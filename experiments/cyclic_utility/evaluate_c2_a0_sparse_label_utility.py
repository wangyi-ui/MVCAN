"""Materialize and evaluate the preregistered C2-A0 calibration.

All calibration is completed from frozen C0 arrays and fourteen frozen sparse
labels before the durable seal.  Full GT, the C0 post-GT audit, and Bridge
correctness are reachable only after the NPZ and seal have been fsynced,
reloaded, and hash-verified.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.b7_sparse_supervision.b7_sparse_anchor_protocol import (
    DEFAULT_LABEL_SPLIT_DIR,
    load_fixed_label_split,
)
from experiments.cyclic_utility import (
    c2_a0_sparse_label_utility_protocol as c2,
)
from experiments.cyclic_utility import (
    evaluate_bridge_p0_residual_utility as bridge_evaluator,
)
from irv.b4_information_utility import tensor_sha256
from weak_quality import ndarray_sha256


STAGE = c2.STAGE
DATASET = "Caltech-6V"
PREREGISTRATION_DIR = (
    REPOSITORY_ROOT
    / "experiment_freeze/c2_a0_fallback_protocol_preregistered_20260904"
)
DEFAULT_FULL_GT_PATH = bridge_evaluator.DEFAULT_FULL_GT_PATH
PRE_GT_NPZ_NAME = "c2_calibrated_utility_pre_gt.npz"
CALIBRATION_SEAL_NAME = "c2_calibration_seal.json"

FORMAL_INPUT_NAMES = (
    "U_cycle",
    "C_conf",
    "y_gen",
    "generator_subsets",
    "verifier_subsets",
)
PRE_GT_ARRAY_NAMES = (
    "sample_ids",
    "R",
    "anchors",
    "soft_contingency",
    "sparse_mapping",
    "E_label",
    "U_cycle",
    "U_tilde",
    "anchors_shuffle",
    "soft_contingency_shuffle",
    "sparse_mapping_shuffle",
    "E_label_shuffle",
    "U_tilde_shuffle",
    "C_conf",
    "y_gen",
    "generator_subsets",
    "verifier_subsets",
    "labeled_sample_ids",
    "labeled_targets",
    "shuffled_labeled_targets",
    "unlabeled_ids",
)
PER_SEED_OUTPUT_FILES = (
    PRE_GT_NPZ_NAME,
    CALIBRATION_SEAL_NAME,
    "confidence_strata_audit.json",
    "c2_metrics_by_direction.json",
    "c2_seed_summary.json",
    "c2_audit.json",
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _display(path):
    value = _resolve(path).resolve()
    try:
        return str(value.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(value)


def _same_path(left, right):
    return _resolve(left).resolve() == _resolve(right).resolve()


def _readonly(value, dtype=None):
    array = np.array(np.asarray(value, dtype=dtype), copy=True, order="C")
    array.setflags(write=False)
    return array


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def write_json_durable(path, value):
    with open(path, "x", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_sha256_manifest(manifest_path):
    """Verify one repository-relative GNU SHA256 manifest."""
    path = _resolve(manifest_path)
    _require(path.is_file(), "C2-A0 preregistration manifest is missing")
    entries = OrderedDict()
    with open(path, "r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split(None, 1)
            _require(
                len(fields) == 2 and len(fields[0]) == 64,
                "invalid C2-A0 manifest line " + str(line_number),
            )
            expected, relative_name = fields
            relative_name = relative_name.lstrip("*")
            _require(relative_name not in entries, "duplicate C2-A0 manifest path")
            artifact_path = (REPOSITORY_ROOT / relative_name).resolve()
            try:
                artifact_path.relative_to(REPOSITORY_ROOT.resolve())
            except ValueError as error:
                raise RuntimeError("C2-A0 manifest path escapes repository") from error
            _require(artifact_path.is_file(), "manifest artifact is missing: " + relative_name)
            actual = file_sha256(artifact_path)
            _require(actual == expected, "SHA256 mismatch: " + relative_name)
            entries[relative_name] = actual
    _require(entries, "C2-A0 preregistration manifest is empty")
    return {
        "path": _display(path),
        "file_sha256": file_sha256(path),
        "entries": dict(entries),
        "all_entries_exact_match": True,
    }


def verify_preregistration_freeze(preregistration_dir=PREREGISTRATION_DIR):
    root = _resolve(preregistration_dir)
    required = (
        "STATUS.txt",
        "PROTOCOL.txt",
        "source_sha256.txt",
        "c0_input_sha256.txt",
        "bridge_final_sha256.txt",
        "CONCLUSION.txt",
    )
    _require(root.is_dir(), "C2-A0 preregistration directory is missing")
    _require(all((root / name).is_file() for name in required), "C2-A0 preregistration is incomplete")
    status = (root / "STATUS.txt").read_text(encoding="utf-8")
    protocol = (root / "PROTOCOL.txt").read_text(encoding="utf-8")
    conclusion = (root / "CONCLUSION.txt").read_text(encoding="utf-8")
    _require(
        "C2-A0 FALLBACK PROTOCOL PREREGISTERED" in status
        and "C2-A0 scientific calibration:\nNOT RUN" in status
        and "Formal C2-A0 MUST NOT use recovered p_gen" in protocol
        and conclusion.strip() == "C2-A0 protocol frozen before implementation.",
        "C2-A0 preregistration state boundary mismatch",
    )
    manifests = {
        name: verify_sha256_manifest(root / name)
        for name in ("source_sha256.txt", "c0_input_sha256.txt", "bridge_final_sha256.txt")
    }
    return {
        "directory": _display(root),
        "required_files": list(required),
        "file_sha256": {name: file_sha256(root / name) for name in required},
        "manifests": manifests,
        "all_manifest_checks_pass": True,
    }


@dataclass(frozen=True)
class C2Paths:
    c0_artifact_path: Path
    c0_seal_path: Path
    c0_audit_path: Path
    label_split_dir: Path
    full_gt_path: Path
    output_dir: Path


def default_paths(seed, output_dir=None):
    active_seed = c2.validate_seed(seed)
    c0_root = (
        REPOSITORY_ROOT
        / "outputs/cyclic_utility"
        / ("c0_complementary_semantic_verification_seed" + str(active_seed))
    )
    target = (
        REPOSITORY_ROOT
        / "outputs/cyclic_utility"
        / ("c2_a0_sparse_label_utility_seed" + str(active_seed))
        if output_dir is None
        else _resolve(output_dir)
    )
    return C2Paths(
        c0_artifact_path=c0_root / "c0_predictions_and_scores.npz",
        c0_seal_path=c0_root / "c0_prediction_seal.json",
        c0_audit_path=c0_root / "c0_audit.json",
        label_split_dir=_resolve(DEFAULT_LABEL_SPLIT_DIR),
        full_gt_path=_resolve(DEFAULT_FULL_GT_PATH),
        output_dir=target,
    )


def _require_frozen_c0_manifest_members(seed, paths, preregistration_audit):
    active_seed = c2.validate_seed(seed)
    entries = preregistration_audit["manifests"]["c0_input_sha256.txt"]["entries"]
    required_paths = (paths.c0_artifact_path, paths.c0_seal_path)
    for path in required_paths:
        relative_name = _display(path)
        _require(
            relative_name in entries and file_sha256(path) == entries[relative_name],
            "selected C0 seed input is not the preregistered artifact: " + relative_name,
        )
    _require(
        ("seed" + str(active_seed)) in _display(paths.c0_artifact_path)
        and ("seed" + str(active_seed)) in _display(paths.c0_seal_path),
        "selected C0 artifact path does not match requested seed",
    )


def _validate_direction_subsets(generator_subsets, verifier_subsets):
    generator = np.asarray(generator_subsets, dtype=np.int64)
    verifier = np.asarray(verifier_subsets, dtype=np.int64)
    all_views = np.arange(6, dtype=np.int64)
    _require(
        generator.shape == verifier.shape == (c2.DIRECTION_COUNT, 3)
        and np.all((generator >= 0) & (generator < 6))
        and np.all((verifier >= 0) & (verifier < 6)),
        "C2-A0 direction-subset boundary mismatch",
    )
    for direction_id in range(c2.DIRECTION_COUNT):
        _require(
            np.unique(generator[direction_id]).size == 3
            and np.unique(verifier[direction_id]).size == 3
            and np.intersect1d(generator[direction_id], verifier[direction_id]).size == 0
            and np.array_equal(
                np.sort(np.concatenate((generator[direction_id], verifier[direction_id]))),
                all_views,
            ),
            "C2-A0 generator/verifier subset partition mismatch",
        )


def load_c0_inputs_before_gt(seed, c0_artifact_path, c0_seal_path):
    """Load only the five frozen formal arrays; no post-GT record is accepted."""
    active_seed = c2.validate_seed(seed)
    artifact_path = _resolve(c0_artifact_path)
    seal_path = _resolve(c0_seal_path)
    _require(artifact_path.is_file() and seal_path.is_file(), "C2-A0 frozen C0 input is missing")
    artifact_hash = file_sha256(artifact_path)
    seal_hash = file_sha256(seal_path)
    seal = read_json(seal_path)
    _require(
        seal.get("stage") == "C0"
        and _same_path(seal.get("artifact_path", ""), artifact_path)
        and seal.get("artifact_file_sha256") == artifact_hash
        and seal.get("scores_completed_before_GT") is True
        and seal.get("scores_saved_before_GT") is True
        and seal.get("scores_hashed_before_GT") is True
        and seal.get("scores_reloaded_before_GT") is True
        and seal.get("full_GT_loaded_before_seal") is False
        and seal.get("R_loaded_before_seal") is False
        and seal.get("sparse_labels_loaded_before_seal") is False
        and seal.get("corruption_mask_loaded_before_seal") is False
        and seal.get("oracle_loaded_before_seal") is False,
        "C2-A0 C0 prediction-seal boundary mismatch",
    )
    expected_shapes = {
        "U_cycle": (c2.SAMPLE_NUM, c2.DIRECTION_COUNT),
        "C_conf": (c2.SAMPLE_NUM, c2.DIRECTION_COUNT),
        "y_gen": (c2.SAMPLE_NUM, c2.DIRECTION_COUNT),
        "generator_subsets": (c2.DIRECTION_COUNT, 3),
        "verifier_subsets": (c2.DIRECTION_COUNT, 3),
    }
    expected_dtypes = {
        "U_cycle": np.dtype(np.float32),
        "C_conf": np.dtype(np.float32),
        "y_gen": np.dtype(np.int64),
        "generator_subsets": np.dtype(np.int64),
        "verifier_subsets": np.dtype(np.int64),
    }
    arrays = {}
    with np.load(artifact_path, allow_pickle=False) as archive:
        archive_fields = tuple(archive.files)
        _require(set(FORMAL_INPUT_NAMES).issubset(archive_fields), "C2-A0 C0 archive is incomplete")
        for name in FORMAL_INPUT_NAMES:
            value = np.array(archive[name], copy=True, order="C")
            own_record = seal.get("arrays", {}).get(name, {})
            _require(
                value.shape == expected_shapes[name]
                and value.dtype == expected_dtypes[name]
                and np.isfinite(value).all()
                and own_record.get("shape") == list(value.shape)
                and own_record.get("dtype") == str(value.dtype)
                and own_record.get("logical_sha256") == tensor_sha256(value),
                "C2-A0 C0 array own-seal mismatch: " + name,
            )
            value.setflags(write=False)
            arrays[name] = value
        if "conf_gen" in archive_fields:
            conf_gen = np.array(archive["conf_gen"], copy=True, order="C")
            own_record = seal.get("arrays", {}).get("conf_gen", {})
            _require(
                conf_gen.shape == expected_shapes["C_conf"]
                and conf_gen.dtype == expected_dtypes["C_conf"]
                and own_record.get("logical_sha256") == tensor_sha256(conf_gen)
                and np.array_equal(conf_gen, arrays["C_conf"]),
                "C2-A0 conf_gen must exactly equal C_conf when present",
            )
    _require(
        np.all((arrays["U_cycle"] >= 0.0) & (arrays["U_cycle"] <= 1.0))
        and np.all((arrays["C_conf"] >= 0.0) & (arrays["C_conf"] <= 1.0))
        and np.all((arrays["y_gen"] >= 0) & (arrays["y_gen"] < c2.CLASS_NUM)),
        "C2-A0 formal input value boundary mismatch",
    )
    _validate_direction_subsets(arrays["generator_subsets"], arrays["verifier_subsets"])
    sample_ids = c2.canonical_sample_ids()
    arrays["sample_ids"] = sample_ids
    return arrays, {
        "seed": active_seed,
        "C0_artifact_path": _display(artifact_path),
        "C0_artifact_file_sha256": artifact_hash,
        "C0_prediction_seal_path": _display(seal_path),
        "C0_prediction_seal_file_sha256": seal_hash,
        "C0_array_tensor_sha256": {
            name: seal["arrays"][name]["logical_sha256"] for name in FORMAL_INPUT_NAMES
        },
        "archive_fields": list(archive_fields),
        "conf_gen_present_and_exact_C_conf": "conf_gen" in archive_fields,
        "sample_ids_logical_sha256": ndarray_sha256(sample_ids),
        "legacy_C0_seed_field": seal.get("seed"),
        "legacy_C0_seed_field_used_as_authority": False,
        "C0_prediction_seal": seal,
        "GT_loaded": False,
        "c0_audit_loaded": False,
        "Bridge_correctness_loaded": False,
        "corruption_mask_loaded": False,
        "p_gen_loaded": False,
        "recovered_p_gen_loaded": False,
    }


def load_fixed_sparse_labels(split_dir=DEFAULT_LABEL_SPLIT_DIR):
    """Reuse B7's loader, deriving targets only from its frozen class-ID schema."""
    split = load_fixed_label_split(split_dir)
    labeled_ids = _readonly(split["labeled_sample_ids"], dtype=np.int64)
    unlabeled_ids = _readonly(split["unlabeled_sample_ids"], dtype=np.int64)
    per_class = split["record"].get("per_class_labeled_ids", {})
    target_by_id = {}
    for class_id in range(c2.CLASS_NUM):
        class_ids = per_class.get(str(class_id))
        _require(isinstance(class_ids, list) and len(class_ids) == 2, "B7 sparse target schema mismatch")
        for sample_id in class_ids:
            _require(int(sample_id) not in target_by_id, "B7 sparse sample appears in two classes")
            target_by_id[int(sample_id)] = class_id
    _require(set(target_by_id) == set(labeled_ids.tolist()), "B7 sparse class schema/ID mismatch")
    targets = _readonly([target_by_id[int(sample_id)] for sample_id in labeled_ids], dtype=np.int64)
    validated = c2.validate_fixed_sparse_split(labeled_ids, unlabeled_ids, targets)
    shuffled = c2.validate_negative_control(targets, c2.fixed_shuffled_targets())
    return {
        **validated,
        "shuffled_labeled_targets": shuffled,
        "label_split_sha256": split["label_split_sha256"],
        "labeled_ids_sha256": split["labeled_ids_sha256"],
        "label_split_path": split["label_split_path"],
        "labeled_ids_path": split["labeled_ids_path"],
        "unlabeled_ids_path": split["unlabeled_ids_path"],
        "targets_constructed_from_per_class_labeled_ids": True,
        "full_GT_used_for_sparse_targets": False,
    }


def build_pre_gt_bundle(seed, c0_arrays, c0_provenance, sparse_split, preregistration_audit):
    active_seed = c2.validate_seed(seed)
    calibrated = c2.build_calibration_outputs(
        y_gen=c0_arrays["y_gen"],
        U_cycle=c0_arrays["U_cycle"],
        labeled_ids=sparse_split["labeled_sample_ids"],
        labeled_targets=sparse_split["labeled_targets"],
        shuffled_targets=sparse_split["shuffled_labeled_targets"],
    )
    arrays = OrderedDict((
        ("sample_ids", c0_arrays["sample_ids"]),
        ("R", calibrated["R"]),
        ("anchors", calibrated["anchors"]),
        ("soft_contingency", calibrated["soft_contingency"]),
        ("sparse_mapping", calibrated["sparse_mapping"]),
        ("E_label", calibrated["E_label"]),
        ("U_cycle", c0_arrays["U_cycle"]),
        ("U_tilde", calibrated["U_tilde"]),
        ("anchors_shuffle", calibrated["anchors_shuffle"]),
        ("soft_contingency_shuffle", calibrated["soft_contingency_shuffle"]),
        ("sparse_mapping_shuffle", calibrated["sparse_mapping_shuffle"]),
        ("E_label_shuffle", calibrated["E_label_shuffle"]),
        ("U_tilde_shuffle", calibrated["U_tilde_shuffle"]),
        ("C_conf", c0_arrays["C_conf"]),
        ("y_gen", c0_arrays["y_gen"]),
        ("generator_subsets", c0_arrays["generator_subsets"]),
        ("verifier_subsets", c0_arrays["verifier_subsets"]),
        ("labeled_sample_ids", sparse_split["labeled_sample_ids"]),
        ("labeled_targets", sparse_split["labeled_targets"]),
        ("shuffled_labeled_targets", sparse_split["shuffled_labeled_targets"]),
        ("unlabeled_ids", sparse_split["unlabeled_ids"]),
    ))
    _require(tuple(arrays) == PRE_GT_ARRAY_NAMES, "C2-A0 pre-GT array order mismatch")
    return {
        "seed": active_seed,
        "arrays": arrays,
        "C0_provenance": c0_provenance,
        "sparse_split": sparse_split,
        "preregistration": preregistration_audit,
    }


def _array_records(arrays):
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
            "logical_sha256": ndarray_sha256(value),
        }
        for name, value in arrays.items()
    }


def persist_calibration_bundle(bundle, output_dir):
    """Fsync, reload, and hash-verify every calibration array before its seal."""
    output_root = _resolve(output_dir)
    _require(not output_root.exists(), "refusing to overwrite C2-A0 seed output")
    arrays = bundle["arrays"]
    _require(tuple(arrays) == PRE_GT_ARRAY_NAMES, "C2-A0 persistence array schema mismatch")
    array_records = _array_records(arrays)
    output_root.mkdir(parents=True, exist_ok=False)
    npz_path = output_root / PRE_GT_NPZ_NAME
    with open(npz_path, "xb") as output_file:
        np.savez(output_file, **arrays)
        output_file.flush()
        os.fsync(output_file.fileno())
    _fsync_directory(output_root)
    with np.load(npz_path, allow_pickle=False) as archive:
        _require(tuple(archive.files) == PRE_GT_ARRAY_NAMES, "C2-A0 pre-GT NPZ field mismatch")
        for name in PRE_GT_ARRAY_NAMES:
            reloaded = np.array(archive[name], copy=True, order="C")
            _require(
                np.array_equal(reloaded, np.asarray(arrays[name]))
                and ndarray_sha256(reloaded) == array_records[name]["logical_sha256"],
                "C2-A0 pre-GT NPZ reload/hash mismatch: " + name,
            )
    c0_provenance = bundle["C0_provenance"]
    split = bundle["sparse_split"]
    seal = {
        "stage": STAGE,
        "seed": bundle["seed"],
        "dataset": DATASET,
        "N": c2.SAMPLE_NUM,
        "K": c2.CLASS_NUM,
        "direction_count": c2.DIRECTION_COUNT,
        "label_count": c2.LABEL_COUNT,
        "unlabeled_evaluation_count": c2.UNLABELED_EVAL_COUNT,
        "pre_gt_npz_path": _display(npz_path),
        "pre_gt_npz_file_sha256": file_sha256(npz_path),
        "arrays": array_records,
        "input_C0_NPZ_SHA256": c0_provenance["C0_artifact_file_sha256"],
        "input_C0_prediction_seal_SHA256": c0_provenance["C0_prediction_seal_file_sha256"],
        "U_cycle_logical_SHA256": array_records["U_cycle"]["logical_sha256"],
        "C_conf_logical_SHA256": array_records["C_conf"]["logical_sha256"],
        "y_gen_logical_SHA256": array_records["y_gen"]["logical_sha256"],
        "sample_ids_logical_SHA256": array_records["sample_ids"]["logical_sha256"],
        "fixed_labeled_IDs_hash": array_records["labeled_sample_ids"]["logical_sha256"],
        "fixed_labeled_targets_hash": array_records["labeled_targets"]["logical_sha256"],
        "shuffle_target_hash": array_records["shuffled_labeled_targets"]["logical_sha256"],
        "R_hash": array_records["R"]["logical_sha256"],
        "anchors_hash": array_records["anchors"]["logical_sha256"],
        "mapping_hash": array_records["sparse_mapping"]["logical_sha256"],
        "E_label_hash": array_records["E_label"]["logical_sha256"],
        "U_tilde_hash": array_records["U_tilde"]["logical_sha256"],
        "E_label_shuffle_hash": array_records["E_label_shuffle"]["logical_sha256"],
        "U_tilde_shuffle_hash": array_records["U_tilde_shuffle"]["logical_sha256"],
        "label_split_sha256": split["label_split_sha256"],
        "preregistration_file_sha256": bundle["preregistration"]["file_sha256"],
        "C2_source_SHA256": {
            "protocol": file_sha256(Path(c2.__file__)),
            "evaluator": file_sha256(Path(__file__)),
        },
        "NPZ_saved_fsynced_reloaded_hash_verified": True,
        "GT_loaded_before_calibration_seal": False,
        "c0_audit_loaded_before_calibration_seal": False,
        "Bridge_correctness_loaded_before_calibration_seal": False,
        "corruption_mask_loaded": False,
        "p_gen_loaded": False,
        "recovered_p_gen_loaded": False,
        "training_performed": False,
        "backward_called": False,
        "optimizer_step_called": False,
        "C_conf_used_for_calibration": False,
        "FORMAL_C2_USES_PGEN": c2.FORMAL_C2_USES_PGEN,
        "FORMAL_C2_USES_YGEN_VOTE_STATE": c2.FORMAL_C2_USES_YGEN_VOTE_STATE,
        "GT_PRESEAL_ISOLATION": c2.GT_PRESEAL_ISOLATION,
        "TRAINING_PATH_PRESENT": c2.TRAINING_PATH_PRESENT,
    }
    seal_path = output_root / CALIBRATION_SEAL_NAME
    write_json_durable(seal_path, seal)
    _fsync_directory(output_root)
    _require(read_json(seal_path) == seal, "C2-A0 calibration seal durable reload mismatch")
    return {
        "output_dir": output_root,
        "npz_path": npz_path,
        "seal_path": seal_path,
        "seal": seal,
        "seal_file_sha256": file_sha256(seal_path),
        "durable_calibration_seal_verified": True,
    }


def load_c0_audit_after_calibration_seal(c0_audit_path, c0_prediction_seal):
    path = _resolve(c0_audit_path)
    _require(path.is_file(), "C2-A0 post-seal C0 audit is missing")
    audit = read_json(path)
    mapping = audit.get("global_mapping", {})
    _require(
        audit.get("stage") == "C0"
        and audit.get("C0_AUDIT_PASS") is True
        and audit.get("prediction_seal") == c0_prediction_seal
        and audit.get("decision", {}).get("final_decision") == "C0_COMPLEMENTARY_CYCLE_VALIDITY_PASS"
        and mapping.get("mapping_fit_count") == 1
        and mapping.get("mapping_fit_after_score_seal") is True
        and mapping.get("single_global_mapping_reused_all_directions") is True
        and mapping.get("direction_specific_mapping_used") is False,
        "C2-A0 frozen C0 post-seal mapping boundary mismatch",
    )
    return audit, {"path": _display(path), "file_sha256": file_sha256(path), "loaded_after_calibration_seal": True}


def load_native_global_cluster_after_calibration_seal(c0_artifact_path, c0_prediction_seal):
    path = _resolve(c0_artifact_path)
    with np.load(path, allow_pickle=False) as archive:
        _require("native_global_cluster" in archive.files, "C0 native global cluster is missing")
        native = np.array(archive["native_global_cluster"], copy=True, order="C")
    record = c0_prediction_seal.get("arrays", {}).get("native_global_cluster", {})
    _require(
        native.shape == (c2.SAMPLE_NUM,)
        and native.dtype == np.dtype(np.int64)
        and record.get("logical_sha256") == tensor_sha256(native)
        and np.array_equal(np.unique(native), np.arange(c2.CLASS_NUM, dtype=np.int64)),
        "C2-A0 native global cluster own-seal mismatch",
    )
    native.setflags(write=False)
    return native


def evaluate_after_calibration_seal(bundle, persistence, paths):
    """The only post-seal path: GT, then C0 audit, then frozen correctness."""
    _require(
        persistence.get("durable_calibration_seal_verified") is True
        and read_json(persistence["seal_path"]) == persistence["seal"],
        "C2-A0 durable calibration seal is not verified",
    )
    full_GT, full_GT_audit = bridge_evaluator.load_bridge_full_ground_truth_after_seal(paths.full_gt_path)
    c0_audit, c0_audit_provenance = load_c0_audit_after_calibration_seal(
        paths.c0_audit_path, bundle["C0_provenance"]["C0_prediction_seal"]
    )
    native = load_native_global_cluster_after_calibration_seal(
        paths.c0_artifact_path, bundle["C0_provenance"]["C0_prediction_seal"]
    )
    correctness_inputs = {
        "y_gen": bundle["arrays"]["y_gen"],
        "native_global_cluster": native,
    }
    correct, mapping_audit = bridge_evaluator.build_correctness_from_frozen_c0_mapping(
        correctness_inputs, c0_audit, full_GT
    )
    unlabeled = bundle["arrays"]["unlabeled_ids"]
    labeled = bundle["arrays"]["labeled_sample_ids"]
    _require(
        unlabeled.shape == (c2.UNLABELED_EVAL_COUNT,)
        and np.intersect1d(labeled, unlabeled).size == 0,
        "C2-A0 post-seal evaluation split boundary mismatch",
    )
    arrays = bundle["arrays"]
    directional = c2.analyze_all_directions(
        correct=correct[unlabeled],
        U_cycle=arrays["U_cycle"][unlabeled],
        U_tilde=arrays["U_tilde"][unlabeled],
        U_tilde_shuffle=arrays["U_tilde_shuffle"][unlabeled],
        E_label=arrays["E_label"][unlabeled],
        C_conf=arrays["C_conf"][unlabeled],
        sample_ids=unlabeled,
    )
    summary = c2.build_seed_summary(bundle["seed"], directional)
    return {
        "directional": directional,
        "summary": summary,
        "full_GT_audit": {**full_GT_audit, "loaded_after_calibration_seal": True},
        "c0_audit_provenance": c0_audit_provenance,
        "mapping_audit": mapping_audit,
        "correctness_logical_sha256": ndarray_sha256(correct),
    }


def save_postseal_results(bundle, persistence, evaluation):
    output_root = persistence["output_dir"]
    directional = evaluation["directional"]
    strata = {
        "stage": STAGE,
        "seed": bundle["seed"],
        "confidence_bin_count": c2.CONFIDENCE_BIN_COUNT,
        "evaluation_sample_count": c2.UNLABELED_EVAL_COUNT,
        "GT_used_for_stratification": False,
        "stable_sort_key": ["C_conf", "sample_id"],
        "directions": [record["confidence_strata"] for record in directional["directions"]],
    }
    metrics = {
        key: value for key, value in directional.items()
    }
    audit = {
        "stage": STAGE,
        "seed": bundle["seed"],
        "dataset": DATASET,
        "N": c2.SAMPLE_NUM,
        "K": c2.CLASS_NUM,
        "direction_count": c2.DIRECTION_COUNT,
        "label_count": c2.LABEL_COUNT,
        "unlabeled_evaluation_count": c2.UNLABELED_EVAL_COUNT,
        "C2_A0_AUDIT_PASS": True,
        "calibration_seal": persistence["seal"],
        "calibration_seal_file_sha256": persistence["seal_file_sha256"],
        "durable_calibration_seal_verified_before_GT": True,
        "GT_loaded_before_calibration_seal": False,
        "GT_loaded_after_calibration_seal": True,
        "c0_audit_loaded_before_calibration_seal": False,
        "c0_audit_loaded_after_calibration_seal": True,
        "Bridge_correctness_loaded_before_calibration_seal": False,
        "Bridge_correctness_built_after_calibration_seal": True,
        "full_GT": evaluation["full_GT_audit"],
        "c0_audit": evaluation["c0_audit_provenance"],
        "frozen_C0_global_mapping": evaluation["mapping_audit"],
        "correctness_logical_sha256": evaluation["correctness_logical_sha256"],
        "evaluation_only_unlabeled": True,
        "labeled_evaluation_intersection_count": 0,
        "corruption_mask_loaded": False,
        "p_gen_loaded": False,
        "recovered_p_gen_loaded": False,
        "model_forward_called": False,
        "training_performed": False,
        "backward_called": False,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "C_conf_used_for_calibration": False,
        "C_conf_used_for_postseal_stratification": True,
        "C2_specific_GT_mapping_fit": False,
        "direction_specific_mapping_fit": False,
        "bin_specific_mapping_fit": False,
        "FORMAL_C2_USES_PGEN": c2.FORMAL_C2_USES_PGEN,
        "FORMAL_C2_USES_YGEN_VOTE_STATE": c2.FORMAL_C2_USES_YGEN_VOTE_STATE,
        "GT_PRESEAL_ISOLATION": c2.GT_PRESEAL_ISOLATION,
        "TRAINING_PATH_PRESENT": c2.TRAINING_PATH_PRESENT,
    }
    records = OrderedDict((
        ("confidence_strata_audit.json", strata),
        ("c2_metrics_by_direction.json", metrics),
        ("c2_seed_summary.json", evaluation["summary"]),
        ("c2_audit.json", audit),
    ))
    for name, record in records.items():
        write_json_durable(output_root / name, record)
    _fsync_directory(output_root)
    _require(
        all(read_json(output_root / name) == record for name, record in records.items()),
        "C2-A0 post-seal result reload mismatch",
    )
    return records


def run_evaluation(seed, output_dir=None, full_gt_path=None):
    """Run one formal seed.  This entry point is intentionally not used by tests."""
    active_seed = c2.validate_seed(seed)
    paths = default_paths(active_seed, output_dir=output_dir)
    if full_gt_path is not None:
        paths = C2Paths(
            c0_artifact_path=paths.c0_artifact_path,
            c0_seal_path=paths.c0_seal_path,
            c0_audit_path=paths.c0_audit_path,
            label_split_dir=paths.label_split_dir,
            full_gt_path=_resolve(full_gt_path),
            output_dir=paths.output_dir,
        )
    preregistration = verify_preregistration_freeze()
    _require_frozen_c0_manifest_members(active_seed, paths, preregistration)
    arrays, provenance = load_c0_inputs_before_gt(
        active_seed, paths.c0_artifact_path, paths.c0_seal_path
    )
    sparse_split = load_fixed_sparse_labels(paths.label_split_dir)
    bundle = build_pre_gt_bundle(active_seed, arrays, provenance, sparse_split, preregistration)
    persistence = persist_calibration_bundle(bundle, paths.output_dir)
    evaluation = evaluate_after_calibration_seal(bundle, persistence, paths)
    records = save_postseal_results(bundle, persistence, evaluation)
    print("C2_A0_SEED_PASS=" + str(evaluation["summary"]["C2_A0_SEED_PASS"]))
    print("Saved: " + _display(paths.output_dir))
    return {
        "bundle": bundle,
        "persistence": persistence,
        "evaluation": evaluation,
        "records": records,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=c2.SEEDS)
    parser.add_argument("--output-dir")
    parser.add_argument("--full-gt-path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_evaluation(args.seed, output_dir=args.output_dir, full_gt_path=args.full_gt_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
