"""Evaluate preregistered C2-B0 with a strict pre-GT residual seal.

Formal execution is exposed for the later authorized phase, but this
implementation turn exercises only the pure unit-test paths.
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

from experiments.cyclic_utility import c2_b0_sparse_utility_residual_protocol as c2
from experiments.cyclic_utility import evaluate_c2_a0_sparse_label_utility as c2_a0_evaluator
from experiments.cyclic_utility import evaluate_bridge_p0_residual_utility as bridge_evaluator
from weak_quality import ndarray_sha256


STAGE = c2.STAGE
DATASET = "Caltech-6V"
PREREGISTRATION_DIR = (
    REPOSITORY_ROOT
    / "experiment_freeze/c2_b0_sparse_utility_residual_preregistered_20260905"
)
PRE_GT_NPZ_NAME = "c2_b0_residual_pre_gt.npz"
RESIDUAL_SEAL_NAME = "c2_b0_residual_seal.json"
PRE_GT_ARRAY_NAMES = (
    "R",
    "z_L",
    "e_L",
    "e_hat",
    "z_L_shuffle",
    "e_L_shuffle",
    "e_hat_shuffle",
    "U_cycle",
    "C_conf",
    "y_gen",
    "sample_ids",
    "labeled_sample_ids",
    "labeled_targets",
    "shuffled_labeled_targets",
    "unlabeled_ids",
)
PER_SEED_OUTPUT_FILES = (
    PRE_GT_NPZ_NAME,
    RESIDUAL_SEAL_NAME,
    "c2_b0_confidence_strata_audit.json",
    "c2_b0_metrics_by_direction.json",
    "c2_b0_seed_summary.json",
    "c2_b0_audit.json",
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
    path = _resolve(manifest_path)
    _require(path.is_file(), "C2-B0 SHA256 manifest is missing")
    entries = OrderedDict()
    with open(path, "r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split(None, 1)
            _require(
                len(fields) == 2 and len(fields[0]) == 64,
                "invalid C2-B0 manifest line " + str(line_number),
            )
            expected, relative_name = fields
            relative_name = relative_name.lstrip("*")
            artifact_path = (REPOSITORY_ROOT / relative_name).resolve()
            try:
                artifact_path.relative_to(REPOSITORY_ROOT.resolve())
            except ValueError as error:
                raise RuntimeError("C2-B0 manifest path escapes repository") from error
            _require(
                artifact_path.is_file() and file_sha256(artifact_path) == expected,
                "C2-B0 frozen SHA256 mismatch: " + relative_name,
            )
            entries[relative_name] = expected
    _require(entries, "C2-B0 SHA256 manifest is empty")
    return {
        "path": _display(path),
        "file_sha256": file_sha256(path),
        "entries": dict(entries),
        "all_entries_exact_match": True,
    }


def verify_preregistration_freeze(preregistration_dir=PREREGISTRATION_DIR):
    root = _resolve(preregistration_dir)
    required = ("PROTOCOL.txt", "STATUS.txt", "CONCLUSION.txt")
    _require(
        root.is_dir() and all((root / name).is_file() for name in required),
        "C2-B0 preregistration is incomplete",
    )
    protocol = (root / "PROTOCOL.txt").read_text(encoding="utf-8")
    status = (root / "STATUS.txt").read_text(encoding="utf-8")
    conclusion = (root / "CONCLUSION.txt").read_text(encoding="utf-8")
    _require(
        "C2-B0 — Sparse Utility Residual Calibration" in protocol
        and "No backward()." in protocol
        and "C2-B0 source:\nNOT IMPLEMENTED" in status
        and conclusion.startswith("C2-B0 scientific protocol is frozen before implementation.")
        and "Formal seed20 / seed30 / seed50 execution is NOT yet allowed." in conclusion,
        "C2-B0 preregistration state boundary mismatch",
    )
    manifests = {
        name: verify_sha256_manifest(root / name)
        for name in ("input_source_sha256.txt", "preregistration_sha256.txt")
    }
    return {
        "directory": _display(root),
        "required_files": list(required),
        "file_sha256": {name: file_sha256(root / name) for name in required},
        "manifests": manifests,
        "C2_A0_frozen_source_verified": True,
    }


@dataclass(frozen=True)
class C2B0Paths:
    c0_artifact_path: Path
    c0_seal_path: Path
    c0_audit_path: Path
    label_split_dir: Path
    full_gt_path: Path
    output_dir: Path


def default_paths(seed, output_dir=None):
    active_seed = c2.validate_seed(seed)
    c2_a0_paths = c2_a0_evaluator.default_paths(active_seed)
    target = (
        REPOSITORY_ROOT
        / "outputs/cyclic_utility"
        / ("c2_b0_sparse_utility_residual_seed" + str(active_seed))
        if output_dir is None
        else _resolve(output_dir)
    )
    return C2B0Paths(
        c0_artifact_path=c2_a0_paths.c0_artifact_path,
        c0_seal_path=c2_a0_paths.c0_seal_path,
        c0_audit_path=c2_a0_paths.c0_audit_path,
        label_split_dir=c2_a0_paths.label_split_dir,
        full_gt_path=c2_a0_paths.full_gt_path,
        output_dir=target,
    )


def load_c0_inputs_before_gt(seed, c0_artifact_path, c0_seal_path):
    """Reuse C2-A0's sealed loader, accepting only pre-GT C0 arrays."""
    arrays, provenance = c2_a0_evaluator.load_c0_inputs_before_gt(
        seed, c0_artifact_path, c0_seal_path
    )
    _require(
        provenance["GT_loaded"] is False
        and provenance["c0_audit_loaded"] is False
        and provenance["Bridge_correctness_loaded"] is False
        and provenance["corruption_mask_loaded"] is False,
        "C2-B0 pre-GT C0 loader crossed an oracle boundary",
    )
    return arrays, provenance


def load_fixed_sparse_labels(split_dir):
    split = c2_a0_evaluator.load_fixed_sparse_labels(split_dir)
    validated = c2.validate_fixed_sparse_split(
        split["labeled_sample_ids"], split["unlabeled_ids"], split["labeled_targets"]
    )
    shuffled = c2.validate_negative_control(
        validated["labeled_targets"], split["shuffled_labeled_targets"]
    )
    return {**split, **validated, "shuffled_labeled_targets": shuffled}


def build_pre_gt_bundle(seed, c0_arrays, c0_provenance, sparse_split, preregistration):
    """Construct all residual objects without accepting any GT-bearing input."""
    active_seed = c2.validate_seed(seed)
    _require(
        c0_provenance.get("GT_loaded") is False
        and c0_provenance.get("c0_audit_loaded") is False
        and c0_provenance.get("Bridge_correctness_loaded") is False
        and c0_provenance.get("corruption_mask_loaded") is False,
        "C2-B0 pre-GT bundle received leaked provenance",
    )
    outputs = c2.build_residual_calibration_outputs(
        y_gen=c0_arrays["y_gen"],
        U_cycle=c0_arrays["U_cycle"],
        labeled_ids=sparse_split["labeled_sample_ids"],
        labeled_targets=sparse_split["labeled_targets"],
        shuffled_targets=sparse_split["shuffled_labeled_targets"],
    )
    arrays = OrderedDict((
        ("R", outputs["R"]),
        ("z_L", outputs["z_L"]),
        ("e_L", outputs["e_L"]),
        ("e_hat", outputs["e_hat"]),
        ("z_L_shuffle", outputs["z_L_shuffle"]),
        ("e_L_shuffle", outputs["e_L_shuffle"]),
        ("e_hat_shuffle", outputs["e_hat_shuffle"]),
        ("U_cycle", c0_arrays["U_cycle"]),
        ("C_conf", c0_arrays["C_conf"]),
        ("y_gen", c0_arrays["y_gen"]),
        ("sample_ids", c0_arrays["sample_ids"]),
        ("labeled_sample_ids", sparse_split["labeled_sample_ids"]),
        ("labeled_targets", sparse_split["labeled_targets"]),
        ("shuffled_labeled_targets", sparse_split["shuffled_labeled_targets"]),
        ("unlabeled_ids", sparse_split["unlabeled_ids"]),
    ))
    _require(tuple(arrays) == PRE_GT_ARRAY_NAMES, "C2-B0 pre-GT array order mismatch")
    return {
        "seed": active_seed,
        "arrays": arrays,
        "outputs": outputs,
        "C0_provenance": c0_provenance,
        "sparse_split": sparse_split,
        "preregistration": preregistration,
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


def persist_residual_bundle(bundle, output_dir):
    """Save, flush, fsync, reload, and logically verify every pre-GT array."""
    output_root = _resolve(output_dir)
    _require(not output_root.exists(), "refusing to overwrite C2-B0 seed output")
    arrays = bundle["arrays"]
    _require(tuple(arrays) == PRE_GT_ARRAY_NAMES, "C2-B0 persistence schema mismatch")
    array_records = _array_records(arrays)
    output_root.mkdir(parents=True, exist_ok=False)
    npz_path = output_root / PRE_GT_NPZ_NAME
    with open(npz_path, "xb") as output_file:
        np.savez(output_file, **arrays)
        output_file.flush()
        os.fsync(output_file.fileno())
    _fsync_directory(output_root)
    with np.load(npz_path, allow_pickle=False) as archive:
        _require(tuple(archive.files) == PRE_GT_ARRAY_NAMES, "C2-B0 sealed NPZ field mismatch")
        for name in PRE_GT_ARRAY_NAMES:
            reloaded = np.array(archive[name], copy=True, order="C")
            _require(
                np.array_equal(reloaded, np.asarray(arrays[name]))
                and ndarray_sha256(reloaded) == array_records[name]["logical_sha256"],
                "C2-B0 sealed array reload/hash mismatch: " + name,
            )
    provenance = bundle["C0_provenance"]
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
        "input_C0_NPZ_SHA256": provenance["C0_artifact_file_sha256"],
        "input_C0_prediction_seal_SHA256": provenance["C0_prediction_seal_file_sha256"],
        "label_split_sha256": split["label_split_sha256"],
        "preregistration_file_sha256": bundle["preregistration"]["file_sha256"],
        "true_target_hash": array_records["labeled_targets"]["logical_sha256"],
        "shuffle_target_hash": array_records["shuffled_labeled_targets"]["logical_sha256"],
        "true_residual_hashes": {
            name: array_records[name]["logical_sha256"]
            for name in ("z_L", "e_L", "e_hat")
        },
        "shuffle_residual_hashes": {
            name: array_records[name]["logical_sha256"]
            for name in ("z_L_shuffle", "e_L_shuffle", "e_hat_shuffle")
        },
        "NPZ_saved_fsynced_reloaded_hash_verified": True,
        "GT_loaded_before_residual_seal": False,
        "c0_audit_loaded_before_residual_seal": False,
        "Bridge_correctness_loaded_before_residual_seal": False,
        "corruption_mask_loaded": False,
        "training_performed": False,
        "backward_called": False,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "model_forward_called": False,
        "C_conf_used_for_residual_propagation": False,
        "shuffle_recomputed_from_targets": True,
        "shuffle_only_final_residual": False,
    }
    _require(
        seal["true_target_hash"] != seal["shuffle_target_hash"],
        "C2-B0 true/shuffle target hashes unexpectedly match",
    )
    seal_path = output_root / RESIDUAL_SEAL_NAME
    write_json_durable(seal_path, seal)
    _fsync_directory(output_root)
    _require(read_json(seal_path) == seal, "C2-B0 residual seal durable reload mismatch")
    return {
        "output_dir": output_root,
        "npz_path": npz_path,
        "seal_path": seal_path,
        "seal": seal,
        "seal_file_sha256": file_sha256(seal_path),
        "durable_residual_seal_verified": True,
    }


def load_full_ground_truth_after_residual_seal(path):
    return bridge_evaluator.load_bridge_full_ground_truth_after_seal(path)


def load_c0_audit_after_residual_seal(c0_audit_path, c0_prediction_seal):
    return c2_a0_evaluator.load_c0_audit_after_calibration_seal(
        c0_audit_path, c0_prediction_seal
    )


def build_bridge_correctness_after_residual_seal(arrays, c0_audit, full_GT):
    return bridge_evaluator.build_correctness_from_frozen_c0_mapping(
        arrays, c0_audit, full_GT
    )


def evaluate_after_residual_seal(bundle, persistence, paths):
    """Load all truth-bearing objects only after the durable residual seal."""
    _require(
        persistence.get("durable_residual_seal_verified") is True
        and read_json(persistence["seal_path"]) == persistence["seal"],
        "C2-B0 durable residual seal is not verified",
    )
    full_GT, full_GT_audit = load_full_ground_truth_after_residual_seal(
        paths.full_gt_path
    )
    c0_audit, c0_audit_provenance = load_c0_audit_after_residual_seal(
        paths.c0_audit_path, bundle["C0_provenance"]["C0_prediction_seal"]
    )
    native = c2_a0_evaluator.load_native_global_cluster_after_calibration_seal(
        paths.c0_artifact_path, bundle["C0_provenance"]["C0_prediction_seal"]
    )
    correct, mapping_audit = build_bridge_correctness_after_residual_seal(
        {"y_gen": bundle["arrays"]["y_gen"], "native_global_cluster": native},
        c0_audit,
        full_GT,
    )
    arrays = bundle["arrays"]
    unlabeled = arrays["unlabeled_ids"]
    _require(
        unlabeled.shape == (c2.UNLABELED_EVAL_COUNT,)
        and np.intersect1d(unlabeled, arrays["labeled_sample_ids"]).size == 0,
        "C2-B0 post-seal evaluation split mismatch",
    )
    # z_GT/e_GT are defined only on the fixed 1386 unlabeled samples.
    z_gt = np.ascontiguousarray(correct[unlabeled], dtype=np.float64)
    e_gt = np.ascontiguousarray(z_gt - arrays["U_cycle"][unlabeled], dtype=np.float64)
    residual = c2.analyze_residual_directions(
        arrays["e_hat"][unlabeled], arrays["e_hat_shuffle"][unlabeled],
        e_gt, unlabeled,
    )
    U_tilde = c2.bounded_utility_update(arrays["U_cycle"], arrays["e_hat"])
    U_tilde_shuffle = c2.bounded_utility_update(
        arrays["U_cycle"], arrays["e_hat_shuffle"]
    )
    calibration = c2.analyze_calibration_directions(
        correct[unlabeled], arrays["U_cycle"][unlabeled], U_tilde[unlabeled],
        U_tilde_shuffle[unlabeled], arrays["C_conf"][unlabeled], unlabeled,
    )
    summary = c2.build_seed_summary(bundle["seed"], residual, calibration)
    return {
        "residual": residual,
        "calibration": calibration,
        "summary": summary,
        "full_GT_audit": {**full_GT_audit, "loaded_after_residual_seal": True},
        "c0_audit_provenance": {
            **c0_audit_provenance, "loaded_after_residual_seal": True
        },
        "mapping_audit": mapping_audit,
        "correctness_logical_sha256": ndarray_sha256(correct),
        "e_GT_logical_sha256": ndarray_sha256(e_gt),
    }


def save_postseal_results(bundle, persistence, evaluation):
    output_root = persistence["output_dir"]
    calibration = evaluation["calibration"]
    strata = {
        "stage": STAGE,
        "seed": bundle["seed"],
        "confidence_bin_count": c2.CONFIDENCE_BIN_COUNT,
        "evaluation_sample_count": c2.UNLABELED_EVAL_COUNT,
        "GT_used_for_stratification": False,
        "stable_sort_key": ["C_conf", "sample_id"],
        "directions": [
            record["confidence_strata"] for record in calibration["directions"]
        ],
    }
    metrics = {
        "stage": STAGE,
        "seed": bundle["seed"],
        "evaluation_sample_count": c2.UNLABELED_EVAL_COUNT,
        "residual": evaluation["residual"],
        "calibration": calibration,
    }
    audit = {
        "stage": STAGE,
        "seed": bundle["seed"],
        "C2_B0_AUDIT_PASS": True,
        "residual_seal": persistence["seal"],
        "residual_seal_file_sha256": persistence["seal_file_sha256"],
        "durable_residual_seal_verified_before_GT": True,
        "GT_loaded_before_residual_seal": False,
        "GT_loaded_after_residual_seal": True,
        "c0_audit_loaded_before_residual_seal": False,
        "c0_audit_loaded_after_residual_seal": True,
        "Bridge_correctness_loaded_before_residual_seal": False,
        "Bridge_correctness_built_after_residual_seal": True,
        "corruption_mask_loaded": False,
        "training_performed": False,
        "backward_called": False,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "model_forward_called": False,
        "C_conf_used_for_residual_propagation": False,
        "C_conf_used_for_postseal_stratification": True,
        "C2_specific_GT_mapping_fit": False,
        "direction_specific_mapping_fit": False,
        "bin_specific_mapping_fit": False,
        "evaluation_only_unlabeled": True,
        "evaluation_sample_count": c2.UNLABELED_EVAL_COUNT,
        "labeled_evaluation_intersection_count": 0,
        "full_GT": evaluation["full_GT_audit"],
        "c0_audit": evaluation["c0_audit_provenance"],
        "frozen_C0_global_mapping": evaluation["mapping_audit"],
        "correctness_logical_sha256": evaluation["correctness_logical_sha256"],
        "e_GT_logical_sha256": evaluation["e_GT_logical_sha256"],
    }
    records = OrderedDict((
        ("c2_b0_confidence_strata_audit.json", strata),
        ("c2_b0_metrics_by_direction.json", metrics),
        ("c2_b0_seed_summary.json", evaluation["summary"]),
        ("c2_b0_audit.json", audit),
    ))
    for name, record in records.items():
        write_json_durable(output_root / name, record)
    _fsync_directory(output_root)
    _require(
        all(read_json(output_root / name) == record for name, record in records.items()),
        "C2-B0 post-seal result durable reload mismatch",
    )
    return records


def run_evaluation(seed, output_dir=None, full_gt_path=None):
    """Run one formal seed; callers must obtain separate formal authorization."""
    active_seed = c2.validate_seed(seed)
    paths = default_paths(active_seed, output_dir=output_dir)
    if full_gt_path is not None:
        paths = C2B0Paths(
            paths.c0_artifact_path, paths.c0_seal_path, paths.c0_audit_path,
            paths.label_split_dir, _resolve(full_gt_path), paths.output_dir,
        )
    preregistration = verify_preregistration_freeze()
    arrays, provenance = load_c0_inputs_before_gt(
        active_seed, paths.c0_artifact_path, paths.c0_seal_path
    )
    split = load_fixed_sparse_labels(paths.label_split_dir)
    bundle = build_pre_gt_bundle(
        active_seed, arrays, provenance, split, preregistration
    )
    persistence = persist_residual_bundle(bundle, paths.output_dir)
    evaluation = evaluate_after_residual_seal(bundle, persistence, paths)
    records = save_postseal_results(bundle, persistence, evaluation)
    print("C2_B0_SEED_PASS=" + str(evaluation["summary"]["C2_B0_SEED_PASS"]))
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
