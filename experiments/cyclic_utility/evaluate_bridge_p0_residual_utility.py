"""Evaluate one Bridge-P0 seed without training or model loading."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import (
    bridge_p0_residual_utility_protocol as bridge,
)
from experiments.cyclic_utility import (
    c0_complementary_semantic_verification as c0_protocol,
)
from experiments.cyclic_utility import (
    evaluate_c0_complementary_semantic_verification as c0_evaluator,
)
from irv.b4_information_utility import tensor_sha256


STAGE = "Bridge-P0"
DATASET = "Caltech-6V"
C0_REQUIRED_DECISION = "C0_COMPLEMENTARY_CYCLE_VALIDITY_PASS"
REQUIRED_ARRAYS = (
    "y_gen",
    "U_cycle",
    "C_conf",
    "U_cycle_shuffle",
    "native_global_cluster",
)
FIXED_FEATURE_PATH = (
    REPOSITORY_ROOT
    / "outputs/e0_glgc_adapter/caltech6v_snr2p5_k3_seed20.npz"
)
DEFAULT_FULL_GT_PATH = c0_evaluator.DEFAULT_FULL_GT_PATH


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


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def default_c0_dir(seed):
    active_seed = bridge.validate_seed(seed)
    return (
        REPOSITORY_ROOT
        / "outputs/cyclic_utility"
        / ("c0_complementary_semantic_verification_seed" + str(active_seed))
    )


def default_e1_audit_path(seed):
    active_seed = bridge.validate_seed(seed)
    return (
        REPOSITORY_ROOT
        / "outputs/e1_pairwise_utility"
        / ("lwc_100ep_seed" + str(active_seed))
        / "e1_audit.json"
    )


def default_output_dir(seed):
    active_seed = bridge.validate_seed(seed)
    return (
        REPOSITORY_ROOT
        / "outputs/cyclic_utility"
        / ("bridge_p0_residual_utility_seed" + str(active_seed))
    )


def verify_frozen_source_integrity(repository_root=REPOSITORY_ROOT):
    root = Path(repository_root)
    actual = {}
    for relative_path, expected_hash in bridge.FROZEN_SOURCE_SHA256.items():
        path = root / relative_path
        _require(path.is_file(), "frozen scientific source missing: " + relative_path)
        actual_hash = file_sha256(path)
        _require(
            actual_hash == expected_hash,
            "frozen scientific source hash mismatch: " + relative_path,
        )
        actual[relative_path] = actual_hash
    return {
        "expected_sha256": dict(bridge.FROZEN_SOURCE_SHA256),
        "actual_sha256": actual,
        "all_existing_scientific_sources_unchanged_pass": True,
    }


def _audit_training_seed(e1_audit):
    value = e1_audit.get("training_seed", e1_audit.get("seed"))
    _require(value is not None, "E1 audit training seed is missing")
    active_seed = bridge.validate_seed(value)
    _require(
        int(e1_audit.get("seed", active_seed)) == active_seed,
        "E1 audit seed fields disagree",
    )
    return active_seed



def load_bridge_inputs_before_gt(
    seed,
    c0_artifact_path,
    c0_seal_path,
    c0_audit_path,
    c0_diagnostic_path,
    e1_audit_path,
):
    """Verify and reload the frozen C0/E1 lineage without loading GT."""
    active_seed = bridge.validate_seed(seed)
    artifact_path = _resolve(c0_artifact_path)
    seal_path = _resolve(c0_seal_path)
    audit_path = _resolve(c0_audit_path)
    diagnostic_path = _resolve(c0_diagnostic_path)
    e1_path = _resolve(e1_audit_path)
    for name, path in (
        ("C0 artifact", artifact_path),
        ("C0 seal", seal_path),
        ("C0 audit", audit_path),
        ("C0 diagnostic", diagnostic_path),
        ("E1 audit", e1_path),
    ):
        _require(path.is_file(), name + " is missing")

    artifact_hash = file_sha256(artifact_path)
    seal_hash = file_sha256(seal_path)
    audit_hash = file_sha256(audit_path)
    diagnostic_hash = file_sha256(diagnostic_path)
    e1_hash = file_sha256(e1_path)
    seal = read_json(seal_path)
    c0_audit = read_json(audit_path)
    diagnostic = read_json(diagnostic_path)
    e1_audit = read_json(e1_path)

    _require(
        c0_audit.get("stage") == "C0"
        and c0_audit.get("C0_AUDIT_PASS") is True
        and c0_audit.get("N") == bridge.SAMPLE_NUM
        and c0_audit.get("V") == bridge.VIEW_NUM
        and c0_audit.get("K") == bridge.CLASS_NUM
        and c0_audit.get("decision", {}).get("final_decision")
        == C0_REQUIRED_DECISION,
        "C0 audit/decision boundary mismatch",
    )
    _require(
        diagnostic.get("stage") == "C0"
        and diagnostic.get("decision", {}).get("final_decision")
        == C0_REQUIRED_DECISION,
        "C0 diagnostic decision boundary mismatch",
    )
    _require(
        seal.get("stage") == "C0"
        and seal.get("artifact_file_sha256") == artifact_hash
        and _same_path(seal.get("artifact_path", ""), artifact_path)
        and seal.get("scores_completed_before_GT") is True
        and seal.get("scores_saved_before_GT") is True
        and seal.get("scores_hashed_before_GT") is True
        and seal.get("scores_reloaded_before_GT") is True
        and seal.get("full_GT_loaded_before_seal") is False
        and seal.get("R_loaded_before_seal") is False
        and seal.get("sparse_labels_loaded_before_seal") is False
        and seal.get("corruption_mask_loaded_before_seal") is False
        and seal.get("oracle_loaded_before_seal") is False
        and c0_audit.get("prediction_seal") == seal,
        "C0 artifact own prediction-seal boundary mismatch",
    )

    e1_seed = _audit_training_seed(e1_audit)
    _require(
        e1_seed == active_seed
        and e1_audit.get("stage") == "E1"
        and e1_audit.get("arm") == "LWC"
        and e1_audit.get("epochs") == 100
        and e1_audit.get("N") == bridge.SAMPLE_NUM
        and e1_audit.get("V") == bridge.VIEW_NUM
        and e1_audit.get("K") == bridge.CLASS_NUM,
        "requested seed and formal E1 audit lineage mismatch",
    )
    representation = c0_audit.get("representation", {})
    checkpoint = representation.get("checkpoint", {})
    e1_final_hash = e1_audit.get("final_model_hash")
    e1_outputs = e1_audit.get("final_model_outputs", {})
    _require(
        representation.get("source_stage") == "E1"
        and representation.get("source_arm") == "LWC"
        and representation.get("source_epochs") == 100
        and checkpoint.get("arm") == "LWC"
        and checkpoint.get("epochs") == 100
        and _same_path(checkpoint.get("audit_path", ""), e1_path)
        and checkpoint.get("checkpoint_file_sha256")
        == e1_outputs.get("file_sha256")
        and representation.get("parameter_hash_before") == e1_final_hash
        and representation.get("parameter_hash_after") == e1_final_hash
        and representation.get("parameters_unchanged_pass") is True,
        "C0 source-model provenance does not match same-seed E1 audit",
    )
    c0_feature = representation.get("feature", {})
    e1_feature = e1_audit.get("feature_provenance", {})
    fixed_feature_hash = file_sha256(FIXED_FEATURE_PATH)
    _require(
        FIXED_FEATURE_PATH.is_file()
        and _same_path(c0_feature.get("path", ""), FIXED_FEATURE_PATH)
        and _same_path(e1_feature.get("path", ""), FIXED_FEATURE_PATH)
        and c0_feature.get("file_sha256") == fixed_feature_hash
        and e1_feature.get("file_sha256") == fixed_feature_hash
        and c0_feature.get("view_content_sha256")
        == e1_feature.get("view_content_sha256")
        and c0_feature.get("corruption_mask_present") is False
        and e1_feature.get("corruption_mask_present") is False,
        "fixed weak-quality feature lineage mismatch",
    )
    execution = c0_audit.get("execution", {})
    leakage = c0_audit.get("leakage", {})
    _require(
        execution.get("training_used") is False
        and execution.get("optimizer_used") is False
        and execution.get("backward_used") is False
        and execution.get("parameter_update_used") is False
        and leakage.get("R_loaded") is False
        and leakage.get("sparse_labels_loaded") is False
        and leakage.get("corruption_mask_loaded") is False
        and leakage.get("full_GT_loaded_before_seal") is False,
        "C0 leakage/execution boundary mismatch",
    )

    expected_shapes = {
        "y_gen": (bridge.SAMPLE_NUM, bridge.DIRECTION_COUNT),
        "U_cycle": (bridge.SAMPLE_NUM, bridge.DIRECTION_COUNT),
        "C_conf": (bridge.SAMPLE_NUM, bridge.DIRECTION_COUNT),
        "U_cycle_shuffle": (
            bridge.SAMPLE_NUM, bridge.DIRECTION_COUNT
        ),
        "native_global_cluster": (bridge.SAMPLE_NUM,),
    }
    expected_dtypes = {
        "y_gen": np.dtype(np.int64),
        "U_cycle": np.dtype(np.float32),
        "C_conf": np.dtype(np.float32),
        "U_cycle_shuffle": np.dtype(np.float32),
        "native_global_cluster": np.dtype(np.int64),
    }
    arrays = {}
    with np.load(artifact_path, allow_pickle=False) as archive:
        archive_fields = tuple(archive.files)
        _require(
            set(REQUIRED_ARRAYS).issubset(archive_fields),
            "C0 artifact is missing a required Bridge array",
        )
        for name in REQUIRED_ARRAYS:
            value = np.array(archive[name], copy=True, order="C")
            own_record = seal.get("arrays", {}).get(name, {})
            _require(
                value.shape == expected_shapes[name]
                and value.dtype == expected_dtypes[name]
                and np.isfinite(value).all()
                and own_record.get("shape") == list(value.shape)
                and own_record.get("dtype") == str(value.dtype)
                and own_record.get("logical_sha256")
                == tensor_sha256(value),
                "C0 required array own-seal mismatch: " + name,
            )
            value.setflags(write=False)
            arrays[name] = value
    _require(
        int(arrays["y_gen"].min()) >= 0
        and int(arrays["y_gen"].max()) < bridge.CLASS_NUM
        and int(arrays["native_global_cluster"].min()) >= 0
        and int(arrays["native_global_cluster"].max())
        < bridge.CLASS_NUM
        and np.array_equal(
            np.unique(arrays["native_global_cluster"]),
            np.arange(bridge.CLASS_NUM, dtype=np.int64),
        )
        and all(
            np.all((arrays[name] >= 0.0) & (arrays[name] <= 1.0))
            for name in ("U_cycle", "C_conf", "U_cycle_shuffle")
        ),
        "C0 required array value boundary mismatch",
    )
    mapping_record = c0_audit.get("global_mapping", {})
    mapping = np.asarray(mapping_record.get("mapping"), dtype=np.int64)
    _require(
        mapping.shape == (bridge.CLASS_NUM,)
        and np.array_equal(
            np.sort(mapping), np.arange(bridge.CLASS_NUM, dtype=np.int64)
        )
        and mapping_record.get("mapping_fit_count") == 1
        and mapping_record.get(
            "single_global_mapping_reused_all_directions"
        ) is True
        and mapping_record.get("direction_specific_mapping_used") is False
        and mapping_record.get("sparse_labels_used_for_mapping") is False,
        "C0 frozen global mapping boundary mismatch",
    )
    return arrays, {
        "seed": active_seed,
        "C0_artifact_path": _display(artifact_path),
        "C0_artifact_file_sha256": artifact_hash,
        "C0_seal_path": _display(seal_path),
        "C0_seal_file_sha256": seal_hash,
        "C0_audit_path": _display(audit_path),
        "C0_audit_file_sha256": audit_hash,
        "C0_diagnostic_path": _display(diagnostic_path),
        "C0_diagnostic_file_sha256": diagnostic_hash,
        "E1_audit_path": _display(e1_path),
        "E1_audit_file_sha256": e1_hash,
        "E1_training_seed": e1_seed,
        "E1_final_model_hash": e1_final_hash,
        "C0_source_model_hash": representation[
            "parameter_hash_before"
        ],
        "C0_E1_same_seed_lineage_pass": True,
        "legacy_C0_seed_field": c0_audit.get("seed"),
        "legacy_C0_seed_field_used_as_authority": False,
        "required_arrays": list(REQUIRED_ARRAYS),
        "archive_fields": list(archive_fields),
        "array_logical_sha256": {
            name: seal["arrays"][name]["logical_sha256"]
            for name in REQUIRED_ARRAYS
        },
        "C0_AUDIT_PASS": True,
        "C0_final_decision": C0_REQUIRED_DECISION,
        "own_prediction_seal_validation_pass": True,
        "all_required_arrays_finite_pass": True,
        "fixed_weak_quality_condition": bridge.WEAK_QUALITY_CONDITION,
        "fixed_feature_path": _display(FIXED_FEATURE_PATH),
        "fixed_feature_file_sha256": fixed_feature_hash,
        "frozen_global_mapping_logical_sha256": tensor_sha256(mapping),
        "GT_loaded": False,
        "R_loaded": False,
        "corruption_mask_loaded": False,
        "sparse_labels_loaded": False,
        "B7_artifact_loaded": False,
        "c0_audit": c0_audit,
    }



def bridge_source_hashes():
    return {
        "protocol": file_sha256(Path(bridge.__file__)),
        "evaluator": file_sha256(Path(__file__)),
    }


def save_bridge_input_seal(
    output_dir,
    provenance,
    source_integrity,
):
    """Durably seal verified Bridge inputs before the sole GT load."""
    output_root = Path(output_dir)
    seal_path = output_root / "bridge_input_seal.json"
    _require(
        output_root.is_dir() and not seal_path.exists(),
        "Bridge input seal output boundary mismatch",
    )
    payload = {
        "stage": STAGE,
        "seed": provenance["seed"],
        "dataset": DATASET,
        "N": bridge.SAMPLE_NUM,
        "V": bridge.VIEW_NUM,
        "K": bridge.CLASS_NUM,
        "direction_count": bridge.DIRECTION_COUNT,
        "required_arrays": list(REQUIRED_ARRAYS),
        "array_logical_sha256": provenance["array_logical_sha256"],
        "C0_artifact_path": provenance["C0_artifact_path"],
        "C0_artifact_file_sha256": provenance[
            "C0_artifact_file_sha256"
        ],
        "C0_seal_path": provenance["C0_seal_path"],
        "C0_seal_file_sha256": provenance["C0_seal_file_sha256"],
        "C0_audit_path": provenance["C0_audit_path"],
        "C0_audit_file_sha256": provenance["C0_audit_file_sha256"],
        "C0_diagnostic_path": provenance["C0_diagnostic_path"],
        "C0_diagnostic_file_sha256": provenance[
            "C0_diagnostic_file_sha256"
        ],
        "E1_audit_path": provenance["E1_audit_path"],
        "E1_audit_file_sha256": provenance["E1_audit_file_sha256"],
        "E1_training_seed": provenance["E1_training_seed"],
        "C0_E1_same_seed_lineage_pass": provenance[
            "C0_E1_same_seed_lineage_pass"
        ],
        "frozen_global_mapping_logical_sha256": provenance[
            "frozen_global_mapping_logical_sha256"
        ],
        "fixed_weak_quality_condition": bridge.WEAK_QUALITY_CONDITION,
        "fixed_feature_path": provenance["fixed_feature_path"],
        "fixed_feature_file_sha256": provenance[
            "fixed_feature_file_sha256"
        ],
        "bridge_source_sha256": bridge_source_hashes(),
        "frozen_source_integrity": source_integrity,
        "inputs_verified_before_GT": True,
        "input_seal_saved_before_GT": True,
        "input_seal_fsynced_before_GT": True,
        "GT_loaded_before_bridge_input_seal": False,
        "R_loaded": False,
        "corruption_mask_loaded": False,
        "sparse_labels_loaded": False,
        "B7_artifact_loaded": False,
        "training_entered": False,
        "new_utility_constructed": False,
    }
    write_json(seal_path, payload)
    _require(seal_path.is_file(), "Bridge input seal was not durable")
    reloaded = read_json(seal_path)
    _require(reloaded == payload, "Bridge input seal reload mismatch")
    return payload, {
        "path": _display(seal_path),
        "file_sha256": file_sha256(seal_path),
        "durable_before_GT_pass": True,
    }


def build_correctness_from_frozen_c0_mapping(
    arrays,
    c0_audit,
    full_GT,
):
    """Apply C0's one frozen mapping; never fit a Bridge mapping."""
    labels = np.ascontiguousarray(full_GT, dtype=np.int64)
    native = np.ascontiguousarray(
        arrays["native_global_cluster"], dtype=np.int64
    )
    mapping_record = c0_audit.get("global_mapping", {})
    mapping = np.ascontiguousarray(
        mapping_record.get("mapping"), dtype=np.int64
    )
    stored_contingency = np.asarray(
        mapping_record.get("contingency"), dtype=np.int64
    )
    _require(
        labels.shape == native.shape == (bridge.SAMPLE_NUM,)
        and mapping.shape == (bridge.CLASS_NUM,)
        and stored_contingency.shape
        == (bridge.CLASS_NUM, bridge.CLASS_NUM)
        and np.array_equal(
            np.sort(mapping), np.arange(bridge.CLASS_NUM, dtype=np.int64)
        )
        and mapping_record.get("mapping_fit_count") == 1
        and mapping_record.get("mapping_fit_after_score_seal") is True
        and mapping_record.get(
            "single_global_mapping_reused_all_directions"
        ) is True
        and mapping_record.get("direction_specific_mapping_used") is False,
        "frozen C0 mapping post-seal boundary mismatch",
    )
    contingency = np.zeros(
        (bridge.CLASS_NUM, bridge.CLASS_NUM), dtype=np.int64
    )
    np.add.at(contingency, (native, labels), 1)
    matched_count = int(
        contingency[
            np.arange(bridge.CLASS_NUM, dtype=np.int64), mapping
        ].sum()
    )
    _require(
        np.array_equal(contingency, stored_contingency)
        and matched_count == int(mapping_record.get("matched_count")),
        "frozen C0 mapping/GT conservation mismatch",
    )
    y_gen_semantic = c0_protocol.apply_global_mapping(
        arrays["y_gen"], mapping
    )
    correct = np.ascontiguousarray(
        y_gen_semantic == labels[:, None], dtype=np.bool_
    )
    _require(
        correct.shape
        == (bridge.SAMPLE_NUM, bridge.DIRECTION_COUNT),
        "Bridge diagnostic correctness shape mismatch",
    )
    correct.setflags(write=False)
    return correct, {
        "mapping_source": "C0_audit.global_mapping",
        "mapping": mapping.tolist(),
        "mapping_logical_sha256": tensor_sha256(mapping),
        "stored_contingency_conserved_pass": True,
        "stored_matched_count_conserved_pass": True,
        "C0_apply_global_mapping_reused": True,
        "Bridge_mapping_fit_count": 0,
        "Hungarian_fit_in_Bridge": False,
        "one_frozen_global_mapping_only": True,
        "no_per_direction_GT_mapping": True,
        "no_per_bin_GT_mapping": True,
        "correctness_definition": (
            "z[i,d] = 1 iff sealed y_gen[i,d] is correct under "
            "the frozen C0 global mapping"
        ),
        "correctness_logical_sha256": tensor_sha256(correct),
        "correctness_diagnostic_only": True,
    }


PER_SEED_OUTPUT_FILES = (
    "bridge_input_seal.json",
    "bridge_audit.json",
    "confidence_strata_audit.json",
    "conditional_auc_by_direction.json",
    "action_lift_by_direction.json",
    "bridge_seed_summary.json",
)


def load_bridge_full_ground_truth_after_seal(path):
    """Unpack and validate the exact structured C0 GT-loader result."""
    loaded = c0_evaluator.load_full_ground_truth_after_seal(path)
    _require(
        isinstance(loaded, tuple) and len(loaded) == 2,
        "C0 GT loader must return (labels, audit)",
    )
    labels_raw, gt_audit = loaded
    _require(
        isinstance(labels_raw, np.ndarray)
        and isinstance(gt_audit, dict),
        "C0 GT loader structured result schema mismatch",
    )
    labels_array = np.asarray(labels_raw)
    _require(
        labels_array.shape == (bridge.SAMPLE_NUM,)
        and np.issubdtype(labels_array.dtype, np.integer)
        and np.isfinite(labels_array).all()
        and np.array_equal(
            np.unique(labels_array),
            np.arange(bridge.CLASS_NUM, dtype=np.int64),
        ),
        "C0-normalized GT labels must be int [1400] with classes 0..6",
    )
    _require(
        gt_audit.get("shape") == [bridge.SAMPLE_NUM]
        and gt_audit.get("loaded_after_prediction_seal") is True
        and isinstance(gt_audit.get("path"), str)
        and isinstance(gt_audit.get("file_sha256"), str),
        "C0 GT loader audit schema mismatch",
    )
    labels = np.ascontiguousarray(labels_array, dtype=np.int64)
    labels.setflags(write=False)
    return labels, dict(gt_audit)



def run_evaluation(
    seed,
    c0_artifact_path,
    c0_seal_path,
    c0_audit_path,
    c0_diagnostic_path,
    e1_audit_path,
    full_gt_path=DEFAULT_FULL_GT_PATH,
    output_dir=None,
):
    """Run one read-only Bridge-P0 seed with an explicit GT boundary."""
    active_seed = bridge.validate_seed(seed)
    output_root = _resolve(
        default_output_dir(active_seed) if output_dir is None else output_dir
    )
    _require(
        not output_root.exists(),
        "refusing to overwrite Bridge-P0 seed output",
    )
    source_integrity = verify_frozen_source_integrity()
    arrays, provenance = load_bridge_inputs_before_gt(
        seed=active_seed,
        c0_artifact_path=c0_artifact_path,
        c0_seal_path=c0_seal_path,
        c0_audit_path=c0_audit_path,
        c0_diagnostic_path=c0_diagnostic_path,
        e1_audit_path=e1_audit_path,
    )
    c0_audit = provenance["c0_audit"]
    output_root.mkdir(parents=True, exist_ok=False)
    input_seal, input_seal_audit = save_bridge_input_seal(
        output_root,
        provenance,
        source_integrity,
    )

    # This is the sole GT load and must remain after the durable input seal.
    gt_path = _resolve(full_gt_path)
    full_GT, full_GT_audit = load_bridge_full_ground_truth_after_seal(
        gt_path
    )
    gt_hash = file_sha256(gt_path)
    c0_gt = c0_audit.get("postseal_ground_truth", {})
    _require(
        full_GT_audit == c0_gt
        and _same_path(full_GT_audit.get("path", ""), gt_path)
        and full_GT_audit.get("file_sha256") == gt_hash
        and c0_gt.get("loaded_after_prediction_seal") is True,
        "Bridge full GT does not match the frozen C0 post-seal GT",
    )
    correct, mapping_audit = build_correctness_from_frozen_c0_mapping(
        arrays,
        c0_audit,
        full_GT,
    )
    directional = bridge.analyze_all_directions(
        correct=correct,
        U_cycle=arrays["U_cycle"],
        C_conf=arrays["C_conf"],
        U_cycle_shuffle=arrays["U_cycle_shuffle"],
        sample_ids=np.arange(bridge.SAMPLE_NUM, dtype=np.int64),
    )
    seed_summary = bridge.build_seed_summary(
        active_seed,
        directional["conditional_auc"],
        directional["action_lift"],
    )
    confidence_strata = {
        "stage": STAGE,
        "seed": active_seed,
        "N": bridge.SAMPLE_NUM,
        "direction_count": bridge.DIRECTION_COUNT,
        "confidence_bin_count": bridge.CONFIDENCE_BIN_COUNT,
        "sample_ids_canonical": True,
        "sample_ids_logical_sha256": tensor_sha256(
            np.arange(bridge.SAMPLE_NUM, dtype=np.int64)
        ),
        **directional["confidence_strata"],
    }
    conditional_auc = {
        "stage": STAGE,
        "seed": active_seed,
        "shared_correctness_derived_valid_bin_mask": True,
        **directional["conditional_auc"],
    }
    action_lift = {
        "stage": STAGE,
        "seed": active_seed,
        "matched_within_confidence_quintile": True,
        **directional["action_lift"],
    }
    leakage = bridge.leakage_audit()
    audit = {
        "stage": STAGE,
        "seed": active_seed,
        "dataset": DATASET,
        "N": bridge.SAMPLE_NUM,
        "V": bridge.VIEW_NUM,
        "K": bridge.CLASS_NUM,
        "direction_count": bridge.DIRECTION_COUNT,
        "confidence_bin_count": bridge.CONFIDENCE_BIN_COUNT,
        "BRIDGE_AUDIT_PASS": True,
        "bridge_input_seal": input_seal,
        "bridge_input_seal_audit": input_seal_audit,
        "bridge_input_seal_reloaded_after_GT_pass": (
            read_json(output_root / "bridge_input_seal.json") == input_seal
        ),
        "source_integrity": source_integrity,
        "bridge_source_sha256": input_seal["bridge_source_sha256"],
        "input_provenance": {
            key: value
            for key, value in provenance.items()
            if key != "c0_audit"
        },
        "full_GT": {
            **full_GT_audit,
            "loaded_through_C0_loader": True,
            "loaded_after_bridge_input_seal": True,
            "C0_loader_audit_preserved": True,
        },
        "frozen_C0_global_mapping": mapping_audit,
        "correctness_diagnostic_only": True,
        "required_output_files": list(PER_SEED_OUTPUT_FILES),
        "fixed_weak_quality_condition": bridge.WEAK_QUALITY_CONDITION,
        **leakage,
    }
    _require(
        audit["bridge_input_seal_reloaded_after_GT_pass"]
        and audit["frozen_C0_global_mapping"][
            "one_frozen_global_mapping_only"
        ]
        and all(
            audit[key] is False
            for key in (
                "training_entered",
                "optimizer_created",
                "backward_called",
                "model_loaded_for_training",
                "R_loaded",
                "corruption_mask_loaded",
                "sparse_labels_loaded",
                "sparse_label_ids_loaded",
                "sparse_label_targets_loaded",
                "B7_artifact_loaded",
                "GT_loaded_before_bridge_input_seal",
                "new_utility_constructed",
                "VSA_modified",
                "C0_modified",
            )
        )
        and audit["GT_loaded_after_bridge_input_seal"] is True
        and audit["no_per_direction_GT_mapping"] is True
        and audit["no_per_bin_GT_mapping"] is True,
        "Bridge-P0 leakage audit boundary mismatch",
    )
    records = {
        "bridge_audit.json": audit,
        "confidence_strata_audit.json": confidence_strata,
        "conditional_auc_by_direction.json": conditional_auc,
        "action_lift_by_direction.json": action_lift,
        "bridge_seed_summary.json": seed_summary,
    }
    for name, record in records.items():
        write_json(output_root / name, record)
    _require(
        all((output_root / name).is_file() for name in PER_SEED_OUTPUT_FILES),
        "Bridge-P0 per-seed output schema incomplete",
    )
    print(
        "RESIDUAL_INFORMATION_SEED_PASS="
        + str(seed_summary["RESIDUAL_INFORMATION_SEED_PASS"])
    )
    print("SEED=" + str(active_seed))
    print("Saved: " + _display(output_root))
    return {
        "output_dir": output_root,
        "bridge_input_seal": input_seal,
        **records,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=bridge.SEEDS)
    parser.add_argument("--c0-artifact-path", default=None)
    parser.add_argument("--c0-seal-path", default=None)
    parser.add_argument("--c0-audit-path", default=None)
    parser.add_argument("--c0-diagnostic-path", default=None)
    parser.add_argument("--e1-audit-path", default=None)
    parser.add_argument("--full-gt-path", default=str(DEFAULT_FULL_GT_PATH))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    c0_dir = default_c0_dir(args.seed)
    if args.c0_artifact_path is None:
        args.c0_artifact_path = str(c0_dir / "c0_predictions_and_scores.npz")
    if args.c0_seal_path is None:
        args.c0_seal_path = str(c0_dir / "c0_prediction_seal.json")
    if args.c0_audit_path is None:
        args.c0_audit_path = str(c0_dir / "c0_audit.json")
    if args.c0_diagnostic_path is None:
        args.c0_diagnostic_path = str(c0_dir / "diagnostic_results.json")
    if args.e1_audit_path is None:
        args.e1_audit_path = str(default_e1_audit_path(args.seed))
    if args.output_dir is None:
        args.output_dir = str(default_output_dir(args.seed))
    return args


def main(argv=None):
    args = parse_args(argv)
    run_evaluation(
        seed=args.seed,
        c0_artifact_path=args.c0_artifact_path,
        c0_seal_path=args.c0_seal_path,
        c0_audit_path=args.c0_audit_path,
        c0_diagnostic_path=args.c0_diagnostic_path,
        e1_audit_path=args.e1_audit_path,
        full_gt_path=args.full_gt_path,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
