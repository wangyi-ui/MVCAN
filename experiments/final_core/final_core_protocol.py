"""Fail-closed protocol primitives for F0-A0 Final-Core Exact Replay.

F0-A0 adds no scientific mechanism.  Its only permitted training path is the
frozen C3-B0 ``TRUE_U`` path: sparse labels define relation semantics and the
frozen directional ``U_cycle`` defines action validity.
"""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from experiments.cyclic_utility import c3_b0_relation_action_protocol as c3b0
from experiments.cyclic_utility import c3_b0_true_u_carrier_protocol as carrier


STAGE = "F0-A0"
PURPOSE = "FINAL_CORE_EXACT_REPLAY_OF_FROZEN_C3_B0_TRUE_U"
SCIENTIFIC_LINEAGE = ("C0", "C3-A0", "C3-B0 TRUE_U", "F0-A0")
SEEDS = (20, 30, 50)
N = 1400
V = 6
K = 7
L = 14
NU = 1386
S = 20
EPOCHS = 20

PREREGISTERED_PROTOCOL_SHA256 = (
    "65fd6f66a09730bdf17b5fc0baf5e1f117ff9a6d6895acaefc8d999a1c857023"
)
FRAMEWORK_DECISION_SHA256 = (
    "af562c7cd42a241f5d026393685ba9fc578b73c5e760d00daa30721bc28c15da"
)
C3_SOURCE_MANIFEST_SHA256 = (
    "2f6584e568bb3c7a0ab1539c78c94c192f38570fdb787dae63cd630fe7dedc5c"
)
C3_FORMAL_MANIFEST_SHA256 = (
    "5a55875f4aa6cb4f1252e53a2150d5de217b2294811d94d2b15ffd8434f8dd08"
)
CARRIER_SOURCE_MANIFEST_SHA256 = (
    "edc29636763fc569ba6249708b0f58fec93775580bf44897187950f90f0829c6"
)
CARRIER_OUTPUT_MANIFEST_SHA256 = (
    "3b0c8f8ce5ddf7e7283e01d44907fe6220a04fe99effe26f5b1b11dac5549bab"
)

FROZEN_METRICS = {
    20: {"ACC": 0.8614285714285714, "NMI": 0.7734181958700173,
         "ARI": 0.753482395557614},
    30: {"ACC": 0.8592857142857143, "NMI": 0.7703779562016145,
         "ARI": 0.749533486243257},
    50: {"ACC": 0.8607142857142858, "NMI": 0.7763024116674858,
         "ARI": 0.7538086225874288},
}

FORBIDDEN_FLAGS = (
    "c4_memory_used",
    "c5_a0_training_used",
    "c5_b0_used",
    "pseudo_label_used",
    "pseudo_CE_used",
    "memory_used",
    "prototype_memory_used",
    "continuous_U_weighting_used",
    "scalar_U_recalibration_used",
    "feature_gate_used",
    "fusion_gate_used",
    "recursive_expansion_used",
    "new_loss_term_used",
    "new_loss_coefficient_used",
)

FROZEN_INPUT_HASH_NAMES = (
    "U_cycle", "PredRelation_true", "relation_balance_weights_true",
)
PARENT_LINEAGE_HASH_NAMES = (
    "F0_preregistered_protocol",
    "final_core_framework_decision",
    "C3_B0_source_manifest",
    "C3_B0_formal_manifest",
    "C3_B0_carrier_source_manifest",
    "C3_B0_carrier_output_manifest",
)
PRE_GT_ARRAY_KEYS = (
    "sample_ids",
    "final_predictions",
    "labeled_ids",
    "q_local",
    "q_aligned",
    "M_v",
    "frozen_input_logical_hashes",
    "parent_lineage_hashes",
    "final_model_hash_aggregate",
    "reference_prediction_file_sha256",
)


def _require(condition, message="F0_A0_FAIL_CLOSED"):
    if not condition:
        raise RuntimeError(message)


def validate_seed(seed):
    value = int(seed)
    if value not in SEEDS:
        raise ValueError("F0-A0 seed must be one of " + str(SEEDS))
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with open(Path(path), "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def logical_sha256(value):
    return carrier.logical_sha256(value)


def verify_pinned_file(path, expected_sha256, failure):
    target = Path(path)
    _require(target.is_file(), failure)
    actual = file_sha256(target)
    _require(actual == expected_sha256, failure)
    return {
        "path": str(target),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual,
        "exact_match": True,
    }


def default_forbidden_flags():
    return OrderedDict((name, False) for name in FORBIDDEN_FLAGS)


def validate_forbidden_flags(flags):
    _require(
        isinstance(flags, dict) and set(flags) == set(FORBIDDEN_FLAGS),
        "F0_A0_FORBIDDEN_FLAG_SCHEMA_FAIL_CLOSED",
    )
    _require(
        all(flags[name] is False for name in FORBIDDEN_FLAGS),
        "F0_A0_FORBIDDEN_PATH_ENABLED_FAIL_CLOSED",
    )
    return OrderedDict((name, False) for name in FORBIDDEN_FLAGS)


def validate_parent_hash_summary(summary):
    required = (
        "all_parent_hashes_pass",
        "frozen_C3_B0_source_hashes_pass",
        "frozen_C3_B0_formal_output_hashes_pass",
        "frozen_carrier_source_hashes_pass",
        "frozen_carrier_output_hashes_pass",
    )
    _require(
        isinstance(summary, dict)
        and all(summary.get(name) is True for name in required),
        "F0_A0_PARENT_HASH_FAIL_CLOSED",
    )
    return {name: True for name in required}


def validate_exact_array(name, actual, expected, shape):
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    _require(
        actual_array.shape == expected_array.shape == tuple(shape)
        and actual_array.dtype == expected_array.dtype
        and np.array_equal(actual_array, expected_array),
        "F0_A0_" + name.upper() + "_MISMATCH_FAIL_CLOSED",
    )
    return {
        "shape": list(actual_array.shape),
        "dtype": str(actual_array.dtype),
        "logical_sha256": logical_sha256(actual_array),
        "exact_equal": True,
    }


def validate_sample_ids(actual, expected):
    record = validate_exact_array("sample_ids", actual, expected, (N,))
    _require(
        np.array_equal(np.asarray(actual, dtype=np.int64), np.arange(N)),
        "F0_A0_SAMPLE_IDS_FAIL_CLOSED",
    )
    record["exact_arange"] = True
    return record


def validate_labeled_ids(actual, expected):
    return validate_exact_array("labeled_ids", actual, expected, (L,))


def validate_utility(U_cycle, expected_logical_sha256):
    # Frozen full-data directional utility: U_cycle[N,S] = [1400,20].
    utility = np.asarray(U_cycle)
    _require(
        utility.shape == (N, S)
        and np.issubdtype(utility.dtype, np.floating)
        and np.isfinite(utility).all()
        and np.all(utility >= 0.0),
        "F0_A0_UTILITY_SHAPE_FAIL_CLOSED",
    )
    actual_hash = logical_sha256(utility)
    _require(
        actual_hash == expected_logical_sha256,
        "F0_A0_UTILITY_LOGICAL_HASH_FAIL_CLOSED",
    )
    return {
        "shape": [N, S],
        "logical_sha256": actual_hash,
        "utility_shape_equal": True,
        "utility_logical_sha256_equal": True,
        "directional_form_preserved": True,
    }


def validate_training_action_view(full_U_cycle, unlabeled_ids, training_U_cycle):
    # Training consumes U_cycle[Nu,S] by canonical sample-ID lookup.
    full = np.asarray(full_U_cycle)
    ids = np.asarray(unlabeled_ids, dtype=np.int64)
    training = np.asarray(training_U_cycle)
    _require(
        ids.shape == (NU,)
        and training.shape == (NU, S)
        and np.array_equal(training, full[ids]),
        "F0_A0_UTILITY_SAMPLE_ID_MAPPING_FAIL_CLOSED",
    )
    return {
        "training_U_cycle_shape": [NU, S],
        "selected_by_canonical_sample_id": True,
        "exact_equal_to_full_U_at_unlabeled_ids": True,
    }


def validate_relation_inputs(
    relation_target, expected_target, balance_weights, expected_weights,
):
    relation = validate_exact_array(
        "relation_target", relation_target, expected_target, (NU, L, S)
    )
    balance = validate_exact_array(
        "relation_balance_weights", balance_weights, expected_weights,
        (NU, L, S),
    )
    _require(
        np.asarray(relation_target).dtype == np.bool_
        and np.isfinite(np.asarray(balance_weights)).all()
        and np.all(np.asarray(balance_weights) > 0.0),
        "F0_A0_RELATION_INPUT_BOUNDARY_FAIL_CLOSED",
    )
    return {
        "relation_target_equal": True,
        "relation_balance_weights_equal": True,
        "relation_target": relation,
        "relation_balance_weights": balance,
    }


def validate_coordinate_mapping(q_local, q_aligned, M_v):
    # q_local/q_aligned: [N,V,K]; M_v: [V,K,K] with [global,local].
    return carrier.validate_coordinate_snapshot(q_local, q_aligned, M_v)


def validate_replay_parity(
    expected_model_hash, replayed_model_hash, expected_predictions,
    replayed_predictions, expected_sample_ids, replayed_sample_ids,
):
    return carrier.validate_replay_parity(
        expected_model_hash, replayed_model_hash,
        expected_predictions, replayed_predictions,
        expected_sample_ids, replayed_sample_ids,
    )


def frozen_metrics(seed):
    return dict(FROZEN_METRICS[validate_seed(seed)])


def validate_metric_parity(actual, expected):
    _require(
        isinstance(actual, dict) and isinstance(expected, dict),
        "F0_A0_METRIC_SCHEMA_FAIL_CLOSED",
    )
    results = {}
    for name in c3b0.METRICS:
        actual_value = actual.get(name)
        expected_value = expected.get(name)
        _require(
            isinstance(actual_value, (int, float))
            and isinstance(expected_value, (int, float))
            and np.isfinite(actual_value)
            and np.isfinite(expected_value)
            and float(actual_value) == float(expected_value),
            "F0_A0_" + name + "_MISMATCH_FAIL_CLOSED",
        )
        results[name] = {
            "actual": float(actual_value),
            "expected": float(expected_value),
            "exact_equal": True,
        }
    return {
        "ACC_exact_equal": True,
        "NMI_exact_equal": True,
        "ARI_exact_equal": True,
        "metrics": results,
    }


def validate_pre_gt_arrays(arrays):
    _require(
        tuple(arrays) == PRE_GT_ARRAY_KEYS,
        "F0_A0_PRE_GT_ARRAY_SCHEMA_FAIL_CLOSED",
    )
    sample_ids = np.asarray(arrays["sample_ids"])
    predictions = np.asarray(arrays["final_predictions"])
    labeled_ids = np.asarray(arrays["labeled_ids"])
    q_local = np.asarray(arrays["q_local"])
    q_aligned = np.asarray(arrays["q_aligned"])
    M_v = np.asarray(arrays["M_v"])
    _require(
        sample_ids.shape == predictions.shape == (N,)
        and labeled_ids.shape == (L,)
        and q_local.shape == q_aligned.shape == (N, V, K)
        and M_v.shape == (V, K, K)
        and np.asarray(arrays["frozen_input_logical_hashes"]).shape
        == (len(FROZEN_INPUT_HASH_NAMES),)
        and np.asarray(arrays["parent_lineage_hashes"]).shape
        == (len(PARENT_LINEAGE_HASH_NAMES),)
        and np.asarray(arrays["final_model_hash_aggregate"]).shape == ()
        and np.asarray(arrays["reference_prediction_file_sha256"]).shape == ()
        and np.isfinite(q_local).all()
        and np.isfinite(q_aligned).all()
        and np.isfinite(M_v).all(),
        "F0_A0_PRE_GT_ARRAY_BOUNDARY_FAIL_CLOSED",
    )
    _require(
        np.array_equal(sample_ids.astype(np.int64), np.arange(N)),
        "F0_A0_PRE_GT_SAMPLE_IDS_FAIL_CLOSED",
    )
    return True


def array_records(arrays):
    validate_pre_gt_arrays(arrays)
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
            "logical_sha256": logical_sha256(np.asarray(value)),
        }
        for name, value in arrays.items()
    }


def build_pre_gt_seal(audit, artifact_path, audit_path):
    flags = validate_forbidden_flags(audit.get("forbidden_flags"))
    required_gates = (
        "parent_hashes_pass", "sample_ids_equal", "labeled_ids_equal",
        "utility_shape_equal", "utility_logical_sha256_equal",
        "relation_target_equal", "relation_balance_weights_equal",
        "coordinate_mapping_audit_pass", "final_sample_ids_equal",
        "final_predictions_equal", "final_model_hash_equal",
    )
    _require(
        audit.get("stage") == STAGE
        and audit.get("scientific_lineage") == list(SCIENTIFIC_LINEAGE)
        and audit.get("final_model_hash_supported") is True
        and audit.get("GT_loaded_before_pre_gt_seal") is False
        and audit.get("GT_loaded_during_training") is False
        and all(audit.get(name) is True for name in required_gates),
        "F0_A0_PRE_GT_SEAL_FAIL_CLOSED",
    )
    artifact = Path(artifact_path)
    audit_file = Path(audit_path)
    _require(artifact.is_file() and audit_file.is_file(),
             "F0_A0_PRE_GT_SEAL_INPUT_MISSING")
    return {
        "stage": STAGE,
        "seed": validate_seed(audit["seed"]),
        "purpose": PURPOSE,
        "scientific_lineage": list(SCIENTIFIC_LINEAGE),
        "pre_gt_seal_valid": True,
        "GT_loaded_before_pre_gt_seal": False,
        "GT_loaded_during_training": False,
        "all_parent_hashes_pass": True,
        "final_sample_ids_equal": True,
        "final_predictions_equal": True,
        "final_model_hash_supported": True,
        "final_model_hash_equal": True,
        "coordinate_mapping_audit_pass": True,
        "forbidden_flags": dict(flags),
        "array_whitelist": list(PRE_GT_ARRAY_KEYS),
        "arrays": audit["arrays"],
        "artifact_path": str(artifact),
        "artifact_file_sha256": file_sha256(artifact),
        "audit_path": str(audit_file),
        "audit_file_sha256": file_sha256(audit_file),
    }


def validate_pre_gt_seal(seed, artifact_path, audit_path, seal_path):
    active_seed = validate_seed(seed)
    artifact = Path(artifact_path)
    audit_file = Path(audit_path)
    seal_file = Path(seal_path)
    _require(
        artifact.is_file() and audit_file.is_file() and seal_file.is_file(),
        "F0_A0_VALID_PRE_GT_SEAL_REQUIRED",
    )
    with open(seal_file, "r", encoding="utf-8") as input_file:
        seal = json.load(input_file)
    with open(audit_file, "r", encoding="utf-8") as input_file:
        audit = json.load(input_file)
    _require(
        seal.get("stage") == STAGE
        and int(seal.get("seed", -1)) == active_seed
        and seal.get("pre_gt_seal_valid") is True
        and seal.get("GT_loaded_before_pre_gt_seal") is False
        and seal.get("all_parent_hashes_pass") is True
        and seal.get("final_predictions_equal") is True
        and seal.get("final_model_hash_supported") is True
        and seal.get("final_model_hash_equal") is True
        and seal.get("coordinate_mapping_audit_pass") is True
        and seal.get("array_whitelist") == list(PRE_GT_ARRAY_KEYS)
        and file_sha256(artifact) == seal.get("artifact_file_sha256")
        and file_sha256(audit_file) == seal.get("audit_file_sha256"),
        "F0_A0_PRE_GT_SEAL_FAIL_CLOSED",
    )
    validate_forbidden_flags(seal.get("forbidden_flags"))
    validate_forbidden_flags(audit.get("forbidden_flags"))
    with np.load(artifact, allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == PRE_GT_ARRAY_KEYS,
            "F0_A0_PRE_GT_ARRAY_SCHEMA_FAIL_CLOSED",
        )
        arrays = OrderedDict(
            (name, np.array(archive[name], copy=True, order="C"))
            for name in PRE_GT_ARRAY_KEYS
        )
    validate_pre_gt_arrays(arrays)
    for name, value in arrays.items():
        record = seal.get("arrays", {}).get(name, {})
        _require(
            record.get("shape") == list(value.shape)
            and record.get("dtype") == str(value.dtype)
            and record.get("logical_sha256") == logical_sha256(value),
            "F0_A0_PRE_GT_ARRAY_HASH_FAIL_CLOSED",
        )
    return arrays, {
        "pre_gt_seal_valid": True,
        "pre_gt_seal_verified_before_GT_load": True,
        "seal_file_sha256": file_sha256(seal_file),
        "artifact_file_sha256": file_sha256(artifact),
        "audit_file_sha256": file_sha256(audit_file),
    }
