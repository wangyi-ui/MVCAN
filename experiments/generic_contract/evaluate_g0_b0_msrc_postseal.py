"""Read-only, non-gating post-seal metrics for the frozen G0-B0 MSRC pilot."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import scipy.io
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import ClusteringTest
from weak_quality import ndarray_sha256

from .generic_final_core_adapter import validate_structural_pre_gt_seal


STAGE = "G0-B0-E0"
DATASET = "MSRC-v1"
TRAINING_SEED = 20
PARENT_COMMIT = "2bfacc6988eaae5a78db2dc9b49da03c22a30482"
PARENT_TAG = "g0-b0-msrc-structural-pilot-pre-gt-pass-20260916"

DATASET_FILE_SHA256 = (
    "38d89aa41ae984f0b026f4baa8b5dcb7c569c471e2fb42b73a0f6743398e9103"
)
PRE_GT_ARTIFACT_SHA256 = (
    "b04f107f6279839d534e55500fbb0f9aba94d7686944ab20f255337a511af234"
)
PRE_GT_AUDIT_SHA256 = (
    "7674a6d46f5092e20a9be8f166cb6d28c29f10422176f79aad259b20ce291b1e"
)
PRE_GT_SEAL_SHA256 = (
    "c2fd8b35ffe0a065dbc8df69b7488fd75d88d39c2d3a100e665cc014729fa50e"
)
PREDICTION_NDARRAY_SHA256 = (
    "d5679cff6f7e6faa6a269dd4deeb020e64e331511ed9cb6632be18459d48251d"
)
PREDICTION_TENSOR_SHA256 = (
    "2fda08173f5ae2254bc397666c9717976a3cb27cfa770195ae800eae1e6a3e83"
)

REQUIRED_PRE_GT_ARRAYS = {
    "sample_ids",
    "labeled_ids",
    "unlabeled_ids",
    "final_predictions",
    "q_local",
    "q_aligned",
    "M_v",
    "U_cycle",
    "PredRelation_true",
    "relation_balance_weights_true",
    "generator_membership",
    "verifier_membership",
}

PRE_GT_FALSE_FLAGS = (
    "FULL_GT_LOADED_DURING_TRAINING",
    "GT_USED_FOR_U",
    "GT_USED_FOR_RELATION",
    "METRICS_RUN",
    "ACC_RUN",
    "NMI_RUN",
    "ARI_RUN",
    "BACC_RUN",
    "full_GT_present_in_pre_gt_artifact",
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value):
    """Match the frozen MVCAN tensor logical hash without importing torch."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    for component in (
        str(array.dtype).encode("ascii"),
        ",".join(str(int(size)) for size in array.shape).encode("ascii"),
        array.tobytes(order="C"),
    ):
        digest.update(struct.pack(">Q", len(component)))
        digest.update(component)
    return digest.hexdigest()


def _read_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_exclusive(path, value):
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _verify_authoritative_pre_gt_files(artifact_path, audit_path, seal_path):
    observed = {
        "artifact": file_sha256(artifact_path),
        "audit": file_sha256(audit_path),
        "seal": file_sha256(seal_path),
    }
    _require(
        observed
        == {
            "artifact": PRE_GT_ARTIFACT_SHA256,
            "audit": PRE_GT_AUDIT_SHA256,
            "seal": PRE_GT_SEAL_SHA256,
        },
        "authoritative pre-GT file SHA256 mismatch",
    )
    return observed


def _verify_pre_gt_isolation(audit, seal):
    for record_name, record in (("audit", audit), ("seal", seal)):
        _require(
            all(record.get(name) is False for name in PRE_GT_FALSE_FLAGS),
            record_name + " violates pre-GT isolation",
        )
        _require(
            record.get("Gate6_A_through_O_pass") is True,
            record_name + " does not bind Gate6 A-through-O PASS",
        )
        forbidden = record.get("forbidden_paths")
        _require(
            isinstance(forbidden, dict)
            and forbidden
            and all(value is False for value in forbidden.values()),
            record_name + " has a forbidden scientific path",
        )


def _verify_frozen_predictions(arrays):
    _require(
        REQUIRED_PRE_GT_ARRAYS.issubset(arrays),
        "validated pre-GT arrays are incomplete",
    )
    sample_ids = np.asarray(arrays["sample_ids"])
    _require(
        sample_ids.dtype == np.dtype(np.int64)
        and sample_ids.shape == (210,)
        and np.array_equal(sample_ids, np.arange(210, dtype=np.int64)),
        "sample IDs do not equal int64 arange(210)",
    )
    predictions = np.asarray(arrays["final_predictions"])
    _require(predictions.shape == (210,), "frozen prediction shape mismatch")
    _require(np.isfinite(predictions).all(), "frozen predictions are non-finite")
    _require(
        np.issubdtype(predictions.dtype, np.integer)
        or np.equal(predictions, np.rint(predictions)).all(),
        "frozen predictions are not integer-compatible",
    )
    prediction_ndarray_hash = ndarray_sha256(predictions)
    prediction_tensor_hash = _tensor_sha256(predictions)
    _require(
        prediction_ndarray_hash == PREDICTION_NDARRAY_SHA256,
        "frozen prediction ndarray hash mismatch",
    )
    _require(
        prediction_tensor_hash == PREDICTION_TENSOR_SHA256,
        "frozen prediction tensor hash mismatch",
    )
    return predictions, prediction_ndarray_hash, prediction_tensor_hash


def _normalize_gt(mat):
    _require("gt" in mat, "MSRC gt variable missing")
    raw_gt = np.squeeze(np.asarray(mat["gt"]))
    _require(
        raw_gt.shape == (210,) and np.isfinite(raw_gt).all(),
        "MSRC raw gt boundary mismatch",
    )
    _, labels = np.unique(raw_gt, return_inverse=True)
    labels = labels.astype(np.int64, copy=False)
    _require(
        labels.shape == (210,)
        and np.array_equal(np.unique(labels), np.arange(7, dtype=np.int64)),
        "MSRC normalized gt boundary mismatch",
    )
    return np.ascontiguousarray(labels)


def _compute_metrics(labels, predictions):
    result = {
        "ACC": float(ClusteringTest.acc(labels, predictions)),
        "NMI": float(normalized_mutual_info_score(labels, predictions)),
        "ARI": float(adjusted_rand_score(labels, predictions)),
    }
    _require(all(np.isfinite(value) for value in result.values()), "non-finite metric")
    _require(0.0 <= result["ACC"] <= 1.0, "ACC outside [0,1]")
    _require(0.0 <= result["NMI"] <= 1.0, "NMI outside [0,1]")
    _require(-1.0 <= result["ARI"] <= 1.0, "ARI outside sklearn range")
    return result


def evaluate_postseal(
    artifact_path,
    audit_path,
    seal_path,
    dataset_path,
    output_dir,
):
    """Validate frozen state, then load only GT and emit non-gating metrics."""
    artifact_path = Path(artifact_path)
    audit_path = Path(audit_path)
    seal_path = Path(seal_path)
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    _require(not output_dir.exists(), "refusing to overwrite evaluation output")

    access_order = []

    # 1. Whole-file identity is established before the seal is interpreted.
    pre_gt_hashes = _verify_authoritative_pre_gt_files(
        artifact_path, audit_path, seal_path
    )
    access_order.append("pre_gt_file_hashes_verified")

    # 2. This is the frozen, read-only seal validator used by G0-B0 itself.
    arrays, seal = validate_structural_pre_gt_seal(
        artifact_path, audit_path, seal_path
    )
    access_order.append("pre_gt_seal_validated")

    # 3. Isolation is checked in both independently frozen metadata records.
    audit = _read_json(audit_path)
    _verify_pre_gt_isolation(audit, seal)
    access_order.append("pre_gt_isolation_verified")

    # 4-5. Sample and prediction identities are checked while GT is unavailable.
    predictions, prediction_ndarray_hash, prediction_tensor_hash = (
        _verify_frozen_predictions(arrays)
    )
    access_order.extend(("sample_ids_verified", "prediction_hashes_verified"))

    # 6. Only the authoritative dataset may cross the subsequent GT boundary.
    dataset_hash = file_sha256(dataset_path)
    _require(dataset_hash == DATASET_FILE_SHA256, "authoritative dataset SHA mismatch")
    access_order.append("dataset_file_hash_verified")

    # 7. This is the first and only full-GT access in the evaluator.
    mat = scipy.io.loadmat(dataset_path, variable_names=["gt"])
    access_order.append("gt_loaded")

    # 8-9. GT is normalized only for the three preregistered metrics.
    labels = _normalize_gt(mat)
    access_order.append("gt_normalized")
    gt_hash = ndarray_sha256(labels)
    metric_values = _compute_metrics(labels, predictions)
    access_order.append("metrics_computed")

    metrics = {
        "stage": STAGE,
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "evaluation_population": "all_210_samples",
        "N": 210,
        "K": 7,
        "labeled_anchor_count": 14,
        "unlabeled_count": 196,
        **metric_values,
        "metrics_are_G0_gate": False,
        "Gate6_status": "PASS",
        "hyperparameter_tuning": False,
    }
    evaluation_audit = {
        "stage": STAGE,
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "parent_commit": PARENT_COMMIT,
        "parent_tag": PARENT_TAG,
        "dataset_file_sha256": dataset_hash,
        "pre_gt_artifact_sha256": pre_gt_hashes["artifact"],
        "pre_gt_audit_sha256": pre_gt_hashes["audit"],
        "pre_gt_seal_sha256": pre_gt_hashes["seal"],
        "pre_gt_seal_valid": True,
        "Gate6_A_through_O_pass": True,
        "FULL_GT_LOADED_DURING_TRAINING": False,
        "GT_LOADED_ONLY_AFTER_PRE_GT_SEAL_VERIFIED": True,
        "prediction_shape": list(predictions.shape),
        "prediction_ndarray_sha256": prediction_ndarray_hash,
        "prediction_tensor_sha256": prediction_tensor_hash,
        "normalized_GT_shape": list(labels.shape),
        "normalized_GT_class_count": int(np.unique(labels).size),
        "normalized_GT_ndarray_sha256": gt_hash,
        "evaluation_population": "all_210_samples",
        "labeled_anchor_count": 14,
        "unlabeled_count": 196,
        **metric_values,
        "GT_used_only_for_metrics": True,
        "MODEL_EXECUTED": False,
        "TRAINING_RUN": False,
        "OPTIMIZER_RUN": False,
        "BACKWARD_RUN": False,
        "KMEANS_RUN": False,
        "U_RECOMPUTED": False,
        "RELATION_RECOMPUTED": False,
        "SPARSE_SPLIT_RECOMPUTED": False,
        "CORRUPTION_RECOMPUTED": False,
        "metrics_are_G0_gate": False,
        "Gate6_changed_by_metrics": False,
        "Gate6_status_before_evaluation": "PASS",
        "Gate6_status_after_evaluation": "PASS",
        "hyperparameter_tuning_authorized": False,
        "scientific_definition_change_authorized": False,
        "access_order": access_order,
    }

    # 10. Diagnostic output is additive and the directory is never reused.
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "msrc_postseal_metrics.json"
    evaluation_audit_path = output_dir / "msrc_postseal_evaluation_audit.json"
    evaluation_seal_path = output_dir / "msrc_postseal_evaluation_seal.json"
    _write_json_exclusive(metrics_path, metrics)
    _write_json_exclusive(evaluation_audit_path, evaluation_audit)
    evaluation_seal = {
        "stage": STAGE,
        "dataset": DATASET,
        "metrics_file_sha256": file_sha256(metrics_path),
        "audit_file_sha256": file_sha256(evaluation_audit_path),
        "parent_pre_gt_artifact_sha256": pre_gt_hashes["artifact"],
        "parent_pre_gt_seal_sha256": pre_gt_hashes["seal"],
        "prediction_ndarray_sha256": prediction_ndarray_hash,
        "prediction_tensor_sha256": prediction_tensor_hash,
        "dataset_file_sha256": dataset_hash,
        "normalized_GT_ndarray_sha256": gt_hash,
        "postseal_evaluation_valid": True,
        "Gate6_remains_PASS": True,
        "metrics_non_gating": True,
        "scientific_core_unchanged": True,
        "training_not_run": True,
    }
    _write_json_exclusive(evaluation_seal_path, evaluation_seal)
    return {
        "metrics": metrics,
        "audit": evaluation_audit,
        "seal": evaluation_seal,
        "paths": {
            "metrics": metrics_path,
            "audit": evaluation_audit_path,
            "seal": evaluation_seal_path,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--audit-path", required=True)
    parser.add_argument("--seal-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args(argv)
    result = evaluate_postseal(
        arguments.artifact_path,
        arguments.audit_path,
        arguments.seal_path,
        arguments.dataset_path,
        arguments.output_dir,
    )
    print(json.dumps(result["metrics"], sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
