"""Evaluate a sealed C5-B0 training output; full GT is loaded only post-seal."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import c3_b0_true_u_carrier_protocol as carrier
from experiments.cyclic_utility import c5_b0_utility_validated_semantic_expansion_protocol as c5b0
from experiments.cyclic_utility import run_c5_b0_utility_validated_semantic_expansion as runner
from experiments.cyclic_utility import train_c3_b0_relation_action_pilot as c3_train
from experiments.e1_pairwise_utility import train_e1_pairwise_utility as e1_train


def _require(condition, message="C5_B0_EVALUATOR_FAIL_CLOSED"):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, record):
    with open(path, "x", encoding="utf-8") as output_file:
        json.dump(record, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def default_paths(seed, arm):
    root = runner.default_output_dir(seed, arm)
    return {
        "artifact": root / runner.ARTIFACT_NAME,
        "audit": root / runner.AUDIT_NAME,
        "seal": root / runner.SEAL_NAME,
        "metrics": root / "c5_b0_metrics.json",
    }


def validate_pre_gt_seal(seed, arm, artifact_path, audit_path, seal_path):
    active_seed = c5b0.validate_seed(seed)
    active_arm = c5b0.validate_arm(arm)
    artifact = _resolve(artifact_path)
    audit_file = _resolve(audit_path)
    seal_file = _resolve(seal_path)
    _require(
        artifact.is_file() and audit_file.is_file() and seal_file.is_file(),
        "C5_B0_SEALED_PRE_GT_OUTPUT_REQUIRED",
    )
    audit = _read_json(audit_file)
    seal = _read_json(seal_file)
    _require(
        seal.get("stage") == c5b0.STAGE
        and int(seal.get("seed", -1)) == active_seed
        and seal.get("arm") == active_arm
        and seal.get("pre_gt_seal_valid") is True
        and seal.get("GT_loaded_before_pre_gt_seal") is False
        and seal.get("all_parent_hashes_pass") is True
        and seal.get("BASE_replay_parity_pass") is True
        and seal.get("array_whitelist") == list(c5b0.PRE_GT_ARRAY_KEYS)
        and carrier.file_sha256(artifact) == seal.get("artifact_file_sha256")
        and carrier.file_sha256(audit_file) == seal.get("audit_file_sha256"),
        "C5_B0_PRE_GT_SEAL_FAIL_CLOSED",
    )
    _require(
        audit.get("stage") == c5b0.STAGE
        and int(audit.get("seed", -1)) == active_seed
        and audit.get("arm") == active_arm
        and audit.get("all_parent_hashes_pass") is True
        and audit.get("BASE_replay_parity_pass") is True
        and audit.get("GT_loaded_before_pre_gt_seal") is False
        and audit.get("C4_used") is False
        and audit.get("pseudo_CE_used") is False
        and audit.get("continuous_U_weighting_used") is False
        and audit.get("memory_used") is False
        and audit.get("recursive_expansion_used") is False
        and audit.get("self_relation_count_used") == 0,
        "C5_B0_PRE_GT_AUDIT_FAIL_CLOSED",
    )
    with np.load(artifact, allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == c5b0.PRE_GT_ARRAY_KEYS,
            "C5_B0_PRE_GT_ARRAY_SCHEMA_FAIL_CLOSED",
        )
        arrays = {
            key: np.array(archive[key], copy=True, order="C")
            for key in c5b0.PRE_GT_ARRAY_KEYS
        }
    c5b0.validate_pre_gt_array_keys(arrays)
    for key, value in arrays.items():
        record = seal.get("arrays", {}).get(key, {})
        _require(
            list(value.shape) == record.get("shape")
            and str(value.dtype) == record.get("dtype")
            and c5b0.logical_sha256(value) == record.get("logical_sha256"),
            "C5_B0_PRE_GT_ARRAY_HASH_FAIL_CLOSED",
        )
    expansion_count = int(arrays["selected_pseudo_anchor_ids"].size)
    expected_count = 0 if active_arm == "BASE" else c5b0.expansion_count(
        arrays["eligible_ids"].size
    )
    _require(
        np.array_equal(arrays["sample_ids"], np.arange(c5b0.N, dtype=np.int64))
        and arrays["final_predictions"].shape == (c5b0.N,)
        and arrays["real_anchor_ids"].shape == (c5b0.L,)
        and arrays["real_anchor_labels"].shape == (c5b0.L,)
        and expansion_count == expected_count
        and arrays["selected_pseudo_anchor_labels"].shape == (expected_count,)
        and arrays["expansion_mask"].shape == (arrays["eligible_ids"].size,)
        and arrays["expansion_mask"].dtype == np.dtype(np.bool_)
        and arrays["pseudo_semantic_label"].shape == (
            0 if active_arm == "BASE" else arrays["eligible_ids"].size,
        )
        and int(np.count_nonzero(arrays["expansion_mask"])) == expected_count
        and arrays["combined_anchor_ids"].shape == (c5b0.L + expected_count,)
        and arrays["combined_anchor_labels"].shape == (c5b0.L + expected_count,)
        and arrays["combined_anchor_is_pseudo"].shape == (c5b0.L + expected_count,)
        and np.isin(
            arrays["selected_pseudo_anchor_ids"], arrays["eligible_ids"]
        ).all()
        and np.intersect1d(
            arrays["selected_pseudo_anchor_ids"], arrays["real_anchor_ids"]
        ).size == 0,
        "C5_B0_PRE_GT_ARRAY_BOUNDARY_FAIL_CLOSED",
    )
    return arrays, {
        "artifact_file_sha256": carrier.file_sha256(artifact),
        "audit_file_sha256": carrier.file_sha256(audit_file),
        "seal_file_sha256": carrier.file_sha256(seal_file),
        "pre_gt_seal_verified_before_GT_load": True,
        "BASE_replay_parity_pass": True,
    }


def align_clusters(labels, predictions):
    truth = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    _require(
        truth.shape == predicted.shape == (c5b0.N,),
        "C5_B0_METRIC_VECTOR_SHAPE_FAIL_CLOSED",
    )
    contingency = np.zeros((c5b0.K, c5b0.K), dtype=np.int64)
    for true_value, predicted_value in zip(truth, predicted):
        _require(
            0 <= true_value < c5b0.K and 0 <= predicted_value < c5b0.K,
            "C5_B0_METRIC_CLASS_RANGE_FAIL_CLOSED",
        )
        contingency[true_value, predicted_value] += 1
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    mapping = np.full(c5b0.K, -1, dtype=np.int64)
    mapping[columns] = rows
    return mapping[predicted]


def evaluate(seed, arm, artifact_path, audit_path, seal_path, full_gt_path, output_path):
    active_seed = c5b0.validate_seed(seed)
    active_arm = c5b0.validate_arm(arm)
    output = _resolve(output_path)
    _require(not output.exists(), "refusing to overwrite C5-B0 metrics")
    arrays, provenance = validate_pre_gt_seal(
        active_seed, active_arm, artifact_path, audit_path, seal_path
    )
    # Sole GT boundary: every pre-GT file, logical array, and BASE gate is sealed.
    labels = e1_train.load_labels_after_predictions(
        _resolve(full_gt_path), _resolve(artifact_path)
    )
    labels = np.asarray(labels, dtype=np.int64)
    predictions = arrays["final_predictions"]
    metrics = e1_train.evaluate_predictions(labels, predictions)
    metrics["Balanced_ACC"] = float(
        balanced_accuracy_score(labels, align_clusters(labels, predictions))
    )
    record = {
        "stage": c5b0.STAGE,
        "seed": active_seed,
        "arm": active_arm,
        "primary_metric": c5b0.PRIMARY_METRIC,
        "secondary_metrics": list(c5b0.SECONDARY_METRICS),
        "metrics": metrics,
        "GT_loaded_only_after_pre_gt_seal": True,
        "training_performed_by_evaluator": False,
        "bundle_modified": False,
        "pre_gt_provenance": provenance,
        "formal_multiseed_decision_made": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, record)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=c5b0.ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=c5b0.SEEDS)
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--audit-path", default=None)
    parser.add_argument("--seal-path", default=None)
    parser.add_argument("--full-gt-path", default=str(c3_train.DEFAULT_FULL_GT_PATH))
    parser.add_argument("--output-path", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = default_paths(args.seed, args.arm)
    evaluate(
        args.seed, args.arm,
        paths["artifact"] if args.artifact_path is None else args.artifact_path,
        paths["audit"] if args.audit_path is None else args.audit_path,
        paths["seal"] if args.seal_path is None else args.seal_path,
        args.full_gt_path,
        paths["metrics"] if args.output_path is None else args.output_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
