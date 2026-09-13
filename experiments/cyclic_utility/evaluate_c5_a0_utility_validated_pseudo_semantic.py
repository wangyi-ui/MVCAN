"""Evaluate C5-A0 only after validating its immutable pre-GT seal."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    normalized_mutual_info_score,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import c3_b0_true_u_carrier_protocol as carrier_protocol
from experiments.cyclic_utility import c5_a0_utility_validated_pseudo_semantic_protocol as c5
from experiments.cyclic_utility import train_c3_b0_relation_action_pilot as c3_train
from experiments.e1_pairwise_utility import train_e1_pairwise_utility as e1_train


def _require(condition, message="C5_A0_EVALUATION_FAIL_CLOSED"):
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


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, record):
    with open(path, "x", encoding="utf-8") as output_file:
        json.dump(record, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def default_paths(seed):
    active_seed = c5.validate_seed(seed)
    root = REPOSITORY_ROOT / "outputs/cyclic_utility" / (
        "c5_a0_utility_validated_pseudo_semantic_seed" + str(active_seed)
    )
    return {
        "artifact": root / "c5_pre_gt_bundle.npz",
        "audit": root / "c5_pre_gt_audit.json",
        "seal": root / "c5_pre_gt_seal.json",
        "metrics": root / "c5_metrics.json",
    }


def validate_pre_gt_seal(seed, artifact_path, audit_path, seal_path):
    active_seed = c5.validate_seed(seed)
    artifact = _resolve(artifact_path)
    audit_file = _resolve(audit_path)
    seal_file = _resolve(seal_path)
    _require(
        artifact.is_file() and audit_file.is_file() and seal_file.is_file(),
        "C5_A0_SEALED_PRE_GT_ARTIFACT_REQUIRED",
    )
    seal = _read_json(seal_file)
    audit = _read_json(audit_file)
    _require(
        seal.get("stage") == c5.STAGE
        and int(seal.get("seed", -1)) == active_seed
        and seal.get("pre_gt_seal_valid") is True
        and seal.get("GT_loaded_before_pre_gt_seal") is False
        and seal.get("training_performed") is False
        and seal.get("C4_used_as_parent") is False
        and seal.get("all_parent_hashes_pass") is True
        and seal.get("carrier_seal_verified") is True
        and seal.get("array_whitelist") == list(c5.PRE_GT_ARRAY_KEYS)
        and carrier_protocol.file_sha256(artifact) == seal.get("artifact_file_sha256")
        and carrier_protocol.file_sha256(audit_file) == seal.get("audit_file_sha256"),
        "C5_A0_PRE_GT_SEAL_FAIL_CLOSED",
    )
    _require(
        audit.get("stage") == c5.STAGE
        and int(audit.get("seed", -1)) == active_seed
        and audit.get("GT_loaded") is False
        and audit.get("training_performed") is False
        and audit.get("C4_used_as_parent") is False
        and audit.get("all_parent_hashes_pass") is True
        and audit.get("pre_gt_array_whitelist") == list(c5.PRE_GT_ARRAY_KEYS)
        and audit.get("same_eligible_set_for_all_arms") is True
        and audit.get("uniform_fallback_for_abstained_rows") is False,
        "C5_A0_PRE_GT_AUDIT_FAIL_CLOSED",
    )
    with np.load(artifact, allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == c5.PRE_GT_ARRAY_KEYS,
            "C5_A0_PRE_GT_ARRAY_SCHEMA_FAIL_CLOSED",
        )
        arrays = {
            key: np.array(archive[key], copy=True, order="C")
            for key in c5.PRE_GT_ARRAY_KEYS
        }
    c5.validate_pre_gt_array_keys(arrays)
    for key, value in arrays.items():
        record = seal.get("arrays", {}).get(key, {})
        _require(
            list(value.shape) == record.get("shape")
            and str(value.dtype) == record.get("dtype")
            and c5.logical_sha256(value) == record.get("logical_sha256"),
            "C5_A0_PRE_GT_ARRAY_HASH_FAIL_CLOSED",
        )
    eligible_count = int(arrays["eligible_ids"].size)
    abstention_count = int(arrays["abstained_ids"].size)
    expected_mask = arrays["utility_mass"] > c5.EPSILON
    rebuilt_permutation = c5.build_permuted_utility(
        arrays["U_true_eligible"], arrays["eligible_ids"]
    )
    _require(
        np.array_equal(arrays["sample_ids"], np.arange(c5.N, dtype=np.int64))
        and arrays["unlabeled_ids"].shape == (c5.NU,)
        and arrays["utility_mass"].shape == (c5.NU,)
        and arrays["eligible_mask"].shape == (c5.NU,)
        and arrays["eligible_mask"].dtype == np.dtype(np.bool_)
        and eligible_count + abstention_count == c5.NU
        and np.array_equal(arrays["eligible_mask"], expected_mask)
        and np.array_equal(arrays["eligible_ids"], arrays["unlabeled_ids"][expected_mask])
        and np.array_equal(arrays["abstained_ids"], arrays["unlabeled_ids"][~expected_mask])
        and int(audit.get("Nu", -1)) == c5.NU
        and int(audit.get("eligible_count", -1)) == eligible_count
        and int(audit.get("abstention_count", -1)) == abstention_count
        and float(audit.get("eligibility_rate", -1.0)) == eligible_count / c5.NU
        and float(audit.get("abstention_rate", -1.0)) == abstention_count / c5.NU
        and int(seal.get("Nu", -1)) == c5.NU
        and int(seal.get("eligible_count", -1)) == eligible_count
        and int(seal.get("abstention_count", -1)) == abstention_count
        and seal.get("same_eligible_set_for_all_arms") is True
        and arrays["directional_pred"].shape == (c5.NU, c5.S)
        and arrays["eligible_directional_posterior"].shape == (eligible_count, c5.S, c5.K)
        and np.array_equal(
            arrays["eligible_directional_posterior"],
            arrays["directional_posterior"][expected_mask],
        )
        and arrays["U_true"].shape == (c5.NU, c5.S)
        and arrays["U_true_eligible"].shape == arrays["U_permuted"].shape == (eligible_count, c5.S)
        and np.array_equal(arrays["U_true_eligible"], arrays["U_true"][expected_mask])
        and np.all(arrays["U_true_eligible"].sum(axis=1) > c5.EPSILON)
        and np.all(arrays["U_permuted"].sum(axis=1) > c5.EPSILON)
        and np.array_equal(
            arrays["permutation_source_ids"],
            rebuilt_permutation["permutation_source_ids"],
        )
        and np.array_equal(arrays["U_permuted"], rebuilt_permutation["U_permuted"])
        and all(arrays["posterior_" + arm].shape == (eligible_count, c5.K) for arm in c5.ARMS)
        and all(arrays["prediction_" + arm].shape == (eligible_count,) for arm in c5.ARMS)
        and all(arrays["confidence_" + arm].shape == (eligible_count,) for arm in c5.ARMS),
        "C5_A0_EVALUATOR_SHAPE_FAIL_CLOSED",
    )
    return arrays, {
        "artifact_path": _display(artifact),
        "artifact_file_sha256": carrier_protocol.file_sha256(artifact),
        "audit_path": _display(audit_file),
        "audit_file_sha256": carrier_protocol.file_sha256(audit_file),
        "seal_path": _display(seal_file),
        "seal_file_sha256": carrier_protocol.file_sha256(seal_file),
        "pre_gt_seal_verified_before_GT_load": True,
    }


def sample_metrics(y_true, prediction):
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(prediction, dtype=np.int64)
    _require(truth.ndim == pred.ndim == 1 and truth.shape == pred.shape and truth.size > 0, "C5_A0_METRIC_SHAPE_FAIL_CLOSED")
    return {
        "ACC": float(accuracy_score(truth, pred)),
        "Balanced_ACC": float(balanced_accuracy_score(truth, pred)),
        "NMI": float(normalized_mutual_info_score(truth, pred)),
        "ARI": float(adjusted_rand_score(truth, pred)),
    }


def fixed_coverage_precision(y_true, prediction, confidence, sample_ids):
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(prediction, dtype=np.int64)
    ids = np.asarray(sample_ids, dtype=np.int64)
    _require(truth.ndim == pred.ndim == ids.ndim == 1
             and truth.shape == pred.shape == ids.shape
             and truth.size > 0,
             "C5_A0_COVERAGE_SHAPE_FAIL_CLOSED")
    row_for_id = {int(sample_id): row for row, sample_id in enumerate(ids)}
    output = {}
    for coverage in c5.COVERAGES:
        selected = c5.deterministic_coverage_selection(confidence, ids, coverage)
        rows = np.asarray([row_for_id[int(value)] for value in selected], dtype=np.int64)
        label = str(int(round(coverage * 100)))
        output["Precision@" + label] = float(np.mean(pred[rows] == truth[rows]))
        output["selection_count@" + label] = int(selected.size)
        output["effective_coverage_vs_all_unlabeled@" + label] = float(selected.size / c5.NU)
        output["selected_sample_ids_logical_sha256@" + label] = c5.logical_sha256(selected)
    return output


def evaluate(
    seed,
    artifact_path,
    audit_path,
    seal_path,
    full_gt_path,
    output_path,
):
    active_seed = c5.validate_seed(seed)
    output = _resolve(output_path)
    _require(not output.exists(), "refusing to overwrite C5-A0 metrics")
    arrays, pre_gt = validate_pre_gt_seal(
        active_seed, artifact_path, audit_path, seal_path
    )
    # The only GT load occurs after all artifact, audit, seal, schema, and hash checks.
    labels = e1_train.load_labels_after_predictions(
        _resolve(full_gt_path), _resolve(artifact_path)
    )
    labels = np.asarray(labels, dtype=np.int64)
    _require(labels.shape == (c5.N,), "C5_A0_FULL_GT_SHAPE_FAIL_CLOSED")
    eligible_ids = arrays["eligible_ids"]
    y_eligible = np.ascontiguousarray(labels[eligible_ids], dtype=np.int64)
    eligible_directional_pred = arrays["directional_pred"][arrays["eligible_mask"]]
    correct_action = np.ascontiguousarray(
        eligible_directional_pred == y_eligible[:, None], dtype=np.int8
    )
    auc_true = c5.binary_auc_fail_closed(correct_action, arrays["U_true_eligible"])
    auc_permuted = c5.binary_auc_fail_closed(correct_action, arrays["U_permuted"])
    arms = {}
    for arm in c5.ARMS:
        metrics = sample_metrics(y_eligible, arrays["prediction_" + arm])
        coverage = fixed_coverage_precision(
            y_eligible,
            arrays["prediction_" + arm],
            arrays["confidence_" + arm],
            eligible_ids,
        )
        arms[arm] = {**metrics, **coverage}
    record = {
        "stage": c5.STAGE,
        "seed": active_seed,
        "GT_loaded_only_after_pre_gt_seal": True,
        "training_performed": False,
        "bundle_modified": False,
        "Nu": c5.NU,
        "eligible_count": int(arrays["eligible_ids"].size),
        "abstention_count": int(arrays["abstained_ids"].size),
        "eligibility_rate": float(arrays["eligible_ids"].size / c5.NU),
        "abstention_rate": float(arrays["abstained_ids"].size / c5.NU),
        "identical_eligible_ids_used_for_all_arms": True,
        "eligible_ids_logical_sha256": c5.logical_sha256(arrays["eligible_ids"]),
        "pre_gt_provenance": pre_gt,
        "Action_Utility_Predictiveness": {
            "AUC_TRUE": auc_true,
            "AUC_PERMUTED": auc_permuted,
            "Delta_AUC": float(auc_true - auc_permuted),
        },
        "arms": arms,
        "Precision@20": {
            "TRUE_U": arms["TRUE_U"]["Precision@20"],
            "UNIFORM": arms["UNIFORM"]["Precision@20"],
            "PERMUTED_U": arms["PERMUTED_U"]["Precision@20"],
            "Delta_TRUE_U_UNIFORM": float(
                arms["TRUE_U"]["Precision@20"] - arms["UNIFORM"]["Precision@20"]
            ),
            "Delta_TRUE_U_PERMUTED_U": float(
                arms["TRUE_U"]["Precision@20"] - arms["PERMUTED_U"]["Precision@20"]
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, record)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=c5.SEEDS)
    defaults = default_paths(20)
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--audit-path", default=None)
    parser.add_argument("--seal-path", default=None)
    parser.add_argument("--full-gt-path", default=str(c3_train.DEFAULT_FULL_GT_PATH))
    parser.add_argument("--output-path", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = default_paths(args.seed)
    evaluate(
        args.seed,
        paths["artifact"] if args.artifact_path is None else args.artifact_path,
        paths["audit"] if args.audit_path is None else args.audit_path,
        paths["seal"] if args.seal_path is None else args.seal_path,
        args.full_gt_path,
        paths["metrics"] if args.output_path is None else args.output_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
