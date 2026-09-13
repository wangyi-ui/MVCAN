"""Post-seal GT-only evaluator for C4-A0 memory diagnostics."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ClusteringTest import acc as clustering_accuracy
from experiments.cyclic_utility import (
    c4_a0_utility_conditioned_semantic_memory_protocol as c4,
)
from experiments.cyclic_utility import (
    run_c4_a0_utility_conditioned_semantic_memory as c4_run,
)
from experiments.cyclic_utility import train_vsa_a0_decoupled_action as vsa_train
from experiments.e1_pairwise_utility import train_e1_pairwise_utility as e1_train


STAGE = c4.STAGE
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/cyclic_utility"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_and_load_pre_gt_bundle(artifact_path, audit_path, seal_path):
    """Authenticate all pre-GT gates before the sole GT boundary is reachable."""
    artifact = Path(artifact_path)
    audit_file = Path(audit_path)
    seal_file = Path(seal_path)
    _require(artifact.is_file() and audit_file.is_file() and seal_file.is_file(),
             "sealed C4-A0 pre-GT artifact is required")
    with open(seal_file, "r", encoding="utf-8") as input_file:
        seal = json.load(input_file)
    _require(
        seal.get("stage") == STAGE
        and seal.get("GT_loaded_before_seal") is False
        and seal.get("carrier_parity_pass") is True
        and seal.get("all_snapshot_side_effect_audits_pass") is True
        and seal.get("semantic_identity_safety_pass") is True
        and seal.get("artifact_file_sha256") == c4_run.file_sha256(artifact)
        and seal.get("audit_file_sha256") == c4_run.file_sha256(audit_file)
        and tuple(seal.get("array_whitelist", ())) == c4.PRE_GT_ARRAY_WHITELIST,
        "C4-A0 pre-GT seal/parity boundary mismatch",
    )
    with open(audit_file, "r", encoding="utf-8") as input_file:
        audit = json.load(input_file)
    _require(
        audit.get("GT_loaded") is False
        and audit.get("carrier_parity", {}).get("carrier_parity_pass") is True
        and audit.get("all_snapshot_side_effect_audits_pass") is True
        and audit.get("semantic_identity_safety_pass") is True,
        "C4-A0 audit gate is not closed",
    )
    arrays = OrderedArrays()
    with np.load(artifact, allow_pickle=False) as archive:
        _require(tuple(archive.files) == c4.PRE_GT_ARRAY_WHITELIST,
                 "C4-A0 pre-GT archive whitelist mismatch")
        for name in c4.PRE_GT_ARRAY_WHITELIST:
            arrays[name] = np.array(archive[name], copy=True, order="C")
    _require(arrays["H"].shape == (c4.HOLDOUT_COUNT,),
             "C4-A0 sealed holdout shape mismatch")
    return arrays, audit, seal


class OrderedArrays(dict):
    """Insertion-preserving semantic name for authenticated NPZ arrays."""


def align_clusters_for_balanced_accuracy(labels, predictions):
    truth = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    _require(truth.shape == predicted.shape == (c4.HOLDOUT_COUNT,),
             "C4-A0 evaluator vector shape mismatch")
    contingency = np.zeros((c4.K, c4.K), dtype=np.int64)
    for true_value, predicted_value in zip(truth, predicted):
        _require(0 <= true_value < c4.K and 0 <= predicted_value < c4.K,
                 "C4-A0 evaluator class ID out of range")
        contingency[true_value, predicted_value] += 1
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    mapping = np.full((c4.K,), -1, dtype=np.int64)
    mapping[columns] = rows
    return mapping[predicted]


def balanced_clustering_accuracy(labels, predictions):
    truth = np.asarray(labels, dtype=np.int64)
    aligned = align_clusters_for_balanced_accuracy(truth, predictions)
    recalls = []
    for class_id in range(c4.K):
        mask = truth == class_id
        _require(bool(np.any(mask)), "C4-A0 holdout lacks a GT class")
        recalls.append(float(np.mean(aligned[mask] == class_id)))
    value = float(np.mean(recalls))
    _require(np.isfinite(value), "C4-A0 Balanced ACC non-finite")
    return value


def evaluate(seed, artifact_path=None, audit_path=None, seal_path=None,
             full_gt_path=vsa_train.DEFAULT_FULL_GT_PATH, output_path=None):
    active_seed = c4.validate_seed(seed)
    root = c4_run.default_output_dir(active_seed)
    artifact = root / "c4_pre_gt_bundle.npz" if artifact_path is None else Path(artifact_path)
    audit_file = root / "c4_pre_gt_audit.json" if audit_path is None else Path(audit_path)
    seal_file = root / "c4_pre_gt_seal.json" if seal_path is None else Path(seal_path)
    arrays, audit, seal = validate_and_load_pre_gt_bundle(
        artifact, audit_file, seal_file
    )
    # Sole full-GT boundary: authentication and carrier parity are complete.
    full_labels = e1_train.load_labels_after_predictions(full_gt_path, artifact)
    labels = full_labels[arrays["H"]]
    metrics = {}
    for arm in c4.MEMORY_ARMS:
        predictions = arrays["prediction_" + arm]
        metrics[arm] = {
            "Balanced_ACC": balanced_clustering_accuracy(labels, predictions),
            "ACC": float(clustering_accuracy(labels, predictions)),
            "NMI": float(normalized_mutual_info_score(labels, predictions)),
            "ARI": float(adjusted_rand_score(labels, predictions)),
        }
    result = {
        "stage": STAGE,
        "seed": active_seed,
        "primary_metric": "Balanced_ACC",
        "secondary_metrics": ["ACC", "NMI", "ARI"],
        "metrics": metrics,
        "pre_gt_artifact_file_sha256": seal["artifact_file_sha256"],
        "pre_gt_audit_file_sha256": seal["audit_file_sha256"],
        "carrier_parity_pass": audit["carrier_parity"]["carrier_parity_pass"],
        "GT_loaded_only_after_pre_gt_seal": True,
        "final_PASS_FAIL_decision_implemented": False,
    }
    destination = root / "c4_metrics.json" if output_path is None else Path(output_path)
    _require(not destination.exists(), "refusing to overwrite C4-A0 metrics")
    destination.parent.mkdir(parents=True, exist_ok=True)
    c4_run._write_json_durable(destination, result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=c4.SEEDS)
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--audit-path", default=None)
    parser.add_argument("--seal-path", default=None)
    parser.add_argument("--full-gt-path", default=str(vsa_train.DEFAULT_FULL_GT_PATH))
    parser.add_argument("--output-path", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    evaluate(args.seed, args.artifact_path, args.audit_path, args.seal_path,
             args.full_gt_path, args.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
