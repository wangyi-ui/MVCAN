"""Gate C: post-seal relation quality by frozen information utility."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.paper.transfer_audit.input_artifacts import file_sha256
from experiments.paper.transfer_audit import msrc_p0_protocol
from release_core.data import load_dataset
from release_core.utility import build_directional_actions

from . import p0_a3_protocol as protocol
from .msrc_g0b0_parity_audit import (
    reconstruct_current_msrc,
    validate_reconstruction_against_manifest,
    verify_current_true_u,
    verify_historical_msrc_files,
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _verify_files(expected):
    observed = {}
    for path, digest in expected.items():
        _require(path.is_file(), "frozen input missing: " + str(path))
        actual = file_sha256(path)
        _require(actual == digest, "frozen input SHA256 mismatch: " + str(path))
        observed[str(path)] = actual
    return observed


def _weighted_quality(correct, weights, utility=None):
    effective = weights if utility is None else weights * utility[:, None, :]
    denominator = float(np.sum(effective, dtype=np.float64))
    _require(denominator > 0.0, "semantic-quality denominator is zero")
    return float(np.sum(effective * correct, dtype=np.float64) / denominator)


def deterministic_positive_quartiles(utility):
    """Rank positive entries by (U, flat index), then split into Q1..Q4."""
    values = np.asarray(utility, dtype=np.float64).reshape(-1)
    groups = np.zeros(values.size, dtype=np.int8)
    positive = np.flatnonzero(values > 0.0)
    order = positive[np.lexsort((positive, values[positive]))]
    for quartile, indices in enumerate(np.array_split(order, 4), start=1):
        groups[indices] = quartile
    return groups.reshape(np.asarray(utility).shape)


def permutation_control(utility, correct, balance, count=1000, seed=20):
    """Shuffle each action's U only over the unlabeled sample axis."""
    utility = np.asarray(utility, dtype=np.float64)
    rng = np.random.RandomState(seed)
    true_quality = _weighted_quality(correct, balance, utility)
    shuffled = np.empty(count, dtype=np.float64)
    for permutation_id in range(count):
        permuted = np.empty_like(utility)
        for action_id in range(utility.shape[1]):
            permuted[:, action_id] = utility[
                rng.permutation(utility.shape[0]), action_id
            ]
        shuffled[permutation_id] = _weighted_quality(correct, balance, permuted)
    mean = float(shuffled.mean())
    return {
        "random_state": int(seed),
        "permutations": int(count),
        "true_Q_U": true_quality,
        "shuffle_mean": mean,
        "shuffle_std": float(shuffled.std(ddof=0)),
        "true_minus_shuffle_mean": float(true_quality - mean),
        "empirical_one_sided_p": float(
            (1 + np.count_nonzero(shuffled >= true_quality)) / (count + 1)
        ),
        "shuffle_axis": "unlabeled sample axis independently per action",
    }


def compute_diagnostic(
    *, dataset, utility, class_pred, pred_relation, balance, sample_ids,
    labeled_ids, unlabeled_ids, labels, generators, verifiers, closure=None,
    permutation_count=1000, permutation_seed=20,
):
    """Compute preregistered aggregate diagnostics from already-sealed tensors."""
    utility = np.asarray(utility, dtype=np.float64)
    class_pred = np.asarray(class_pred, dtype=np.int64)
    pred_relation = np.asarray(pred_relation, dtype=np.bool_)
    balance = np.asarray(balance, dtype=np.float64)
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    labeled_ids = np.asarray(labeled_ids, dtype=np.int64)
    unlabeled_ids = np.asarray(unlabeled_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    row = {int(sample_id): index for index, sample_id in enumerate(sample_ids)}
    unlabeled_rows = np.asarray([row[int(value)] for value in unlabeled_ids])
    labeled_rows = np.asarray([row[int(value)] for value in labeled_ids])
    if utility.shape[0] == sample_ids.size:
        utility = utility[unlabeled_rows]
    nu, action_count = utility.shape
    expected_relation_shape = (nu, labeled_ids.size, action_count)
    _require(class_pred.shape == (nu, action_count), "class_pred shape mismatch")
    _require(pred_relation.shape == expected_relation_shape,
             "PredRelation shape mismatch")
    _require(balance.shape == expected_relation_shape, "balance shape mismatch")
    _require(labels.shape == sample_ids.shape, "full GT/sample shape mismatch")
    y_u, y_l = labels[unlabeled_rows], labels[labeled_rows]
    truth_relation = y_u[:, None, None] == y_l[None, :, None]
    relation_correct = pred_relation == truth_relation
    semantic_correct = class_pred == y_u[:, None]
    denominator = balance.sum(axis=1)
    _require(np.all(denominator > 0.0), "anchor-balanced denominator is zero")
    relation_by_pair = (balance * relation_correct).sum(axis=1) / denominator
    q_uniform = _weighted_quality(relation_correct, balance)
    q_u = _weighted_quality(relation_correct, balance, utility)
    flat_u, flat_c, flat_r = utility.ravel(), semantic_correct.ravel(), relation_by_pair.ravel()
    both_classes = np.unique(flat_c).size == 2
    auc = float(roc_auc_score(flat_c, flat_u)) if both_classes else None
    ap = float(average_precision_score(flat_c, flat_u)) if both_classes else None
    rho_result = spearmanr(flat_u, flat_r)
    rho = float(rho_result.statistic) if np.isfinite(rho_result.statistic) else None
    quartiles = deterministic_positive_quartiles(utility)
    strata = []
    for group_id, name in enumerate(("U_EQ_0", "Q1", "Q2", "Q3", "Q4")):
        selected = quartiles == group_id
        strata.append({
            "group": name,
            "count": int(selected.sum()),
            "mean_U": float(utility[selected].mean()) if selected.any() else None,
            "action_semantic_accuracy": (
                float(semantic_correct[selected].mean()) if selected.any() else None
            ),
            "relation_correctness": (
                float(relation_by_pair[selected].mean()) if selected.any() else None
            ),
        })
    if closure is None:
        closure = utility > 0.0
        closure_source = "U_cycle > 0; exact because posterior support is strictly positive"
    else:
        closure = np.asarray(closure, dtype=np.bool_)
        if closure.shape[0] == sample_ids.size:
            closure = closure[unlabeled_rows]
        closure_source = "release_core.compute_directional_cycle_utility"
    _require(closure.shape == utility.shape, "closure shape mismatch")
    per_action = []
    for action_id in range(action_count):
        per_action.append({
            "action_id": action_id,
            "generator": list(generators[action_id]),
            "verifier": list(verifiers[action_id]),
            "mean_U": float(utility[:, action_id].mean()),
            "U_zero_fraction": float((utility[:, action_id] == 0.0).mean()),
            "closure_rate": float(closure[:, action_id].mean()),
            "class_pred_accuracy": float(semantic_correct[:, action_id].mean()),
            "relation_correctness": float(relation_by_pair[:, action_id].mean()),
            "utility_weighted_relation_correctness": _weighted_quality(
                relation_correct[:, :, action_id:action_id + 1],
                balance[:, :, action_id:action_id + 1],
                utility[:, action_id:action_id + 1],
            ) if np.any(utility[:, action_id] > 0.0) else None,
            "sample_count": nu,
        })
    return {
        "schema": "paper-p0-a3-relation-utility-quality-v1",
        "dataset": dataset,
        "tensor_shapes": {
            "U": list(utility.shape),
            "C": list(semantic_correct.shape),
            "G": list(expected_relation_shape),
            "E": list(relation_correct.shape),
            "R": list(relation_by_pair.shape),
            "PredRelation": list(pred_relation.shape),
            "balance": list(balance.shape),
        },
        "relation_class_accuracy": float(semantic_correct.mean()),
        "Q_uniform": q_uniform,
        "Q_U": q_u,
        "utility_semantic_lift": float(q_u - q_uniform),
        "utility_action_validity": {
            "both_classes_present": bool(both_classes),
            "roc_auc": auc,
            "average_precision": ap,
            "spearman_U_vs_anchor_balanced_R": rho,
        },
        "stratification": {
            "positive_quartile_rule": "stable rank by (U, flat index), np.array_split into four",
            "groups": strata,
            "high_vs_low_U_semantic_gap": (
                None if strata[1]["action_semantic_accuracy"] is None
                or strata[4]["action_semantic_accuracy"] is None
                else float(strata[4]["action_semantic_accuracy"]
                           - strata[1]["action_semantic_accuracy"])
            ),
        },
        "closure_source": closure_source,
        "per_action": per_action,
        "permutation_control": permutation_control(
            utility, relation_correct, balance,
            count=permutation_count, seed=permutation_seed,
        ),
        "training_run": False,
        "selection_or_thresholding": False,
        "full_gt_use": "post-seal diagnostic labels only",
    }


def _load_labels(dataset, path):
    # Deliberately the first full-GT access in each CLI path.
    _, label_sets = load_dataset(dataset, path)
    _require(isinstance(label_sets, list) and len(label_sets) == 1,
             "dataset GT contract mismatch")
    return np.ascontiguousarray(np.squeeze(label_sets[0]), dtype=np.int64)


def run_msrc(output, device="cpu"):
    verify_historical_msrc_files()
    manifest = verify_current_true_u()
    arrays, split = reconstruct_current_msrc(device)
    identity = validate_reconstruction_against_manifest(arrays, manifest)
    labels = _load_labels("MSRC-v1", msrc_p0_protocol.DATASET_PATH)
    actions = build_directional_actions(5)
    record = compute_diagnostic(
        dataset="MSRC-v1", utility=arrays["U_cycle"],
        class_pred=arrays["class_pred"],
        pred_relation=arrays["PredRelation_true"],
        balance=arrays["relation_balance_weights_true"],
        sample_ids=split.sample_ids, labeled_ids=split.labeled_ids,
        unlabeled_ids=split.unlabeled_ids, labels=labels,
        generators=actions.generators, verifiers=actions.verifiers,
        closure=arrays["closure"], permutation_count=protocol.PERMUTATION_COUNT,
        permutation_seed=protocol.PERMUTATION_SEED,
    )
    record["pre_gt_validation"] = {
        "current_true_u_manifest_and_seal": True,
        "historical_msrc_files": True,
        "reconstructed_action_hashes": identity,
        "completed_before_full_gt_load": True,
    }
    _write(output, record)
    return record


def run_caltech(output):
    input_hashes = _verify_files(protocol.CALTECH_INPUT_FILES)
    reference_hashes = _verify_files(protocol.CALTECH_REFERENCE_FILES)
    with np.load(protocol.CALTECH_ACTION, allow_pickle=False) as archive:
        names = (
            "sample_ids", "labeled_ids", "unlabeled_ids", "U_cycle", "y_gen",
            "class_pred_true", "PredRelation_true",
            "relation_balance_weights_true",
        )
        arrays = {name: np.ascontiguousarray(archive[name]) for name in names}
    labels = _load_labels("Caltech-6V", protocol.CALTECH_FULL_GT)
    actions = build_directional_actions(6)
    record = compute_diagnostic(
        dataset="Caltech-6V", utility=arrays["U_cycle"],
        class_pred=arrays["class_pred_true"],
        pred_relation=arrays["PredRelation_true"],
        balance=arrays["relation_balance_weights_true"],
        sample_ids=arrays["sample_ids"], labeled_ids=arrays["labeled_ids"],
        unlabeled_ids=arrays["unlabeled_ids"], labels=labels,
        generators=actions.generators, verifiers=actions.verifiers,
        closure=None, permutation_count=protocol.PERMUTATION_COUNT,
        permutation_seed=protocol.PERMUTATION_SEED,
    )
    record["pre_gt_validation"] = {
        "input_file_sha256": input_hashes,
        "reference_file_sha256": reference_hashes,
        "completed_before_full_gt_load": True,
    }
    _write(output, record)
    return record


def _write(path, record):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("MSRC-v1", "Caltech-6V"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_msrc(args.output, args.device) if args.dataset == "MSRC-v1" \
        else run_caltech(args.output)
    print(json.dumps({"dataset": result["dataset"], "output": args.output}, sort_keys=True))


if __name__ == "__main__":
    main()
