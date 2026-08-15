"""Offline replication and specificity controls for B3-A2.2."""

import math

import numpy as np

from irv.b3_predictability_diagnostics import bootstrap_reliability
from irv.b3_predictability_diagnostics import null_distribution_summary
from irv.b3_predictability_diagnostics import reliability_metrics


def permute_corruption_mask_rows(corruption_mask, permutation):
    """Permute complete sample rows while retaining view dependencies."""
    mask = np.asarray(corruption_mask, dtype=bool)
    permutation = np.asarray(permutation, dtype=np.int64)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("corruption_mask must have shape [N,V]")
    if permutation.shape != (mask.shape[0],):
        raise ValueError("permutation must have shape [N]")
    if not np.array_equal(np.sort(permutation), np.arange(mask.shape[0])):
        raise ValueError("permutation must contain every sample ID exactly once")
    permuted = mask[permutation]
    if permuted.shape != mask.shape:
        raise RuntimeError("row permutation changed mask shape")
    if not np.array_equal(permuted.sum(axis=0), mask.sum(axis=0)):
        raise RuntimeError("row permutation changed global per-view counts")
    if not np.array_equal(
        np.sort(permuted.sum(axis=1)), np.sort(mask.sum(axis=1))
    ):
        raise RuntimeError("row permutation changed the row-count multiset")
    return permuted


def corruption_label_permutation_test(
    scores,
    corruption_mask,
    repeats=2000,
    seed=20260815,
):
    """Evaluate fixed scores against a row-permuted corruption-mask null."""
    values = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(corruption_mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("scores and corruption_mask must have equal [N,V] shape")
    repeats = int(repeats)
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    real = reliability_metrics(values, mask)
    random_state = np.random.RandomState(int(seed))
    auc_null = np.empty(repeats, dtype=np.float64)
    gap_null = np.empty(repeats, dtype=np.float64)
    for repeat_id in range(repeats):
        permuted_mask = permute_corruption_mask_rows(
            mask,
            random_state.permutation(mask.shape[0]),
        )
        metrics = reliability_metrics(values, permuted_mask)
        auc_null[repeat_id] = metrics["clean_corrupted_auc"]
        gap_null[repeat_id] = metrics["predictability_gap"]
    real_auc = real["clean_corrupted_auc"]
    real_gap = real["predictability_gap"]
    result = {
        "permutation_repeats": repeats,
        "permutation_seed": int(seed),
        "permutation_unit": "sample_row_with_all_views",
        "real_auc": real_auc,
        "permutation_auc_mean": float(np.mean(auc_null)),
        "permutation_auc_p95": float(np.percentile(auc_null, 95)),
        "p_auc": float((1 + np.sum(auc_null >= real_auc)) / (1 + repeats)),
        "real_gap": real_gap,
        "permutation_gap_mean": float(np.mean(gap_null)),
        "permutation_gap_p95": float(np.percentile(gap_null, 95)),
        "p_gap": float((1 + np.sum(gap_null >= real_gap)) / (1 + repeats)),
        "auc_null_summary": null_distribution_summary(auc_null),
        "gap_null_summary": null_distribution_summary(gap_null),
        "row_sum_multiset_preserved_pass": True,
        "column_global_counts_preserved_pass": True,
    }
    result["specificity_pass"] = bool(
        result["real_auc"] > result["permutation_auc_p95"]
        and result["p_auc"] < 0.01
        and result["real_gap"] > result["permutation_gap_p95"]
        and result["p_gap"] < 0.01
    )
    return result


def oof_target_only_centroid_score(target_view, fold_assignment):
    """Score one target view using only its OOF training-fold centroid."""
    values = np.asarray(target_view, dtype=np.float64)
    assignment = np.asarray(fold_assignment, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("target_view must have shape [N,D]")
    if assignment.shape != (values.shape[0],):
        raise ValueError("fold_assignment must have shape [N]")
    if not np.isfinite(values).all():
        raise ValueError("target_view must contain only finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    scores = np.empty(values.shape[0], dtype=np.float64)
    coverage = np.zeros(values.shape[0], dtype=np.int64)
    fold_ids = np.unique(assignment)
    if fold_ids.size < 2:
        raise ValueError("at least two folds are required")
    for fold_id in fold_ids:
        train_ids = np.flatnonzero(assignment != fold_id)
        test_ids = np.flatnonzero(assignment == fold_id)
        if train_ids.size == 0 or test_ids.size == 0:
            raise ValueError("each fold must have non-empty train and test IDs")
        centroid = normalized[train_ids].mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
        scores[test_ids] = normalized[test_ids].dot(centroid)
        coverage[test_ids] += 1
    if not np.all(coverage == 1) or not np.isfinite(scores).all():
        raise RuntimeError("target-only OOF score integrity failed")
    return scores


def per_view_reliability(
    scores,
    corruption_mask,
    repeats=2000,
    seed=20260815,
):
    """Bootstrap reliability separately for every target view."""
    values = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(corruption_mask, dtype=bool)
    if values.ndim != 2 or values.shape != mask.shape:
        raise ValueError("scores and corruption_mask must have equal [N,V] shape")
    records = []
    for target_view in range(values.shape[1]):
        bootstrap = bootstrap_reliability(
            {"view": values[:, target_view:target_view + 1]},
            mask[:, target_view:target_view + 1],
            repeats=repeats,
            seed=seed,
        )["arms"]["view"]
        records.append({
            "target_view": target_view,
            "clean_count": bootstrap["clean_pair_count"],
            "corrupted_count": bootstrap["corrupted_pair_count"],
            "clean_mean": bootstrap["clean_predictability_mean"],
            "corrupted_mean": bootstrap["corrupted_predictability_mean"],
            "gap": bootstrap["predictability_gap"],
            "gap_ci_low": bootstrap["gap_ci_low"],
            "gap_ci_high": bootstrap["gap_ci_high"],
            "auc": bootstrap["clean_corrupted_auc"],
            "auc_ci_low": bootstrap["auc_ci_low"],
            "auc_ci_high": bootstrap["auc_ci_high"],
        })
    auc_values = np.asarray([record["auc"] for record in records])
    auc_pass_count = sum(record["auc_ci_low"] > 0.5 for record in records)
    gap_pass_count = sum(record["gap_ci_low"] > 0.0 for record in records)
    return {
        "records": records,
        "min_view_auc": float(np.min(auc_values)),
        "median_view_auc": float(np.median(auc_values)),
        "max_view_auc": float(np.max(auc_values)),
        "number_of_views_auc_ci_lower_gt_0p5": int(auc_pass_count),
        "number_of_views_gap_ci_lower_gt_0": int(gap_pass_count),
        "view_robustness_pass": bool(auc_pass_count >= 4 and gap_pass_count >= 4),
    }


def _metric_aggregate(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("aggregate values must be a finite non-empty vector")
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def aggregate_multiseed(records, expected_seeds=(20, 30, 50)):
    """Aggregate fixed-seed results without replacing per-seed gates by a mean."""
    ordered = sorted(records, key=lambda record: int(record["model_seed"]))
    seeds = [int(record["model_seed"]) for record in ordered]
    if seeds != sorted(int(seed) for seed in expected_seeds):
        raise ValueError("multiseed records do not match the expected seeds")
    per_seed_pass = {
        str(record["model_seed"]): bool(
            record["auc_ci_low"] > 0.5 and record["gap_ci_low"] > 0.0
        )
        for record in ordered
    }
    return {
        "seeds": seeds,
        "per_seed": ordered,
        "auc_across_seeds": _metric_aggregate(
            [record["auc"] for record in ordered]
        ),
        "gap_across_seeds": _metric_aggregate(
            [record["gap"] for record in ordered]
        ),
        "per_seed_absolute_pass": per_seed_pass,
        "multiseed_replication_pass": all(per_seed_pass.values()),
    }


def aggregate_severity(records, expected_snr=(0.0, 2.5, 5.0)):
    """Aggregate severity results without imposing monotonicity."""
    ordered = sorted(records, key=lambda record: float(record["snr_db"]))
    severities = [float(record["snr_db"]) for record in ordered]
    if severities != sorted(float(value) for value in expected_snr):
        raise ValueError("severity records do not match the expected SNR values")
    individual_pass = {
        str(float(record["snr_db"])): bool(
            record["auc_ci_low"] > 0.5 and record["gap_ci_low"] > 0.0
        )
        for record in ordered
    }
    return {
        "snr_db": severities,
        "per_severity": ordered,
        "individual_absolute_pass": individual_pass,
        "monotonicity_required": False,
        "snr5_mild_condition_report_only": True,
        "severity_robustness_pass": bool(
            individual_pass["0.0"] and individual_pass["2.5"]
        ),
    }


def paired_score_control(
    real_scores,
    control_scores,
    corruption_mask,
    repeats=2000,
    seed=20260815,
    real_name="real",
    control_name="control",
):
    """Bootstrap a paired real-minus-control reliability difference."""
    result = bootstrap_reliability(
        {real_name: real_scores, control_name: control_scores},
        corruption_mask,
        repeats=repeats,
        seed=seed,
        paired_comparisons=((real_name, control_name),),
    )
    return {
        "real": result["arms"][real_name],
        "control": result["arms"][control_name],
        "real_minus_control": result["paired"][
            real_name + "_minus_" + control_name
        ],
        "bootstrap_repeats": int(repeats),
        "bootstrap_seed": int(seed),
        "sampling_unit": "sample_id_with_all_views",
    }


def all_finite_nested(value):
    """Return whether every numeric leaf is finite."""
    if isinstance(value, dict):
        return all(all_finite_nested(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite_nested(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True
