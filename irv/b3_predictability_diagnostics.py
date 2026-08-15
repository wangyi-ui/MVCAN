"""Pure offline diagnostics for B3-A2.1 cross-view predictability."""

import hashlib
import math

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


NULL_PERCENTILES = (5, 25, 50, 75, 95)


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _row_normalize(array, eps=1e-12):
    array = np.asarray(array, dtype=np.float64)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, float(eps))


def normalize_representation_views(representation_views):
    """Return row-normalized representations with shape [N,V,D]."""
    if not isinstance(representation_views, (list, tuple)):
        raise ValueError("representation_views must be a list or tuple")
    if len(representation_views) < 2:
        raise ValueError("at least two representation views are required")
    arrays = [_as_numpy(view) for view in representation_views]
    expected_shape = arrays[0].shape
    if len(expected_shape) != 2 or min(expected_shape) <= 0:
        raise ValueError("each representation view must have shape [N,D]")
    if any(array.shape != expected_shape for array in arrays):
        raise ValueError("all representation views must have equal shape")
    stacked = np.stack(arrays, axis=1).astype(np.float64, copy=False)
    if not np.isfinite(stacked).all():
        raise ValueError("representation views must contain only finite values")
    return _row_normalize(stacked)


def make_fold_assignment(sample_num, n_splits=5, random_state=20):
    """Assign each sample ID to exactly one deterministic OOF fold."""
    sample_num = int(sample_num)
    n_splits = int(n_splits)
    if sample_num < n_splits or n_splits < 2:
        raise ValueError("sample_num must be at least n_splits >= 2")
    assignment = np.full(sample_num, -1, dtype=np.int64)
    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(random_state),
    )
    sample_ids = np.arange(sample_num, dtype=np.int64)
    for fold_id, (train_ids, test_ids) in enumerate(splitter.split(sample_ids)):
        if np.intersect1d(train_ids, test_ids).size:
            raise RuntimeError("OOF train/test sample IDs overlap")
        if np.any(assignment[test_ids] != -1):
            raise RuntimeError("a sample was assigned to more than one fold")
        assignment[test_ids] = fold_id
    if np.any(assignment < 0):
        raise RuntimeError("OOF assignment does not cover every sample")
    return assignment


def fold_assignment_csv_text(fold_assignment):
    """Return the canonical on-disk CSV serialization for fold assignments."""
    assignment = np.asarray(fold_assignment, dtype=np.int64)
    if assignment.ndim != 1 or assignment.size == 0:
        raise ValueError("fold_assignment must have shape [N]")
    lines = ["sample_id,fold_id"]
    lines.extend(
        str(sample_id) + "," + str(int(fold_id))
        for sample_id, fold_id in enumerate(assignment)
    )
    return "\n".join(lines) + "\n"


def fold_assignment_sha256(fold_assignment):
    """Hash the exact canonical CSV bytes for a fold assignment."""
    serialized = fold_assignment_csv_text(fold_assignment).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def ordered_view_pairs(view_num):
    """Return all ordered (source_view, target_view) pairs with source != target."""
    view_num = int(view_num)
    if view_num < 2:
        raise ValueError("view_num must be at least two")
    return [
        (source_view, target_view)
        for target_view in range(view_num)
        for source_view in range(view_num)
        if source_view != target_view
    ]


def oof_ridge_predictability(
    representation_views,
    fold_assignment,
    alpha=1.0,
):
    """Fit label-free ordered cross-view Ridge models and score OOF predictions.

    This API intentionally accepts representations, fold IDs, and Ridge alpha
    only. Class labels and corruption masks cannot enter predictor fitting.
    """
    representations = normalize_representation_views(representation_views)
    sample_num, view_num, feature_dim = representations.shape
    assignment = np.asarray(fold_assignment, dtype=np.int64)
    if assignment.shape != (sample_num,):
        raise ValueError("fold_assignment must have shape [N]")
    fold_ids = np.unique(assignment)
    if (
        fold_ids.size < 2
        or fold_ids[0] != 0
        or not np.array_equal(fold_ids, np.arange(fold_ids.size))
    ):
        raise ValueError("fold IDs must be contiguous from zero")
    if float(alpha) < 0:
        raise ValueError("Ridge alpha must be non-negative")

    source_view_ids = np.empty((view_num, view_num - 1), dtype=np.int64)
    predictions = np.empty(
        (sample_num, view_num, view_num - 1, feature_dim),
        dtype=np.float64,
    )
    oof_coverage = np.zeros(sample_num, dtype=np.int64)
    train_test_disjoint_pass = True

    for fold_id in fold_ids:
        test_ids = np.flatnonzero(assignment == fold_id)
        train_ids = np.flatnonzero(assignment != fold_id)
        train_test_disjoint_pass = bool(
            train_test_disjoint_pass
            and test_ids.size > 0
            and train_ids.size > 0
            and np.intersect1d(train_ids, test_ids).size == 0
        )
        oof_coverage[test_ids] += 1
        for target_view in range(view_num):
            sources = [
                source_view
                for source_view in range(view_num)
                if source_view != target_view
            ]
            source_view_ids[target_view] = sources
            for source_position, source_view in enumerate(sources):
                predictor = Ridge(alpha=float(alpha), fit_intercept=True)
                predictor.fit(
                    representations[train_ids, source_view],
                    representations[train_ids, target_view],
                )
                predicted = predictor.predict(
                    representations[test_ids, source_view]
                )
                predictions[
                    test_ids, target_view, source_position
                ] = _row_normalize(predicted)

    pairwise_cosine = np.sum(
        predictions * representations[:, :, None, :], axis=3
    )
    consensus_prediction = _row_normalize(predictions.mean(axis=2))
    oof_consensus_cosine = np.sum(
        representations * consensus_prediction, axis=2
    )
    consensus_negative_mse = -np.mean(
        np.square(representations - consensus_prediction), axis=2
    )
    all_finite_pass = bool(
        np.isfinite(predictions).all()
        and np.isfinite(pairwise_cosine).all()
        and np.isfinite(consensus_prediction).all()
        and np.isfinite(oof_consensus_cosine).all()
        and np.isfinite(consensus_negative_mse).all()
    )
    if not all_finite_pass:
        raise RuntimeError("OOF predictability produced non-finite values")

    return {
        "predicted_target_views": predictions,
        "consensus_prediction": consensus_prediction,
        "oof_consensus_cosine": oof_consensus_cosine,
        "pairwise_cosine_mean": pairwise_cosine.mean(axis=2),
        "pairwise_cosine_median": np.median(pairwise_cosine, axis=2),
        "consensus_negative_mse": consensus_negative_mse,
        "source_view_ids": source_view_ids,
        "audit": {
            "ordered_view_pairs": [
                {"source_view": source, "target_view": target}
                for source, target in ordered_view_pairs(view_num)
            ],
            "ordered_view_pair_count": view_num * (view_num - 1),
            "fold_count": int(fold_ids.size),
            "train_test_disjoint_pass": train_test_disjoint_pass,
            "oof_coverage_pass": bool(np.all(oof_coverage == 1)),
            "oof_coverage_count": int(np.sum(oof_coverage == 1)),
            "all_finite_pass": all_finite_pass,
        },
    }


def score_summary(scores):
    """Return a compact finite summary for an [N,V] score matrix."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite [N,V] matrix")
    flat = values.reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "p05": float(np.percentile(flat, 5)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
    }


def reliability_metrics(scores, corruption_mask):
    """Evaluate a score as evidence for clean (1) vs corrupted (0)."""
    values = np.asarray(scores, dtype=np.float64)
    corrupted = np.asarray(corruption_mask, dtype=bool)
    if values.ndim != 2 or values.shape != corrupted.shape:
        raise ValueError("scores and corruption_mask must have equal [N,V] shape")
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    clean = ~corrupted
    if not np.any(clean) or not np.any(corrupted):
        raise ValueError("both clean and corrupted sample-views are required")
    flat_scores = values.reshape(-1)
    clean_indicator = clean.astype(np.int64).reshape(-1)
    clean_scores = values[clean]
    corrupted_scores = values[corrupted]
    spearman = float(spearmanr(clean_indicator, flat_scores).correlation)
    result = {
        "clean_pair_count": int(clean.sum()),
        "corrupted_pair_count": int(corrupted.sum()),
        "clean_predictability_mean": float(np.mean(clean_scores)),
        "corrupted_predictability_mean": float(np.mean(corrupted_scores)),
        "predictability_gap": float(
            np.mean(clean_scores) - np.mean(corrupted_scores)
        ),
        "clean_corrupted_auc": float(
            roc_auc_score(clean_indicator, flat_scores)
        ),
        "spearman_clean_indicator_vs_predictability": spearman,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("reliability metrics contain non-finite values")
    return result


def draw_sample_bootstrap_ids(sample_num, repeats=2000, seed=20260815):
    """Draw complete sample rows; all views remain grouped by construction."""
    sample_num = int(sample_num)
    repeats = int(repeats)
    if sample_num <= 0 or repeats <= 0:
        raise ValueError("sample_num and repeats must be positive")
    random_state = np.random.RandomState(int(seed))
    return random_state.randint(0, sample_num, size=(repeats, sample_num))


def _percentile_ci(values):
    low, high = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return float(low), float(high)


def bootstrap_reliability(
    scores_by_arm,
    corruption_mask,
    repeats=2000,
    seed=20260815,
    paired_comparisons=(),
):
    """Run deterministic sample-level bootstrap for arms and paired deltas."""
    if not isinstance(scores_by_arm, dict) or not scores_by_arm:
        raise ValueError("scores_by_arm must be a non-empty dict")
    mask = np.asarray(corruption_mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("corruption_mask must have shape [N,V]")
    score_arrays = {}
    for arm_name, scores in scores_by_arm.items():
        array = np.asarray(scores, dtype=np.float64)
        if array.shape != mask.shape or not np.isfinite(array).all():
            raise ValueError("every arm score must be finite and match the mask")
        score_arrays[str(arm_name)] = array

    draws = draw_sample_bootstrap_ids(mask.shape[0], repeats=repeats, seed=seed)
    point = {
        arm_name: reliability_metrics(scores, mask)
        for arm_name, scores in score_arrays.items()
    }
    distributions = {
        arm_name: {"gap": [], "auc": [], "spearman": []}
        for arm_name in score_arrays
    }
    for sample_ids in draws:
        sampled_mask = mask[sample_ids]
        for arm_name, scores in score_arrays.items():
            metrics = reliability_metrics(scores[sample_ids], sampled_mask)
            distributions[arm_name]["gap"].append(
                metrics["predictability_gap"]
            )
            distributions[arm_name]["auc"].append(
                metrics["clean_corrupted_auc"]
            )
            distributions[arm_name]["spearman"].append(
                metrics["spearman_clean_indicator_vs_predictability"]
            )

    arms = {}
    for arm_name, metrics in point.items():
        gap_low, gap_high = _percentile_ci(distributions[arm_name]["gap"])
        auc_low, auc_high = _percentile_ci(distributions[arm_name]["auc"])
        spearman_low, spearman_high = _percentile_ci(
            distributions[arm_name]["spearman"]
        )
        arms[arm_name] = {
            **metrics,
            "gap_ci_low": gap_low,
            "gap_ci_high": gap_high,
            "auc_ci_low": auc_low,
            "auc_ci_high": auc_high,
            "spearman_ci_low": spearman_low,
            "spearman_ci_high": spearman_high,
        }

    paired = {}
    for first_arm, second_arm in paired_comparisons:
        if first_arm not in score_arrays or second_arm not in score_arrays:
            raise ValueError("paired comparison references an unknown arm")
        key = str(first_arm) + "_minus_" + str(second_arm)
        delta_gap = np.asarray(distributions[first_arm]["gap"]) - np.asarray(
            distributions[second_arm]["gap"]
        )
        delta_auc = np.asarray(distributions[first_arm]["auc"]) - np.asarray(
            distributions[second_arm]["auc"]
        )
        gap_low, gap_high = _percentile_ci(delta_gap)
        auc_low, auc_high = _percentile_ci(delta_auc)
        paired[key] = {
            "delta_gap": (
                point[first_arm]["predictability_gap"]
                - point[second_arm]["predictability_gap"]
            ),
            "delta_gap_ci_low": gap_low,
            "delta_gap_ci_high": gap_high,
            "delta_auc": (
                point[first_arm]["clean_corrupted_auc"]
                - point[second_arm]["clean_corrupted_auc"]
            ),
            "delta_auc_ci_low": auc_low,
            "delta_auc_ci_high": auc_high,
        }

    return {
        "bootstrap_repeats": int(repeats),
        "bootstrap_seed": int(seed),
        "sampling_unit": "sample_id_with_all_views",
        "arms": arms,
        "paired": paired,
    }


def null_distribution_summary(values):
    """Summarize one finite one-dimensional random-head null metric."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("null values must be a finite non-empty vector")
    percentiles = np.percentile(array, NULL_PERCENTILES)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p05": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
    }


def value_position_in_null(value, null_values):
    """Return empirical weak percentile rank and population-standardized z."""
    array = np.asarray(null_values, dtype=np.float64)
    value = float(value)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("null values must be a finite non-empty vector")
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    standard_deviation = float(np.std(array))
    return {
        "value": value,
        "percentile": float(100.0 * np.mean(array <= value)),
        "z_score": (
            float((value - np.mean(array)) / standard_deviation)
            if standard_deviation > 1e-12
            else None
        ),
        "percentile_definition": "100 * mean(null_value <= observed_value)",
    }
