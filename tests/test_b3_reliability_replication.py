"""Synthetic tests for B3-A2.2 replication and specificity controls."""

import inspect

import numpy as np

from irv.b3_predictability_diagnostics import deterministic_derangement
from irv.b3_predictability_diagnostics import draw_sample_bootstrap_ids
from irv.b3_predictability_diagnostics import make_fold_assignment
from irv.b3_predictability_diagnostics import oof_ridge_predictability
from irv.b3_predictability_diagnostics import (
    oof_ridge_predictability_with_shuffled_correspondence,
)
from irv.b3_reliability_replication import aggregate_multiseed
from irv.b3_reliability_replication import aggregate_severity
from irv.b3_reliability_replication import corruption_label_permutation_test
from irv.b3_reliability_replication import oof_target_only_centroid_score
from irv.b3_reliability_replication import per_view_reliability
from irv.b3_reliability_replication import permute_corruption_mask_rows


def _balanced_mask(sample_num=210, view_num=5):
    mask = np.zeros((sample_num, view_num), dtype=bool)
    for sample_id in range(sample_num):
        mask[sample_id, sample_id % view_num] = True
        mask[sample_id, (sample_id + 1) % view_num] = True
    return mask


def _linear_views(sample_num=100, view_num=5, feature_dim=4, seed=11):
    random_state = np.random.RandomState(seed)
    shared = random_state.normal(size=(sample_num, feature_dim))
    views = []
    for _ in range(view_num):
        matrix, _ = np.linalg.qr(
            random_state.normal(size=(feature_dim, feature_dim))
        )
        views.append(shared.dot(matrix))
    return views


def _condition_record(seed, snr, auc=0.8, gap=0.2):
    return {
        "model_seed": seed,
        "snr_db": snr,
        "auc": auc,
        "auc_ci_low": auc - 0.05,
        "auc_ci_high": auc + 0.05,
        "gap": gap,
        "gap_ci_low": gap - 0.05,
        "gap_ci_high": gap + 0.05,
    }


def test_rowwise_mask_permutation_preserves_shape_rows_and_columns():
    mask = _balanced_mask()
    permutation = np.random.RandomState(4).permutation(mask.shape[0])

    permuted = permute_corruption_mask_rows(mask, permutation)

    assert permuted.shape == mask.shape
    assert np.array_equal(permuted.sum(axis=1), mask.sum(axis=1))
    assert np.array_equal(permuted.sum(axis=0), mask.sum(axis=0))


def test_rowwise_mask_permutation_is_deterministic():
    mask = _balanced_mask()
    first = np.random.RandomState(8).permutation(mask.shape[0])
    repeated = np.random.RandomState(8).permutation(mask.shape[0])

    assert np.array_equal(
        permute_corruption_mask_rows(mask, first),
        permute_corruption_mask_rows(mask, repeated),
    )


def test_strong_real_signal_is_significant_against_permutation_null():
    mask = _balanced_mask()
    scores = np.where(mask, -1.0, 1.0)

    result = corruption_label_permutation_test(
        scores, mask, repeats=300, seed=20260815
    )

    assert result["real_auc"] > result["permutation_auc_p95"]
    assert result["p_auc"] < 0.01
    assert result["real_gap"] > result["permutation_gap_p95"]
    assert result["p_gap"] < 0.01
    assert result["specificity_pass"] is True


def test_random_signal_is_not_stably_significant_under_permutation():
    mask = _balanced_mask()
    scores = np.random.RandomState(19).normal(size=mask.shape)

    result = corruption_label_permutation_test(
        scores, mask, repeats=300, seed=20260815
    )

    assert result["specificity_pass"] is False


def test_shuffled_correspondence_keeps_oof_train_test_disjoint():
    views = _linear_views()
    folds = make_fold_assignment(100, n_splits=5, random_state=20)

    shuffled = oof_ridge_predictability_with_shuffled_correspondence(
        views, folds, alpha=1.0, shuffle_seed=20260815
    )

    assert shuffled["audit"]["train_test_disjoint_pass"] is True
    assert shuffled["audit"]["oof_coverage_pass"] is True
    assert shuffled["audit"]["target_training_permutation_pass"] is True


def test_shuffled_training_derangement_has_zero_fixed_points():
    permutation = deterministic_derangement(168, seed=99)
    views = _linear_views()
    folds = make_fold_assignment(100, n_splits=5, random_state=20)
    shuffled = oof_ridge_predictability_with_shuffled_correspondence(
        views, folds, alpha=1.0, shuffle_seed=99
    )

    assert np.sum(permutation == np.arange(168)) == 0
    assert shuffled["audit"]["target_training_fixed_point_count"] == 0


def test_real_linear_correspondence_exceeds_shuffled_correspondence():
    views = _linear_views()
    folds = make_fold_assignment(100, n_splits=5, random_state=20)
    real = oof_ridge_predictability(views, folds, alpha=1.0)
    shuffled = oof_ridge_predictability_with_shuffled_correspondence(
        views, folds, alpha=1.0, shuffle_seed=20260815
    )

    assert np.mean(real["oof_consensus_cosine"]) > (
        np.mean(shuffled["oof_consensus_cosine"]) + 0.5
    )


def test_per_view_output_contains_five_records():
    mask = _balanced_mask()
    scores = np.where(mask, 0.0, 1.0)

    result = per_view_reliability(scores, mask, repeats=100, seed=7)

    assert len(result["records"]) == 5
    assert [record["target_view"] for record in result["records"]] == list(range(5))


def test_per_view_clean_and_corrupted_counts_are_exact():
    mask = _balanced_mask()
    scores = np.where(mask, 0.0, 1.0)

    result = per_view_reliability(scores, mask, repeats=100, seed=7)

    assert all(record["clean_count"] == 126 for record in result["records"])
    assert all(record["corrupted_count"] == 84 for record in result["records"])


def test_sample_bootstrap_keeps_all_five_views_together():
    draws = draw_sample_bootstrap_ids(210, repeats=11, seed=20260815)
    row_ids = np.repeat(np.arange(210)[:, None], 5, axis=1)

    selected = row_ids[draws]

    assert selected.shape == (11, 210, 5)
    assert np.all(selected == draws[:, :, None])


def test_three_seed_aggregation_is_deterministic_and_sorted():
    records = [
        _condition_record(50, 2.5, auc=0.9),
        _condition_record(20, 2.5, auc=0.8),
        _condition_record(30, 2.5, auc=0.85),
    ]

    first = aggregate_multiseed(records)
    repeated = aggregate_multiseed(list(reversed(records)))

    assert first == repeated
    assert first["seeds"] == [20, 30, 50]
    assert first["multiseed_replication_pass"] is True


def test_severity_aggregation_does_not_require_monotonic_auc():
    records = [
        _condition_record(20, 5.0, auc=0.95),
        _condition_record(20, 2.5, auc=0.80),
        _condition_record(20, 0.0, auc=0.90),
    ]

    result = aggregate_severity(records)

    assert result["monotonicity_required"] is False
    assert result["severity_robustness_pass"] is True


def test_target_only_api_cannot_access_other_views():
    parameters = inspect.signature(oof_target_only_centroid_score).parameters
    target = np.random.RandomState(3).normal(size=(30, 4))
    folds = make_fold_assignment(30, n_splits=5, random_state=20)

    scores = oof_target_only_centroid_score(target, folds)

    assert set(parameters) == {"target_view", "fold_assignment"}
    assert scores.shape == (30,)
    assert np.isfinite(scores).all()
