"""Synthetic tests for dataset-agnostic D2-A0 Utility transfer logic."""

import inspect

import numpy as np
import pytest

from experiments.d2_caltech6v.evaluate_d2_a0_utility_transfer import (
    admission_diagnostics,
)
from experiments.d2_caltech6v.evaluate_d2_a0_utility_transfer import (
    compute_transfer_scores,
)
from experiments.d2_caltech6v.evaluate_d2_a0_utility_transfer import (
    random_admission_baselines,
)
from experiments.d2_caltech6v.evaluate_d2_a0_utility_transfer import topk_admission
from irv.b3_predictability_diagnostics import make_fold_assignment
from irv.b3_predictability_diagnostics import ordered_view_pairs


def _linear_views(sample_num=60, view_num=6, feature_dim=4, seed=17):
    random_state = np.random.RandomState(seed)
    shared = random_state.normal(size=(sample_num, feature_dim))
    views = []
    for _ in range(view_num):
        matrix, _ = np.linalg.qr(
            random_state.normal(size=(feature_dim, feature_dim))
        )
        views.append(shared.dot(matrix))
    return views


def _balanced_three_of_six_mask(sample_num):
    mask = np.zeros((sample_num, 6), dtype=bool)
    for sample_id in range(sample_num):
        mask[sample_id, (sample_id + np.arange(3)) % 6] = True
    return mask


def test_six_views_generate_thirty_dynamic_ordered_predictors():
    pairs = ordered_view_pairs(6)

    assert len(pairs) == 6 * (6 - 1) == 30
    assert len(set(pairs)) == 30
    assert all(source != target for source, target in pairs)


def test_six_view_transfer_scores_have_dynamic_n_by_v_shapes():
    views = _linear_views()
    folds = make_fold_assignment(60, n_splits=5, random_state=20)

    result = compute_transfer_scores(views, folds, ridge_alpha=1.0)

    assert result["T"].shape == (60, 6)
    assert result["U"].shape == (60, 6)
    assert result["prediction"]["audit"]["ordered_view_pair_count"] == 30
    assert result["prediction"]["audit"]["oof_coverage_pass"] is True


def test_msrc_sized_five_view_path_remains_backward_compatible():
    views = _linear_views(sample_num=210, view_num=5, seed=23)
    folds = make_fold_assignment(210, n_splits=5, random_state=20)

    result = compute_transfer_scores(views, folds, ridge_alpha=1.0)

    assert result["T"].shape == (210, 5)
    assert result["U"].shape == (210, 5)
    assert result["prediction"]["audit"]["ordered_view_pair_count"] == 20
    assert result["prediction"]["audit"]["oof_coverage_pass"] is True


def test_top3_indices_are_exact_with_stable_tie_breaking():
    utility = np.array([
        [0.2, 0.9, 0.1, 0.8, 0.7, 0.0],
        [0.5, 0.5, 0.1, 0.5, 0.2, 0.0],
    ])

    selected, mask = topk_admission(utility, admitted_view_num=3)

    assert np.array_equal(selected[0], [1, 3, 4])
    assert np.array_equal(selected[1], [0, 1, 3])
    assert np.all(mask.sum(axis=1) == 3)


def test_oracle_clean_top3_is_perfect_when_clean_utilities_are_higher():
    sample_num = 8
    corruption_mask = _balanced_three_of_six_mask(sample_num)
    utility = np.where(corruption_mask, 0.1, 0.9).astype(np.float64)
    # Add row-varying offsets without changing clean/corrupt ordering, ensuring
    # admission margins include both correct and (below) synthetic error rows.
    utility += np.arange(sample_num, dtype=np.float64)[:, None] * 1e-3

    selected, admission_mask = topk_admission(utility, admitted_view_num=3)

    assert selected.shape == (sample_num, 3)
    assert np.array_equal(admission_mask, np.logical_not(corruption_mask))
    assert float(np.mean(np.all(admission_mask == ~corruption_mask, axis=1))) == 1.0


def test_admission_metrics_report_perfect_clean_selection_with_mixed_errors():
    corruption_mask = _balanced_three_of_six_mask(4)
    utility = np.where(corruption_mask, 0.0, 1.0).astype(np.float64)
    utility[3] = np.where(corruption_mask[3], 1.0, 0.0)

    metrics = admission_diagnostics(utility, corruption_mask, admitted_view_num=3)

    assert metrics["topk_clean_fraction"] == pytest.approx(0.75)
    assert metrics["topk_all_clean_rate"] == pytest.approx(0.75)
    assert metrics["corrupted_views_admitted_mean"] == pytest.approx(0.75)


def test_random_baselines_are_derived_from_view_and_clean_counts():
    six_view = random_admission_baselines(6, 3, 3)
    eight_view = random_admission_baselines(8, 5, 3)

    assert six_view["random_topk_clean_fraction"] == pytest.approx(0.5)
    assert six_view["random_topk_all_clean_rate"] == pytest.approx(1.0 / 20.0)
    assert eight_view["random_topk_clean_fraction"] == pytest.approx(5.0 / 8.0)
    assert eight_view["random_topk_all_clean_rate"] == pytest.approx(10.0 / 56.0)


def test_label_permutation_cannot_change_t_or_u():
    views = _linear_views(sample_num=40)
    folds = make_fold_assignment(40, n_splits=5, random_state=20)
    labels = np.arange(40, dtype=np.int64) % 7
    first = compute_transfer_scores(views, folds, ridge_alpha=1.0)
    labels = labels[np.random.RandomState(3).permutation(labels.size)]
    second = compute_transfer_scores(views, folds, ridge_alpha=1.0)

    assert labels.shape == (40,)
    assert np.array_equal(first["T"], second["T"])
    assert np.array_equal(first["U"], second["U"])


def test_transfer_score_api_cannot_accept_labels_or_corruption_masks():
    parameters = inspect.signature(compute_transfer_scores).parameters

    assert set(parameters) == {
        "representation_views",
        "fold_assignment",
        "ridge_alpha",
    }
    assert all(
        "label" not in name and "mask" not in name and "oracle" not in name
        for name in parameters
    )
