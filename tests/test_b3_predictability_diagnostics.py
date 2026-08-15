"""Synthetic tests for the pure B3-A2.1 predictability diagnostics."""

import inspect

import numpy as np
import pytest

from irv.b3_audit import hash_semantic_heads
from irv.b3_predictability_diagnostics import bootstrap_reliability
from irv.b3_predictability_diagnostics import draw_sample_bootstrap_ids
from irv.b3_predictability_diagnostics import fold_assignment_sha256
from irv.b3_predictability_diagnostics import make_fold_assignment
from irv.b3_predictability_diagnostics import oof_ridge_predictability
from irv.b3_predictability_diagnostics import ordered_view_pairs
from irv.semantic_head import DetachedSemanticHeadBank


def _linear_views(sample_num=80, view_num=5, feature_dim=4, seed=7):
    random_state = np.random.RandomState(seed)
    shared = random_state.normal(size=(sample_num, feature_dim))
    views = []
    for _ in range(view_num):
        matrix, _ = np.linalg.qr(
            random_state.normal(size=(feature_dim, feature_dim))
        )
        views.append(shared.dot(matrix))
    return views


def _predictability_result():
    views = _linear_views()
    folds = make_fold_assignment(len(views[0]), n_splits=5, random_state=20)
    return oof_ridge_predictability(views, folds, alpha=1.0)


def test_kfold_assigns_every_sample_exactly_once_oof():
    assignment = make_fold_assignment(210, n_splits=5, random_state=20)

    assert assignment.shape == (210,)
    assert np.array_equal(np.unique(assignment), np.arange(5))
    assert sum(np.sum(assignment == fold_id) for fold_id in range(5)) == 210


def test_fold_train_and_test_sample_ids_are_disjoint():
    assignment = make_fold_assignment(210, n_splits=5, random_state=20)

    for fold_id in range(5):
        train_ids = np.flatnonzero(assignment != fold_id)
        test_ids = np.flatnonzero(assignment == fold_id)
        assert test_ids.size > 0
        assert np.intersect1d(train_ids, test_ids).size == 0


def test_same_fold_seed_has_exact_same_hash():
    first = make_fold_assignment(210, n_splits=5, random_state=20)
    repeated = make_fold_assignment(210, n_splits=5, random_state=20)
    different = make_fold_assignment(210, n_splits=5, random_state=21)

    assert fold_assignment_sha256(first) == fold_assignment_sha256(repeated)
    assert fold_assignment_sha256(first) != fold_assignment_sha256(different)


def test_linear_cross_view_mapping_has_high_oof_predictability():
    result = _predictability_result()

    assert np.mean(result["oof_consensus_cosine"]) > 0.98
    assert result["audit"]["train_test_disjoint_pass"] is True
    assert result["audit"]["oof_coverage_pass"] is True


def test_permuted_target_negative_control_reduces_predictability():
    views = _linear_views(view_num=2)
    folds = make_fold_assignment(len(views[0]), n_splits=5, random_state=20)
    aligned = oof_ridge_predictability(views, folds, alpha=1.0)
    permuted_views = [views[0], views[1][np.random.RandomState(9).permutation(80)]]
    permuted = oof_ridge_predictability(permuted_views, folds, alpha=1.0)

    assert np.mean(permuted["oof_consensus_cosine"]) < (
        np.mean(aligned["oof_consensus_cosine"]) - 0.5
    )


def test_five_views_generate_twenty_ordered_pairs():
    pairs = ordered_view_pairs(5)

    assert len(pairs) == 20
    assert len(set(pairs)) == 20
    assert all(source != target for source, target in pairs)


def test_consensus_prediction_shape_is_n_v_d():
    result = _predictability_result()

    assert result["consensus_prediction"].shape == (80, 5, 4)
    assert result["predicted_target_views"].shape == (80, 5, 4, 4)


def test_primary_predictability_score_shape_is_n_v():
    result = _predictability_result()

    assert result["oof_consensus_cosine"].shape == (80, 5)
    assert result["pairwise_cosine_mean"].shape == (80, 5)
    assert result["pairwise_cosine_median"].shape == (80, 5)
    assert result["consensus_negative_mse"].shape == (80, 5)


def test_sample_bootstrap_keeps_all_views_of_each_drawn_sample():
    draws = draw_sample_bootstrap_ids(20, repeats=7, seed=123)
    sample_view_ids = np.repeat(np.arange(20)[:, None], 5, axis=1)

    selected = sample_view_ids[draws]
    assert draws.shape == (7, 20)
    assert selected.shape == (7, 20, 5)
    assert np.all(selected == draws[:, :, None])


def test_separated_scores_have_auc_and_gap_lower_ci_above_null():
    mask = np.zeros((60, 5), dtype=bool)
    mask[:, :2] = True
    scores = np.where(mask, 0.0, 1.0)

    result = bootstrap_reliability(
        {"a1": scores}, mask, repeats=300, seed=20260815
    )["arms"]["a1"]

    assert result["auc_ci_low"] > 0.5
    assert result["gap_ci_low"] > 0.0


def test_random_scores_have_auc_ci_covering_half():
    mask = np.zeros((210, 5), dtype=bool)
    mask[:, :2] = True
    scores = np.random.RandomState(44).normal(size=mask.shape)

    result = bootstrap_reliability(
        {"random": scores}, mask, repeats=500, seed=20260815
    )["arms"]["random"]

    assert result["auc_ci_low"] < 0.5 < result["auc_ci_high"]


def test_paired_bootstrap_delta_is_positive_for_stronger_a1():
    mask = np.zeros((80, 5), dtype=bool)
    mask[:, :2] = True
    a0 = np.random.RandomState(1).normal(size=mask.shape)
    a1 = np.where(mask, -1.0, 1.0)

    paired = bootstrap_reliability(
        {"a1": a1, "a0": a0},
        mask,
        repeats=300,
        seed=20260815,
        paired_comparisons=(("a1", "a0"),),
    )["paired"]["a1_minus_a0"]

    assert paired["delta_auc"] > 0.0
    assert paired["delta_auc_ci_low"] > 0.0
    assert paired["delta_gap"] > 0.0


def test_random_semantic_seed_hash_is_deterministic_and_seed_sensitive():
    kwargs = {"view_num": 5, "latent_dim": 10, "semantic_dim": 10}
    first = DetachedSemanticHeadBank(semantic_seed=1020, **kwargs)
    repeated = DetachedSemanticHeadBank(semantic_seed=1020, **kwargs)
    different = DetachedSemanticHeadBank(semantic_seed=1021, **kwargs)

    assert hash_semantic_heads(first) == hash_semantic_heads(repeated)
    assert hash_semantic_heads(first) != hash_semantic_heads(different)


def test_ridge_fit_helper_exposes_no_label_or_corruption_input():
    parameters = inspect.signature(oof_ridge_predictability).parameters

    assert set(parameters) == {
        "representation_views",
        "fold_assignment",
        "alpha",
    }
    assert all("label" not in name and "corrupt" not in name for name in parameters)


def test_invalid_fold_assignment_is_rejected():
    views = _linear_views(sample_num=20, view_num=2)

    with pytest.raises(ValueError):
        oof_ridge_predictability(views, np.zeros(20, dtype=np.int64))
