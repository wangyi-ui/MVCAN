"""Tests for preregistered B4-A1a Information Utility admission."""

import copy
import inspect

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from irv.b3_audit import hash_semantic_heads
from irv.b3_predictability_diagnostics import make_fold_assignment
from irv.b3_predictability_diagnostics import oof_ridge_predictability
from irv.b4_information_utility import compute_information_utility
from irv.b4_information_utility import normalized_weighted_mean
from irv.b4_information_utility import pair_admission_weight
from irv.b4_information_utility import uniform_equivalence_max_abs_error
from irv.b4_information_utility import unordered_view_pairs
from irv.b4_information_utility import weighted_symmetric_semantic_infonce
from irv.semantic_head import DetachedSemanticHeadBank
from irv.semantic_loss import uniform_cross_view_infonce


def _scores(sample_num=12, view_num=5):
    random_state = np.random.RandomState(5)
    return random_state.normal(size=(sample_num, view_num))


def _z_views(sample_num=12, view_num=5, feature_dim=10):
    random_state = np.random.RandomState(9)
    return [
        torch.from_numpy(
            random_state.normal(size=(sample_num, feature_dim))
        ).float()
        for _ in range(view_num)
    ]


def _semantic_views(sample_num=12):
    heads = DetachedSemanticHeadBank(5, 10, 10, 1020)
    return heads.forward_views(_z_views(sample_num=sample_num))


def test_per_view_rank_utility_preserves_shape():
    scores = _scores()
    utility = compute_information_utility(scores)

    assert utility.shape == scores.shape


def test_utility_is_in_closed_unit_interval():
    utility = compute_information_utility(_scores())

    assert np.min(utility) >= 0.0
    assert np.max(utility) <= 1.0


def test_utility_is_exactly_deterministic():
    scores = _scores()

    assert np.array_equal(
        compute_information_utility(scores),
        compute_information_utility(scores.copy()),
    )


def test_rank_is_per_view_not_global_flattened():
    scores = np.array([[0.0, 100.0], [1.0, 0.0], [2.0, 50.0]])
    utility = compute_information_utility(scores)

    assert np.array_equal(utility[:, 0], [0.0, 0.5, 1.0])
    assert np.array_equal(utility[:, 1], [1.0, 0.0, 0.5])


def test_utility_api_exposes_no_label_or_mask_input():
    parameters = inspect.signature(compute_information_utility).parameters

    assert set(parameters) == {"predictability_scores"}
    assert all("label" not in name and "mask" not in name for name in parameters)


def test_strictly_increasing_view_scores_reach_zero_and_one():
    base = np.arange(20, dtype=np.float64)
    scores = np.stack([base + 100.0 * view for view in range(5)], axis=1)
    utility = compute_information_utility(scores)

    assert np.all(utility.min(axis=0) == 0.0)
    assert np.all(utility.max(axis=0) == 1.0)


def test_pair_weight_has_shape_n():
    utility = torch.from_numpy(compute_information_utility(_scores())).float()

    assert pair_admission_weight(utility, 1, 4).shape == (12,)


def test_pair_weight_is_exact_product():
    utility = torch.from_numpy(compute_information_utility(_scores())).float()

    actual = pair_admission_weight(utility, 0, 3)

    assert torch.equal(actual, utility[:, 0] * utility[:, 3])


def test_pair_weight_is_finite_and_nonnegative():
    utility = torch.from_numpy(compute_information_utility(_scores())).float()
    weight = pair_admission_weight(utility, 0, 1)

    assert torch.isfinite(weight).all()
    assert torch.all(weight >= 0.0)


def test_normalized_weighted_ce_matches_manual_formula():
    losses = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    weight = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    expected = torch.sum(losses * weight) / (torch.sum(weight) + 1e-12)

    actual = normalized_weighted_mean(losses, weight)

    assert torch.equal(actual, expected)


def test_all_one_weighted_loss_matches_frozen_b3_uniform_loss():
    semantic_views = _semantic_views()
    utility = torch.ones(12, 5)

    weighted, _ = weighted_symmetric_semantic_infonce(
        semantic_views, utility, temperature=0.2
    )
    original, _ = uniform_cross_view_infonce(semantic_views, temperature=0.2)

    assert abs(weighted.item() - original.item()) <= 1e-7
    assert uniform_equivalence_max_abs_error(semantic_views, 0.2) <= 1e-7


def test_two_view_symmetric_pair_loss_is_exact_bidirectional_average():
    semantic_views = _semantic_views()[:2]
    utility = torch.from_numpy(compute_information_utility(_scores(view_num=2))).float()
    actual, _ = weighted_symmetric_semantic_infonce(
        semantic_views, utility, temperature=0.2
    )
    logits = semantic_views[0] @ semantic_views[1].T / 0.2
    target = torch.arange(12)
    weight = utility[:, 0] * utility[:, 1]
    forward = normalized_weighted_mean(
        F.cross_entropy(logits, target, reduction="none"), weight
    )
    backward = normalized_weighted_mean(
        F.cross_entropy(logits.T, target, reduction="none"), weight
    )

    assert torch.equal(actual, 0.5 * (forward + backward))


def test_five_views_have_ten_unordered_pairs():
    assert len(unordered_view_pairs(5)) == 10


def test_utility_tensor_is_detached():
    scores = torch.from_numpy(_scores()).float().requires_grad_(True)
    utility = compute_information_utility(scores)

    assert utility.requires_grad is False
    assert utility.grad_fn is None


def test_frozen_z_receives_no_gradient():
    z_views = [value.requires_grad_(True) for value in _z_views()]
    heads = DetachedSemanticHeadBank(5, 10, 10, 1020)
    semantic_views = heads.forward_views(z_views)
    utility = torch.from_numpy(compute_information_utility(_scores())).float()
    loss, _ = weighted_symmetric_semantic_infonce(
        semantic_views, utility, temperature=0.2
    )

    loss.backward()

    assert all(value.grad is None for value in z_views)


def test_semantic_heads_receive_finite_nonzero_gradients():
    heads = DetachedSemanticHeadBank(5, 10, 10, 1020)
    semantic_views = heads.forward_views(_z_views())
    utility = torch.from_numpy(compute_information_utility(_scores())).float()
    loss, _ = weighted_symmetric_semantic_infonce(
        semantic_views, utility, temperature=0.2
    )

    loss.backward()

    assert all(parameter.grad is not None for parameter in heads.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in heads.parameters())
    assert sum(parameter.grad.abs().sum().item() for parameter in heads.parameters()) > 0


def test_frozen_backbone_parameters_keep_none_grad():
    backbone = torch.nn.Linear(6, 10)
    backbone.requires_grad_(False)
    features = [torch.randn(12, 6) for _ in range(5)]
    z_views = [backbone(value).detach() for value in features]
    heads = DetachedSemanticHeadBank(5, 10, 10, 1020)
    utility = torch.from_numpy(compute_information_utility(_scores())).float()
    loss, _ = weighted_symmetric_semantic_infonce(
        heads.forward_views(z_views), utility, temperature=0.2
    )

    loss.backward()

    assert all(parameter.grad is None for parameter in backbone.parameters())


def test_template_initialization_gives_exact_arm_hash_match():
    template = DetachedSemanticHeadBank(5, 10, 10, 1020)
    state = copy.deepcopy(template.state_dict())
    uniform = DetachedSemanticHeadBank(5, 10, 10, 1020)
    utility = DetachedSemanticHeadBank(5, 10, 10, 1020)
    uniform.load_state_dict(state, strict=True)
    utility.load_state_dict(state, strict=True)

    assert hash_semantic_heads(uniform) == hash_semantic_heads(utility)


def test_one_optimizer_step_changes_semantic_hash():
    heads = DetachedSemanticHeadBank(5, 10, 10, 1020)
    initial = hash_semantic_heads(heads)
    optimizer = torch.optim.Adam(heads.parameters(), lr=1e-4)
    utility = torch.from_numpy(compute_information_utility(_scores())).float()
    loss, _ = weighted_symmetric_semantic_infonce(
        heads.forward_views(_z_views()), utility, temperature=0.2
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert hash_semantic_heads(heads) != initial


def test_frozen_b3_oof_predictability_reuse_is_deterministic():
    views = [value.numpy() for value in _z_views(sample_num=30)]
    folds = make_fold_assignment(30, n_splits=5, random_state=20)

    first = oof_ridge_predictability(views, folds, alpha=1.0)
    repeated = oof_ridge_predictability(views, folds, alpha=1.0)

    assert np.array_equal(
        first["oof_consensus_cosine"], repeated["oof_consensus_cosine"]
    )


def test_same_b3_t_always_produces_same_utility():
    views = [value.numpy() for value in _z_views(sample_num=30)]
    folds = make_fold_assignment(30, n_splits=5, random_state=20)
    t_scores = oof_ridge_predictability(
        views, folds, alpha=1.0
    )["oof_consensus_cosine"]

    assert np.array_equal(
        compute_information_utility(t_scores),
        compute_information_utility(t_scores),
    )


def test_zero_and_tiny_total_weights_are_numerically_stable():
    losses = torch.tensor([1.0, 2.0, 3.0])
    zero = normalized_weighted_mean(losses, torch.zeros(3))
    tiny = normalized_weighted_mean(losses, torch.full((3,), 1e-30))

    assert zero.item() == 0.0
    assert torch.isfinite(tiny)
    assert tiny.item() >= 0.0


def test_full_batch_shapes_n210_v5_d10():
    scores = _scores(sample_num=210, view_num=5)
    utility = torch.from_numpy(compute_information_utility(scores)).float()
    semantic_views = _semantic_views(sample_num=210)

    loss, diagnostics = weighted_symmetric_semantic_infonce(
        semantic_views, utility, temperature=0.2
    )

    assert utility.shape == (210, 5)
    assert all(value.shape == (210, 10) for value in semantic_views)
    assert loss.ndim == 0
    assert diagnostics["pair_count"] == 10


def test_invalid_nonfinite_predictability_is_rejected():
    scores = _scores()
    scores[0, 0] = np.nan

    with pytest.raises(ValueError):
        compute_information_utility(scores)
