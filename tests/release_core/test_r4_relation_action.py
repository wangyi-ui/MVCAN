import inspect

import pytest
import torch

from release_core.action import (
    RelationActionAudit,
    build_action_weight,
    relation_bce,
    utility_conditioned_relation_loss,
)
from release_core.semantics import compute_view_relations


def _posterior(rows, classes, offset=0, dtype=torch.float64):
    values = torch.arange(
        1 + offset, 1 + offset + rows * classes, dtype=dtype
    ).reshape(rows, classes)
    return values / values.sum(dim=1, keepdim=True)


def _inputs(dtype=torch.float64):
    queries = [_posterior(4, 3, 5 * view, dtype) for view in range(3)]
    anchors = [_posterior(2, 3, 7 * view + 2, dtype) for view in range(3)]
    target = (torch.arange(4 * 2 * 5).reshape(4, 2, 5) % 3 == 0)
    cycle = torch.arange(1, 21, dtype=dtype).reshape(4, 5) / 21.0
    balance = torch.arange(1, 41, dtype=dtype).reshape(4, 2, 5) / 40.0
    return queries, anchors, target, cycle, balance


def test_build_action_weight_exact_shape_broadcast_and_detach():
    _, _, _, cycle, balance = _inputs()
    cycle.requires_grad_()
    balance.requires_grad_()
    weight = build_action_weight(
        cycle, balance, like=torch.ones(1, 1, dtype=cycle.dtype)
    )
    expected = cycle.detach()[:, None, :] * balance.detach()
    assert weight.shape == (4, 2, 5)
    assert torch.equal(weight, expected)
    assert torch.equal(
        weight[:, 0, :], cycle.detach() * balance.detach()[:, 0, :]
    )
    assert torch.equal(
        weight[:, 1, :], cycle.detach() * balance.detach()[:, 1, :]
    )
    assert weight.requires_grad is False
    assert weight.grad_fn is None


@pytest.mark.parametrize(
    "cycle,balance",
    (
        (torch.tensor([[-1.0, 1.0]]), torch.ones(1, 2, 2)),
        (torch.tensor([[float("nan"), 1.0]]), torch.ones(1, 2, 2)),
        (torch.ones(1, 2), torch.tensor([[[0.0, 1.0], [1.0, 1.0]]])),
        (torch.ones(1, 2), torch.ones(2, 2, 2)),
    ),
)
def test_build_action_weight_invalid_inputs_fail_closed(cycle, balance):
    with pytest.raises(ValueError):
        build_action_weight(cycle, balance, like=torch.ones(1, 1))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_relation_bce_exact_dtype_epsilon_clamp_and_broadcast(dtype):
    probability = torch.tensor([[0.0, 1.0], [0.25, 0.75]], dtype=dtype)
    target = torch.tensor(
        [[[0, 1, 0], [1, 0, 1]], [[1, 0, 1], [0, 1, 0]]],
        dtype=torch.bool,
    )
    actual = relation_bce(probability, target)
    epsilon = torch.finfo(dtype).eps
    clamped = probability.clamp(epsilon, 1.0 - epsilon).unsqueeze(-1)
    converted = target.to(dtype=dtype)
    expected = -converted * torch.log(clamped) - (1.0 - converted) * torch.log(
        1.0 - clamped
    )
    assert actual.shape == (2, 2, 3)
    assert torch.equal(actual, expected)
    assert torch.equal(clamped[0, 0], torch.tensor([epsilon], dtype=dtype))
    assert torch.equal(
        clamped[0, 1], torch.tensor([1.0 - epsilon], dtype=dtype)
    )


def test_relation_bce_binary_target_and_shape_fail_closed():
    probability = torch.full((2, 3), 0.5)
    with pytest.raises(ValueError, match="binary"):
        relation_bce(probability, torch.full((2, 3, 2), 0.5))
    with pytest.raises(ValueError, match="shape"):
        relation_bce(probability, torch.zeros(2, 4, 2))


def test_loss_exact_global_normalization_per_view_and_arithmetic_mean():
    queries, anchors, target, cycle, balance = _inputs()
    loss, audit = utility_conditioned_relation_loss(
        queries, anchors, target, cycle, balance
    )
    weight = (cycle[:, None, :] * balance).detach()
    denominator = weight.sum()
    expected_views = []
    for query, anchor in zip(queries, anchors):
        probability = compute_view_relations(query, anchor)
        bce = relation_bce(probability, target)
        expected_views.append(torch.sum(weight * bce) / denominator)
    expected = torch.stack(expected_views).mean()
    assert torch.equal(loss, expected)
    assert audit == RelationActionAudit(
        view_count=3,
        view_losses=tuple(float(value.item()) for value in expected_views),
        denominator=float(denominator.item()),
        view_arithmetic_mean=True,
        query_gradient_capable=False,
        anchor_stop_gradient_enforced=True,
        u_cycle_detached=True,
        pred_relation_detached=True,
        balance_detached=True,
        action_weight_detached=True,
        numerical_epsilon_only=True,
    )
    assert bool(torch.isfinite(loss).item())


def test_denominator_zero_too_small_and_nonfinite_fail_closed():
    queries, anchors, target, cycle, balance = _inputs(torch.float32)
    with pytest.raises(ValueError, match="denominator"):
        utility_conditioned_relation_loss(
            queries, anchors, target, torch.zeros_like(cycle), balance
        )
    tiny = torch.full_like(cycle, torch.finfo(torch.float32).tiny)
    with pytest.raises(ValueError, match="denominator"):
        utility_conditioned_relation_loss(queries, anchors, target, tiny, balance)
    huge = torch.full_like(cycle, torch.finfo(torch.float32).max)
    with pytest.raises(ValueError, match="denominator"):
        utility_conditioned_relation_loss(
            queries, anchors, target, huge, balance * 2.0
        )


def test_query_gradient_capability_and_anchor_stop_are_structural_only():
    queries, anchors, target, cycle, balance = _inputs(torch.float32)
    for value in queries + anchors:
        value.requires_grad_()
    cycle.requires_grad_()
    balance.requires_grad_()
    loss, audit = utility_conditioned_relation_loss(
        queries, anchors, target, cycle, balance
    )
    assert loss.requires_grad is True
    assert loss.grad_fn is not None
    assert audit.query_gradient_capable is True
    assert "compute_view_relations(query, anchor)" in inspect.getsource(
        utility_conditioned_relation_loss
    )
    assert "q_anchor.detach()" in inspect.getsource(compute_view_relations)
    assert ".backward(" not in inspect.getsource(utility_conditioned_relation_loss)


@pytest.mark.parametrize("scale", (0.5, 2.0, 5.0))
def test_positive_global_utility_scaling_is_numerically_invariant(scale):
    queries, anchors, target, cycle, balance = _inputs(torch.float64)
    original, _ = utility_conditioned_relation_loss(
        queries, anchors, target, cycle, balance
    )
    scaled, _ = utility_conditioned_relation_loss(
        queries, anchors, target, scale * cycle, balance
    )
    assert torch.allclose(original, scaled, rtol=1e-15, atol=1e-15)


def test_runtime_shape_and_view_contracts_fail_closed():
    queries, anchors, target, cycle, balance = _inputs()
    with pytest.raises(ValueError, match="V >= 2"):
        utility_conditioned_relation_loss(
            queries[:1], anchors[:1], target, cycle, balance
        )
    with pytest.raises(ValueError, match="query views"):
        utility_conditioned_relation_loss(
            queries + [torch.ones(3, 3)], anchors + [torch.ones(2, 3)],
            target, cycle, balance,
        )
