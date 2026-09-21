import inspect

import pytest
import torch

from experiments.cyclic_utility import c3_b0_relation_action_protocol as c3
from experiments.generic_contract import generic_relation_action as authoritative
from release_core.action import (
    build_action_weight,
    relation_bce,
    utility_conditioned_relation_loss,
)
from release_core.semantics import compute_view_relations


def _posterior(rows, classes, offset, dtype):
    values = torch.arange(
        1 + offset, 1 + offset + rows * classes, dtype=dtype
    ).reshape(rows, classes)
    return values / values.sum(dim=1, keepdim=True)


def _case(V, S, B, K, L, dtype):
    queries = [_posterior(B, K, 11 * view, dtype) for view in range(V)]
    anchors = [_posterior(L, K, 13 * view + 3, dtype) for view in range(V)]
    target = (torch.arange(B * L * S).reshape(B, L, S) % 3 == 1)
    cycle = torch.arange(1, B * S + 1, dtype=dtype).reshape(B, S)
    cycle = cycle / (B * S + 1.0)
    balance = torch.arange(1, B * L * S + 1, dtype=dtype).reshape(B, L, S)
    balance = balance / (B * L * S + 1.0)
    return queries, anchors, target, cycle, balance


@pytest.mark.parametrize(
    "V,S,B,K,L", ((2, 2, 4, 3, 2), (5, 20, 3, 4, 5), (6, 20, 5, 7, 4))
)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_generic_intermediate_and_final_exact_parity(V, S, B, K, L, dtype):
    queries, anchors, target, cycle, balance = _case(V, S, B, K, L, dtype)
    clean_weight = build_action_weight(cycle, balance, like=queries[0])
    expected_weight = (cycle.detach()[:, None, :] * balance.detach()).detach()
    assert torch.equal(clean_weight, expected_weight)
    clean_denominator = clean_weight.sum()
    expected_denominator = expected_weight.sum()
    assert torch.equal(clean_denominator, expected_denominator)

    expected_views = []
    for query, anchor in zip(queries, anchors):
        clean_probability = compute_view_relations(query, anchor)
        expected_probability = authoritative.relation_probability(query, anchor)
        assert torch.equal(clean_probability, expected_probability)
        clean_bce = relation_bce(clean_probability, target)
        epsilon = torch.finfo(dtype).eps
        clamped = expected_probability.clamp(
            min=epsilon, max=1.0 - epsilon
        ).unsqueeze(-1)
        converted = target.to(dtype=dtype)
        expected_bce = -converted * torch.log(clamped) - (
            1.0 - converted
        ) * torch.log(1.0 - clamped)
        assert torch.equal(clean_bce, expected_bce)
        clean_numerator = torch.sum(clean_weight * clean_bce)
        expected_numerator = torch.sum(expected_weight * expected_bce)
        assert torch.equal(clean_numerator, expected_numerator)
        expected_views.append(expected_numerator / expected_denominator)

    clean_loss, clean_audit = utility_conditioned_relation_loss(
        queries, anchors, target, cycle, balance
    )
    frozen_loss, frozen_audit = authoritative.relation_semantic_loss(
        queries, anchors, target, cycle, balance
    )
    assert torch.equal(clean_loss, frozen_loss)
    assert clean_audit.view_losses == frozen_audit["view_losses"]
    assert clean_audit.denominator == frozen_audit["denominator"]
    assert torch.equal(clean_loss, torch.stack(expected_views).mean())


def test_uniform_utility_is_test_only_historical_control_parity():
    queries, anchors, target, cycle, balance = _case(
        6, 20, 4, 7, 14, torch.float32
    )
    uniform = torch.ones_like(cycle)
    clean_weight = build_action_weight(uniform, balance, like=queries[0])
    clean_loss, _ = utility_conditioned_relation_loss(
        queries, anchors, target, uniform, balance
    )
    frozen_loss, _ = c3.relation_semantic_loss(
        queries, anchors, target, cycle, balance, "TRUE_UNIFORM"
    )
    assert torch.equal(clean_weight, balance)
    assert torch.equal(clean_loss, frozen_loss)


def test_historical_shuffle_u_is_semantic_lineage_not_u_value_shuffle():
    queries, anchors, target, cycle, balance = _case(
        6, 20, 4, 7, 14, torch.float32
    )
    shuffled_target = torch.flip(target, dims=(0, 1))
    shuffled_balance = torch.flip(balance, dims=(0, 1))
    clean_loss, _ = utility_conditioned_relation_loss(
        queries, anchors, shuffled_target, cycle, shuffled_balance
    )
    frozen_loss, _ = c3.relation_semantic_loss(
        queries, anchors, shuffled_target, cycle, shuffled_balance, "SHUFFLE_U"
    )
    assert torch.equal(clean_loss, frozen_loss)


def test_primary_api_has_no_control_arm_or_base_execution():
    parameters = inspect.signature(utility_conditioned_relation_loss).parameters
    assert "arm" not in parameters
    assert tuple(parameters) == (
        "q_query_views", "q_anchor_views", "pred_relation", "u_cycle",
        "balance_weight",
    )
