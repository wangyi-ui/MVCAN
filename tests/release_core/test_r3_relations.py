import inspect

import numpy as np
import pytest
import torch

from experiments.generic_contract import generic_relation_action as authoritative
from release_core.semantics import (
    SparseLabelSplit,
    build_relation_balance_weights,
    build_relation_semantics,
    build_vote_semantic_state,
    compute_view_relations,
    fit_sparse_mapping,
)
from release_core.utility import build_directional_actions


def _synthetic(N, K, S):
    sample_ids = np.arange(N, dtype=np.int64)
    labeled_ids = np.arange(2 * K, dtype=np.int64)
    targets = np.repeat(np.arange(K, dtype=np.int64), 2)
    unlabeled_ids = sample_ids[~np.isin(sample_ids, labeled_ids)]
    y_gen = np.asarray(
        [[(3 * row + 2 * action + row * action) % K for action in range(S)]
         for row in range(N)],
        dtype=np.int64,
    )
    split = SparseLabelSplit(
        sample_ids,
        labeled_ids,
        targets,
        unlabeled_ids,
        K,
        2,
        20,
        "synthetic-%d-%d-%d" % (N, K, S),
    )
    return y_gen, split


@pytest.mark.parametrize("N,K,S", ((13, 3, 2), (19, 4, 20), (23, 5, 20)))
def test_vote_state_exact_authoritative_parity(N, K, S):
    y_gen, _ = _synthetic(N, K, S)
    expected = authoritative.build_vote_semantic_state(y_gen, K)
    actual = build_vote_semantic_state(y_gen, K)
    assert actual.dtype == expected.dtype == np.float64
    assert np.array_equal(actual, expected)


def test_hungarian_mapping_orientation_exact_and_not_reversed():
    y_gen, split = _synthetic(9, 3, 2)
    # Cluster 0 -> class 1, cluster 1 -> class 2, cluster 2 -> class 0.
    cluster_for_class = np.asarray([2, 0, 1], dtype=np.int64)
    y_gen[:6] = np.repeat(cluster_for_class, 2)[:, None]
    state = build_vote_semantic_state(y_gen, 3)
    clean = fit_sparse_mapping(state, split)
    frozen = authoritative.build_sparse_mapping(
        state,
        split.labeled_ids,
        split.labeled_targets,
        class_count=3,
        labels_per_class=2,
    )
    expected = np.asarray([1, 2, 0], dtype=np.int64)
    assert np.array_equal(clean.anchors, frozen["anchors"])
    assert np.array_equal(clean.soft_contingency, frozen["soft_contingency"])
    assert np.array_equal(clean.mapping, frozen["sparse_mapping"])
    assert np.array_equal(clean.mapping, expected)
    assert not np.array_equal(clean.mapping, np.argsort(expected))


@pytest.mark.parametrize("N,K,S", ((13, 3, 2), (19, 4, 20), (23, 5, 20)))
def test_class_pred_predrelation_and_balance_exact_parity(N, K, S):
    y_gen, split = _synthetic(N, K, S)
    clean = build_relation_semantics(y_gen, split)
    frozen = authoritative.build_relation_semantics(
        y_gen,
        split.labeled_ids,
        split.labeled_targets,
        split.unlabeled_ids,
        class_count=K,
        labels_per_class=2,
    )
    assert clean.class_pred.shape == (N - 2 * K, S)
    assert clean.pred_relation.shape == (N - 2 * K, 2 * K, S)
    assert clean.pred_relation.dtype == np.bool_
    assert clean.balance_weights.dtype == np.float64
    assert np.array_equal(clean.sparse_mapping, frozen["sparse_mapping"])
    assert np.array_equal(clean.class_pred, frozen["class_pred"])
    assert np.array_equal(clean.pred_relation, frozen["PredRelation_true"])
    assert np.array_equal(
        clean.balance_weights, frozen["relation_balance_weights_true"]
    )


def test_unlabeled_and_labeled_axis_order_is_not_silently_sorted():
    N, K, S = 10, 2, 2
    sample_ids = np.arange(N, dtype=np.int64)
    labeled_ids = np.asarray([8, 1, 6, 3], dtype=np.int64)
    targets = np.asarray([1, 0, 0, 1], dtype=np.int64)
    unlabeled_ids = sample_ids[~np.isin(sample_ids, labeled_ids)]
    split = SparseLabelSplit(
        sample_ids, labeled_ids, targets, unlabeled_ids, K, 2, 20, "axis-order"
    )
    y_gen = np.indices((N, S)).sum(axis=0) % K
    result = build_relation_semantics(y_gen.astype(np.int64), split)
    expected = (
        result.sparse_mapping[y_gen[unlabeled_ids]][:, None, :]
        == targets[None, :, None]
    )
    assert np.array_equal(result.labeled_ids, labeled_ids)
    assert np.array_equal(result.unlabeled_ids, unlabeled_ids)
    assert np.array_equal(result.pred_relation, expected)


def test_action_metadata_is_reused_and_validated():
    y_gen, split = _synthetic(13, 3, 2)
    result = build_relation_semantics(y_gen, split, build_directional_actions(2))
    assert result.action_count == 2
    with pytest.raises(ValueError):
        build_relation_semantics(y_gen, split, build_directional_actions(5))


def test_balance_normalization_and_empty_groups_fail_closed():
    predicted = np.asarray(
        [[[True, False], [False, True]], [[False, True], [True, False]]],
        dtype=np.bool_,
    )
    clean = build_relation_balance_weights(predicted)
    frozen = authoritative.build_relation_balance_weights(predicted)
    assert np.array_equal(clean, frozen)
    for action in range(predicted.shape[2]):
        same = predicted[:, :, action]
        assert clean[:, :, action][same].sum() == 0.5
        assert clean[:, :, action][~same].sum() == 0.5
        assert clean[:, :, action].sum() == 1.0
    with pytest.raises(ValueError, match="empty"):
        build_relation_balance_weights(np.ones((2, 3, 2), dtype=np.bool_))
    with pytest.raises(ValueError, match="empty"):
        build_relation_balance_weights(np.zeros((2, 3, 2), dtype=np.bool_))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_raw_view_relation_exact_authoritative_parity(dtype):
    query = torch.tensor(
        [[0.2, 0.3, 0.5], [0.7, 0.1, 0.2]], dtype=dtype
    )
    anchor = torch.tensor(
        [[0.1, 0.6, 0.3], [0.8, 0.1, 0.1], [0.4, 0.4, 0.2]],
        dtype=dtype,
    )
    actual = compute_view_relations(query, anchor)
    expected = authoritative.relation_probability(query, anchor)
    assert actual.shape == (2, 3)
    assert torch.equal(actual, expected)
    assert torch.equal(actual, query @ anchor.transpose(0, 1))


def test_query_gradient_capability_and_anchor_stop_are_structural():
    query = torch.rand((4, 3), generator=torch.Generator().manual_seed(20))
    anchor = torch.rand((5, 3), generator=torch.Generator().manual_seed(21))
    query.requires_grad_()
    anchor.requires_grad_()
    output = compute_view_relations(query, anchor)
    assert output.requires_grad is True
    assert query.requires_grad is True
    assert anchor.requires_grad is True
    assert "q_anchor.detach()" in inspect.getsource(compute_view_relations)
    assert "return q_query @" in inspect.getsource(compute_view_relations)


@pytest.mark.parametrize(
    "value,K",
    (
        (np.zeros((3,), dtype=np.int64), 2),
        (np.zeros((3, 2), dtype=np.float64), 2),
        (np.full((3, 2), 2, dtype=np.int64), 2),
    ),
)
def test_invalid_y_gen_fails_closed(value, K):
    with pytest.raises(ValueError):
        build_vote_semantic_state(value, K)

