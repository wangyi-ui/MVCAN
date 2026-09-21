"""Frozen sparse relation targets and raw local-view relation primitive."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from release_core.utility.action_space import ActionSpace

from .sparse_labels import SparseLabelSplit, validate_sparse_label_split


def _readonly(value, dtype=None):
    result = np.array(np.asarray(value, dtype=dtype), copy=True, order="C")
    result.setflags(write=False)
    return result


def _integer_matrix(value, name):
    if isinstance(value, torch.Tensor):
        if value.requires_grad or value.device.type != "cpu":
            raise ValueError(name + " must be a detached CPU integer tensor")
        value = value.numpy()
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.size == 0
        or array.dtype == np.bool_
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(name + " must be a non-empty rank-2 integer array")
    return _readonly(array, np.int64)


@dataclass(frozen=True)
class SparseMapping:
    """Frozen cluster-to-class assignment fitted from labeled rows."""

    anchors: np.ndarray
    soft_contingency: np.ndarray
    mapping: np.ndarray
    fit_label_count: int

    @property
    def sparse_mapping(self):
        return self.mapping


@dataclass(frozen=True)
class RelationSemantics:
    """Immutable action-indexed semantic targets on the split axes."""

    vote_state: np.ndarray
    anchors: np.ndarray
    soft_contingency: np.ndarray
    sparse_mapping: np.ndarray
    class_pred: np.ndarray
    pred_relation: np.ndarray
    balance_weights: np.ndarray
    sample_ids: np.ndarray
    labeled_ids: np.ndarray
    unlabeled_ids: np.ndarray
    action_count: int

def build_vote_semantic_state(y_gen, class_count):
    """Build the exact unweighted cross-action cluster vote state."""
    actions = _integer_matrix(y_gen, "y_gen")
    if isinstance(class_count, (bool, np.bool_)) or not isinstance(
        class_count, (int, np.integer)
    ):
        raise ValueError("class_count must be an integer")
    class_count = int(class_count)
    if class_count < 2 or np.any((actions < 0) | (actions >= class_count)):
        raise ValueError("y_gen contains an invalid cluster index")
    state = np.ascontiguousarray(
        np.eye(class_count, dtype=np.float64)[actions].mean(axis=1),
        dtype=np.float64,
    )
    if (
        not np.isfinite(state).all()
        or np.any(state < 0.0)
        or not np.allclose(
            state.sum(axis=1), 1.0, rtol=0.0, atol=1e-15
        )
    ):
        raise ValueError("vote semantic state normalization failed")
    state.setflags(write=False)
    return state


def fit_sparse_mapping(vote_state, split):
    """Fit the historical Hungarian cluster-to-class orientation."""
    split = validate_sparse_label_split(split)
    state = np.asarray(vote_state)
    if (
        state.ndim != 2
        or state.shape != (split.sample_ids.size, split.class_count)
        or not np.issubdtype(state.dtype, np.number)
    ):
        raise ValueError("vote_state shape or dtype is invalid")
    state = _readonly(state, np.float64)
    if (
        not np.isfinite(state).all()
        or np.any(state < 0.0)
        or not np.allclose(state.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)
    ):
        raise ValueError("vote_state values are invalid")

    row_lookup = {int(sample_id): index for index, sample_id in enumerate(
        split.sample_ids.tolist()
    )}
    labeled_rows = np.asarray(
        [row_lookup[int(sample_id)] for sample_id in split.labeled_ids],
        dtype=np.int64,
    )
    class_count = split.class_count
    anchors = np.empty((class_count, class_count), dtype=np.float64)
    contingency = np.zeros((class_count, class_count), dtype=np.float64)
    for class_id in range(class_count):
        rows = labeled_rows[split.labeled_targets == class_id]
        anchors[class_id] = state[rows].mean(axis=0)
        contingency[:, class_id] = state[rows].sum(axis=0)

    row_indices, column_indices = linear_sum_assignment(-contingency)
    mapping = np.empty(class_count, dtype=np.int64)
    mapping[row_indices] = column_indices
    if not np.array_equal(
        np.sort(mapping), np.arange(class_count, dtype=np.int64)
    ):
        raise ValueError("sparse mapping is not a class permutation")
    return SparseMapping(
        anchors=_readonly(anchors),
        soft_contingency=_readonly(contingency),
        mapping=_readonly(mapping),
        fit_label_count=int(split.labeled_ids.size),
    )


def build_relation_balance_weights(pred_relation):
    """Balance same/different edges independently for every action."""
    value = np.asarray(pred_relation)
    if value.ndim != 3 or value.dtype != np.bool_ or value.size == 0:
        raise ValueError("pred_relation must be a non-empty bool [Nu,L,S] array")
    predicted = _readonly(value, np.bool_)
    weights = np.empty(predicted.shape, dtype=np.float64)
    for action_id in range(predicted.shape[2]):
        same = predicted[:, :, action_id]
        same_count = int(np.count_nonzero(same))
        different_count = int(same.size - same_count)
        if same_count == 0 or different_count == 0:
            raise ValueError("relation group is empty")
        action_weights = weights[:, :, action_id]
        action_weights[same] = 1.0 / (2.0 * same_count)
        action_weights[~same] = 1.0 / (2.0 * different_count)
        if not (
            np.isclose(
                action_weights[same].sum(), 0.5, rtol=0.0, atol=1e-15
            )
            and np.isclose(
                action_weights[~same].sum(), 0.5, rtol=0.0, atol=1e-15
            )
        ):
            raise ValueError("relation balance normalization failed")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("relation balance weights must be finite and positive")
    return _readonly(weights)


def build_relation_semantics(y_gen, split, actions=None):
    """Construct action-indexed sparse relation targets from frozen assignments."""
    split = validate_sparse_label_split(split)
    assignments = _integer_matrix(y_gen, "y_gen")
    if assignments.shape[0] != split.sample_ids.size:
        raise ValueError("y_gen sample axis does not match sample_ids")
    if actions is not None:
        if not isinstance(actions, ActionSpace):
            raise ValueError("actions must be an R2 ActionSpace")
        if assignments.shape[1] != actions.S:
            raise ValueError("y_gen action axis does not match actions")

    vote_state = build_vote_semantic_state(assignments, split.class_count)
    fitted = fit_sparse_mapping(vote_state, split)
    row_lookup = {int(sample_id): index for index, sample_id in enumerate(
        split.sample_ids.tolist()
    )}
    unlabeled_rows = np.asarray(
        [row_lookup[int(sample_id)] for sample_id in split.unlabeled_ids],
        dtype=np.int64,
    )
    class_pred = np.ascontiguousarray(
        fitted.mapping[assignments[unlabeled_rows]], dtype=np.int64
    )
    pred_relation = np.ascontiguousarray(
        class_pred[:, None, :]
        == split.labeled_targets[None, :, None],
        dtype=np.bool_,
    )
    balance_weights = build_relation_balance_weights(pred_relation)
    expected_shape = (
        split.unlabeled_ids.size,
        split.labeled_ids.size,
        assignments.shape[1],
    )
    if pred_relation.shape != expected_shape or balance_weights.shape != expected_shape:
        raise ValueError("relation semantic shape construction failed")
    return RelationSemantics(
        vote_state=vote_state,
        anchors=fitted.anchors,
        soft_contingency=fitted.soft_contingency,
        sparse_mapping=fitted.mapping,
        class_pred=_readonly(class_pred),
        pred_relation=_readonly(pred_relation),
        balance_weights=balance_weights,
        sample_ids=split.sample_ids,
        labeled_ids=split.labeled_ids,
        unlabeled_ids=split.unlabeled_ids,
        action_count=int(assignments.shape[1]),
    )


def compute_view_relations(q_query, q_anchor):
    """Compute raw local-coordinate dot products with stopped anchor gradient."""
    if not isinstance(q_query, torch.Tensor) or not isinstance(q_anchor, torch.Tensor):
        raise ValueError("q_query and q_anchor must be torch tensors")
    if (
        q_query.ndim != 2
        or q_anchor.ndim != 2
        or q_query.shape[0] == 0
        or q_anchor.shape[0] == 0
        or q_query.shape[1] == 0
        or q_query.shape[1] != q_anchor.shape[1]
    ):
        raise ValueError("query and anchor must have compatible [B,K] and [L,K] shapes")
    if (
        not q_query.is_floating_point()
        or not q_anchor.is_floating_point()
        or q_query.dtype != q_anchor.dtype
        or q_query.device != q_anchor.device
    ):
        raise ValueError("query and anchor dtype/device contract mismatch")
    if not bool(torch.isfinite(q_query).all().item()) or not bool(
        torch.isfinite(q_anchor).all().item()
    ):
        raise ValueError("query and anchor must be finite")
    return q_query @ q_anchor.detach().transpose(0, 1)
