"""Runtime-dimensional sparse relation semantics with frozen C3 behavior."""

from types import MappingProxyType

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _readonly(value, dtype=None):
    result = np.array(np.asarray(value, dtype=dtype), copy=True, order="C")
    result.setflags(write=False)
    return result


def build_vote_semantic_state(y_gen, class_count):
    actions = _readonly(y_gen, np.int64)
    K = int(class_count)
    _require(
        actions.ndim == 2
        and K >= 2
        and np.all((actions >= 0) & (actions < K)),
        "y_gen runtime boundary mismatch",
    )
    state = np.ascontiguousarray(
        np.eye(K, dtype=np.float64)[actions].mean(axis=1),
        dtype=np.float64,
    )
    _require(
        np.isfinite(state).all()
        and np.all(state >= 0.0)
        and np.allclose(state.sum(axis=1), 1.0, rtol=0.0, atol=1e-15),
        "vote semantic state boundary mismatch",
    )
    state.setflags(write=False)
    return state


def build_sparse_mapping(
    vote_state,
    labeled_ids,
    labeled_targets,
    *,
    class_count,
    labels_per_class,
):
    state = _readonly(vote_state, np.float64)
    labeled = _readonly(labeled_ids, np.int64)
    targets = _readonly(labeled_targets, np.int64)
    K = int(class_count)
    budget = int(labels_per_class)
    _require(
        state.ndim == 2
        and state.shape[1] == K
        and labeled.shape == targets.shape == (K * budget,)
        and np.unique(labeled).size == labeled.size
        and np.all((labeled >= 0) & (labeled < state.shape[0]))
        and np.array_equal(np.bincount(targets, minlength=K), np.full(K, budget))
        and np.isfinite(state).all(),
        "sparse mapping input boundary mismatch",
    )
    anchors = np.empty((K, K), dtype=np.float64)
    contingency = np.zeros((K, K), dtype=np.float64)
    for class_id in range(K):
        rows = labeled[targets == class_id]
        anchors[class_id] = state[rows].mean(axis=0)
        contingency[:, class_id] = state[rows].sum(axis=0)
    row_indices, column_indices = linear_sum_assignment(-contingency)
    mapping = np.empty(K, dtype=np.int64)
    mapping[row_indices] = column_indices
    _require(
        np.array_equal(np.sort(mapping), np.arange(K, dtype=np.int64))
        and np.isfinite(anchors).all()
        and np.isfinite(contingency).all(),
        "sparse mapping construction mismatch",
    )
    return MappingProxyType({
        "anchors": _readonly(anchors),
        "soft_contingency": _readonly(contingency),
        "sparse_mapping": _readonly(mapping),
        "mapping_fit_label_count": int(labeled.size),
        "mapping_fit_full_GT_used": False,
    })


def build_relation_balance_weights(predicted_relation):
    predicted = _readonly(predicted_relation, np.bool_)
    _require(predicted.ndim == 3, "PredRelation must have shape [N_u,L,S]")
    weights = np.empty(predicted.shape, dtype=np.float64)
    for direction_id in range(predicted.shape[2]):
        same = predicted[:, :, direction_id]
        same_count = int(np.count_nonzero(same))
        different_count = int(same.size - same_count)
        _require(
            same_count > 0 and different_count > 0,
            "relation group is empty",
        )
        direction = weights[:, :, direction_id]
        direction[same] = 1.0 / (2.0 * same_count)
        direction[~same] = 1.0 / (2.0 * different_count)
        _require(
            np.isclose(direction[same].sum(), 0.5, rtol=0.0, atol=1e-15)
            and np.isclose(direction[~same].sum(), 0.5, rtol=0.0, atol=1e-15),
            "relation balance normalization mismatch",
        )
    _require(
        np.isfinite(weights).all() and np.all(weights > 0.0),
        "relation weights must be finite and positive",
    )
    return _readonly(weights)


def build_relation_semantics(
    y_gen,
    labeled_ids,
    labeled_targets,
    unlabeled_ids,
    *,
    class_count,
    labels_per_class=2,
):
    actions = _readonly(y_gen, np.int64)
    labeled = _readonly(labeled_ids, np.int64)
    targets = _readonly(labeled_targets, np.int64)
    unlabeled = _readonly(unlabeled_ids, np.int64)
    state = build_vote_semantic_state(actions, class_count)
    fitted = build_sparse_mapping(
        state,
        labeled,
        targets,
        class_count=class_count,
        labels_per_class=labels_per_class,
    )
    class_pred = np.ascontiguousarray(
        fitted["sparse_mapping"][actions[unlabeled]], dtype=np.int64
    )
    predicted = np.ascontiguousarray(
        class_pred[:, None, :] == targets[None, :, None], dtype=np.bool_
    )
    weights = build_relation_balance_weights(predicted)
    expected = (unlabeled.size, labeled.size, actions.shape[1])
    _require(
        predicted.shape == weights.shape == expected,
        "relation semantic runtime shape mismatch",
    )
    return MappingProxyType({
        "vote_state": state,
        "anchors": fitted["anchors"],
        "sparse_mapping": fitted["sparse_mapping"],
        "class_pred": _readonly(class_pred),
        "PredRelation_true": _readonly(predicted),
        "relation_balance_weights_true": weights,
        "mapping_fit_full_GT_used": False,
    })


def relation_probability(q_query, q_anchor):
    _require(
        q_query.ndim == q_anchor.ndim == 2
        and q_query.shape[1] == q_anchor.shape[1],
        "relation posterior shape mismatch",
    )
    return q_query @ q_anchor.detach().transpose(0, 1)


def relation_semantic_loss(
    q_query_views,
    q_anchor_views,
    PredRelation,
    U_cycle,
    balance_weight,
):
    """Compute independently normalized per-view TRUE_U losses and their mean."""
    _require(
        len(q_query_views) == len(q_anchor_views) >= 2,
        "relation action view count mismatch",
    )
    reference = q_query_views[0]
    target = torch.as_tensor(
        PredRelation, device=reference.device, dtype=reference.dtype
    ).detach()
    cycle = torch.as_tensor(
        U_cycle, device=reference.device, dtype=reference.dtype
    ).detach()
    balance = torch.as_tensor(
        balance_weight, device=reference.device, dtype=reference.dtype
    ).detach()
    _require(
        target.ndim == 3
        and cycle.ndim == 2
        and balance.shape == target.shape
        and target.shape[0] == cycle.shape[0]
        and target.shape[2] == cycle.shape[1],
        "relation action tensor shape mismatch",
    )
    action_weight = (cycle[:, None, :] * balance).detach()
    denominator = action_weight.sum()
    epsilon = torch.finfo(reference.dtype).eps
    _require(
        bool(torch.isfinite(denominator).item())
        and bool((denominator > epsilon).item()),
        "relation denominator is non-finite or too small",
    )
    view_losses = []
    for query, anchor in zip(q_query_views, q_anchor_views):
        probability = relation_probability(query, anchor)
        target_tensor = target
        probability = probability.clamp(
            min=epsilon, max=1.0 - epsilon
        )[:, :, None]
        bce = -target_tensor * torch.log(probability) - (
            1.0 - target_tensor
        ) * torch.log(1.0 - probability)
        view_loss = torch.sum(action_weight * bce) / denominator
        _require(bool(torch.isfinite(view_loss).item()), "non-finite relation loss")
        view_losses.append(view_loss)
    loss = torch.stack(view_losses).mean()
    return loss, MappingProxyType({
        "view_count": len(view_losses),
        "view_losses": tuple(float(v.detach().item()) for v in view_losses),
        "denominator": float(denominator.detach().item()),
        "view_arithmetic_mean": True,
        "anchor_posterior_detached": all(
            not value.requires_grad and value.grad_fn is None
            for value in q_anchor_views
        ),
        "U_cycle_detached": not cycle.requires_grad,
        "PredRelation_detached": not target.requires_grad,
        "balance_weights_detached": not balance.requires_grad,
    })
