"""Pre-registered C5-B0 utility-validated semantic expansion primitives.

Sparse labels determine semantic content. Frozen ``U_cycle`` determines action
validity. Utility is used only once to select fixed pseudo-semantic anchors;
training reuses the frozen C3-B0 TRUE_U relation action without pseudo-label CE.
"""

from collections import OrderedDict

import numpy as np
import torch

from experiments.cyclic_utility import c3_a0_utility_conditioned_action_granularity_protocol as c3a0
from experiments.cyclic_utility import c3_b0_relation_action_protocol as c3b0
from irv.b4_information_utility import tensor_sha256


STAGE = "C5-B0"
SEEDS = (20, 30, 50)
ARMS = ("BASE", "TRUE_U_EXPAND", "PERMUTED_U_EXPAND")
N = 1400
V = 6
K = 7
S = 20
L = 14
NU = 1386
EXPANSION_BUDGET_FRACTION = 0.20
FORMAL_EPOCHS = c3b0.FORMAL_EPOCHS
BATCH_SIZE = c3b0.BATCH_SIZE
PREREGISTERED_PROTOCOL_SHA256 = (
    "88d1c395946c008ef64c117c6c56ff608d82ba2671658b6dad606d352af9b385"
)
RELATION_TARGET_AXIS_ORDER = (
    "unlabeled_query", "combined_anchor", "direction"
)
PRIMARY_METRIC = "ACC"
SECONDARY_METRICS = ("NMI", "ARI", "Balanced_ACC")
PRE_GT_ARRAY_KEYS = (
    "sample_ids",
    "final_predictions",
    "eligible_ids",
    "selected_direction",
    "admission_score",
    "expansion_mask",
    "pseudo_semantic_label",
    "selected_pseudo_anchor_ids",
    "selected_pseudo_anchor_labels",
    "real_anchor_ids",
    "real_anchor_labels",
    "combined_anchor_ids",
    "combined_anchor_labels",
    "combined_anchor_is_pseudo",
)


def _require(condition, message="C5_B0_PROTOCOL_FAIL_CLOSED"):
    if not condition:
        raise RuntimeError(message)


def _readonly(value, dtype=None):
    array = np.array(np.asarray(value, dtype=dtype), copy=True, order="C")
    array.setflags(write=False)
    return array


def logical_sha256(value):
    return tensor_sha256(value)


def validate_seed(seed):
    value = int(seed)
    if value not in SEEDS:
        raise ValueError("C5-B0 seed must be one of " + str(SEEDS))
    return value


def validate_arm(arm):
    if arm not in ARMS:
        raise ValueError("C5-B0 arm must be one of " + str(ARMS))
    return arm


def expansion_count(eligible_count):
    count = int(eligible_count)
    _require(count > 0, "C5_B0_EMPTY_ELIGIBLE_SET_FAIL_CLOSED")
    budget = int(np.floor(EXPANSION_BUDGET_FRACTION * count))
    _require(budget > 0, "C5_B0_EMPTY_EXPANSION_FAIL_CLOSED")
    return budget


def select_pseudo_semantic_anchors(
    arm, eligible_ids, action_utility, directional_pred,
):
    """Select fixed pseudo anchors from [Ne,20] actions and [Ne,20] labels."""
    active_arm = validate_arm(arm)
    _require(active_arm != "BASE", "C5_B0_BASE_HAS_NO_EXPANSION")
    ids = _readonly(eligible_ids, np.int64)
    utility = _readonly(action_utility, np.float64)
    predicted = _readonly(directional_pred, np.int64)
    eligible_count = int(ids.size)
    _require(
        ids.ndim == 1
        and np.unique(ids).size == eligible_count
        and utility.shape == predicted.shape == (eligible_count, S)
        and np.isfinite(utility).all()
        and np.all(utility >= 0.0)
        and np.all(utility.sum(axis=1) > 0.0)
        and np.all((predicted >= 0) & (predicted < K)),
        "C5_B0_EXPANSION_INPUT_FAIL_CLOSED",
    )
    # np.argmax returns the smallest index for exact ties.
    selected_direction = np.ascontiguousarray(
        np.argmax(utility, axis=1), dtype=np.int64
    )
    rows = np.arange(eligible_count, dtype=np.int64)
    admission_score = np.ascontiguousarray(
        utility[rows, selected_direction], dtype=np.float64
    )
    pseudo_semantic_label = np.ascontiguousarray(
        predicted[rows, selected_direction], dtype=np.int64
    )
    budget = expansion_count(eligible_count)
    ranking = np.lexsort((ids, -admission_score))
    selected_rows = np.ascontiguousarray(ranking[:budget], dtype=np.int64)
    expansion_mask = np.zeros(eligible_count, dtype=np.bool_)
    expansion_mask[selected_rows] = True
    expansion_mask = np.ascontiguousarray(expansion_mask)
    selected_ids = np.ascontiguousarray(ids[selected_rows], dtype=np.int64)
    selected_labels = np.ascontiguousarray(
        pseudo_semantic_label[selected_rows], dtype=np.int64
    )
    for value in (
        selected_direction, admission_score, pseudo_semantic_label,
        selected_rows, expansion_mask, selected_ids, selected_labels,
    ):
        value.setflags(write=False)
    return {
        "arm": active_arm,
        "eligible_ids": ids,
        "eligible_count": eligible_count,
        "expansion_budget_fraction": EXPANSION_BUDGET_FRACTION,
        "expansion_count": budget,
        "selected_direction": selected_direction,
        "admission_score": admission_score,
        "expansion_mask": expansion_mask,
        "pseudo_semantic_label": pseudo_semantic_label,
        "selected_rows": selected_rows,
        "selected_pseudo_anchor_ids": selected_ids,
        "selected_pseudo_anchor_labels": selected_labels,
        "selection_order": "(-admission_score, sample_id)",
        "argmax_tie_break": "smallest_direction_index",
        "RNG_used": False,
        "continuous_U_weighting_used": False,
        "posterior_TRUE_U_used": False,
    }


def build_anchor_plan(
    arm, eligible_ids, action_utility, directional_pred,
    real_anchor_ids, real_anchor_labels,
):
    """Build [14+B] fixed semantic support; BASE remains exactly 14 anchors."""
    active_arm = validate_arm(arm)
    eligible = _readonly(eligible_ids, np.int64)
    real_ids = _readonly(real_anchor_ids, np.int64)
    real_labels = _readonly(real_anchor_labels, np.int64)
    _require(
        eligible.ndim == 1
        and real_ids.shape == real_labels.shape == (L,)
        and np.unique(real_ids).size == L
        and np.intersect1d(eligible, real_ids).size == 0
        and np.all((real_labels >= 0) & (real_labels < K)),
        "C5_B0_REAL_ANCHOR_BOUNDARY_FAIL_CLOSED",
    )
    if active_arm == "BASE":
        selected_direction = _readonly([], np.int64)
        admission_score = _readonly([], np.float64)
        expansion_mask = _readonly(np.zeros(eligible.size, dtype=np.bool_))
        selected_ids = _readonly([], np.int64)
        selected_labels = _readonly([], np.int64)
        pseudo_labels = _readonly([], np.int64)
        count = 0
    else:
        selected = select_pseudo_semantic_anchors(
            active_arm, eligible, action_utility, directional_pred
        )
        selected_direction = selected["selected_direction"]
        admission_score = selected["admission_score"]
        expansion_mask = selected["expansion_mask"]
        pseudo_labels = selected["pseudo_semantic_label"]
        selected_ids = selected["selected_pseudo_anchor_ids"]
        selected_labels = selected["selected_pseudo_anchor_labels"]
        count = selected["expansion_count"]
        _require(
            np.intersect1d(selected_ids, real_ids).size == 0
            and np.isin(selected_ids, eligible).all(),
            "C5_B0_SELECTED_ANCHOR_LINEAGE_FAIL_CLOSED",
        )
    combined_ids = np.ascontiguousarray(
        np.concatenate((real_ids, selected_ids)), dtype=np.int64
    )
    combined_labels = np.ascontiguousarray(
        np.concatenate((real_labels, selected_labels)), dtype=np.int64
    )
    is_pseudo = np.ascontiguousarray(
        np.concatenate((
            np.zeros(L, dtype=np.bool_),
            np.ones(count, dtype=np.bool_),
        )),
        dtype=np.bool_,
    )
    _require(
        combined_ids.shape == combined_labels.shape == is_pseudo.shape
        == (L + count,)
        and np.unique(combined_ids).size == L + count,
        "C5_B0_COMBINED_ANCHOR_BOUNDARY_FAIL_CLOSED",
    )
    for value in (combined_ids, combined_labels, is_pseudo):
        value.setflags(write=False)
    return {
        "arm": active_arm,
        "eligible_ids": eligible,
        "eligible_count": int(eligible.size),
        "expansion_budget_fraction": EXPANSION_BUDGET_FRACTION,
        "expansion_count": count,
        "selected_direction": selected_direction,
        "admission_score": admission_score,
        "expansion_mask": expansion_mask,
        "pseudo_semantic_label": pseudo_labels,
        "selected_pseudo_anchor_ids": selected_ids,
        "selected_pseudo_anchor_labels": selected_labels,
        "real_anchor_ids": real_ids,
        "real_anchor_labels": real_labels,
        "combined_anchor_ids": combined_ids,
        "combined_anchor_labels": combined_labels,
        "combined_anchor_is_pseudo": is_pseudo,
    }


def build_relation_targets(query_directional_labels, anchor_labels):
    """C3 operator: [Nu,20] equality with anchor labels -> [Nu,A,20]."""
    query = _readonly(query_directional_labels, np.int64)
    labels = _readonly(anchor_labels, np.int64)
    _require(
        query.shape == (NU, S)
        and labels.ndim == 1
        and labels.size >= L
        and np.all((query >= 0) & (query < K))
        and np.all((labels >= 0) & (labels < K)),
        "C5_B0_RELATION_TARGET_INPUT_FAIL_CLOSED",
    )
    target = np.ascontiguousarray(
        query[:, None, :] == labels[None, :, None], dtype=np.bool_
    )
    _require(
        target.shape == (NU, labels.size, S),
        "C5_B0_RELATION_TARGET_SHAPE_FAIL_CLOSED",
    )
    target.setflags(write=False)
    return target


def build_relation_balance_weights(predicted_relation):
    """Exact C3-A0 same/different balancing generalized only over anchor axis."""
    predicted = _readonly(predicted_relation, np.bool_)
    _require(
        predicted.ndim == 3
        and predicted.shape[0] == NU
        and predicted.shape[1] >= L
        and predicted.shape[2] == S,
        "C5_B0_RELATION_BALANCE_INPUT_FAIL_CLOSED",
    )
    weights = np.empty(predicted.shape, dtype=np.float64)
    for direction_id in range(S):
        same = predicted[:, :, direction_id]
        same_count = int(np.count_nonzero(same))
        different_count = int(same.size - same_count)
        _require(
            same_count > 0 and different_count > 0,
            "C5_B0_RELATION_BALANCE_GROUP_EMPTY_FAIL_CLOSED",
        )
        direction_weights = weights[:, :, direction_id]
        direction_weights[same] = 1.0 / (2.0 * same_count)
        direction_weights[~same] = 1.0 / (2.0 * different_count)
        _require(
            np.isclose(direction_weights[same].sum(), 0.5, rtol=0.0, atol=1e-15)
            and np.isclose(
                direction_weights[~same].sum(), 0.5, rtol=0.0, atol=1e-15
            ),
            "C5_B0_RELATION_BALANCE_NORMALIZATION_FAIL_CLOSED",
        )
    _require(
        np.isfinite(weights).all() and np.all(weights > 0.0),
        "C5_B0_RELATION_BALANCE_FINITE_FAIL_CLOSED",
    )
    weights.setflags(write=False)
    return weights


def relation_equivalence_gate(
    class_pred_true, real_anchor_labels,
    frozen_relation_target, frozen_balance_weight,
):
    """Require bitwise target and strict exact balance parity for 14 anchors."""
    rebuilt_target = build_relation_targets(class_pred_true, real_anchor_labels)
    rebuilt_balance = build_relation_balance_weights(rebuilt_target)
    frozen_operator_balance = c3a0.build_relation_balance_weights(rebuilt_target)
    expected_target = np.asarray(frozen_relation_target, dtype=np.bool_)
    expected_balance = np.asarray(frozen_balance_weight, dtype=np.float64)
    target_equal = np.array_equal(rebuilt_target, expected_target)
    balance_equal = np.array_equal(rebuilt_balance, expected_balance)
    _require(
        target_equal and balance_equal
        and np.array_equal(rebuilt_balance, frozen_operator_balance),
        "C5_B0_ORIGINAL_RELATION_EQUIVALENCE_FAIL_CLOSED",
    )
    return {
        "PredRelation_true_bitwise_equal": True,
        "relation_balance_weights_true_exact_equal": True,
        "frozen_C3_A0_balance_operator_equal": True,
        "relation_target_logical_sha256": logical_sha256(rebuilt_target),
        "relation_balance_weight_logical_sha256": logical_sha256(rebuilt_balance),
    }


def build_self_relation_mask(query_ids, anchor_ids, anchor_is_pseudo):
    """Return valid relation entries [Nu,A,20], excluding pseudo self-pairs."""
    query = _readonly(query_ids, np.int64)
    anchors = _readonly(anchor_ids, np.int64)
    pseudo = _readonly(anchor_is_pseudo, np.bool_)
    _require(
        query.shape == (NU,)
        and anchors.ndim == pseudo.ndim == 1
        and anchors.shape == pseudo.shape
        and np.unique(query).size == NU
        and np.unique(anchors).size == anchors.size,
        "C5_B0_SELF_RELATION_INPUT_FAIL_CLOSED",
    )
    self_pair = (query[:, None] == anchors[None, :]) & pseudo[None, :]
    valid = np.broadcast_to(~self_pair[:, :, None], (NU, anchors.size, S)).copy()
    _require(
        not np.any(valid & np.broadcast_to(self_pair[:, :, None], valid.shape)),
        "C5_B0_SELF_RELATION_EXCLUSION_FAIL_CLOSED",
    )
    valid.setflags(write=False)
    return valid, {
        "self_relation_pair_count_excluded": int(np.count_nonzero(self_pair)),
        "self_relation_entry_count_excluded": int(np.count_nonzero(~valid)),
        "self_relation_count_used": 0,
    }


def build_training_action(
    U_cycle, unlabeled_ids, class_pred_true, anchor_plan,
):
    """Create frozen [Nu,A,20] relation action with true directional utility."""
    utility = _readonly(U_cycle, np.float64)
    query_ids = _readonly(unlabeled_ids, np.int64)
    if utility.shape == (N, S):
        utility = _readonly(utility[query_ids], np.float64)
    _require(
        utility.shape == (NU, S)
        and np.isfinite(utility).all()
        and np.all(utility >= 0.0),
        "C5_B0_U_CYCLE_BOUNDARY_FAIL_CLOSED",
    )
    target = build_relation_targets(
        class_pred_true, anchor_plan["combined_anchor_labels"]
    )
    balance = build_relation_balance_weights(target)
    valid_mask, self_audit = build_self_relation_mask(
        query_ids,
        anchor_plan["combined_anchor_ids"],
        anchor_plan["combined_anchor_is_pseudo"],
    )
    return {
        "U_cycle": utility,
        "unlabeled_ids": query_ids,
        "anchor_ids": anchor_plan["combined_anchor_ids"],
        "anchor_labels": anchor_plan["combined_anchor_labels"],
        "anchor_is_pseudo": anchor_plan["combined_anchor_is_pseudo"],
        "PredRelation": target,
        "balance_weight": balance,
        "valid_relation_mask": valid_mask,
        "relation_target_axis_order": RELATION_TARGET_AXIS_ORDER,
        **self_audit,
    }


def rows_for_sample_ids(unlabeled_ids, query_sample_ids):
    source_ids = np.asarray(unlabeled_ids, dtype=np.int64)
    query_ids = np.asarray(query_sample_ids, dtype=np.int64)
    order = np.argsort(source_ids)
    sorted_ids = source_ids[order]
    positions = np.searchsorted(sorted_ids, query_ids)
    valid = positions < sorted_ids.size
    _require(
        query_ids.ndim == 1
        and query_ids.size > 0
        and np.all(valid)
        and np.array_equal(sorted_ids[positions], query_ids),
        "C5_B0_QUERY_SAMPLE_ID_FAIL_CLOSED",
    )
    return np.ascontiguousarray(order[positions], dtype=np.int64)


def action_batch_by_sample_ids(action, query_sample_ids):
    rows = rows_for_sample_ids(action["unlabeled_ids"], query_sample_ids)
    return {
        "query_sample_ids": _readonly(query_sample_ids, np.int64),
        "U_cycle": _readonly(action["U_cycle"][rows], np.float64),
        "PredRelation": _readonly(action["PredRelation"][rows], np.bool_),
        "balance_weight": _readonly(action["balance_weight"][rows], np.float64),
        "valid_relation_mask": _readonly(
            action["valid_relation_mask"][rows], np.bool_
        ),
        "source_rows": _readonly(rows, np.int64),
    }


def relation_probability(q_sample, q_anchor):
    """Exact C3 dot product [B,K] @ detach([A,K]).T -> [B,A]."""
    _require(
        q_sample.ndim == q_anchor.ndim == 2
        and q_sample.shape[1] == q_anchor.shape[1] == K
        and q_anchor.shape[0] >= L,
        "C5_B0_RELATION_POSTERIOR_SHAPE_FAIL_CLOSED",
    )
    return q_sample @ q_anchor.detach().transpose(0, 1)


def relation_semantic_loss(
    q_sample_views, q_anchor_views, PredRelation, U_cycle,
    balance_weight, valid_relation_mask,
):
    """Frozen C3 TRUE_U relation loss with only pseudo-self entries masked."""
    _require(
        len(q_sample_views) == len(q_anchor_views) == V,
        "C5_B0_RELATION_VIEW_COUNT_FAIL_CLOSED",
    )
    reference = q_sample_views[0]
    target = torch.as_tensor(
        PredRelation, device=reference.device, dtype=reference.dtype
    ).detach()
    utility = torch.as_tensor(
        U_cycle, device=reference.device, dtype=reference.dtype
    ).detach()
    balance = torch.as_tensor(
        balance_weight, device=reference.device, dtype=reference.dtype
    ).detach()
    valid = torch.as_tensor(
        valid_relation_mask, device=reference.device, dtype=reference.dtype
    ).detach()
    _require(
        target.ndim == balance.ndim == valid.ndim == 3
        and target.shape == balance.shape == valid.shape
        and utility.shape == (target.shape[0], S),
        "C5_B0_RELATION_ACTION_SHAPE_FAIL_CLOSED",
    )
    action_weight = (utility[:, None, :] * balance * valid).detach()
    denominator = action_weight.sum()
    epsilon = torch.finfo(reference.dtype).eps
    _require(
        bool(torch.isfinite(denominator).item())
        and bool((denominator > epsilon).item()),
        "C5_B0_RELATION_DENOMINATOR_FAIL_CLOSED",
    )
    view_losses = []
    for q_sample, q_anchor in zip(q_sample_views, q_anchor_views):
        bce = c3b0.relation_bce(relation_probability(q_sample, q_anchor), target)
        view_loss = torch.sum(action_weight * bce) / denominator
        _require(bool(torch.isfinite(view_loss).item()), "C5_B0_NONFINITE_LOSS")
        view_losses.append(view_loss)
    loss = torch.stack(view_losses).mean()
    invalid = valid == 0
    return loss, {
        "view_count": V,
        "view_losses": [float(value.detach().item()) for value in view_losses],
        "denominator": float(denominator.detach().item()),
        "six_view_arithmetic_mean": True,
        "anchor_posterior_detached": True,
        "U_cycle_detached": not utility.requires_grad,
        "PredRelation_detached": not target.requires_grad,
        "balance_weights_detached": not balance.requires_grad,
        "expansion_mask_detached": not valid.requires_grad,
        "self_relation_count_used": int(
            torch.count_nonzero(action_weight[invalid]).detach().item()
        ),
        "new_loss_created": False,
        "pseudo_anchor_loss_weight_used": False,
    }


def validate_pre_gt_array_keys(arrays):
    keys = tuple(arrays.keys())
    forbidden = ("gt", "truth", "metric", "acc", "nmi", "ari")
    _require(
        keys == PRE_GT_ARRAY_KEYS
        and not any(token in key.lower() for key in keys for token in forbidden),
        "C5_B0_PRE_GT_SCHEMA_FAIL_CLOSED",
    )
    return True


def summarize_multiseed_metrics(metrics_by_seed):
    _require(set(metrics_by_seed) == set(SEEDS), "C5_B0_SEED_SET_FAIL_CLOSED")
    for seed in SEEDS:
        _require(
            set(metrics_by_seed[seed]) == set(ARMS),
            "C5_B0_ARM_SET_FAIL_CLOSED",
        )
    deltas_base = [
        float(metrics_by_seed[seed]["TRUE_U_EXPAND"][PRIMARY_METRIC])
        - float(metrics_by_seed[seed]["BASE"][PRIMARY_METRIC])
        for seed in SEEDS
    ]
    deltas_permuted = [
        float(metrics_by_seed[seed]["TRUE_U_EXPAND"][PRIMARY_METRIC])
        - float(metrics_by_seed[seed]["PERMUTED_U_EXPAND"][PRIMARY_METRIC])
        for seed in SEEDS
    ]
    gates = OrderedDict((
        ("mean_ACC_TRUE_U_EXPAND_minus_BASE_gt_0", np.mean(deltas_base) > 0.0),
        ("ACC_TRUE_U_EXPAND_minus_BASE_positive_at_least_2_of_3",
         sum(value > 0.0 for value in deltas_base) >= 2),
        ("mean_ACC_TRUE_U_EXPAND_minus_PERMUTED_U_EXPAND_gt_0",
         np.mean(deltas_permuted) > 0.0),
        ("ACC_TRUE_U_EXPAND_minus_PERMUTED_U_EXPAND_positive_at_least_2_of_3",
         sum(value > 0.0 for value in deltas_permuted) >= 2),
    ))
    return {
        "stage": STAGE,
        "primary_metric": PRIMARY_METRIC,
        "delta_ACC_TRUE_U_EXPAND_minus_BASE_by_seed": dict(
            zip((str(seed) for seed in SEEDS), deltas_base)
        ),
        "delta_ACC_TRUE_U_EXPAND_minus_PERMUTED_U_EXPAND_by_seed": dict(
            zip((str(seed) for seed in SEEDS), deltas_permuted)
        ),
        "mean_delta_ACC_TRUE_U_EXPAND_minus_BASE": float(np.mean(deltas_base)),
        "mean_delta_ACC_TRUE_U_EXPAND_minus_PERMUTED_U_EXPAND": float(
            np.mean(deltas_permuted)
        ),
        "positive_seed_count_vs_BASE": int(sum(v > 0.0 for v in deltas_base)),
        "positive_seed_count_vs_PERMUTED_U_EXPAND": int(
            sum(v > 0.0 for v in deltas_permuted
        )),
        "gate_checks": gates,
        "C5_B0_PASS": bool(all(gates.values())),
        "secondary_metrics_not_used_in_gate": list(SECONDARY_METRICS),
    }
