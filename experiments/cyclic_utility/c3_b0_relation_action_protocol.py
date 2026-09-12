"""Frozen protocol primitives for the C3-B0 relation-action pilot.

Only the query posterior receives Phase-A gradients. Every semantic target,
utility factor, balance weight, and anchor posterior is a detached input.
"""

from collections import OrderedDict

import numpy as np
import torch


STAGE = "C3-B0"
SEEDS = (20, 30, 50)
ARMS = ("BASE", "TRUE_U", "TRUE_UNIFORM", "SHUFFLE_U")
ARM_DIRECTORY_NAMES = OrderedDict((
    ("BASE", "base"),
    ("TRUE_U", "true_u"),
    ("TRUE_UNIFORM", "true_uniform"),
    ("SHUFFLE_U", "shuffle_u"),
))
PRIMARY_ARM = "TRUE_U"
METRICS = ("ACC", "NMI", "ARI")
SAMPLE_NUM = 1400
CLASS_NUM = 7
VIEW_NUM = 6
LABEL_COUNT = 14
UNLABELED_COUNT = 1386
DIRECTION_COUNT = 20
BATCH_SIZE = 256
FORMAL_EPOCHS = 20
NATIVE_OBJECTIVE = "REC + 0.01 * CLU"
RELATION_OBJECTIVE = (
    "mean_v(sum(action_weight * BCE(dot(q_sample_v, "
    "detach(q_anchor_v)), PredRelation)) / sum(action_weight))"
)
PASS_DECISION = "C3_B0_RELATION_ACTION_PILOT_PASS"
FAIL_DECISION = "C3-B1 CLASS-LEVEL UTILITY-CONDITIONED ACTION PILOT"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_seed(seed):
    value = int(seed)
    if value not in SEEDS:
        raise ValueError("C3-B0 seed must be one of " + str(SEEDS))
    return value


def validate_arm(arm):
    if arm not in ARMS:
        raise ValueError("C3-B0 arm must be one of " + str(ARMS))
    return arm


def _readonly(value, dtype=None):
    array = np.array(np.asarray(value, dtype=dtype), copy=True, order="C")
    array.setflags(write=False)
    return array


def validate_action_arrays(arrays):
    """Validate and expose only the seven sealed C3-A0 action inputs."""
    required = (
        "U_cycle", "unlabeled_ids", "labeled_ids",
        "PredRelation_true", "PredRelation_shuffle",
        "relation_balance_weights_true",
        "relation_balance_weights_shuffle",
    )
    _require(all(name in arrays for name in required), "action field missing")
    unlabeled = _readonly(arrays["unlabeled_ids"], np.int64)
    labeled = _readonly(arrays["labeled_ids"], np.int64)
    _require(
        unlabeled.shape == (UNLABELED_COUNT,)
        and labeled.shape == (LABEL_COUNT,)
        and np.unique(unlabeled).size == UNLABELED_COUNT
        and np.unique(labeled).size == LABEL_COUNT
        and np.intersect1d(unlabeled, labeled).size == 0
        and np.array_equal(
            np.sort(np.concatenate((unlabeled, labeled))),
            np.arange(SAMPLE_NUM, dtype=np.int64),
        ),
        "C3-B0 sparse split boundary mismatch",
    )
    cycle_source = _readonly(arrays["U_cycle"], np.float64)
    if cycle_source.shape == (SAMPLE_NUM, DIRECTION_COUNT):
        cycle = _readonly(cycle_source[unlabeled], np.float64)
    else:
        _require(
            cycle_source.shape == (UNLABELED_COUNT, DIRECTION_COUNT),
            "C3-B0 U_cycle shape mismatch",
        )
        cycle = cycle_source
    expected = (UNLABELED_COUNT, LABEL_COUNT, DIRECTION_COUNT)
    true_relation = _readonly(arrays["PredRelation_true"], np.bool_)
    shuffled_relation = _readonly(arrays["PredRelation_shuffle"], np.bool_)
    true_balance = _readonly(
        arrays["relation_balance_weights_true"], np.float64
    )
    shuffled_balance = _readonly(
        arrays["relation_balance_weights_shuffle"], np.float64
    )
    _require(
        cycle.shape == (UNLABELED_COUNT, DIRECTION_COUNT)
        and true_relation.shape == shuffled_relation.shape == expected
        and true_balance.shape == shuffled_balance.shape == expected
        and np.isfinite(cycle).all()
        and np.isfinite(true_balance).all()
        and np.isfinite(shuffled_balance).all()
        and np.all(cycle >= 0.0)
        and np.all(true_balance > 0.0)
        and np.all(shuffled_balance > 0.0),
        "C3-B0 relation action boundary mismatch",
    )
    return {
        "U_cycle": cycle,
        "unlabeled_ids": unlabeled,
        "labeled_ids": labeled,
        "PredRelation_true": true_relation,
        "PredRelation_shuffle": shuffled_relation,
        "relation_balance_weights_true": true_balance,
        "relation_balance_weights_shuffle": shuffled_balance,
    }


def unlabeled_query_ids(batch_sample_ids, unlabeled_ids):
    """Filter a shuffled ID batch; labeled rows never become Phase-A queries."""
    batch_ids = np.asarray(batch_sample_ids, dtype=np.int64)
    allowed = np.asarray(unlabeled_ids, dtype=np.int64)
    _require(
        batch_ids.ndim == 1 and np.unique(batch_ids).size == batch_ids.size,
        "invalid sample-ID batch",
    )
    mask = np.isin(batch_ids, allowed, assume_unique=True)
    return np.ascontiguousarray(batch_ids[mask])


def _rows_for_sample_ids(unlabeled_ids, query_sample_ids):
    """Map canonical sample IDs to sealed action rows, never batch positions."""
    source_ids = np.asarray(unlabeled_ids, dtype=np.int64)
    query_ids = np.asarray(query_sample_ids, dtype=np.int64)
    order = np.argsort(source_ids)
    sorted_ids = source_ids[order]
    positions = np.searchsorted(sorted_ids, query_ids)
    valid = positions < sorted_ids.size
    _require(
        query_ids.ndim == 1 and query_ids.size > 0 and np.all(valid)
        and np.array_equal(sorted_ids[positions], query_ids),
        "query sample ID is absent from sealed unlabeled IDs",
    )
    return np.ascontiguousarray(order[positions], dtype=np.int64)


def action_batch_by_sample_ids(action_arrays, arm, query_sample_ids):
    """Select detached [B,S] and [B,14,S] arrays by canonical sample ID."""
    validate_arm(arm)
    _require(arm != "BASE", "BASE has no Phase-A action batch")
    arrays = validate_action_arrays(action_arrays)
    rows = _rows_for_sample_ids(arrays["unlabeled_ids"], query_sample_ids)
    suffix = "shuffle" if arm == "SHUFFLE_U" else "true"
    return {
        "query_sample_ids": _readonly(query_sample_ids, np.int64),
        "U_cycle": _readonly(arrays["U_cycle"][rows], np.float64),
        "PredRelation": _readonly(
            arrays["PredRelation_" + suffix][rows], np.bool_
        ),
        "balance_weight": _readonly(
            arrays["relation_balance_weights_" + suffix][rows], np.float64
        ),
        "source_rows": _readonly(rows, np.int64),
    }


def action_weight_for_arm(U_cycle, balance_weight, arm):
    """Construct only a loss weight; this value is not persisted as utility."""
    validate_arm(arm)
    _require(arm != "BASE", "BASE has no action weight")
    cycle = torch.as_tensor(U_cycle).detach()
    balance = torch.as_tensor(balance_weight).detach()
    _require(
        cycle.ndim == 2 and balance.ndim == 3
        and balance.shape[0] == cycle.shape[0]
        and balance.shape[2] == cycle.shape[1],
        "action-weight tensor shape mismatch",
    )
    factor = torch.ones_like(cycle) if arm == "TRUE_UNIFORM" else cycle
    return factor[:, None, :] * balance


def relation_probability(q_sample, q_anchor):
    """Return dot-product probability [B,14]; anchor side is stop-gradient."""
    _require(
        q_sample.ndim == q_anchor.ndim == 2
        and q_sample.shape[1] == q_anchor.shape[1] == CLASS_NUM
        and q_anchor.shape[0] == LABEL_COUNT,
        "relation posterior shape mismatch",
    )
    return q_sample @ q_anchor.detach().transpose(0, 1)


def relation_bce(relation_prob, target):
    """Elementwise BCE [B,14,S] with dtype epsilon only."""
    target_tensor = torch.as_tensor(
        target, device=relation_prob.device, dtype=relation_prob.dtype
    ).detach()
    _require(
        relation_prob.ndim == 2 and target_tensor.ndim == 3
        and target_tensor.shape[:2] == relation_prob.shape,
        "relation BCE shape mismatch",
    )
    eps = torch.finfo(relation_prob.dtype).eps
    probability = relation_prob.clamp(min=eps, max=1.0 - eps)[:, :, None]
    return -target_tensor * torch.log(probability) - (
        1.0 - target_tensor
    ) * torch.log(1.0 - probability)


def relation_semantic_loss(
    q_sample_views, q_anchor_views, PredRelation, U_cycle,
    balance_weight, arm,
):
    """Compute six independently normalized view losses and their mean."""
    validate_arm(arm)
    _require(arm != "BASE", "BASE must not call relation semantic loss")
    _require(
        len(q_sample_views) == len(q_anchor_views) == VIEW_NUM,
        "relation action requires exactly six views",
    )
    reference = q_sample_views[0]
    target = torch.as_tensor(
        PredRelation, device=reference.device, dtype=reference.dtype
    ).detach()
    cycle = torch.as_tensor(
        U_cycle, device=reference.device, dtype=reference.dtype
    ).detach()
    balance = torch.as_tensor(
        balance_weight, device=reference.device, dtype=reference.dtype
    ).detach()
    action_weight = action_weight_for_arm(cycle, balance, arm).detach()
    epsilon = torch.finfo(reference.dtype).eps
    denominator = action_weight.sum()
    _require(
        bool(torch.isfinite(denominator).item())
        and bool((denominator > epsilon).item()),
        "relation action denominator is non-finite or <= numerical epsilon",
    )
    view_losses = []
    for q_sample, q_anchor in zip(q_sample_views, q_anchor_views):
        bce = relation_bce(relation_probability(q_sample, q_anchor), target)
        view_loss = torch.sum(action_weight * bce) / denominator
        _require(
            bool(torch.isfinite(view_loss).item()),
            "non-finite relation action loss",
        )
        view_losses.append(view_loss)
    loss = torch.stack(view_losses).mean()
    return loss, {
        "view_count": VIEW_NUM,
        "view_losses": [float(value.detach().item()) for value in view_losses],
        "denominator": float(denominator.detach().item()),
        "six_view_arithmetic_mean": True,
        "anchor_posterior_detached": True,
        "U_cycle_detached": not cycle.requires_grad,
        "PredRelation_detached": not target.requires_grad,
        "balance_weights_detached": not balance.requires_grad,
        "action_weight_name": "action_weight",
        "numerical_epsilon_only": True,
    }


def summarize_multiseed_metrics(metrics_by_seed):
    """Apply the preregistered eight-condition C3-B0 decision exactly."""
    _require(set(metrics_by_seed) == set(SEEDS), "seed set mismatch")
    for seed in SEEDS:
        _require(set(metrics_by_seed[seed]) == set(ARMS), "arm set mismatch")
        for arm in ARMS:
            _require(
                set(metrics_by_seed[seed][arm]) >= set(METRICS)
                and all(np.isfinite(metrics_by_seed[seed][arm][name])
                        for name in METRICS),
                "metric record mismatch",
            )
    comparisons = OrderedDict()
    for comparator in ("BASE", "TRUE_UNIFORM", "SHUFFLE_U"):
        deltas = {
            metric: [
                float(metrics_by_seed[seed][PRIMARY_ARM][metric])
                - float(metrics_by_seed[seed][comparator][metric])
                for seed in SEEDS
            ]
            for metric in METRICS
        }
        comparisons[comparator] = {
            "delta_by_seed": {
                metric: dict(zip((str(seed) for seed in SEEDS), values))
                for metric, values in deltas.items()
            },
            "mean_delta": {
                metric: float(np.mean(values))
                for metric, values in deltas.items()
            },
            "positive_ACC_seed_count": int(
                sum(value > 0.0 for value in deltas["ACC"])
            ),
        }
    checks = OrderedDict((
        ("mean_DeltaACC_TRUE_U_vs_BASE_gt_0",
         comparisons["BASE"]["mean_delta"]["ACC"] > 0.0),
        ("mean_DeltaACC_TRUE_U_vs_TRUE_UNIFORM_gt_0",
         comparisons["TRUE_UNIFORM"]["mean_delta"]["ACC"] > 0.0),
        ("mean_DeltaACC_TRUE_U_vs_SHUFFLE_U_gt_0",
         comparisons["SHUFFLE_U"]["mean_delta"]["ACC"] > 0.0),
        ("TRUE_U_vs_BASE_positive_ACC_at_least_2_of_3",
         comparisons["BASE"]["positive_ACC_seed_count"] >= 2),
        ("TRUE_U_vs_TRUE_UNIFORM_positive_ACC_at_least_2_of_3",
         comparisons["TRUE_UNIFORM"]["positive_ACC_seed_count"] >= 2),
        ("TRUE_U_vs_SHUFFLE_U_positive_ACC_at_least_2_of_3",
         comparisons["SHUFFLE_U"]["positive_ACC_seed_count"] >= 2),
        ("mean_DeltaNMI_TRUE_U_vs_BASE_gt_0",
         comparisons["BASE"]["mean_delta"]["NMI"] > 0.0),
        ("mean_DeltaARI_TRUE_U_vs_BASE_gt_0",
         comparisons["BASE"]["mean_delta"]["ARI"] > 0.0),
    ))
    passed = all(checks.values())
    return {
        "stage": STAGE,
        "primary_arm": PRIMARY_ARM,
        "comparisons": dict(comparisons),
        "gate_checks": dict(checks),
        "C3_B0_RELATION_ACTION_PILOT_PASS": passed,
        "decision": PASS_DECISION if passed else FAIL_DECISION,
        "next_stage_if_fail": FAIL_DECISION,
        "memory_eligible": passed,
        "memory_automatically_enabled": False,
    }
