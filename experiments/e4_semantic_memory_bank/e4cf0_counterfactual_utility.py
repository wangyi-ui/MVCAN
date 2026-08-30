"""Counterfactual marginal memory-action utility primitives for E4-CF0."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

from experiments.e4_semantic_memory_bank.semantic_memory_bank import (
    CLASS_NUM,
    LABELED_NUM,
    VIEW_NUM,
    build_classwise_matched_shuffle,
    validate_sparse_labels,
)


WRITER_NUM = LABELED_NUM * VIEW_NUM
MATCHED_PAIR_NUM = CLASS_NUM * VIEW_NUM
ZERO_TOLERANCE = 1e-12
DELTA_NONDEGENERACY_TOLERANCE = 1e-6
PROXY_NAMES = ("R", "S", "U_RS")
MIXED_PROXY_ACTION_VALUE_EVIDENCE_EXPLANATION = (
    "Counterfactual memory-writer action values are identifiable, "
    "but the preregistered handcrafted proxy conclusions are mixed: "
    "at least one proxy exhibits non-null action-value association, "
    "while the primary RS proxy does not satisfy the full "
    "E4CF0_PROXY_HAS_ACTION_VALUE gate."
)


def _frozen_tensor(value, *, dtype=None, device=None):
    if torch.is_tensor(value):
        tensor = value.detach()
        return tensor.to(
            dtype=tensor.dtype if dtype is None else dtype,
            device=tensor.device if device is None else device,
        ).detach()
    array = np.array(value, copy=True, order="C")
    return torch.as_tensor(array, dtype=dtype, device=device).detach()


@torch.no_grad()
def build_counterfactual_marginal_utility(
    h_labeled,
    labels_labeled,
):
    """Measure each writer view's leave-one-view-out held-out margin change."""
    h_values = _frozen_tensor(h_labeled)
    if not h_values.is_floating_point():
        raise ValueError("h_labeled must be floating point")
    if h_values.shape != (LABELED_NUM, VIEW_NUM, CLASS_NUM):
        raise ValueError("h_labeled must have shape [14,6,7]")
    if not bool(torch.isfinite(h_values).all().item()):
        raise ValueError("h_labeled must be finite")
    targets, _ = validate_sparse_labels(labels_labeled)
    targets = targets.to(device=h_values.device)

    heldout_peer_rows = torch.empty(
        LABELED_NUM, dtype=torch.long, device=h_values.device
    )
    peer_count = torch.empty(
        LABELED_NUM, dtype=torch.long, device=h_values.device
    )
    all_rows = torch.arange(LABELED_NUM, device=h_values.device)
    for writer_row in range(LABELED_NUM):
        peers = torch.nonzero(
            (targets == targets[writer_row])
            & (all_rows != writer_row),
            as_tuple=False,
        ).flatten()
        peer_count[writer_row] = peers.numel()
        if peers.shape != (1,):
            raise RuntimeError(
                "every writer must have exactly one held-out same-class peer"
            )
        heldout_peer_rows[writer_row] = peers[0]

    heldout_query = F.normalize(
        h_values[heldout_peer_rows].mean(dim=1), dim=-1
    ).detach()
    if heldout_query.shape != (LABELED_NUM, CLASS_NUM):
        raise RuntimeError("heldout_query must have shape [14,7]")
    if not bool(torch.isfinite(heldout_query).all().item()):
        raise RuntimeError("heldout_query contains non-finite values")

    class_all_view_prototypes = []
    for class_id in range(CLASS_NUM):
        class_sum = h_values[targets == class_id].sum(dim=(0, 1))
        if not bool((torch.linalg.vector_norm(class_sum) > 0.0).item()):
            raise RuntimeError("a negative-class prototype has zero norm")
        class_all_view_prototypes.append(F.normalize(class_sum, dim=0))
    class_all_view_prototypes = torch.stack(
        class_all_view_prototypes, dim=0
    ).detach()

    writer_all_view_sum = h_values.sum(dim=1)
    true_prototype_plus = F.normalize(
        writer_all_view_sum, dim=-1
    ).detach()
    true_prototype_minus = F.normalize(
        writer_all_view_sum[:, None, :] - h_values,
        dim=-1,
    ).detach()
    if true_prototype_plus.shape != (LABELED_NUM, CLASS_NUM):
        raise RuntimeError("plus true prototype must have shape [14,7]")
    if true_prototype_minus.shape != (
        LABELED_NUM,
        VIEW_NUM,
        CLASS_NUM,
    ):
        raise RuntimeError("minus true prototype must have shape [14,6,7]")

    negative_class_mask = (
        torch.arange(CLASS_NUM, device=h_values.device)[None, :]
        != targets[:, None]
    )
    negative_similarity = heldout_query @ class_all_view_prototypes.T
    hard_negative_similarity = negative_similarity.masked_fill(
        ~negative_class_mask, -torch.inf
    ).max(dim=1).values.detach()
    Q_plus_per_sample = (
        torch.sum(heldout_query * true_prototype_plus, dim=1)
        - hard_negative_similarity
    )
    Q_plus = Q_plus_per_sample[:, None].expand(
        LABELED_NUM, VIEW_NUM
    ).clone().detach()
    Q_minus = (
        torch.einsum(
            "ik,ivk->iv", heldout_query, true_prototype_minus
        )
        - hard_negative_similarity[:, None]
    ).detach()
    Delta_mem = (Q_plus - Q_minus).detach()
    if not (
        Q_plus.shape
        == Q_minus.shape
        == Delta_mem.shape
        == (LABELED_NUM, VIEW_NUM)
    ):
        raise RuntimeError("Q_plus/Q_minus/Delta_mem must have shape [14,6]")
    if not bool(
        torch.isfinite(Q_plus).all().item()
        and torch.isfinite(Q_minus).all().item()
        and torch.isfinite(Delta_mem).all().item()
    ):
        raise RuntimeError("counterfactual action values are non-finite")
    if Delta_mem.requires_grad or Delta_mem.grad_fn is not None:
        raise RuntimeError("Delta_mem did not stop gradients")

    plus_included_view_mask = torch.ones(
        LABELED_NUM,
        VIEW_NUM,
        VIEW_NUM,
        dtype=torch.bool,
        device=h_values.device,
    )
    minus_included_view_mask = plus_included_view_mask.clone()
    writer_views = torch.arange(VIEW_NUM, device=h_values.device)
    minus_included_view_mask[:, writer_views, writer_views] = False
    plus_view_count = plus_included_view_mask.sum(dim=2)
    minus_view_count = minus_included_view_mask.sum(dim=2)
    writer_rows = torch.arange(
        LABELED_NUM, device=h_values.device
    )[:, None].expand(LABELED_NUM, VIEW_NUM)
    writer_view_grid = torch.arange(
        VIEW_NUM, device=h_values.device
    )[None, :].expand(LABELED_NUM, VIEW_NUM)
    removed_view_exact = bool(
        torch.all(
            ~minus_included_view_mask[
                writer_rows,
                writer_view_grid,
                writer_view_grid,
            ]
        ).item()
        and torch.all(
            minus_included_view_mask.sum(dim=2) == VIEW_NUM - 1
        ).item()
    )

    evaluator_differs = bool(
        torch.all(heldout_peer_rows != all_rows).item()
    )
    evaluator_same_class = bool(
        torch.equal(targets[heldout_peer_rows], targets)
    )
    heldout_excluded = bool(
        evaluator_differs
        and torch.all(plus_view_count == VIEW_NUM).item()
        and torch.all(minus_view_count == VIEW_NUM - 1).item()
    )
    if not (
        evaluator_differs
        and evaluator_same_class
        and heldout_excluded
        and removed_view_exact
    ):
        raise RuntimeError("counterfactual held-out isolation failed")

    return {
        "Delta_mem": Delta_mem.detach(),
        "Q_plus": Q_plus.detach(),
        "Q_minus": Q_minus.detach(),
        "heldout_peer_rows": heldout_peer_rows.detach(),
        "heldout_query": heldout_query.detach(),
        "true_prototype_plus": true_prototype_plus.detach(),
        "true_prototype_minus": true_prototype_minus.detach(),
        "class_all_view_prototypes": (
            class_all_view_prototypes.detach()
        ),
        "hard_negative_similarity": hard_negative_similarity.detach(),
        "plus_included_view_mask": plus_included_view_mask.detach(),
        "minus_included_view_mask": minus_included_view_mask.detach(),
        "audit": {
            "writer_count": WRITER_NUM,
            "Delta_mem_shape": list(Delta_mem.shape),
            "Q_plus_shape": list(Q_plus.shape),
            "Q_minus_shape": list(Q_minus.shape),
            "heldout_query_shape": list(heldout_query.shape),
            "heldout_peer_rows": heldout_peer_rows.cpu().tolist(),
            "heldout_peer_count": peer_count.cpu().tolist(),
            "heldout_peer_differs_from_writer_pass": evaluator_differs,
            "heldout_peer_same_class_pass": evaluator_same_class,
            "heldout_sample_excluded_from_true_class_memory": (
                heldout_excluded
            ),
            "plus_writer_source_view_count": (
                plus_view_count.cpu().tolist()
            ),
            "plus_has_exactly_6_writer_source_views_pass": bool(
                torch.all(plus_view_count == VIEW_NUM).item()
            ),
            "minus_writer_source_view_count": (
                minus_view_count.cpu().tolist()
            ),
            "minus_has_exactly_5_writer_source_views_pass": bool(
                torch.all(minus_view_count == VIEW_NUM - 1).item()
            ),
            "removed_view_is_exactly_writer_view_pass": (
                removed_view_exact
            ),
            "negative_prototypes_shared_plus_minus_pass": True,
            "Delta_mem_finite_pass": True,
            "Delta_mem_stop_gradient_pass": True,
            "unlabeled_GT_used_for_delta": False,
            "oracle_used_for_delta": False,
            "corruption_mask_used_for_delta": False,
            "R_used_for_delta": False,
            "S_used_for_delta": False,
            "U_used_for_delta": False,
        },
    }


def summarize_delta(
    Delta_mem,
    zero_tolerance=ZERO_TOLERANCE,
):
    """Summarize all 84 action values with the frozen numerical-zero rule."""
    tolerance = float(zero_tolerance)
    if tolerance != ZERO_TOLERANCE:
        raise ValueError("CF0 zero tolerance is frozen at 1e-12")
    values = np.asarray(
        _frozen_tensor(Delta_mem).cpu().numpy(), dtype=np.float64
    )
    if values.shape != (LABELED_NUM, VIEW_NUM):
        raise ValueError("Delta_mem must have shape [14,6]")
    if not np.isfinite(values).all():
        raise ValueError("Delta_mem must be finite")
    flattened = values.reshape(-1)
    positive = flattened > tolerance
    negative = flattened < -tolerance
    zero = np.abs(flattened) <= tolerance
    if not np.all(positive | negative | zero):
        raise RuntimeError("Delta sign partition failed")
    return {
        "writer_count": int(flattened.size),
        "delta_mean": float(np.mean(flattened)),
        "delta_std": float(np.std(flattened)),
        "delta_min": float(np.min(flattened)),
        "delta_max": float(np.max(flattened)),
        "delta_positive_fraction": float(np.mean(positive)),
        "delta_negative_fraction": float(np.mean(negative)),
        "delta_zero_fraction": float(np.mean(zero)),
        "delta_abs_mean": float(np.mean(np.abs(flattened))),
        "delta_p25": float(np.percentile(flattened, 25)),
        "delta_p50": float(np.percentile(flattened, 50)),
        "delta_p75": float(np.percentile(flattened, 75)),
        "zero_numerical_tolerance": tolerance,
    }


def _correlation(left, right):
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size != WRITER_NUM:
        raise ValueError("correlation inputs must contain 84 aligned writers")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("correlation inputs must be finite")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(
        np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    )
    if denominator == 0.0:
        return None
    return float(np.sum(x_centered * y_centered) / denominator)


def proxy_correlations(proxy_score, Delta_mem):
    """Return Pearson and Spearman correlations for one aligned proxy."""
    proxy = np.asarray(
        _frozen_tensor(proxy_score).cpu().numpy(), dtype=np.float64
    )
    delta = np.asarray(
        _frozen_tensor(Delta_mem).cpu().numpy(), dtype=np.float64
    )
    if proxy.shape != (LABELED_NUM, VIEW_NUM):
        raise ValueError("proxy score must have shape [14,6]")
    if delta.shape != (LABELED_NUM, VIEW_NUM):
        raise ValueError("Delta_mem must have shape [14,6]")
    pearson = _correlation(proxy, delta)
    spearman = _correlation(
        rankdata(proxy.reshape(-1), method="average"),
        rankdata(delta.reshape(-1), method="average"),
    )
    return {
        "Pearson_correlation": pearson,
        "Spearman_correlation": spearman,
    }


def matched_pair_ranking(
    proxy_score,
    Delta_mem,
    labels_labeled,
):
    """Evaluate 42 within-class/view pair orderings, ignoring exact ties."""
    proxy = np.asarray(
        _frozen_tensor(proxy_score).cpu().numpy(), dtype=np.float64
    )
    delta = np.asarray(
        _frozen_tensor(Delta_mem).cpu().numpy(), dtype=np.float64
    )
    targets, _ = validate_sparse_labels(labels_labeled)
    targets = targets.cpu().numpy()
    if proxy.shape != (LABELED_NUM, VIEW_NUM):
        raise ValueError("proxy score must have shape [14,6]")
    if delta.shape != (LABELED_NUM, VIEW_NUM):
        raise ValueError("Delta_mem must have shape [14,6]")

    correct_count = 0
    valid_pair_count = 0
    tie_pair_count = 0
    records = []
    for class_id in range(CLASS_NUM):
        class_rows = np.flatnonzero(targets == class_id)
        if class_rows.shape != (2,):
            raise RuntimeError("matched ranking requires two rows/class")
        first, second = int(class_rows[0]), int(class_rows[1])
        for view_id in range(VIEW_NUM):
            proxy_difference = float(
                proxy[first, view_id] - proxy[second, view_id]
            )
            delta_difference = float(
                delta[first, view_id] - delta[second, view_id]
            )
            proxy_sign = int(np.sign(proxy_difference))
            delta_sign = int(np.sign(delta_difference))
            exact_tie = proxy_sign == 0 or delta_sign == 0
            if exact_tie:
                tie_pair_count += 1
                correct = None
            else:
                valid_pair_count += 1
                correct = proxy_sign == delta_sign
                correct_count += int(correct)
            records.append({
                "class_id": class_id,
                "view_id": view_id,
                "writer_rows": [first, second],
                "proxy_difference": proxy_difference,
                "delta_difference": delta_difference,
                "proxy_sign": proxy_sign,
                "delta_sign": delta_sign,
                "exact_tie": exact_tie,
                "correct": correct,
            })
    if valid_pair_count + tie_pair_count != MATCHED_PAIR_NUM:
        raise RuntimeError("matched pair accounting must total 42")
    accuracy = (
        None
        if valid_pair_count == 0
        else float(correct_count / valid_pair_count)
    )
    return {
        "total_pair_count": MATCHED_PAIR_NUM,
        "valid_pair_count": valid_pair_count,
        "tie_pair_count": tie_pair_count,
        "correct_pair_count": correct_count,
        "pairwise_rank_accuracy": accuracy,
        "pairs": records,
    }


@torch.no_grad()
def build_matched_proxy_shuffles(proxy_scores, labels_labeled):
    """Matched-swap R, S, and U_RS independently without random sampling."""
    if tuple(proxy_scores) != PROXY_NAMES:
        raise ValueError("proxy score set/order must be R, S, U_RS")
    shuffled = {}
    audits = {}
    for proxy_name in PROXY_NAMES:
        shuffled_score, shuffle_audit = (
            build_classwise_matched_shuffle(
                proxy_scores[proxy_name], labels_labeled
            )
        )
        shuffled[proxy_name] = shuffled_score.detach()
        audits[proxy_name] = shuffle_audit
    return shuffled, {
        "matched_pair_count": MATCHED_PAIR_NUM,
        "per_proxy": audits,
        "deterministic_no_RNG_pass": True,
    }


def compare_proxies_to_delta(
    proxy_scores,
    Delta_mem,
    labels_labeled,
):
    """Compare R/S/U_RS and their matched swaps with true action utility."""
    if tuple(proxy_scores) != PROXY_NAMES:
        raise ValueError("proxy score set/order must be R, S, U_RS")
    shuffled_scores, shuffle_audit = build_matched_proxy_shuffles(
        proxy_scores, labels_labeled
    )
    comparisons = {}
    for proxy_name in PROXY_NAMES:
        direct_ranking = matched_pair_ranking(
            proxy_scores[proxy_name], Delta_mem, labels_labeled
        )
        shuffled_ranking = matched_pair_ranking(
            shuffled_scores[proxy_name], Delta_mem, labels_labeled
        )
        comparisons[proxy_name] = {
            **proxy_correlations(proxy_scores[proxy_name], Delta_mem),
            **direct_ranking,
        }
        comparisons["SHUFFLED_" + proxy_name] = {
            **proxy_correlations(shuffled_scores[proxy_name], Delta_mem),
            **shuffled_ranking,
        }
    return {
        "comparisons": comparisons,
        "R_pairwise_rank_accuracy": comparisons["R"][
            "pairwise_rank_accuracy"
        ],
        "S_pairwise_rank_accuracy": comparisons["S"][
            "pairwise_rank_accuracy"
        ],
        "U_pairwise_rank_accuracy": comparisons["U_RS"][
            "pairwise_rank_accuracy"
        ],
        "matched_shuffled_U_pairwise_rank_accuracy": comparisons[
            "SHUFFLED_U_RS"
        ]["pairwise_rank_accuracy"],
        "matched_shuffles": shuffle_audit,
        "shuffled_scores": shuffled_scores,
    }


def _greater(value, reference):
    return value is not None and bool(value > reference)


def build_e4cf0_decision(delta_summary, proxy_comparison):
    """Apply the exhaustive CF0 decision tree without changing A/B/C."""
    delta_std = float(delta_summary["delta_std"])
    comparisons = proxy_comparison["comparisons"]
    delta_identifiable = bool(
        delta_std > DELTA_NONDEGENERACY_TOLERANCE
    )

    proxy_component_metrics = {}
    proxy_positive = {}
    for proxy_name in PROXY_NAMES:
        direct = comparisons[proxy_name]
        shuffled = comparisons["SHUFFLED_" + proxy_name]
        proxy_component_metrics[proxy_name] = {
            "pearson": direct["Pearson_correlation"],
            "spearman": direct["Spearman_correlation"],
            "pairwise_rank_accuracy": direct[
                "pairwise_rank_accuracy"
            ],
            "shuffled_pearson": shuffled["Pearson_correlation"],
            "shuffled_spearman": shuffled["Spearman_correlation"],
            "shuffled_pairwise_rank_accuracy": shuffled[
                "pairwise_rank_accuracy"
            ],
        }
        proxy_positive[proxy_name] = bool(
            _greater(
                comparisons[proxy_name]["pairwise_rank_accuracy"],
                0.5,
            )
            and _greater(
                comparisons[proxy_name]["Spearman_correlation"],
                0.0,
            )
        )

    U_accuracy = comparisons["U_RS"]["pairwise_rank_accuracy"]
    shuffled_U_accuracy = comparisons["SHUFFLED_U_RS"][
        "pairwise_rank_accuracy"
    ]
    U_spearman = comparisons["U_RS"]["Spearman_correlation"]
    U_pairwise_gt_half = _greater(U_accuracy, 0.5)
    U_pairwise_gt_shuffle = bool(
        U_accuracy is not None
        and shuffled_U_accuracy is not None
        and U_accuracy > shuffled_U_accuracy
    )
    U_spearman_positive = _greater(U_spearman, 0.0)

    if delta_std <= DELTA_NONDEGENERACY_TOLERANCE:
        condition_A = False
        condition_B = False
        condition_C = True
    else:
        condition_A = bool(
            _greater(U_accuracy, 0.5)
            and U_accuracy is not None
            and shuffled_U_accuracy is not None
            and U_accuracy > shuffled_U_accuracy
            and _greater(U_spearman, 0.0)
        )
        condition_B = bool(not any(proxy_positive.values()))
        condition_C = False
    condition_D = bool(
        delta_std > DELTA_NONDEGENERACY_TOLERANCE
        and not condition_A
        and not condition_B
        and not condition_C
    )

    decision_conditions = {
        "Decision_A": condition_A,
        "Decision_B": condition_B,
        "Decision_C": condition_C,
        "Decision_D": condition_D,
    }
    if sum(decision_conditions.values()) != 1:
        raise RuntimeError("E4CF0_DECISION_TREE_LOGICALLY_IMPOSSIBLE_STATE")
    if condition_A:
        decision = "E4CF0_PROXY_HAS_ACTION_VALUE"
    elif condition_B:
        decision = "E4CF0_HANDCRAFTED_PROXY_FAIL"
    elif condition_C:
        decision = "E4CF0_MEMORY_WRITER_ACTION_UNIDENTIFIABLE"
    else:
        decision = "E4CF0_MIXED_PROXY_ACTION_VALUE_EVIDENCE"

    return {
        "delta_nondegeneracy_tolerance": (
            DELTA_NONDEGENERACY_TOLERANCE
        ),
        "delta_identifiable": delta_identifiable,
        "proxy_component_metrics": proxy_component_metrics,
        "U_pairwise_gt_half": U_pairwise_gt_half,
        "U_pairwise_gt_shuffle": U_pairwise_gt_shuffle,
        "U_spearman_positive": U_spearman_positive,
        "R_proxy_joint_positive": proxy_positive["R"],
        "S_proxy_joint_positive": proxy_positive["S"],
        "U_proxy_joint_positive": proxy_positive["U_RS"],
        "any_proxy_joint_positive": bool(
            proxy_positive["R"]
            or proxy_positive["S"]
            or proxy_positive["U_RS"]
        ),
        "Decision_A_condition": condition_A,
        "Decision_B_condition": condition_B,
        "Decision_C_condition": condition_C,
        "Decision_D_condition": condition_D,
        "decision_conditions": decision_conditions,
        "final_decision": decision,
        "decision_explanation": (
            MIXED_PROXY_ACTION_VALUE_EVIDENCE_EXPLANATION
            if condition_D
            else None
        ),
        "Decision_D_guardrails": {
            "is_PASS": False,
            "allows_online_memory_training": False,
            "interpreted_as_RS_PASS": False,
            "interpreted_as_memory_FAIL": False,
            "automatically_enters_estimator_training": False,
        },
        "corruption_posthoc_entered_gate": False,
    }
