"""Pure-array protocol for C2-B0 sparse utility residual calibration.

The fourteen sparse labels estimate only ``z_L - U_cycle``.  ``R`` defines
sample similarity, while frozen ``C_conf`` is deliberately absent from every
pre-GT residual-construction API.  This module performs no artifact I/O,
ground-truth loading, model execution, or training.
"""

from collections import OrderedDict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from experiments.cyclic_utility import bridge_p0_residual_utility_protocol as bridge
from weak_quality import ndarray_sha256


STAGE = "C2-B0"
SEEDS = (20, 30, 50)
SAMPLE_NUM = 1400
CLASS_NUM = 7
DIRECTION_COUNT = 20
LABEL_COUNT = 14
UNLABELED_EVAL_COUNT = 1386
CONFIDENCE_BIN_COUNT = 5
MIN_VALID_BINS_PER_DIRECTION = 3
MIN_VALID_DIRECTIONS_PER_SEED = 16
POSITIVE_DIRECTION_MIN_COUNT = 12
SEED_PASS_MIN_COUNT = 2
ALPHA = 0.05
ZERO_SUPPORT_EPSILON = 1e-12

FIXED_LABELED_IDS = (
    67, 82, 90, 111, 200, 365, 440, 513, 536, 983, 1027, 1250, 1316, 1385,
)
FIXED_LABELED_TARGETS = (4, 6, 2, 3, 0, 4, 1, 5, 6, 3, 0, 2, 5, 1)
FIXED_SHUFFLED_TARGETS = (6, 2, 3, 0, 4, 1, 5, 6, 3, 0, 2, 5, 1, 4)

FINAL_DECISIONS = (
    "C2_B0_SPARSE_UTILITY_RESIDUAL_CALIBRATION_PASS",
    "C2_B0_SPARSE_UTILITY_RESIDUAL_CALIBRATION_FAIL",
)

# Exact frozen Bridge primitives.  Confidence is used only by the post-seal
# calibration diagnostic below, never by residual propagation.
confidence_equal_count_strata = bridge.confidence_equal_count_strata
binary_auc_or_none = bridge.binary_auc_or_none
one_sided_wilcoxon_greater = bridge.one_sided_wilcoxon_greater


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _readonly(value, dtype=None):
    array = np.array(np.asarray(value, dtype=dtype), copy=True, order="C")
    array.setflags(write=False)
    return array


def validate_seed(seed):
    value = int(seed)
    if value not in SEEDS:
        raise ValueError("C2-B0 seed must be one of " + str(SEEDS))
    return value


def canonical_sample_ids():
    result = np.arange(SAMPLE_NUM, dtype=np.int64)
    result.setflags(write=False)
    return result


def fixed_labeled_ids():
    return _readonly(FIXED_LABELED_IDS, dtype=np.int64)


def fixed_labeled_targets():
    return _readonly(FIXED_LABELED_TARGETS, dtype=np.int64)


def fixed_shuffled_targets():
    return _readonly(FIXED_SHUFFLED_TARGETS, dtype=np.int64)


def validate_fixed_sparse_split(labeled_ids, unlabeled_ids, labeled_targets):
    labeled = _readonly(labeled_ids, dtype=np.int64)
    unlabeled = _readonly(unlabeled_ids, dtype=np.int64)
    targets = _readonly(labeled_targets, dtype=np.int64)
    expected_unlabeled = np.setdiff1d(canonical_sample_ids(), fixed_labeled_ids())
    _require(
        labeled.shape == targets.shape == (LABEL_COUNT,)
        and unlabeled.shape == (UNLABELED_EVAL_COUNT,)
        and np.array_equal(labeled, fixed_labeled_ids())
        and np.array_equal(targets, fixed_labeled_targets())
        and np.array_equal(unlabeled, expected_unlabeled)
        and np.array_equal(
            np.bincount(targets, minlength=CLASS_NUM),
            np.full(CLASS_NUM, 2, dtype=np.int64),
        ),
        "C2-B0 fixed sparse split mismatch",
    )
    return {
        "labeled_sample_ids": labeled,
        "labeled_targets": targets,
        "unlabeled_ids": unlabeled,
    }


def validate_negative_control(original_targets, shuffled_targets):
    original = _readonly(original_targets, dtype=np.int64)
    shuffled = _readonly(shuffled_targets, dtype=np.int64)
    expected = np.full(CLASS_NUM, 2, dtype=np.int64)
    _require(
        original.shape == shuffled.shape == (LABEL_COUNT,)
        and np.array_equal(original, fixed_labeled_targets())
        and np.array_equal(shuffled, fixed_shuffled_targets())
        and np.array_equal(np.bincount(original, minlength=CLASS_NUM), expected)
        and np.array_equal(np.bincount(shuffled, minlength=CLASS_NUM), expected)
        and ndarray_sha256(original) != ndarray_sha256(shuffled),
        "C2-B0 frozen negative-control targets mismatch",
    )
    return shuffled


def build_vote_semantic_state(y_gen):
    """Build R[i,k] = mean_s 1[y_gen[i,s] == k]."""
    actions = _readonly(y_gen, dtype=np.int64)
    # y_gen: [N, D]
    _require(
        actions.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and np.all((actions >= 0) & (actions < CLASS_NUM)),
        "C2-B0 y_gen boundary mismatch",
    )
    one_hot = np.eye(CLASS_NUM, dtype=np.float64)[actions]
    # one_hot: [N, D, K]; R: [N, K]
    state = np.ascontiguousarray(one_hot.mean(axis=1), dtype=np.float64)
    _require(
        state.shape == (SAMPLE_NUM, CLASS_NUM)
        and np.isfinite(state).all()
        and np.all(state >= 0.0)
        and np.allclose(state.sum(axis=1), 1.0, rtol=0.0, atol=1e-15),
        "C2-B0 vote semantic-state boundary mismatch",
    )
    state.setflags(write=False)
    return state


def build_leave_one_out_action_correctness(
    vote_state, y_gen, labeled_ids, labeled_targets
):
    """Construct all 14 LOO mappings and the corresponding sparse actions."""
    state = _readonly(vote_state, dtype=np.float64)
    actions = _readonly(y_gen, dtype=np.int64)
    labeled = _readonly(labeled_ids, dtype=np.int64)
    targets = _readonly(labeled_targets, dtype=np.int64)
    _require(
        state.shape == (SAMPLE_NUM, CLASS_NUM)
        and actions.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and labeled.shape == targets.shape == (LABEL_COUNT,)
        and np.unique(labeled).size == LABEL_COUNT
        and np.all((labeled >= 0) & (labeled < SAMPLE_NUM))
        and np.all((targets >= 0) & (targets < CLASS_NUM))
        and np.isfinite(state).all()
        and np.all(state >= 0.0)
        and np.allclose(state.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)
        and np.all((actions >= 0) & (actions < CLASS_NUM)),
        "C2-B0 LOO sparse-mapping input boundary mismatch",
    )

    # contingencies: [L, K, K]; mappings: [L, K]; z_L: [L, D]
    contingencies = np.zeros(
        (LABEL_COUNT, CLASS_NUM, CLASS_NUM), dtype=np.float64
    )
    mappings = np.empty((LABEL_COUNT, CLASS_NUM), dtype=np.int64)
    z_labeled = np.empty((LABEL_COUNT, DIRECTION_COUNT), dtype=np.float64)
    included_masks = np.ones((LABEL_COUNT, LABEL_COUNT), dtype=np.bool_)
    for held_out in range(LABEL_COUNT):
        included_masks[held_out, held_out] = False
        remaining_ids = labeled[included_masks[held_out]]
        remaining_targets = targets[included_masks[held_out]]
        contingency = contingencies[held_out]
        for native_id in range(CLASS_NUM):
            for semantic_class in range(CLASS_NUM):
                rows = remaining_ids[remaining_targets == semantic_class]
                contingency[native_id, semantic_class] = state[rows, native_id].sum()
        row_indices, column_indices = linear_sum_assignment(-contingency)
        mapping = mappings[held_out]
        mapping[row_indices] = column_indices
        _require(
            np.array_equal(np.sort(mapping), np.arange(CLASS_NUM, dtype=np.int64)),
            "C2-B0 LOO Hungarian result is not a class permutation",
        )
        z_labeled[held_out] = (
            mapping[actions[labeled[held_out]]] == targets[held_out]
        ).astype(np.float64)

    _require(
        np.all(included_masks.sum(axis=1) == LABEL_COUNT - 1)
        and not np.any(np.diag(included_masks))
        and np.all((z_labeled == 0.0) | (z_labeled == 1.0)),
        "C2-B0 leave-one-out exclusion/action boundary mismatch",
    )
    for value in (contingencies, mappings, z_labeled, included_masks):
        value.setflags(write=False)
    return {
        "loo_soft_contingency": contingencies,
        "loo_mappings": mappings,
        "loo_included_label_mask": included_masks,
        "z_L": z_labeled,
        "mapping_fit_label_count_each": LABEL_COUNT - 1,
        "held_out_sample_used_in_own_mapping": False,
    }


def build_labeled_residual(z_L, U_cycle, labeled_ids):
    correctness = _readonly(z_L, dtype=np.float64)
    cycle = _readonly(U_cycle, dtype=np.float64)
    labeled = _readonly(labeled_ids, dtype=np.int64)
    _require(
        correctness.shape == (LABEL_COUNT, DIRECTION_COUNT)
        and cycle.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and labeled.shape == (LABEL_COUNT,)
        and np.all((correctness == 0.0) | (correctness == 1.0))
        and np.isfinite(cycle).all()
        and np.all((cycle >= 0.0) & (cycle <= 1.0)),
        "C2-B0 labeled-residual input boundary mismatch",
    )
    # z_L: [L, D]; U_cycle[labeled]: [L, D]; e_L: [L, D]
    residual = np.ascontiguousarray(correctness - cycle[labeled], dtype=np.float64)
    _require(
        residual.shape == (LABEL_COUNT, DIRECTION_COUNT)
        and np.isfinite(residual).all()
        and np.all((residual >= -1.0) & (residual <= 1.0)),
        "C2-B0 labeled residual boundary mismatch",
    )
    residual.setflags(write=False)
    return residual


def propagate_action_context_residuals(vote_state, y_gen, labeled_ids, e_L):
    """Propagate e_L using only semantic cosine and same-direction action match."""
    state = _readonly(vote_state, dtype=np.float64)
    actions = _readonly(y_gen, dtype=np.int64)
    labeled = _readonly(labeled_ids, dtype=np.int64)
    residual = _readonly(e_L, dtype=np.float64)
    _require(
        state.shape == (SAMPLE_NUM, CLASS_NUM)
        and actions.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and labeled.shape == (LABEL_COUNT,)
        and residual.shape == (LABEL_COUNT, DIRECTION_COUNT)
        and np.isfinite(state).all()
        and np.isfinite(residual).all()
        and np.all((residual >= -1.0) & (residual <= 1.0)),
        "C2-B0 residual-propagation input boundary mismatch",
    )
    # R: [N, K]; R_labeled: [L, K]; semantic_sim: [N, L]
    labeled_state = state[labeled]
    denominator = (
        np.linalg.norm(state, axis=1)[:, None]
        * np.linalg.norm(labeled_state, axis=1)[None, :]
    )
    semantic_sim = np.divide(
        state @ labeled_state.T,
        denominator,
        out=np.zeros((SAMPLE_NUM, LABEL_COUNT), dtype=np.float64),
        where=denominator > 0.0,
    )
    semantic_sim = np.clip(semantic_sim, 0.0, 1.0)
    # action_match: [N, L, D]; weights: [N, L, D]
    action_match = actions[:, None, :] == actions[labeled][None, :, :]
    weights = semantic_sim[:, :, None] * action_match.astype(np.float64)
    support = weights.sum(axis=1)
    numerator = (weights * residual[None, :, :]).sum(axis=1)
    # e_hat: [N, D]; unsupported cells remain exactly zero.
    propagated = np.divide(
        numerator,
        support,
        out=np.zeros((SAMPLE_NUM, DIRECTION_COUNT), dtype=np.float64),
        where=support > ZERO_SUPPORT_EPSILON,
    )
    _require(
        semantic_sim.shape == (SAMPLE_NUM, LABEL_COUNT)
        and action_match.shape == weights.shape
        == (SAMPLE_NUM, LABEL_COUNT, DIRECTION_COUNT)
        and propagated.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and np.isfinite(propagated).all()
        and np.all((propagated >= -1.0) & (propagated <= 1.0))
        and np.all(propagated[support <= ZERO_SUPPORT_EPSILON] == 0.0),
        "C2-B0 propagated residual boundary mismatch",
    )
    for value in (semantic_sim, action_match, weights, support, propagated):
        value.setflags(write=False)
    return {
        "semantic_similarity": semantic_sim,
        "action_match": action_match,
        "weights": weights,
        "support": support,
        "e_hat": propagated,
        "C_conf_used": False,
        "correction_capacity_used_in_weights": False,
        "top_k_used": False,
        "threshold_used": False,
        "temperature_used": False,
    }


def bounded_utility_update(U_cycle, e_hat):
    cycle = _readonly(U_cycle, dtype=np.float64)
    residual = _readonly(e_hat, dtype=np.float64)
    _require(
        cycle.shape == residual.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and np.isfinite(cycle).all()
        and np.isfinite(residual).all()
        and np.all((cycle >= 0.0) & (cycle <= 1.0))
        and np.all((residual >= -1.0) & (residual <= 1.0)),
        "C2-B0 bounded-update input boundary mismatch",
    )
    calibrated = cycle + cycle * (1.0 - cycle) * residual
    calibrated[cycle == 0.0] = 0.0
    _require(
        np.isfinite(calibrated).all()
        and np.all((calibrated >= 0.0) & (calibrated <= 1.0))
        and np.all(calibrated[cycle == 0.0] == 0.0),
        "C2-B0 bounded utility update failed",
    )
    calibrated.setflags(write=False)
    return calibrated


def _build_one_residual_path(vote_state, y_gen, U_cycle, labeled_ids, targets):
    loo = build_leave_one_out_action_correctness(
        vote_state, y_gen, labeled_ids, targets
    )
    e_labeled = build_labeled_residual(loo["z_L"], U_cycle, labeled_ids)
    propagation = propagate_action_context_residuals(
        vote_state, y_gen, labeled_ids, e_labeled
    )
    calibrated = bounded_utility_update(U_cycle, propagation["e_hat"])
    return {
        **loo,
        "e_L": e_labeled,
        **propagation,
        "U_tilde": calibrated,
    }


def build_residual_calibration_outputs(
    y_gen, U_cycle, labeled_ids, labeled_targets, shuffled_targets
):
    """Build true and negative-control residual paths independently at source."""
    actions = _readonly(y_gen, dtype=np.int64)
    cycle = _readonly(U_cycle, dtype=np.float64)
    labeled = _readonly(labeled_ids, dtype=np.int64)
    targets = _readonly(labeled_targets, dtype=np.int64)
    shuffled = validate_negative_control(targets, shuffled_targets)
    state = build_vote_semantic_state(actions)
    true_path = _build_one_residual_path(state, actions, cycle, labeled, targets)
    # Recompute LOO mapping, z_L, e_L, and e_hat from the frozen shuffled targets.
    shuffle_path = _build_one_residual_path(
        state, actions, cycle, labeled, shuffled
    )
    return {
        "R": state,
        "z_L": true_path["z_L"],
        "e_L": true_path["e_L"],
        "e_hat": true_path["e_hat"],
        "U_tilde": true_path["U_tilde"],
        "loo_mappings": true_path["loo_mappings"],
        "loo_soft_contingency": true_path["loo_soft_contingency"],
        "loo_included_label_mask": true_path["loo_included_label_mask"],
        "semantic_similarity": true_path["semantic_similarity"],
        "action_match": true_path["action_match"],
        "weights": true_path["weights"],
        "support": true_path["support"],
        "z_L_shuffle": shuffle_path["z_L"],
        "e_L_shuffle": shuffle_path["e_L"],
        "e_hat_shuffle": shuffle_path["e_hat"],
        "U_tilde_shuffle": shuffle_path["U_tilde"],
        "loo_mappings_shuffle": shuffle_path["loo_mappings"],
        "loo_soft_contingency_shuffle": shuffle_path["loo_soft_contingency"],
        "shuffle_recomputed_from_targets": True,
        "shuffle_only_final_residual": False,
    }


def analyze_residual_directions(e_hat, e_hat_shuffle, e_GT, sample_ids):
    predicted = _readonly(e_hat, dtype=np.float64)
    shuffled = _readonly(e_hat_shuffle, dtype=np.float64)
    truth = _readonly(e_GT, dtype=np.float64)
    ids = _readonly(sample_ids, dtype=np.int64)
    expected = (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
    _require(
        predicted.shape == shuffled.shape == truth.shape == expected
        and ids.shape == (UNLABELED_EVAL_COUNT,)
        and np.unique(ids).size == UNLABELED_EVAL_COUNT
        and np.intersect1d(ids, fixed_labeled_ids()).size == 0,
        "C2-B0 residual identification must use exactly 1386 unlabeled samples",
    )
    directions = []
    for direction_id in range(DIRECTION_COUNT):
        arrays = (
            predicted[:, direction_id],
            shuffled[:, direction_id],
            truth[:, direction_id],
        )
        all_finite = all(np.isfinite(value).all() for value in arrays)
        all_nonconstant = all(np.unique(value).size >= 2 for value in arrays)
        valid = bool(all_finite and all_nonconstant)
        record = {
            "direction_id": direction_id,
            "joint_valid": valid,
            "all_arrays_finite": all_finite,
            "e_hat_has_two_distinct_values": bool(np.unique(arrays[0]).size >= 2),
            "e_hat_shuffle_has_two_distinct_values": bool(np.unique(arrays[1]).size >= 2),
            "e_GT_has_two_distinct_values": bool(np.unique(arrays[2]).size >= 2),
            "rho_true": None,
            "rho_shuffle": None,
            "ResidualSpecificityGap": None,
        }
        if valid:
            rho_true = float(spearmanr(arrays[0], arrays[2]).statistic)
            rho_shuffle = float(spearmanr(arrays[1], arrays[2]).statistic)
            _require(
                np.isfinite(rho_true) and np.isfinite(rho_shuffle),
                "C2-B0 valid Spearman direction produced non-finite output",
            )
            record.update({
                "rho_true": rho_true,
                "rho_shuffle": rho_shuffle,
                "ResidualSpecificityGap": rho_true - rho_shuffle,
            })
        directions.append(record)
    return {
        "direction_count": DIRECTION_COUNT,
        "evaluation_sample_count": UNLABELED_EVAL_COUNT,
        "evaluation_only_unlabeled": True,
        "invalid_spearman_not_replaced_with_zero": True,
        "directions": directions,
    }


def _weighted_mean(records, field):
    weights = np.asarray([record["sample_count"] for record in records], dtype=np.float64)
    values = np.asarray([record[field] for record in records], dtype=np.float64)
    _require(
        weights.size > 0 and np.all(weights > 0.0) and np.isfinite(values).all(),
        "C2-B0 weighted calibration metric boundary mismatch",
    )
    return float(np.average(values, weights=weights))


def analyze_calibration_direction(
    correct, U_cycle, U_tilde, U_tilde_shuffle, C_conf, sample_ids, direction_id
):
    outcomes = _readonly(correct, dtype=np.int64)
    cycle = _readonly(U_cycle, dtype=np.float64)
    calibrated = _readonly(U_tilde, dtype=np.float64)
    shuffled = _readonly(U_tilde_shuffle, dtype=np.float64)
    confidence = _readonly(C_conf, dtype=np.float64)
    ids = _readonly(sample_ids, dtype=np.int64)
    _require(
        outcomes.shape == cycle.shape == calibrated.shape == shuffled.shape
        == confidence.shape == ids.shape == (UNLABELED_EVAL_COUNT,)
        and np.all((outcomes == 0) | (outcomes == 1))
        and all(np.isfinite(value).all() for value in (cycle, calibrated, shuffled, confidence))
        and np.unique(ids).size == UNLABELED_EVAL_COUNT,
        "C2-B0 calibration direction input boundary mismatch",
    )
    assignments, strata_audit = confidence_equal_count_strata(confidence, ids)
    bins = []
    valid_mask = []
    for bin_id in range(CONFIDENCE_BIN_COUNT):
        indices = np.flatnonzero(assignments == bin_id)
        bin_outcomes = outcomes[indices]
        valid = bool(np.unique(bin_outcomes).size == 2)
        valid_mask.append(valid)
        record = {
            "bin_id": bin_id,
            "sample_count": int(indices.size),
            "valid_both_correctness_classes": valid,
            "AUC_Ucycle": None,
            "AUC_U_tilde": None,
            "AUC_U_tilde_shuffle": None,
        }
        if valid:
            record.update({
                "AUC_Ucycle": binary_auc_or_none(bin_outcomes, cycle[indices]),
                "AUC_U_tilde": binary_auc_or_none(bin_outcomes, calibrated[indices]),
                "AUC_U_tilde_shuffle": binary_auc_or_none(
                    bin_outcomes, shuffled[indices]
                ),
            })
        bins.append(record)
    valid_bins = [record for record in bins if record["valid_both_correctness_classes"]]
    direction_valid = len(valid_bins) >= MIN_VALID_BINS_PER_DIRECTION
    result = {
        "direction_id": int(direction_id),
        "valid_bin_count": len(valid_bins),
        "valid_bin_mask": valid_mask,
        "same_valid_bin_mask_all_utility_arms": True,
        "direction_valid": direction_valid,
        "bins": bins,
        "CondAUC_Ucycle": None,
        "CondAUC_U_tilde": None,
        "CondAUC_U_tilde_shuffle": None,
        "DeltaCalAUC": None,
        "TrueVsShuffleCalGap": None,
        "confidence_strata": strata_audit,
    }
    if direction_valid:
        baseline_auc = _weighted_mean(valid_bins, "AUC_Ucycle")
        calibrated_auc = _weighted_mean(valid_bins, "AUC_U_tilde")
        shuffled_auc = _weighted_mean(valid_bins, "AUC_U_tilde_shuffle")
        result.update({
            "CondAUC_Ucycle": baseline_auc,
            "CondAUC_U_tilde": calibrated_auc,
            "CondAUC_U_tilde_shuffle": shuffled_auc,
            "DeltaCalAUC": calibrated_auc - baseline_auc,
            "TrueVsShuffleCalGap": calibrated_auc - shuffled_auc,
        })
    return result


def analyze_calibration_directions(
    correct, U_cycle, U_tilde, U_tilde_shuffle, C_conf, sample_ids
):
    arrays = (
        _readonly(correct, dtype=np.int64),
        _readonly(U_cycle, dtype=np.float64),
        _readonly(U_tilde, dtype=np.float64),
        _readonly(U_tilde_shuffle, dtype=np.float64),
        _readonly(C_conf, dtype=np.float64),
    )
    ids = _readonly(sample_ids, dtype=np.int64)
    expected = (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
    _require(
        all(value.shape == expected for value in arrays)
        and ids.shape == (UNLABELED_EVAL_COUNT,)
        and np.array_equal(ids, np.setdiff1d(canonical_sample_ids(), fixed_labeled_ids())),
        "C2-B0 calibration evaluation must use fixed 1386 unlabeled samples",
    )
    directions = [
        analyze_calibration_direction(
            arrays[0][:, direction_id], arrays[1][:, direction_id],
            arrays[2][:, direction_id], arrays[3][:, direction_id],
            arrays[4][:, direction_id], ids, direction_id,
        )
        for direction_id in range(DIRECTION_COUNT)
    ]
    return {
        "direction_count": DIRECTION_COUNT,
        "evaluation_sample_count": UNLABELED_EVAL_COUNT,
        "evaluation_only_unlabeled": True,
        "GT_used_for_confidence_stratification": False,
        "directions": directions,
    }


def _mean(records, field):
    if not records:
        return None
    values = np.asarray([record[field] for record in records], dtype=np.float64)
    _require(np.isfinite(values).all(), "C2-B0 non-finite seed metric")
    return float(values.mean())


def build_seed_summary(seed, residual_metrics, calibration_metrics):
    active_seed = validate_seed(seed)
    residual_directions = residual_metrics.get("directions", ())
    calibration_directions = calibration_metrics.get("directions", ())
    _require(
        len(residual_directions) == len(calibration_directions) == DIRECTION_COUNT
        and all(
            residual_directions[index].get("direction_id")
            == calibration_directions[index].get("direction_id") == index
            for index in range(DIRECTION_COUNT)
        ),
        "C2-B0 seed direction schema mismatch",
    )
    valid_residual = [record for record in residual_directions if record["joint_valid"]]
    valid_calibration = [
        record for record in calibration_directions if record["direction_valid"]
    ]
    rho = np.asarray([record["rho_true"] for record in valid_residual], dtype=np.float64)
    deltas = np.asarray(
        [record["DeltaCalAUC"] for record in valid_calibration], dtype=np.float64
    )
    residual_p = one_sided_wilcoxon_greater(rho)
    calibration_p = one_sided_wilcoxon_greater(deltas)
    mean_rho = _mean(valid_residual, "rho_true")
    mean_residual_gap = _mean(valid_residual, "ResidualSpecificityGap")
    mean_delta = _mean(valid_calibration, "DeltaCalAUC")
    mean_calibration_gap = _mean(valid_calibration, "TrueVsShuffleCalGap")
    residual_conditions = OrderedDict((
        ("minimum_valid_residual_directions", len(valid_residual) >= MIN_VALID_DIRECTIONS_PER_SEED),
        ("minimum_positive_rho_true_directions", int(np.count_nonzero(rho > 0.0)) >= POSITIVE_DIRECTION_MIN_COUNT),
        ("positive_mean_rho_true", mean_rho is not None and mean_rho > 0.0),
        ("one_sided_wilcoxon_rho_true_greater", residual_p is not None and residual_p < ALPHA),
        ("positive_mean_ResidualSpecificityGap", mean_residual_gap is not None and mean_residual_gap > 0.0),
    ))
    calibration_conditions = OrderedDict((
        ("minimum_valid_calibration_directions", len(valid_calibration) >= MIN_VALID_DIRECTIONS_PER_SEED),
        ("minimum_positive_DeltaCalAUC_directions", int(np.count_nonzero(deltas > 0.0)) >= POSITIVE_DIRECTION_MIN_COUNT),
        ("positive_mean_DeltaCalAUC", mean_delta is not None and mean_delta > 0.0),
        ("one_sided_wilcoxon_DeltaCalAUC_greater", calibration_p is not None and calibration_p < ALPHA),
        ("positive_mean_TrueVsShuffleCalGap", mean_calibration_gap is not None and mean_calibration_gap > 0.0),
    ))
    residual_pass = bool(all(residual_conditions.values()))
    calibration_pass = bool(all(calibration_conditions.values()))
    return {
        "stage": STAGE,
        "seed": active_seed,
        "label_count": LABEL_COUNT,
        "unlabeled_evaluation_count": UNLABELED_EVAL_COUNT,
        "valid_residual_direction_count": len(valid_residual),
        "positive_rho_true_direction_count": int(np.count_nonzero(rho > 0.0)),
        "mean_rho_true": mean_rho,
        "mean_ResidualSpecificityGap": mean_residual_gap,
        "residual_wilcoxon_greater_p": residual_p,
        "valid_calibration_direction_count": len(valid_calibration),
        "positive_DeltaCalAUC_direction_count": int(np.count_nonzero(deltas > 0.0)),
        "mean_DeltaCalAUC": mean_delta,
        "mean_TrueVsShuffleCalGap": mean_calibration_gap,
        "calibration_wilcoxon_greater_p": calibration_p,
        "wilcoxon_alternative": "greater",
        "ResidualGate_conditions": dict(residual_conditions),
        "CalibrationGate_conditions": dict(calibration_conditions),
        "ResidualGate": residual_pass,
        "CalibrationGate": calibration_pass,
        "C2_B0_SEED_PASS": residual_pass and calibration_pass,
    }


def _mean_or_none(values):
    if any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=np.float64)
    _require(np.isfinite(array).all(), "C2-B0 non-finite multi-seed metric")
    return float(array.mean())


def build_multiseed_decision(seed_summaries):
    _require(
        tuple(seed_summaries) == SEEDS
        and all(seed_summaries[seed].get("seed") == seed for seed in SEEDS),
        "C2-B0 multi-seed set/order mismatch",
    )
    pass_count = sum(bool(seed_summaries[seed]["C2_B0_SEED_PASS"]) for seed in SEEDS)
    fields = (
        "rho_true", "ResidualSpecificityGap", "DeltaCalAUC", "TrueVsShuffleCalGap"
    )
    aggregate = {
        field: _mean_or_none(
            [seed_summaries[seed]["mean_" + field] for seed in SEEDS]
        )
        for field in fields
    }
    conditions = OrderedDict((
        ("at_least_two_of_three_seed_passes", pass_count >= SEED_PASS_MIN_COUNT),
        ("positive_aggregate_mean_rho_true", aggregate["rho_true"] is not None and aggregate["rho_true"] > 0.0),
        ("positive_aggregate_mean_ResidualSpecificityGap", aggregate["ResidualSpecificityGap"] is not None and aggregate["ResidualSpecificityGap"] > 0.0),
        ("positive_aggregate_mean_DeltaCalAUC", aggregate["DeltaCalAUC"] is not None and aggregate["DeltaCalAUC"] > 0.0),
        ("positive_aggregate_mean_TrueVsShuffleCalGap", aggregate["TrueVsShuffleCalGap"] is not None and aggregate["TrueVsShuffleCalGap"] > 0.0),
    ))
    passed = bool(all(conditions.values()))
    return {
        "summary": {
            "stage": STAGE,
            "seeds": list(SEEDS),
            "seed_pass_count": pass_count,
            **{"aggregate_mean_" + key: value for key, value in aggregate.items()},
            "seed_summaries": {str(seed): seed_summaries[seed] for seed in SEEDS},
        },
        "decision": {
            "decision_conditions": dict(conditions),
            "final_decision": FINAL_DECISIONS[0] if passed else FINAL_DECISIONS[1],
            "C2_B0_SPARSE_UTILITY_RESIDUAL_CALIBRATION_PASS": passed,
            "post_hoc_gate_change": False,
        },
    }
