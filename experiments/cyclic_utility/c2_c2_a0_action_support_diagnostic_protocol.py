"""Pure-array protocol for C2-C2-A0 action-support sufficiency diagnosis.

This stage reproduces the frozen C2-C0 true/shuffled directional paths and
only diagnoses their sparse-anchor support.  It never rewrites U_cycle and
defines no utility-calibration or action-admission method.
"""

from collections import OrderedDict

import numpy as np
from scipy.stats import spearmanr

from experiments.cyclic_utility import c2_c0_directional_consensus_utility_protocol as c0


STAGE = "C2-C2-A0"
SEEDS = c0.SEEDS
SAMPLE_NUM = c0.SAMPLE_NUM
CLASS_NUM = c0.CLASS_NUM
DIRECTION_COUNT = c0.DIRECTION_COUNT
LABEL_COUNT = c0.LABEL_COUNT
UNLABELED_EVAL_COUNT = c0.UNLABELED_EVAL_COUNT
SUPPORT_QUANTILE_COUNT = 5
MIN_VALID_DIRECTIONS_PER_SEED = c0.MIN_VALID_DIRECTIONS_PER_SEED
POSITIVE_DIRECTION_MIN_COUNT = c0.POSITIVE_DIRECTION_MIN_COUNT
SEED_PASS_MIN_COUNT = c0.SEED_PASS_MIN_COUNT
SIGNIFICANCE_LEVEL = c0.SIGNIFICANCE_LEVEL
ZERO_SUPPORT_EPSILON = c0.ZERO_SUPPORT_EPSILON

FIXED_LABELED_IDS = c0.FIXED_LABELED_IDS
FIXED_LABELED_TARGETS = c0.FIXED_LABELED_TARGETS
FIXED_SHUFFLED_TARGETS = c0.FIXED_SHUFFLED_TARGETS

FINAL_DECISIONS = (
    "C2_C2_A0_ACTION_SUPPORT_SUFFICIENCY_DIAGNOSTIC_PASS",
    "C2_C2_A0_ACTION_SUPPORT_SUFFICIENCY_DIAGNOSTIC_FAIL",
)
PASS_BRANCH_DECISION = (
    "SUPPORT_SUFFICIENCY_VALIDATED; A LATER NEW STAGE MAY TEST "
    "SUPPORT_AWARE_UTILITY_ACTION_USAGE"
)
FAIL_BRANCH_DECISION = (
    "CLOSE_SCALAR_UTILITY_CORRECTION_BRANCH; MOVE TO "
    "U_CONDITIONED_WEAK_LABEL_ACTION_POLICY"
)

# Frozen C2-C0/B0 primitives are aliases, not reimplementations.
canonical_sample_ids = c0.canonical_sample_ids
fixed_labeled_ids = c0.fixed_labeled_ids
fixed_labeled_targets = c0.fixed_labeled_targets
fixed_shuffled_targets = c0.fixed_shuffled_targets
validate_fixed_sparse_split = c0.validate_fixed_sparse_split
validate_negative_control = c0.validate_negative_control
build_vote_semantic_state = c0.build_vote_semantic_state
build_leave_one_out_action_correctness = c0.build_leave_one_out_action_correctness
build_labeled_residual = c0.build_labeled_residual
build_labeled_residual_directions = c0.build_labeled_residual_directions
binary_auc_or_none = c0.binary_auc_or_none
one_sided_wilcoxon_greater = c0.one_sided_wilcoxon_greater


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
        raise ValueError("C2-C2-A0 seed must be one of " + str(SEEDS))
    return value


def compute_effective_support(weights):
    """Compute M/Q/ESS:[N,S] and active count from weights:[N,L,S]."""
    values = _readonly(weights, dtype=np.float64)
    _require(
        values.shape == (SAMPLE_NUM, LABEL_COUNT, DIRECTION_COUNT)
        and np.isfinite(values).all()
        and np.all(values >= 0.0),
        "C2-C2-A0 support-weight boundary mismatch",
    )
    # weights:[N,L,S]; M/Q/ESS/active_anchor_count:[N,S].
    support_mass = np.ascontiguousarray(values.sum(axis=1), dtype=np.float64)
    support_square_mass = np.ascontiguousarray(
        np.square(values).sum(axis=1), dtype=np.float64
    )
    active_anchor_count = np.ascontiguousarray(
        np.count_nonzero(values > 0.0, axis=1), dtype=np.int64
    )
    supported = support_mass > ZERO_SUPPORT_EPSILON
    _require(
        np.isfinite(support_mass).all()
        and np.isfinite(support_square_mass).all()
        and np.all(support_mass >= 0.0)
        and np.all(support_square_mass >= 0.0)
        and np.all(support_square_mass[supported] > 0.0),
        "C2-C2-A0 support mass boundary mismatch",
    )
    raw_ess = np.divide(
        np.square(support_mass),
        support_square_mass,
        out=np.zeros((SAMPLE_NUM, DIRECTION_COUNT), dtype=np.float64),
        where=supported,
    )
    float64_epsilon = np.finfo(np.float64).eps
    bound_tolerance = 32.0 * float64_epsilon * LABEL_COUNT
    _require(
        np.isfinite(raw_ess).all()
        and np.all(raw_ess[~supported] == 0.0)
        and np.all(raw_ess >= -bound_tolerance)
        and np.all(raw_ess <= LABEL_COUNT + bound_tolerance),
        "C2-C2-A0 ESS exceeds float64 numerical tolerance",
    )
    ess = np.ascontiguousarray(
        np.clip(raw_ess, 0.0, float(LABEL_COUNT)), dtype=np.float64
    )
    _require(
        np.isfinite(ess).all()
        and np.all((ess >= 0.0) & (ess <= LABEL_COUNT))
        and np.all(ess[~supported] == 0.0)
        and np.all((active_anchor_count >= 0) & (active_anchor_count <= LABEL_COUNT)),
        "C2-C2-A0 canonical ESS boundary mismatch",
    )
    for value in (
        support_mass, support_square_mass, ess, active_anchor_count
    ):
        value.setflags(write=False)
    return {
        "support_mass": support_mass,
        "support_square_mass": support_square_mass,
        "ESS": ess,
        "active_anchor_count": active_anchor_count,
    }


def build_action_support_outputs(
    y_gen, U_cycle, labeled_ids, labeled_targets, shuffled_targets
):
    """Rebuild both frozen C2-C0 paths and diagnose support independently."""
    c0_outputs = c0.build_directional_consensus_outputs(
        y_gen, U_cycle, labeled_ids, labeled_targets, shuffled_targets
    )
    true_support = compute_effective_support(c0_outputs["weights"])
    shuffle_support = compute_effective_support(c0_outputs["shuffle_weights"])
    return {
        "R": c0_outputs["R"],
        "z_L": c0_outputs["z_L"],
        "e_L": c0_outputs["e_L"],
        "d_L": c0_outputs["d_L"],
        "weights": c0_outputs["weights"],
        "D": c0_outputs["D"],
        **true_support,
        "loo_mappings": c0_outputs["loo_mappings"],
        "loo_soft_contingency": c0_outputs["loo_soft_contingency"],
        "loo_included_label_mask": c0_outputs["loo_included_label_mask"],
        "z_L_shuffle": c0_outputs["z_L_shuffle"],
        "e_L_shuffle": c0_outputs["e_L_shuffle"],
        "d_L_shuffle": c0_outputs["d_L_shuffle"],
        "weights_shuffle": c0_outputs["shuffle_weights"],
        "D_shuffle": c0_outputs["D_shuffle"],
        "support_mass_shuffle": shuffle_support["support_mass"],
        "support_square_mass_shuffle": shuffle_support["support_square_mass"],
        "ESS_shuffle": shuffle_support["ESS"],
        "active_anchor_count_shuffle": shuffle_support[
            "active_anchor_count"
        ],
        "loo_mappings_shuffle": c0_outputs["loo_mappings_shuffle"],
        "loo_soft_contingency_shuffle": c0_outputs[
            "loo_soft_contingency_shuffle"
        ],
        "loo_included_label_mask_shuffle": c0_outputs[
            "loo_included_label_mask_shuffle"
        ],
        "C2_C0_weights_reproduced_exactly": True,
        "C2_C0_D_reproduced_exactly": True,
        "shuffle_recomputed_from_targets": c0_outputs[
            "shuffle_recomputed_from_targets"
        ],
        "weights_shuffle_independently_rebuilt": True,
        "true_weights_reused_for_shuffle": False,
        "ESS_shuffle_independently_computed": True,
        "GT_used_for_support": False,
    }


def build_directional_correctness(D, D_shuffle, e_GT, sample_ids):
    """Build DirCorrect:[N,S] only on valid [1386,20] directional actions."""
    consensus = _readonly(D, dtype=np.float64)
    consensus_shuffle = _readonly(D_shuffle, dtype=np.float64)
    truth_residual = _readonly(e_GT, dtype=np.float64)
    ids = _readonly(sample_ids, dtype=np.int64)
    expected = (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
    _require(
        consensus.shape == consensus_shuffle.shape == truth_residual.shape
        == expected
        and ids.shape == (UNLABELED_EVAL_COUNT,)
        and np.array_equal(
            ids, np.setdiff1d(canonical_sample_ids(), fixed_labeled_ids())
        )
        and all(
            np.isfinite(value).all()
            for value in (consensus, consensus_shuffle, truth_residual)
        ),
        "C2-C2-A0 directional correctness must use 1386 unlabeled samples",
    )
    valid = (consensus != 0.0) & (truth_residual != 0.0)
    valid_shuffle = (consensus_shuffle != 0.0) & (truth_residual != 0.0)
    correct = np.zeros(expected, dtype=np.int8)
    correct_shuffle = np.zeros(expected, dtype=np.int8)
    correct[valid] = (
        np.sign(consensus[valid]) == np.sign(truth_residual[valid])
    ).astype(np.int8)
    correct_shuffle[valid_shuffle] = (
        np.sign(consensus_shuffle[valid_shuffle])
        == np.sign(truth_residual[valid_shuffle])
    ).astype(np.int8)
    for value in (valid, valid_shuffle, correct, correct_shuffle):
        value.setflags(write=False)
    return {
        "DirCorrect": correct,
        "valid_directional_action_mask": valid,
        "DirCorrect_shuffle": correct_shuffle,
        "valid_directional_action_mask_shuffle": valid_shuffle,
    }


def support_equal_count_quantiles(ESS, sample_ids):
    """Assign five deterministic equal-count bins using ESS only."""
    support = _readonly(ESS, dtype=np.float64)
    ids = _readonly(sample_ids, dtype=np.int64)
    assignments, source = c0.confidence_equal_count_strata(support, ids)
    bins = []
    for record in source["bins"]:
        bins.append({
            "bin_id": record["bin_id"],
            "bin_label": record["bin_label"],
            "sample_count": record["sample_count"],
            "rank_start_inclusive": record["rank_start_inclusive"],
            "rank_end_exclusive": record["rank_end_exclusive"],
            "sample_ids_logical_sha256": record[
                "sample_ids_logical_sha256"
            ],
            "ESS_min": record["confidence_min"],
            "ESS_max": record["confidence_max"],
        })
    return assignments, {
        "bin_count": SUPPORT_QUANTILE_COUNT,
        "sample_count": int(support.size),
        "assignment_rule": (
            "stable_ESS_ascending_then_sample_id; equal_count_rank_chunks"
        ),
        "ESS_only_for_assignment": True,
        "GT_used_for_assignment": False,
        "sample_id_only_tie_break": True,
        "approximately_equal_count_pass": source[
            "approximately_equal_count_pass"
        ],
        "bin_sizes": source["bin_sizes"],
        "assignment_logical_sha256": source[
            "assignment_logical_sha256"
        ],
        "bins": bins,
    }


def frozen_c0_action_and_benefit(z_GT, U_cycle, D):
    """Reproduce frozen C0 action and ActionBenefit on [1386,20] arrays."""
    target = _readonly(z_GT, dtype=np.float64)
    cycle = _readonly(U_cycle, dtype=np.float64)
    consensus = _readonly(D, dtype=np.float64)
    expected = (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
    _require(
        target.shape == cycle.shape == consensus.shape == expected
        and np.all((target == 0.0) | (target == 1.0))
        and all(np.isfinite(value).all() for value in (cycle, consensus))
        and np.all((cycle >= 0.0) & (cycle <= 1.0))
        and np.all((consensus >= -1.0) & (consensus <= 1.0)),
        "C2-C2-A0 frozen C0 action input boundary mismatch",
    )
    frozen_action = cycle + cycle * (1.0 - cycle) * consensus
    frozen_action[cycle == 0.0] = 0.0
    benefit = np.ascontiguousarray(
        np.abs(target - cycle) - np.abs(target - frozen_action),
        dtype=np.float64,
    )
    _require(
        np.isfinite(frozen_action).all()
        and np.all((frozen_action >= 0.0) & (frozen_action <= 1.0))
        and np.isfinite(benefit).all(),
        "C2-C2-A0 frozen C0 action-benefit boundary mismatch",
    )
    frozen_action.setflags(write=False)
    benefit.setflags(write=False)
    return {"U_C0": frozen_action, "ActionBenefit": benefit}


def analyze_support_direction(
    ESS, ESS_shuffle, DirCorrect, valid_mask,
    DirCorrect_shuffle, valid_mask_shuffle,
    ActionBenefit, sample_ids, direction_id,
):
    support = _readonly(ESS, dtype=np.float64)
    support_shuffle = _readonly(ESS_shuffle, dtype=np.float64)
    correct = _readonly(DirCorrect, dtype=np.int8)
    valid = _readonly(valid_mask, dtype=np.bool_)
    correct_shuffle = _readonly(DirCorrect_shuffle, dtype=np.int8)
    valid_shuffle = _readonly(valid_mask_shuffle, dtype=np.bool_)
    benefit = _readonly(ActionBenefit, dtype=np.float64)
    ids = _readonly(sample_ids, dtype=np.int64)
    expected = (UNLABELED_EVAL_COUNT,)
    _require(
        support.shape == support_shuffle.shape == correct.shape == valid.shape
        == correct_shuffle.shape == valid_shuffle.shape == benefit.shape
        == ids.shape == expected
        and np.unique(ids).size == UNLABELED_EVAL_COUNT
        and all(
            np.isfinite(value).all()
            for value in (support, support_shuffle, benefit)
        )
        and np.all((correct == 0) | (correct == 1))
        and np.all((correct_shuffle == 0) | (correct_shuffle == 1)),
        "C2-C2-A0 support-direction input boundary mismatch",
    )
    true_outcomes = correct[valid]
    shuffle_outcomes = correct_shuffle[valid_shuffle]
    true_support = support[valid]
    shuffle_support = support_shuffle[valid_shuffle]
    true_valid = bool(
        np.unique(true_outcomes).size == 2
        and np.unique(true_support).size >= 2
    )
    shuffle_valid = bool(
        np.unique(shuffle_outcomes).size == 2
        and np.unique(shuffle_support).size >= 2
    )
    joint_valid = true_valid and shuffle_valid
    result = {
        "direction_id": int(direction_id),
        "true_valid_action_count": int(np.count_nonzero(valid)),
        "shuffle_valid_action_count": int(np.count_nonzero(valid_shuffle)),
        "true_target_has_both_classes": bool(np.unique(true_outcomes).size == 2),
        "shuffle_target_has_both_classes": bool(
            np.unique(shuffle_outcomes).size == 2
        ),
        "true_ESS_has_two_distinct_values": bool(
            np.unique(true_support).size >= 2
        ),
        "shuffle_ESS_has_two_distinct_values": bool(
            np.unique(shuffle_support).size >= 2
        ),
        "true_support_AUC_valid": true_valid,
        "shuffle_support_AUC_valid": shuffle_valid,
        "joint_valid": joint_valid,
        "AUC_support_true": None,
        "AUC_support_shuffle": None,
        "SupportSpecificityGap": None,
        "Agree_Q1": None,
        "Agree_Q2": None,
        "Agree_Q3": None,
        "Agree_Q4": None,
        "Agree_Q5": None,
        "HighLowAgreementGap": None,
        "SpearmanSupportCorrectness": None,
        "SpearmanSupportActionBenefit": None,
        "mean_ActionBenefit_Q1": None,
        "mean_ActionBenefit_Q2": None,
        "mean_ActionBenefit_Q3": None,
        "mean_ActionBenefit_Q4": None,
        "mean_ActionBenefit_Q5": None,
        "support_quantiles": None,
    }
    if not joint_valid:
        return result

    auc_true = binary_auc_or_none(true_outcomes, true_support)
    auc_shuffle = binary_auc_or_none(shuffle_outcomes, shuffle_support)
    assignments, quantile_audit = support_equal_count_quantiles(
        true_support, ids[valid]
    )
    bins = []
    agreements = []
    benefit_means = []
    true_benefit = benefit[valid]
    for bin_id in range(SUPPORT_QUANTILE_COUNT):
        indices = np.flatnonzero(assignments == bin_id)
        agreement = float(np.mean(true_outcomes[indices]))
        benefit_mean = float(np.mean(true_benefit[indices]))
        agreements.append(agreement)
        benefit_means.append(benefit_mean)
        bins.append({
            **quantile_audit["bins"][bin_id],
            "Agree": agreement,
            "mean_ActionBenefit": benefit_mean,
        })
    support_correctness_rho = float(
        spearmanr(true_support, true_outcomes).statistic
    )
    support_benefit_rho_raw = float(
        spearmanr(true_support, true_benefit).statistic
    )
    support_benefit_rho = (
        support_benefit_rho_raw
        if np.isfinite(support_benefit_rho_raw)
        else None
    )
    _require(
        auc_true is not None
        and auc_shuffle is not None
        and np.isfinite(support_correctness_rho),
        "C2-C2-A0 valid support direction produced invalid primary metric",
    )
    result.update({
        "AUC_support_true": float(auc_true),
        "AUC_support_shuffle": float(auc_shuffle),
        "SupportSpecificityGap": float(auc_true - auc_shuffle),
        **{
            "Agree_Q" + str(index + 1): value
            for index, value in enumerate(agreements)
        },
        "HighLowAgreementGap": agreements[-1] - agreements[0],
        "SpearmanSupportCorrectness": support_correctness_rho,
        "SpearmanSupportActionBenefit": support_benefit_rho,
        **{
            "mean_ActionBenefit_Q" + str(index + 1): value
            for index, value in enumerate(benefit_means)
        },
        "support_quantiles": {
            **quantile_audit,
            "GT_used_for_assignment": False,
            "bins": bins,
        },
    })
    return result


def analyze_support_directions(
    ESS, ESS_shuffle, D, D_shuffle, e_GT,
    ActionBenefit, sample_ids,
):
    arrays = (
        _readonly(ESS, dtype=np.float64),
        _readonly(ESS_shuffle, dtype=np.float64),
        _readonly(D, dtype=np.float64),
        _readonly(D_shuffle, dtype=np.float64),
        _readonly(e_GT, dtype=np.float64),
        _readonly(ActionBenefit, dtype=np.float64),
    )
    ids = _readonly(sample_ids, dtype=np.int64)
    expected = (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
    _require(
        all(value.shape == expected for value in arrays)
        and ids.shape == (UNLABELED_EVAL_COUNT,)
        and np.array_equal(
            ids, np.setdiff1d(canonical_sample_ids(), fixed_labeled_ids())
        ),
        "C2-C2-A0 support evaluation must use fixed 1386 unlabeled samples",
    )
    correctness = build_directional_correctness(
        arrays[2], arrays[3], arrays[4], ids
    )
    directions = [
        analyze_support_direction(
            arrays[0][:, direction_id],
            arrays[1][:, direction_id],
            correctness["DirCorrect"][:, direction_id],
            correctness["valid_directional_action_mask"][:, direction_id],
            correctness["DirCorrect_shuffle"][:, direction_id],
            correctness["valid_directional_action_mask_shuffle"][:, direction_id],
            arrays[5][:, direction_id],
            ids,
            direction_id,
        )
        for direction_id in range(DIRECTION_COUNT)
    ]
    return {
        "direction_count": DIRECTION_COUNT,
        "evaluation_sample_count": UNLABELED_EVAL_COUNT,
        "evaluation_only_unlabeled": True,
        "GT_used_for_support": False,
        "GT_used_for_quantile_assignment": False,
        "ActionBenefit_used_in_gate": False,
        "correctness": correctness,
        "directions": directions,
    }


def _mean(records, field):
    if not records:
        return None
    values = np.asarray([record[field] for record in records], dtype=np.float64)
    _require(np.isfinite(values).all(), "C2-C2-A0 non-finite seed metric")
    return float(values.mean())


def build_seed_summary(seed, support_metrics):
    active_seed = validate_seed(seed)
    directions = support_metrics.get("directions", ())
    _require(
        len(directions) == DIRECTION_COUNT
        and all(
            directions[index].get("direction_id") == index
            for index in range(DIRECTION_COUNT)
        ),
        "C2-C2-A0 seed direction schema mismatch",
    )
    valid = [record for record in directions if record["joint_valid"]]
    auc = np.asarray(
        [record["AUC_support_true"] for record in valid], dtype=np.float64
    )
    centered_auc = auc - 0.5
    high_low = np.asarray(
        [record["HighLowAgreementGap"] for record in valid], dtype=np.float64
    )
    auc_p = one_sided_wilcoxon_greater(centered_auc)
    mean_auc = _mean(valid, "AUC_support_true")
    mean_specificity = _mean(valid, "SupportSpecificityGap")
    mean_high_low = _mean(valid, "HighLowAgreementGap")
    benefit_records = [
        record for record in valid
        if record["SpearmanSupportActionBenefit"] is not None
    ]
    conditions = OrderedDict((
        (
            "minimum_valid_support_AUC_directions",
            len(valid) >= MIN_VALID_DIRECTIONS_PER_SEED,
        ),
        (
            "minimum_AUC_support_true_above_half_directions",
            int(np.count_nonzero(auc > 0.5)) >= POSITIVE_DIRECTION_MIN_COUNT,
        ),
        (
            "mean_AUC_support_true_above_half",
            mean_auc is not None and mean_auc > 0.5,
        ),
        (
            "one_sided_wilcoxon_AUC_minus_half_greater",
            auc_p is not None and auc_p < SIGNIFICANCE_LEVEL,
        ),
        (
            "positive_mean_SupportSpecificityGap",
            mean_specificity is not None and mean_specificity > 0.0,
        ),
        (
            "minimum_positive_HighLowAgreementGap_directions",
            int(np.count_nonzero(high_low > 0.0))
            >= POSITIVE_DIRECTION_MIN_COUNT,
        ),
        (
            "positive_mean_HighLowAgreementGap",
            mean_high_low is not None and mean_high_low > 0.0,
        ),
    ))
    passed = bool(all(conditions.values()))
    return {
        "stage": STAGE,
        "seed": active_seed,
        "label_count": LABEL_COUNT,
        "unlabeled_evaluation_count": UNLABELED_EVAL_COUNT,
        "valid_support_AUC_direction_count": len(valid),
        "AUC_support_true_above_half_direction_count": int(
            np.count_nonzero(auc > 0.5)
        ),
        "mean_AUC_support_true": mean_auc,
        "AUC_minus_half_wilcoxon_greater_p": auc_p,
        "mean_SupportSpecificityGap": mean_specificity,
        "positive_HighLowAgreementGap_direction_count": int(
            np.count_nonzero(high_low > 0.0)
        ),
        "mean_HighLowAgreementGap": mean_high_low,
        "mean_SpearmanSupportCorrectness": _mean(
            valid, "SpearmanSupportCorrectness"
        ),
        "mean_SpearmanSupportActionBenefit": _mean(
            benefit_records, "SpearmanSupportActionBenefit"
        ),
        "wilcoxon_alternative": "greater",
        "SeedGate_conditions": dict(conditions),
        "ActionBenefit_used_in_gate": False,
        "support_mass_used_in_gate": False,
        "active_anchor_count_used_in_gate": False,
        "C1_admission_rate_used_in_gate": False,
        "C2_C2_A0_SEED_PASS": passed,
    }


def _mean_or_none(values):
    if any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=np.float64)
    _require(np.isfinite(array).all(), "C2-C2-A0 non-finite multi-seed metric")
    return float(array.mean())


def build_multiseed_decision(seed_summaries):
    _require(
        tuple(seed_summaries) == SEEDS
        and all(seed_summaries[seed].get("seed") == seed for seed in SEEDS),
        "C2-C2-A0 multi-seed set/order mismatch",
    )
    pass_count = sum(
        bool(seed_summaries[seed]["C2_C2_A0_SEED_PASS"])
        for seed in SEEDS
    )
    aggregate_auc = _mean_or_none([
        seed_summaries[seed]["mean_AUC_support_true"] for seed in SEEDS
    ])
    aggregate_specificity = _mean_or_none([
        seed_summaries[seed]["mean_SupportSpecificityGap"] for seed in SEEDS
    ])
    aggregate_high_low = _mean_or_none([
        seed_summaries[seed]["mean_HighLowAgreementGap"] for seed in SEEDS
    ])
    conditions = OrderedDict((
        ("at_least_two_of_three_seed_passes", pass_count >= SEED_PASS_MIN_COUNT),
        (
            "aggregate_mean_AUC_support_true_above_half",
            aggregate_auc is not None and aggregate_auc > 0.5,
        ),
        (
            "positive_aggregate_mean_SupportSpecificityGap",
            aggregate_specificity is not None and aggregate_specificity > 0.0,
        ),
        (
            "positive_aggregate_mean_HighLowAgreementGap",
            aggregate_high_low is not None and aggregate_high_low > 0.0,
        ),
    ))
    passed = bool(all(conditions.values()))
    return {
        "summary": {
            "stage": STAGE,
            "seeds": list(SEEDS),
            "seed_pass_count": pass_count,
            "aggregate_mean_AUC_support_true": aggregate_auc,
            "aggregate_mean_SupportSpecificityGap": aggregate_specificity,
            "aggregate_mean_HighLowAgreementGap": aggregate_high_low,
            "seed_summaries": {
                str(seed): seed_summaries[seed] for seed in SEEDS
            },
        },
        "decision": {
            "decision_conditions": dict(conditions),
            "final_decision": FINAL_DECISIONS[0] if passed else FINAL_DECISIONS[1],
            "C2_C2_A0_ACTION_SUPPORT_SUFFICIENCY_DIAGNOSTIC_PASS": passed,
            "precommitted_branch_decision": (
                PASS_BRANCH_DECISION if passed else FAIL_BRANCH_DECISION
            ),
            "ActionBenefit_used_in_gate": False,
            "post_hoc_gate_change": False,
        },
    }
