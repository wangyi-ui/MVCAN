"""Pure-array protocol for C2-C0 directional consensus utility calibration.

The frozen C2-B0 construction is reused exactly through ``R``, leave-one-out
``z_L``, ``e_L``, semantic similarity, action matching, and propagation
weights.  C2-C0 then discards residual magnitude: sparse residual signs are
combined into ``D`` and the frozen cycle utility supplies correction capacity.

This module performs no artifact I/O, ground-truth loading, model execution,
or training.
"""

from collections import OrderedDict

import numpy as np

from experiments.cyclic_utility import c2_b0_sparse_utility_residual_protocol as b0


STAGE = "C2-C0"
SEEDS = b0.SEEDS
SAMPLE_NUM = b0.SAMPLE_NUM
CLASS_NUM = b0.CLASS_NUM
DIRECTION_COUNT = b0.DIRECTION_COUNT
LABEL_COUNT = b0.LABEL_COUNT
UNLABELED_EVAL_COUNT = b0.UNLABELED_EVAL_COUNT
CONFIDENCE_BIN_COUNT = b0.CONFIDENCE_BIN_COUNT
MIN_VALID_BINS_PER_DIRECTION = b0.MIN_VALID_BINS_PER_DIRECTION
MIN_VALID_DIRECTIONS_PER_SEED = b0.MIN_VALID_DIRECTIONS_PER_SEED
POSITIVE_DIRECTION_MIN_COUNT = b0.POSITIVE_DIRECTION_MIN_COUNT
SEED_PASS_MIN_COUNT = b0.SEED_PASS_MIN_COUNT
SIGNIFICANCE_LEVEL = 0.05
ZERO_SUPPORT_EPSILON = b0.ZERO_SUPPORT_EPSILON

FIXED_LABELED_IDS = b0.FIXED_LABELED_IDS
FIXED_LABELED_TARGETS = b0.FIXED_LABELED_TARGETS
FIXED_SHUFFLED_TARGETS = b0.FIXED_SHUFFLED_TARGETS

FINAL_DECISIONS = (
    "C2_C0_DIRECTIONAL_CONSENSUS_UTILITY_CALIBRATION_PASS",
    "C2_C0_DIRECTIONAL_CONSENSUS_UTILITY_CALIBRATION_FAIL",
)

# Frozen C2-B0 primitives.  These aliases make the scientific inheritance
# explicit and prevent a second implementation from drifting from B0.
canonical_sample_ids = b0.canonical_sample_ids
fixed_labeled_ids = b0.fixed_labeled_ids
fixed_labeled_targets = b0.fixed_labeled_targets
fixed_shuffled_targets = b0.fixed_shuffled_targets
validate_fixed_sparse_split = b0.validate_fixed_sparse_split
validate_negative_control = b0.validate_negative_control
build_vote_semantic_state = b0.build_vote_semantic_state
build_leave_one_out_action_correctness = b0.build_leave_one_out_action_correctness
build_labeled_residual = b0.build_labeled_residual
confidence_equal_count_strata = b0.confidence_equal_count_strata
binary_auc_or_none = b0.binary_auc_or_none
one_sided_wilcoxon_greater = b0.one_sided_wilcoxon_greater


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
        raise ValueError("C2-C0 seed must be one of " + str(SEEDS))
    return value


def build_labeled_residual_directions(e_L):
    """Convert each labeled residual to the standard mathematical sign."""
    residual = _readonly(e_L, dtype=np.float64)
    _require(
        residual.shape == (LABEL_COUNT, DIRECTION_COUNT)
        and np.isfinite(residual).all()
        and np.all((residual >= -1.0) & (residual <= 1.0)),
        "C2-C0 labeled-residual sign input boundary mismatch",
    )
    directions = np.ascontiguousarray(np.sign(residual), dtype=np.float64)
    _require(
        np.all((directions == -1.0) | (directions == 0.0) | (directions == 1.0))
        and np.all(directions[residual > 0.0] == 1.0)
        and np.all(directions[residual < 0.0] == -1.0)
        and np.all(directions[residual == 0.0] == 0.0),
        "C2-C0 labeled-residual sign conversion failed",
    )
    directions.setflags(write=False)
    return directions


def propagate_action_context_directional_consensus(
    vote_state, y_gen, labeled_ids, d_L
):
    """Combine residual signs with the exact frozen B0 weight topology."""
    directions = _readonly(d_L, dtype=np.float64)
    _require(
        directions.shape == (LABEL_COUNT, DIRECTION_COUNT)
        and np.isfinite(directions).all()
        and np.all(
            (directions == -1.0) | (directions == 0.0) | (directions == 1.0)
        ),
        "C2-C0 labeled-direction propagation input boundary mismatch",
    )

    # B0 constructs exactly:
    # w[i,ell,s] = cosine(R_i, R_ell) * 1[y_gen[i,s] == y_gen[ell,s]].
    # Passing signs is safe because this helper's topology is independent of
    # residual values.  Its residual output is deliberately ignored below.
    topology = b0.propagate_action_context_residuals(
        vote_state, y_gen, labeled_ids, directions
    )
    weights = topology["weights"]
    denominator = topology["support"]
    numerator = np.ascontiguousarray(
        (weights * directions[None, :, :]).sum(axis=1), dtype=np.float64
    )
    consensus = np.divide(
        numerator,
        denominator,
        out=np.zeros((SAMPLE_NUM, DIRECTION_COUNT), dtype=np.float64),
        where=denominator > ZERO_SUPPORT_EPSILON,
    )
    _require(
        consensus.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and np.isfinite(consensus).all()
        and np.all((consensus >= -1.0) & (consensus <= 1.0))
        and np.all(consensus[denominator <= ZERO_SUPPORT_EPSILON] == 0.0),
        "C2-C0 directional consensus boundary mismatch",
    )
    numerator.setflags(write=False)
    consensus.setflags(write=False)
    return {
        "semantic_similarity": topology["semantic_similarity"],
        "action_match": topology["action_match"],
        "weights": weights,
        "denom": denominator,
        "directional_numerator": numerator,
        "D": consensus,
        "C_conf_used": False,
        "residual_magnitude_propagated": False,
        "top_k_used": False,
        "temperature_used": False,
    }


def directional_utility_update(U_cycle, D):
    """Apply U + U(1-U)D with no fitted or tunable coefficient."""
    cycle = _readonly(U_cycle, dtype=np.float64)
    consensus = _readonly(D, dtype=np.float64)
    _require(
        cycle.shape == consensus.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and np.isfinite(cycle).all()
        and np.isfinite(consensus).all()
        and np.all((cycle >= 0.0) & (cycle <= 1.0))
        and np.all((consensus >= -1.0) & (consensus <= 1.0)),
        "C2-C0 directional utility-update input boundary mismatch",
    )
    calibrated = cycle + cycle * (1.0 - cycle) * consensus
    calibrated[cycle == 0.0] = 0.0
    _require(
        np.isfinite(calibrated).all()
        and np.all((calibrated >= 0.0) & (calibrated <= 1.0))
        and np.all(calibrated[cycle == 0.0] == 0.0),
        "C2-C0 directional utility update failed",
    )
    calibrated.setflags(write=False)
    return calibrated


def _build_one_directional_path(vote_state, y_gen, U_cycle, labeled_ids, targets):
    loo = build_leave_one_out_action_correctness(
        vote_state, y_gen, labeled_ids, targets
    )
    residual = build_labeled_residual(loo["z_L"], U_cycle, labeled_ids)
    directions = build_labeled_residual_directions(residual)
    propagation = propagate_action_context_directional_consensus(
        vote_state, y_gen, labeled_ids, directions
    )
    calibrated = directional_utility_update(U_cycle, propagation["D"])
    return {
        **loo,
        "e_L": residual,
        "d_L": directions,
        **propagation,
        "U_tilde_dir": calibrated,
    }


def build_directional_consensus_outputs(
    y_gen, U_cycle, labeled_ids, labeled_targets, shuffled_targets
):
    """Build true and shuffled directional paths independently from targets."""
    actions = _readonly(y_gen, dtype=np.int64)
    cycle = _readonly(U_cycle, dtype=np.float64)
    labeled = _readonly(labeled_ids, dtype=np.int64)
    targets = _readonly(labeled_targets, dtype=np.int64)
    shuffled = validate_negative_control(targets, shuffled_targets)
    state = build_vote_semantic_state(actions)
    true_path = _build_one_directional_path(
        state, actions, cycle, labeled, targets
    )
    shuffle_path = _build_one_directional_path(
        state, actions, cycle, labeled, shuffled
    )
    return {
        "R": state,
        "z_L": true_path["z_L"],
        "e_L": true_path["e_L"],
        "d_L": true_path["d_L"],
        "D": true_path["D"],
        "U_tilde_dir": true_path["U_tilde_dir"],
        "loo_mappings": true_path["loo_mappings"],
        "loo_soft_contingency": true_path["loo_soft_contingency"],
        "loo_included_label_mask": true_path["loo_included_label_mask"],
        "semantic_similarity": true_path["semantic_similarity"],
        "action_match": true_path["action_match"],
        "weights": true_path["weights"],
        "denom": true_path["denom"],
        "z_L_shuffle": shuffle_path["z_L"],
        "e_L_shuffle": shuffle_path["e_L"],
        "d_L_shuffle": shuffle_path["d_L"],
        "D_shuffle": shuffle_path["D"],
        "U_tilde_dir_shuffle": shuffle_path["U_tilde_dir"],
        "loo_mappings_shuffle": shuffle_path["loo_mappings"],
        "loo_soft_contingency_shuffle": shuffle_path["loo_soft_contingency"],
        "loo_included_label_mask_shuffle": shuffle_path[
            "loo_included_label_mask"
        ],
        "shuffle_semantic_similarity": shuffle_path["semantic_similarity"],
        "shuffle_action_match": shuffle_path["action_match"],
        "shuffle_weights": shuffle_path["weights"],
        "shuffle_denom": shuffle_path["denom"],
        "shuffle_recomputed_from_targets": True,
        "shuffle_only_final_D": False,
        "residual_magnitude_propagated": False,
    }


def analyze_directional_sign_agreement(D, D_shuffle, e_GT, sample_ids):
    """Post-seal diagnostic of predicted versus true correction direction."""
    predicted = _readonly(D, dtype=np.float64)
    shuffled = _readonly(D_shuffle, dtype=np.float64)
    truth = _readonly(e_GT, dtype=np.float64)
    ids = _readonly(sample_ids, dtype=np.int64)
    expected = (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
    _require(
        predicted.shape == shuffled.shape == truth.shape == expected
        and ids.shape == (UNLABELED_EVAL_COUNT,)
        and np.array_equal(
            ids, np.setdiff1d(canonical_sample_ids(), fixed_labeled_ids())
        )
        and all(np.isfinite(value).all() for value in (predicted, shuffled, truth))
        and np.all((predicted >= -1.0) & (predicted <= 1.0))
        and np.all((shuffled >= -1.0) & (shuffled <= 1.0)),
        "C2-C0 directional diagnostic must use exactly 1386 unlabeled samples",
    )
    records = []
    for direction_id in range(DIRECTION_COUNT):
        e_direction = truth[:, direction_id]
        predicted_direction = predicted[:, direction_id]
        shuffled_direction = shuffled[:, direction_id]
        true_mask = (predicted_direction != 0.0) & (e_direction != 0.0)
        shuffle_mask = (shuffled_direction != 0.0) & (e_direction != 0.0)
        true_count = int(np.count_nonzero(true_mask))
        shuffle_count = int(np.count_nonzero(shuffle_mask))
        true_agreement = (
            float(np.mean(
                np.sign(predicted_direction[true_mask])
                == np.sign(e_direction[true_mask])
            ))
            if true_count
            else None
        )
        shuffle_agreement = (
            float(np.mean(
                np.sign(shuffled_direction[shuffle_mask])
                == np.sign(e_direction[shuffle_mask])
            ))
            if shuffle_count
            else None
        )
        valid = true_agreement is not None and shuffle_agreement is not None
        records.append({
            "direction_id": direction_id,
            "true_eligible_count": true_count,
            "shuffle_eligible_count": shuffle_count,
            "directional_sign_agreement_true": true_agreement,
            "directional_sign_agreement_shuffle": shuffle_agreement,
            "DirectionalSpecificityGap": (
                true_agreement - shuffle_agreement if valid else None
            ),
            "diagnostic_valid": valid,
        })
    return {
        "direction_count": DIRECTION_COUNT,
        "evaluation_sample_count": UNLABELED_EVAL_COUNT,
        "evaluation_only_unlabeled": True,
        "used_to_construct_U_tilde_dir": False,
        "used_in_seed_gate": False,
        "directions": records,
    }


def analyze_calibration_direction(
    correct, U_cycle, U_tilde_dir, U_tilde_dir_shuffle,
    C_conf, sample_ids, direction_id,
):
    """Reuse B0 confidence-matched AUC evaluation with directional names."""
    source = b0.analyze_calibration_direction(
        correct, U_cycle, U_tilde_dir, U_tilde_dir_shuffle,
        C_conf, sample_ids, direction_id,
    )
    bins = [{
        "bin_id": record["bin_id"],
        "sample_count": record["sample_count"],
        "valid_both_correctness_classes": record[
            "valid_both_correctness_classes"
        ],
        "AUC_Ucycle": record["AUC_Ucycle"],
        "AUC_U_tilde_dir": record["AUC_U_tilde"],
        "AUC_U_tilde_dir_shuffle": record["AUC_U_tilde_shuffle"],
    } for record in source["bins"]]
    return {
        "direction_id": source["direction_id"],
        "valid_bin_count": source["valid_bin_count"],
        "valid_bin_mask": source["valid_bin_mask"],
        "same_valid_bin_mask_all_utility_arms": source[
            "same_valid_bin_mask_all_utility_arms"
        ],
        "direction_valid": source["direction_valid"],
        "bins": bins,
        "CondAUC_Ucycle": source["CondAUC_Ucycle"],
        "CondAUC_U_tilde_dir": source["CondAUC_U_tilde"],
        "CondAUC_U_tilde_dir_shuffle": source[
            "CondAUC_U_tilde_shuffle"
        ],
        "DeltaCalAUC_dir": source["DeltaCalAUC"],
        "TrueVsShuffleCalGap_dir": source["TrueVsShuffleCalGap"],
        "confidence_strata": source["confidence_strata"],
    }


def analyze_calibration_directions(
    correct, U_cycle, U_tilde_dir, U_tilde_dir_shuffle, C_conf, sample_ids
):
    arrays = (
        _readonly(correct, dtype=np.int64),
        _readonly(U_cycle, dtype=np.float64),
        _readonly(U_tilde_dir, dtype=np.float64),
        _readonly(U_tilde_dir_shuffle, dtype=np.float64),
        _readonly(C_conf, dtype=np.float64),
    )
    ids = _readonly(sample_ids, dtype=np.int64)
    expected = (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
    _require(
        all(value.shape == expected for value in arrays)
        and ids.shape == (UNLABELED_EVAL_COUNT,)
        and np.array_equal(
            ids, np.setdiff1d(canonical_sample_ids(), fixed_labeled_ids())
        ),
        "C2-C0 calibration evaluation must use fixed 1386 unlabeled samples",
    )
    records = [
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
        "directions": records,
    }


def _mean(records, field):
    if not records:
        return None
    values = np.asarray([record[field] for record in records], dtype=np.float64)
    _require(np.isfinite(values).all(), "C2-C0 non-finite seed metric")
    return float(values.mean())


def build_seed_summary(seed, directional_metrics, calibration_metrics):
    active_seed = validate_seed(seed)
    diagnostic = directional_metrics.get("directions", ())
    calibration = calibration_metrics.get("directions", ())
    _require(
        len(diagnostic) == len(calibration) == DIRECTION_COUNT
        and all(
            diagnostic[index].get("direction_id")
            == calibration[index].get("direction_id") == index
            for index in range(DIRECTION_COUNT)
        ),
        "C2-C0 seed direction schema mismatch",
    )
    valid_diagnostic = [
        record for record in diagnostic if record["diagnostic_valid"]
    ]
    valid_calibration = [
        record for record in calibration if record["direction_valid"]
    ]
    deltas = np.asarray(
        [record["DeltaCalAUC_dir"] for record in valid_calibration],
        dtype=np.float64,
    )
    calibration_p = one_sided_wilcoxon_greater(deltas)
    mean_delta = _mean(valid_calibration, "DeltaCalAUC_dir")
    mean_calibration_gap = _mean(
        valid_calibration, "TrueVsShuffleCalGap_dir"
    )
    conditions = OrderedDict((
        (
            "minimum_valid_calibration_directions",
            len(valid_calibration) >= MIN_VALID_DIRECTIONS_PER_SEED,
        ),
        (
            "minimum_positive_DeltaCalAUC_dir_directions",
            int(np.count_nonzero(deltas > 0.0)) >= POSITIVE_DIRECTION_MIN_COUNT,
        ),
        ("positive_mean_DeltaCalAUC_dir", mean_delta is not None and mean_delta > 0.0),
        (
            "one_sided_wilcoxon_DeltaCalAUC_dir_greater",
            calibration_p is not None and calibration_p < SIGNIFICANCE_LEVEL,
        ),
        (
            "positive_mean_TrueVsShuffleCalGap_dir",
            mean_calibration_gap is not None and mean_calibration_gap > 0.0,
        ),
    ))
    passed = bool(all(conditions.values()))
    return {
        "stage": STAGE,
        "seed": active_seed,
        "label_count": LABEL_COUNT,
        "unlabeled_evaluation_count": UNLABELED_EVAL_COUNT,
        "valid_directional_diagnostic_count": len(valid_diagnostic),
        "mean_directional_sign_agreement_true": _mean(
            valid_diagnostic, "directional_sign_agreement_true"
        ),
        "mean_directional_sign_agreement_shuffle": _mean(
            valid_diagnostic, "directional_sign_agreement_shuffle"
        ),
        "mean_DirectionalSpecificityGap": _mean(
            valid_diagnostic, "DirectionalSpecificityGap"
        ),
        "valid_calibration_direction_count": len(valid_calibration),
        "positive_DeltaCalAUC_dir_direction_count": int(
            np.count_nonzero(deltas > 0.0)
        ),
        "mean_DeltaCalAUC_dir": mean_delta,
        "mean_TrueVsShuffleCalGap_dir": mean_calibration_gap,
        "calibration_wilcoxon_greater_p": calibration_p,
        "wilcoxon_alternative": "greater",
        "CalibrationGate_conditions": dict(conditions),
        "DirectionalDiagnostic_used_in_gate": False,
        "CalibrationGate": passed,
        "C2_C0_SEED_PASS": passed,
    }


def _mean_or_none(values):
    if any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=np.float64)
    _require(np.isfinite(array).all(), "C2-C0 non-finite multi-seed metric")
    return float(array.mean())


def build_multiseed_decision(seed_summaries):
    _require(
        tuple(seed_summaries) == SEEDS
        and all(seed_summaries[seed].get("seed") == seed for seed in SEEDS),
        "C2-C0 multi-seed set/order mismatch",
    )
    pass_count = sum(
        bool(seed_summaries[seed]["C2_C0_SEED_PASS"]) for seed in SEEDS
    )
    aggregate_delta = _mean_or_none(
        [seed_summaries[seed]["mean_DeltaCalAUC_dir"] for seed in SEEDS]
    )
    aggregate_gap = _mean_or_none([
        seed_summaries[seed]["mean_TrueVsShuffleCalGap_dir"] for seed in SEEDS
    ])
    aggregate_directional_gap = _mean_or_none([
        seed_summaries[seed]["mean_DirectionalSpecificityGap"] for seed in SEEDS
    ])
    conditions = OrderedDict((
        ("at_least_two_of_three_seed_passes", pass_count >= SEED_PASS_MIN_COUNT),
        (
            "positive_aggregate_mean_DeltaCalAUC_dir",
            aggregate_delta is not None and aggregate_delta > 0.0,
        ),
        (
            "positive_aggregate_mean_TrueVsShuffleCalGap_dir",
            aggregate_gap is not None and aggregate_gap > 0.0,
        ),
    ))
    passed = bool(all(conditions.values()))
    return {
        "summary": {
            "stage": STAGE,
            "seeds": list(SEEDS),
            "seed_pass_count": pass_count,
            "aggregate_mean_DeltaCalAUC_dir": aggregate_delta,
            "aggregate_mean_TrueVsShuffleCalGap_dir": aggregate_gap,
            "aggregate_mean_DirectionalSpecificityGap": aggregate_directional_gap,
            "seed_summaries": {
                str(seed): seed_summaries[seed] for seed in SEEDS
            },
        },
        "decision": {
            "decision_conditions": dict(conditions),
            "final_decision": FINAL_DECISIONS[0] if passed else FINAL_DECISIONS[1],
            "C2_C0_DIRECTIONAL_CONSENSUS_UTILITY_CALIBRATION_PASS": passed,
            "DirectionalDiagnostic_used_in_gate": False,
            "post_hoc_gate_change": False,
        },
    }
