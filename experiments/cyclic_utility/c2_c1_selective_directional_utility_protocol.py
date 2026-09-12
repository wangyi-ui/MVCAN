"""Pure-array protocol for C2-C1 selective directional utility admission.

C2-C1 reproduces the frozen C2-C0 directional pipeline exactly, then changes
only whether an already-defined C2-C0 action is executed.  Admission is the
parameter-free unanimous stability of the full direction under deletion of
each one of the fourteen propagation anchors.  This module performs no I/O,
ground-truth loading, model execution, or training.
"""

from collections import OrderedDict

import numpy as np

from experiments.cyclic_utility import c2_c0_directional_consensus_utility_protocol as c0


STAGE = "C2-C1"
SEEDS = c0.SEEDS
SAMPLE_NUM = c0.SAMPLE_NUM
CLASS_NUM = c0.CLASS_NUM
DIRECTION_COUNT = c0.DIRECTION_COUNT
LABEL_COUNT = c0.LABEL_COUNT
UNLABELED_EVAL_COUNT = c0.UNLABELED_EVAL_COUNT
CONFIDENCE_BIN_COUNT = c0.CONFIDENCE_BIN_COUNT
MIN_VALID_BINS_PER_DIRECTION = c0.MIN_VALID_BINS_PER_DIRECTION
MIN_VALID_DIRECTIONS_PER_SEED = c0.MIN_VALID_DIRECTIONS_PER_SEED
POSITIVE_DIRECTION_MIN_COUNT = c0.POSITIVE_DIRECTION_MIN_COUNT
SEED_PASS_MIN_COUNT = c0.SEED_PASS_MIN_COUNT
SIGNIFICANCE_LEVEL = c0.SIGNIFICANCE_LEVEL
ZERO_SUPPORT_EPSILON = c0.ZERO_SUPPORT_EPSILON

FIXED_LABELED_IDS = c0.FIXED_LABELED_IDS
FIXED_LABELED_TARGETS = c0.FIXED_LABELED_TARGETS
FIXED_SHUFFLED_TARGETS = c0.FIXED_SHUFFLED_TARGETS

FINAL_DECISIONS = (
    "C2_C1_SELECTIVE_DIRECTIONAL_UTILITY_ACTION_ADMISSION_PASS",
    "C2_C1_SELECTIVE_DIRECTIONAL_UTILITY_ACTION_ADMISSION_FAIL",
)

# Frozen C2-C0/B0 primitives are exposed as aliases rather than duplicated.
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
confidence_equal_count_strata = c0.confidence_equal_count_strata
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
        raise ValueError("C2-C1 seed must be one of " + str(SEEDS))
    return value


def unanimous_jackknife_admission(D, q_minus):
    """Build strict admission from D:[N,S] and q_minus:[N,S,L]."""
    full = _readonly(D, dtype=np.float64)
    jackknife_signs = _readonly(q_minus)
    _require(
        full.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and jackknife_signs.shape
        == (SAMPLE_NUM, DIRECTION_COUNT, LABEL_COUNT)
        and np.isfinite(full).all()
        and np.all((full >= -1.0) & (full <= 1.0))
        and np.all(
            (jackknife_signs == -1)
            | (jackknife_signs == 0)
            | (jackknife_signs == 1)
        ),
        "C2-C1 unanimous-admission input boundary mismatch",
    )
    # q:[N,S], q_minus:[N,S,L], A:[N,S].
    full_sign = np.ascontiguousarray(np.sign(full), dtype=np.int8)
    stable = np.all(
        jackknife_signs == full_sign[:, :, None], axis=2
    )
    admission = np.ascontiguousarray(
        (full_sign != 0) & stable, dtype=np.int8
    )
    _require(
        np.all((full_sign == -1) | (full_sign == 0) | (full_sign == 1))
        and np.all((admission == 0) | (admission == 1))
        and np.all(admission[full_sign == 0] == 0),
        "C2-C1 unanimous-admission construction failed",
    )
    full_sign.setflags(write=False)
    admission.setflags(write=False)
    return {"q": full_sign, "A": admission}


def canonicalize_jackknife_consensus(raw_D_minus, denominator_minus):
    """Canonicalize float64 roundoff for D_minus:[N,S,L], and nothing else."""
    raw_consensus = _readonly(raw_D_minus, dtype=np.float64)
    denominator = _readonly(denominator_minus, dtype=np.float64)
    expected = (SAMPLE_NUM, DIRECTION_COUNT, LABEL_COUNT)
    unsupported = denominator <= ZERO_SUPPORT_EPSILON
    float64_epsilon = np.finfo(np.float64).eps
    bound_tolerance = 32.0 * float64_epsilon
    _require(
        raw_consensus.shape == denominator.shape == expected
        and np.isfinite(raw_consensus).all()
        and np.isfinite(denominator).all()
        and np.all(raw_consensus[unsupported] == 0.0)
        and np.all(raw_consensus >= -1.0 - bound_tolerance)
        and np.all(raw_consensus <= 1.0 + bound_tolerance),
        "C2-C1 raw jackknife consensus exceeds float64 numerical tolerance",
    )
    consensus = np.ascontiguousarray(
        np.clip(raw_consensus, -1.0, 1.0), dtype=np.float64
    )
    _require(
        np.isfinite(consensus).all()
        and np.all((consensus >= -1.0) & (consensus <= 1.0))
        and np.all(consensus[unsupported] == 0.0),
        "C2-C1 canonical jackknife consensus boundary mismatch",
    )
    _require(
        np.array_equal(np.sign(raw_consensus), np.sign(consensus)),
        "C2-C1 jackknife numerical canonicalization changed a direction",
    )
    consensus.setflags(write=False)
    return consensus


def build_jackknife_directional_admission(weights, d_L, D):
    """Delete each final propagation anchor without refitting semantic LOO.

    Shapes:
      weights: [N,L,S] = [1400,14,20]
      d_L: [L,S] = [14,20]
      D: [N,S] = [1400,20]
      D_minus/q_minus: [N,S,L] = [1400,20,14]
      A: [N,S] = [1400,20]
    """
    propagation_weights = _readonly(weights, dtype=np.float64)
    labeled_directions = _readonly(d_L, dtype=np.float64)
    full_consensus = _readonly(D, dtype=np.float64)
    _require(
        propagation_weights.shape
        == (SAMPLE_NUM, LABEL_COUNT, DIRECTION_COUNT)
        and labeled_directions.shape == (LABEL_COUNT, DIRECTION_COUNT)
        and full_consensus.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and np.isfinite(propagation_weights).all()
        and np.all(propagation_weights >= 0.0)
        and np.all(
            (labeled_directions == -1.0)
            | (labeled_directions == 0.0)
            | (labeled_directions == 1.0)
        )
        and np.isfinite(full_consensus).all()
        and np.all((full_consensus >= -1.0) & (full_consensus <= 1.0)),
        "C2-C1 jackknife input boundary mismatch",
    )

    # contribution:[N,L,S], full numerator/denominator:[N,S].
    contribution = propagation_weights * labeled_directions[None, :, :]
    full_numerator = np.ascontiguousarray(contribution.sum(axis=1))
    full_denominator = np.ascontiguousarray(propagation_weights.sum(axis=1))
    reproduced = np.divide(
        full_numerator,
        full_denominator,
        out=np.zeros((SAMPLE_NUM, DIRECTION_COUNT), dtype=np.float64),
        where=full_denominator > ZERO_SUPPORT_EPSILON,
    )
    _require(
        np.array_equal(reproduced, full_consensus),
        "C2-C1 did not reproduce frozen C2-C0 full D exactly",
    )

    # First compute [N,L,S], then transpose once to the registered [N,S,L].
    numerator_minus_nls = full_numerator[:, None, :] - contribution
    denominator_minus_nls = (
        full_denominator[:, None, :] - propagation_weights
    )
    numerator_minus = np.ascontiguousarray(
        np.transpose(numerator_minus_nls, (0, 2, 1)), dtype=np.float64
    )
    denominator_minus = np.ascontiguousarray(
        np.transpose(denominator_minus_nls, (0, 2, 1)), dtype=np.float64
    )
    raw_consensus_minus = np.divide(
        numerator_minus,
        denominator_minus,
        out=np.zeros(
            (SAMPLE_NUM, DIRECTION_COUNT, LABEL_COUNT), dtype=np.float64
        ),
        where=denominator_minus > ZERO_SUPPORT_EPSILON,
    )
    consensus_minus = canonicalize_jackknife_consensus(
        raw_consensus_minus, denominator_minus
    )
    jackknife_signs = np.ascontiguousarray(
        np.sign(consensus_minus), dtype=np.int8
    )
    selection = unanimous_jackknife_admission(
        full_consensus, jackknife_signs
    )
    for value in (
        contribution,
        full_numerator,
        full_denominator,
        numerator_minus,
        denominator_minus,
        consensus_minus,
        jackknife_signs,
    ):
        value.setflags(write=False)
    return {
        "anchor_contribution": contribution,
        "num": full_numerator,
        "denom": full_denominator,
        "num_minus": numerator_minus,
        "denom_minus": denominator_minus,
        "D_minus": consensus_minus,
        "q": selection["q"],
        "q_minus": jackknife_signs,
        "A": selection["A"],
        "semantic_LOO_recomputed_for_jackknife": False,
    }


def selective_directional_utility_update(U_cycle, D, A):
    """Apply U + A*U*(1-U)*D for [N,S] arrays, preserving C0 magnitude."""
    cycle = _readonly(U_cycle, dtype=np.float64)
    consensus = _readonly(D, dtype=np.float64)
    admission = _readonly(A)
    _require(
        cycle.shape == consensus.shape == admission.shape
        == (SAMPLE_NUM, DIRECTION_COUNT)
        and np.isfinite(cycle).all()
        and np.isfinite(consensus).all()
        and np.isfinite(admission).all()
        and np.all((cycle >= 0.0) & (cycle <= 1.0))
        and np.all((consensus >= -1.0) & (consensus <= 1.0))
        and np.all((admission == 0) | (admission == 1)),
        "C2-C1 selective update input boundary mismatch",
    )
    selected = cycle + admission * cycle * (1.0 - cycle) * consensus
    selected[cycle == 0.0] = 0.0
    c0_baseline = c0.directional_utility_update(cycle, consensus)
    _require(
        np.isfinite(selected).all()
        and np.all((selected >= 0.0) & (selected <= 1.0))
        and np.array_equal(selected[admission == 0], cycle[admission == 0])
        and np.array_equal(
            selected[admission == 1], c0_baseline[admission == 1]
        )
        and np.all(selected[cycle == 0.0] == 0.0),
        "C2-C1 selective utility update failed",
    )
    selected.setflags(write=False)
    return selected


def build_selective_directional_outputs(
    y_gen, U_cycle, labeled_ids, labeled_targets, shuffled_targets
):
    """Rebuild true and shuffled C0 paths, then admit each independently."""
    c0_outputs = c0.build_directional_consensus_outputs(
        y_gen, U_cycle, labeled_ids, labeled_targets, shuffled_targets
    )
    true_jackknife = build_jackknife_directional_admission(
        c0_outputs["weights"], c0_outputs["d_L"], c0_outputs["D"]
    )
    shuffle_jackknife = build_jackknife_directional_admission(
        c0_outputs["shuffle_weights"],
        c0_outputs["d_L_shuffle"],
        c0_outputs["D_shuffle"],
    )
    selected = selective_directional_utility_update(
        U_cycle, c0_outputs["D"], true_jackknife["A"]
    )
    selected_shuffle = selective_directional_utility_update(
        U_cycle, c0_outputs["D_shuffle"], shuffle_jackknife["A"]
    )
    return {
        "R": c0_outputs["R"],
        "z_L": c0_outputs["z_L"],
        "e_L": c0_outputs["e_L"],
        "d_L": c0_outputs["d_L"],
        "weights": c0_outputs["weights"],
        "D": c0_outputs["D"],
        "q": true_jackknife["q"],
        "q_minus": true_jackknife["q_minus"],
        "D_minus": true_jackknife["D_minus"],
        "num": true_jackknife["num"],
        "denom": true_jackknife["denom"],
        "num_minus": true_jackknife["num_minus"],
        "denom_minus": true_jackknife["denom_minus"],
        "A": true_jackknife["A"],
        "U_tilde_dir_c0": c0_outputs["U_tilde_dir"],
        "U_tilde_select": selected,
        "loo_mappings": c0_outputs["loo_mappings"],
        "loo_soft_contingency": c0_outputs["loo_soft_contingency"],
        "loo_included_label_mask": c0_outputs["loo_included_label_mask"],
        "z_L_shuffle": c0_outputs["z_L_shuffle"],
        "e_L_shuffle": c0_outputs["e_L_shuffle"],
        "d_L_shuffle": c0_outputs["d_L_shuffle"],
        "shuffle_weights": c0_outputs["shuffle_weights"],
        "D_shuffle": c0_outputs["D_shuffle"],
        "q_shuffle": shuffle_jackknife["q"],
        "q_minus_shuffle": shuffle_jackknife["q_minus"],
        "D_minus_shuffle": shuffle_jackknife["D_minus"],
        "num_shuffle": shuffle_jackknife["num"],
        "denom_shuffle": shuffle_jackknife["denom"],
        "num_minus_shuffle": shuffle_jackknife["num_minus"],
        "denom_minus_shuffle": shuffle_jackknife["denom_minus"],
        "A_shuffle": shuffle_jackknife["A"],
        "U_tilde_dir_c0_shuffle": c0_outputs["U_tilde_dir_shuffle"],
        "U_tilde_select_shuffle": selected_shuffle,
        "loo_mappings_shuffle": c0_outputs["loo_mappings_shuffle"],
        "loo_soft_contingency_shuffle": c0_outputs[
            "loo_soft_contingency_shuffle"
        ],
        "loo_included_label_mask_shuffle": c0_outputs[
            "loo_included_label_mask_shuffle"
        ],
        "C2_C0_D_reproduced_exactly": True,
        "C2_C0_utility_reproduced_exactly": True,
        "shuffle_recomputed_from_targets": c0_outputs[
            "shuffle_recomputed_from_targets"
        ],
        "A_shuffle_independently_recomputed": True,
        "true_A_reused_for_shuffle": False,
        "semantic_LOO_recomputed_for_jackknife": False,
        "C_conf_used_for_admission": False,
    }


def analyze_admission_directions(
    A, A_shuffle, D, D_shuffle, e_GT, sample_ids
):
    """Post-seal admission diagnostics on fixed [1386,20] unlabeled arrays."""
    admission = _readonly(A)
    admission_shuffle = _readonly(A_shuffle)
    consensus = _readonly(D, dtype=np.float64)
    consensus_shuffle = _readonly(D_shuffle, dtype=np.float64)
    truth = _readonly(e_GT, dtype=np.float64)
    ids = _readonly(sample_ids, dtype=np.int64)
    expected = (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
    _require(
        admission.shape == admission_shuffle.shape == consensus.shape
        == consensus_shuffle.shape == truth.shape == expected
        and ids.shape == (UNLABELED_EVAL_COUNT,)
        and np.array_equal(
            ids, np.setdiff1d(canonical_sample_ids(), fixed_labeled_ids())
        )
        and np.all((admission == 0) | (admission == 1))
        and np.all((admission_shuffle == 0) | (admission_shuffle == 1))
        and all(
            np.isfinite(value).all()
            for value in (consensus, consensus_shuffle, truth)
        ),
        "C2-C1 admission diagnostic must use exactly 1386 unlabeled samples",
    )
    records = []
    for direction_id in range(DIRECTION_COUNT):
        true_admitted = admission[:, direction_id] == 1
        shuffle_admitted = admission_shuffle[:, direction_id] == 1
        e_direction = truth[:, direction_id]
        d_direction = consensus[:, direction_id]
        d_shuffle_direction = consensus_shuffle[:, direction_id]
        true_eligible = (
            true_admitted & (d_direction != 0.0) & (e_direction != 0.0)
        )
        shuffle_eligible = (
            shuffle_admitted
            & (d_shuffle_direction != 0.0)
            & (e_direction != 0.0)
        )
        true_eligible_count = int(np.count_nonzero(true_eligible))
        shuffle_eligible_count = int(np.count_nonzero(shuffle_eligible))
        true_agreement = (
            float(np.mean(
                np.sign(d_direction[true_eligible])
                == np.sign(e_direction[true_eligible])
            ))
            if true_eligible_count
            else None
        )
        shuffle_agreement = (
            float(np.mean(
                np.sign(d_shuffle_direction[shuffle_eligible])
                == np.sign(e_direction[shuffle_eligible])
            ))
            if shuffle_eligible_count
            else None
        )
        valid = true_agreement is not None and shuffle_agreement is not None
        true_count = int(np.count_nonzero(true_admitted))
        shuffle_count = int(np.count_nonzero(shuffle_admitted))
        records.append({
            "direction_id": direction_id,
            "admission_count_true": true_count,
            "admission_rate_true": true_count / UNLABELED_EVAL_COUNT,
            "admission_count_shuffle": shuffle_count,
            "admission_rate_shuffle": shuffle_count / UNLABELED_EVAL_COUNT,
            "admitted_agreement_eligible_count_true": true_eligible_count,
            "admitted_agreement_eligible_count_shuffle": shuffle_eligible_count,
            "AdmittedDirectionalAgreement_true": true_agreement,
            "AdmittedDirectionalAgreement_shuffle": shuffle_agreement,
            "AdmittedDirectionalSpecificityGap": (
                true_agreement - shuffle_agreement if valid else None
            ),
            "diagnostic_valid": valid,
        })
    return {
        "direction_count": DIRECTION_COUNT,
        "evaluation_sample_count": UNLABELED_EVAL_COUNT,
        "evaluation_only_unlabeled": True,
        "GT_used_to_construct_admission": False,
        "used_in_seed_gate": False,
        "directions": records,
    }


def _one_nondegeneracy_record(A):
    admission = np.asarray(A)
    counts = np.count_nonzero(admission, axis=0)
    total = int(admission.size)
    admitted = int(np.count_nonzero(admission))
    return {
        "total_action_count": total,
        "total_admitted_action_count": admitted,
        "overall_admission_rate": admitted / total,
        "directions_with_at_least_one_admitted_action": int(
            np.count_nonzero(counts > 0)
        ),
        "directions_with_both_admitted_and_abstained_actions": int(
            np.count_nonzero((counts > 0) & (counts < admission.shape[0]))
        ),
        "all_zero_admission": admitted == 0,
    }


def analyze_admission_nondegeneracy(A, A_shuffle, sample_ids):
    admission = _readonly(A)
    admission_shuffle = _readonly(A_shuffle)
    ids = _readonly(sample_ids, dtype=np.int64)
    expected = (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
    _require(
        admission.shape == admission_shuffle.shape == expected
        and ids.shape == (UNLABELED_EVAL_COUNT,)
        and np.array_equal(
            ids, np.setdiff1d(canonical_sample_ids(), fixed_labeled_ids())
        )
        and np.all((admission == 0) | (admission == 1))
        and np.all((admission_shuffle == 0) | (admission_shuffle == 1)),
        "C2-C1 non-degeneracy audit input boundary mismatch",
    )
    return {
        "true": _one_nondegeneracy_record(admission),
        "shuffle": _one_nondegeneracy_record(admission_shuffle),
        "admission_rate_used_in_gate": False,
    }


def analyze_calibration_direction(
    correct, U_cycle, U_tilde_dir_c0, U_tilde_select,
    U_tilde_select_shuffle, C_conf, sample_ids, direction_id,
):
    """Apply the exact C0 confidence strata/validity to all four arms."""
    # First frozen call evaluates Ucycle/C0/C1; the second evaluates
    # Ucycle/C1/shuffled-C1.  Strata and valid masks must match exactly.
    c0_and_select = c0.analyze_calibration_direction(
        correct, U_cycle, U_tilde_dir_c0, U_tilde_select,
        C_conf, sample_ids, direction_id,
    )
    select_and_shuffle = c0.analyze_calibration_direction(
        correct, U_cycle, U_tilde_select, U_tilde_select_shuffle,
        C_conf, sample_ids, direction_id,
    )
    _require(
        c0_and_select["valid_bin_mask"]
        == select_and_shuffle["valid_bin_mask"]
        and c0_and_select["direction_valid"]
        == select_and_shuffle["direction_valid"]
        and c0_and_select["confidence_strata"]
        == select_and_shuffle["confidence_strata"],
        "C2-C1 utility arms did not share an identical confidence-bin mask",
    )
    bins = []
    for first, second in zip(
        c0_and_select["bins"], select_and_shuffle["bins"]
    ):
        _require(
            first["valid_both_correctness_classes"]
            == second["valid_both_correctness_classes"]
            and first["AUC_Ucycle"] == second["AUC_Ucycle"]
            and first["AUC_U_tilde_dir_shuffle"] == second["AUC_U_tilde_dir"],
            "C2-C1 confidence-bin metric alignment mismatch",
        )
        bins.append({
            "bin_id": first["bin_id"],
            "sample_count": first["sample_count"],
            "valid_both_correctness_classes": first[
                "valid_both_correctness_classes"
            ],
            "AUC_Ucycle": first["AUC_Ucycle"],
            "AUC_C0_dir": first["AUC_U_tilde_dir"],
            "AUC_C1_select": second["AUC_U_tilde_dir"],
            "AUC_C1_select_shuffle": second["AUC_U_tilde_dir_shuffle"],
        })
    valid = c0_and_select["direction_valid"]
    baseline_auc = c0_and_select["CondAUC_Ucycle"]
    c0_auc = c0_and_select["CondAUC_U_tilde_dir"]
    select_auc = select_and_shuffle["CondAUC_U_tilde_dir"]
    shuffle_auc = select_and_shuffle["CondAUC_U_tilde_dir_shuffle"]
    return {
        "direction_id": int(direction_id),
        "valid_bin_count": c0_and_select["valid_bin_count"],
        "valid_bin_mask": c0_and_select["valid_bin_mask"],
        "same_valid_bin_mask_all_utility_arms": True,
        "direction_valid": valid,
        "bins": bins,
        "CondAUC_Ucycle": baseline_auc,
        "CondAUC_C0_dir": c0_auc,
        "CondAUC_C1_select": select_auc,
        "CondAUC_C1_select_shuffle": shuffle_auc,
        "DeltaCalAUC_select": (
            select_auc - baseline_auc if valid else None
        ),
        "SelectiveGainVsC0": select_auc - c0_auc if valid else None,
        "TrueVsShuffleCalGap_select": (
            select_auc - shuffle_auc if valid else None
        ),
        "confidence_strata": c0_and_select["confidence_strata"],
    }


def analyze_calibration_directions(
    correct, U_cycle, U_tilde_dir_c0, U_tilde_select,
    U_tilde_select_shuffle, C_conf, sample_ids,
):
    arrays = (
        _readonly(correct, dtype=np.int64),
        _readonly(U_cycle, dtype=np.float64),
        _readonly(U_tilde_dir_c0, dtype=np.float64),
        _readonly(U_tilde_select, dtype=np.float64),
        _readonly(U_tilde_select_shuffle, dtype=np.float64),
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
        "C2-C1 calibration evaluation must use fixed 1386 unlabeled samples",
    )
    records = [
        analyze_calibration_direction(
            arrays[0][:, direction_id],
            arrays[1][:, direction_id],
            arrays[2][:, direction_id],
            arrays[3][:, direction_id],
            arrays[4][:, direction_id],
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
        "GT_used_for_confidence_stratification": False,
        "directions": records,
    }


def _mean(records, field):
    if not records:
        return None
    values = np.asarray([record[field] for record in records], dtype=np.float64)
    _require(np.isfinite(values).all(), "C2-C1 non-finite seed metric")
    return float(values.mean())


def build_seed_summary(seed, admission_metrics, calibration_metrics):
    active_seed = validate_seed(seed)
    diagnostics = admission_metrics.get("directions", ())
    calibration = calibration_metrics.get("directions", ())
    _require(
        len(diagnostics) == len(calibration) == DIRECTION_COUNT
        and all(
            diagnostics[index].get("direction_id")
            == calibration[index].get("direction_id") == index
            for index in range(DIRECTION_COUNT)
        ),
        "C2-C1 seed direction schema mismatch",
    )
    valid_diagnostics = [
        record for record in diagnostics if record["diagnostic_valid"]
    ]
    valid_calibration = [
        record for record in calibration if record["direction_valid"]
    ]
    deltas = np.asarray(
        [record["DeltaCalAUC_select"] for record in valid_calibration],
        dtype=np.float64,
    )
    gains = np.asarray(
        [record["SelectiveGainVsC0"] for record in valid_calibration],
        dtype=np.float64,
    )
    delta_p = one_sided_wilcoxon_greater(deltas)
    gain_p = one_sided_wilcoxon_greater(gains)
    mean_delta = _mean(valid_calibration, "DeltaCalAUC_select")
    mean_gain = _mean(valid_calibration, "SelectiveGainVsC0")
    mean_label_gap = _mean(
        valid_calibration, "TrueVsShuffleCalGap_select"
    )
    conditions = OrderedDict((
        (
            "minimum_valid_calibration_directions",
            len(valid_calibration) >= MIN_VALID_DIRECTIONS_PER_SEED,
        ),
        (
            "minimum_positive_DeltaCalAUC_select_directions",
            int(np.count_nonzero(deltas > 0.0)) >= POSITIVE_DIRECTION_MIN_COUNT,
        ),
        (
            "positive_mean_DeltaCalAUC_select",
            mean_delta is not None and mean_delta > 0.0,
        ),
        (
            "one_sided_wilcoxon_DeltaCalAUC_select_greater",
            delta_p is not None and delta_p < SIGNIFICANCE_LEVEL,
        ),
        (
            "minimum_positive_SelectiveGainVsC0_directions",
            int(np.count_nonzero(gains > 0.0)) >= POSITIVE_DIRECTION_MIN_COUNT,
        ),
        (
            "positive_mean_SelectiveGainVsC0",
            mean_gain is not None and mean_gain > 0.0,
        ),
        (
            "one_sided_wilcoxon_SelectiveGainVsC0_greater",
            gain_p is not None and gain_p < SIGNIFICANCE_LEVEL,
        ),
        (
            "positive_mean_TrueVsShuffleCalGap_select",
            mean_label_gap is not None and mean_label_gap > 0.0,
        ),
    ))
    passed = bool(all(conditions.values()))
    return {
        "stage": STAGE,
        "seed": active_seed,
        "label_count": LABEL_COUNT,
        "unlabeled_evaluation_count": UNLABELED_EVAL_COUNT,
        "total_admission_count_true": int(sum(
            record["admission_count_true"] for record in diagnostics
        )),
        "overall_admission_rate_true": _mean(
            diagnostics, "admission_rate_true"
        ),
        "total_admission_count_shuffle": int(sum(
            record["admission_count_shuffle"] for record in diagnostics
        )),
        "overall_admission_rate_shuffle": _mean(
            diagnostics, "admission_rate_shuffle"
        ),
        "valid_admitted_directional_diagnostic_count": len(valid_diagnostics),
        "mean_AdmittedDirectionalAgreement_true": _mean(
            valid_diagnostics, "AdmittedDirectionalAgreement_true"
        ),
        "mean_AdmittedDirectionalAgreement_shuffle": _mean(
            valid_diagnostics, "AdmittedDirectionalAgreement_shuffle"
        ),
        "mean_AdmittedDirectionalSpecificityGap": _mean(
            valid_diagnostics, "AdmittedDirectionalSpecificityGap"
        ),
        "valid_calibration_direction_count": len(valid_calibration),
        "positive_DeltaCalAUC_select_direction_count": int(
            np.count_nonzero(deltas > 0.0)
        ),
        "mean_DeltaCalAUC_select": mean_delta,
        "DeltaCalAUC_select_wilcoxon_greater_p": delta_p,
        "positive_SelectiveGainVsC0_direction_count": int(
            np.count_nonzero(gains > 0.0)
        ),
        "mean_SelectiveGainVsC0": mean_gain,
        "SelectiveGainVsC0_wilcoxon_greater_p": gain_p,
        "mean_TrueVsShuffleCalGap_select": mean_label_gap,
        "wilcoxon_alternative": "greater",
        "SeedGate_conditions": dict(conditions),
        "admission_diagnostics_used_in_gate": False,
        "C2_C1_SEED_PASS": passed,
    }


def _mean_or_none(values):
    if any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=np.float64)
    _require(np.isfinite(array).all(), "C2-C1 non-finite multi-seed metric")
    return float(array.mean())


def build_multiseed_decision(seed_summaries):
    _require(
        tuple(seed_summaries) == SEEDS
        and all(seed_summaries[seed].get("seed") == seed for seed in SEEDS),
        "C2-C1 multi-seed set/order mismatch",
    )
    pass_count = sum(
        bool(seed_summaries[seed]["C2_C1_SEED_PASS"]) for seed in SEEDS
    )
    aggregate_delta = _mean_or_none([
        seed_summaries[seed]["mean_DeltaCalAUC_select"] for seed in SEEDS
    ])
    aggregate_gain = _mean_or_none([
        seed_summaries[seed]["mean_SelectiveGainVsC0"] for seed in SEEDS
    ])
    aggregate_gap = _mean_or_none([
        seed_summaries[seed]["mean_TrueVsShuffleCalGap_select"]
        for seed in SEEDS
    ])
    conditions = OrderedDict((
        ("at_least_two_of_three_seed_passes", pass_count >= SEED_PASS_MIN_COUNT),
        (
            "positive_aggregate_mean_DeltaCalAUC_select",
            aggregate_delta is not None and aggregate_delta > 0.0,
        ),
        (
            "positive_aggregate_mean_SelectiveGainVsC0",
            aggregate_gain is not None and aggregate_gain > 0.0,
        ),
        (
            "positive_aggregate_mean_TrueVsShuffleCalGap_select",
            aggregate_gap is not None and aggregate_gap > 0.0,
        ),
    ))
    passed = bool(all(conditions.values()))
    return {
        "summary": {
            "stage": STAGE,
            "seeds": list(SEEDS),
            "seed_pass_count": pass_count,
            "aggregate_mean_DeltaCalAUC_select": aggregate_delta,
            "aggregate_mean_SelectiveGainVsC0": aggregate_gain,
            "aggregate_mean_TrueVsShuffleCalGap_select": aggregate_gap,
            "seed_summaries": {
                str(seed): seed_summaries[seed] for seed in SEEDS
            },
        },
        "decision": {
            "decision_conditions": dict(conditions),
            "final_decision": FINAL_DECISIONS[0] if passed else FINAL_DECISIONS[1],
            "C2_C1_SELECTIVE_DIRECTIONAL_UTILITY_ACTION_ADMISSION_PASS": passed,
            "admission_diagnostics_used_in_gate": False,
            "post_hoc_gate_change": False,
        },
    }
