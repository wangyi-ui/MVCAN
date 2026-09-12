"""Frozen pure-array protocol for C2-A0 sparse-label utility calibration.

The fourteen labels calibrate already-frozen C0 action utilities.  This module
contains no artifact I/O, model execution, trainable mechanism, or ground-truth
loader.  ``C_conf`` is deliberately absent from every calibration API and is
used only by the post-seal confidence-stratified diagnostics.
"""

from collections import OrderedDict

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.cyclic_utility import (
    bridge_p0_residual_utility_protocol as bridge,
)
from weak_quality import ndarray_sha256


STAGE = "C2-A0"
SEEDS = (20, 30, 50)
SAMPLE_NUM = 1400
CLASS_NUM = 7
DIRECTION_COUNT = 20
CONFIDENCE_BIN_COUNT = 5
MIN_VALID_BINS_PER_DIRECTION = 3
MIN_VALID_DIRECTIONS_PER_SEED = 16
POSITIVE_DIRECTION_MIN_COUNT = 12
SEED_PASS_MIN_COUNT = 2
ALPHA = 0.05
LABEL_COUNT = 14
UNLABELED_EVAL_COUNT = 1386
EXPECTED_SAMPLE_IDS_LOGICAL_SHA256 = (
    "887cb8dff9918e38f0b0c7f467d3b9aec75dcef9bf821b2307fd076ca5c42237"
)

FIXED_LABELED_IDS = (
    67, 82, 90, 111, 200, 365, 440, 513, 536, 983, 1027, 1250, 1316, 1385,
)
FIXED_LABELED_TARGETS = (4, 6, 2, 3, 0, 4, 1, 5, 6, 3, 0, 2, 5, 1)
FIXED_SHUFFLED_TARGETS = (6, 2, 3, 0, 4, 1, 5, 6, 3, 0, 2, 5, 1, 4)

FORMAL_C2_USES_PGEN = False
FORMAL_C2_USES_YGEN_VOTE_STATE = True
GT_PRESEAL_ISOLATION = True
TRAINING_PATH_PRESENT = False

FINAL_DECISIONS = (
    "C2_A0_SPARSE_LABEL_UTILITY_CALIBRATION_PASS",
    "C2_A0_SPARSE_LABEL_UTILITY_CALIBRATION_FAIL",
)

# Re-export the frozen Bridge statistical primitives used by C2-A0.
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
        raise ValueError("C2-A0 seed must be one of " + str(SEEDS))
    return value


def canonical_sample_ids():
    sample_ids = np.arange(SAMPLE_NUM, dtype=np.int64)
    _require(
        all(int(sample_ids[index]) == index for index in range(SAMPLE_NUM))
        and ndarray_sha256(sample_ids) == EXPECTED_SAMPLE_IDS_LOGICAL_SHA256,
        "C2-A0 canonical sample-ID boundary mismatch",
    )
    sample_ids.setflags(write=False)
    return sample_ids


def fixed_labeled_ids():
    return _readonly(FIXED_LABELED_IDS, dtype=np.int64)


def fixed_labeled_targets():
    return _readonly(FIXED_LABELED_TARGETS, dtype=np.int64)


def fixed_shuffled_targets():
    return _readonly(FIXED_SHUFFLED_TARGETS, dtype=np.int64)


def is_global_class_permutation(original_targets, candidate_targets):
    """Return whether one fixed class permutation explains every pair."""
    original = np.asarray(original_targets, dtype=np.int64)
    candidate = np.asarray(candidate_targets, dtype=np.int64)
    _require(
        original.shape == candidate.shape == (LABEL_COUNT,),
        "C2-A0 permutation-control target shape mismatch",
    )
    mapping = {}
    for source, target in zip(original.tolist(), candidate.tolist()):
        if source in mapping and mapping[source] != target:
            return False
        mapping[source] = target
    return (
        set(mapping) == set(range(CLASS_NUM))
        and set(mapping.values()) == set(range(CLASS_NUM))
    )


def validate_fixed_sparse_split(labeled_ids, unlabeled_ids, labeled_targets):
    labeled = _readonly(labeled_ids, dtype=np.int64)
    unlabeled = _readonly(unlabeled_ids, dtype=np.int64)
    targets = _readonly(labeled_targets, dtype=np.int64)
    canonical = canonical_sample_ids()
    expected_unlabeled = np.setdiff1d(canonical, fixed_labeled_ids())
    _require(
        labeled.shape == (LABEL_COUNT,)
        and unlabeled.shape == (UNLABELED_EVAL_COUNT,)
        and targets.shape == (LABEL_COUNT,)
        and np.array_equal(labeled, fixed_labeled_ids())
        and np.array_equal(targets, fixed_labeled_targets()),
        "C2-A0 fixed fourteen-label split mismatch",
    )
    _require(
        np.array_equal(np.bincount(targets, minlength=CLASS_NUM), np.full(CLASS_NUM, 2))
        and np.array_equal(unlabeled, expected_unlabeled)
        and np.intersect1d(labeled, unlabeled).size == 0
        and np.array_equal(np.sort(np.concatenate((labeled, unlabeled))), canonical),
        "C2-A0 labeled/unlabeled partition mismatch",
    )
    return {
        "labeled_sample_ids": labeled,
        "labeled_targets": targets,
        "unlabeled_ids": unlabeled,
    }


def validate_negative_control(original_targets, shuffled_targets):
    original = _readonly(original_targets, dtype=np.int64)
    shuffled = _readonly(shuffled_targets, dtype=np.int64)
    expected_histogram = np.full(CLASS_NUM, 2, dtype=np.int64)
    _require(
        original.shape == shuffled.shape == (LABEL_COUNT,)
        and np.array_equal(original, fixed_labeled_targets())
        and np.array_equal(shuffled, fixed_shuffled_targets())
        and np.array_equal(np.bincount(original, minlength=CLASS_NUM), expected_histogram)
        and np.array_equal(np.bincount(shuffled, minlength=CLASS_NUM), expected_histogram),
        "C2-A0 fixed negative-control targets mismatch",
    )
    _require(
        not is_global_class_permutation(original, shuffled),
        "C2-A0 negative control is absorbable by a global class permutation",
    )
    return shuffled


def build_cross_direction_vote_state(y_gen):
    """Construct R[i,k] as the frequency of frozen C0 actions across directions."""
    actions = _readonly(y_gen, dtype=np.int64)
    _require(
        actions.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and np.all((actions >= 0) & (actions < CLASS_NUM)),
        "C2-A0 y_gen boundary mismatch",
    )
    one_hot = np.eye(CLASS_NUM, dtype=np.float64)[actions]
    _require(
        one_hot.shape == (SAMPLE_NUM, DIRECTION_COUNT, CLASS_NUM),
        "C2-A0 one-hot action shape mismatch",
    )
    vote_state = np.ascontiguousarray(one_hot.mean(axis=1), dtype=np.float64)
    _require(
        vote_state.shape == (SAMPLE_NUM, CLASS_NUM)
        and np.isfinite(vote_state).all()
        and np.all(vote_state >= 0.0)
        and np.allclose(vote_state.sum(axis=1), 1.0, rtol=0.0, atol=1e-15),
        "C2-A0 semantic vote-state boundary mismatch",
    )
    vote_state.setflags(write=False)
    return vote_state


def build_sparse_anchors_and_mapping(vote_state, labeled_ids, labeled_targets):
    """Fit the one sparse global cluster->class mapping from exactly 14 labels."""
    state = _readonly(vote_state, dtype=np.float64)
    labeled = _readonly(labeled_ids, dtype=np.int64)
    targets = _readonly(labeled_targets, dtype=np.int64)
    _require(
        state.shape == (SAMPLE_NUM, CLASS_NUM)
        and np.isfinite(state).all()
        and np.all(state >= 0.0)
        and np.allclose(state.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)
        and labeled.shape == targets.shape == (LABEL_COUNT,)
        and np.unique(labeled).size == LABEL_COUNT
        and np.all((labeled >= 0) & (labeled < SAMPLE_NUM))
        and np.all((targets >= 0) & (targets < CLASS_NUM))
        and np.array_equal(np.bincount(targets, minlength=CLASS_NUM), np.full(CLASS_NUM, 2)),
        "C2-A0 sparse-anchor input boundary mismatch",
    )
    anchors = np.empty((CLASS_NUM, CLASS_NUM), dtype=np.float64)
    contingency = np.zeros((CLASS_NUM, CLASS_NUM), dtype=np.float64)
    for class_id in range(CLASS_NUM):
        rows = labeled[targets == class_id]
        _require(rows.shape == (2,), "C2-A0 requires exactly two anchors per class")
        anchors[class_id] = state[rows].mean(axis=0)
        contingency[:, class_id] = state[rows].sum(axis=0)
    row_indices, column_indices = linear_sum_assignment(-contingency)
    mapping = np.empty(CLASS_NUM, dtype=np.int64)
    mapping[row_indices] = column_indices
    _require(
        anchors.shape == contingency.shape == (CLASS_NUM, CLASS_NUM)
        and np.isfinite(anchors).all()
        and np.isfinite(contingency).all()
        and np.array_equal(np.sort(mapping), np.arange(CLASS_NUM, dtype=np.int64)),
        "C2-A0 sparse Hungarian mapping boundary mismatch",
    )
    anchors.setflags(write=False)
    contingency.setflags(write=False)
    mapping.setflags(write=False)
    return {
        "anchors": anchors,
        "soft_contingency": contingency,
        "sparse_mapping": mapping,
        "mapping_fit_label_count": LABEL_COUNT,
        "mapping_fit_full_GT_used": False,
        "mapping_fit_bridge_mapping_used": False,
    }


def build_action_label_evidence(vote_state, anchors, sparse_mapping, y_gen):
    state = _readonly(vote_state, dtype=np.float64)
    anchor_values = _readonly(anchors, dtype=np.float64)
    mapping = _readonly(sparse_mapping, dtype=np.int64)
    actions = _readonly(y_gen, dtype=np.int64)
    _require(
        state.shape == (SAMPLE_NUM, CLASS_NUM)
        and anchor_values.shape == (CLASS_NUM, CLASS_NUM)
        and mapping.shape == (CLASS_NUM,)
        and np.array_equal(np.sort(mapping), np.arange(CLASS_NUM, dtype=np.int64))
        and actions.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and np.all((actions >= 0) & (actions < CLASS_NUM))
        and np.isfinite(state).all()
        and np.isfinite(anchor_values).all()
        and np.all(state >= 0.0)
        and np.all(anchor_values >= 0.0),
        "C2-A0 label-evidence input boundary mismatch",
    )
    state_norm = np.linalg.norm(state, axis=1)
    anchor_norm = np.linalg.norm(anchor_values, axis=1)
    denominator = state_norm[:, None] * anchor_norm[None, :]
    similarities = np.divide(
        state @ anchor_values.T,
        denominator,
        out=np.zeros((SAMPLE_NUM, CLASS_NUM), dtype=np.float64),
        where=denominator > 0.0,
    )
    similarities = np.clip(similarities, 0.0, 1.0)
    action_classes = mapping[actions]
    rows = np.arange(SAMPLE_NUM, dtype=np.int64)[:, None]
    g_plus = similarities[rows, action_classes]
    competitors = np.broadcast_to(similarities[:, None, :], (SAMPLE_NUM, DIRECTION_COUNT, CLASS_NUM)).copy()
    np.put_along_axis(competitors, action_classes[:, :, None], -np.inf, axis=2)
    g_minus = competitors.max(axis=2)
    evidence_denominator = g_plus + g_minus
    evidence = np.full((SAMPLE_NUM, DIRECTION_COUNT), 0.5, dtype=np.float64)
    informative = evidence_denominator > 1e-12
    evidence[informative] = g_plus[informative] / evidence_denominator[informative]
    _require(
        similarities.shape == (SAMPLE_NUM, CLASS_NUM)
        and evidence.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and np.isfinite(evidence).all()
        and np.all((evidence >= 0.0) & (evidence <= 1.0)),
        "C2-A0 action-specific label evidence boundary mismatch",
    )
    similarities.setflags(write=False)
    evidence.setflags(write=False)
    return {"similarity": similarities, "E_label": evidence}


def calibrate_utility(U_cycle, E_label):
    cycle = _readonly(U_cycle, dtype=np.float64)
    evidence = _readonly(E_label, dtype=np.float64)
    _require(
        cycle.shape == evidence.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and np.isfinite(cycle).all()
        and np.isfinite(evidence).all()
        and np.all((cycle >= 0.0) & (cycle <= 1.0))
        and np.all((evidence >= 0.0) & (evidence <= 1.0)),
        "C2-A0 utility-calibration input boundary mismatch",
    )
    calibrated = np.zeros_like(cycle, dtype=np.float64)
    active = cycle > 0.0
    numerator = cycle[active] * evidence[active]
    denominator = numerator + (1.0 - cycle[active]) * (1.0 - evidence[active])
    _require(np.all(denominator > 0.0), "C2-A0 calibration denominator is not positive")
    calibrated[active] = numerator / denominator
    neutral = active & (evidence == 0.5)
    calibrated[neutral] = cycle[neutral]
    _require(
        np.isfinite(calibrated).all()
        and np.all((calibrated >= 0.0) & (calibrated <= 1.0))
        and np.all(calibrated[~active] == 0.0)
        and np.array_equal(calibrated[neutral], cycle[neutral]),
        "C2-A0 calibrated utility boundary mismatch",
    )
    calibrated.setflags(write=False)
    return calibrated


def build_calibration_outputs(y_gen, U_cycle, labeled_ids, labeled_targets, shuffled_targets):
    """Build primary and shuffled calibration paths independently from y_gen."""
    actions = _readonly(y_gen, dtype=np.int64)
    cycle = _readonly(U_cycle, dtype=np.float32)
    labeled = _readonly(labeled_ids, dtype=np.int64)
    targets = _readonly(labeled_targets, dtype=np.int64)
    shuffled = validate_negative_control(targets, shuffled_targets)
    vote_state = build_cross_direction_vote_state(actions)

    primary = build_sparse_anchors_and_mapping(vote_state, labeled, targets)
    primary_evidence = build_action_label_evidence(
        vote_state, primary["anchors"], primary["sparse_mapping"], actions
    )
    calibrated = calibrate_utility(cycle, primary_evidence["E_label"])

    control = build_sparse_anchors_and_mapping(vote_state, labeled, shuffled)
    control_evidence = build_action_label_evidence(
        vote_state, control["anchors"], control["sparse_mapping"], actions
    )
    calibrated_control = calibrate_utility(cycle, control_evidence["E_label"])
    return {
        "R": vote_state,
        "anchors": primary["anchors"],
        "soft_contingency": primary["soft_contingency"],
        "sparse_mapping": primary["sparse_mapping"],
        "similarity": primary_evidence["similarity"],
        "E_label": primary_evidence["E_label"],
        "U_tilde": calibrated,
        "anchors_shuffle": control["anchors"],
        "soft_contingency_shuffle": control["soft_contingency"],
        "sparse_mapping_shuffle": control["sparse_mapping"],
        "similarity_shuffle": control_evidence["similarity"],
        "E_label_shuffle": control_evidence["E_label"],
        "U_tilde_shuffle": calibrated_control,
    }


def _weighted_mean(records, field):
    weights = np.asarray([record["sample_count"] for record in records], dtype=np.float64)
    values = np.asarray([record[field] for record in records], dtype=np.float64)
    _require(
        weights.size > 0 and np.all(weights > 0.0) and np.isfinite(values).all(),
        "C2-A0 weighted metric boundary mismatch",
    )
    return float(np.average(values, weights=weights))


def analyze_direction(correct, U_cycle, U_tilde, U_tilde_shuffle, E_label, C_conf, sample_ids, direction_id):
    """Compute matched conditional AUC and signed correction for one direction."""
    outcomes = _readonly(correct, dtype=np.int64)
    cycle = _readonly(U_cycle, dtype=np.float64)
    calibrated = _readonly(U_tilde, dtype=np.float64)
    shuffled = _readonly(U_tilde_shuffle, dtype=np.float64)
    evidence = _readonly(E_label, dtype=np.float64)
    confidence = _readonly(C_conf, dtype=np.float64)
    ids = _readonly(sample_ids, dtype=np.int64)
    _require(
        outcomes.shape == cycle.shape == calibrated.shape == shuffled.shape
        == evidence.shape == confidence.shape == ids.shape == (UNLABELED_EVAL_COUNT,)
        and np.all((outcomes == 0) | (outcomes == 1))
        and all(np.isfinite(value).all() for value in (cycle, calibrated, shuffled, evidence, confidence))
        and np.unique(ids).size == UNLABELED_EVAL_COUNT,
        "C2-A0 direction input boundary mismatch",
    )
    assignments, strata_audit = confidence_equal_count_strata(confidence, ids)
    delta = calibrated - cycle
    delta_shuffle = shuffled - cycle
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
            "AUC_E_label": None,
            "SCG_bin": None,
            "SCG_shuffle_bin": None,
        }
        if valid:
            correct_indices = indices[bin_outcomes == 1]
            incorrect_indices = indices[bin_outcomes == 0]
            record.update({
                "AUC_Ucycle": binary_auc_or_none(bin_outcomes, cycle[indices]),
                "AUC_U_tilde": binary_auc_or_none(bin_outcomes, calibrated[indices]),
                "AUC_U_tilde_shuffle": binary_auc_or_none(bin_outcomes, shuffled[indices]),
                "AUC_E_label": binary_auc_or_none(bin_outcomes, evidence[indices]),
                "SCG_bin": float(delta[correct_indices].mean() - delta[incorrect_indices].mean()),
                "SCG_shuffle_bin": float(
                    delta_shuffle[correct_indices].mean() - delta_shuffle[incorrect_indices].mean()
                ),
            })
        bins.append(record)
    valid_bins = [record for record in bins if record["valid_both_correctness_classes"]]
    direction_valid = len(valid_bins) >= MIN_VALID_BINS_PER_DIRECTION
    result = {
        "direction_id": int(direction_id),
        "valid_bin_count": len(valid_bins),
        "valid_bin_mask": valid_mask,
        "direction_valid": direction_valid,
        "bins": bins,
        "CondAUC_Ucycle": None,
        "CondAUC_U_tilde": None,
        "CondAUC_U_tilde_shuffle": None,
        "CondAUC_E_label": None,
        "DeltaCalAUC": None,
        "LabelSpecificityGap": None,
        "SCG": None,
        "SCG_shuffle": None,
        "SCGGap": None,
        "confidence_strata": strata_audit,
    }
    if direction_valid:
        cond_cycle = _weighted_mean(valid_bins, "AUC_Ucycle")
        cond_calibrated = _weighted_mean(valid_bins, "AUC_U_tilde")
        cond_shuffle = _weighted_mean(valid_bins, "AUC_U_tilde_shuffle")
        scg = _weighted_mean(valid_bins, "SCG_bin")
        scg_shuffle = _weighted_mean(valid_bins, "SCG_shuffle_bin")
        result.update({
            "CondAUC_Ucycle": cond_cycle,
            "CondAUC_U_tilde": cond_calibrated,
            "CondAUC_U_tilde_shuffle": cond_shuffle,
            "CondAUC_E_label": _weighted_mean(valid_bins, "AUC_E_label"),
            "DeltaCalAUC": cond_calibrated - cond_cycle,
            "LabelSpecificityGap": cond_calibrated - cond_shuffle,
            "SCG": scg,
            "SCG_shuffle": scg_shuffle,
            "SCGGap": scg - scg_shuffle,
        })
    return result


def analyze_all_directions(correct, U_cycle, U_tilde, U_tilde_shuffle, E_label, C_conf, sample_ids):
    arrays = [
        _readonly(correct, dtype=np.int64),
        _readonly(U_cycle, dtype=np.float64),
        _readonly(U_tilde, dtype=np.float64),
        _readonly(U_tilde_shuffle, dtype=np.float64),
        _readonly(E_label, dtype=np.float64),
        _readonly(C_conf, dtype=np.float64),
    ]
    ids = _readonly(sample_ids, dtype=np.int64)
    expected_shape = (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
    expected_ids = np.setdiff1d(canonical_sample_ids(), fixed_labeled_ids())
    _require(
        all(value.shape == expected_shape for value in arrays)
        and ids.shape == (UNLABELED_EVAL_COUNT,)
        and np.array_equal(ids, expected_ids)
        and np.intersect1d(ids, fixed_labeled_ids()).size == 0,
        "C2-A0 unlabeled-only evaluation boundary mismatch",
    )
    directions = [
        analyze_direction(
            arrays[0][:, direction_id], arrays[1][:, direction_id],
            arrays[2][:, direction_id], arrays[3][:, direction_id],
            arrays[4][:, direction_id], arrays[5][:, direction_id], ids,
            direction_id,
        )
        for direction_id in range(DIRECTION_COUNT)
    ]
    return {
        "direction_count": DIRECTION_COUNT,
        "evaluation_sample_count": UNLABELED_EVAL_COUNT,
        "labeled_evaluation_intersection_count": 0,
        "GT_used_for_confidence_stratification": False,
        "directions": directions,
    }


def _mean(records, field):
    if not records:
        return None
    values = np.asarray([record[field] for record in records], dtype=np.float64)
    _require(np.isfinite(values).all(), "C2-A0 non-finite seed metric")
    return float(values.mean())


def build_seed_summary(seed, directional_metrics):
    active_seed = validate_seed(seed)
    directions = directional_metrics.get("directions", ())
    _require(
        len(directions) == DIRECTION_COUNT
        and all(record.get("direction_id") == index for index, record in enumerate(directions)),
        "C2-A0 seed direction schema mismatch",
    )
    valid = [record for record in directions if record["direction_valid"]]
    deltas = np.asarray([record["DeltaCalAUC"] for record in valid], dtype=np.float64)
    positive_count = int(np.count_nonzero(deltas > 0.0))
    p_value = one_sided_wilcoxon_greater(deltas)
    mean_delta = _mean(valid, "DeltaCalAUC")
    mean_specificity = _mean(valid, "LabelSpecificityGap")
    mean_scg = _mean(valid, "SCG")
    mean_scg_gap = _mean(valid, "SCGGap")
    conditions = OrderedDict((
        ("minimum_valid_directions", len(valid) >= MIN_VALID_DIRECTIONS_PER_SEED),
        ("minimum_positive_DeltaCalAUC_directions", positive_count >= POSITIVE_DIRECTION_MIN_COUNT),
        ("positive_mean_DeltaCalAUC", mean_delta is not None and mean_delta > 0.0),
        ("one_sided_wilcoxon_greater", p_value is not None and p_value < ALPHA),
        ("positive_mean_LabelSpecificityGap", mean_specificity is not None and mean_specificity > 0.0),
        ("positive_mean_SCG", mean_scg is not None and mean_scg > 0.0),
        ("positive_mean_SCGGap", mean_scg_gap is not None and mean_scg_gap > 0.0),
    ))
    return {
        "stage": STAGE,
        "seed": active_seed,
        "label_count": LABEL_COUNT,
        "unlabeled_evaluation_count": UNLABELED_EVAL_COUNT,
        "valid_direction_count": len(valid),
        "positive_DeltaCalAUC_direction_count": positive_count,
        "mean_DeltaCalAUC": mean_delta,
        "mean_LabelSpecificityGap": mean_specificity,
        "mean_SCG": mean_scg,
        "mean_SCGGap": mean_scg_gap,
        "mean_CondAUC_E_label_secondary": _mean(valid, "CondAUC_E_label"),
        "wilcoxon_DeltaCalAUC_greater_p": p_value,
        "wilcoxon_alternative": "greater",
        "wilcoxon_unit": "direction-wise paired consistency test",
        "directions_are_not_independent_supervised_samples": True,
        "seed_gate_conditions": dict(conditions),
        "C2_A0_SEED_PASS": bool(all(conditions.values())),
    }


def _mean_or_none(values):
    if any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=np.float64)
    _require(np.isfinite(array).all(), "C2-A0 non-finite multi-seed metric")
    return float(array.mean())


def build_multiseed_decision(seed_summaries):
    _require(
        tuple(seed_summaries) == SEEDS
        and all(seed_summaries[seed].get("seed") == seed for seed in SEEDS),
        "C2-A0 multi-seed set/order mismatch",
    )
    pass_count = sum(bool(seed_summaries[seed]["C2_A0_SEED_PASS"]) for seed in SEEDS)
    aggregates = {
        name: _mean_or_none([seed_summaries[seed]["mean_" + name] for seed in SEEDS])
        for name in ("DeltaCalAUC", "LabelSpecificityGap", "SCG", "SCGGap")
    }
    conditions = OrderedDict((
        ("at_least_two_of_three_seed_passes", pass_count >= SEED_PASS_MIN_COUNT),
        ("positive_aggregate_mean_DeltaCalAUC", aggregates["DeltaCalAUC"] is not None and aggregates["DeltaCalAUC"] > 0.0),
        ("positive_aggregate_mean_LabelSpecificityGap", aggregates["LabelSpecificityGap"] is not None and aggregates["LabelSpecificityGap"] > 0.0),
        ("positive_aggregate_mean_SCG", aggregates["SCG"] is not None and aggregates["SCG"] > 0.0),
        ("positive_aggregate_mean_SCGGap", aggregates["SCGGap"] is not None and aggregates["SCGGap"] > 0.0),
    ))
    passed = bool(all(conditions.values()))
    return {
        "summary": {
            "stage": STAGE,
            "seeds": list(SEEDS),
            "seed_pass_count": pass_count,
            "aggregate_mean_DeltaCalAUC": aggregates["DeltaCalAUC"],
            "aggregate_mean_LabelSpecificityGap": aggregates["LabelSpecificityGap"],
            "aggregate_mean_SCG": aggregates["SCG"],
            "aggregate_mean_SCGGap": aggregates["SCGGap"],
            "seed_summaries": {str(seed): seed_summaries[seed] for seed in SEEDS},
        },
        "decision": {
            "decision_conditions": dict(conditions),
            "final_decision": FINAL_DECISIONS[0] if passed else FINAL_DECISIONS[1],
            "C2_A0_SPARSE_LABEL_UTILITY_CALIBRATION_PASS": passed,
            "label_count_is_fourteen_not_directions_times_labels": True,
            "post_hoc_gate_change": False,
        },
    }
