"""Frozen statistical protocol for the read-only Bridge-P0 diagnostic.

This module contains no artifact I/O and no trainable mechanism. It defines
the pre-registered strata, diagnostics, and fixed decision gates.
"""

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from irv.b4_information_utility import tensor_sha256


SEEDS = (20, 30, 50)
SAMPLE_NUM = 1400
VIEW_NUM = 6
CLASS_NUM = 7
DIRECTION_COUNT = 20
CONFIDENCE_BIN_COUNT = 5
MIN_VALID_BINS_PER_DIRECTION = 3
MIN_VALID_DIRECTIONS_PER_SEED = 16
POSITIVE_DIRECTION_MIN_COUNT = 12
ALPHA = 0.05
SEED_PASS_MIN_COUNT = 2
WEAK_QUALITY_CONDITION = "snr2p5_k3_seed20"
FINAL_DECISIONS = (
    "BRIDGE_P0_RESIDUAL_UTILITY_PASS",
    "BRIDGE_P0_RESIDUAL_UTILITY_NOT_CONFIRMED",
)

FROZEN_SOURCE_SHA256 = {
    "experiments/cyclic_utility/c0_complementary_semantic_verification.py": (
        "d52fbf0816557ba57a11fc490ead1a26598b35e68bac7808b7077c448bc7a4a9"
    ),
    "experiments/cyclic_utility/evaluate_c0_complementary_semantic_verification.py": (
        "adb420bbc59833ff7eca58f3b5d54162a45b4106227d8f6540fa69fc22ac691b"
    ),
    "experiments/cyclic_utility/vsa_decoupled_action_protocol.py": (
        "f5ad5d2d81416802999b657ee04a8c01896debcb7c522523395b49729b95ca48"
    ),
    "experiments/cyclic_utility/train_vsa_a0_decoupled_action.py": (
        "b3a354beaa4824717ed90f8a533cf1ddf5ef3a658b636380d2c5b3f0782afdd3"
    ),
    "experiments/cyclic_utility/summarize_vsa_a0_decoupled_action.py": (
        "b346825ce6785f837d81ea805067db0825dcffa7c83a06dd3fd9d9f7a564f156"
    ),
    "experiments/e1_pairwise_utility/train_e1_pairwise_utility.py": (
        "ad6dcb187ccb39d36f5f5839130b0345053b23d92df1c00d9db98d41f8b1ca73"
    ),
}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_seed(seed):
    value = int(seed)
    if value not in SEEDS:
        raise ValueError("Bridge-P0 seed must be one of " + str(SEEDS))
    return value


def _readonly(value, dtype=None):
    array = np.array(np.asarray(value, dtype=dtype), copy=True, order="C")
    array.setflags(write=False)
    return array



def confidence_equal_count_strata(
    confidence,
    sample_ids,
    bin_count=CONFIDENCE_BIN_COUNT,
):
    """Assign deterministic rank-based confidence strata without GT."""
    values = _readonly(confidence, dtype=np.float64)
    ids = _readonly(sample_ids, dtype=np.int64)
    count = int(bin_count)
    _require(
        values.ndim == ids.ndim == 1
        and values.shape == ids.shape
        and values.size >= count
        and count == CONFIDENCE_BIN_COUNT
        and np.isfinite(values).all()
        and np.unique(ids).size == ids.size,
        "confidence stratification input boundary mismatch",
    )
    # Python sorting is stable; sample_id is the sole exact-score tie break.
    order = np.asarray(
        sorted(
            range(values.size),
            key=lambda index: (
                float(values[index]), int(ids[index])
            ),
        ),
        dtype=np.int64,
    )
    rank_chunks = np.array_split(order, count)
    assignments = np.full(values.size, -1, dtype=np.int64)
    bin_records = []
    rank_start = 0
    for bin_id, indices in enumerate(rank_chunks):
        assignments[indices] = bin_id
        member_ids = np.ascontiguousarray(ids[indices], dtype=np.int64)
        bin_records.append({
            "bin_id": bin_id,
            "bin_label": "Q" + str(bin_id + 1),
            "sample_count": int(indices.size),
            "rank_start_inclusive": rank_start,
            "rank_end_exclusive": rank_start + int(indices.size),
            "sample_ids_logical_sha256": tensor_sha256(member_ids),
            "confidence_min": float(values[indices].min()),
            "confidence_max": float(values[indices].max()),
        })
        rank_start += int(indices.size)
    sizes = [record["sample_count"] for record in bin_records]
    _require(
        np.all(assignments >= 0)
        and sum(sizes) == values.size
        and max(sizes) - min(sizes) <= 1,
        "confidence equal-count assignment failed",
    )
    assignments.setflags(write=False)
    return assignments, {
        "bin_count": count,
        "sample_count": int(values.size),
        "assignment_rule": (
            "stable_confidence_ascending_then_sample_id; "
            "equal_count_rank_chunks"
        ),
        "confidence_only_for_assignment": True,
        "GT_used_for_assignment": False,
        "sample_id_only_tie_break": True,
        "approximately_equal_count_pass": True,
        "bin_sizes": sizes,
        "assignment_logical_sha256": tensor_sha256(assignments),
        "bins": bin_records,
    }


def binary_auc_or_none(correct, score):
    outcomes = _readonly(correct, dtype=np.int64)
    values = _readonly(score, dtype=np.float64)
    _require(
        outcomes.ndim == values.ndim == 1
        and outcomes.shape == values.shape
        and outcomes.size > 0
        and np.isfinite(values).all()
        and np.all((outcomes == 0) | (outcomes == 1)),
        "Bridge-P0 binary AUC input boundary mismatch",
    )
    if np.unique(outcomes).size != 2:
        return None
    return float(roc_auc_score(outcomes, values))


def matched_confidence_lift(correct, score, sample_ids):
    """Upper-minus-lower correctness under deterministic equal halves."""
    outcomes = _readonly(correct, dtype=np.float64)
    values = _readonly(score, dtype=np.float64)
    ids = _readonly(sample_ids, dtype=np.int64)
    _require(
        outcomes.ndim == values.ndim == ids.ndim == 1
        and outcomes.shape == values.shape == ids.shape
        and outcomes.size >= 2
        and np.isfinite(outcomes).all()
        and np.isfinite(values).all()
        and np.unique(ids).size == ids.size,
        "Bridge-P0 action-lift input boundary mismatch",
    )
    order = np.asarray(
        sorted(
            range(values.size),
            key=lambda index: (
                float(values[index]), int(ids[index])
            ),
        ),
        dtype=np.int64,
    )
    lower, upper = np.array_split(order, 2)
    _require(
        lower.size > 0
        and upper.size > 0
        and abs(int(lower.size) - int(upper.size)) <= 1,
        "Bridge-P0 equal-count half assignment failed",
    )
    return {
        "lift": float(outcomes[upper].mean() - outcomes[lower].mean()),
        "lower_sample_count": int(lower.size),
        "upper_sample_count": int(upper.size),
        "sample_id_only_tie_break": True,
        "GT_used_for_half_assignment": False,
    }


def _weighted_mean(records, value_name):
    weights = np.asarray(
        [record["sample_count"] for record in records], dtype=np.float64
    )
    values = np.asarray(
        [record[value_name] for record in records], dtype=np.float64
    )
    _require(
        weights.size > 0
        and np.all(weights > 0)
        and np.isfinite(values).all(),
        "Bridge-P0 weighted aggregation boundary mismatch",
    )
    return float(np.average(values, weights=weights))



def analyze_direction(
    correct,
    U_cycle,
    C_conf,
    U_cycle_shuffle,
    sample_ids,
    direction_id,
):
    """Analyze one direction using one confidence-derived valid-bin mask."""
    outcomes = _readonly(correct, dtype=np.int64)
    cycle = _readonly(U_cycle, dtype=np.float64)
    confidence = _readonly(C_conf, dtype=np.float64)
    shuffle = _readonly(U_cycle_shuffle, dtype=np.float64)
    ids = _readonly(sample_ids, dtype=np.int64)
    _require(
        outcomes.shape
        == cycle.shape
        == confidence.shape
        == shuffle.shape
        == ids.shape
        and outcomes.ndim == 1
        and np.isfinite(cycle).all()
        and np.isfinite(confidence).all()
        and np.isfinite(shuffle).all()
        and np.all((outcomes == 0) | (outcomes == 1)),
        "Bridge-P0 direction input boundary mismatch",
    )
    assignments, strata_audit = confidence_equal_count_strata(
        confidence, ids
    )
    auc_bins = []
    lift_bins = []
    valid_mask = []
    for bin_id in range(CONFIDENCE_BIN_COUNT):
        indices = np.flatnonzero(assignments == bin_id)
        bin_outcomes = outcomes[indices]
        valid = bool(np.unique(bin_outcomes).size == 2)
        valid_mask.append(valid)
        auc_record = {
            "bin_id": bin_id,
            "bin_label": "Q" + str(bin_id + 1),
            "sample_count": int(indices.size),
            "valid_both_correctness_classes": valid,
            "AUC_U_cycle": None,
            "AUC_conf": None,
            "AUC_shuffle": None,
        }
        lift_record = {
            "bin_id": bin_id,
            "bin_label": "Q" + str(bin_id + 1),
            "sample_count": int(indices.size),
            "valid_both_correctness_classes": valid,
            "Lift_cycle": None,
            "Lift_shuffle": None,
        }
        if valid:
            auc_record.update({
                "AUC_U_cycle": binary_auc_or_none(
                    bin_outcomes, cycle[indices]
                ),
                "AUC_conf": binary_auc_or_none(
                    bin_outcomes, confidence[indices]
                ),
                "AUC_shuffle": binary_auc_or_none(
                    bin_outcomes, shuffle[indices]
                ),
            })
            cycle_lift = matched_confidence_lift(
                bin_outcomes, cycle[indices], ids[indices]
            )
            shuffle_lift = matched_confidence_lift(
                bin_outcomes, shuffle[indices], ids[indices]
            )
            lift_record.update({
                "Lift_cycle": cycle_lift["lift"],
                "Lift_shuffle": shuffle_lift["lift"],
                "cycle_half_audit": cycle_lift,
                "shuffle_half_audit": shuffle_lift,
            })
        auc_bins.append(auc_record)
        lift_bins.append(lift_record)
    valid_auc_bins = [
        record
        for record in auc_bins
        if record["valid_both_correctness_classes"]
    ]
    valid_lift_bins = [
        record
        for record in lift_bins
        if record["valid_both_correctness_classes"]
    ]
    valid_bin_count = len(valid_auc_bins)
    direction_valid = valid_bin_count >= MIN_VALID_BINS_PER_DIRECTION
    # Aggregates are exposed only for valid directions.
    conditional = {
        "direction_id": int(direction_id),
        "valid_bin_count": valid_bin_count,
        "valid_bin_mask": valid_mask,
        "same_valid_bin_mask_U_conf_shuffle": True,
        "direction_valid": direction_valid,
        "bins": auc_bins,
        "CondAUC_U": None,
        "CondAUC_conf": None,
        "CondAUC_shuffle": None,
        "DeltaCondAUC_cycle": None,
        "DeltaCondAUC_shuffle": None,
        "CorrespondenceResidualGap": None,
    }
    action = {
        "direction_id": int(direction_id),
        "valid_bin_count": valid_bin_count,
        "valid_bin_mask": valid_mask,
        "direction_valid": direction_valid,
        "bins": lift_bins,
        "ActionLift_cycle": None,
        "ActionLift_shuffle": None,
        "CorrespondenceLiftGap": None,
    }
    if direction_valid:
        cond_cycle = _weighted_mean(valid_auc_bins, "AUC_U_cycle")
        cond_conf = _weighted_mean(valid_auc_bins, "AUC_conf")
        cond_shuffle = _weighted_mean(valid_auc_bins, "AUC_shuffle")
        action_cycle = _weighted_mean(valid_lift_bins, "Lift_cycle")
        action_shuffle = _weighted_mean(valid_lift_bins, "Lift_shuffle")
        conditional.update({
            "CondAUC_U": cond_cycle,
            "CondAUC_conf": cond_conf,
            "CondAUC_shuffle": cond_shuffle,
            "DeltaCondAUC_cycle": cond_cycle - cond_conf,
            "DeltaCondAUC_shuffle": cond_shuffle - cond_conf,
            "CorrespondenceResidualGap": cond_cycle - cond_shuffle,
        })
        action.update({
            "ActionLift_cycle": action_cycle,
            "ActionLift_shuffle": action_shuffle,
            "CorrespondenceLiftGap": action_cycle - action_shuffle,
        })
    return {
        "confidence_strata": {
            "direction_id": int(direction_id),
            **strata_audit,
        },
        "conditional_auc": conditional,
        "action_lift": action,
    }


def analyze_all_directions(
    correct,
    U_cycle,
    C_conf,
    U_cycle_shuffle,
    sample_ids=None,
):
    outcomes = _readonly(correct, dtype=np.int64)
    cycle = _readonly(U_cycle, dtype=np.float64)
    confidence = _readonly(C_conf, dtype=np.float64)
    shuffle = _readonly(U_cycle_shuffle, dtype=np.float64)
    ids = (
        np.arange(SAMPLE_NUM, dtype=np.int64)
        if sample_ids is None
        else _readonly(sample_ids, dtype=np.int64)
    )
    expected_shape = (SAMPLE_NUM, DIRECTION_COUNT)
    _require(
        outcomes.shape
        == cycle.shape
        == confidence.shape
        == shuffle.shape
        == expected_shape
        and ids.shape == (SAMPLE_NUM,),
        "Bridge-P0 full directional input shape mismatch",
    )
    results = [
        analyze_direction(
            outcomes[:, direction_id],
            cycle[:, direction_id],
            confidence[:, direction_id],
            shuffle[:, direction_id],
            ids,
            direction_id,
        )
        for direction_id in range(DIRECTION_COUNT)
    ]
    return {
        "confidence_strata": {
            "confidence_bin_count": CONFIDENCE_BIN_COUNT,
            "GT_used_for_stratification": False,
            "directions": [
                result["confidence_strata"] for result in results
            ],
        },
        "conditional_auc": {
            "direction_count": DIRECTION_COUNT,
            "minimum_valid_bins_per_direction": (
                MIN_VALID_BINS_PER_DIRECTION
            ),
            "directions": [
                result["conditional_auc"] for result in results
            ],
        },
        "action_lift": {
            "direction_count": DIRECTION_COUNT,
            "hard_admission_constructed": False,
            "top_k_constructed": False,
            "directions": [
                result["action_lift"] for result in results
            ],
        },
    }



def one_sided_wilcoxon_greater(values):
    differences = _readonly(values, dtype=np.float64)
    _require(
        differences.ndim == 1 and np.isfinite(differences).all(),
        "Bridge-P0 Wilcoxon input boundary mismatch",
    )
    if differences.size == 0:
        return None
    if np.all(differences == 0.0):
        return 1.0
    return float(
        wilcoxon(
            differences,
            alternative="greater",
            zero_method="wilcox",
        ).pvalue
    )


def build_seed_summary(seed, conditional_auc, action_lift):
    active_seed = validate_seed(seed)
    conditional = conditional_auc["directions"]
    action = action_lift["directions"]
    _require(
        len(conditional) == len(action) == DIRECTION_COUNT
        and all(
            conditional[index]["direction_id"]
            == action[index]["direction_id"]
            == index
            for index in range(DIRECTION_COUNT)
        ),
        "Bridge-P0 seed summary direction boundary mismatch",
    )
    valid_pairs = [
        (conditional[index], action[index])
        for index in range(DIRECTION_COUNT)
        if conditional[index]["direction_valid"]
        and action[index]["direction_valid"]
    ]
    valid_direction_count = len(valid_pairs)

    def mean_field(record_index, field):
        if not valid_pairs:
            return None
        values = np.asarray(
            [pair[record_index][field] for pair in valid_pairs],
            dtype=np.float64,
        )
        _require(np.isfinite(values).all(), "non-finite seed metric")
        return float(values.mean())

    deltas = np.asarray(
        [pair[0]["DeltaCondAUC_cycle"] for pair in valid_pairs],
        dtype=np.float64,
    )
    positive_count = int(np.count_nonzero(deltas > 0.0))
    p_value = one_sided_wilcoxon_greater(deltas)
    mean_delta = mean_field(0, "DeltaCondAUC_cycle")
    mean_action = mean_field(1, "ActionLift_cycle")
    mean_residual_gap = mean_field(0, "CorrespondenceResidualGap")
    mean_lift_gap = mean_field(1, "CorrespondenceLiftGap")
    conditions = {
        "minimum_valid_directions": (
            valid_direction_count >= MIN_VALID_DIRECTIONS_PER_SEED
        ),
        "positive_mean_DeltaCondAUC_cycle": (
            mean_delta is not None and mean_delta > 0.0
        ),
        "positive_direction_count": (
            positive_count >= POSITIVE_DIRECTION_MIN_COUNT
        ),
        "one_sided_wilcoxon_greater": (
            p_value is not None and p_value < ALPHA
        ),
        "positive_mean_ActionLift_cycle": (
            mean_action is not None and mean_action > 0.0
        ),
        "positive_mean_CorrespondenceResidualGap": (
            mean_residual_gap is not None and mean_residual_gap > 0.0
        ),
        "positive_mean_CorrespondenceLiftGap": (
            mean_lift_gap is not None and mean_lift_gap > 0.0
        ),
    }
    seed_pass = bool(all(conditions.values()))
    return {
        "stage": "Bridge-P0",
        "seed": active_seed,
        "valid_direction_count": valid_direction_count,
        "mean_CondAUC_U": mean_field(0, "CondAUC_U"),
        "mean_CondAUC_conf": mean_field(0, "CondAUC_conf"),
        "mean_CondAUC_shuffle": mean_field(0, "CondAUC_shuffle"),
        "mean_DeltaCondAUC_cycle": mean_delta,
        "positive_DeltaCondAUC_direction_count": positive_count,
        "wilcoxon_cycle_vs_conf_p": p_value,
        "wilcoxon_alternative": "greater",
        "wilcoxon_unit": "directional_split",
        "wilcoxon_direction_count": valid_direction_count,
        "mean_CorrespondenceResidualGap": mean_residual_gap,
        "mean_ActionLift_cycle": mean_action,
        "mean_ActionLift_shuffle": mean_field(1, "ActionLift_shuffle"),
        "mean_CorrespondenceLiftGap": mean_lift_gap,
        "seed_gate_conditions": conditions,
        "RESIDUAL_INFORMATION_SEED_PASS": seed_pass,
        "new_utility_constructed": False,
    }


def _mean_or_none(values):
    if any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=np.float64)
    _require(np.isfinite(array).all(), "non-finite multi-seed metric")
    return float(array.mean())


def build_multiseed_decision(seed_summaries):
    _require(
        tuple(seed_summaries) == SEEDS
        and all(seed_summaries[seed]["seed"] == seed for seed in SEEDS),
        "Bridge-P0 multi-seed set/order mismatch",
    )
    seed_pass_count = sum(
        bool(seed_summaries[seed]["RESIDUAL_INFORMATION_SEED_PASS"])
        for seed in SEEDS
    )
    mean_delta = _mean_or_none([
        seed_summaries[seed]["mean_DeltaCondAUC_cycle"]
        for seed in SEEDS
    ])
    mean_action = _mean_or_none([
        seed_summaries[seed]["mean_ActionLift_cycle"]
        for seed in SEEDS
    ])
    mean_residual_gap = _mean_or_none([
        seed_summaries[seed]["mean_CorrespondenceResidualGap"]
        for seed in SEEDS
    ])
    mean_lift_gap = _mean_or_none([
        seed_summaries[seed]["mean_CorrespondenceLiftGap"]
        for seed in SEEDS
    ])
    conditions = {
        "at_least_two_seed_passes": seed_pass_count >= SEED_PASS_MIN_COUNT,
        "positive_mean_seed_DeltaCondAUC_cycle": (
            mean_delta is not None and mean_delta > 0.0
        ),
        "positive_mean_seed_ActionLift_cycle": (
            mean_action is not None and mean_action > 0.0
        ),
        "positive_mean_seed_CorrespondenceResidualGap": (
            mean_residual_gap is not None and mean_residual_gap > 0.0
        ),
        "positive_mean_seed_CorrespondenceLiftGap": (
            mean_lift_gap is not None and mean_lift_gap > 0.0
        ),
    }
    passed = bool(all(conditions.values()))
    final_decision = FINAL_DECISIONS[0] if passed else FINAL_DECISIONS[1]
    summary = {
        "stage": "Bridge-P0",
        "seeds": list(SEEDS),
        "seed_pass_count": seed_pass_count,
        "mean_seed_DeltaCondAUC_cycle": mean_delta,
        "mean_seed_ActionLift_cycle": mean_action,
        "mean_seed_CorrespondenceResidualGap": mean_residual_gap,
        "mean_seed_CorrespondenceLiftGap": mean_lift_gap,
        "seed_summaries": {
            str(seed): seed_summaries[seed] for seed in SEEDS
        },
    }
    decision = {
        **conditions,
        "decision_conditions": conditions,
        "final_decision": final_decision,
        "no_statistical_significance_claim_from_three_seeds": True,
        "post_hoc_threshold_change": False,
        "new_utility_constructed": False,
    }
    return {"summary": summary, "decision": decision}


def leakage_audit():
    """Return the hard, JSON-safe Bridge execution boundary."""
    return {
        "training_entered": False,
        "optimizer_created": False,
        "backward_called": False,
        "model_loaded_for_training": False,
        "R_loaded": False,
        "corruption_mask_loaded": False,
        "sparse_labels_loaded": False,
        "sparse_label_ids_loaded": False,
        "sparse_label_targets_loaded": False,
        "B7_artifact_loaded": False,
        "GT_loaded_before_bridge_input_seal": False,
        "GT_loaded_after_bridge_input_seal": True,
        "no_per_direction_GT_mapping": True,
        "no_per_bin_GT_mapping": True,
        "new_utility_constructed": False,
        "VSA_modified": False,
        "C0_modified": False,
    }
