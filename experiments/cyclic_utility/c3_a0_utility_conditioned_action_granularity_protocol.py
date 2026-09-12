"""Pure-array protocol for C3-A0 semantic action-granularity auditing.

The stage keeps frozen U_cycle unchanged and asks whether it ranks the validity
of class, relation, and one-step prototype-write actions.  Sparse semantic
mapping is inherited verbatim from C2-A0.  This module performs no I/O, model
execution, optimization, training, or persistent memory update.
"""

from collections import OrderedDict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from experiments.cyclic_utility import c2_a0_sparse_label_utility_protocol as parent


STAGE = "C3-A0"
SEEDS = parent.SEEDS
SAMPLE_NUM = parent.SAMPLE_NUM
CLASS_NUM = parent.CLASS_NUM
DIRECTION_COUNT = parent.DIRECTION_COUNT
LABEL_COUNT = parent.LABEL_COUNT
UNLABELED_EVAL_COUNT = parent.UNLABELED_EVAL_COUNT
UTILITY_QUINTILE_COUNT = 5
MIN_VALID_DIRECTIONS_PER_SEED = 16
POSITIVE_DIRECTION_MIN_COUNT = 12
SEED_PASS_MIN_COUNT = 2
SIGNIFICANCE_LEVEL = 0.05

FIXED_LABELED_IDS = parent.FIXED_LABELED_IDS
FIXED_LABELED_TARGETS = parent.FIXED_LABELED_TARGETS
FIXED_SHUFFLED_TARGETS = parent.FIXED_SHUFFLED_TARGETS

RELATION_PRIMARY_DECISION = "C3-B0 RELATION-LEVEL SEMANTIC ACTION PILOT"
CLASS_PRIMARY_DECISION = (
    "C3-B0 CLASS-LEVEL UTILITY-CONDITIONED PSEUDO ACTION"
)
PRIMARY_INTERFACE_FAIL_DECISION = (
    "CURRENT U_CYCLE -> WEAK-LABEL SEMANTIC ACTION INTERFACE FAILS; "
    "PIVOT WEAK-QUANTITY INTERFACE OR BASELINE; DO NOT ADD MEMORY"
)
MEMORY_ELIGIBLE_DECISION = (
    "MEMORY ELIGIBLE ONLY AFTER A C3-B0 PRIMARY ACTION PILOT DEMONSTRATES "
    "DOWNSTREAM CLUSTERING GAIN"
)
MEMORY_CLOSED_DECISION = "SEMANTIC MEMORY BRANCH REMAINS CLOSED"

# Frozen parent primitives.  The mapping helper is intentionally an alias.
canonical_sample_ids = parent.canonical_sample_ids
fixed_labeled_ids = parent.fixed_labeled_ids
fixed_labeled_targets = parent.fixed_labeled_targets
fixed_shuffled_targets = parent.fixed_shuffled_targets
validate_fixed_sparse_split = parent.validate_fixed_sparse_split
validate_negative_control = parent.validate_negative_control
build_vote_semantic_state = parent.build_cross_direction_vote_state
build_sparse_anchors_and_mapping = parent.build_sparse_anchors_and_mapping
binary_auc_or_none = parent.binary_auc_or_none
one_sided_wilcoxon_greater = parent.one_sided_wilcoxon_greater


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
        raise ValueError("C3-A0 seed must be one of " + str(SEEDS))
    return value


def utility_equal_count_quintiles(U_cycle, sample_ids):
    """Assign five stable rank bins using only U_cycle and sample_id."""
    cycle = _readonly(U_cycle, dtype=np.float64)
    ids = _readonly(sample_ids, dtype=np.int64)
    _require(
        cycle.shape == (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
        and ids.shape == (UNLABELED_EVAL_COUNT,)
        and np.unique(ids).size == ids.size
        and np.isfinite(cycle).all(),
        "C3-A0 utility-quintile input boundary mismatch",
    )
    assignments = np.full(cycle.shape, -1, dtype=np.int8)
    expected_sizes = [
        int(chunk.size)
        for chunk in np.array_split(
            np.arange(UNLABELED_EVAL_COUNT), UTILITY_QUINTILE_COUNT
        )
    ]
    for direction_id in range(DIRECTION_COUNT):
        order = np.lexsort((ids, cycle[:, direction_id]))
        for bin_id, indices in enumerate(
            np.array_split(order, UTILITY_QUINTILE_COUNT)
        ):
            assignments[indices, direction_id] = bin_id
            _require(
                indices.size == expected_sizes[bin_id],
                "C3-A0 utility-quintile size mismatch",
            )
    _require(
        np.all((assignments >= 0) & (assignments < UTILITY_QUINTILE_COUNT)),
        "C3-A0 utility-quintile assignment failed",
    )
    assignments.setflags(write=False)
    return assignments


def build_relation_balance_weights(predicted_relation):
    """Balance predicted-same and predicted-different edges per direction."""
    predicted = _readonly(predicted_relation, dtype=np.bool_)
    expected = (UNLABELED_EVAL_COUNT, LABEL_COUNT, DIRECTION_COUNT)
    _require(
        predicted.shape == expected,
        "C3-A0 predicted-relation weight boundary mismatch",
    )
    weights = np.empty(expected, dtype=np.float64)
    for direction_id in range(DIRECTION_COUNT):
        same = predicted[:, :, direction_id]
        same_count = int(np.count_nonzero(same))
        different_count = int(same.size - same_count)
        _require(
            same_count > 0 and different_count > 0,
            "C3-A0 relation group is empty",
        )
        direction_weights = weights[:, :, direction_id]
        direction_weights[same] = 1.0 / (2.0 * same_count)
        direction_weights[~same] = 1.0 / (2.0 * different_count)
        _require(
            np.isclose(
                np.sum(direction_weights[same]), 0.5, rtol=0.0, atol=1e-15
            )
            and np.isclose(
                np.sum(direction_weights[~same]), 0.5, rtol=0.0, atol=1e-15
            ),
            "C3-A0 relation balance normalization failed",
        )
    _require(
        np.isfinite(weights).all() and np.all(weights > 0.0),
        "C3-A0 relation weights must be finite and positive",
    )
    weights.setflags(write=False)
    return weights


def build_semantic_action_path(
    R,
    y_gen,
    unlabeled_ids,
    labeled_ids,
    labeled_targets,
):
    """Use the frozen C2-A0 mapping to build one complete semantic arm."""
    state = _readonly(R, dtype=np.float64)
    actions = _readonly(y_gen, dtype=np.int64)
    unlabeled = _readonly(unlabeled_ids, dtype=np.int64)
    labeled = _readonly(labeled_ids, dtype=np.int64)
    targets = _readonly(labeled_targets, dtype=np.int64)
    _require(
        state.shape == (SAMPLE_NUM, CLASS_NUM)
        and actions.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and unlabeled.shape == (UNLABELED_EVAL_COUNT,)
        and labeled.shape == targets.shape == (LABEL_COUNT,)
        and np.isfinite(state).all()
        and np.all((actions >= 0) & (actions < CLASS_NUM))
        and np.array_equal(
            np.bincount(targets, minlength=CLASS_NUM),
            np.full(CLASS_NUM, 2, dtype=np.int64),
        ),
        "C3-A0 semantic-action input boundary mismatch",
    )
    fitted = build_sparse_anchors_and_mapping(
        state, labeled, targets
    )
    sparse_mapping = fitted["sparse_mapping"]
    class_pred = np.ascontiguousarray(
        sparse_mapping[actions[unlabeled]], dtype=np.int64
    )
    predicted_relation = np.ascontiguousarray(
        class_pred[:, None, :] == targets[None, :, None],
        dtype=np.bool_,
    )
    relation_weights = build_relation_balance_weights(predicted_relation)
    prototypes = np.ascontiguousarray(fitted["anchors"], dtype=np.float64)
    candidates = np.ascontiguousarray(
        (
            2.0 * prototypes[class_pred]
            + state[unlabeled, None, :]
        )
        / 3.0,
        dtype=np.float64,
    )
    _require(
        class_pred.shape == (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
        and predicted_relation.shape
        == (UNLABELED_EVAL_COUNT, LABEL_COUNT, DIRECTION_COUNT)
        and prototypes.shape == (CLASS_NUM, CLASS_NUM)
        and candidates.shape
        == (UNLABELED_EVAL_COUNT, DIRECTION_COUNT, CLASS_NUM)
        and np.isfinite(prototypes).all()
        and np.isfinite(candidates).all(),
        "C3-A0 semantic-action construction failed",
    )
    for value in (class_pred, predicted_relation, prototypes, candidates):
        value.setflags(write=False)
    return {
        "sparse_mapping": sparse_mapping,
        "class_pred": class_pred,
        "PredRelation": predicted_relation,
        "relation_balance_weights": relation_weights,
        "P": prototypes,
        "P_candidate": candidates,
        "mapping_fit_label_count": fitted["mapping_fit_label_count"],
        "mapping_fit_full_GT_used": fitted["mapping_fit_full_GT_used"],
    }


def build_pre_gt_actions(
    sample_ids,
    unlabeled_ids,
    labeled_ids,
    labeled_targets,
    shuffled_labeled_targets,
    U_cycle,
    y_gen,
    R,
):
    """Build true/shuffle actions independently without accepting full GT."""
    ids = _readonly(sample_ids, dtype=np.int64)
    unlabeled = _readonly(unlabeled_ids, dtype=np.int64)
    labeled = _readonly(labeled_ids, dtype=np.int64)
    targets = _readonly(labeled_targets, dtype=np.int64)
    shuffled = validate_negative_control(
        targets, shuffled_labeled_targets
    )
    split = validate_fixed_sparse_split(labeled, unlabeled, targets)
    cycle = _readonly(U_cycle, dtype=np.float64)
    actions = _readonly(y_gen, dtype=np.int64)
    state = _readonly(R, dtype=np.float64)
    rebuilt_state = build_vote_semantic_state(actions)
    _require(
        ids.shape == (SAMPLE_NUM,)
        and np.array_equal(ids, canonical_sample_ids())
        and cycle.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and actions.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and state.shape == (SAMPLE_NUM, CLASS_NUM)
        and np.isfinite(cycle).all()
        and np.array_equal(state, rebuilt_state)
        and np.array_equal(split["unlabeled_ids"], unlabeled),
        "C3-A0 frozen pre-GT input boundary mismatch",
    )
    true_path = build_semantic_action_path(
        state, actions, unlabeled, labeled, targets
    )
    shuffle_path = build_semantic_action_path(
        state, actions, unlabeled, labeled, shuffled
    )
    quintiles = utility_equal_count_quintiles(
        cycle[unlabeled], unlabeled
    )
    return {
        "sample_ids": ids,
        "unlabeled_ids": unlabeled,
        "labeled_ids": labeled,
        "labeled_targets": targets,
        "shuffled_labeled_targets": shuffled,
        "U_cycle": cycle,
        "y_gen": actions,
        "R": state,
        "class_pred_true": true_path["class_pred"],
        "class_pred_shuffle": shuffle_path["class_pred"],
        "PredRelation_true": true_path["PredRelation"],
        "PredRelation_shuffle": shuffle_path["PredRelation"],
        "relation_balance_weights_true": true_path[
            "relation_balance_weights"
        ],
        "relation_balance_weights_shuffle": shuffle_path[
            "relation_balance_weights"
        ],
        "utility_quintile_assignment": quintiles,
        "P_true": true_path["P"],
        "P_shuffle": shuffle_path["P"],
        "P_candidate_true": true_path["P_candidate"],
        "P_candidate_shuffle": shuffle_path["P_candidate"],
        "sparse_mapping_true": true_path["sparse_mapping"],
        "sparse_mapping_shuffle": shuffle_path["sparse_mapping"],
        "true_mapping_fit_full_GT_used": true_path[
            "mapping_fit_full_GT_used"
        ],
        "shuffle_mapping_fit_full_GT_used": shuffle_path[
            "mapping_fit_full_GT_used"
        ],
        "shuffle_mapping_independently_rebuilt": True,
        "shuffle_relation_independently_rebuilt": True,
        "shuffle_prototype_independently_rebuilt": True,
        "shuffle_candidate_independently_rebuilt": True,
    }


def build_class_correctness(
    class_pred_true,
    class_pred_shuffle,
    full_GT,
    unlabeled_ids,
):
    true_pred = _readonly(class_pred_true, dtype=np.int64)
    shuffle_pred = _readonly(class_pred_shuffle, dtype=np.int64)
    truth = _readonly(full_GT, dtype=np.int64)
    unlabeled = _readonly(unlabeled_ids, dtype=np.int64)
    expected = (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
    _require(
        true_pred.shape == shuffle_pred.shape == expected
        and truth.shape == (SAMPLE_NUM,)
        and unlabeled.shape == (UNLABELED_EVAL_COUNT,)
        and np.all((truth >= 0) & (truth < CLASS_NUM)),
        "C3-A0 class-correctness input boundary mismatch",
    )
    target = truth[unlabeled, None]
    true_correct = np.ascontiguousarray(true_pred == target, dtype=np.int8)
    shuffle_correct = np.ascontiguousarray(
        shuffle_pred == target, dtype=np.int8
    )
    true_correct.setflags(write=False)
    shuffle_correct.setflags(write=False)
    return {
        "ClassCorrect_true": true_correct,
        "ClassCorrect_shuffle": shuffle_correct,
    }


def build_relation_correctness(
    PredRelation_true,
    PredRelation_shuffle,
    full_GT,
    unlabeled_ids,
    true_labeled_targets,
):
    true_relation = _readonly(PredRelation_true, dtype=np.bool_)
    shuffle_relation = _readonly(PredRelation_shuffle, dtype=np.bool_)
    truth = _readonly(full_GT, dtype=np.int64)
    unlabeled = _readonly(unlabeled_ids, dtype=np.int64)
    targets = _readonly(true_labeled_targets, dtype=np.int64)
    expected = (UNLABELED_EVAL_COUNT, LABEL_COUNT, DIRECTION_COUNT)
    _require(
        true_relation.shape == shuffle_relation.shape == expected
        and truth.shape == (SAMPLE_NUM,)
        and unlabeled.shape == (UNLABELED_EVAL_COUNT,)
        and targets.shape == (LABEL_COUNT,),
        "C3-A0 relation-correctness input boundary mismatch",
    )
    gt_relation = np.ascontiguousarray(
        truth[unlabeled, None] == targets[None, :], dtype=np.bool_
    )
    true_correct = np.ascontiguousarray(
        true_relation == gt_relation[:, :, None], dtype=np.int8
    )
    shuffle_correct = np.ascontiguousarray(
        shuffle_relation == gt_relation[:, :, None], dtype=np.int8
    )
    for value in (gt_relation, true_correct, shuffle_correct):
        value.setflags(write=False)
    return {
        "GT_Relation": gt_relation,
        "RelationCorrect_true": true_correct,
        "RelationCorrect_shuffle": shuffle_correct,
    }


def build_candidate_gt_centers(R, full_GT, unlabeled_ids, class_pred):
    """Build target-class centers, excluding candidate i when i belongs to it."""
    state = _readonly(R, dtype=np.float64)
    truth = _readonly(full_GT, dtype=np.int64)
    unlabeled = _readonly(unlabeled_ids, dtype=np.int64)
    predicted = _readonly(class_pred, dtype=np.int64)
    _require(
        state.shape == (SAMPLE_NUM, CLASS_NUM)
        and truth.shape == (SAMPLE_NUM,)
        and unlabeled.shape == (UNLABELED_EVAL_COUNT,)
        and predicted.shape == (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
        and np.isfinite(state).all()
        and np.all((truth >= 0) & (truth < CLASS_NUM))
        and np.all((predicted >= 0) & (predicted < CLASS_NUM)),
        "C3-A0 GT-center input boundary mismatch",
    )
    class_counts = np.bincount(truth, minlength=CLASS_NUM).astype(np.int64)
    class_sums = np.stack(
        [state[truth == class_id].sum(axis=0) for class_id in range(CLASS_NUM)]
    )
    _require(
        np.all(class_counts >= 2) and np.isfinite(class_sums).all(),
        "C3-A0 each GT class must support leave-one-candidate-out",
    )
    centers = np.empty(
        (UNLABELED_EVAL_COUNT, DIRECTION_COUNT, CLASS_NUM),
        dtype=np.float64,
    )
    for direction_id in range(DIRECTION_COUNT):
        predicted_class = predicted[:, direction_id]
        numerator = np.array(class_sums[predicted_class], copy=True)
        denominator = class_counts[predicted_class].astype(np.int64)
        belongs_to_target = truth[unlabeled] == predicted_class
        numerator[belongs_to_target] -= state[unlabeled[belongs_to_target]]
        denominator[belongs_to_target] -= 1
        _require(
            np.all(denominator > 0),
            "C3-A0 empty leave-one-candidate-out GT center",
        )
        centers[:, direction_id, :] = numerator / denominator[:, None]
    _require(
        np.isfinite(centers).all(),
        "C3-A0 non-finite leave-one-candidate-out GT center",
    )
    centers.setflags(write=False)
    return centers


def build_memory_write_outcomes(
    R,
    full_GT,
    unlabeled_ids,
    class_pred,
    P,
    P_candidate,
):
    state = _readonly(R, dtype=np.float64)
    predicted = _readonly(class_pred, dtype=np.int64)
    prototypes = _readonly(P, dtype=np.float64)
    candidates = _readonly(P_candidate, dtype=np.float64)
    centers = build_candidate_gt_centers(
        state, full_GT, unlabeled_ids, predicted
    )
    expected_candidates = (
        UNLABELED_EVAL_COUNT,
        DIRECTION_COUNT,
        CLASS_NUM,
    )
    _require(
        prototypes.shape == (CLASS_NUM, CLASS_NUM)
        and candidates.shape == expected_candidates,
        "C3-A0 memory-write input boundary mismatch",
    )
    original_distance = np.sum(
        np.square(prototypes[predicted] - centers), axis=2
    )
    candidate_distance = np.sum(
        np.square(candidates - centers), axis=2
    )
    benefit = np.ascontiguousarray(
        original_distance - candidate_distance, dtype=np.float64
    )
    safe = np.ascontiguousarray(benefit > 0.0, dtype=np.int8)
    _require(
        benefit.shape
        == safe.shape
        == (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
        and np.isfinite(benefit).all(),
        "C3-A0 memory-write outcome construction failed",
    )
    benefit.setflags(write=False)
    safe.setflags(write=False)
    return {
        "GT_center_for_candidate": centers,
        "MemoryWriteBenefit": benefit,
        "MemoryWriteSafe": safe,
    }


def weighted_binary_auc_or_none(correct, score, weights):
    outcomes = _readonly(correct, dtype=np.int8)
    values = _readonly(score, dtype=np.float64)
    sample_weights = _readonly(weights, dtype=np.float64)
    _require(
        outcomes.ndim == values.ndim == sample_weights.ndim == 1
        and outcomes.shape == values.shape == sample_weights.shape
        and outcomes.size > 0
        and np.all((outcomes == 0) | (outcomes == 1))
        and np.isfinite(values).all()
        and np.isfinite(sample_weights).all()
        and np.all(sample_weights > 0.0),
        "C3-A0 weighted AUC input boundary mismatch",
    )
    if np.unique(outcomes).size != 2 or np.unique(values).size < 2:
        return None
    return float(
        roc_auc_score(outcomes, values, sample_weight=sample_weights)
    )


def _direction_is_valid(cycle, true_correct, shuffle_correct):
    return bool(
        np.isfinite(cycle).all()
        and np.isfinite(true_correct).all()
        and np.isfinite(shuffle_correct).all()
        and np.unique(cycle).size >= 2
        and np.unique(true_correct).size == 2
        and np.unique(shuffle_correct).size == 2
    )

def spearman_memory_write_benefit_or_none(U_cycle, MemoryWriteBenefit):
    """Return the preregistered benefit correlation or None if invalid."""
    cycle = _readonly(U_cycle, dtype=np.float64)
    benefit = _readonly(MemoryWriteBenefit, dtype=np.float64)
    _require(
        cycle.ndim == benefit.ndim == 1
        and cycle.shape == benefit.shape
        and cycle.size > 0,
        "C3-A0 memory-write correlation input boundary mismatch",
    )
    if np.unique(cycle).size < 2 or np.unique(benefit).size < 2:
        return None
    statistic = float(spearmanr(cycle, benefit).statistic)
    return statistic if np.isfinite(statistic) else None



def analyze_class_metrics(
    U_cycle,
    ClassCorrect_true,
    ClassCorrect_shuffle,
    utility_quintile_assignment,
):
    cycle = _readonly(U_cycle, dtype=np.float64)
    true_correct = _readonly(ClassCorrect_true, dtype=np.int8)
    shuffle_correct = _readonly(ClassCorrect_shuffle, dtype=np.int8)
    bins = _readonly(utility_quintile_assignment, dtype=np.int8)
    expected = (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
    _require(
        cycle.shape == true_correct.shape == shuffle_correct.shape
        == bins.shape == expected,
        "C3-A0 class-metric input boundary mismatch",
    )
    directions = []
    for direction_id in range(DIRECTION_COUNT):
        values = cycle[:, direction_id]
        true_outcomes = true_correct[:, direction_id]
        shuffle_outcomes = shuffle_correct[:, direction_id]
        valid = _direction_is_valid(
            values, true_outcomes, shuffle_outcomes
        )
        auc_true = (
            binary_auc_or_none(true_outcomes, values) if valid else None
        )
        auc_shuffle = (
            binary_auc_or_none(shuffle_outcomes, values) if valid else None
        )
        accuracies = [
            float(np.mean(true_outcomes[bins[:, direction_id] == bin_id]))
            for bin_id in range(UTILITY_QUINTILE_COUNT)
        ]
        directions.append({
            "direction_id": direction_id,
            "joint_valid": valid,
            "AUC_class_true": auc_true,
            "AUC_class_shuffle": auc_shuffle,
            "ClassSpecificityGap": (
                None if not valid else float(auc_true - auc_shuffle)
            ),
            **{
                "ClassAccuracy_Q" + str(index + 1): value
                for index, value in enumerate(accuracies)
            },
            "ClassAccuracy_Q5_minus_Q1": accuracies[-1] - accuracies[0],
        })
    return {
        "direction_count": DIRECTION_COUNT,
        "evaluation_sample_count": UNLABELED_EVAL_COUNT,
        "U_cycle_only": True,
        "directions": directions,
    }


def analyze_relation_metrics(
    U_cycle,
    PredRelation_true,
    PredRelation_shuffle,
    RelationCorrect_true,
    RelationCorrect_shuffle,
    relation_balance_weights_true,
    relation_balance_weights_shuffle,
    utility_quintile_assignment,
):
    cycle = _readonly(U_cycle, dtype=np.float64)
    true_pred = _readonly(PredRelation_true, dtype=np.bool_)
    shuffle_pred = _readonly(PredRelation_shuffle, dtype=np.bool_)
    true_correct = _readonly(RelationCorrect_true, dtype=np.int8)
    shuffle_correct = _readonly(RelationCorrect_shuffle, dtype=np.int8)
    true_weights = _readonly(relation_balance_weights_true, dtype=np.float64)
    shuffle_weights = _readonly(
        relation_balance_weights_shuffle, dtype=np.float64
    )
    bins = _readonly(utility_quintile_assignment, dtype=np.int8)
    edge_shape = (UNLABELED_EVAL_COUNT, LABEL_COUNT, DIRECTION_COUNT)
    _require(
        cycle.shape == bins.shape
        == (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
        and true_pred.shape == shuffle_pred.shape == true_correct.shape
        == shuffle_correct.shape == true_weights.shape
        == shuffle_weights.shape == edge_shape,
        "C3-A0 relation-metric input boundary mismatch",
    )
    directions = []
    for direction_id in range(DIRECTION_COUNT):
        values = cycle[:, direction_id]
        edge_values = np.broadcast_to(
            values[:, None], (UNLABELED_EVAL_COUNT, LABEL_COUNT)
        ).reshape(-1)
        true_outcomes = true_correct[:, :, direction_id].reshape(-1)
        shuffle_outcomes = shuffle_correct[:, :, direction_id].reshape(-1)
        valid = _direction_is_valid(
            edge_values, true_outcomes, shuffle_outcomes
        )
        auc_true = (
            weighted_binary_auc_or_none(
                true_outcomes,
                edge_values,
                true_weights[:, :, direction_id].reshape(-1),
            )
            if valid else None
        )
        auc_shuffle = (
            weighted_binary_auc_or_none(
                shuffle_outcomes,
                edge_values,
                shuffle_weights[:, :, direction_id].reshape(-1),
            )
            if valid else None
        )
        same_accuracies = []
        different_accuracies = []
        balanced_accuracies = []
        for bin_id in range(UTILITY_QUINTILE_COUNT):
            row_mask = bins[:, direction_id] == bin_id
            predicted_same = true_pred[row_mask, :, direction_id]
            outcomes = true_correct[row_mask, :, direction_id]
            _require(
                np.any(predicted_same) and np.any(~predicted_same),
                "C3-A0 relation quintile group is empty",
            )
            same_accuracy = float(np.mean(outcomes[predicted_same]))
            different_accuracy = float(np.mean(outcomes[~predicted_same]))
            same_accuracies.append(same_accuracy)
            different_accuracies.append(different_accuracy)
            balanced_accuracies.append(
                0.5 * (same_accuracy + different_accuracy)
            )
        record = {
            "direction_id": direction_id,
            "joint_valid": valid,
            "U_relation_is_broadcast_U_cycle": True,
            "AUC_relation_true": auc_true,
            "AUC_relation_shuffle": auc_shuffle,
            "RelationSpecificityGap": (
                None if not valid else float(auc_true - auc_shuffle)
            ),
        }
        for index in range(UTILITY_QUINTILE_COUNT):
            label = str(index + 1)
            record["Acc_same_Q" + label] = same_accuracies[index]
            record["Acc_different_Q" + label] = different_accuracies[index]
            record[
                "BalancedRelationAccuracy_Q" + label
            ] = balanced_accuracies[index]
        record["BalancedRelationAccuracy_Q5_minus_Q1"] = (
            balanced_accuracies[-1] - balanced_accuracies[0]
        )
        directions.append(record)
    return {
        "direction_count": DIRECTION_COUNT,
        "evaluation_sample_count": UNLABELED_EVAL_COUNT,
        "pair_utility_constructed": False,
        "U_relation_is_broadcast_U_cycle": True,
        "directions": directions,
    }


def analyze_memory_metrics(
    U_cycle,
    MemoryWriteSafe_true,
    MemoryWriteSafe_shuffle,
    MemoryWriteBenefit_true,
    MemoryWriteBenefit_shuffle,
    utility_quintile_assignment,
):
    cycle = _readonly(U_cycle, dtype=np.float64)
    true_safe = _readonly(MemoryWriteSafe_true, dtype=np.int8)
    shuffle_safe = _readonly(MemoryWriteSafe_shuffle, dtype=np.int8)
    true_benefit = _readonly(MemoryWriteBenefit_true, dtype=np.float64)
    shuffle_benefit = _readonly(
        MemoryWriteBenefit_shuffle, dtype=np.float64
    )
    bins = _readonly(utility_quintile_assignment, dtype=np.int8)
    expected = (UNLABELED_EVAL_COUNT, DIRECTION_COUNT)
    _require(
        cycle.shape == true_safe.shape == shuffle_safe.shape
        == true_benefit.shape == shuffle_benefit.shape == bins.shape
        == expected,
        "C3-A0 memory-metric input boundary mismatch",
    )
    directions = []
    for direction_id in range(DIRECTION_COUNT):
        values = cycle[:, direction_id]
        true_outcomes = true_safe[:, direction_id]
        shuffle_outcomes = shuffle_safe[:, direction_id]
        valid = _direction_is_valid(
            values, true_outcomes, shuffle_outcomes
        )
        benefit_correlation = spearman_memory_write_benefit_or_none(
            values, true_benefit[:, direction_id]
        )
        auc_true = (
            binary_auc_or_none(true_outcomes, values) if valid else None
        )
        auc_shuffle = (
            binary_auc_or_none(shuffle_outcomes, values) if valid else None
        )
        safe_rates = []
        benefit_means = []
        shuffle_benefit_means = []
        for bin_id in range(UTILITY_QUINTILE_COUNT):
            mask = bins[:, direction_id] == bin_id
            safe_rates.append(float(np.mean(true_outcomes[mask])))
            benefit_means.append(
                float(np.mean(true_benefit[mask, direction_id]))
            )
            shuffle_benefit_means.append(
                float(np.mean(shuffle_benefit[mask, direction_id]))
            )
        record = {
            "direction_id": direction_id,
            "joint_valid": valid,
            "AUC_memory_true": auc_true,
            "AUC_memory_shuffle": auc_shuffle,
            "MemorySpecificityGap": (
                None if not valid else float(auc_true - auc_shuffle)
            ),
            "SpearmanMemoryWriteBenefit": benefit_correlation,
            "mean_MemoryWriteBenefit_true": float(
                np.mean(true_benefit[:, direction_id])
            ),
            "mean_MemoryWriteBenefit_shuffle": float(
                np.mean(shuffle_benefit[:, direction_id])
            ),
        }
        for index in range(UTILITY_QUINTILE_COUNT):
            label = str(index + 1)
            record["MemorySafeRate_Q" + label] = safe_rates[index]
            record[
                "mean_MemoryWriteBenefit_true_Q" + label
            ] = benefit_means[index]
            record[
                "mean_MemoryWriteBenefit_shuffle_Q" + label
            ] = shuffle_benefit_means[index]
        record["MemorySafeRate_Q5_minus_Q1"] = (
            safe_rates[-1] - safe_rates[0]
        )
        directions.append(record)
    return {
        "direction_count": DIRECTION_COUNT,
        "evaluation_sample_count": UNLABELED_EVAL_COUNT,
        "benefit_used_in_gate": False,
        "MemoryWriteBenefitCorrelation_used_in_gate": False,
        "persistent_memory_created": False,
        "directions": directions,
    }


def build_postseal_evaluation(pre_gt_arrays, full_GT):
    arrays = pre_gt_arrays
    required = (
        "unlabeled_ids",
        "labeled_targets",
        "U_cycle",
        "R",
        "class_pred_true",
        "class_pred_shuffle",
        "PredRelation_true",
        "PredRelation_shuffle",
        "relation_balance_weights_true",
        "relation_balance_weights_shuffle",
        "utility_quintile_assignment",
        "P_true",
        "P_shuffle",
        "P_candidate_true",
        "P_candidate_shuffle",
    )
    _require(
        all(name in arrays for name in required),
        "C3-A0 post-seal input schema mismatch",
    )
    truth = _readonly(full_GT, dtype=np.int64)
    class_outcomes = build_class_correctness(
        arrays["class_pred_true"],
        arrays["class_pred_shuffle"],
        truth,
        arrays["unlabeled_ids"],
    )
    relation_outcomes = build_relation_correctness(
        arrays["PredRelation_true"],
        arrays["PredRelation_shuffle"],
        truth,
        arrays["unlabeled_ids"],
        arrays["labeled_targets"],
    )
    memory_true = build_memory_write_outcomes(
        arrays["R"],
        truth,
        arrays["unlabeled_ids"],
        arrays["class_pred_true"],
        arrays["P_true"],
        arrays["P_candidate_true"],
    )
    memory_shuffle = build_memory_write_outcomes(
        arrays["R"],
        truth,
        arrays["unlabeled_ids"],
        arrays["class_pred_shuffle"],
        arrays["P_shuffle"],
        arrays["P_candidate_shuffle"],
    )
    unlabeled = arrays["unlabeled_ids"]
    cycle = arrays["U_cycle"][unlabeled]
    bins = arrays["utility_quintile_assignment"]
    class_metrics = analyze_class_metrics(
        cycle,
        class_outcomes["ClassCorrect_true"],
        class_outcomes["ClassCorrect_shuffle"],
        bins,
    )
    relation_metrics = analyze_relation_metrics(
        cycle,
        arrays["PredRelation_true"],
        arrays["PredRelation_shuffle"],
        relation_outcomes["RelationCorrect_true"],
        relation_outcomes["RelationCorrect_shuffle"],
        arrays["relation_balance_weights_true"],
        arrays["relation_balance_weights_shuffle"],
        bins,
    )
    memory_metrics = analyze_memory_metrics(
        cycle,
        memory_true["MemoryWriteSafe"],
        memory_shuffle["MemoryWriteSafe"],
        memory_true["MemoryWriteBenefit"],
        memory_shuffle["MemoryWriteBenefit"],
        bins,
    )
    return {
        "class_metrics": class_metrics,
        "relation_metrics": relation_metrics,
        "memory_metrics": memory_metrics,
        "outcomes": {
            **class_outcomes,
            **relation_outcomes,
            "GT_center_for_candidate_true": memory_true[
                "GT_center_for_candidate"
            ],
            "MemoryWriteBenefit_true": memory_true[
                "MemoryWriteBenefit"
            ],
            "MemoryWriteSafe_true": memory_true["MemoryWriteSafe"],
            "GT_center_for_candidate_shuffle": memory_shuffle[
                "GT_center_for_candidate"
            ],
            "MemoryWriteBenefit_shuffle": memory_shuffle[
                "MemoryWriteBenefit"
            ],
            "MemoryWriteSafe_shuffle": memory_shuffle["MemoryWriteSafe"],
        },
    }


def _mean(records, field):
    if not records:
        return None
    values = np.asarray([record[field] for record in records], dtype=np.float64)
    _require(
        np.isfinite(values).all(),
        "C3-A0 non-finite aggregate metric",
    )
    return float(values.mean())


def _build_action_seed_gate(
    action_name,
    metrics,
    auc_field,
    specificity_field,
    gap_field,
    pass_field,
):
    directions = metrics.get("directions", ())
    _require(
        len(directions) == DIRECTION_COUNT
        and all(
            directions[index].get("direction_id") == index
            for index in range(DIRECTION_COUNT)
        ),
        "C3-A0 seed direction schema mismatch for " + action_name,
    )
    valid = [record for record in directions if record["joint_valid"]]
    auc = np.asarray(
        [record[auc_field] for record in valid], dtype=np.float64
    )
    gaps = np.asarray(
        [record[gap_field] for record in valid], dtype=np.float64
    )
    mean_auc = _mean(valid, auc_field)
    mean_specificity = _mean(valid, specificity_field)
    mean_gap = _mean(valid, gap_field)
    p_value = one_sided_wilcoxon_greater(auc - 0.5)
    conditions = OrderedDict((
        ("minimum_valid_directions", len(valid) >= MIN_VALID_DIRECTIONS_PER_SEED),
        (
            "minimum_AUC_true_above_half_directions",
            int(np.count_nonzero(auc > 0.5)) >= POSITIVE_DIRECTION_MIN_COUNT,
        ),
        (
            "mean_AUC_true_above_half",
            mean_auc is not None and mean_auc > 0.5,
        ),
        (
            "one_sided_wilcoxon_AUC_minus_half_greater",
            p_value is not None and p_value < SIGNIFICANCE_LEVEL,
        ),
        (
            "positive_mean_specificity_gap",
            mean_specificity is not None and mean_specificity > 0.0,
        ),
        (
            "minimum_positive_Q5_minus_Q1_directions",
            int(np.count_nonzero(gaps > 0.0)) >= POSITIVE_DIRECTION_MIN_COUNT,
        ),
        (
            "positive_mean_Q5_minus_Q1_gap",
            mean_gap is not None and mean_gap > 0.0,
        ),
    ))
    return {
        "action": action_name,
        "valid_direction_count": len(valid),
        "AUC_true_above_half_direction_count": int(
            np.count_nonzero(auc > 0.5)
        ),
        "mean_AUC_true": mean_auc,
        "AUC_minus_half_wilcoxon_greater_p": p_value,
        "mean_specificity_gap": mean_specificity,
        "positive_Q5_minus_Q1_direction_count": int(
            np.count_nonzero(gaps > 0.0)
        ),
        "mean_Q5_minus_Q1_gap": mean_gap,
        "wilcoxon_alternative": "greater",
        "SeedGate_conditions": dict(conditions),
        pass_field: bool(all(conditions.values())),
    }


def build_seed_summary(
    seed,
    class_metrics,
    relation_metrics,
    memory_metrics,
):
    active_seed = validate_seed(seed)
    class_summary = _build_action_seed_gate(
        "class",
        class_metrics,
        "AUC_class_true",
        "ClassSpecificityGap",
        "ClassAccuracy_Q5_minus_Q1",
        "CLASS_SEED_PASS",
    )
    relation_summary = _build_action_seed_gate(
        "relation",
        relation_metrics,
        "AUC_relation_true",
        "RelationSpecificityGap",
        "BalancedRelationAccuracy_Q5_minus_Q1",
        "RELATION_SEED_PASS",
    )
    memory_summary = _build_action_seed_gate(
        "memory",
        memory_metrics,
        "AUC_memory_true",
        "MemorySpecificityGap",
        "MemorySafeRate_Q5_minus_Q1",
        "MEMORY_SEED_PASS",
    )
    memory_summary["mean_MemoryWriteBenefit_diagnostic"] = _mean(
        memory_metrics["directions"], "mean_MemoryWriteBenefit_true"
    )
    correlation_records = [
        record for record in memory_metrics["directions"]
        if record.get("SpearmanMemoryWriteBenefit") is not None
    ]
    memory_summary["mean_SpearmanMemoryWriteBenefit"] = _mean(
        correlation_records, "SpearmanMemoryWriteBenefit"
    )
    memory_summary[
        "valid_MemoryWriteBenefitCorrelation_direction_count"
    ] = len(correlation_records)
    memory_summary["MemoryWriteBenefit_used_in_gate"] = False
    memory_summary[
        "MemoryWriteBenefitCorrelation_used_in_gate"
    ] = False
    return {
        "stage": STAGE,
        "seed": active_seed,
        "class": class_summary,
        "relation": relation_summary,
        "memory": memory_summary,
    }


def _mean_seed_field(seed_summaries, action, field):
    values = [
        seed_summaries[seed][action][field] for seed in SEEDS
    ]
    if any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=np.float64)
    _require(
        np.isfinite(array).all(),
        "C3-A0 non-finite multi-seed metric",
    )
    return float(array.mean())


def _build_action_multi_gate(seed_summaries, action, pass_field, result_field):
    pass_count = sum(
        bool(seed_summaries[seed][action][pass_field]) for seed in SEEDS
    )
    aggregate_auc = _mean_seed_field(
        seed_summaries, action, "mean_AUC_true"
    )
    aggregate_specificity = _mean_seed_field(
        seed_summaries, action, "mean_specificity_gap"
    )
    aggregate_gap = _mean_seed_field(
        seed_summaries, action, "mean_Q5_minus_Q1_gap"
    )
    conditions = OrderedDict((
        ("at_least_two_of_three_seed_passes", pass_count >= SEED_PASS_MIN_COUNT),
        (
            "aggregate_mean_AUC_above_half",
            aggregate_auc is not None and aggregate_auc > 0.5,
        ),
        (
            "positive_aggregate_mean_specificity_gap",
            aggregate_specificity is not None and aggregate_specificity > 0.0,
        ),
        (
            "positive_aggregate_mean_Q5_minus_Q1_gap",
            aggregate_gap is not None and aggregate_gap > 0.0,
        ),
    ))
    return {
        "action": action,
        "seed_pass_count": pass_count,
        "aggregate_mean_AUC": aggregate_auc,
        "aggregate_mean_specificity_gap": aggregate_specificity,
        "aggregate_mean_Q5_minus_Q1_gap": aggregate_gap,
        "MultiSeedGate_conditions": dict(conditions),
        result_field: bool(all(conditions.values())),
    }


def build_multiseed_decision(seed_summaries):
    _require(
        tuple(seed_summaries) == SEEDS
        and all(
            seed_summaries[seed].get("seed") == seed for seed in SEEDS
        ),
        "C3-A0 multi-seed set/order mismatch",
    )
    class_result = _build_action_multi_gate(
        seed_summaries, "class", "CLASS_SEED_PASS", "CLASS_MULTI_PASS"
    )
    relation_result = _build_action_multi_gate(
        seed_summaries,
        "relation",
        "RELATION_SEED_PASS",
        "RELATION_MULTI_PASS",
    )
    memory_result = _build_action_multi_gate(
        seed_summaries,
        "memory",
        "MEMORY_SEED_PASS",
        "MEMORY_MULTI_PASS",
    )
    class_pass = class_result["CLASS_MULTI_PASS"]
    relation_pass = relation_result["RELATION_MULTI_PASS"]
    memory_pass = memory_result["MEMORY_MULTI_PASS"]
    if relation_pass:
        primary_decision = RELATION_PRIMARY_DECISION
    elif class_pass:
        primary_decision = CLASS_PRIMARY_DECISION
    else:
        primary_decision = PRIMARY_INTERFACE_FAIL_DECISION
    memory_eligible = bool(memory_pass and (class_pass or relation_pass))
    return {
        "summary": {
            "stage": STAGE,
            "seeds": list(SEEDS),
            "class": class_result,
            "relation": relation_result,
            "memory": memory_result,
            "seed_summaries": {
                str(seed): seed_summaries[seed] for seed in SEEDS
            },
        },
        "decision": {
            "stage": STAGE,
            "CLASS_MULTI_PASS": class_pass,
            "RELATION_MULTI_PASS": relation_pass,
            "MEMORY_MULTI_PASS": memory_pass,
            "next_primary_stage": primary_decision,
            "MEMORY_ELIGIBLE": memory_eligible,
            "memory_decision": (
                MEMORY_ELIGIBLE_DECISION
                if memory_eligible else MEMORY_CLOSED_DECISION
            ),
            "memory_cannot_rescue_primary_route": True,
            "branch_decision_precommitted_before_formal_results": True,
            "post_hoc_gate_change": False,
        },
    }
