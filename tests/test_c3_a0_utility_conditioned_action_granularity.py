import ast
import inspect
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from experiments.cyclic_utility import c2_a0_sparse_label_utility_protocol as parent
from experiments.cyclic_utility import c3_a0_utility_conditioned_action_granularity_protocol as c3
from experiments.cyclic_utility import evaluate_c3_a0_utility_conditioned_action_granularity as evaluate
from experiments.cyclic_utility import summarize_c3_a0_utility_conditioned_action_granularity as summarize
from weak_quality import ndarray_sha256


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def formal_inputs():
    rows = np.arange(c3.SAMPLE_NUM, dtype=np.int64)[:, None]
    directions = np.arange(c3.DIRECTION_COUNT, dtype=np.int64)[None, :]
    y_gen = (rows + 3 * directions) % c3.CLASS_NUM
    U_cycle = (
        (17 * rows + 31 * directions) % 997
    ).astype(np.float64) / 996.0
    sample_ids = c3.canonical_sample_ids()
    labeled_ids = c3.fixed_labeled_ids()
    unlabeled_ids = np.setdiff1d(sample_ids, labeled_ids)
    return {
        "sample_ids": sample_ids,
        "unlabeled_ids": unlabeled_ids,
        "labeled_ids": labeled_ids,
        "labeled_targets": c3.fixed_labeled_targets(),
        "shuffled_labeled_targets": c3.fixed_shuffled_targets(),
        "U_cycle": U_cycle,
        "y_gen": y_gen,
        "R": c3.build_vote_semantic_state(y_gen),
    }


@pytest.fixture(scope="module")
def actions(formal_inputs):
    return c3.build_pre_gt_actions(**formal_inputs)


@pytest.fixture(scope="module")
def full_GT():
    return np.arange(c3.SAMPLE_NUM, dtype=np.int64) % c3.CLASS_NUM


@pytest.fixture(scope="module")
def postseal(actions, full_GT):
    return c3.build_postseal_evaluation(actions, full_GT)


def _called_names(*paths):
    called = set()
    for path in paths:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        called.update(
            node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
        )
    return called


def _direction_metrics(action, valid_count=20):
    specifications = {
        "class": (
            "AUC_class_true",
            "ClassSpecificityGap",
            "ClassAccuracy_Q5_minus_Q1",
        ),
        "relation": (
            "AUC_relation_true",
            "RelationSpecificityGap",
            "BalancedRelationAccuracy_Q5_minus_Q1",
        ),
        "memory": (
            "AUC_memory_true",
            "MemorySpecificityGap",
            "MemorySafeRate_Q5_minus_Q1",
        ),
    }
    auc_name, specificity_name, gap_name = specifications[action]
    directions = []
    for direction_id in range(c3.DIRECTION_COUNT):
        valid = direction_id < valid_count
        record = {
            "direction_id": direction_id,
            "joint_valid": valid,
            auc_name: 0.7 if valid else None,
            specificity_name: 0.1 if valid else None,
            gap_name: 0.2 if valid else None,
        }
        if action == "memory":
            record["mean_MemoryWriteBenefit_true"] = 0.01
            record["SpearmanMemoryWriteBenefit"] = 0.3 if valid else None
        directions.append(record)
    return {"directions": directions}


def _seed_summary(seed, class_pass, relation_pass, memory_pass):
    def action_record(passed):
        return {
            "mean_AUC_true": 0.7,
            "mean_specificity_gap": 0.1,
            "mean_Q5_minus_Q1_gap": 0.2,
            "CLASS_SEED_PASS": passed,
            "RELATION_SEED_PASS": passed,
            "MEMORY_SEED_PASS": passed,
        }

    return {
        "stage": c3.STAGE,
        "seed": seed,
        "class": {
            **action_record(class_pass),
            "CLASS_SEED_PASS": class_pass,
        },
        "relation": {
            **action_record(relation_pass),
            "RELATION_SEED_PASS": relation_pass,
        },
        "memory": {
            **action_record(memory_pass),
            "MEMORY_SEED_PASS": memory_pass,
        },
    }


def _multiseed(class_pass, relation_pass, memory_pass):
    summaries = OrderedDict(
        (
            seed,
            _seed_summary(seed, class_pass, relation_pass, memory_pass),
        )
        for seed in c3.SEEDS
    )
    return c3.build_multiseed_decision(summaries)


def test_01_all_parent_fail_closed_hashes_pass():
    audit = evaluate.verify_frozen_parent_sources()
    assert audit["C2_B0_frozen_hashes_pass"] is True
    assert audit["C2_C0_frozen_hashes_pass"] is True
    assert audit["C2_C1_frozen_hashes_pass"] is True
    assert audit["C2_C2_A0_frozen_hashes_pass"] is True
    assert audit["all_frozen_parent_hashes_pass"] is True
    assert tuple(audit["manifests"]) == tuple(
        evaluate.FROZEN_PARENT_MANIFESTS
    )


def test_02_sparse_semantic_mapping_is_exact_parent_alias():
    assert (
        c3.build_sparse_anchors_and_mapping
        is parent.build_sparse_anchors_and_mapping
    )


def test_03_full_gt_cannot_enter_mapping_or_pre_gt_builder():
    assert "full_GT" not in inspect.signature(
        c3.build_semantic_action_path
    ).parameters
    assert "full_GT" not in inspect.signature(
        c3.build_pre_gt_actions
    ).parameters
    forbidden = {
        "load_full_ground_truth_after_action_seal",
        "build_postseal_evaluation",
    }
    source = inspect.getsource(evaluate.build_pre_gt_bundle)
    assert all(name not in source for name in forbidden)


def test_04_true_and_shuffle_mapping_are_independent(actions):
    assert actions["shuffle_mapping_independently_rebuilt"] is True
    assert actions["class_pred_true"] is not actions["class_pred_shuffle"]
    assert actions["sparse_mapping_true"] is not actions["sparse_mapping_shuffle"]
    assert not np.array_equal(
        actions["sparse_mapping_true"], actions["sparse_mapping_shuffle"]
    )
    assert actions["true_mapping_fit_full_GT_used"] is False
    assert actions["shuffle_mapping_fit_full_GT_used"] is False


def test_05_class_action_shapes_and_unlabeled_boundary(actions):
    assert actions["class_pred_true"].shape == (1386, 20)
    assert actions["class_pred_shuffle"].shape == (1386, 20)
    assert actions["unlabeled_ids"].shape == (1386,)
    assert np.intersect1d(
        actions["unlabeled_ids"], actions["labeled_ids"]
    ).size == 0


def test_06_class_correctness_formula_is_exact(actions, full_GT):
    result = c3.build_class_correctness(
        actions["class_pred_true"],
        actions["class_pred_shuffle"],
        full_GT,
        actions["unlabeled_ids"],
    )
    expected_true = (
        actions["class_pred_true"]
        == full_GT[actions["unlabeled_ids"], None]
    ).astype(np.int8)
    expected_shuffle = (
        actions["class_pred_shuffle"]
        == full_GT[actions["unlabeled_ids"], None]
    ).astype(np.int8)
    assert np.array_equal(result["ClassCorrect_true"], expected_true)
    assert np.array_equal(
        result["ClassCorrect_shuffle"], expected_shuffle
    )


def test_07_relation_shapes_and_independent_reconstruction(actions):
    expected = (1386, 14, 20)
    assert actions["PredRelation_true"].shape == expected
    assert actions["PredRelation_shuffle"].shape == expected
    assert actions["PredRelation_true"] is not actions["PredRelation_shuffle"]
    assert actions["shuffle_relation_independently_rebuilt"] is True
    rebuilt_true = (
        actions["class_pred_true"][:, None, :]
        == actions["labeled_targets"][None, :, None]
    )
    rebuilt_shuffle = (
        actions["class_pred_shuffle"][:, None, :]
        == actions["shuffled_labeled_targets"][None, :, None]
    )
    assert np.array_equal(actions["PredRelation_true"], rebuilt_true)
    assert np.array_equal(actions["PredRelation_shuffle"], rebuilt_shuffle)


def test_08_relation_balance_weights_are_exact(actions):
    for pred_name, weight_name in (
        ("PredRelation_true", "relation_balance_weights_true"),
        ("PredRelation_shuffle", "relation_balance_weights_shuffle"),
    ):
        predicted = actions[pred_name]
        weights = actions[weight_name]
        assert weights.shape == predicted.shape
        for direction_id in range(c3.DIRECTION_COUNT):
            same = predicted[:, :, direction_id]
            assert np.sum(
                weights[:, :, direction_id][same]
            ) == pytest.approx(0.5)
            assert np.sum(
                weights[:, :, direction_id][~same]
            ) == pytest.approx(0.5)


def test_09_relation_weights_cannot_use_gt():
    assert tuple(
        inspect.signature(c3.build_relation_balance_weights).parameters
    ) == ("predicted_relation",)


def test_10_gt_relation_and_relation_correctness_are_exact(actions, full_GT):
    result = c3.build_relation_correctness(
        actions["PredRelation_true"],
        actions["PredRelation_shuffle"],
        full_GT,
        actions["unlabeled_ids"],
        actions["labeled_targets"],
    )
    expected_gt = (
        full_GT[actions["unlabeled_ids"], None]
        == actions["labeled_targets"][None, :]
    )
    assert np.array_equal(result["GT_Relation"], expected_gt)
    assert np.array_equal(
        result["RelationCorrect_true"],
        (
            actions["PredRelation_true"]
            == expected_gt[:, :, None]
        ).astype(np.int8),
    )
    assert np.array_equal(
        result["RelationCorrect_shuffle"],
        (
            actions["PredRelation_shuffle"]
            == expected_gt[:, :, None]
        ).astype(np.int8),
    )


def test_11_relation_utility_is_only_broadcast_u_cycle(actions, postseal):
    record = postseal["relation_metrics"]
    assert record["U_relation_is_broadcast_U_cycle"] is True
    assert record["pair_utility_constructed"] is False
    direction_id = 0
    expected = np.broadcast_to(
        actions["U_cycle"][actions["unlabeled_ids"], direction_id, None],
        (c3.UNLABELED_EVAL_COUNT, c3.LABEL_COUNT),
    )
    assert np.array_equal(expected[:, 0], expected[:, -1])


def test_12_no_pair_utility_primitive_exists():
    local_names = {
        name for name, value in vars(c3).items()
        if inspect.isfunction(value) and value.__module__ == c3.__name__
    }
    assert {
        "pair_utility",
        "relation_utility",
        "build_pair_utility",
    }.isdisjoint(local_names)


def test_13_utility_quintiles_use_only_u_cycle_and_sample_id():
    assert tuple(
        inspect.signature(c3.utility_equal_count_quintiles).parameters
    ) == ("U_cycle", "sample_ids")
    ids = np.arange(c3.UNLABELED_EVAL_COUNT, dtype=np.int64)[::-1]
    cycle = np.zeros(
        (c3.UNLABELED_EVAL_COUNT, c3.DIRECTION_COUNT), dtype=np.float64
    )
    assignments = c3.utility_equal_count_quintiles(cycle, ids)
    q1_ids = ids[assignments[:, 0] == 0]
    expected_size = len(
        np.array_split(ids, c3.UTILITY_QUINTILE_COUNT)[0]
    )
    assert np.array_equal(
        np.sort(q1_ids), np.arange(expected_size, dtype=np.int64)
    )


def test_14_exactly_five_approximately_equal_bins(actions):
    bins = actions["utility_quintile_assignment"]
    assert set(np.unique(bins)) == set(range(5))
    for direction_id in range(c3.DIRECTION_COUNT):
        sizes = np.bincount(bins[:, direction_id], minlength=5)
        assert sizes.sum() == 1386
        assert sizes.max() - sizes.min() <= 1


def test_15_r_and_prototypes_are_exact_sparse_anchor_means(actions):
    assert actions["R"].shape == (1400, 7)
    assert actions["P_true"].shape == (7, 7)
    assert actions["P_shuffle"].shape == (7, 7)
    for targets_name, prototype_name in (
        ("labeled_targets", "P_true"),
        ("shuffled_labeled_targets", "P_shuffle"),
    ):
        targets = actions[targets_name]
        assert np.array_equal(
            np.bincount(targets, minlength=7), np.full(7, 2)
        )
        expected = np.stack([
            actions["R"][
                actions["labeled_ids"][targets == class_id]
            ].mean(axis=0)
            for class_id in range(7)
        ])
        assert np.array_equal(actions[prototype_name], expected)
    assert actions["P_true"] is not actions["P_shuffle"]
    assert actions["shuffle_prototype_independently_rebuilt"] is True


def test_16_one_step_candidate_formula_is_exact(actions):
    rows = actions["R"][actions["unlabeled_ids"], None, :]
    expected_true = (
        2.0 * actions["P_true"][actions["class_pred_true"]] + rows
    ) / 3.0
    expected_shuffle = (
        2.0 * actions["P_shuffle"][actions["class_pred_shuffle"]] + rows
    ) / 3.0
    assert actions["P_candidate_true"].shape == (1386, 20, 7)
    assert actions["P_candidate_shuffle"].shape == (1386, 20, 7)
    assert np.array_equal(actions["P_candidate_true"], expected_true)
    assert np.array_equal(actions["P_candidate_shuffle"], expected_shuffle)


def test_17_candidate_build_does_not_accumulate_or_update_state(formal_inputs):
    state_before = formal_inputs["R"].copy()
    result = c3.build_pre_gt_actions(**formal_inputs)
    assert np.array_equal(formal_inputs["R"], state_before)
    assert result["shuffle_candidate_independently_rebuilt"] is True
    assert "memory_state" not in result


def test_18_gt_center_leave_one_candidate_out_is_exact(actions, full_GT):
    predicted = np.array(actions["class_pred_true"], copy=True)
    row_id = int(actions["unlabeled_ids"][0])
    own_class = int(full_GT[row_id])
    other_class = (own_class + 1) % c3.CLASS_NUM
    predicted[0, 0] = own_class
    predicted[0, 1] = other_class
    centers = c3.build_candidate_gt_centers(
        actions["R"], full_GT, actions["unlabeled_ids"], predicted
    )
    own_rows = np.flatnonzero(full_GT == own_class)
    own_rows = own_rows[own_rows != row_id]
    expected_own = actions["R"][own_rows].mean(axis=0)
    expected_other = actions["R"][full_GT == other_class].mean(axis=0)
    assert np.allclose(centers[0, 0], expected_own)
    assert np.allclose(centers[0, 1], expected_other)


def test_19_memory_benefit_and_safe_formulas_are_exact(actions, full_GT):
    result = c3.build_memory_write_outcomes(
        actions["R"],
        full_GT,
        actions["unlabeled_ids"],
        actions["class_pred_true"],
        actions["P_true"],
        actions["P_candidate_true"],
    )
    predicted = actions["class_pred_true"]
    centers = result["GT_center_for_candidate"]
    expected = np.sum(
        np.square(actions["P_true"][predicted] - centers), axis=2
    ) - np.sum(
        np.square(actions["P_candidate_true"] - centers), axis=2
    )
    assert np.allclose(result["MemoryWriteBenefit"], expected)
    assert np.array_equal(
        result["MemoryWriteSafe"], (expected > 0.0).astype(np.int8)
    )


def test_20_class_auc_specificity_and_q5_q1_are_exact(actions):
    count = c3.UNLABELED_EVAL_COUNT
    cycle = np.tile(
        np.linspace(0.0, 1.0, count)[:, None],
        (1, c3.DIRECTION_COUNT),
    )
    true_correct = np.tile(
        (np.arange(count) >= count // 2).astype(np.int8)[:, None],
        (1, c3.DIRECTION_COUNT),
    )
    shuffle_correct = np.tile(
        (np.arange(count) % 2).astype(np.int8)[:, None],
        (1, c3.DIRECTION_COUNT),
    )
    bins = c3.utility_equal_count_quintiles(
        cycle, actions["unlabeled_ids"]
    )
    metrics = c3.analyze_class_metrics(
        cycle, true_correct, shuffle_correct, bins
    )
    record = metrics["directions"][0]
    expected_true = roc_auc_score(true_correct[:, 0], cycle[:, 0])
    expected_shuffle = roc_auc_score(
        shuffle_correct[:, 0], cycle[:, 0]
    )
    assert record["AUC_class_true"] == expected_true
    assert record["AUC_class_shuffle"] == expected_shuffle
    assert record["ClassSpecificityGap"] == (
        expected_true - expected_shuffle
    )
    assert record["ClassAccuracy_Q5_minus_Q1"] == (
        record["ClassAccuracy_Q5"] - record["ClassAccuracy_Q1"]
    )


def test_21_weighted_relation_auc_specificity_and_gap_are_exact(actions):
    count = c3.UNLABELED_EVAL_COUNT
    cycle = np.tile(
        np.linspace(0.0, 1.0, count)[:, None],
        (1, c3.DIRECTION_COUNT),
    )
    true_correct = np.broadcast_to(
        (cycle[:, None, :] > 0.5),
        (count, c3.LABEL_COUNT, c3.DIRECTION_COUNT),
    ).astype(np.int8)
    shuffle_correct = np.broadcast_to(
        (np.arange(count)[:, None, None] % 2),
        (count, c3.LABEL_COUNT, c3.DIRECTION_COUNT),
    ).astype(np.int8)
    bins = c3.utility_equal_count_quintiles(
        cycle, actions["unlabeled_ids"]
    )
    metrics = c3.analyze_relation_metrics(
        cycle,
        actions["PredRelation_true"],
        actions["PredRelation_shuffle"],
        true_correct,
        shuffle_correct,
        actions["relation_balance_weights_true"],
        actions["relation_balance_weights_shuffle"],
        bins,
    )
    record = metrics["directions"][0]
    edge_cycle = np.broadcast_to(
        cycle[:, 0, None], (count, c3.LABEL_COUNT)
    ).reshape(-1)
    expected_true = roc_auc_score(
        true_correct[:, :, 0].reshape(-1),
        edge_cycle,
        sample_weight=actions[
            "relation_balance_weights_true"
        ][:, :, 0].reshape(-1),
    )
    expected_shuffle = roc_auc_score(
        shuffle_correct[:, :, 0].reshape(-1),
        edge_cycle,
        sample_weight=actions[
            "relation_balance_weights_shuffle"
        ][:, :, 0].reshape(-1),
    )
    assert record["AUC_relation_true"] == expected_true
    assert record["AUC_relation_shuffle"] == expected_shuffle
    assert record["RelationSpecificityGap"] == (
        expected_true - expected_shuffle
    )
    assert record["BalancedRelationAccuracy_Q5_minus_Q1"] == (
        record["BalancedRelationAccuracy_Q5"]
        - record["BalancedRelationAccuracy_Q1"]
    )


def test_22_memory_auc_specificity_and_gap_are_exact(actions):
    count = c3.UNLABELED_EVAL_COUNT
    cycle = np.tile(
        np.linspace(0.0, 1.0, count)[:, None],
        (1, c3.DIRECTION_COUNT),
    )
    true_safe = np.tile(
        (np.arange(count) >= count // 2).astype(np.int8)[:, None],
        (1, c3.DIRECTION_COUNT),
    )
    shuffle_safe = np.tile(
        (np.arange(count) % 2).astype(np.int8)[:, None],
        (1, c3.DIRECTION_COUNT),
    )
    true_benefit = true_safe.astype(np.float64) - 0.25
    shuffle_benefit = shuffle_safe.astype(np.float64) - 0.25
    bins = c3.utility_equal_count_quintiles(
        cycle, actions["unlabeled_ids"]
    )
    metrics = c3.analyze_memory_metrics(
        cycle,
        true_safe,
        shuffle_safe,
        true_benefit,
        shuffle_benefit,
        bins,
    )
    record = metrics["directions"][0]
    expected_true = roc_auc_score(true_safe[:, 0], cycle[:, 0])
    expected_shuffle = roc_auc_score(shuffle_safe[:, 0], cycle[:, 0])
    assert record["AUC_memory_true"] == expected_true
    assert record["AUC_memory_shuffle"] == expected_shuffle
    assert record["MemorySpecificityGap"] == (
        expected_true - expected_shuffle
    )
    assert record["MemorySafeRate_Q5_minus_Q1"] == (
        record["MemorySafeRate_Q5"] - record["MemorySafeRate_Q1"]
    )


def test_23_class_seed_gate_is_exact():
    summary = c3.build_seed_summary(
        20,
        _direction_metrics("class"),
        _direction_metrics("relation"),
        _direction_metrics("memory"),
    )
    assert all(summary["class"]["SeedGate_conditions"].values())
    assert summary["class"]["CLASS_SEED_PASS"] is True
    failed = c3.build_seed_summary(
        20,
        _direction_metrics("class", valid_count=15),
        _direction_metrics("relation"),
        _direction_metrics("memory"),
    )
    assert failed["class"]["CLASS_SEED_PASS"] is False


def test_24_relation_seed_gate_is_exact():
    summary = c3.build_seed_summary(
        20,
        _direction_metrics("class"),
        _direction_metrics("relation"),
        _direction_metrics("memory"),
    )
    assert all(summary["relation"]["SeedGate_conditions"].values())
    assert summary["relation"]["RELATION_SEED_PASS"] is True
    failed = c3.build_seed_summary(
        20,
        _direction_metrics("class"),
        _direction_metrics("relation", valid_count=15),
        _direction_metrics("memory"),
    )
    assert failed["relation"]["RELATION_SEED_PASS"] is False


def test_25_memory_seed_gate_is_exact_and_benefit_is_diagnostic_only():
    summary = c3.build_seed_summary(
        20,
        _direction_metrics("class"),
        _direction_metrics("relation"),
        _direction_metrics("memory"),
    )
    assert all(summary["memory"]["SeedGate_conditions"].values())
    assert summary["memory"]["MEMORY_SEED_PASS"] is True
    assert summary["memory"]["MemoryWriteBenefit_used_in_gate"] is False
    first_conditions = summary["memory"]["SeedGate_conditions"]
    changed = _direction_metrics("memory")
    for record in changed["directions"]:
        record["mean_MemoryWriteBenefit_true"] = -1000.0
    second = c3.build_seed_summary(
        20,
        _direction_metrics("class"),
        _direction_metrics("relation"),
        changed,
    )
    assert second["memory"]["SeedGate_conditions"] == first_conditions


def test_26_all_three_multiseed_gates_are_exact():
    result = _multiseed(True, True, True)
    assert result["summary"]["class"]["CLASS_MULTI_PASS"] is True
    assert result["summary"]["relation"]["RELATION_MULTI_PASS"] is True
    assert result["summary"]["memory"]["MEMORY_MULTI_PASS"] is True
    assert all(
        result["summary"]["class"]["MultiSeedGate_conditions"].values()
    )


def test_27_relation_primary_branch_is_precommitted():
    result = _multiseed(True, True, False)["decision"]
    assert result["next_primary_stage"] == c3.RELATION_PRIMARY_DECISION
    assert (
        result["branch_decision_precommitted_before_formal_results"]
        is True
    )


def test_28_class_primary_branch_is_precommitted():
    result = _multiseed(True, False, False)["decision"]
    assert result["next_primary_stage"] == c3.CLASS_PRIMARY_DECISION


def test_29_failed_primary_interface_pivots_without_memory():
    result = _multiseed(False, False, False)["decision"]
    assert result["next_primary_stage"] == c3.PRIMARY_INTERFACE_FAIL_DECISION
    assert result["MEMORY_ELIGIBLE"] is False


def test_30_memory_is_eligible_only_with_a_passing_primary_action():
    relation = _multiseed(False, True, True)["decision"]
    class_action = _multiseed(True, False, True)["decision"]
    assert relation["MEMORY_ELIGIBLE"] is True
    assert class_action["MEMORY_ELIGIBLE"] is True
    assert relation["memory_decision"] == c3.MEMORY_ELIGIBLE_DECISION


def test_31_failed_memory_gate_keeps_memory_closed():
    result = _multiseed(True, True, False)["decision"]
    assert result["MEMORY_ELIGIBLE"] is False
    assert result["memory_decision"] == c3.MEMORY_CLOSED_DECISION


def test_32_memory_cannot_rescue_a_failed_primary_route():
    result = _multiseed(False, False, True)["decision"]
    assert result["MEMORY_MULTI_PASS"] is True
    assert result["MEMORY_ELIGIBLE"] is False
    assert result["memory_cannot_rescue_primary_route"] is True
    assert result["next_primary_stage"] == c3.PRIMARY_INTERFACE_FAIL_DECISION


def test_33_no_gt_or_posthoc_input_before_durable_seal():
    preseal_functions = (
        evaluate.load_c0_inputs_before_gt,
        evaluate.load_fixed_sparse_labels,
        evaluate.build_pre_gt_bundle,
        evaluate.persist_action_bundle,
    )
    forbidden = {
        "load_full_ground_truth_after_action_seal",
        "build_postseal_evaluation",
        "load_c0_audit_after_action_seal",
        "build_bridge_correctness_after_action_seal",
    }
    assert all(
        forbidden.isdisjoint({
            node.func.id
            if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(ast.parse(inspect.getsource(function)))
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
        })
        for function in preseal_functions
    )


def test_34_durable_npz_reload_hash_and_seal_boundaries(
    tmp_path, formal_inputs
):
    provenance = {
        "GT_loaded": False,
        "c0_audit_loaded": False,
        "Bridge_correctness_loaded": False,
        "corruption_mask_loaded": False,
        "C0_artifact_file_sha256": "a" * 64,
        "C0_prediction_seal_file_sha256": "b" * 64,
    }
    split = {
        "labeled_sample_ids": formal_inputs["labeled_ids"],
        "labeled_targets": formal_inputs["labeled_targets"],
        "shuffled_labeled_targets": formal_inputs[
            "shuffled_labeled_targets"
        ],
        "unlabeled_ids": formal_inputs["unlabeled_ids"],
        "label_split_sha256": "unit-test-split",
    }
    frozen = {"all_frozen_parent_hashes_pass": True}
    c0_arrays = {
        "sample_ids": formal_inputs["sample_ids"],
        "U_cycle": formal_inputs["U_cycle"],
        "y_gen": formal_inputs["y_gen"],
    }
    bundle = evaluate.build_pre_gt_bundle(
        20, c0_arrays, provenance, split, frozen
    )
    persisted = evaluate.persist_action_bundle(
        bundle, tmp_path / "sealed"
    )
    seal = persisted["seal"]
    assert tuple(bundle["arrays"]) == evaluate.PRE_GT_ARRAY_NAMES
    assert seal["NPZ_saved_fsynced_reloaded_hash_verified"] is True
    false_boundaries = (
        "GT_loaded_before_action_seal",
        "C0_audit_loaded_before_action_seal",
        "Bridge_loaded_before_action_seal",
        "GT_used_for_class_action",
        "GT_used_for_relation_action",
        "GT_used_for_relation_weights",
        "GT_used_for_utility_bins",
        "GT_used_for_memory_candidate",
        "training_performed",
        "model_forward_called",
        "optimizer_created",
        "optimizer_step_called",
        "backward_called",
    )
    assert all(seal[name] is False for name in false_boundaries)
    with np.load(persisted["npz_path"], allow_pickle=False) as archive:
        assert tuple(archive.files) == evaluate.PRE_GT_ARRAY_NAMES
        for name in evaluate.PRE_GT_ARRAY_NAMES:
            assert np.array_equal(archive[name], bundle["arrays"][name])
            assert (
                ndarray_sha256(archive[name])
                == seal["arrays"][name]["logical_sha256"]
            )


def test_35_no_model_optimizer_backward_or_training_calls():
    called = _called_names(
        c3.__file__, evaluate.__file__, summarize.__file__
    )
    assert {
        "forward",
        "optimizer",
        "step",
        "backward",
        "train",
        "fit_transform",
    }.isdisjoint(called)


def test_36_no_forbidden_selector_or_new_utility_surface():
    local_names = {
        name.lower()
        for name, value in vars(c3).items()
        if inspect.isfunction(value) and value.__module__ == c3.__name__
    }
    forbidden = {
        "u_tilde",
        "ess",
        "inverse_ess",
        "reliability",
        "confidence",
        "threshold",
        "top_k",
        "temperature",
        "alpha",
        "beta",
        "lambda",
        "mlp",
        "learned_selector",
        "parameter_sweep",
    }
    assert forbidden.isdisjoint(local_names)
    assert tuple(
        name for name in evaluate.PRE_GT_ARRAY_NAMES
        if name.startswith("U")
    ) == ("U_cycle",)


def test_37_no_actual_memory_bank_is_created(actions, postseal):
    assert actions["P_candidate_true"].shape == (1386, 20, 7)
    assert postseal["memory_metrics"]["persistent_memory_created"] is False
    assert "memory_bank" not in vars(c3)
    assert "updated_memory" not in actions


def test_38_exactly_four_c3_a0_python_files_exist():
    expected = {
        ROOT / (
            "experiments/cyclic_utility/"
            "c3_a0_utility_conditioned_action_granularity_protocol.py"
        ),
        ROOT / (
            "experiments/cyclic_utility/"
            "evaluate_c3_a0_utility_conditioned_action_granularity.py"
        ),
        ROOT / (
            "experiments/cyclic_utility/"
            "summarize_c3_a0_utility_conditioned_action_granularity.py"
        ),
        ROOT / (
            "tests/test_c3_a0_utility_conditioned_action_granularity.py"
        ),
    }
    discovered = set(
        ROOT.glob("experiments/cyclic_utility/*c3_a0*.py")
    )
    discovered.update(ROOT.glob("tests/*c3_a0*.py"))
    assert discovered == expected


def test_39_output_schemas_and_fixed_seed_paths_are_exact():
    assert evaluate.PER_SEED_OUTPUT_FILES == (
        "c3_a0_action_pre_gt.npz",
        "c3_a0_action_seal.json",
        "c3_a0_class_metrics.json",
        "c3_a0_relation_metrics.json",
        "c3_a0_memory_write_metrics.json",
        "c3_a0_seed_summary.json",
        "c3_a0_audit.json",
    )
    assert summarize.MULTISEED_OUTPUT_FILES == (
        "c3_a0_multiseed_summary.json",
        "c3_a0_multiseed_decision.json",
        "c3_a0_multiseed_audit.json",
    )
    assert tuple(c3.SEEDS) == (20, 30, 50)
    for seed in c3.SEEDS:
        assert "seed" + str(seed) in str(summarize.default_seed_dir(seed))


def test_40_u_cycle_is_bitwise_unchanged(formal_inputs):
    before = formal_inputs["U_cycle"].copy()
    result = c3.build_pre_gt_actions(**formal_inputs)
    assert np.array_equal(formal_inputs["U_cycle"], before)
    assert np.array_equal(result["U_cycle"], before)
    assert result["U_cycle"].flags.writeable is False


def test_41_precommitted_decision_constants_are_exact():
    assert c3.RELATION_PRIMARY_DECISION == (
        "C3-B0 RELATION-LEVEL SEMANTIC ACTION PILOT"
    )
    assert c3.CLASS_PRIMARY_DECISION == (
        "C3-B0 CLASS-LEVEL UTILITY-CONDITIONED PSEUDO ACTION"
    )
    assert "PIVOT WEAK-QUANTITY INTERFACE OR BASELINE" in (
        c3.PRIMARY_INTERFACE_FAIL_DECISION
    )
    assert "DO NOT ADD MEMORY" in c3.PRIMARY_INTERFACE_FAIL_DECISION


def test_42_existing_scientific_files_match_all_parent_manifests():
    audit = evaluate.verify_frozen_parent_sources()
    assert all(
        record["all_entries_exact_match"]
        for record in audit["manifests"].values()
    )


def test_43_spearman_memory_write_benefit_exact_formula():
    cycle = np.asarray([0.4, 0.1, 0.3, 0.2], dtype=np.float64)
    benefit = np.asarray([2.0, 4.0, 1.0, 3.0], dtype=np.float64)
    expected = float(spearmanr(cycle, benefit).statistic)
    assert c3.spearman_memory_write_benefit_or_none(
        cycle, benefit
    ) == expected


def test_44_positive_monotonic_benefit_has_rho_positive_one():
    values = np.arange(8, dtype=np.float64)
    assert c3.spearman_memory_write_benefit_or_none(
        values, values
    ) == 1.0


def test_45_negative_monotonic_benefit_has_rho_negative_one():
    values = np.arange(8, dtype=np.float64)
    assert c3.spearman_memory_write_benefit_or_none(
        values, values[::-1]
    ) == -1.0


def test_46_constant_utility_has_no_benefit_correlation():
    assert c3.spearman_memory_write_benefit_or_none(
        np.ones(8), np.arange(8)
    ) is None


def test_47_constant_benefit_has_no_benefit_correlation():
    assert c3.spearman_memory_write_benefit_or_none(
        np.arange(8), np.ones(8)
    ) is None


def test_48_nonfinite_correlation_is_none_not_zero(monkeypatch):
    def nonfinite_spearman(*args):
        return SimpleNamespace(statistic=np.nan)

    monkeypatch.setattr(c3, "spearmanr", nonfinite_spearman)
    result = c3.spearman_memory_write_benefit_or_none(
        np.arange(8), np.arange(8)
    )
    assert result is None
    assert result != 0


def test_49_mean_spearman_uses_only_non_none_directions():
    memory = _direction_metrics("memory")
    for record in memory["directions"]:
        record["SpearmanMemoryWriteBenefit"] = None
    memory["directions"][0]["SpearmanMemoryWriteBenefit"] = 0.6
    memory["directions"][3]["SpearmanMemoryWriteBenefit"] = -0.2
    summary = c3.build_seed_summary(
        20,
        _direction_metrics("class"),
        _direction_metrics("relation"),
        memory,
    )
    assert summary["memory"][
        "mean_SpearmanMemoryWriteBenefit"
    ] == pytest.approx(0.2)


def test_50_valid_benefit_correlation_direction_count_is_exact():
    memory = _direction_metrics("memory")
    for index, record in enumerate(memory["directions"]):
        record["SpearmanMemoryWriteBenefit"] = (
            0.1 if index < 7 else None
        )
    summary = c3.build_seed_summary(
        20,
        _direction_metrics("class"),
        _direction_metrics("relation"),
        memory,
    )
    assert summary["memory"][
        "valid_MemoryWriteBenefitCorrelation_direction_count"
    ] == 7


def test_51_benefit_correlation_is_explicitly_excluded_from_gate(postseal):
    metrics = postseal["memory_metrics"]
    summary = c3.build_seed_summary(
        20,
        postseal["class_metrics"],
        postseal["relation_metrics"],
        metrics,
    )
    assert all(
        "SpearmanMemoryWriteBenefit" in record
        for record in metrics["directions"]
    )
    assert (
        metrics["MemoryWriteBenefitCorrelation_used_in_gate"] is False
    )
    assert summary["memory"][
        "MemoryWriteBenefitCorrelation_used_in_gate"
    ] is False


def test_52_changing_correlation_does_not_change_memory_seed_pass():
    memory = _direction_metrics("memory")
    first = c3.build_seed_summary(
        20,
        _direction_metrics("class"),
        _direction_metrics("relation"),
        memory,
    )
    for record in memory["directions"]:
        record["SpearmanMemoryWriteBenefit"] = -0.99
    second = c3.build_seed_summary(
        20,
        _direction_metrics("class"),
        _direction_metrics("relation"),
        memory,
    )
    assert (
        first["memory"]["SeedGate_conditions"]
        == second["memory"]["SeedGate_conditions"]
    )
    assert (
        first["memory"]["MEMORY_SEED_PASS"]
        == second["memory"]["MEMORY_SEED_PASS"]
    )


def test_53_changing_correlation_does_not_change_multiseed_pass():
    summaries = OrderedDict(
        (seed, _seed_summary(seed, True, True, True))
        for seed in c3.SEEDS
    )
    first = c3.build_multiseed_decision(summaries)
    for summary in summaries.values():
        summary["memory"]["mean_SpearmanMemoryWriteBenefit"] = -0.99
    second = c3.build_multiseed_decision(summaries)
    assert (
        first["summary"]["memory"]["MEMORY_MULTI_PASS"]
        == second["summary"]["memory"]["MEMORY_MULTI_PASS"]
    )
    assert (
        first["summary"]["memory"]["MultiSeedGate_conditions"]
        == second["summary"]["memory"]["MultiSeedGate_conditions"]
    )


def test_54_branch_decision_does_not_use_correlation():
    summaries = OrderedDict(
        (seed, _seed_summary(seed, True, False, True))
        for seed in c3.SEEDS
    )
    first = c3.build_multiseed_decision(summaries)["decision"]
    for index, summary in enumerate(summaries.values()):
        summary["memory"]["mean_SpearmanMemoryWriteBenefit"] = float(
            index - 1
        )
    second = c3.build_multiseed_decision(summaries)["decision"]
    assert first == second
