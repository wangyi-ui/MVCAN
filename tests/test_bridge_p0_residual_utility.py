"""Engineering tests for the frozen read-only Bridge-P0 protocol."""

import ast
import inspect
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from experiments.cyclic_utility import (
    bridge_p0_residual_utility_protocol as bridge,
)
from experiments.cyclic_utility import (
    evaluate_bridge_p0_residual_utility as evaluate,
)
from experiments.cyclic_utility import (
    summarize_bridge_p0_residual_utility as summarize,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _ast_callable_name(node):
    """Return a dotted callable name using Python 3.8 AST APIs only."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _ast_callable_name(node.value)
        return prefix + "." + node.attr if prefix else node.attr
    if isinstance(node, ast.Call):
        return _ast_callable_name(node.func)
    return ""


def _c0_paths(seed):
    root = evaluate.default_c0_dir(seed)
    return {
        "c0_artifact_path": root / "c0_predictions_and_scores.npz",
        "c0_seal_path": root / "c0_prediction_seal.json",
        "c0_audit_path": root / "c0_audit.json",
        "c0_diagnostic_path": root / "diagnostic_results.json",
        "e1_audit_path": evaluate.default_e1_audit_path(seed),
    }


@pytest.fixture(scope="session")
def seed20_inputs():
    return evaluate.load_bridge_inputs_before_gt(seed=20, **_c0_paths(20))


def _direction_records(seed, deltas, valid_count=20):
    conditional = []
    action = []
    for direction_id in range(bridge.DIRECTION_COUNT):
        valid = direction_id < valid_count
        delta = float(deltas[direction_id])
        conditional.append({
            "direction_id": direction_id,
            "direction_valid": valid,
            "CondAUC_U": 0.5 + delta if valid else None,
            "CondAUC_conf": 0.5 if valid else None,
            "CondAUC_shuffle": 0.45 if valid else None,
            "DeltaCondAUC_cycle": delta if valid else None,
            "DeltaCondAUC_shuffle": -0.05 if valid else None,
            "CorrespondenceResidualGap": 0.05 + delta if valid else None,
        })
        action.append({
            "direction_id": direction_id,
            "direction_valid": valid,
            "ActionLift_cycle": 0.20 if valid else None,
            "ActionLift_shuffle": 0.05 if valid else None,
            "CorrespondenceLiftGap": 0.15 if valid else None,
        })
    return bridge.build_seed_summary(
        seed,
        {"directions": conditional},
        {"directions": action},
    )


def test_01_seed_set_is_exact():
    assert bridge.SEEDS == (20, 30, 50)


@pytest.mark.parametrize("seed", bridge.SEEDS)
def test_02_validate_allowed_seeds(seed):
    assert bridge.validate_seed(seed) == seed


@pytest.mark.parametrize("seed", (0, 10, 40, 60))
def test_03_reject_unknown_seed(seed):
    with pytest.raises(ValueError, match="one of"):
        bridge.validate_seed(seed)


def test_04_dataset_dimensions_are_frozen():
    assert bridge.SAMPLE_NUM == 1400
    assert bridge.VIEW_NUM == 6
    assert bridge.CLASS_NUM == 7


def test_05_direction_count_is_frozen():
    assert bridge.DIRECTION_COUNT == 20


def test_06_confidence_bin_count_is_five():
    assert bridge.CONFIDENCE_BIN_COUNT == 5


def test_07_equal_count_confidence_strata():
    confidence = np.arange(23, dtype=np.float64)
    sample_ids = np.arange(23, dtype=np.int64)
    assignments, audit = bridge.confidence_equal_count_strata(
        confidence, sample_ids
    )
    assert assignments.shape == (23,)
    assert sorted(audit["bin_sizes"]) == [4, 4, 5, 5, 5]
    assert max(audit["bin_sizes"]) - min(audit["bin_sizes"]) == 1


def test_08_confidence_then_sample_id_tie_break():
    confidence = np.ones(10, dtype=np.float64)
    sample_ids = np.arange(9, -1, -1, dtype=np.int64)
    assignments, _ = bridge.confidence_equal_count_strata(
        confidence, sample_ids
    )
    first_bin_original_rows = np.flatnonzero(assignments == 0)
    assert np.array_equal(
        np.sort(sample_ids[first_bin_original_rows]),
        np.array([0, 1]),
    )


def test_09_confidence_strata_are_deterministic():
    confidence = np.repeat(np.arange(5), 4).astype(np.float64)
    sample_ids = np.arange(20, dtype=np.int64)[::-1]
    first, first_audit = bridge.confidence_equal_count_strata(
        confidence, sample_ids
    )
    second, second_audit = bridge.confidence_equal_count_strata(
        confidence, sample_ids
    )
    assert np.array_equal(first, second)
    assert first_audit["assignment_logical_sha256"] == (
        second_audit["assignment_logical_sha256"]
    )


def test_10_GT_cannot_affect_confidence_strata():
    signature = inspect.signature(bridge.confidence_equal_count_strata)
    assert "correct" not in signature.parameters
    assert "GT" not in signature.parameters
    _, audit = bridge.confidence_equal_count_strata(
        np.arange(10), np.arange(10)
    )
    assert audit["GT_used_for_assignment"] is False
    assert audit["confidence_only_for_assignment"] is True


def test_11_single_class_bin_auc_is_invalid_none():
    assert bridge.binary_auc_or_none(
        np.zeros(8, dtype=np.int64),
        np.arange(8, dtype=np.float64),
    ) is None


def test_12_binary_auc_formula():
    correct = np.array([0, 0, 1, 1], dtype=np.int64)
    score = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    assert bridge.binary_auc_or_none(correct, score) == 1.0


@pytest.fixture
def three_valid_bin_direction():
    correct = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1]
        + [0, 1, 0, 1] * 3,
        dtype=np.int64,
    )
    confidence = np.arange(20, dtype=np.float64)
    cycle = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]
        + [0.1, 0.9, 0.2, 0.8] * 3,
        dtype=np.float64,
    )
    shuffle = np.array(
        [0.4, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2, 0.1]
        + [0.9, 0.1, 0.8, 0.2] * 3,
        dtype=np.float64,
    )
    return bridge.analyze_direction(
        correct,
        cycle,
        confidence,
        shuffle,
        np.arange(20, dtype=np.int64),
        direction_id=3,
    )


def test_13_shared_valid_mask_U_conf_shuffle(three_valid_bin_direction):
    conditional = three_valid_bin_direction["conditional_auc"]
    assert conditional["valid_bin_mask"] == [False, False, True, True, True]
    assert conditional["same_valid_bin_mask_U_conf_shuffle"] is True
    for record in conditional["bins"][:2]:
        assert record["AUC_U_cycle"] is None
        assert record["AUC_conf"] is None
        assert record["AUC_shuffle"] is None


def test_14_three_bins_make_direction_valid(three_valid_bin_direction):
    record = three_valid_bin_direction["conditional_auc"]
    assert record["valid_bin_count"] == 3
    assert record["direction_valid"] is True


def test_15_fewer_than_three_bins_make_direction_invalid():
    correct = np.array([0] * 12 + [0, 1, 0, 1] * 2, dtype=np.int64)
    score = np.arange(20, dtype=np.float64)
    result = bridge.analyze_direction(
        correct, score, score, score, np.arange(20), 0
    )
    assert result["conditional_auc"]["valid_bin_count"] == 2
    assert result["conditional_auc"]["direction_valid"] is False
    assert result["conditional_auc"]["CondAUC_U"] is None


def test_16_conditional_auc_weighted_formula(three_valid_bin_direction):
    record = three_valid_bin_direction["conditional_auc"]
    valid = [
        item for item in record["bins"]
        if item["valid_both_correctness_classes"]
    ]
    expected = np.average(
        [item["AUC_U_cycle"] for item in valid],
        weights=[item["sample_count"] for item in valid],
    )
    assert record["CondAUC_U"] == pytest.approx(expected)


def test_17_delta_conditional_auc_formula(three_valid_bin_direction):
    record = three_valid_bin_direction["conditional_auc"]
    assert record["DeltaCondAUC_cycle"] == pytest.approx(
        record["CondAUC_U"] - record["CondAUC_conf"]
    )
    assert record["DeltaCondAUC_shuffle"] == pytest.approx(
        record["CondAUC_shuffle"] - record["CondAUC_conf"]
    )


def test_18_correspondence_residual_formula(three_valid_bin_direction):
    record = three_valid_bin_direction["conditional_auc"]
    assert record["CorrespondenceResidualGap"] == pytest.approx(
        record["CondAUC_U"] - record["CondAUC_shuffle"]
    )


def test_19_action_lift_formula():
    result = bridge.matched_confidence_lift(
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.arange(4),
    )
    assert result["lift"] == 1.0
    assert result["lower_sample_count"] == result["upper_sample_count"] == 2


def test_20_action_lift_uses_sample_id_tie_break():
    result = bridge.matched_confidence_lift(
        np.array([1, 1, 0, 0], dtype=np.int64),
        np.ones(4),
        np.array([3, 2, 1, 0]),
    )
    assert result["lift"] == 1.0
    assert result["sample_id_only_tie_break"] is True
    assert result["GT_used_for_half_assignment"] is False


def test_21_action_lift_directional_formulas(three_valid_bin_direction):
    action = three_valid_bin_direction["action_lift"]
    valid = [
        item for item in action["bins"]
        if item["valid_both_correctness_classes"]
    ]
    expected_cycle = np.average(
        [item["Lift_cycle"] for item in valid],
        weights=[item["sample_count"] for item in valid],
    )
    assert action["ActionLift_cycle"] == pytest.approx(expected_cycle)
    assert action["CorrespondenceLiftGap"] == pytest.approx(
        action["ActionLift_cycle"] - action["ActionLift_shuffle"]
    )


def test_22_minimum_valid_directions_is_16():
    assert bridge.MIN_VALID_DIRECTIONS_PER_SEED == 16


def test_23_positive_direction_minimum_is_12():
    assert bridge.POSITIVE_DIRECTION_MIN_COUNT == 12


def test_24_alpha_is_point_zero_five():
    assert bridge.ALPHA == 0.05


def test_25_seed_pass_exact_all_conditions():
    summary = _direction_records(20, [0.10] * 20)
    assert summary["valid_direction_count"] == 20
    assert all(summary["seed_gate_conditions"].values())
    assert summary["RESIDUAL_INFORMATION_SEED_PASS"] is True


def test_26_seed_fails_with_only_15_valid_directions():
    summary = _direction_records(20, [0.10] * 20, valid_count=15)
    assert summary["seed_gate_conditions"]["minimum_valid_directions"] is False
    assert summary["RESIDUAL_INFORMATION_SEED_PASS"] is False


def test_27_twelve_positive_directions_satisfy_count_gate():
    summary = _direction_records(20, [0.10] * 12 + [0.0] * 8)
    assert summary["positive_DeltaCondAUC_direction_count"] == 12
    assert summary["seed_gate_conditions"]["positive_direction_count"] is True


def test_28_eleven_positive_directions_fail_count_gate():
    summary = _direction_records(20, [0.10] * 11 + [0.0] * 9)
    assert summary["positive_DeltaCondAUC_direction_count"] == 11
    assert summary["seed_gate_conditions"]["positive_direction_count"] is False
    assert summary["RESIDUAL_INFORMATION_SEED_PASS"] is False


def test_29_wilcoxon_is_one_sided_greater():
    p_value = bridge.one_sided_wilcoxon_greater(
        np.full(20, 0.1, dtype=np.float64)
    )
    assert p_value < bridge.ALPHA
    source = inspect.getsource(bridge.one_sided_wilcoxon_greater)
    assert 'alternative="greater"' in source


def test_30_zero_wilcoxon_vector_returns_one():
    assert bridge.one_sided_wilcoxon_greater(np.zeros(20)) == 1.0


def test_31_seed_summary_required_fields():
    summary = _direction_records(20, [0.10] * 20)
    required = {
        "seed",
        "valid_direction_count",
        "mean_CondAUC_U",
        "mean_CondAUC_conf",
        "mean_CondAUC_shuffle",
        "mean_DeltaCondAUC_cycle",
        "positive_DeltaCondAUC_direction_count",
        "wilcoxon_cycle_vs_conf_p",
        "mean_CorrespondenceResidualGap",
        "mean_ActionLift_cycle",
        "mean_ActionLift_shuffle",
        "mean_CorrespondenceLiftGap",
        "RESIDUAL_INFORMATION_SEED_PASS",
    }
    assert required.issubset(summary)


def test_32_multiseed_minimum_is_two():
    assert bridge.SEED_PASS_MIN_COUNT == 2


def test_33_two_of_three_seed_passes_can_pass():
    summaries = OrderedDict((
        (20, _direction_records(20, [0.10] * 20)),
        (30, _direction_records(30, [0.10] * 20)),
        (50, _direction_records(50, [-0.01] * 20)),
    ))
    result = bridge.build_multiseed_decision(summaries)
    assert result["summary"]["seed_pass_count"] == 2
    assert result["decision"]["final_decision"] == (
        "BRIDGE_P0_RESIDUAL_UTILITY_PASS"
    )


def test_34_one_of_three_seed_passes_fails():
    summaries = OrderedDict((
        (20, _direction_records(20, [0.10] * 20)),
        (30, _direction_records(30, [-0.01] * 20)),
        (50, _direction_records(50, [-0.01] * 20)),
    ))
    result = bridge.build_multiseed_decision(summaries)
    assert result["summary"]["seed_pass_count"] == 1
    assert result["decision"]["at_least_two_seed_passes"] is False
    assert result["decision"]["final_decision"] == (
        "BRIDGE_P0_RESIDUAL_UTILITY_NOT_CONFIRMED"
    )


def test_35_multiseed_seed_set_order_hard_fails():
    summaries = OrderedDict((
        (30, _direction_records(30, [0.10] * 20)),
        (20, _direction_records(20, [0.10] * 20)),
        (50, _direction_records(50, [0.10] * 20)),
    ))
    with pytest.raises(RuntimeError, match="set/order"):
        bridge.build_multiseed_decision(summaries)


def test_36_final_decisions_are_exact():
    assert bridge.FINAL_DECISIONS == (
        "BRIDGE_P0_RESIDUAL_UTILITY_PASS",
        "BRIDGE_P0_RESIDUAL_UTILITY_NOT_CONFIRMED",
    )


def test_37_evaluator_consumes_exact_C0_arrays():
    assert evaluate.REQUIRED_ARRAYS == (
        "y_gen",
        "U_cycle",
        "C_conf",
        "U_cycle_shuffle",
        "native_global_cluster",
    )


def test_38_real_seed20_pre_GT_inputs_validate(seed20_inputs):
    arrays, provenance = seed20_inputs
    assert tuple(arrays) == evaluate.REQUIRED_ARRAYS
    assert provenance["C0_AUDIT_PASS"] is True
    assert provenance["C0_final_decision"] == evaluate.C0_REQUIRED_DECISION
    assert provenance["GT_loaded"] is False


@pytest.mark.parametrize("seed", bridge.SEEDS)
def test_39_all_seed_default_inputs_exist(seed):
    paths = _c0_paths(seed)
    assert all(Path(path).is_file() for path in paths.values())


def test_40_C0_audit_and_decision_are_required():
    source = inspect.getsource(evaluate.load_bridge_inputs_before_gt)
    assert 'c0_audit.get("C0_AUDIT_PASS") is True' in source
    assert "C0_REQUIRED_DECISION" in source
    assert "diagnostic.get" in source


def test_41_C0_own_seal_is_required(seed20_inputs):
    _, provenance = seed20_inputs
    assert provenance["own_prediction_seal_validation_pass"] is True
    source = inspect.getsource(evaluate.load_bridge_inputs_before_gt)
    assert 'c0_audit.get("prediction_seal") == seal' in source


def test_42_same_seed_E1_C0_lineage(seed20_inputs):
    _, provenance = seed20_inputs
    assert provenance["E1_training_seed"] == 20
    assert provenance["C0_E1_same_seed_lineage_pass"] is True


def test_43_cross_seed_E1_mismatch_hard_fails():
    paths = _c0_paths(20)
    paths["e1_audit_path"] = evaluate.default_e1_audit_path(30)
    with pytest.raises(RuntimeError, match="same-seed|requested seed"):
        evaluate.load_bridge_inputs_before_gt(seed=20, **paths)


def test_44_fixed_weak_quality_realization(seed20_inputs):
    _, provenance = seed20_inputs
    assert provenance["fixed_weak_quality_condition"] == (
        bridge.WEAK_QUALITY_CONDITION
    )
    assert provenance["fixed_feature_path"] == (
        "outputs/e0_glgc_adapter/caltech6v_snr2p5_k3_seed20.npz"
    )


def _synthetic_mapping_case():
    native = np.arange(bridge.SAMPLE_NUM, dtype=np.int64) % bridge.CLASS_NUM
    mapping = np.array([4, 2, 3, 1, 5, 6, 0], dtype=np.int64)
    labels = mapping[native]
    y_gen = np.tile(
        np.arange(bridge.DIRECTION_COUNT, dtype=np.int64)
        % bridge.CLASS_NUM,
        (bridge.SAMPLE_NUM, 1),
    )
    contingency = np.zeros((bridge.CLASS_NUM, bridge.CLASS_NUM), dtype=np.int64)
    np.add.at(contingency, (native, labels), 1)
    audit = {
        "global_mapping": {
            "mapping": mapping.tolist(),
            "contingency": contingency.tolist(),
            "matched_count": bridge.SAMPLE_NUM,
            "mapping_fit_count": 1,
            "mapping_fit_after_score_seal": True,
            "single_global_mapping_reused_all_directions": True,
            "direction_specific_mapping_used": False,
        }
    }
    arrays = {
        "native_global_cluster": native,
        "y_gen": y_gen,
    }
    return arrays, audit, labels, mapping


def test_45_frozen_C0_global_mapping_reused():
    arrays, audit, labels, mapping = _synthetic_mapping_case()
    correct, mapping_audit = (
        evaluate.build_correctness_from_frozen_c0_mapping(
            arrays, audit, labels
        )
    )
    expected = mapping[arrays["y_gen"]] == labels[:, None]
    assert np.array_equal(correct, expected)
    assert mapping_audit["C0_apply_global_mapping_reused"] is True
    assert mapping_audit["Bridge_mapping_fit_count"] == 0


def test_46_no_Hungarian_fit_in_Bridge():
    source = inspect.getsource(evaluate)
    assert "linear_sum_assignment" not in source
    assert "fit_global_cluster_mapping_once(" not in source
    assert '"Hungarian_fit_in_Bridge": False' in source


def test_47_no_per_direction_or_bin_mapping():
    audit = bridge.leakage_audit()
    assert audit["no_per_direction_GT_mapping"] is True
    assert audit["no_per_bin_GT_mapping"] is True
    source = inspect.getsource(evaluate.build_correctness_from_frozen_c0_mapping)
    assert "for direction" not in source
    assert "for bin" not in source


def test_48_GT_load_occurs_after_durable_input_seal():
    source = inspect.getsource(evaluate.run_evaluation)
    assert source.index("save_bridge_input_seal") < source.index(
        "load_bridge_full_ground_truth_after_seal"
    )
    assert source.index("load_bridge_full_ground_truth_after_seal") < source.index(
        "build_correctness_from_frozen_c0_mapping"
    )


def test_49_same_C0_GT_loader_is_reused():
    source = inspect.getsource(evaluate.run_evaluation)
    assert "load_bridge_full_ground_truth_after_seal" in source


def test_49a_real_C0_GT_loader_contract_is_structured_tuple():
    source = inspect.getsource(evaluate.c0_evaluator.run_evaluation)
    assert "full_GT, full_GT_audit = load_full_ground_truth_after_seal" in source


def test_49b_bridge_unpacks_labels_and_preserves_GT_audit(monkeypatch):
    arrays, c0_audit, labels, _ = _synthetic_mapping_case()
    gt_audit = {
        "path": "data/Caltech.mat",
        "file_sha256": "frozen-gt-hash",
        "shape": [bridge.SAMPLE_NUM],
        "loaded_after_prediction_seal": True,
    }

    def fake_loader(path):
        return labels, gt_audit

    monkeypatch.setattr(
        evaluate.c0_evaluator,
        "load_full_ground_truth_after_seal",
        fake_loader,
    )
    extracted, preserved_audit = (
        evaluate.load_bridge_full_ground_truth_after_seal("unused.mat")
    )
    correct, _ = evaluate.build_correctness_from_frozen_c0_mapping(
        arrays,
        c0_audit,
        extracted,
    )
    assert extracted.shape == (bridge.SAMPLE_NUM,)
    assert extracted.dtype == np.int64
    assert preserved_audit == gt_audit
    assert correct.shape == (
        bridge.SAMPLE_NUM,
        bridge.DIRECTION_COUNT,
    )


@pytest.mark.parametrize(
    "malformed",
    (
        np.zeros(bridge.SAMPLE_NUM, dtype=np.int64),
        (np.zeros(bridge.SAMPLE_NUM - 1, dtype=np.int64), {}),
        (np.zeros(bridge.SAMPLE_NUM, dtype=np.float64), {}),
        (np.zeros(bridge.SAMPLE_NUM, dtype=np.int64), []),
    ),
)
def test_49c_malformed_C0_GT_loader_output_hard_fails(
    monkeypatch,
    malformed,
):
    monkeypatch.setattr(
        evaluate.c0_evaluator,
        "load_full_ground_truth_after_seal",
        lambda path: malformed,
    )
    with pytest.raises(RuntimeError, match="C0 GT loader|C0-normalized"):
        evaluate.load_bridge_full_ground_truth_after_seal("unused.mat")



def test_50_per_seed_output_schema_is_exact():
    assert evaluate.PER_SEED_OUTPUT_FILES == (
        "bridge_input_seal.json",
        "bridge_audit.json",
        "confidence_strata_audit.json",
        "conditional_auc_by_direction.json",
        "action_lift_by_direction.json",
        "bridge_seed_summary.json",
    )


@pytest.mark.parametrize("seed", bridge.SEEDS)
def test_51_evaluator_CLI_seed_defaults(seed):
    args = evaluate.parse_args(["--seed", str(seed)])
    root = "c0_complementary_semantic_verification_seed" + str(seed)
    assert root in args.c0_artifact_path
    assert "lwc_100ep_seed" + str(seed) in args.e1_audit_path
    assert args.output_dir.endswith(
        "bridge_p0_residual_utility_seed" + str(seed)
    )


def test_52_evaluator_CLI_preserves_explicit_paths(tmp_path):
    values = [str(tmp_path / str(name)) for name in range(8)]
    args = evaluate.parse_args([
        "--seed", "20",
        "--c0-artifact-path", values[0],
        "--c0-seal-path", values[1],
        "--c0-audit-path", values[2],
        "--c0-diagnostic-path", values[3],
        "--e1-audit-path", values[4],
        "--full-gt-path", values[5],
        "--output-dir", values[6],
    ])
    assert args.c0_artifact_path == values[0]
    assert args.full_gt_path == values[5]
    assert args.output_dir == values[6]


def test_53_summarizer_seed_files_match_evaluator():
    assert summarize.REQUIRED_SEED_FILES == evaluate.PER_SEED_OUTPUT_FILES


def test_54_summarizer_output_schema_is_exact():
    assert summarize.MULTISEED_OUTPUT_FILES == (
        "bridge_multiseed_summary.json",
        "bridge_multiseed_decision.json",
        "bridge_multiseed_audit.json",
    )


def test_55_summarizer_CLI_is_explicit():
    args = summarize.parse_args([
        "--seed20-dir", "a",
        "--seed30-dir", "b",
        "--seed50-dir", "c",
        "--output-dir", "d",
    ])
    assert vars(args) == {
        "seed20_dir": "a",
        "seed30_dir": "b",
        "seed50_dir": "c",
        "output_dir": "d",
    }


def test_56_summarizer_uses_only_frozen_aggregate_helper():
    source = inspect.getsource(summarize.summarize)
    assert "bridge.build_multiseed_decision" in source
    assert "SEED_PASS_MIN_COUNT" not in source
    assert "ALPHA" not in source


def test_57_summarizer_requires_exact_seed_order():
    source = inspect.getsource(summarize.summarize)
    assert "(20, seed20_dir)" in source
    assert "(30, seed30_dir)" in source
    assert "(50, seed50_dir)" in source
    assert "bridge.SEEDS" in source


def test_58_leakage_audit_required_false_fields():
    audit = bridge.leakage_audit()
    fields = (
        "training_entered",
        "optimizer_created",
        "backward_called",
        "model_loaded_for_training",
        "R_loaded",
        "corruption_mask_loaded",
        "sparse_labels_loaded",
        "sparse_label_ids_loaded",
        "sparse_label_targets_loaded",
        "B7_artifact_loaded",
        "GT_loaded_before_bridge_input_seal",
        "new_utility_constructed",
        "VSA_modified",
        "C0_modified",
    )
    assert all(audit[field] is False for field in fields)
    assert audit["GT_loaded_after_bridge_input_seal"] is True


def _called_names(module):
    tree = ast.parse(inspect.getsource(module))
    return [
        _ast_callable_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]


@pytest.mark.parametrize(
    "forbidden",
    (
        "load_frozen_reliability",
        "load_sparse_label_protocol",
        "load_corruption_mask",
        "load_B7",
        "torch.optim",
        "backward",
        "optimizer.step",
        "train_model",
    ),
)
def test_59_forbidden_calls_absent(forbidden):
    called = _called_names(evaluate) + _called_names(summarize)
    assert all(forbidden not in name for name in called)


def test_60_no_R_or_sparse_or_B7_imports():
    source = inspect.getsource(evaluate) + inspect.getsource(summarize)
    tree = ast.parse(source)
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    assert all("b7" not in name.lower() for name in imports)
    assert all("reliability" not in name.lower() for name in imports)
    assert all("sparse" not in name.lower() for name in imports)


def test_61_no_new_utility_formula():
    protocol_source = inspect.getsource(bridge)
    evaluator_source = inspect.getsource(evaluate)
    assert "new_utility_constructed" in protocol_source
    assert "new_utility_constructed" in evaluator_source
    assert "P_corr" not in evaluator_source
    assert "P_util" not in evaluator_source


def test_62_no_training_or_model_loading():
    evaluator_source = inspect.getsource(evaluate)
    assert "torch" not in evaluator_source
    assert "load_trainable" not in evaluator_source
    assert "optimizer_created" in evaluator_source
    assert '"optimizer_created": False' not in evaluator_source
    assert "bridge.leakage_audit()" in evaluator_source


def test_63_frozen_existing_source_integrity():
    audit = evaluate.verify_frozen_source_integrity()
    assert audit["all_existing_scientific_sources_unchanged_pass"] is True
    assert audit["actual_sha256"] == bridge.FROZEN_SOURCE_SHA256


@pytest.mark.parametrize(
    "expression,expected",
    (
        ("foo()", "foo"),
        ("module.foo()", "module.foo"),
        ("package.module.foo()", "package.module.foo"),
        ("self.foo()", "self.foo"),
    ),
)
def test_64_python38_callable_name_helper(expression, expected):
    tree = ast.parse(expression)
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    assert _ast_callable_name(call.func) == expected


def test_65_python38_nested_forbidden_detection():
    tree = ast.parse("some_module.load_sparse_label_protocol()")
    names = [
        _ast_callable_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    assert any("load_sparse_label_protocol" in name for name in names)


def test_66_no_ast_unparse_dependency():
    source = Path(__file__).read_text(encoding="utf-8")
    source += inspect.getsource(bridge)
    source += inspect.getsource(evaluate)
    source += inspect.getsource(summarize)
    forbidden_api = "ast." + "unparse"
    assert forbidden_api not in source


def test_67_protocol_source_is_not_an_I_O_runner():
    source = inspect.getsource(bridge)
    assert "np.load(" not in source
    assert "write_json" not in source
    assert "argparse" not in source


def test_68_protocol_gate_constants_unchanged():
    assert bridge.MIN_VALID_BINS_PER_DIRECTION == 3
    assert bridge.MIN_VALID_DIRECTIONS_PER_SEED == 16
    assert bridge.POSITIVE_DIRECTION_MIN_COUNT == 12
    assert bridge.ALPHA == 0.05
    assert bridge.SEED_PASS_MIN_COUNT == 2


def test_69_no_formal_bridge_outputs_exist_before_scientific_run():
    for seed in bridge.SEEDS:
        assert not evaluate.default_output_dir(seed).exists()


def test_70_no_multiseed_scientific_output_exists():
    candidates = list(
        (REPOSITORY_ROOT / "outputs/cyclic_utility").glob(
            "bridge_p0_residual_utility_multiseed*"
        )
    )
    assert candidates == []
