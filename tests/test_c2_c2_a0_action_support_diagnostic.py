import ast
import inspect
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from experiments.cyclic_utility import c2_c0_directional_consensus_utility_protocol as c0
from experiments.cyclic_utility import c2_c2_a0_action_support_diagnostic_protocol as c2
from experiments.cyclic_utility import evaluate_c2_c2_a0_action_support_diagnostic as evaluate
from experiments.cyclic_utility import summarize_c2_c2_a0_action_support_diagnostic as summarize
from weak_quality import ndarray_sha256


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def formal_arrays():
    rows = np.arange(c2.SAMPLE_NUM, dtype=np.int64)[:, None]
    directions = np.arange(c2.DIRECTION_COUNT, dtype=np.int64)[None, :]
    y_gen = (rows + 2 * directions) % c2.CLASS_NUM
    U_cycle = (0.05 + ((3 * rows + 7 * directions) % 90) / 100.0).astype(
        np.float64
    )
    U_cycle[0, 0] = 0.0
    C_conf = (0.01 + ((11 * rows + directions) % 98) / 100.0).astype(
        np.float64
    )
    return {
        "y_gen": y_gen,
        "U_cycle": U_cycle,
        "C_conf": C_conf,
        "sample_ids": c2.canonical_sample_ids(),
    }


@pytest.fixture(scope="module")
def split():
    labeled = c2.fixed_labeled_ids()
    return {
        "labeled_sample_ids": labeled,
        "labeled_targets": c2.fixed_labeled_targets(),
        "shuffled_labeled_targets": c2.fixed_shuffled_targets(),
        "unlabeled_ids": np.setdiff1d(c2.canonical_sample_ids(), labeled),
        "label_split_sha256": "unit-test-split",
    }


@pytest.fixture(scope="module")
def frozen_c0_outputs(formal_arrays, split):
    return c0.build_directional_consensus_outputs(
        formal_arrays["y_gen"],
        formal_arrays["U_cycle"],
        split["labeled_sample_ids"],
        split["labeled_targets"],
        split["shuffled_labeled_targets"],
    )


@pytest.fixture(scope="module")
def outputs(formal_arrays, split):
    return c2.build_action_support_outputs(
        formal_arrays["y_gen"],
        formal_arrays["U_cycle"],
        split["labeled_sample_ids"],
        split["labeled_targets"],
        split["shuffled_labeled_targets"],
    )


@pytest.fixture(scope="module")
def support_record(split):
    count = c2.UNLABELED_EVAL_COUNT
    support = np.linspace(1.0, 10.0, count)
    support_shuffle = np.linspace(2.0, 9.0, count)[::-1]
    correct = (np.arange(count) % 3 != 0).astype(np.int8)
    correct_shuffle = (np.arange(count) % 2).astype(np.int8)
    valid = np.ones(count, dtype=np.bool_)
    benefit = np.linspace(-0.2, 0.3, count)
    return c2.analyze_support_direction(
        support,
        support_shuffle,
        correct,
        valid,
        correct_shuffle,
        valid,
        benefit,
        split["unlabeled_ids"],
        0,
    )


def _bundle(formal_arrays, split):
    provenance = {
        "GT_loaded": False,
        "c0_audit_loaded": False,
        "Bridge_correctness_loaded": False,
        "corruption_mask_loaded": False,
        "C0_artifact_file_sha256": "a" * 64,
        "C0_prediction_seal_file_sha256": "b" * 64,
        "C0_prediction_seal": {},
    }
    frozen = {
        "manifests": {},
        "C2_B0_frozen_hashes_pass": True,
        "C2_C0_frozen_hashes_pass": True,
        "C2_C1_frozen_hashes_pass": True,
        "all_frozen_source_hashes_pass": True,
    }
    return evaluate.build_pre_gt_bundle(
        20, formal_arrays, provenance, split, frozen
    )


@pytest.fixture()
def persisted(tmp_path, formal_arrays, split):
    return evaluate.persist_support_bundle(
        _bundle(formal_arrays, split), tmp_path / "sealed"
    )


def _weights():
    return np.zeros((1400, 14, 20), dtype=np.float64)


def _direction_records(
    auc=0.7, specificity=0.1, high_low=0.2, valid_count=20
):
    directions = []
    for direction_id in range(c2.DIRECTION_COUNT):
        valid = direction_id < valid_count
        directions.append({
            "direction_id": direction_id,
            "joint_valid": valid,
            "AUC_support_true": auc if valid else None,
            "SupportSpecificityGap": specificity if valid else None,
            "HighLowAgreementGap": high_low if valid else None,
            "SpearmanSupportCorrectness": 0.3 if valid else None,
            "SpearmanSupportActionBenefit": 0.2 if valid else None,
        })
    return {"directions": directions}


def _called_names(function):
    tree = ast.parse(inspect.getsource(function))
    return {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }


def _all_called_names(*paths):
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


def _all_parameter_names():
    functions = (
        value for value in vars(c2).values()
        if inspect.isfunction(value) and value.__module__ == c2.__name__
    )
    return {
        name.lower()
        for function in functions
        for name in inspect.signature(function).parameters
    }


def test_01_ess_exact_formula():
    weights = _weights()
    weights[0, :3, 0] = (1.0, 2.0, 3.0)
    result = c2.compute_effective_support(weights)
    assert result["ESS"][0, 0] == pytest.approx(6.0 ** 2 / 14.0)


def test_02_one_active_anchor_has_ess_one():
    weights = _weights()
    weights[0, 3, 0] = 7.5
    assert c2.compute_effective_support(weights)["ESS"][0, 0] == 1.0


def test_03_three_equal_active_anchors_have_ess_three():
    weights = _weights()
    weights[0, :3, 0] = 4.0
    assert c2.compute_effective_support(weights)["ESS"][0, 0] == 3.0


def test_04_ess_is_scale_invariant():
    weights = _weights()
    weights[0, :4, 0] = (1.0, 2.0, 3.0, 4.0)
    first = c2.compute_effective_support(weights)["ESS"][0, 0]
    second = c2.compute_effective_support(weights * 10.0)["ESS"][0, 0]
    assert first == pytest.approx(second)


def test_05_zero_support_has_exact_zero_ess():
    result = c2.compute_effective_support(_weights())
    assert np.all(result["ESS"] == 0.0)


def test_06_ess_is_finite(outputs):
    assert np.isfinite(outputs["ESS"]).all()
    assert np.isfinite(outputs["ESS_shuffle"]).all()


def test_07_ess_is_bounded_by_label_count(outputs):
    for name in ("ESS", "ESS_shuffle"):
        assert np.all((0.0 <= outputs[name]) & (outputs[name] <= 14.0))


def test_08_support_mass_is_exact():
    weights = _weights()
    weights[0, :3, 0] = (1.0, 2.0, 3.0)
    assert c2.compute_effective_support(weights)["support_mass"][0, 0] == 6.0


def test_09_squared_support_mass_is_exact():
    weights = _weights()
    weights[0, :3, 0] = (1.0, 2.0, 3.0)
    result = c2.compute_effective_support(weights)
    assert result["support_square_mass"][0, 0] == 14.0


def test_10_active_anchor_count_is_exact():
    weights = _weights()
    weights[0, (0, 4, 13), 0] = (1.0, 2.0, 3.0)
    result = c2.compute_effective_support(weights)
    assert result["active_anchor_count"][0, 0] == 3


def test_11_no_handcrafted_combined_support_score():
    names = set(vars(c2))
    source = inspect.getsource(c2.compute_effective_support)
    assert "support_score" not in names
    assert "alpha *" not in source
    assert "beta *" not in source


def test_12_true_c2_c0_weights_are_reproduced_exactly(outputs, frozen_c0_outputs):
    assert np.array_equal(outputs["weights"], frozen_c0_outputs["weights"])


def test_13_shuffle_weights_are_rebuilt_from_shuffled_labels(
    outputs, frozen_c0_outputs
):
    assert np.array_equal(
        outputs["weights_shuffle"], frozen_c0_outputs["shuffle_weights"]
    )
    assert outputs["shuffle_recomputed_from_targets"] is True


def test_14_true_weights_are_not_reused_for_shuffle(outputs):
    assert outputs["true_weights_reused_for_shuffle"] is False
    assert outputs["weights_shuffle_independently_rebuilt"] is True
    assert outputs["weights"] is not outputs["weights_shuffle"]


def test_15_ess_shuffle_is_independently_computed(outputs):
    rebuilt = c2.compute_effective_support(outputs["weights_shuffle"])
    assert np.array_equal(outputs["ESS_shuffle"], rebuilt["ESS"])
    assert outputs["ESS_shuffle_independently_computed"] is True


def test_16_support_diagnostic_does_not_modify_u_cycle(formal_arrays, split):
    before = formal_arrays["U_cycle"].copy()
    c2.build_action_support_outputs(
        formal_arrays["y_gen"],
        formal_arrays["U_cycle"],
        split["labeled_sample_ids"],
        split["labeled_targets"],
        split["shuffled_labeled_targets"],
    )
    assert np.array_equal(formal_arrays["U_cycle"], before)


def test_17_no_new_u_tilde_method_exists():
    local_functions = {
        name for name, value in vars(c2).items()
        if inspect.isfunction(value) and value.__module__ == c2.__name__
    }
    assert all("tilde" not in name.lower() for name in local_functions)


def test_18_pre_gt_array_whitelist_is_exact(formal_arrays, split):
    bundle = _bundle(formal_arrays, split)
    assert tuple(bundle["arrays"]) == evaluate.PRE_GT_ARRAY_NAMES


def test_19_gt_is_inaccessible_before_support_seal():
    forbidden = {
        "load_full_ground_truth_after_support_seal",
        "load_c0_audit_after_support_seal",
        "load_native_global_cluster_after_support_seal",
        "build_bridge_correctness_after_support_seal",
    }
    functions = (
        evaluate.load_c0_inputs_before_gt,
        evaluate.load_fixed_sparse_labels,
        evaluate.build_pre_gt_bundle,
        evaluate.persist_support_bundle,
    )
    assert all(_called_names(function).isdisjoint(forbidden) for function in functions)


def test_20_c0_audit_is_inaccessible_before_support_seal():
    assert "load_c0_audit_after_support_seal" not in _called_names(
        evaluate.persist_support_bundle
    )


def test_21_bridge_correctness_is_inaccessible_before_support_seal():
    assert "build_bridge_correctness_after_support_seal" not in _called_names(
        evaluate.build_pre_gt_bundle
    )


def test_22_corruption_mask_loader_is_absent():
    assert "load_corruption_mask" not in (set(vars(c2)) | set(vars(evaluate)))


def test_23_no_model_forward():
    assert "forward" not in _all_called_names(c2.__file__, evaluate.__file__)


def test_24_no_optimizer():
    called = _all_called_names(c2.__file__, evaluate.__file__)
    assert {"optimizer", "step"}.isdisjoint(called)


def test_25_no_backward():
    assert "backward" not in _all_called_names(c2.__file__, evaluate.__file__)


def test_26_no_training():
    assert "train" not in _all_called_names(c2.__file__, evaluate.__file__)


def test_27_durable_pre_gt_save_reload_and_hash(persisted):
    seal = persisted["seal"]
    assert seal["NPZ_saved_fsynced_reloaded_hash_verified"] is True
    with np.load(persisted["npz_path"], allow_pickle=False) as archive:
        assert tuple(archive.files) == evaluate.PRE_GT_ARRAY_NAMES
        for name in evaluate.PRE_GT_ARRAY_NAMES:
            assert ndarray_sha256(archive[name]) == seal["arrays"][name]["logical_sha256"]


def test_28_evaluation_requires_exactly_1386_unlabeled_samples(split):
    zeros = np.zeros((1386, 20), dtype=np.float64)
    result = c2.build_directional_correctness(
        zeros, zeros, zeros, split["unlabeled_ids"]
    )
    assert result["DirCorrect"].shape == (1386, 20)
    with pytest.raises(RuntimeError, match="1386 unlabeled"):
        c2.build_directional_correctness(
            np.zeros((1400, 20)), np.zeros((1400, 20)),
            np.zeros((1400, 20)), np.arange(1400),
        )


def test_29_directional_correctness_formula_is_exact(split):
    d = np.zeros((1386, 20), dtype=np.float64)
    d_shuffle = np.zeros_like(d)
    truth = np.zeros_like(d)
    d[:3, 0] = (1.0, -1.0, 1.0)
    d_shuffle[:3, 0] = (-1.0, -1.0, 1.0)
    truth[:3, 0] = (1.0, -1.0, -1.0)
    result = c2.build_directional_correctness(
        d, d_shuffle, truth, split["unlabeled_ids"]
    )
    assert np.array_equal(result["DirCorrect"][:3, 0], (1, 1, 0))
    assert np.array_equal(result["DirCorrect_shuffle"][:3, 0], (0, 1, 0))


def test_30_zero_d_is_excluded_from_directional_correctness(split):
    d = np.ones((1386, 20), dtype=np.float64)
    truth = np.ones_like(d)
    d[0, 0] = 0.0
    result = c2.build_directional_correctness(
        d, d, truth, split["unlabeled_ids"]
    )
    assert result["valid_directional_action_mask"][0, 0] == 0


def test_31_zero_e_gt_is_excluded_from_directional_correctness(split):
    d = np.ones((1386, 20), dtype=np.float64)
    truth = np.ones_like(d)
    truth[0, 0] = 0.0
    result = c2.build_directional_correctness(
        d, d, truth, split["unlabeled_ids"]
    )
    assert result["valid_directional_action_mask"][0, 0] == 0


def test_32_gt_cannot_influence_ess():
    assert "e_GT" not in inspect.signature(c2.compute_effective_support).parameters
    assert "z_GT" not in inspect.signature(c2.compute_effective_support).parameters


def test_33_roc_auc_calculation_is_exact(support_record):
    count = c2.UNLABELED_EVAL_COUNT
    support = np.linspace(1.0, 10.0, count)
    correct = (np.arange(count) % 3 != 0).astype(np.int8)
    expected = c2.binary_auc_or_none(correct, support)
    assert support_record["AUC_support_true"] == expected


def test_34_single_class_target_direction_is_invalid(split):
    count = c2.UNLABELED_EVAL_COUNT
    support = np.linspace(1.0, 2.0, count)
    correct = np.ones(count, dtype=np.int8)
    valid = np.ones(count, dtype=np.bool_)
    record = c2.analyze_support_direction(
        support, support, correct, valid, correct, valid,
        support, split["unlabeled_ids"], 0,
    )
    assert record["joint_valid"] is False
    assert record["AUC_support_true"] is None


def test_35_constant_ess_direction_is_invalid(split):
    count = c2.UNLABELED_EVAL_COUNT
    support = np.ones(count, dtype=np.float64)
    correct = (np.arange(count) % 2).astype(np.int8)
    valid = np.ones(count, dtype=np.bool_)
    record = c2.analyze_support_direction(
        support, support, correct, valid, correct, valid,
        support, split["unlabeled_ids"], 0,
    )
    assert record["joint_valid"] is False
    assert record["AUC_support_true"] is None


def test_36_support_specificity_gap_formula_is_exact(support_record):
    assert support_record["SupportSpecificityGap"] == (
        support_record["AUC_support_true"]
        - support_record["AUC_support_shuffle"]
    )


def test_37_five_ess_quantile_bins_are_exact(split):
    support = np.linspace(0.0, 1.0, 1386)
    assignments, audit = c2.support_equal_count_quantiles(
        support, split["unlabeled_ids"]
    )
    assert audit["bin_count"] == 5
    assert len(audit["bins"]) == 5
    assert max(np.bincount(assignments)) - min(np.bincount(assignments)) <= 1


def test_38_quantile_assignment_uses_ess_only():
    signature = inspect.signature(c2.support_equal_count_quantiles)
    assert tuple(signature.parameters) == ("ESS", "sample_ids")


def test_39_gt_cannot_change_quantile_assignment(split):
    support = np.linspace(0.0, 1.0, 1386)
    first, audit = c2.support_equal_count_quantiles(
        support, split["unlabeled_ids"]
    )
    second, _ = c2.support_equal_count_quantiles(
        support, split["unlabeled_ids"]
    )
    assert np.array_equal(first, second)
    assert audit["GT_used_for_assignment"] is False


def test_40_high_low_agreement_gap_is_exact(support_record):
    assert support_record["HighLowAgreementGap"] == (
        support_record["Agree_Q5"] - support_record["Agree_Q1"]
    )


def test_41_spearman_support_correctness_is_exact(support_record):
    count = c2.UNLABELED_EVAL_COUNT
    support = np.linspace(1.0, 10.0, count)
    correct = (np.arange(count) % 3 != 0).astype(np.int8)
    expected = float(spearmanr(support, correct).statistic)
    assert support_record["SpearmanSupportCorrectness"] == expected


def test_42_frozen_c2_c0_action_is_exact():
    target = np.zeros((1386, 20), dtype=np.float64)
    cycle = np.full((1386, 20), 0.4, dtype=np.float64)
    consensus = np.full((1386, 20), 0.25, dtype=np.float64)
    result = c2.frozen_c0_action_and_benefit(target, cycle, consensus)
    expected = cycle + cycle * (1.0 - cycle) * consensus
    assert np.array_equal(result["U_C0"], expected)


def test_43_action_benefit_formula_is_exact():
    target = np.zeros((1386, 20), dtype=np.float64)
    target[::2] = 1.0
    cycle = np.full((1386, 20), 0.4, dtype=np.float64)
    consensus = np.full((1386, 20), 0.25, dtype=np.float64)
    result = c2.frozen_c0_action_and_benefit(target, cycle, consensus)
    expected = np.abs(target - cycle) - np.abs(target - result["U_C0"])
    assert np.array_equal(result["ActionBenefit"], expected)


def test_44_action_benefit_does_not_influence_formal_gate():
    metrics = _direction_records()
    first = c2.build_seed_summary(20, metrics)
    for record in metrics["directions"]:
        record["SpearmanSupportActionBenefit"] = -0.9
    second = c2.build_seed_summary(20, metrics)
    assert first["SeedGate_conditions"] == second["SeedGate_conditions"]
    assert second["ActionBenefit_used_in_gate"] is False


def test_45_per_seed_gate_is_exact():
    summary = c2.build_seed_summary(20, _direction_records())
    assert summary["valid_support_AUC_direction_count"] == 20
    assert summary["AUC_support_true_above_half_direction_count"] == 20
    assert summary["AUC_minus_half_wilcoxon_greater_p"] < 0.05
    assert summary["positive_HighLowAgreementGap_direction_count"] == 20
    assert all(summary["SeedGate_conditions"].values())
    assert summary["C2_C2_A0_SEED_PASS"] is True
    failed = c2.build_seed_summary(20, _direction_records(valid_count=15))
    assert failed["C2_C2_A0_SEED_PASS"] is False


def test_46_multi_seed_gate_and_branch_decision_are_exact():
    summaries = OrderedDict()
    for seed, passed in zip(c2.SEEDS, (True, True, False)):
        summaries[seed] = {
            "seed": seed,
            "C2_C2_A0_SEED_PASS": passed,
            "mean_AUC_support_true": 0.7,
            "mean_SupportSpecificityGap": 0.1,
            "mean_HighLowAgreementGap": 0.2,
        }
    result = c2.build_multiseed_decision(summaries)
    assert all(result["decision"]["decision_conditions"].values())
    assert result["decision"][
        "C2_C2_A0_ACTION_SUPPORT_SUFFICIENCY_DIAGNOSTIC_PASS"
    ] is True
    assert result["decision"]["precommitted_branch_decision"] == c2.PASS_BRANCH_DECISION
    summaries[30]["mean_HighLowAgreementGap"] = -1.0
    failed = c2.build_multiseed_decision(summaries)
    assert failed["decision"]["precommitted_branch_decision"] == c2.FAIL_BRANCH_DECISION


def test_47_no_threshold_parameter():
    assert "threshold" not in _all_parameter_names()
    assert "THRESHOLD" not in vars(c2)


def test_48_no_top_k_parameter():
    assert "top_k" not in _all_parameter_names()
    assert "TOP_K" not in vars(c2)


def test_49_no_temperature_parameter():
    assert "temperature" not in _all_parameter_names()
    assert "TEMPERATURE" not in vars(c2)


def test_50_no_alpha_beta_or_lambda_parameter():
    forbidden = {"alpha", "beta", "lambda"}
    assert forbidden.isdisjoint(_all_parameter_names())
    assert {"ALPHA", "BETA", "LAMBDA"}.isdisjoint(vars(c2))


def test_51_no_hyperparameter_sweep():
    names = set(vars(c2)) | set(vars(evaluate))
    assert {"grid_search", "parameter_sweep", "hyperparameter_sweep"}.isdisjoint(names)


def test_52_no_mlp_or_learned_selector():
    names = set(vars(c2)) | set(vars(evaluate))
    assert {"MLP", "mlp", "learned_selector"}.isdisjoint(names)


def test_53_no_memory_or_caip():
    names = set(vars(c2)) | set(vars(evaluate))
    assert {"memory_bank", "CAIP", "caip"}.isdisjoint(names)


def test_54_no_pseudo_label_or_training_path():
    names = set(vars(c2)) | set(vars(evaluate))
    called = _all_called_names(c2.__file__, evaluate.__file__)
    assert {"pseudo_label", "train", "backward", "optimizer"}.isdisjoint(
        names | called
    )


def test_55_all_required_preseal_false_boundaries_are_recorded(persisted):
    keys = (
        "GT_loaded_before_support_seal",
        "c0_audit_loaded_before_support_seal",
        "Bridge_correctness_loaded_before_support_seal",
        "corruption_mask_loaded",
        "training_performed",
        "model_forward_called",
        "optimizer_created",
        "optimizer_step_called",
        "backward_called",
    )
    assert all(persisted["seal"][key] is False for key in keys)


def test_56_frozen_parent_manifests_all_pass():
    audit = evaluate.verify_frozen_sources()
    assert audit["C2_B0_frozen_hashes_pass"] is True
    assert audit["C2_C0_frozen_hashes_pass"] is True
    assert audit["C2_C1_frozen_hashes_pass"] is True
    assert tuple(audit["manifests"]) == evaluate.FROZEN_SOURCE_MANIFESTS


def test_57_only_fixed_seeds_and_default_paths_are_supported():
    assert tuple(c2.SEEDS) == (20, 30, 50)
    for seed in c2.SEEDS:
        assert "seed" + str(seed) in str(summarize.default_seed_dir(seed))
    assert str(summarize.default_multiseed_dir()).endswith(
        "c2_c2_a0_action_support_diagnostic_multiseed"
    )
    with pytest.raises(ValueError):
        summarize.default_seed_dir(21)


def test_58_exactly_four_c2_c2_a0_python_files_exist():
    expected = {
        ROOT / "experiments/cyclic_utility/c2_c2_a0_action_support_diagnostic_protocol.py",
        ROOT / "experiments/cyclic_utility/evaluate_c2_c2_a0_action_support_diagnostic.py",
        ROOT / "experiments/cyclic_utility/summarize_c2_c2_a0_action_support_diagnostic.py",
        ROOT / "tests/test_c2_c2_a0_action_support_diagnostic.py",
    }
    discovered = set(ROOT.glob("experiments/cyclic_utility/*c2_c2_a0*.py"))
    discovered.update(ROOT.glob("tests/*c2_c2_a0*.py"))
    assert discovered == expected


def test_59_numpy_int8_ndarray_serializes_successfully(tmp_path):
    original = np.asarray([[0, 1], [1, 0]], dtype=np.int8)
    path = tmp_path / "int8.json"
    evaluate.write_json_durable(path, {"value": original})
    assert evaluate.read_json(path)["value"] == [[0, 1], [1, 0]]


def test_60_numpy_bool_ndarray_serializes_successfully(tmp_path):
    original = np.asarray([[True, False], [False, True]], dtype=np.bool_)
    path = tmp_path / "bool.json"
    evaluate.write_json_durable(path, {"value": original})
    assert evaluate.read_json(path)["value"] == [
        [True, False], [False, True]
    ]


def test_61_nested_ndarray_serializes_successfully():
    original = {"outer": {"inner": np.asarray([[1]], dtype=np.int8)}}
    encoded = json.dumps(evaluate._json_native(original), allow_nan=False)
    assert json.loads(encoded) == {"outer": {"inner": [[1]]}}


def test_62_json_round_trip_preserves_dircorrect_exactly():
    original = np.asarray([[0, 1, 1], [1, 0, 1]], dtype=np.int8)
    encoded = json.dumps(evaluate._json_native(original), allow_nan=False)
    restored = np.asarray(json.loads(encoded), dtype=np.int8)
    assert np.array_equal(restored, original)


def test_63_json_round_trip_preserves_dircorrect_shuffle_exactly():
    original = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.int8)
    payload = {"DirCorrect_shuffle": original}
    encoded = json.dumps(evaluate._json_native(payload), allow_nan=False)
    restored = np.asarray(
        json.loads(encoded)["DirCorrect_shuffle"], dtype=np.int8
    )
    assert np.array_equal(restored, original)


def test_64_json_round_trip_preserves_valid_masks_exactly():
    true_mask = np.asarray([[True, False], [True, True]], dtype=np.bool_)
    shuffle_mask = np.asarray([[False, True], [True, False]], dtype=np.bool_)
    payload = {
        "valid_directional_action_mask": true_mask,
        "valid_directional_action_mask_shuffle": shuffle_mask,
    }
    restored = json.loads(
        json.dumps(evaluate._json_native(payload), allow_nan=False)
    )
    assert np.array_equal(
        np.asarray(restored["valid_directional_action_mask"], dtype=np.bool_),
        true_mask,
    )
    assert np.array_equal(
        np.asarray(
            restored["valid_directional_action_mask_shuffle"], dtype=np.bool_
        ),
        shuffle_mask,
    )


def test_65_two_dimensional_n_by_s_structure_is_preserved():
    original = np.arange(15, dtype=np.int8).reshape(3, 5)
    restored = json.loads(
        json.dumps(evaluate._json_native(original), allow_nan=False)
    )
    assert len(restored) == 3
    assert all(len(row) == 5 for row in restored)
    assert np.asarray(restored, dtype=np.int8).shape == original.shape


def test_66_numpy_scalars_convert_to_exact_json_native_types():
    native = evaluate._json_native({
        "integer": np.int64(7),
        "floating": np.float64(1.25),
        "boolean": np.bool_(True),
    })
    assert type(native["integer"]) is int and native["integer"] == 7
    assert type(native["floating"]) is float and native["floating"] == 1.25
    assert type(native["boolean"]) is bool and native["boolean"] is True


def test_67_allow_nan_false_remains_enforced(tmp_path):
    with pytest.raises(ValueError):
        evaluate.write_json_durable(
            tmp_path / "nonfinite.json", {"value": np.float64(np.nan)}
        )


def test_68_scientific_protocol_and_gates_remain_frozen():
    protocol_path = (
        ROOT
        / "experiments/cyclic_utility/"
        "c2_c2_a0_action_support_diagnostic_protocol.py"
    )
    assert evaluate.file_sha256(protocol_path) == (
        "15b2b3c2dc08db85c77ef28e9fa2f2eb7ebf15e10b9e41528cc8168b2397d2fa"
    )


def test_69_ess_formula_is_unchanged_by_serialization_repair():
    weights = _weights()
    weights[0, :4, 0] = (1.0, 2.0, 3.0, 4.0)
    result = c2.compute_effective_support(weights)
    expected = np.square(np.sum(weights[0, :, 0])) / np.sum(
        np.square(weights[0, :, 0])
    )
    assert result["ESS"][0, 0] == pytest.approx(expected)


def test_70_precommitted_branch_decisions_are_unchanged():
    assert c2.FINAL_DECISIONS == (
        "C2_C2_A0_ACTION_SUPPORT_SUFFICIENCY_DIAGNOSTIC_PASS",
        "C2_C2_A0_ACTION_SUPPORT_SUFFICIENCY_DIAGNOSTIC_FAIL",
    )
    assert c2.PASS_BRANCH_DECISION == (
        "SUPPORT_SUFFICIENCY_VALIDATED; A LATER NEW STAGE MAY TEST "
        "SUPPORT_AWARE_UTILITY_ACTION_USAGE"
    )
    assert c2.FAIL_BRANCH_DECISION == (
        "CLOSE_SCALAR_UTILITY_CORRECTION_BRANCH; MOVE TO "
        "U_CONDITIONED_WEAK_LABEL_ACTION_POLICY"
    )


def test_71_readback_uses_json_native_expected_record(tmp_path):
    original = {"array": np.asarray([[0, 1], [1, 0]], dtype=np.int8)}
    path = tmp_path / "native-expected.json"
    evaluate.write_json_durable(path, original)
    actual = evaluate.read_json(path)
    with pytest.raises(ValueError, match="truth value of an array"):
        actual == original
    evaluate._verify_json_durable_readback(path, original)


def test_72_complete_nested_record_round_trip_is_exact(tmp_path):
    arrays = {
        "DirCorrect": np.asarray([[0, 1], [1, 0]], dtype=np.int8),
        "DirCorrect_shuffle": np.asarray([[1, 0], [0, 1]], dtype=np.int8),
        "valid_directional_action_mask": np.asarray(
            [[True, False], [True, True]], dtype=np.bool_
        ),
        "valid_directional_action_mask_shuffle": np.asarray(
            [[False, True], [True, False]], dtype=np.bool_
        ),
    }
    original = {
        "arrays": arrays,
        "nested": {
            "items": [np.int64(7), np.float64(1.25), np.bool_(True)],
            "labels": ("true", "shuffle"),
        },
    }
    path = tmp_path / "complete-record.json"
    evaluate.write_json_durable(path, original)
    evaluate._verify_json_durable_readback(path, original)
    restored = evaluate.read_json(path)
    for name in ("DirCorrect", "DirCorrect_shuffle"):
        assert np.array_equal(
            np.asarray(restored["arrays"][name], dtype=np.int8), arrays[name]
        )
    for name in (
        "valid_directional_action_mask",
        "valid_directional_action_mask_shuffle",
    ):
        assert np.array_equal(
            np.asarray(restored["arrays"][name], dtype=np.bool_), arrays[name]
        )
    assert restored == evaluate._json_native(original)


def test_73_changed_persisted_integer_is_detected(tmp_path):
    original = {"nested": {"integer": np.int64(7)}}
    path = tmp_path / "changed-integer.json"
    evaluate.write_json_durable(path, original)
    changed = evaluate.read_json(path)
    changed["nested"]["integer"] = 8
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="durable post-seal JSON reload mismatch"
    ):
        evaluate._verify_json_durable_readback(path, original)


def test_74_changed_persisted_boolean_is_detected(tmp_path):
    original = {"nested": {"boolean": np.bool_(True)}}
    path = tmp_path / "changed-boolean.json"
    evaluate.write_json_durable(path, original)
    changed = evaluate.read_json(path)
    changed["nested"]["boolean"] = False
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="durable post-seal JSON reload mismatch"
    ):
        evaluate._verify_json_durable_readback(path, original)


def test_75_missing_nested_key_is_detected(tmp_path):
    original = {"nested": {"kept": 1, "required": np.int64(2)}}
    path = tmp_path / "missing-key.json"
    evaluate.write_json_durable(path, original)
    changed = evaluate.read_json(path)
    del changed["nested"]["required"]
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="durable post-seal JSON reload mismatch"
    ):
        evaluate._verify_json_durable_readback(path, original)


def test_76_changed_array_list_shape_is_detected(tmp_path):
    original = {"array": np.asarray([[0, 1], [1, 0]], dtype=np.int8)}
    path = tmp_path / "changed-shape.json"
    evaluate.write_json_durable(path, original)
    changed = evaluate.read_json(path)
    changed["array"] = [0, 1, 1, 0]
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="durable post-seal JSON reload mismatch"
    ):
        evaluate._verify_json_durable_readback(path, original)


def test_77_durable_verification_reads_and_rejects_mismatch(
    tmp_path, monkeypatch
):
    path = tmp_path / "must-be-read.json"
    calls = []

    def mismatched_read_json(actual_path):
        calls.append(actual_path)
        return {"value": 2}

    monkeypatch.setattr(evaluate, "read_json", mismatched_read_json)
    with pytest.raises(
        RuntimeError, match="durable post-seal JSON reload mismatch"
    ):
        evaluate._verify_json_durable_readback(path, {"value": np.int64(1)})
    assert calls == [path]


def test_78_postseal_save_verifies_every_complete_record(tmp_path, monkeypatch):
    metrics = {
        "directions": [{
            "direction_id": 0,
            "joint_valid": True,
            "support_quantiles": {
                "counts": np.asarray([2, 2], dtype=np.int8)
            },
            "DirCorrect": np.asarray([[0, 1], [1, 0]], dtype=np.int8),
            "valid_directional_action_mask": np.asarray(
                [[True, False], [True, True]], dtype=np.bool_
            ),
        }]
    }
    evaluation = {
        "support_metrics": metrics,
        "summary": {"C2_C2_A0_SEED_PASS": np.bool_(True)},
        "full_GT_audit": {},
        "c0_audit_provenance": {},
        "mapping_audit": {},
        "correctness_logical_sha256": "a" * 64,
        "e_GT_logical_sha256": "b" * 64,
        "U_C0_logical_sha256": "c" * 64,
        "ActionBenefit_logical_sha256": "d" * 64,
    }
    persistence = {
        "output_dir": tmp_path,
        "seal": {},
        "seal_file_sha256": "e" * 64,
    }
    calls = []
    verify = evaluate._verify_json_durable_readback

    def tracked_verify(path, record):
        calls.append((path.name, record))
        verify(path, record)

    monkeypatch.setattr(evaluate, "_verify_json_durable_readback", tracked_verify)
    records = evaluate.save_postseal_results(
        {"seed": 20}, persistence, evaluation
    )
    assert [name for name, _ in calls] == list(records)
    assert [record for _, record in calls] == list(records.values())
    for name, record in records.items():
        assert evaluate.read_json(tmp_path / name) == evaluate._json_native(record)
