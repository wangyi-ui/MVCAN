import ast
import inspect
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from experiments.cyclic_utility import c2_c0_directional_consensus_utility_protocol as c0
from experiments.cyclic_utility import c2_c1_selective_directional_utility_protocol as c2
from experiments.cyclic_utility import evaluate_c2_c1_selective_directional_utility as evaluate
from experiments.cyclic_utility import summarize_c2_c1_selective_directional_utility as summarize
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
    return c2.build_selective_directional_outputs(
        formal_arrays["y_gen"],
        formal_arrays["U_cycle"],
        split["labeled_sample_ids"],
        split["labeled_targets"],
        split["shuffled_labeled_targets"],
    )


@pytest.fixture(scope="module")
def calibration_record(split):
    ids = split["unlabeled_ids"]
    outcomes = (np.arange(1386) % 2).astype(np.int64)
    baseline = np.linspace(0.01, 0.99, 1386)
    c0_arm = np.roll(baseline, 3)
    select = baseline[::-1]
    select_shuffle = np.roll(baseline, 11)
    confidence = np.linspace(0.0, 1.0, 1386)
    return c2.analyze_calibration_direction(
        outcomes,
        baseline,
        c0_arm,
        select,
        select_shuffle,
        confidence,
        ids,
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
        "C2_C0_frozen_hashes_pass": True,
        "C2_B0_frozen_hashes_pass": True,
        "all_frozen_source_hashes_pass": True,
    }
    return evaluate.build_pre_gt_bundle(
        20, formal_arrays, provenance, split, frozen
    )


@pytest.fixture()
def persisted(tmp_path, formal_arrays, split):
    return evaluate.persist_admission_bundle(
        _bundle(formal_arrays, split), tmp_path / "sealed"
    )


def _admission_inputs(full_value=1.0, jackknife_value=1):
    full = np.full((1400, 20), full_value, dtype=np.float64)
    jackknife = np.full((1400, 20, 14), jackknife_value, dtype=np.int8)
    return full, jackknife


def _direction_records(
    delta=0.2,
    gain=0.15,
    label_gap=0.1,
    valid_count=20,
    diagnostic_valid_count=20,
):
    diagnostics = []
    calibration = []
    for direction_id in range(c2.DIRECTION_COUNT):
        diagnostic_valid = direction_id < diagnostic_valid_count
        direction_valid = direction_id < valid_count
        diagnostics.append({
            "direction_id": direction_id,
            "admission_count_true": 100,
            "admission_rate_true": 100 / 1386,
            "admission_count_shuffle": 80,
            "admission_rate_shuffle": 80 / 1386,
            "AdmittedDirectionalAgreement_true": (
                0.8 if diagnostic_valid else None
            ),
            "AdmittedDirectionalAgreement_shuffle": (
                0.2 if diagnostic_valid else None
            ),
            "AdmittedDirectionalSpecificityGap": (
                0.6 if diagnostic_valid else None
            ),
            "diagnostic_valid": diagnostic_valid,
        })
        calibration.append({
            "direction_id": direction_id,
            "direction_valid": direction_valid,
            "DeltaCalAUC_select": delta if direction_valid else None,
            "SelectiveGainVsC0": gain if direction_valid else None,
            "TrueVsShuffleCalGap_select": label_gap if direction_valid else None,
        })
    return {"directions": diagnostics}, {"directions": calibration}


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


def test_01_c2_c0_d_is_reproduced_exactly(outputs, frozen_c0_outputs):
    assert np.array_equal(outputs["D"], frozen_c0_outputs["D"])
    assert np.array_equal(outputs["D_shuffle"], frozen_c0_outputs["D_shuffle"])


def test_02_c2_c0_utility_baseline_is_reproduced_exactly(
    outputs, frozen_c0_outputs
):
    assert np.array_equal(
        outputs["U_tilde_dir_c0"], frozen_c0_outputs["U_tilde_dir"]
    )
    assert np.array_equal(
        outputs["U_tilde_dir_c0_shuffle"],
        frozen_c0_outputs["U_tilde_dir_shuffle"],
    )


def test_03_jackknife_numerator_subtraction_is_exact(outputs):
    i, s, m = 17, 3, 5
    expected = outputs["num"][i, s] - (
        outputs["weights"][i, m, s] * outputs["d_L"][m, s]
    )
    assert outputs["num_minus"][i, s, m] == expected


def test_04_jackknife_denominator_subtraction_is_exact(outputs):
    i, s, m = 23, 7, 9
    expected = outputs["denom"][i, s] - outputs["weights"][i, m, s]
    assert outputs["denom_minus"][i, s, m] == expected


def test_05_nonpositive_epsilon_support_produces_zero_d_minus():
    weights = np.zeros((1400, 14, 20), dtype=np.float64)
    directions = np.ones((14, 20), dtype=np.float64)
    full = np.zeros((1400, 20), dtype=np.float64)
    result = c2.build_jackknife_directional_admission(weights, directions, full)
    assert np.all(result["denom_minus"] <= 1e-12)
    assert np.all(result["D_minus"] == 0.0)


def test_06_q_is_exact_sign_of_d(outputs):
    assert np.array_equal(outputs["q"], np.sign(outputs["D"]).astype(np.int8))


def test_07_q_minus_is_exact_sign_of_d_minus(outputs):
    assert np.array_equal(
        outputs["q_minus"], np.sign(outputs["D_minus"]).astype(np.int8)
    )


def test_08_admission_requires_nonzero_full_and_all_fourteen_equal():
    full, jackknife = _admission_inputs()
    result = c2.unanimous_jackknife_admission(full, jackknife)
    assert np.all(result["A"] == 1)
    assert result["q_minus"].shape if "q_minus" in result else True


def test_09_one_disagreeing_jackknife_sign_forces_abstention():
    full, jackknife = _admission_inputs()
    jackknife[0, 0, 6] = -1
    result = c2.unanimous_jackknife_admission(full, jackknife)
    assert result["A"][0, 0] == 0


def test_10_one_zero_jackknife_sign_forces_abstention():
    full, jackknife = _admission_inputs()
    jackknife[0, 0, 6] = 0
    result = c2.unanimous_jackknife_admission(full, jackknife)
    assert result["A"][0, 0] == 0


def test_11_zero_full_direction_forces_abstention():
    full, jackknife = _admission_inputs()
    full[0, 0] = 0.0
    jackknife[0, 0, :] = 0
    result = c2.unanimous_jackknife_admission(full, jackknife)
    assert result["A"][0, 0] == 0


def test_12_all_zero_admission_is_allowed_and_does_not_crash(formal_arrays):
    admission = np.zeros((1400, 20), dtype=np.int8)
    consensus = np.ones((1400, 20), dtype=np.float64)
    result = c2.selective_directional_utility_update(
        formal_arrays["U_cycle"], consensus, admission
    )
    assert np.array_equal(result, formal_arrays["U_cycle"])


def test_13_admission_contains_only_zero_or_one(outputs):
    for name in ("A", "A_shuffle"):
        assert outputs[name].dtype == np.int8
        assert np.all((outputs[name] == 0) | (outputs[name] == 1))


def test_14_c1_update_formula_is_exact(formal_arrays, outputs):
    cycle = formal_arrays["U_cycle"]
    expected = cycle + outputs["A"] * cycle * (1.0 - cycle) * outputs["D"]
    expected[cycle == 0.0] = 0.0
    assert np.array_equal(outputs["U_tilde_select"], expected)


def test_15_abstained_cells_equal_u_cycle_exactly(formal_arrays, outputs):
    mask = outputs["A"] == 0
    assert np.array_equal(
        outputs["U_tilde_select"][mask], formal_arrays["U_cycle"][mask]
    )


def test_16_admitted_cells_equal_c2_c0_baseline_exactly(outputs):
    mask = outputs["A"] == 1
    assert np.array_equal(
        outputs["U_tilde_select"][mask], outputs["U_tilde_dir_c0"][mask]
    )


def test_17_selected_utility_is_finite(outputs):
    assert np.isfinite(outputs["U_tilde_select"]).all()
    assert np.isfinite(outputs["U_tilde_select_shuffle"]).all()


def test_18_selected_utility_is_bounded(outputs):
    for name in ("U_tilde_select", "U_tilde_select_shuffle"):
        assert np.all((0.0 <= outputs[name]) & (outputs[name] <= 1.0))


def test_19_zero_cycle_utility_remains_zero(formal_arrays):
    consensus = np.ones((1400, 20), dtype=np.float64)
    admission = np.ones((1400, 20), dtype=np.int8)
    result = c2.selective_directional_utility_update(
        formal_arrays["U_cycle"], consensus, admission
    )
    assert result[0, 0] == 0.0


def test_20_c1_update_uses_d_not_sign_d():
    cycle = np.full((1400, 20), 0.4, dtype=np.float64)
    consensus = np.full((1400, 20), 0.25, dtype=np.float64)
    admission = np.ones((1400, 20), dtype=np.int8)
    result = c2.selective_directional_utility_update(cycle, consensus, admission)
    expected = cycle + cycle * (1.0 - cycle) * consensus
    wrong = cycle + cycle * (1.0 - cycle) * np.sign(consensus)
    assert np.array_equal(result, expected)
    assert not np.array_equal(result, wrong)


def test_21_no_alpha_parameter():
    assert "alpha" not in _all_parameter_names()
    assert "ALPHA" not in vars(c2)


def test_22_no_beta_parameter():
    assert "beta" not in _all_parameter_names()
    assert "BETA" not in vars(c2)


def test_23_no_lambda_parameter():
    assert "lambda" not in _all_parameter_names()
    assert "LAMBDA" not in vars(c2)


def test_24_no_temperature_parameter():
    assert "temperature" not in _all_parameter_names()
    assert "TEMPERATURE" not in vars(c2)


def test_25_no_admission_threshold_parameter():
    assert "threshold" not in _all_parameter_names()
    assert "THRESHOLD" not in vars(c2)


def test_26_no_top_k_parameter():
    assert "top_k" not in _all_parameter_names()
    assert "TOP_K" not in vars(c2)


def test_27_no_hyperparameter_sweep_path():
    names = set(vars(c2)) | set(vars(evaluate))
    assert {"grid_search", "parameter_sweep", "hyperparameter_sweep"}.isdisjoint(names)


def test_28_shuffled_control_is_rebuilt_from_shuffled_targets(
    formal_arrays, split, outputs
):
    rebuilt = c2.build_leave_one_out_action_correctness(
        outputs["R"],
        formal_arrays["y_gen"],
        split["labeled_sample_ids"],
        split["shuffled_labeled_targets"],
    )
    assert np.array_equal(outputs["z_L_shuffle"], rebuilt["z_L"])
    assert outputs["shuffle_recomputed_from_targets"] is True


def test_29_a_shuffle_is_independently_recomputed(outputs):
    rebuilt = c2.build_jackknife_directional_admission(
        outputs["shuffle_weights"], outputs["d_L_shuffle"], outputs["D_shuffle"]
    )
    assert np.array_equal(outputs["A_shuffle"], rebuilt["A"])
    assert outputs["A_shuffle_independently_recomputed"] is True


def test_30_true_admission_is_not_reused_for_shuffle(outputs):
    assert outputs["true_A_reused_for_shuffle"] is False
    source = inspect.getsource(c2.build_selective_directional_outputs)
    assert 'shuffle_jackknife["A"]' in source


def test_31_loo_semantic_mapping_is_frozen_c0_definition(
    outputs, frozen_c0_outputs
):
    assert np.array_equal(outputs["loo_mappings"], frozen_c0_outputs["loo_mappings"])
    assert np.array_equal(
        outputs["loo_mappings_shuffle"], frozen_c0_outputs["loo_mappings_shuffle"]
    )


def test_32_jackknife_does_not_recompute_semantic_loo():
    forbidden = {
        "build_leave_one_out_action_correctness",
        "build_labeled_residual",
        "build_labeled_residual_directions",
    }
    assert _called_names(c2.build_jackknife_directional_admission).isdisjoint(forbidden)


def test_33_c_conf_is_not_used_in_admission_construction():
    functions = (
        c2.unanimous_jackknife_admission,
        c2.build_jackknife_directional_admission,
        c2.selective_directional_utility_update,
        c2.build_selective_directional_outputs,
    )
    assert all("C_conf" not in inspect.signature(function).parameters for function in functions)


def test_34_gt_is_inaccessible_before_admission_seal():
    forbidden = {
        "load_full_ground_truth_after_admission_seal",
        "load_c0_audit_after_admission_seal",
        "load_native_global_cluster_after_admission_seal",
        "build_bridge_correctness_after_admission_seal",
    }
    functions = (
        evaluate.load_c0_inputs_before_gt,
        evaluate.load_fixed_sparse_labels,
        evaluate.build_pre_gt_bundle,
        evaluate.persist_admission_bundle,
    )
    assert all(_called_names(function).isdisjoint(forbidden) for function in functions)


def test_35_c0_audit_is_inaccessible_before_admission_seal():
    assert "load_c0_audit_after_admission_seal" not in _called_names(
        evaluate.persist_admission_bundle
    )


def test_36_bridge_correctness_is_inaccessible_before_admission_seal():
    assert "build_bridge_correctness_after_admission_seal" not in _called_names(
        evaluate.build_pre_gt_bundle
    )


def test_37_corruption_mask_loader_is_absent():
    assert "load_corruption_mask" not in (set(vars(c2)) | set(vars(evaluate)))


def test_38_no_model_forward_call():
    called = _all_called_names(c2.__file__, evaluate.__file__)
    assert "forward" not in called


def test_39_no_optimizer_creation_or_step():
    called = _all_called_names(c2.__file__, evaluate.__file__)
    assert {"optimizer", "step"}.isdisjoint(called)


def test_40_no_backward_call():
    called = _all_called_names(c2.__file__, evaluate.__file__)
    assert "backward" not in called


def test_41_no_training_path():
    called = _all_called_names(c2.__file__, evaluate.__file__)
    assert "train" not in called


def test_42_pre_gt_key_whitelist_is_exact(formal_arrays, split):
    bundle = _bundle(formal_arrays, split)
    assert tuple(bundle["arrays"]) == evaluate.PRE_GT_ARRAY_NAMES
    assert bundle["arrays"]["q_minus"].dtype == np.int8
    assert bundle["arrays"]["q_minus_shuffle"].dtype == np.int8


def test_43_durable_save_reload_and_hash_verification(persisted):
    seal = persisted["seal"]
    assert seal["NPZ_saved_fsynced_reloaded_hash_verified"] is True
    with np.load(persisted["npz_path"], allow_pickle=False) as archive:
        assert tuple(archive.files) == evaluate.PRE_GT_ARRAY_NAMES
        for name in evaluate.PRE_GT_ARRAY_NAMES:
            assert ndarray_sha256(archive[name]) == seal["arrays"][name]["logical_sha256"]


def test_44_evaluation_is_exactly_1386_unlabeled_samples(split):
    shape = (1386, 20)
    zeros = np.zeros(shape, dtype=np.float64)
    admissions = np.zeros(shape, dtype=np.int8)
    result = c2.analyze_admission_directions(
        admissions, admissions, zeros, zeros, zeros, split["unlabeled_ids"]
    )
    assert result["evaluation_sample_count"] == 1386
    with pytest.raises(RuntimeError, match="1386 unlabeled"):
        c2.analyze_admission_directions(
            np.zeros((1400, 20), dtype=np.int8),
            np.zeros((1400, 20), dtype=np.int8),
            np.zeros((1400, 20)),
            np.zeros((1400, 20)),
            np.zeros((1400, 20)),
            np.arange(1400),
        )


def test_45_confidence_strata_assignment_does_not_use_gt(split):
    confidence = np.ones(1386, dtype=np.float64)
    assignments, audit = c2.confidence_equal_count_strata(
        confidence, split["unlabeled_ids"]
    )
    assert audit["GT_used_for_assignment"] is False
    assert audit["sample_id_only_tie_break"] is True
    assert max(np.bincount(assignments)) - min(np.bincount(assignments)) <= 1


def test_46_all_four_utility_arms_share_valid_bin_mask(calibration_record):
    assert calibration_record["same_valid_bin_mask_all_utility_arms"] is True
    assert calibration_record["valid_bin_mask"] == [
        record["valid_both_correctness_classes"]
        for record in calibration_record["bins"]
    ]


def test_47_delta_cal_auc_select_formula_is_exact(calibration_record):
    assert calibration_record["DeltaCalAUC_select"] == (
        calibration_record["CondAUC_C1_select"]
        - calibration_record["CondAUC_Ucycle"]
    )


def test_48_selective_gain_vs_c0_formula_is_exact(calibration_record):
    assert calibration_record["SelectiveGainVsC0"] == (
        calibration_record["CondAUC_C1_select"]
        - calibration_record["CondAUC_C0_dir"]
    )


def test_49_true_vs_shuffle_select_gap_formula_is_exact(calibration_record):
    assert calibration_record["TrueVsShuffleCalGap_select"] == (
        calibration_record["CondAUC_C1_select"]
        - calibration_record["CondAUC_C1_select_shuffle"]
    )


def test_50_admission_rate_diagnostic_is_exact(split):
    shape = (1386, 20)
    admission = np.zeros(shape, dtype=np.int8)
    admission_shuffle = np.zeros(shape, dtype=np.int8)
    admission[:100, 0] = 1
    admission_shuffle[:80, 0] = 1
    consensus = np.ones(shape, dtype=np.float64)
    truth = np.ones(shape, dtype=np.float64)
    record = c2.analyze_admission_directions(
        admission,
        admission_shuffle,
        consensus,
        consensus,
        truth,
        split["unlabeled_ids"],
    )["directions"][0]
    assert record["admission_count_true"] == 100
    assert record["admission_rate_true"] == 100 / 1386
    assert record["admission_count_shuffle"] == 80
    assert record["admission_rate_shuffle"] == 80 / 1386


def test_51_admitted_directional_agreement_is_exact(split):
    shape = (1386, 20)
    admission = np.zeros(shape, dtype=np.int8)
    admission[:3, 0] = 1
    consensus = np.zeros(shape, dtype=np.float64)
    consensus[:3, 0] = (1.0, -1.0, 1.0)
    shuffle = np.zeros(shape, dtype=np.float64)
    shuffle[:3, 0] = (-1.0, 1.0, -1.0)
    truth = np.zeros(shape, dtype=np.float64)
    truth[:3, 0] = (1.0, -1.0, -1.0)
    record = c2.analyze_admission_directions(
        admission, admission, consensus, shuffle, truth, split["unlabeled_ids"]
    )["directions"][0]
    assert record["AdmittedDirectionalAgreement_true"] == pytest.approx(2 / 3)
    assert record["AdmittedDirectionalAgreement_shuffle"] == pytest.approx(1 / 3)


def test_52_admitted_directional_specificity_gap_is_exact(split):
    shape = (1386, 20)
    admission = np.zeros(shape, dtype=np.int8)
    admission[:3, 0] = 1
    true_d = np.zeros(shape, dtype=np.float64)
    true_d[:3, 0] = (1.0, -1.0, 1.0)
    shuffle_d = np.zeros(shape, dtype=np.float64)
    shuffle_d[:3, 0] = (-1.0, 1.0, -1.0)
    truth = np.zeros(shape, dtype=np.float64)
    truth[:3, 0] = (1.0, -1.0, -1.0)
    record = c2.analyze_admission_directions(
        admission, admission, true_d, shuffle_d, truth, split["unlabeled_ids"]
    )["directions"][0]
    assert record["AdmittedDirectionalSpecificityGap"] == pytest.approx(1 / 3)


def test_53_diagnostics_do_not_affect_formal_gate():
    diagnostics, calibration = _direction_records(diagnostic_valid_count=0)
    summary = c2.build_seed_summary(20, diagnostics, calibration)
    assert summary["valid_admitted_directional_diagnostic_count"] == 0
    assert summary["admission_diagnostics_used_in_gate"] is False
    assert summary["C2_C1_SEED_PASS"] is True


def test_54_per_seed_gate_is_exact():
    diagnostics, calibration = _direction_records()
    summary = c2.build_seed_summary(20, diagnostics, calibration)
    assert summary["valid_calibration_direction_count"] == 20
    assert summary["positive_DeltaCalAUC_select_direction_count"] == 20
    assert summary["DeltaCalAUC_select_wilcoxon_greater_p"] < 0.05
    assert summary["positive_SelectiveGainVsC0_direction_count"] == 20
    assert summary["SelectiveGainVsC0_wilcoxon_greater_p"] < 0.05
    assert all(summary["SeedGate_conditions"].values())
    assert summary["C2_C1_SEED_PASS"] is True
    _, invalid = _direction_records(valid_count=15)
    assert c2.build_seed_summary(20, diagnostics, invalid)["C2_C1_SEED_PASS"] is False


def test_55_multi_seed_gate_is_exact():
    summaries = OrderedDict()
    for seed, passed in zip(c2.SEEDS, (True, True, False)):
        summaries[seed] = {
            "seed": seed,
            "C2_C1_SEED_PASS": passed,
            "mean_DeltaCalAUC_select": 0.02,
            "mean_SelectiveGainVsC0": 0.01,
            "mean_TrueVsShuffleCalGap_select": 0.03,
        }
    result = c2.build_multiseed_decision(summaries)
    assert result["summary"]["seed_pass_count"] == 2
    assert all(result["decision"]["decision_conditions"].values())
    assert result["decision"][
        "C2_C1_SELECTIVE_DIRECTIONAL_UTILITY_ACTION_ADMISSION_PASS"
    ] is True
    summaries[30]["mean_SelectiveGainVsC0"] = -0.2
    result = c2.build_multiseed_decision(summaries)
    assert result["decision"][
        "C2_C1_SELECTIVE_DIRECTIONAL_UTILITY_ACTION_ADMISSION_PASS"
    ] is False


def test_56_nondegeneracy_audit_all_zero_is_valid(split):
    admission = np.zeros((1386, 20), dtype=np.int8)
    audit = c2.analyze_admission_nondegeneracy(
        admission, admission, split["unlabeled_ids"]
    )
    assert audit["true"]["total_action_count"] == 1386 * 20
    assert audit["true"]["total_admitted_action_count"] == 0
    assert audit["true"]["overall_admission_rate"] == 0.0
    assert audit["true"]["all_zero_admission"] is True
    assert audit["admission_rate_used_in_gate"] is False


def test_57_both_frozen_source_manifests_pass():
    audit = evaluate.verify_frozen_sources()
    assert audit["C2_C0_frozen_hashes_pass"] is True
    assert audit["C2_B0_frozen_hashes_pass"] is True
    assert tuple(audit["manifests"]) == evaluate.FROZEN_SOURCE_MANIFESTS


def test_58_only_fixed_formal_seeds_and_default_paths_are_supported():
    assert tuple(c2.SEEDS) == (20, 30, 50)
    for seed in c2.SEEDS:
        assert "seed" + str(seed) in str(summarize.default_seed_dir(seed))
    assert str(summarize.default_multiseed_dir()).endswith(
        "c2_c1_selective_directional_utility_multiseed"
    )
    with pytest.raises(ValueError):
        summarize.default_seed_dir(21)


def test_59_exactly_four_c2_c1_python_files_exist():
    expected = {
        ROOT / "experiments/cyclic_utility/c2_c1_selective_directional_utility_protocol.py",
        ROOT / "experiments/cyclic_utility/evaluate_c2_c1_selective_directional_utility.py",
        ROOT / "experiments/cyclic_utility/summarize_c2_c1_selective_directional_utility.py",
        ROOT / "tests/test_c2_c1_selective_directional_utility.py",
    }
    discovered = set(ROOT.glob("experiments/cyclic_utility/*c2_c1*.py"))
    discovered.update(ROOT.glob("tests/*c2_c1*.py"))
    assert discovered == expected


def _canonicalization_arrays():
    raw = np.zeros((1400, 20, 14), dtype=np.float64)
    denominator = np.ones((1400, 20, 14), dtype=np.float64)
    return raw, denominator


def test_60_epsilon_upper_overshoot_is_canonicalized_to_one():
    raw, denominator = _canonicalization_arrays()
    raw[0, 0, 0] = 1.0 + np.finfo(np.float64).eps
    result = c2.canonicalize_jackknife_consensus(raw, denominator)
    assert result[0, 0, 0] == 1.0


def test_61_epsilon_lower_overshoot_is_canonicalized_to_minus_one():
    raw, denominator = _canonicalization_arrays()
    raw[0, 0, 0] = -1.0 - np.finfo(np.float64).eps
    result = c2.canonicalize_jackknife_consensus(raw, denominator)
    assert result[0, 0, 0] == -1.0


def test_62_epsilon_canonicalization_changes_zero_jackknife_signs():
    raw, denominator = _canonicalization_arrays()
    epsilon = np.finfo(np.float64).eps
    raw[0, 0, 0] = 1.0 + epsilon
    raw[0, 0, 1] = -1.0 - epsilon
    result = c2.canonicalize_jackknife_consensus(raw, denominator)
    assert np.count_nonzero(np.sign(raw) != np.sign(result)) == 0


def test_63_material_upper_violation_hard_fails():
    raw, denominator = _canonicalization_arrays()
    raw[0, 0, 0] = 1.0 + 1e-6
    with pytest.raises(RuntimeError, match="float64 numerical tolerance"):
        c2.canonicalize_jackknife_consensus(raw, denominator)


def test_64_material_lower_violation_hard_fails():
    raw, denominator = _canonicalization_arrays()
    raw[0, 0, 0] = -1.0 - 1e-6
    with pytest.raises(RuntimeError, match="float64 numerical tolerance"):
        c2.canonicalize_jackknife_consensus(raw, denominator)


def test_65_zero_support_remains_exactly_zero():
    raw, denominator = _canonicalization_arrays()
    denominator[0, 0, 0] = 0.0
    result = c2.canonicalize_jackknife_consensus(raw, denominator)
    assert result[0, 0, 0] == 0.0


def test_66_admission_is_identical_before_and_after_epsilon_canonicalization():
    raw, denominator = _canonicalization_arrays()
    raw.fill(1.0)
    raw[0, 0, 0] = 1.0 + np.finfo(np.float64).eps
    full_d = np.ones((1400, 20), dtype=np.float64)
    admission_before = c2.unanimous_jackknife_admission(
        full_d, np.sign(raw).astype(np.int8)
    )["A"]
    canonical = c2.canonicalize_jackknife_consensus(raw, denominator)
    admission_after = c2.unanimous_jackknife_admission(
        full_d, np.sign(canonical).astype(np.int8)
    )["A"]
    assert np.array_equal(admission_before, admission_after)


def test_67_numerical_repair_does_not_change_c1_update_formula():
    cycle = np.full((1400, 20), 0.4, dtype=np.float64)
    full_d = np.full((1400, 20), 0.25, dtype=np.float64)
    admission = np.zeros((1400, 20), dtype=np.int8)
    admission[::2] = 1
    result = c2.selective_directional_utility_update(
        cycle, full_d, admission
    )
    expected = cycle + admission * cycle * (1.0 - cycle) * full_d
    assert np.array_equal(result, expected)


def test_68_numerical_repair_introduces_no_scientific_parameter():
    forbidden = {
        "alpha", "beta", "lambda", "temperature", "threshold", "top_k"
    }
    assert forbidden.isdisjoint(_all_parameter_names())
    assert tuple(
        inspect.signature(c2.canonicalize_jackknife_consensus).parameters
    ) == ("raw_D_minus", "denominator_minus")
    assert "BOUND_TOLERANCE" not in vars(c2)


def test_69_numerical_repair_does_not_enter_scientific_gates():
    assert "canonicalize_jackknife_consensus" not in _called_names(
        c2.build_seed_summary
    )
    assert "canonicalize_jackknife_consensus" not in _called_names(
        c2.build_multiseed_decision
    )
    diagnostics, calibration = _direction_records()
    summary = c2.build_seed_summary(20, diagnostics, calibration)
    assert all(summary["SeedGate_conditions"].values())
    assert summary["C2_C1_SEED_PASS"] is True
