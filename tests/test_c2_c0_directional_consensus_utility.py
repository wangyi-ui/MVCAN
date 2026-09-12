import ast
import hashlib
import inspect
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from experiments.cyclic_utility import c2_c0_directional_consensus_utility_protocol as c2
from experiments.cyclic_utility import evaluate_c2_c0_directional_consensus_utility as evaluate
from experiments.cyclic_utility import summarize_c2_c0_directional_consensus_utility as summarize
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
def outputs(formal_arrays, split):
    return c2.build_directional_consensus_outputs(
        formal_arrays["y_gen"],
        formal_arrays["U_cycle"],
        split["labeled_sample_ids"],
        split["labeled_targets"],
        split["shuffled_labeled_targets"],
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
        "all_frozen_C2_B0_hashes_pass": True,
    }
    return evaluate.build_pre_gt_bundle(
        20, formal_arrays, provenance, split, frozen
    )


@pytest.fixture()
def persisted(tmp_path, formal_arrays, split):
    return evaluate.persist_directional_bundle(
        _bundle(formal_arrays, split), tmp_path / "sealed"
    )


def _direction_records(
    delta=0.2,
    calibration_gap=0.1,
    calibration_valid_count=20,
    diagnostic_valid_count=20,
):
    diagnostic = []
    calibration = []
    for direction_id in range(c2.DIRECTION_COUNT):
        diagnostic_valid = direction_id < diagnostic_valid_count
        calibration_valid = direction_id < calibration_valid_count
        diagnostic.append({
            "direction_id": direction_id,
            "diagnostic_valid": diagnostic_valid,
            "directional_sign_agreement_true": 0.7 if diagnostic_valid else None,
            "directional_sign_agreement_shuffle": 0.5 if diagnostic_valid else None,
            "DirectionalSpecificityGap": 0.2 if diagnostic_valid else None,
        })
        calibration.append({
            "direction_id": direction_id,
            "direction_valid": calibration_valid,
            "DeltaCalAUC_dir": delta if calibration_valid else None,
            "TrueVsShuffleCalGap_dir": (
                calibration_gap if calibration_valid else None
            ),
        })
    return {"directions": diagnostic}, {"directions": calibration}


def _called_names(function):
    tree = ast.parse(inspect.getsource(function))
    return {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_01_sign_residual_conversion_positive_negative_zero():
    residual = np.zeros((14, 20), dtype=np.float64)
    residual[0, 0] = 0.25
    residual[0, 1] = -0.75
    residual[0, 2] = 0.0
    directions = c2.build_labeled_residual_directions(residual)
    assert directions[0, 0] == 1.0
    assert directions[0, 1] == -1.0
    assert directions[0, 2] == 0.0


def test_02_d_weighted_consensus_formula_is_exact(outputs):
    sample_id, direction_id = 17, 3
    weights = outputs["weights"][sample_id, :, direction_id]
    denominator = weights.sum()
    expected = (
        np.sum(weights * outputs["d_L"][:, direction_id]) / denominator
        if denominator > 1e-12
        else 0.0
    )
    assert outputs["D"][sample_id, direction_id] == expected


def test_03_denominator_at_or_below_epsilon_produces_zero(formal_arrays, split):
    actions = formal_arrays["y_gen"].copy()
    labeled = split["labeled_sample_ids"]
    actions[labeled, 0] = 0
    actions[0, 0] = 6
    directions = np.ones((14, 20), dtype=np.float64)
    result = c2.propagate_action_context_directional_consensus(
        c2.build_vote_semantic_state(formal_arrays["y_gen"]),
        actions,
        labeled,
        directions,
    )
    assert result["denom"][0, 0] <= 1e-12
    assert result["D"][0, 0] == 0.0


def test_04_d_is_finite_and_bounded(outputs):
    for name in ("D", "D_shuffle"):
        assert outputs[name].shape == (1400, 20)
        assert np.isfinite(outputs[name]).all()
        assert np.all((-1.0 <= outputs[name]) & (outputs[name] <= 1.0))


def test_05_utility_update_formula_is_exact(formal_arrays, outputs):
    cycle = formal_arrays["U_cycle"]
    expected = cycle + cycle * (1.0 - cycle) * outputs["D"]
    expected[cycle == 0.0] = 0.0
    actual = c2.directional_utility_update(cycle, outputs["D"])
    assert np.array_equal(actual, expected)


def test_06_utility_update_is_finite_and_bounded(outputs):
    for name in ("U_tilde_dir", "U_tilde_dir_shuffle"):
        assert np.isfinite(outputs[name]).all()
        assert np.all((0.0 <= outputs[name]) & (outputs[name] <= 1.0))


def test_07_zero_cycle_utility_remains_zero(formal_arrays):
    consensus = np.ones((1400, 20), dtype=np.float64)
    calibrated = c2.directional_utility_update(
        formal_arrays["U_cycle"], consensus
    )
    assert calibrated[0, 0] == 0.0


def test_08_no_tunable_update_parameter_exists():
    assert tuple(inspect.signature(c2.directional_utility_update).parameters) == (
        "U_cycle", "D"
    )
    forbidden_constants = {
        "ALPHA", "BETA", "LAMBDA", "TEMPERATURE", "THRESHOLD", "TOP_K"
    }
    assert forbidden_constants.isdisjoint(vars(c2))


def test_09_shuffled_control_is_rebuilt_from_targets(formal_arrays, split, outputs):
    rebuilt = c2.build_leave_one_out_action_correctness(
        outputs["R"],
        formal_arrays["y_gen"],
        split["labeled_sample_ids"],
        split["shuffled_labeled_targets"],
    )
    assert np.array_equal(outputs["z_L_shuffle"], rebuilt["z_L"])
    assert outputs["shuffle_recomputed_from_targets"] is True


def test_10_shuffled_pipeline_is_not_a_final_d_shuffle(outputs):
    assert outputs["shuffle_only_final_D"] is False
    assert ndarray_sha256(outputs["d_L"]) != ndarray_sha256(outputs["d_L_shuffle"])
    source = inspect.getsource(c2.build_directional_consensus_outputs)
    assert "random" not in source
    assert "permutation" not in source


def test_11_leave_one_out_self_exclusion_is_exact(outputs):
    for name in (
        "loo_included_label_mask", "loo_included_label_mask_shuffle"
    ):
        masks = outputs[name]
        assert masks.shape == (14, 14)
        assert not np.any(np.diag(masks))
        assert np.all(masks.sum(axis=1) == 13)


def test_12_frozen_b0_weight_topology_is_used(formal_arrays, split, outputs):
    source = c2.b0.propagate_action_context_residuals(
        outputs["R"], formal_arrays["y_gen"],
        split["labeled_sample_ids"], outputs["d_L"],
    )
    assert np.array_equal(outputs["semantic_similarity"], source["semantic_similarity"])
    assert np.array_equal(outputs["action_match"], source["action_match"])
    assert np.array_equal(outputs["weights"], source["weights"])


def test_13_c_conf_is_absent_from_directional_propagation():
    functions = (
        c2.build_labeled_residual_directions,
        c2.propagate_action_context_directional_consensus,
        c2.directional_utility_update,
        c2.build_directional_consensus_outputs,
    )
    assert all("C_conf" not in inspect.signature(function).parameters for function in functions)


def test_14_gt_cannot_be_loaded_before_directional_seal():
    forbidden = {
        "load_full_ground_truth_after_directional_seal",
        "load_c0_audit_after_directional_seal",
        "load_native_global_cluster_after_directional_seal",
        "build_bridge_correctness_after_directional_seal",
    }
    functions = (
        evaluate.load_c0_inputs_before_gt,
        evaluate.load_fixed_sparse_labels,
        evaluate.build_pre_gt_bundle,
        evaluate.persist_directional_bundle,
    )
    assert all(_called_names(function).isdisjoint(forbidden) for function in functions)


def test_15_c0_audit_cannot_be_loaded_before_directional_seal():
    assert "load_c0_audit_after_directional_seal" not in _called_names(
        evaluate.persist_directional_bundle
    )


def test_16_bridge_correctness_cannot_be_built_before_directional_seal():
    assert "build_bridge_correctness_after_directional_seal" not in _called_names(
        evaluate.build_pre_gt_bundle
    )


def test_17_no_corruption_mask_or_oracle_loader_exists():
    names = set(vars(c2)) | set(vars(evaluate))
    assert "load_corruption_mask" not in names
    assert "load_oracle" not in names


def test_18_no_model_forward_optimizer_backward_or_training_calls():
    trees = [
        ast.parse(Path(c2.__file__).read_text(encoding="utf-8")),
        ast.parse(Path(evaluate.__file__).read_text(encoding="utf-8")),
    ]
    called = set()
    for tree in trees:
        called.update(
            node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))
        )
    assert called.isdisjoint({"forward", "optimizer", "backward", "step", "train"})


def test_19_pre_gt_output_key_whitelist_is_exact(formal_arrays, split):
    bundle = _bundle(formal_arrays, split)
    assert tuple(bundle["arrays"]) == evaluate.PRE_GT_ARRAY_NAMES
    assert set(bundle["arrays"]) == set(evaluate.PRE_GT_ARRAY_NAMES)


def test_20_seal_reloads_and_verifies_every_logical_hash(persisted):
    seal = persisted["seal"]
    assert seal["NPZ_saved_fsynced_reloaded_hash_verified"] is True
    assert tuple(seal["arrays"]) == evaluate.PRE_GT_ARRAY_NAMES
    with np.load(persisted["npz_path"], allow_pickle=False) as archive:
        assert tuple(archive.files) == evaluate.PRE_GT_ARRAY_NAMES
        for name in evaluate.PRE_GT_ARRAY_NAMES:
            assert ndarray_sha256(archive[name]) == seal["arrays"][name]["logical_sha256"]


def test_21_seal_records_all_required_false_boundaries(persisted):
    keys = (
        "GT_loaded_before_directional_seal",
        "c0_audit_loaded_before_directional_seal",
        "Bridge_correctness_loaded_before_directional_seal",
        "corruption_mask_loaded",
        "training_performed",
        "model_forward_called",
        "optimizer_created",
        "backward_called",
        "optimizer_step_called",
    )
    assert all(persisted["seal"][key] is False for key in keys)


def test_22_directional_diagnostic_uses_only_fixed_1386_unlabeled(split):
    zeros = np.zeros((1386, 20), dtype=np.float64)
    result = c2.analyze_directional_sign_agreement(
        zeros, zeros, zeros, split["unlabeled_ids"]
    )
    assert result["evaluation_sample_count"] == 1386
    with pytest.raises(RuntimeError, match="1386 unlabeled"):
        c2.analyze_directional_sign_agreement(
            np.zeros((1400, 20)), np.zeros((1400, 20)),
            np.zeros((1400, 20)), np.arange(1400),
        )


def test_23_confidence_strata_assignment_does_not_use_gt(split):
    confidence = np.ones(1386, dtype=np.float64)
    assignments, audit = c2.confidence_equal_count_strata(
        confidence, split["unlabeled_ids"]
    )
    assert audit["GT_used_for_assignment"] is False
    assert audit["sample_id_only_tie_break"] is True
    assert max(np.bincount(assignments)) - min(np.bincount(assignments)) <= 1


def test_24_identical_valid_bin_mask_for_all_three_utility_arms(split):
    ids = split["unlabeled_ids"]
    outcomes = (np.arange(1386) % 2).astype(np.int64)
    baseline = np.linspace(0.01, 0.99, 1386)
    calibrated = baseline[::-1]
    shuffled = np.roll(baseline, 7)
    confidence = np.linspace(0.0, 1.0, 1386)
    record = c2.analyze_calibration_direction(
        outcomes, baseline, calibrated, shuffled, confidence, ids, 0
    )
    assert record["same_valid_bin_mask_all_utility_arms"] is True
    assert record["valid_bin_mask"] == [
        item["valid_both_correctness_classes"] for item in record["bins"]
    ]


def test_25_delta_cal_auc_formula_is_exact(split):
    ids = split["unlabeled_ids"]
    outcomes = (np.arange(1386) % 2).astype(np.int64)
    baseline = np.linspace(0.01, 0.99, 1386)
    calibrated = baseline[::-1]
    shuffled = np.roll(baseline, 7)
    confidence = np.linspace(0.0, 1.0, 1386)
    record = c2.analyze_calibration_direction(
        outcomes, baseline, calibrated, shuffled, confidence, ids, 0
    )
    assert record["DeltaCalAUC_dir"] == (
        record["CondAUC_U_tilde_dir"] - record["CondAUC_Ucycle"]
    )


def test_26_true_vs_shuffle_calibration_gap_formula_is_exact(split):
    ids = split["unlabeled_ids"]
    outcomes = (np.arange(1386) % 2).astype(np.int64)
    baseline = np.linspace(0.01, 0.99, 1386)
    calibrated = baseline[::-1]
    shuffled = np.roll(baseline, 7)
    confidence = np.linspace(0.0, 1.0, 1386)
    record = c2.analyze_calibration_direction(
        outcomes, baseline, calibrated, shuffled, confidence, ids, 0
    )
    assert record["TrueVsShuffleCalGap_dir"] == (
        record["CondAUC_U_tilde_dir"]
        - record["CondAUC_U_tilde_dir_shuffle"]
    )


def test_27_directional_sign_agreement_formula_is_exact(split):
    predicted = np.zeros((1386, 20), dtype=np.float64)
    shuffled = np.zeros((1386, 20), dtype=np.float64)
    truth = np.zeros((1386, 20), dtype=np.float64)
    predicted[:4, 0] = (1.0, -1.0, 1.0, 0.0)
    shuffled[:4, 0] = (-1.0, 1.0, -1.0, 0.0)
    truth[:4, 0] = (1.0, -1.0, -1.0, 1.0)
    record = c2.analyze_directional_sign_agreement(
        predicted, shuffled, truth, split["unlabeled_ids"]
    )["directions"][0]
    assert record["directional_sign_agreement_true"] == pytest.approx(2.0 / 3.0)
    assert record["directional_sign_agreement_shuffle"] == pytest.approx(1.0 / 3.0)
    assert record["DirectionalSpecificityGap"] == pytest.approx(1.0 / 3.0)


def test_28_directional_diagnostic_is_not_used_in_seed_gate():
    diagnostic, calibration = _direction_records(diagnostic_valid_count=0)
    summary = c2.build_seed_summary(20, diagnostic, calibration)
    assert summary["valid_directional_diagnostic_count"] == 0
    assert summary["DirectionalDiagnostic_used_in_gate"] is False
    assert summary["C2_C0_SEED_PASS"] is True


def test_29_seed_gate_matches_all_five_preregistered_conditions():
    diagnostic, calibration = _direction_records()
    summary = c2.build_seed_summary(20, diagnostic, calibration)
    assert summary["valid_calibration_direction_count"] == 20
    assert summary["positive_DeltaCalAUC_dir_direction_count"] == 20
    assert summary["mean_DeltaCalAUC_dir"] == pytest.approx(0.2)
    assert summary["calibration_wilcoxon_greater_p"] < 0.05
    assert summary["mean_TrueVsShuffleCalGap_dir"] == pytest.approx(0.1)
    assert all(summary["CalibrationGate_conditions"].values())
    assert summary["C2_C0_SEED_PASS"] is True


def test_30_seed_gate_requires_sixteen_valid_and_twelve_positive():
    diagnostic, calibration = _direction_records(calibration_valid_count=15)
    summary = c2.build_seed_summary(20, diagnostic, calibration)
    assert summary["CalibrationGate_conditions"][
        "minimum_valid_calibration_directions"
    ] is False
    assert summary["C2_C0_SEED_PASS"] is False


def test_31_seed_gate_rejects_nonpositive_true_vs_shuffle_gap():
    diagnostic, calibration = _direction_records(calibration_gap=-0.1)
    summary = c2.build_seed_summary(20, diagnostic, calibration)
    assert summary["CalibrationGate_conditions"][
        "positive_mean_TrueVsShuffleCalGap_dir"
    ] is False
    assert summary["C2_C0_SEED_PASS"] is False


def test_32_multiseed_gate_is_exact():
    summaries = OrderedDict()
    for seed, passed in zip(c2.SEEDS, (True, True, False)):
        summaries[seed] = {
            "seed": seed,
            "C2_C0_SEED_PASS": passed,
            "mean_DeltaCalAUC_dir": 0.02,
            "mean_TrueVsShuffleCalGap_dir": 0.01,
            "mean_DirectionalSpecificityGap": 0.03,
        }
    result = c2.build_multiseed_decision(summaries)
    assert result["summary"]["seed_pass_count"] == 2
    assert all(result["decision"]["decision_conditions"].values())
    assert result["decision"][
        "C2_C0_DIRECTIONAL_CONSENSUS_UTILITY_CALIBRATION_PASS"
    ] is True
    summaries[30]["mean_TrueVsShuffleCalGap_dir"] = -0.2
    result = c2.build_multiseed_decision(summaries)
    assert result["decision"][
        "C2_C0_DIRECTIONAL_CONSENSUS_UTILITY_CALIBRATION_PASS"
    ] is False


def test_33_only_fixed_formal_seed_identifiers_are_supported():
    assert tuple(c2.SEEDS) == (20, 30, 50)
    for seed in c2.SEEDS:
        assert "seed" + str(seed) in str(summarize.default_seed_dir(seed))
    assert str(summarize.default_multiseed_dir()).endswith(
        "c2_c0_directional_consensus_utility_multiseed"
    )
    with pytest.raises(ValueError):
        summarize.default_seed_dir(21)


def test_34_both_frozen_c2_b0_sha256_manifests_pass():
    audit = evaluate.verify_frozen_c2_b0_sources()
    assert audit["all_frozen_C2_B0_hashes_pass"] is True
    assert tuple(audit["manifests"]) == evaluate.FROZEN_C2_B0_MANIFESTS


def test_35_frozen_c2_b0_sources_are_unmodified():
    expected = {
        "experiments/cyclic_utility/c2_b0_sparse_utility_residual_protocol.py": (
            "efcb469973e317959648e024d9ab7728311e40569bd538c18ece36d6fdc83515"
        ),
        "experiments/cyclic_utility/evaluate_c2_b0_sparse_utility_residual.py": (
            "ab08a1b60701daa17d4ba4574e63a17fa2f9713f5af3e10f8d387c72a60a009c"
        ),
        "experiments/cyclic_utility/summarize_c2_b0_sparse_utility_residual.py": (
            "52f3ca2317cf6511dacf7cb27aa8b85dd01b29e76b09b90d78aec7311e4c7201"
        ),
        "tests/test_c2_b0_sparse_utility_residual.py": (
            "94f87335b5d506ebcbfdb36245ea0d5963052c91995cff160995bd7b921c5ab7"
        ),
    }
    assert all(_sha256(ROOT / path) == digest for path, digest in expected.items())


def test_36_exactly_four_c2_c0_python_files_exist():
    expected = {
        ROOT / "experiments/cyclic_utility/c2_c0_directional_consensus_utility_protocol.py",
        ROOT / "experiments/cyclic_utility/evaluate_c2_c0_directional_consensus_utility.py",
        ROOT / "experiments/cyclic_utility/summarize_c2_c0_directional_consensus_utility.py",
        ROOT / "tests/test_c2_c0_directional_consensus_utility.py",
    }
    discovered = set(ROOT.glob("experiments/cyclic_utility/*c2_c0*.py"))
    discovered.update(ROOT.glob("tests/*c2_c0*.py"))
    assert discovered == expected
