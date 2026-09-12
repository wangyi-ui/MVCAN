import ast
import hashlib
import inspect
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from experiments.cyclic_utility import c2_b0_sparse_utility_residual_protocol as c2
from experiments.cyclic_utility import evaluate_c2_b0_sparse_utility_residual as evaluate
from experiments.cyclic_utility import summarize_c2_b0_sparse_utility_residual as summarize
from weak_quality import ndarray_sha256


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def formal_arrays():
    rows = np.arange(c2.SAMPLE_NUM, dtype=np.int64)[:, None]
    directions = np.arange(c2.DIRECTION_COUNT, dtype=np.int64)[None, :]
    y_gen = (rows + 2 * directions) % c2.CLASS_NUM
    U_cycle = 0.05 + ((3 * rows + 7 * directions) % 90) / 100.0
    U_cycle = U_cycle.astype(np.float64)
    U_cycle[0, 0] = 0.0
    C_conf = (0.01 + ((11 * rows + directions) % 98) / 100.0).astype(np.float64)
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
    return c2.build_residual_calibration_outputs(
        formal_arrays["y_gen"], formal_arrays["U_cycle"],
        split["labeled_sample_ids"], split["labeled_targets"],
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
    preregistration = {"file_sha256": {"PROTOCOL.txt": "c" * 64}}
    return evaluate.build_pre_gt_bundle(
        20, formal_arrays, provenance, split, preregistration
    )


@pytest.fixture()
def persisted(tmp_path, formal_arrays, split):
    return evaluate.persist_residual_bundle(
        _bundle(formal_arrays, split), tmp_path / "sealed"
    )


def _direction_records(
    residual_value=0.2, residual_gap=0.1, delta=0.2, calibration_gap=0.1,
    residual_valid_count=20, calibration_valid_count=20,
):
    residual = []
    calibration = []
    for direction_id in range(c2.DIRECTION_COUNT):
        residual_valid = direction_id < residual_valid_count
        calibration_valid = direction_id < calibration_valid_count
        residual.append({
            "direction_id": direction_id,
            "joint_valid": residual_valid,
            "rho_true": residual_value if residual_valid else None,
            "ResidualSpecificityGap": residual_gap if residual_valid else None,
        })
        calibration.append({
            "direction_id": direction_id,
            "direction_valid": calibration_valid,
            "DeltaCalAUC": delta if calibration_valid else None,
            "TrueVsShuffleCalGap": calibration_gap if calibration_valid else None,
        })
    return {"directions": residual}, {"directions": calibration}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _called_names(function):
    tree = ast.parse(inspect.getsource(function))
    return {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }


def test_01_vote_state_shape_finite_nonnegative(formal_arrays):
    state = c2.build_vote_semantic_state(formal_arrays["y_gen"])
    assert state.shape == (1400, 7)
    assert np.isfinite(state).all()
    assert np.all(state >= 0.0)


def test_02_vote_state_rows_sum_exactly_one(formal_arrays):
    state = c2.build_vote_semantic_state(formal_arrays["y_gen"])
    assert np.allclose(state.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)


def test_03_leave_one_out_sample_is_excluded(formal_arrays, split, outputs):
    state = outputs["R"]
    masks = outputs["loo_included_label_mask"]
    labeled = split["labeled_sample_ids"]
    targets = split["labeled_targets"]
    assert not np.any(np.diag(masks))
    assert np.all(masks.sum(axis=1) == 13)
    held_out = 5
    manual = np.zeros((7, 7), dtype=np.float64)
    for native_id in range(7):
        for semantic_class in range(7):
            positions = np.flatnonzero(
                (targets == semantic_class) & masks[held_out]
            )
            manual[native_id, semantic_class] = state[
                labeled[positions], native_id
            ].sum()
    assert np.array_equal(outputs["loo_soft_contingency"][held_out], manual)


def test_04_every_loo_mapping_is_a_legal_permutation(outputs):
    assert outputs["loo_mappings"].shape == (14, 7)
    assert outputs["loo_mappings_shuffle"].shape == (14, 7)
    expected = np.arange(7)
    for mappings in (outputs["loo_mappings"], outputs["loo_mappings_shuffle"]):
        assert all(np.array_equal(np.sort(mapping), expected) for mapping in mappings)


def test_05_e_l_exactly_equals_z_l_minus_u_cycle(formal_arrays, split, outputs):
    expected = outputs["z_L"] - formal_arrays["U_cycle"][split["labeled_sample_ids"]]
    assert np.array_equal(outputs["e_L"], expected)


def test_06_e_l_is_finite_and_in_unit_residual_range(outputs):
    assert np.isfinite(outputs["e_L"]).all()
    assert np.all((-1.0 <= outputs["e_L"]) & (outputs["e_L"] <= 1.0))


def test_07_action_match_and_weight_shapes(outputs):
    assert outputs["semantic_similarity"].shape == (1400, 14)
    assert outputs["action_match"].shape == (1400, 14, 20)
    assert outputs["weights"].shape == (1400, 14, 20)
    assert outputs["action_match"].dtype == np.bool_


def test_08_zero_support_produces_exact_zero(formal_arrays, split):
    actions = formal_arrays["y_gen"].copy()
    labeled = split["labeled_sample_ids"]
    actions[labeled, 0] = 0
    actions[0, 0] = 6
    residual = np.full((14, 20), 0.75, dtype=np.float64)
    result = c2.propagate_action_context_residuals(
        c2.build_vote_semantic_state(formal_arrays["y_gen"]),
        actions, labeled, residual,
    )
    assert result["support"][0, 0] == 0.0
    assert result["e_hat"][0, 0] == 0.0


def test_09_e_hat_is_finite_and_bounded(outputs):
    for name in ("e_hat", "e_hat_shuffle"):
        assert outputs[name].shape == (1400, 20)
        assert np.isfinite(outputs[name]).all()
        assert np.all((-1.0 <= outputs[name]) & (outputs[name] <= 1.0))


def test_10_shuffle_path_is_rebuilt_from_frozen_targets(formal_arrays, split, outputs):
    rebuilt = c2.build_leave_one_out_action_correctness(
        outputs["R"], formal_arrays["y_gen"], split["labeled_sample_ids"],
        split["shuffled_labeled_targets"],
    )
    assert np.array_equal(outputs["z_L_shuffle"], rebuilt["z_L"])
    assert outputs["shuffle_recomputed_from_targets"] is True
    assert outputs["shuffle_only_final_residual"] is False


def test_11_true_and_shuffle_hashes_differ(outputs):
    pairs = (
        ("z_L", "z_L_shuffle"),
        ("e_L", "e_L_shuffle"),
        ("e_hat", "e_hat_shuffle"),
    )
    assert any(ndarray_sha256(outputs[a]) != ndarray_sha256(outputs[b]) for a, b in pairs)


def test_12_confidence_cannot_enter_residual_construction():
    for function in (
        c2.build_vote_semantic_state,
        c2.build_leave_one_out_action_correctness,
        c2.build_labeled_residual,
        c2.propagate_action_context_residuals,
        c2.build_residual_calibration_outputs,
    ):
        assert "C_conf" not in inspect.signature(function).parameters


def test_13_pre_gt_functions_cannot_load_gt_or_c0_audit():
    forbidden = {
        "load_full_ground_truth_after_residual_seal",
        "load_c0_audit_after_residual_seal",
        "build_bridge_correctness_after_residual_seal",
    }
    for function in (
        evaluate.load_c0_inputs_before_gt,
        evaluate.build_pre_gt_bundle,
        evaluate.persist_residual_bundle,
    ):
        assert _called_names(function).isdisjoint(forbidden)


def test_14_no_corruption_mask_or_oracle_api_exists():
    protocol_names = set(vars(c2))
    evaluator_names = set(vars(evaluate))
    assert "load_corruption_mask" not in protocol_names | evaluator_names
    assert "load_oracle" not in protocol_names | evaluator_names


def test_15_no_training_backward_optimizer_or_model_forward_calls():
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
    assert called.isdisjoint({"backward", "optimizer", "step", "forward", "train"})


def test_16_seal_reloads_and_logically_verifies_all_arrays(persisted):
    seal = persisted["seal"]
    assert seal["NPZ_saved_fsynced_reloaded_hash_verified"] is True
    assert tuple(seal["arrays"]) == evaluate.PRE_GT_ARRAY_NAMES
    with np.load(persisted["npz_path"], allow_pickle=False) as archive:
        for name in evaluate.PRE_GT_ARRAY_NAMES:
            assert ndarray_sha256(archive[name]) == seal["arrays"][name]["logical_sha256"]


def test_17_seal_records_every_hard_false_boundary(persisted):
    seal = persisted["seal"]
    keys = (
        "GT_loaded_before_residual_seal",
        "c0_audit_loaded_before_residual_seal",
        "Bridge_correctness_loaded_before_residual_seal",
        "corruption_mask_loaded",
        "training_performed",
        "backward_called",
        "optimizer_created",
        "optimizer_step_called",
        "model_forward_called",
        "C_conf_used_for_residual_propagation",
    )
    assert all(seal[key] is False for key in keys)


def test_18_evaluation_accepts_only_fixed_1386_unlabeled_samples(split):
    shape = (1386, 20)
    rows = np.arange(1386, dtype=np.float64)[:, None]
    directions = np.arange(20, dtype=np.float64)[None, :]
    predicted = rows + directions
    shuffled = rows[::-1] + 2.0 * directions
    truth = rows + 3.0 * directions
    result = c2.analyze_residual_directions(
        predicted, shuffled, truth, split["unlabeled_ids"]
    )
    assert result["evaluation_sample_count"] == 1386
    with pytest.raises(RuntimeError, match="1386 unlabeled"):
        c2.analyze_residual_directions(
            np.zeros((1400, 20)), np.zeros((1400, 20)),
            np.zeros((1400, 20)), np.arange(1400),
        )


def test_19_joint_valid_and_invalid_spearman_handling(split):
    rng = np.random.default_rng(4)
    truth = rng.normal(size=(1386, 20))
    predicted = truth + rng.normal(scale=0.01, size=(1386, 20))
    shuffled = rng.normal(size=(1386, 20))
    predicted[:, 1] = 1.0
    shuffled[0, 2] = np.nan
    result = c2.analyze_residual_directions(
        predicted, shuffled, truth, split["unlabeled_ids"]
    )
    assert result["directions"][0]["joint_valid"] is True
    assert result["directions"][1]["joint_valid"] is False
    assert result["directions"][1]["rho_true"] is None
    assert result["directions"][2]["joint_valid"] is False
    assert result["directions"][2]["rho_shuffle"] is None


def test_20_bounded_update_is_finite_and_in_zero_one(formal_arrays, outputs):
    calibrated = c2.bounded_utility_update(
        formal_arrays["U_cycle"], outputs["e_hat"]
    )
    assert np.isfinite(calibrated).all()
    assert np.all((0.0 <= calibrated) & (calibrated <= 1.0))


def test_21_zero_utility_is_never_revived(formal_arrays):
    residual = np.ones((1400, 20), dtype=np.float64)
    calibrated = c2.bounded_utility_update(formal_arrays["U_cycle"], residual)
    assert calibrated[0, 0] == 0.0


def test_22_capacity_is_not_part_of_propagation_weights():
    signature = inspect.signature(c2.propagate_action_context_residuals)
    source = inspect.getsource(c2.propagate_action_context_residuals)
    assert "U_cycle" not in signature.parameters
    assert "1.0 - cycle" not in source


def test_23_bridge_confidence_stratification_is_reused(split):
    ids = split["unlabeled_ids"]
    confidence = np.ones(1386, dtype=np.float64)
    assignments, audit = c2.confidence_equal_count_strata(confidence, ids)
    assert audit["sample_id_only_tie_break"] is True
    assert audit["GT_used_for_assignment"] is False
    assert max(np.bincount(assignments)) - min(np.bincount(assignments)) <= 1


def test_24_seed_gate_matches_all_frozen_conditions():
    residual, calibration = _direction_records()
    summary = c2.build_seed_summary(20, residual, calibration)
    assert summary["valid_residual_direction_count"] == 20
    assert summary["positive_rho_true_direction_count"] == 20
    assert summary["valid_calibration_direction_count"] == 20
    assert summary["positive_DeltaCalAUC_direction_count"] == 20
    assert summary["ResidualGate"] is True
    assert summary["CalibrationGate"] is True
    assert summary["C2_B0_SEED_PASS"] is True


def test_25_seed_gate_requires_sixteen_valid_and_twelve_positive():
    residual, calibration = _direction_records(
        residual_valid_count=15, calibration_valid_count=15
    )
    summary = c2.build_seed_summary(20, residual, calibration)
    assert summary["ResidualGate"] is False
    assert summary["CalibrationGate"] is False


def test_26_wilcoxon_is_one_sided_greater():
    source = inspect.getsource(c2.one_sided_wilcoxon_greater)
    assert 'alternative="greater"' in source
    assert c2.one_sided_wilcoxon_greater(np.ones(20)) < 0.05


def test_27_multiseed_requires_two_of_three_and_all_aggregate_gaps():
    summaries = OrderedDict()
    for seed, passed in zip(c2.SEEDS, (True, True, False)):
        summaries[seed] = {
            "seed": seed,
            "C2_B0_SEED_PASS": passed,
            "mean_rho_true": 0.1,
            "mean_ResidualSpecificityGap": 0.05,
            "mean_DeltaCalAUC": 0.02,
            "mean_TrueVsShuffleCalGap": 0.01,
        }
    result = c2.build_multiseed_decision(summaries)
    assert result["summary"]["seed_pass_count"] == 2
    assert result["decision"]["C2_B0_SPARSE_UTILITY_RESIDUAL_CALIBRATION_PASS"] is True
    summaries[30]["mean_TrueVsShuffleCalGap"] = -0.2
    result = c2.build_multiseed_decision(summaries)
    assert result["decision"]["C2_B0_SPARSE_UTILITY_RESIDUAL_CALIBRATION_PASS"] is False


def test_28_summarizer_accepts_only_fixed_seed_identifiers():
    assert tuple(c2.SEEDS) == (20, 30, 50)
    for seed in c2.SEEDS:
        assert "seed" + str(seed) in str(summarize.default_seed_dir(seed))
    with pytest.raises(ValueError):
        summarize.default_seed_dir(21)


def test_29_preregistration_and_frozen_source_hashes_verify():
    audit = evaluate.verify_preregistration_freeze()
    assert audit["C2_A0_frozen_source_verified"] is True
    assert audit["manifests"]["input_source_sha256.txt"]["all_entries_exact_match"] is True


def test_30_c2_a0_frozen_files_are_unmodified():
    expected = {
        "experiments/cyclic_utility/c2_a0_sparse_label_utility_protocol.py": (
            "ee40a7687cbd41ade5c7ba850d3af92b813321c2f76a73ec115eeb7472fd7db7"
        ),
        "experiments/cyclic_utility/evaluate_c2_a0_sparse_label_utility.py": (
            "91325a0dbfdfe575d551fa9967269e11565176b1bcecc341bd45845e7555a4b2"
        ),
    }
    assert all(_sha256(ROOT / path) == digest for path, digest in expected.items())


def test_31_pre_gt_npz_contains_every_preregistered_residual_array(persisted):
    required = {
        "R", "z_L", "e_L", "e_hat", "z_L_shuffle", "e_L_shuffle",
        "e_hat_shuffle", "U_cycle", "C_conf", "y_gen", "sample_ids",
        "labeled_sample_ids", "labeled_targets", "unlabeled_ids",
    }
    with np.load(persisted["npz_path"], allow_pickle=False) as archive:
        assert required.issubset(archive.files)


def test_32_only_four_c2_b0_implementation_files_exist():
    expected = {
        ROOT / "experiments/cyclic_utility/c2_b0_sparse_utility_residual_protocol.py",
        ROOT / "experiments/cyclic_utility/evaluate_c2_b0_sparse_utility_residual.py",
        ROOT / "experiments/cyclic_utility/summarize_c2_b0_sparse_utility_residual.py",
        ROOT / "tests/test_c2_b0_sparse_utility_residual.py",
    }
    discovered = set(ROOT.glob("experiments/cyclic_utility/*c2_b0*.py"))
    discovered.update(ROOT.glob("tests/*c2_b0*.py"))
    assert discovered == expected
