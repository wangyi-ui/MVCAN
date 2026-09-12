import ast
import inspect
import itertools
import json
from collections import OrderedDict

import numpy as np
import pytest

from experiments.cyclic_utility import (
    c2_a0_sparse_label_utility_protocol as c2,
)
from experiments.cyclic_utility import (
    evaluate_c2_a0_sparse_label_utility as evaluate,
)
from experiments.cyclic_utility import (
    summarize_c2_a0_sparse_label_utility as summarize,
)
from irv.b4_information_utility import tensor_sha256
from weak_quality import ndarray_sha256


def _direction_subsets():
    generator = np.asarray(list(itertools.combinations(range(6), 3)), dtype=np.int64)
    verifier = np.asarray(
        [[view for view in range(6) if view not in row] for row in generator],
        dtype=np.int64,
    )
    return generator, verifier


def _synthetic_formal_arrays():
    sample_ids = c2.canonical_sample_ids()
    row = np.arange(c2.SAMPLE_NUM, dtype=np.int64)[:, None]
    direction = np.arange(c2.DIRECTION_COUNT, dtype=np.int64)[None, :]
    y_gen = np.ascontiguousarray((row + direction) % c2.CLASS_NUM, dtype=np.int64)
    U_cycle = np.ascontiguousarray(
        0.15 + 0.7 * ((row * 3 + direction * 5) % 101) / 100.0,
        dtype=np.float32,
    )
    C_conf = np.ascontiguousarray(
        0.1 + 0.8 * ((row * 7 + direction * 11) % 103) / 102.0,
        dtype=np.float32,
    )
    generator, verifier = _direction_subsets()
    return {
        "sample_ids": sample_ids,
        "U_cycle": U_cycle,
        "C_conf": C_conf,
        "y_gen": y_gen,
        "generator_subsets": generator,
        "verifier_subsets": verifier,
    }


def _synthetic_split():
    labeled = c2.fixed_labeled_ids()
    targets = c2.fixed_labeled_targets()
    unlabeled = np.setdiff1d(c2.canonical_sample_ids(), labeled)
    return {
        **c2.validate_fixed_sparse_split(labeled, unlabeled, targets),
        "shuffled_labeled_targets": c2.validate_negative_control(
            targets, c2.fixed_shuffled_targets()
        ),
        "label_split_sha256": "a" * 64,
    }


@pytest.fixture(scope="module")
def formal_arrays():
    return _synthetic_formal_arrays()


@pytest.fixture(scope="module")
def sparse_split():
    return _synthetic_split()


@pytest.fixture(scope="module")
def calibration(formal_arrays, sparse_split):
    return c2.build_calibration_outputs(
        formal_arrays["y_gen"],
        formal_arrays["U_cycle"],
        sparse_split["labeled_sample_ids"],
        sparse_split["labeled_targets"],
        sparse_split["shuffled_labeled_targets"],
    )


def _subscript_string_keys(module):
    tree = ast.parse(inspect.getsource(module))
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            slice_node = node.slice
            if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                keys.add(slice_node.value)
    return keys


def _called_attribute_names(module):
    tree = ast.parse(inspect.getsource(module))
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def _directional_records(deltas, specificity=0.1, scg=0.1, scg_gap=0.1):
    records = []
    for direction_id, delta in enumerate(deltas):
        records.append({
            "direction_id": direction_id,
            "direction_valid": True,
            "DeltaCalAUC": float(delta),
            "LabelSpecificityGap": float(specificity),
            "SCG": float(scg),
            "SCGGap": float(scg_gap),
            "CondAUC_E_label": 0.6,
        })
    return {"directions": records}


def _seed_summary(seed, passed, metric=0.1):
    return {
        "seed": seed,
        "C2_A0_SEED_PASS": passed,
        "mean_DeltaCalAUC": metric,
        "mean_LabelSpecificityGap": metric,
        "mean_SCG": metric,
        "mean_SCGGap": metric,
    }


def test_01_formal_c2_does_not_read_p_gen():
    assert c2.FORMAL_C2_USES_PGEN is False
    assert "p_gen" not in _subscript_string_keys(evaluate)


def test_02_formal_c2_does_not_read_recovered_seed20_p_gen():
    paths = evaluate.default_paths(20)
    assert "c0_complementary_semantic_verification_seed20" in str(paths.c0_artifact_path)
    assert "c2_a0_pre_gt_input_seed" not in inspect.getsource(evaluate.default_paths)
    assert "materialize_c2_a0_pre_gt_inputs" not in inspect.getsource(evaluate)


def test_03_full_gt_loader_is_post_durable_seal_only():
    run_source = inspect.getsource(evaluate.run_evaluation)
    post_source = inspect.getsource(evaluate.evaluate_after_calibration_seal)
    assert run_source.index("persist_calibration_bundle") < run_source.index(
        "evaluate_after_calibration_seal"
    )
    assert post_source.index("durable_calibration_seal_verified") < post_source.index(
        "load_bridge_full_ground_truth_after_seal"
    )


def test_04_c0_audit_is_not_loaded_before_calibration_seal():
    pre_source = inspect.getsource(evaluate.load_c0_inputs_before_gt)
    post_source = inspect.getsource(evaluate.evaluate_after_calibration_seal)
    assert "load_c0_audit_after_calibration_seal" not in pre_source
    assert post_source.index("load_bridge_full_ground_truth_after_seal") < post_source.index(
        "load_c0_audit_after_calibration_seal"
    )


def test_05_corruption_mask_is_never_read():
    assert "corruption_mask" not in _subscript_string_keys(evaluate)
    assert "load_oracle_corruption_mask" not in inspect.getsource(evaluate)


def test_06_no_model_forward_call_exists():
    modules = (c2, evaluate, summarize)
    assert all("forward" not in _called_attribute_names(module) for module in modules)


def test_07_no_backward_call_exists():
    modules = (c2, evaluate, summarize)
    assert all("backward" not in _called_attribute_names(module) for module in modules)


def test_08_no_optimizer_or_optimizer_step_exists():
    modules = (c2, evaluate, summarize)
    assert all("step" not in _called_attribute_names(module) for module in modules)
    assert all("torch.optim" not in inspect.getsource(module) for module in modules)
    assert c2.TRAINING_PATH_PRESENT is False


def test_09_fixed_fourteen_ids_are_exact_and_loaded_from_b7():
    split = evaluate.load_fixed_sparse_labels()
    assert split["labeled_sample_ids"].tolist() == list(c2.FIXED_LABELED_IDS)
    assert split["targets_constructed_from_per_class_labeled_ids"] is True


def test_10_fixed_fourteen_targets_are_exact():
    assert c2.fixed_labeled_targets().tolist() == [
        4, 6, 2, 3, 0, 4, 1, 5, 6, 3, 0, 2, 5, 1,
    ]


def test_11_every_sparse_class_has_exactly_two_labels():
    assert np.array_equal(
        np.bincount(c2.fixed_labeled_targets(), minlength=c2.CLASS_NUM),
        np.full(c2.CLASS_NUM, 2),
    )


def test_12_unlabeled_partition_has_1386_samples(sparse_split):
    assert sparse_split["unlabeled_ids"].shape == (1386,)


def test_13_vote_state_shape_is_1400_by_7(formal_arrays):
    vote_state = c2.build_cross_direction_vote_state(formal_arrays["y_gen"])
    assert vote_state.shape == (1400, 7)


def test_14_vote_state_rows_sum_to_one(formal_arrays):
    vote_state = c2.build_cross_direction_vote_state(formal_arrays["y_gen"])
    assert np.isfinite(vote_state).all()
    assert np.all(vote_state >= 0.0)
    assert np.allclose(vote_state.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)


def test_15_sparse_mapping_is_a_permutation(calibration):
    assert np.array_equal(
        np.sort(calibration["sparse_mapping"]), np.arange(c2.CLASS_NUM)
    )


def test_16_shuffled_targets_preserve_two_per_class():
    shuffled = c2.fixed_shuffled_targets()
    assert np.array_equal(np.bincount(shuffled, minlength=7), np.full(7, 2))


def test_17_shuffle_is_not_a_global_class_permutation():
    assert not c2.is_global_class_permutation(
        c2.fixed_labeled_targets(), c2.fixed_shuffled_targets()
    )


def test_18_c_conf_changes_cannot_change_evidence_or_calibration(
    formal_arrays, sparse_split
):
    first = dict(formal_arrays)
    second = dict(formal_arrays)
    first["C_conf"] = np.zeros((1400, 20), dtype=np.float32)
    second["C_conf"] = np.ones((1400, 20), dtype=np.float32)
    provenance = {"C0_artifact_file_sha256": "a" * 64, "C0_prediction_seal_file_sha256": "b" * 64}
    preregistration = {"file_sha256": {}}
    left = evaluate.build_pre_gt_bundle(20, first, provenance, sparse_split, preregistration)
    right = evaluate.build_pre_gt_bundle(20, second, provenance, sparse_split, preregistration)
    assert np.array_equal(left["arrays"]["E_label"], right["arrays"]["E_label"])
    assert np.array_equal(left["arrays"]["U_tilde"], right["arrays"]["U_tilde"])


def _calibration_case(evidence_value, utility_value=0.4):
    utility = np.full((1400, 20), utility_value, dtype=np.float64)
    evidence = np.full((1400, 20), evidence_value, dtype=np.float64)
    return utility, c2.calibrate_utility(utility, evidence)


def test_19_neutral_evidence_keeps_utility_exactly_unchanged():
    utility, calibrated = _calibration_case(0.5)
    assert np.array_equal(calibrated, utility)


def test_20_supportive_evidence_increases_positive_utility():
    utility, calibrated = _calibration_case(0.75)
    assert np.all(calibrated > utility)


def test_21_adverse_evidence_decreases_positive_utility():
    utility, calibrated = _calibration_case(0.25)
    assert np.all(calibrated < utility)


def test_22_zero_cycle_utility_is_never_revived():
    utility, calibrated = _calibration_case(0.9, utility_value=0.0)
    assert np.array_equal(calibrated, utility)


def test_23_calibrated_utility_is_finite_and_bounded(calibration):
    for name in ("U_tilde", "U_tilde_shuffle"):
        value = calibration[name]
        assert np.isfinite(value).all()
        assert np.all((value >= 0.0) & (value <= 1.0))


def test_24_failure_before_durable_seal_cannot_reach_gt(monkeypatch, tmp_path):
    reached_postseal = {"value": False}
    monkeypatch.setattr(evaluate, "verify_preregistration_freeze", lambda: {})
    monkeypatch.setattr(evaluate, "_require_frozen_c0_manifest_members", lambda *args: None)
    monkeypatch.setattr(evaluate, "load_c0_inputs_before_gt", lambda *args: ({}, {}))
    monkeypatch.setattr(evaluate, "load_fixed_sparse_labels", lambda *args: {})
    monkeypatch.setattr(evaluate, "build_pre_gt_bundle", lambda *args: {"seed": 20})
    monkeypatch.setattr(
        evaluate,
        "persist_calibration_bundle",
        lambda *args: (_ for _ in ()).throw(RuntimeError("seal failure")),
    )

    def forbidden_postseal(*args):
        reached_postseal["value"] = True

    monkeypatch.setattr(evaluate, "evaluate_after_calibration_seal", forbidden_postseal)
    with pytest.raises(RuntimeError, match="seal failure"):
        evaluate.run_evaluation(20, output_dir=tmp_path / "never-created")
    assert reached_postseal["value"] is False


def test_25_evaluation_accepts_only_1386_unlabeled_samples(monkeypatch):
    ids = np.setdiff1d(c2.canonical_sample_ids(), c2.fixed_labeled_ids())
    shape = (c2.UNLABELED_EVAL_COUNT, c2.DIRECTION_COUNT)
    values = np.zeros(shape, dtype=np.float64)
    correct = np.zeros(shape, dtype=np.int64)
    monkeypatch.setattr(
        c2,
        "analyze_direction",
        lambda *args: {"direction_id": int(args[-1]), "direction_valid": False},
    )
    result = c2.analyze_all_directions(correct, values, values, values, values, values, ids)
    assert result["evaluation_sample_count"] == 1386
    with pytest.raises(RuntimeError, match="unlabeled-only"):
        c2.analyze_all_directions(
            np.zeros((1400, 20), dtype=np.int64),
            np.zeros((1400, 20)),
            np.zeros((1400, 20)),
            np.zeros((1400, 20)),
            np.zeros((1400, 20)),
            np.zeros((1400, 20)),
            c2.canonical_sample_ids(),
        )


def test_26_labeled_and_evaluation_ids_are_disjoint(sparse_split):
    assert np.intersect1d(
        sparse_split["labeled_sample_ids"], sparse_split["unlabeled_ids"]
    ).size == 0


def test_27_confidence_sorting_is_stable_with_sample_id_tie_break():
    ids = np.setdiff1d(c2.canonical_sample_ids(), c2.fixed_labeled_ids())[::-1]
    confidence = np.zeros(ids.size, dtype=np.float64)
    assignments, audit = c2.confidence_equal_count_strata(confidence, ids)
    expected_first = np.sort(ids)[: audit["bin_sizes"][0]]
    assert np.array_equal(np.sort(ids[assignments == 0]), expected_first)
    assert audit["sample_id_only_tie_break"] is True


def test_28_valid_bin_and_valid_direction_gate():
    ids = np.setdiff1d(c2.canonical_sample_ids(), c2.fixed_labeled_ids())
    confidence = np.linspace(0.0, 1.0, ids.size)
    outcomes = np.arange(ids.size) % 2
    cycle = np.linspace(0.1, 0.8, ids.size)
    calibrated = np.clip(cycle + 0.02 * (2 * outcomes - 1), 0.0, 1.0)
    shuffled = cycle.copy()
    evidence = calibrated.copy()
    valid = c2.analyze_direction(
        outcomes, cycle, calibrated, shuffled, evidence, confidence, ids, 0
    )
    invalid = c2.analyze_direction(
        np.zeros_like(outcomes), cycle, calibrated, shuffled, evidence, confidence, ids, 0
    )
    assert valid["valid_bin_count"] == 5 and valid["direction_valid"] is True
    assert invalid["valid_bin_count"] == 0 and invalid["direction_valid"] is False


def test_29_twelve_positive_directions_gate_is_fixed():
    metrics = _directional_records([0.1] * 11 + [-0.01] * 9)
    summary = c2.build_seed_summary(20, metrics)
    assert summary["positive_DeltaCalAUC_direction_count"] == 11
    assert summary["seed_gate_conditions"]["minimum_positive_DeltaCalAUC_directions"] is False
    assert summary["C2_A0_SEED_PASS"] is False


def test_30_one_sided_wilcoxon_greater_gate_is_fixed():
    summary = c2.build_seed_summary(20, _directional_records([0.1] * 20))
    assert summary["wilcoxon_DeltaCalAUC_greater_p"] < 0.05
    assert summary["seed_gate_conditions"]["one_sided_wilcoxon_greater"] is True
    assert summary["wilcoxon_unit"] == "direction-wise paired consistency test"


def test_31_negative_control_specificity_and_scg_gap_are_primary_gates():
    summary = c2.build_seed_summary(
        20, _directional_records([0.1] * 20, specificity=-0.01, scg=0.1, scg_gap=-0.01)
    )
    assert summary["seed_gate_conditions"]["positive_mean_LabelSpecificityGap"] is False
    assert summary["seed_gate_conditions"]["positive_mean_SCGGap"] is False
    assert summary["C2_A0_SEED_PASS"] is False


def test_32_multiseed_two_of_three_and_aggregate_gate():
    passing = OrderedDict((
        (20, _seed_summary(20, True)),
        (30, _seed_summary(30, True)),
        (50, _seed_summary(50, False)),
    ))
    result = c2.build_multiseed_decision(passing)
    assert result["decision"]["C2_A0_SPARSE_LABEL_UTILITY_CALIBRATION_PASS"] is True
    failing = OrderedDict((
        (20, _seed_summary(20, True)),
        (30, _seed_summary(30, False)),
        (50, _seed_summary(50, False)),
    ))
    result = c2.build_multiseed_decision(failing)
    assert result["decision"]["C2_A0_SPARSE_LABEL_UTILITY_CALIBRATION_PASS"] is False


def test_33_pre_gt_npz_and_seal_are_durable_and_complete(
    tmp_path, formal_arrays, sparse_split
):
    provenance = {
        "C0_artifact_file_sha256": "a" * 64,
        "C0_prediction_seal_file_sha256": "b" * 64,
    }
    preregistration = {"file_sha256": {"PROTOCOL.txt": "c" * 64}}
    bundle = evaluate.build_pre_gt_bundle(
        20, formal_arrays, provenance, sparse_split, preregistration
    )
    persisted = evaluate.persist_calibration_bundle(bundle, tmp_path / "sealed")
    seal = persisted["seal"]
    assert persisted["durable_calibration_seal_verified"] is True
    assert seal["GT_loaded_before_calibration_seal"] is False
    assert seal["c0_audit_loaded_before_calibration_seal"] is False
    assert seal["Bridge_correctness_loaded_before_calibration_seal"] is False
    assert seal["sample_ids_logical_SHA256"] == c2.EXPECTED_SAMPLE_IDS_LOGICAL_SHA256
    with np.load(persisted["npz_path"], allow_pickle=False) as archive:
        assert tuple(archive.files) == evaluate.PRE_GT_ARRAY_NAMES


def test_34_c0_loader_ignores_nonformal_arrays_and_checks_conf_gen(tmp_path):
    arrays = _synthetic_formal_arrays()
    artifact_path = tmp_path / "c0_predictions_and_scores.npz"
    np.savez(
        artifact_path,
        **{name: arrays[name] for name in evaluate.FORMAL_INPUT_NAMES},
        conf_gen=arrays["C_conf"],
        p_gen=np.zeros((1,), dtype=np.float32),
    )
    records = {}
    for name in evaluate.FORMAL_INPUT_NAMES + ("conf_gen",):
        value = arrays["C_conf"] if name == "conf_gen" else arrays[name]
        records[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "logical_sha256": tensor_sha256(value),
        }
    seal = {
        "stage": "C0",
        "artifact_path": str(artifact_path),
        "artifact_file_sha256": evaluate.file_sha256(artifact_path),
        "arrays": records,
        "scores_completed_before_GT": True,
        "scores_saved_before_GT": True,
        "scores_hashed_before_GT": True,
        "scores_reloaded_before_GT": True,
        "full_GT_loaded_before_seal": False,
        "R_loaded_before_seal": False,
        "sparse_labels_loaded_before_seal": False,
        "corruption_mask_loaded_before_seal": False,
        "oracle_loaded_before_seal": False,
    }
    seal_path = tmp_path / "c0_prediction_seal.json"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    loaded, provenance = evaluate.load_c0_inputs_before_gt(20, artifact_path, seal_path)
    assert set(loaded) == set(evaluate.FORMAL_INPUT_NAMES) | {"sample_ids"}
    assert provenance["p_gen_loaded"] is False
    assert provenance["conf_gen_present_and_exact_C_conf"] is True


def test_35_expected_sample_id_logical_hash_uses_ndarray_sha256():
    sample_ids = np.arange(1400, dtype=np.int64)
    assert ndarray_sha256(sample_ids) == c2.EXPECTED_SAMPLE_IDS_LOGICAL_SHA256
