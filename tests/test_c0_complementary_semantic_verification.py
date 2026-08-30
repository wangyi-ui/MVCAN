"""Tests for C0 complementary-view reciprocal semantic verification."""

import hashlib
import inspect
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.cyclic_utility import (
    c0_complementary_semantic_verification as c0,
)
from experiments.cyclic_utility import (
    evaluate_c0_complementary_semantic_verification as evaluator,
)
from irv.b4_information_utility import tensor_sha256


@pytest.fixture(scope="module")
def q_aligned():
    generator = torch.Generator().manual_seed(20260827)
    values = torch.rand(
        c0.SAMPLE_NUM,
        c0.VIEW_NUM,
        c0.CLASS_NUM,
        generator=generator,
        dtype=torch.float64,
    )
    values = values / values.sum(dim=-1, keepdim=True)
    for view_id in range(c0.VIEW_NUM):
        for class_id in range(c0.CLASS_NUM):
            values[class_id, view_id] = 0.0
            values[class_id, view_id, class_id] = 1.0
    return values.detach()


@pytest.fixture(scope="module")
def cycle_fixed(q_aligned):
    return c0.build_cycle_scores(q_aligned)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping_inputs():
    labels = np.arange(c0.SAMPLE_NUM, dtype=np.int64) % c0.CLASS_NUM
    mapping = np.array([3, 6, 1, 4, 0, 5, 2], dtype=np.int64)
    clusters = np.argsort(mapping)[labels]
    return labels, clusters, mapping


def _correct_matrix():
    rows = np.arange(c0.SAMPLE_NUM)[:, None]
    directions = np.arange(c0.DIRECTION_COUNT)[None, :]
    return ((rows + directions) % 3 != 0)


def _score_arrays_for_auc():
    correct = _correct_matrix()
    direction_offset = (
        np.arange(c0.DIRECTION_COUNT, dtype=np.float64)[None, :]
        * 1e-5
    )
    cycle = correct.astype(np.float64) + direction_offset
    return correct, {
        "cycle": cycle,
        "confidence": 0.8 * cycle,
        "jsd": 0.7 * cycle,
        "shuffle": 0.6 * cycle,
    }


def _gate_inputs(
    signal=True,
    specificity=True,
    beats_confidence=True,
    beats_jsd=True,
):
    cycle_mean = 0.6 if signal else 0.4
    auc_result = {
        "valid_direction_count": 20 if signal else 19,
        "summary": {
            "cycle": {
                "mean": cycle_mean,
                "median": cycle_mean,
            },
            "shuffle": {
                "mean": (
                    cycle_mean - 0.1
                    if specificity
                    else cycle_mean + 0.1
                )
            },
            "confidence": {
                "mean": (
                    cycle_mean - 0.1
                    if beats_confidence
                    else cycle_mean + 0.1
                )
            },
            "jsd": {
                "mean": (
                    cycle_mean - 0.1
                    if beats_jsd
                    else cycle_mean + 0.1
                )
            },
        },
    }
    paired = {
        "p_cycle_vs_shuffle": 0.01 if specificity else 0.1,
        "p_cycle_vs_conf": 0.01 if beats_confidence else 0.1,
        "p_cycle_vs_jsd": 0.01 if beats_jsd else 0.1,
    }
    separation = {
        "mean_U_correct": 0.7 if signal else 0.3,
        "mean_U_wrong": 0.3 if signal else 0.7,
    }
    return auc_result, paired, separation


def test_frozen_dimensions_are_exact():
    assert (c0.SAMPLE_NUM, c0.VIEW_NUM, c0.CLASS_NUM) == (
        1400,
        6,
        7,
    )
    assert c0.DIRECTION_COUNT == 20
    assert c0.SUBSET_SIZE == 3


def test_validate_q_aligned_shape_finite_mass_and_detached(q_aligned):
    frozen = c0.validate_q_aligned(
        q_aligned.clone().requires_grad_(True)
    )

    assert frozen.shape == (1400, 6, 7)
    assert torch.isfinite(frozen).all()
    torch.testing.assert_close(
        frozen.sum(dim=-1),
        torch.ones(1400, 6, dtype=frozen.dtype),
        rtol=0.0,
        atol=c0.PROBABILITY_ATOL,
    )
    assert not frozen.requires_grad and frozen.grad_fn is None


def test_numpy_to_torch_copies_read_only_input(q_aligned):
    array = q_aligned.numpy().copy()
    array.setflags(write=False)

    frozen = c0.validate_q_aligned(array)

    assert frozen.shape == (1400, 6, 7)
    assert frozen.data_ptr() != torch.as_tensor(
        np.array(array, copy=True)
    ).data_ptr()


def test_non_probability_q_hard_fails(q_aligned):
    invalid = q_aligned.clone()
    invalid[10, 2, 0] += 0.1

    with pytest.raises(RuntimeError, match="probability mass"):
        c0.validate_q_aligned(invalid)


def test_missing_aligned_cluster_hard_fails():
    values = torch.zeros(1400, 6, 7, dtype=torch.float64)
    values[..., 0] = 1.0

    with pytest.raises(RuntimeError, match="missing a cluster"):
        c0.validate_q_aligned(values)


def test_exactly_twenty_generator_and_verifier_subsets():
    generator, verifier = c0.enumerate_complementary_splits()

    assert generator.shape == verifier.shape == (20, 3)
    assert np.unique(generator, axis=0).shape[0] == 20


@pytest.mark.parametrize("direction_id", range(c0.DIRECTION_COUNT))
def test_each_direction_is_exact_complement(direction_id):
    generator, verifier = c0.enumerate_complementary_splits()
    group = set(generator[direction_id].tolist())
    complement = set(verifier[direction_id].tolist())

    assert len(group) == len(complement) == 3
    assert group.isdisjoint(complement)
    assert group | complement == set(range(6))


def test_all_size_three_generators_appear_once():
    generator, _ = c0.enumerate_complementary_splits()
    expected = np.asarray(
        list(itertools.combinations(range(6), 3)),
        dtype=np.int64,
    )

    assert np.array_equal(generator, expected)


def test_complement_directions_are_both_retained():
    generator, verifier = c0.enumerate_complementary_splits()
    first = np.flatnonzero(np.all(generator == [0, 1, 2], axis=1))
    reverse = np.flatnonzero(np.all(generator == [3, 4, 5], axis=1))

    assert first.size == reverse.size == 1
    assert np.array_equal(verifier[first[0]], [3, 4, 5])
    assert np.array_equal(verifier[reverse[0]], [0, 1, 2])


def test_p_gen_and_p_ver_shapes_mass_and_stop_gradient(cycle_fixed):
    p_gen = cycle_fixed["p_gen"]
    p_ver = cycle_fixed["p_ver"]

    assert p_gen.shape == p_ver.shape == (1400, 20, 7)
    for posterior in (p_gen, p_ver):
        assert torch.isfinite(posterior).all()
        torch.testing.assert_close(
            posterior.sum(dim=-1),
            torch.ones(1400, 20, dtype=posterior.dtype),
            rtol=0.0,
            atol=c0.PROBABILITY_ATOL,
        )
        assert not posterior.requires_grad and posterior.grad_fn is None


def test_posteriors_are_exact_subset_means(q_aligned, cycle_fixed):
    generator = cycle_fixed["generator_subsets"]
    verifier = cycle_fixed["verifier_subsets"]
    direction_id = 7

    expected_gen = q_aligned[:, generator[direction_id]].mean(dim=1)
    expected_ver = q_aligned[:, verifier[direction_id]].mean(dim=1)

    assert torch.equal(cycle_fixed["p_gen"][:, direction_id], expected_gen)
    assert torch.equal(cycle_fixed["p_ver"][:, direction_id], expected_ver)


def test_no_second_softmax_or_probability_transform(cycle_fixed):
    assert cycle_fixed["audit"]["second_softmax_used"] is False
    source = inspect.getsource(
        c0.build_complementary_posteriors
    ).split("return {", 1)[0]

    assert "softmax" not in source
    assert "sigmoid" not in source


@pytest.mark.parametrize(
    "name,shape",
    [
        ("y_gen", (1400, 20)),
        ("conf_gen", (1400, 20)),
        ("y_ver", (1400, 20)),
        ("support_ver", (1400, 20)),
        ("closure", (1400, 20)),
        ("U_cycle", (1400, 20)),
        ("C_conf", (1400, 20)),
        ("C_jsd", (1400, 20)),
        ("U_cycle_shuffle", (1400, 20)),
    ],
)
def test_score_tensor_shapes(name, shape, cycle_fixed):
    assert cycle_fixed[name].shape == shape


def test_generator_hypothesis_and_confidence_are_exact(cycle_fixed):
    expected_conf, expected_y = torch.max(
        cycle_fixed["p_gen"], dim=-1
    )

    assert torch.equal(cycle_fixed["y_gen"], expected_y)
    assert torch.equal(cycle_fixed["conf_gen"], expected_conf)


def test_verifier_support_and_closure_are_exact(cycle_fixed):
    expected_y_ver = cycle_fixed["p_ver"].argmax(dim=-1)
    expected_support = torch.gather(
        cycle_fixed["p_ver"],
        -1,
        cycle_fixed["y_gen"].unsqueeze(-1),
    ).squeeze(-1)

    assert torch.equal(cycle_fixed["y_ver"], expected_y_ver)
    assert torch.equal(cycle_fixed["support_ver"], expected_support)
    assert torch.equal(
        cycle_fixed["closure"],
        expected_y_ver == cycle_fixed["y_gen"],
    )


def test_U_cycle_exact_formula(cycle_fixed):
    expected = (
        cycle_fixed["closure"].to(cycle_fixed["conf_gen"].dtype)
        * torch.sqrt(
            cycle_fixed["conf_gen"] * cycle_fixed["support_ver"]
        )
    )

    assert torch.equal(cycle_fixed["U_cycle"], expected)


def test_U_cycle_is_bounded_and_zero_on_disagreement(cycle_fixed):
    utility = cycle_fixed["U_cycle"]
    disagreement = ~cycle_fixed["closure"]

    assert torch.all((utility >= 0.0) & (utility <= 1.0))
    assert torch.any(disagreement)
    assert torch.all(utility[disagreement] == 0.0)


def test_U_cycle_calculation_has_no_R_temperature_topk_or_threshold():
    source = inspect.getsource(c0.build_cycle_scores)
    formula = source.split("# U_cycle:", 1)[1].split(
        "C_conf =", 1
    )[0]

    assert "R" not in formula
    assert "temperature" not in formula
    assert "topk" not in formula.lower()
    assert "threshold" not in formula.lower()


def test_confidence_baseline_is_generator_confidence(cycle_fixed):
    assert torch.equal(cycle_fixed["C_conf"], cycle_fixed["conf_gen"])
    assert cycle_fixed["audit"]["confidence_used_as_baseline_only"]


def test_jsd_is_one_for_identical_posteriors():
    posterior = torch.full(
        (1400, 20, 7), 1.0 / 7.0, dtype=torch.float64
    )

    consistency = c0.historical_jsd_consistency(
        posterior, posterior
    )

    torch.testing.assert_close(
        consistency,
        torch.ones(1400, 20, dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_jsd_is_zero_for_disjoint_one_hot_posteriors():
    first = torch.zeros(1400, 20, 7, dtype=torch.float64)
    second = torch.zeros_like(first)
    first[..., 0] = 1.0
    second[..., 1] = 1.0

    consistency = c0.historical_jsd_consistency(first, second)

    torch.testing.assert_close(
        consistency,
        torch.zeros_like(consistency),
        rtol=0.0,
        atol=1e-12,
    )


def test_jsd_matches_standard_formula(cycle_fixed):
    first = cycle_fixed["p_gen"].double()
    second = cycle_fixed["p_ver"].double()
    midpoint = 0.5 * (first + second)
    expected_js = 0.5 * torch.sum(
        first * torch.log(torch.clamp(first, min=1e-12)
                          / torch.clamp(midpoint, min=1e-12)),
        dim=-1,
    ) + 0.5 * torch.sum(
        second * torch.log(torch.clamp(second, min=1e-12)
                           / torch.clamp(midpoint, min=1e-12)),
        dim=-1,
    )
    expected = 1.0 - expected_js / np.log(2.0)

    torch.testing.assert_close(
        cycle_fixed["C_jsd"].double(),
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


def test_jsd_is_diagnostic_only(cycle_fixed):
    assert cycle_fixed["audit"]["primary_operator_is_jsd"] is False
    assert cycle_fixed["audit"][
        "historical_jsd_used_as_baseline_only"
    ] is True
    formula = inspect.getsource(c0.build_cycle_scores).split(
        "# U_cycle:", 1
    )[1].split("C_conf =", 1)[0]
    assert "C_jsd" not in formula


def test_shuffle_permutation_is_fixed_rotation_derangement(cycle_fixed):
    permutation = cycle_fixed["shuffle_permutation"]
    expected = (torch.arange(1400) + 1) % 1400

    assert torch.equal(permutation.cpu(), expected)
    assert torch.all(permutation != torch.arange(1400))
    assert cycle_fixed["audit"]["deterministic_no_rng"]


def test_shuffle_preserves_verifier_row_multiset(cycle_fixed):
    permutation = cycle_fixed["shuffle_permutation"]
    restored = cycle_fixed["p_ver_null"][torch.argsort(permutation)]

    assert torch.equal(restored, cycle_fixed["p_ver"])
    assert cycle_fixed["audit"]["verifier_row_multiset_preserved"]


def test_shuffle_leaves_generator_and_confidence_unchanged(cycle_fixed):
    expected_support = torch.gather(
        cycle_fixed["p_ver_null"],
        -1,
        cycle_fixed["y_gen"].unsqueeze(-1),
    ).squeeze(-1)

    assert torch.equal(cycle_fixed["support_ver_null"], expected_support)
    assert cycle_fixed["audit"]["generator_unchanged_pass"]


def test_shuffle_utility_exact_formula(cycle_fixed):
    expected = (
        cycle_fixed["closure_null"].to(
            cycle_fixed["conf_gen"].dtype
        )
        * torch.sqrt(
            cycle_fixed["conf_gen"]
            * cycle_fixed["support_ver_null"]
        )
    )

    assert torch.equal(cycle_fixed["U_cycle_shuffle"], expected)


def test_prediction_seal_saves_hashes_reloads_before_GT(
    tmp_path, cycle_fixed
):
    representation = {
        "q_aligned": {
            "shape": [1400, 6, 7],
            "logical_sha256": "q-hash",
        },
        "M": {
            "shape": [6, 7, 7],
            "logical_sha256": "m-hash",
        },
        "native_global_reference": {
            "shape": [1400],
            "logical_sha256": "native-hash",
        },
    }
    native = np.arange(1400, dtype=np.int64) % 7
    output_dir = tmp_path / "c0"
    seal = evaluator.save_prediction_seal(
        cycle_fixed, native, representation, output_dir
    )
    artifact = output_dir / "c0_predictions_and_scores.npz"

    assert artifact.is_file()
    assert (output_dir / "c0_prediction_seal.json").is_file()
    assert seal["scores_completed_before_GT"]
    assert seal["scores_saved_before_GT"]
    assert seal["scores_hashed_before_GT"]
    assert seal["scores_reloaded_before_GT"]
    assert seal["full_GT_loaded_before_seal"] is False
    with np.load(artifact, allow_pickle=False) as archive:
        assert tuple(archive.files) == evaluator.SEALED_ARRAY_NAMES
        for name in evaluator.SEALED_ARRAY_NAMES:
            assert seal["arrays"][name]["logical_sha256"] == (
                tensor_sha256(archive[name])
            )
    with open(
        output_dir / "c0_prediction_seal.json",
        "r",
        encoding="utf-8",
    ) as input_file:
        assert json.load(input_file) == seal


def test_prediction_seal_refuses_overwrite(tmp_path, cycle_fixed):
    representation = {
        "q_aligned": {},
        "M": {},
        "native_global_reference": {},
    }
    native = np.arange(1400, dtype=np.int64) % 7
    output_dir = tmp_path / "sealed"
    evaluator.save_prediction_seal(
        cycle_fixed, native, representation, output_dir
    )

    with pytest.raises(RuntimeError, match="output boundary"):
        evaluator.save_prediction_seal(
            cycle_fixed, native, representation, output_dir
        )


def test_run_evaluation_seals_before_loading_full_GT():
    source = inspect.getsource(evaluator.run_evaluation)

    assert source.index("build_cycle_scores") < source.index(
        "save_prediction_seal"
    )
    assert source.index("save_prediction_seal") < source.index(
        "load_full_ground_truth_after_seal"
    )
    assert source.index("load_full_ground_truth_after_seal") < (
        source.index("evaluate_postseal")
    )


def test_primary_execution_path_has_no_sparse_label_or_R_loader():
    source = (
        inspect.getsource(evaluator.run_evaluation)
        + inspect.getsource(evaluator.load_frozen_e1_aligned_q)
        + inspect.getsource(c0.build_cycle_scores)
    )

    assert "load_sparse" not in source
    assert "load_frozen_reliability" not in source
    assert "load_oracle_corruption" not in source
    assert "load_corruption_mask" not in source
    assert "full_GT" not in inspect.getsource(c0.build_cycle_scores)
    assert tuple(inspect.signature(c0.build_cycle_scores).parameters) == (
        "q_aligned",
    )


def test_one_global_cluster_mapping_is_exact():
    labels, clusters, expected_mapping = _mapping_inputs()

    record = c0.fit_global_cluster_mapping_once(clusters, labels)

    assert np.array_equal(record["mapping"], expected_mapping)
    assert np.array_equal(record["mapping"][clusters], labels)
    assert record["mapping_fit_count"] == 1
    assert record["mapping_fit_after_score_seal"]
    assert record["single_global_mapping_reused_all_directions"]
    assert not record["direction_specific_mapping_used"]
    assert not record["sparse_labels_used_for_mapping"]


def test_one_mapping_applies_to_all_twenty_directions():
    labels, clusters, mapping = _mapping_inputs()
    y_gen = np.tile(clusters[:, None], (1, 20))

    semantic = c0.apply_global_mapping(y_gen, mapping)

    assert semantic.shape == (1400, 20)
    assert np.array_equal(
        semantic, np.tile(labels[:, None], (1, 20))
    )


def test_no_per_direction_hungarian():
    source = inspect.getsource(c0.evaluate_postseal)

    assert source.count("fit_global_cluster_mapping_once") == 1
    assert "linear_sum_assignment" not in source
    assert "for direction" not in inspect.getsource(
        c0.fit_global_cluster_mapping_once
    )


def test_directional_auc_is_computed_per_twenty_splits():
    correct, scores = _score_arrays_for_auc()
    result = c0.evaluate_directional_aucs(correct, scores)

    assert result["valid_direction_count"] == 20
    assert result["invalid_direction_ids"] == []
    assert len(result["AUC_cycle"]) == 20
    assert result["primary_evaluation_unit"].startswith("20 directional")
    assert result[
        "record_count_not_independent_experiments"
    ] == 28000


def test_directional_auc_matches_sklearn_per_direction():
    correct, scores = _score_arrays_for_auc()
    values, invalid = c0.directional_roc_auc(
        correct, scores["cycle"]
    )

    assert invalid == []
    assert values == [1.0] * 20


def test_invalid_direction_is_recorded_not_dropped():
    correct, scores = _score_arrays_for_auc()
    correct[:, 0] = True

    values, invalid = c0.directional_roc_auc(
        correct, scores["cycle"]
    )

    assert invalid == [0]
    assert values[0] is None
    assert len(values) == 20


def test_auc_summary_has_all_fixed_statistics():
    correct, scores = _score_arrays_for_auc()
    result = c0.evaluate_directional_aucs(correct, scores)

    for score_name in c0.SCORE_NAMES:
        assert set(result["summary"][score_name]) == {
            "mean",
            "median",
            "std",
            "min",
            "max",
        }


def test_paired_wilcoxon_uses_greater_alternative():
    first = np.linspace(0.7, 0.9, 20)
    second = first - 0.1

    p_value = c0.paired_wilcoxon_greater(first, second)

    assert p_value < 0.05
    assert 'alternative="greater"' in inspect.getsource(
        c0.paired_wilcoxon_greater
    )


def test_paired_auc_comparison_uses_twenty_direction_pairs():
    base = np.linspace(0.6, 0.8, 20)
    auc_result = {
        "invalid_direction_ids": [],
        "AUC_cycle": base.tolist(),
        "AUC_conf": (base - 0.02).tolist(),
        "AUC_jsd": (base - 0.03).tolist(),
        "AUC_shuffle": (base - 0.04).tolist(),
    }

    result = c0.paired_auc_comparisons(auc_result)

    assert result["pair_count"] == 20
    assert result["alpha"] == 0.05
    assert result["alternative"] == "greater"
    assert result["p_cycle_vs_conf"] < 0.05
    assert result["p_cycle_vs_jsd"] < 0.05
    assert result["p_cycle_vs_shuffle"] < 0.05


def test_score_separation_records_required_values():
    correct = _correct_matrix()
    utility = correct.astype(np.float64) * 0.8
    utility[~correct] = 0.2

    result = c0.score_separation(correct, utility)

    assert result["mean_U_correct"] == pytest.approx(0.8)
    assert result["mean_U_wrong"] == pytest.approx(0.2)
    assert result["median_U_correct"] == pytest.approx(0.8)
    assert result["median_U_wrong"] == pytest.approx(0.2)
    assert result["mean_U_correct_minus_wrong"] == pytest.approx(0.6)
    assert result["pooled_records_secondary_diagnostic_only"]


@pytest.mark.parametrize(
    "coverage,expected_count",
    [(0.2, 280), (0.4, 560), (0.6, 840)],
)
def test_risk_coverage_uses_only_fixed_coverages(
    coverage, expected_count
):
    correct = _correct_matrix()
    score = correct.astype(np.float64)
    result = c0.risk_coverage_diagnostic(
        correct,
        {
            "cycle": score,
            "confidence": score,
            "jsd": score,
        },
    )
    key = str(int(coverage * 100))

    assert result["coverages"] == [0.2, 0.4, 0.6]
    assert result["per_direction"]["cycle"][key][
        "admitted_count_per_direction"
    ] == expected_count
    assert len(
        result["per_direction"]["cycle"][key][
            "precision_by_direction"
        ]
    ) == 20
    assert result["primary_gate_used"] is False


def test_risk_coverage_reports_all_three_comparators():
    correct = _correct_matrix()
    score = correct.astype(np.float64)
    result = c0.risk_coverage_diagnostic(
        correct,
        {
            "cycle": score,
            "confidence": score,
            "jsd": score,
        },
    )

    assert set(result["per_direction"]) == {
        "cycle",
        "confidence",
        "jsd",
    }
    assert set(result["macro_mean"]) == {
        "precision_cycle@20",
        "precision_cycle@40",
        "precision_cycle@60",
        "precision_confidence@20",
        "precision_confidence@40",
        "precision_confidence@60",
        "precision_jsd@20",
        "precision_jsd@40",
        "precision_jsd@60",
    }


def test_gate_1_is_exact():
    auc, paired, separation = _gate_inputs(signal=True)
    decision = c0.build_c0_decision(auc, paired, separation)

    assert decision["C0_SIGNAL_PASS"]

    auc["summary"]["cycle"]["mean"] = 0.5
    decision = c0.build_c0_decision(auc, paired, separation)
    assert not decision["C0_SIGNAL_PASS"]


def test_gate_2_is_exact():
    auc, paired, separation = _gate_inputs(specificity=True)
    decision = c0.build_c0_decision(auc, paired, separation)
    assert decision["C0_CORRESPONDENCE_SPECIFICITY_PASS"]

    paired["p_cycle_vs_shuffle"] = 0.05
    decision = c0.build_c0_decision(auc, paired, separation)
    assert not decision["C0_CORRESPONDENCE_SPECIFICITY_PASS"]


def test_gate_3_is_exact():
    auc, paired, separation = _gate_inputs(beats_confidence=True)
    decision = c0.build_c0_decision(auc, paired, separation)
    assert decision["C0_BEATS_CONFIDENCE_PASS"]

    auc["summary"]["confidence"]["mean"] = (
        auc["summary"]["cycle"]["mean"]
    )
    decision = c0.build_c0_decision(auc, paired, separation)
    assert not decision["C0_BEATS_CONFIDENCE_PASS"]


def test_gate_4_is_exact():
    auc, paired, separation = _gate_inputs(beats_jsd=True)
    decision = c0.build_c0_decision(auc, paired, separation)
    assert decision["C0_BEATS_JSD_PASS"]

    paired["p_cycle_vs_jsd"] = 0.05
    decision = c0.build_c0_decision(auc, paired, separation)
    assert not decision["C0_BEATS_JSD_PASS"]


@pytest.mark.parametrize(
    "signal,specificity,beats_confidence,beats_jsd",
    list(itertools.product((False, True), repeat=4)),
)
def test_final_decision_tree_is_exhaustive(
    signal, specificity, beats_confidence, beats_jsd
):
    auc, paired, separation = _gate_inputs(
        signal,
        specificity,
        beats_confidence,
        beats_jsd,
    )
    decision = c0.build_c0_decision(auc, paired, separation)
    conditions = decision["decision_conditions"]

    assert sum(conditions.values()) == 1
    if not signal:
        assert decision["final_decision"] == (
            "C0_CYCLIC_SEMANTIC_SIGNAL_FAIL"
        )
    elif not specificity:
        assert decision["final_decision"] == (
            "C0_SAMPLE_CORRESPONDENCE_NOT_SPECIFIC"
        )
    elif beats_confidence and beats_jsd:
        assert decision["final_decision"] == (
            "C0_COMPLEMENTARY_CYCLE_VALIDITY_PASS"
        )
    else:
        assert decision["final_decision"] == (
            "C0_CYCLE_REDUNDANT_WITH_EXISTING_CONFIDENCE"
        )


def test_only_full_pass_allows_future_C1():
    auc, paired, separation = _gate_inputs(True, True, True, True)
    decision = c0.build_c0_decision(auc, paired, separation)

    assert decision["C1_pseudo_supervision_training_allowed"]
    assert not decision["training_entered"]


def test_no_training_optimizer_backward_or_parameter_update():
    source = (
        inspect.getsource(c0)
        + inspect.getsource(evaluator)
    )

    for forbidden in (
        "torch.optim",
        ".backward(",
        ".train(",
        "scheduler",
        "parameter.data",
    ):
        assert forbidden not in source


def test_historical_actions_absent_from_primary_operator():
    source = inspect.getsource(c0.build_cycle_scores)

    for forbidden in (
        "memory",
        "P_corr",
        "target_correction",
        "reliability",
        "sparse",
        "topk",
    ):
        assert forbidden not in source.lower()


def test_frozen_representation_loader_uses_audited_alignment():
    source = inspect.getsource(evaluator.load_frozen_e1_aligned_q)

    assert "frozen_e1._load_frozen_lwc_model" in source
    assert "load_frozen_feature_artifact" in source
    assert "replay_native_global_reference" in source
    assert "build_alignment" in source
    assert "frozen_e1.align_q_to_global" in source
    assert '"nvk,vjk->nvj"' in source
    assert "h_sem" not in source


def test_model_parameter_hash_is_checked_before_and_after():
    source = inspect.getsource(evaluator.load_frozen_e1_aligned_q)

    assert source.count("hash_backbone") == 2
    assert "parameter_hash_before == parameter_hash_after" in source
    assert "parameter.requires_grad" in source
    assert "parameter.grad is None" in source


E1_CHECKPOINT_HASHES = (
    "b914af6e3740d1808d44ed2592b012dd67bb90356b745c2c117d63743256d77c",
    "7c5ce3be3fc2a62b0731b58651e29af2da6e2dfd2eb51f53d2e574e019e3c576",
    "69d65bfeb51617a3a560aa040e7be8c2ad9e2ef90368ca052914c49fb04a28ac",
    "cd629a4b604e814595d1f9c9e720a94cdcf805d871863b457b95b4f97866a33c",
    "6268ceeda5c9cf1dedc0de1550888da0d61e25ca1f786cd6c181639257df692d",
    "8a9117c23a45b24e4e6641746e4f67c864c0a0638edbe410f4cf7866c79a1701",
)


@pytest.mark.parametrize(
    "view_id,expected_sha",
    list(enumerate(E1_CHECKPOINT_HASHES)),
)
def test_frozen_E1_LWC_checkpoint_provenance(view_id, expected_sha):
    path = (
        evaluator.DEFAULT_E1_LWC_MODEL_DIR
        / ("Caltech-6V" + str(view_id + 1) + "V.pth")
    )

    assert path.is_file()
    assert _file_sha256(path) == expected_sha


E4_SOURCE_HASHES = {
    "e4a1_memory_specific_utility.py": (
        "ec0b6af7867ac19795abe7cafcc15c5e793421c1fc2d2f4c73920a8be76ac7de"
    ),
    "e4cf0_counterfactual_utility.py": (
        "2d4dbb365089363baa42eca9793e48956f4f40db8f5a404585e9f97129853438"
    ),
    "evaluate_e4_a0_memory_feasibility.py": (
        "5f9e26d4ef3d446f22743dc4f7900369e9a0d82bde6f13a170eaa89278e99e4c"
    ),
    "evaluate_e4_a1_memory_specific_utility.py": (
        "123e04e616df758f38d383d9c5776ef7398f2078b50556919d7faa712e40f621"
    ),
    "evaluate_e4cf0_counterfactual_utility.py": (
        "a353e02c5937e7ba9cac4e6ebf84d03c34af7139871413715dce7d95e215fb25"
    ),
    "semantic_memory_bank.py": (
        "8c4f99e1cf31ec03eb69183445076177362e6b4adc6e6dc56350c76f41a0afe6"
    ),
}


@pytest.mark.parametrize(
    "filename,expected_sha", E4_SOURCE_HASHES.items()
)
def test_E4_source_files_remain_bitwise_frozen(filename, expected_sha):
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments/e4_semantic_memory_bank" / filename

    assert _file_sha256(path) == expected_sha


def test_output_contract_and_formal_path_are_exact():
    assert evaluator.DEFAULT_OUTPUT_DIR == (
        evaluator.REPOSITORY_ROOT
        / "outputs/cyclic_utility"
        / "c0_complementary_semantic_verification_seed20"
    )
    source = inspect.getsource(evaluator.run_evaluation)
    for artifact in (
        "c0_auc_by_direction.json",
        "c0_risk_coverage.json",
        "c0_audit.json",
        "diagnostic_results.json",
    ):
        assert artifact in source
    assert "c0_predictions_and_scores.npz" in inspect.getsource(
        evaluator.save_prediction_seal
    )
    assert "c0_prediction_seal.json" in inspect.getsource(
        evaluator.save_prediction_seal
    )

