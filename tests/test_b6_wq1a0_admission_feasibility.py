"""Tests for the B6-WQ1A-0 admission feasibility audit."""

import inspect

import numpy as np
import pytest

import experiments.b6_weak_quality.audit_b6_wq1a0_admission_feasibility as audit
import experiments.b6_weak_quality.evaluate_b6_wq0_utility_semantic_admission as b6


@pytest.fixture(scope="module")
def seed20_inputs():
    return audit.load_frozen_audit_inputs(20)


@pytest.fixture(scope="module")
def seed20_audit(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("b6_wq1a0_seed20")
    manifest_before = b6.CANONICAL_MANIFEST_PATH.read_bytes()
    result = audit.run_seed_audit(20, output_dir)
    manifest_after = b6.CANONICAL_MANIFEST_PATH.read_bytes()
    return {
        "result": result,
        "output_dir": output_dir,
        "manifest_before": manifest_before,
        "manifest_after": manifest_after,
    }


@pytest.fixture()
def toy_admission_case():
    oracle = np.asarray([
        [True, True, True, False, False],
        [True, True, True, False, False],
    ])
    admitted = np.asarray([
        [True, True, True, False, False],
        [True, True, False, True, False],
    ])
    utility = np.asarray([
        [0.9, 0.8, 0.7, 0.2, 0.1],
        [0.9, 0.8, 0.3, 0.7, 0.1],
    ])
    metrics = audit.admission_metrics(
        admitted,
        oracle,
        utility,
    )
    return admitted, oracle, utility, metrics


def test_utility_shape_range_and_provenance(seed20_inputs):
    utility = seed20_inputs["utility"]
    assert utility.shape == (210, 5)
    assert np.isfinite(utility).all()
    assert utility.min() >= 0.0
    assert utility.max() <= 1.0
    assert (
        seed20_inputs["utility_sha256"]
        == b6.SEED_PROVENANCE[20]["utility_hash"]
    )
    assert seed20_inputs["utility_source"].is_file()
    assert seed20_inputs["utility_provenance"].is_file()


def test_corruption_mask_shape_is_210_by_5(seed20_inputs):
    mask = seed20_inputs["corruption_mask"]
    assert mask.shape == (210, 5)
    assert mask.dtype == np.dtype(bool)


def test_corruption_mask_has_exactly_two_corrupt_views_per_sample(
    seed20_inputs,
):
    assert np.all(
        seed20_inputs["corruption_mask"].sum(axis=1) == 2
    )


def test_oracle_has_exactly_three_clean_views_per_sample(seed20_inputs):
    oracle = seed20_inputs["oracle_clean_mask"]
    assert oracle.shape == (210, 5)
    assert np.all(oracle.sum(axis=1) == 3)
    assert np.array_equal(
        oracle,
        np.logical_not(seed20_inputs["corruption_mask"]),
    )


def test_correct_top3_has_exactly_three_views_per_sample(seed20_inputs):
    mask = audit.top3_admission_mask(seed20_inputs["utility"])
    assert mask.shape == (210, 5)
    assert mask.dtype == np.dtype(bool)
    assert np.all(mask.sum(axis=1) == 3)


def test_top3_ties_are_broken_stably_by_low_view_index():
    utility = np.asarray([
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0, 1.0, 0.0],
    ])
    expected = np.asarray([
        [True, True, True, False, False],
        [True, True, True, False, False],
    ])
    first = audit.top3_admission_mask(utility)
    second = audit.top3_admission_mask(utility)
    assert np.array_equal(first, expected)
    assert np.array_equal(second, expected)


def test_shuffled_utility_preserves_each_view_distribution(seed20_inputs):
    utility = seed20_inputs["utility"]
    bank = audit.generate_permutation_bank()
    shuffled = audit.shuffle_utility_within_views(
        utility,
        bank[0],
    )
    assert bank.shape == (200, 5, 210)
    for view_id in range(5):
        assert np.array_equal(
            np.sort(shuffled[:, view_id]),
            np.sort(utility[:, view_id]),
        )


def test_shuffled_null_contains_exactly_200_repeats(seed20_audit):
    result = seed20_audit["result"]
    null_path = audit._resolve(
        result["shuffled_null_metrics_path"]
    )
    assert result["shuffled_null_repeats"] == 200
    assert result["shuffled_null_seed"] == b6.SHUFFLE_SEED
    with np.load(null_path, allow_pickle=False) as null_values:
        assert set(null_values.files) == set(audit.METRIC_NAMES)
        assert all(
            null_values[name].shape == (200,)
            for name in audit.METRIC_NAMES
        )


def test_admitted_clean_fraction_formula_is_correct(toy_admission_case):
    _, _, _, metrics = toy_admission_case
    assert metrics["admitted_clean_fraction"] == pytest.approx(5.0 / 6.0)


def test_all_clean_top3_fraction_formula_is_correct(toy_admission_case):
    _, _, _, metrics = toy_admission_case
    assert metrics["all_clean_top3_fraction"] == 0.5


def test_corrupted_admission_relation_is_exact(toy_admission_case):
    _, _, _, metrics = toy_admission_case
    corrupt_mean = metrics["corrupted_views_admitted_mean"]
    assert corrupt_mean == 0.5
    assert metrics["admitted_clean_fraction"] == pytest.approx(
        1.0 - corrupt_mean / 3.0,
        abs=1e-15,
    )


def test_oracle_set_jaccard_mean_is_correct(toy_admission_case):
    _, _, _, metrics = toy_admission_case
    assert metrics["oracle_set_jaccard_mean"] == 0.75


def test_pair_win_rate_counts_ties_as_one_half():
    utility = np.ones((2, 5), dtype=np.float64)
    oracle = np.asarray([
        [True, True, True, False, False],
        [False, True, True, True, False],
    ])
    assert (
        audit.within_sample_clean_corrupt_pair_win_rate(
            utility,
            oracle,
        )
        == 0.5
    )


def test_null_summary_uses_preregistered_tail_directions():
    values = np.linspace(0.0, 1.0, 200)
    higher = audit.summarize_null(
        values,
        correct_value=0.75,
        higher_is_better=True,
    )
    lower = audit.summarize_null(
        values,
        correct_value=0.25,
        higher_is_better=False,
    )
    assert higher["count_as_or_more_extreme"] == int(
        np.sum(values >= 0.75)
    )
    assert lower["count_as_or_more_extreme"] == int(
        np.sum(values <= 0.25)
    )
    assert higher["one_sided_empirical_p"] == (
        1 + np.sum(values >= 0.75)
    ) / 201
    assert lower["one_sided_empirical_p"] == (
        1 + np.sum(values <= 0.25)
    ) / 201


def test_seed_gate_requires_a_and_b_and_c():
    correct = {
        "admitted_clean_fraction": 0.8,
        "all_clean_top3_fraction": 0.4,
        "within_sample_pair_win_rate": 0.7,
    }
    null = {
        "admitted_clean_fraction": {
            "p95": 0.7,
            "one_sided_empirical_p": 0.01,
        },
        "all_clean_top3_fraction": {
            "p95": 0.3,
            "one_sided_empirical_p": 0.01,
        },
        "within_sample_pair_win_rate": {
            "p95": 0.6,
            "one_sided_empirical_p": 0.01,
        },
    }
    result = audit.seed_feasibility_gate(correct, null)
    assert result["gate_a_admitted_clean_fraction"]
    assert result["gate_b_all_clean_top3_fraction"]
    assert result["gate_c_within_sample_pair_win_rate"]
    assert result["B6_WQ1A0_ADMISSION_FEASIBILITY_SEED_PASS"]

    null["within_sample_pair_win_rate"][
        "one_sided_empirical_p"
    ] = 0.05
    assert not audit.seed_feasibility_gate(
        correct,
        null,
    )["B6_WQ1A0_ADMISSION_FEASIBILITY_SEED_PASS"]


def _multiseed_row(seed_pass, clean, null_clean, all_clean, null_all, pair):
    return {
        "B6_WQ1A0_ADMISSION_FEASIBILITY_SEED_PASS": seed_pass,
        "correct_metrics": {
            "admitted_clean_fraction": clean,
            "all_clean_top3_fraction": all_clean,
            "within_sample_pair_win_rate": pair,
        },
        "shuffled_null_summary": {
            "admitted_clean_fraction": {"mean": null_clean},
            "all_clean_top3_fraction": {"mean": null_all},
        },
    }


def test_multiseed_gate_uses_preregistered_conditions():
    rows = [
        _multiseed_row(True, 0.8, 0.6, 0.4, 0.1, 0.7),
        _multiseed_row(True, 0.7, 0.6, 0.3, 0.1, 0.6),
        _multiseed_row(False, 0.6, 0.6, 0.1, 0.1, 0.5),
    ]
    result = audit.multiseed_feasibility_gate(rows)
    assert result["seed_pass_count"] == 2
    assert all(result["multiseed_gate_conditions"].values())
    assert result[
        "B6_WQ1A0_ADMISSION_FEASIBILITY_MULTISEED_PASS"
    ]


def test_no_labels_are_loaded_or_used(seed20_audit):
    source = inspect.getsource(audit)
    result = seed20_audit["result"]
    assert "load_data(" not in source
    assert "prepare_seed(" not in source
    assert not result["labels_loaded"]
    assert not result["labels_used"]


def test_no_optimizer_is_created(seed20_audit):
    source = inspect.getsource(audit)
    result = seed20_audit["result"]
    assert "torch.optim" not in source
    assert not result["optimizer_created"]


def test_no_backward_is_performed(seed20_audit):
    source = inspect.getsource(audit)
    result = seed20_audit["result"]
    assert ".backward(" not in source
    assert not result["backward_performed"]


def test_no_model_or_parameter_modification(seed20_audit):
    source = inspect.getsource(audit)
    result = seed20_audit["result"]
    assert "load_frozen_backbone(" not in source
    assert "state_dict" not in source
    assert ".step(" not in source
    assert not result["model_loaded"]
    assert not result["parameter_updates"]


def test_no_clustering_metrics_are_computed(seed20_audit):
    result = seed20_audit["result"]
    assert set(result["correct_metrics"]) == set(audit.METRIC_NAMES)
    assert {"acc", "nmi", "ari"}.isdisjoint(
        result["correct_metrics"]
    )
    assert not result["clustering_metrics_computed"]


def test_existing_b6_canonical_manifest_is_unchanged(seed20_audit):
    assert (
        seed20_audit["manifest_before"]
        == seed20_audit["manifest_after"]
    )
