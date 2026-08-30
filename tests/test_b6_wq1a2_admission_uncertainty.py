"""Tests for B6-WQ1A-2 admission uncertainty diagnostic."""

import csv
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

import experiments.b6_weak_quality.audit_b6_wq1a0_admission_feasibility as wq1a0
import experiments.b6_weak_quality.audit_b6_wq1a2_admission_uncertainty as audit


@pytest.fixture(scope="module")
def frozen_inputs():
    return audit.load_read_only_inputs(20)


@pytest.fixture(scope="module")
def sample_diagnostics(frozen_inputs):
    return audit.build_sample_admission_diagnostics(
        frozen_inputs["utility"],
        frozen_inputs["corruption_mask"],
    )


@pytest.fixture(scope="module")
def shuffled_diagnostics(frozen_inputs):
    return audit.regenerate_formal_shuffled_diagnostics(
        frozen_inputs
    )


@pytest.fixture(scope="module")
def completed_audit(tmp_path_factory):
    output_root = tmp_path_factory.mktemp("b6_wq1a2")
    formal_paths = {
        name: audit._resolve(audit.FORMAL_ROOT) / name
        for name in audit.EXPECTED_FORMAL_FILE_SHA256
    }
    before = {
        str(path): path.read_bytes()
        for path in formal_paths.values()
    }
    protected_before = audit.protected_file_hashes()
    wq1a0_before = audit._wq1a0_artifact_hashes()
    summary = audit.run_audit(
        seed=20,
        output_dir=output_root,
    )
    after = {
        str(path): path.read_bytes()
        for path in formal_paths.values()
    }
    return {
        "summary": summary,
        "output_root": output_root,
        "formal_before": before,
        "formal_after": after,
        "protected_before": protected_before,
        "protected_after": audit.protected_file_hashes(),
        "wq1a0_before": wq1a0_before,
        "wq1a0_after": audit._wq1a0_artifact_hashes(),
    }


def test_frozen_utility_shape_is_210_by_5(frozen_inputs):
    assert frozen_inputs["utility"].shape == (210, 5)


def test_margin34_formula_is_exact():
    utility = np.asarray([
        [0.2, 0.9, 0.5, 0.7, 0.1],
        [5.0, 4.0, 3.0, 2.0, 1.0],
    ])
    ordering = audit.utility_ordering(utility)
    assert np.array_equal(
        ordering["top3_utility"],
        np.asarray([0.5, 3.0]),
    )
    assert np.array_equal(
        ordering["top4_utility"],
        np.asarray([0.2, 2.0]),
    )
    assert np.array_equal(
        ordering["margin_34"],
        np.asarray([0.3, 1.0]),
    )


def test_stable_descending_sort_is_deterministic():
    utility = np.asarray([
        [0.5, 0.5, 0.5, 0.2, 0.2],
        [1.0, 0.7, 0.7, 0.7, 0.1],
    ])
    first = audit.utility_ordering(utility)
    second = audit.utility_ordering(utility.copy())
    expected = np.asarray([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ])
    assert np.array_equal(
        first["descending_view_order"],
        expected,
    )
    assert np.array_equal(
        first["descending_view_order"],
        second["descending_view_order"],
    )


def test_all_clean_target_uses_only_corruption_mask(
    frozen_inputs,
    sample_diagnostics,
):
    oracle = ~frozen_inputs["corruption_mask"]
    correct = wq1a0.top3_admission_mask(
        frozen_inputs["utility"]
    )
    expected = np.all(correct == oracle, axis=1).astype(np.int64)
    assert np.array_equal(
        sample_diagnostics["all_clean_top3"],
        expected,
    )
    assert np.array_equal(
        sample_diagnostics["admission_error"],
        1 - expected,
    )
    parameters = inspect.signature(
        audit.build_sample_admission_diagnostics
    ).parameters
    assert tuple(parameters) == ("utility", "corruption_mask")


def test_no_clustering_labels_are_loaded():
    source = inspect.getsource(audit)
    for forbidden_call in (
        "load_evaluation_labels(",
        "load_labels(",
        "variable_names=[\"gt\"]",
        "sio.loadmat(",
    ):
        assert forbidden_call not in source


def test_no_model_is_loaded():
    source = inspect.getsource(audit)
    frozen_source = inspect.getsource(
        wq1a0.load_frozen_audit_inputs
    )
    for forbidden_call in (
        "load_frozen_backbone(",
        "torch.load(",
        "ViewProjectors(",
        "load_state_dict(",
    ):
        assert forbidden_call not in source
        assert forbidden_call not in frozen_source


def test_no_optimizer_is_created():
    source = inspect.getsource(audit)
    assert "torch.optim" not in source
    assert "optim.Adam(" not in source


def test_no_backward_is_performed():
    source = inspect.getsource(audit)
    assert ".backward(" not in source


def test_no_kmeans_fit_is_performed():
    source = inspect.getsource(audit)
    for forbidden_call in (
        "KMeans(",
        "MiniBatchKMeans(",
        ".fit_predict(",
        ".fit(",
    ):
        assert forbidden_call not in source


def test_formal_shuffled_arrays_are_exact_200_and_finite(
    frozen_inputs,
):
    for metric in ("acc", "nmi", "ari"):
        values = frozen_inputs["shuffled_metrics"][metric]
        assert values.shape == (200,)
        assert np.isfinite(values).all()


def test_regenerated_shuffle_count_is_exactly_200(
    shuffled_diagnostics,
):
    assert audit.FORMAL_SHUFFLE_REPEATS == 200
    assert len(shuffled_diagnostics["rows"]) == 200


def test_shuffle_repeat_alignment_is_deterministic(
    frozen_inputs,
    shuffled_diagnostics,
):
    rows = shuffled_diagnostics["rows"]
    formal_rows = frozen_inputs["formal_summary"]["shuffled_u"]
    for repeat_id, row in enumerate(rows):
        assert row["repeat_id"] == repeat_id
        assert (
            row["shuffled_utility_sha256"]
            == formal_rows[repeat_id][
                "shuffled_utility_sha256"
            ]
        )
        for metric in ("acc", "nmi", "ari"):
            assert row[metric] == frozen_inputs[
                "shuffled_metrics"
            ][metric][repeat_id]
    assert shuffled_diagnostics[
        "utility_hash_alignment_pass"
    ]
    assert shuffled_diagnostics[
        "formal_purity_alignment_pass"
    ]
    assert shuffled_diagnostics[
        "B6_WQ1A2_SHUFFLE_BANK_REPRO_PASS"
    ]


def test_each_shuffle_preserves_per_view_utility_distribution(
    frozen_inputs,
    shuffled_diagnostics,
):
    assert all(
        row["per_view_utility_distribution_preserved"]
        for row in shuffled_diagnostics["rows"]
    )
    bank = wq1a0.generate_permutation_bank()
    shuffled = wq1a0.shuffle_utility_within_views(
        frozen_inputs["utility"],
        bank[0],
    )
    for view_id in range(5):
        assert np.array_equal(
            np.sort(shuffled[:, view_id]),
            np.sort(frozen_inputs["utility"][:, view_id]),
        )


def test_wq1a0_correct_mask_is_reproduced_exactly(
    frozen_inputs,
):
    stored = np.load(
        audit._resolve(audit.WQ1A0_ROOT)
        / "seed20/correct_admission_mask.npy",
        allow_pickle=False,
    )
    assert np.array_equal(
        frozen_inputs["correct_admission_mask"],
        stored,
    )


def test_correct_u_purity_is_reproduced_exactly(frozen_inputs):
    metrics = wq1a0.admission_metrics(
        frozen_inputs["correct_admission_mask"],
        frozen_inputs["oracle_clean_mask"],
        frozen_inputs["utility"],
    )
    assert metrics == audit.EXPECTED_CORRECT_PURITY


def test_margin_auc_formula_is_correct():
    margin = np.asarray([0.8, 0.7, 0.2, 0.1])
    error = np.asarray([0, 0, 1, 1])
    assert audit.margin_error_auc(margin, error) == 1.0
    assert audit.margin_error_auc(-margin, error) == 0.0


def test_margin_auc_tie_handling_is_average_rank():
    margin = np.ones(6, dtype=np.float64)
    error = np.asarray([0, 1, 0, 1, 0, 1])
    assert audit.margin_error_auc(margin, error) == 0.5


def test_margin_quartiles_are_deterministic(
    sample_diagnostics,
):
    args = (
        sample_diagnostics["ordering"]["margin_34"],
        sample_diagnostics["all_clean_top3"],
        sample_diagnostics["admission_error"],
        sample_diagnostics["corrupt_admitted_count"],
    )
    first = audit.margin_quartile_diagnostics(*args)
    second = audit.margin_quartile_diagnostics(*args)
    assert first == second
    assert [row["quartile"] for row in first] == [
        "Q1", "Q2", "Q3", "Q4",
    ]
    assert [row["sample_count"] for row in first] == [
        53, 53, 52, 52,
    ]
    assert sum(row["sample_count"] for row in first) == 210


def test_bootstrap_repeat_count_and_seed_are_exact(
    completed_audit,
):
    path = (
        completed_audit["output_root"]
        / "bootstrap_summary.json"
    )
    bootstrap = json.loads(path.read_text())
    assert bootstrap["bootstrap_repeats"] == 2000
    assert bootstrap["bootstrap_seed"] == 20
    assert len(bootstrap["margin34_error_auc_ci95"]) == 2
    assert bootstrap["auc_bootstrap_all_finite"]
    assert bootstrap[
        "median_difference_bootstrap_all_finite"
    ]


def test_spearman_implementation_matches_scipy_with_ties():
    x = np.asarray([1.0, 2.0, 2.0, 4.0, 5.0])
    y = np.asarray([5.0, 3.0, 3.0, 2.0, 1.0])
    expected = float(spearmanr(x, y).statistic)
    assert audit.spearman_correlation(x, y) == pytest.approx(
        expected,
        rel=0.0,
        abs=1e-15,
    )


def test_permutation_repeat_count_and_seed_are_exact(
    completed_audit,
):
    path = (
        completed_audit["output_root"]
        / "correlation_summary.json"
    )
    correlation = json.loads(path.read_text())
    for name in (
        "clean_fraction_acc_permutation",
        "allclean_fraction_acc_permutation",
    ):
        result = correlation[name]
        assert result["permutation_repeats"] == 10000
        assert result["permutation_seed"] == 20
        assert (
            0
            <= result[
                "count_abs_rho_permuted_ge_observed"
            ]
            <= 10000
        )


def test_empirical_permutation_p_formula_is_exact():
    assert audit.empirical_two_sided_p(0, 10000) == 1 / 10001
    assert (
        audit.empirical_two_sided_p(123, 10000)
        == 124 / 10001
    )
    assert audit.empirical_two_sided_p(10000, 10000) == 1.0


def test_regression_uses_exactly_200_shuffled_points():
    x = np.linspace(0.0, 1.0, 200)
    y = 0.25 + 0.5 * x
    result = audit.fit_null_purity_trend(x, y)
    assert result["fit_population"] == "200 shuffled-U arms only"
    assert result["slope"] == pytest.approx(0.5)
    assert result["intercept"] == pytest.approx(0.25)
    assert result["r2"] == pytest.approx(1.0)


def test_correct_u_is_excluded_from_null_regression_fit(
    completed_audit,
):
    parameters = inspect.signature(
        audit.fit_null_purity_trend
    ).parameters
    assert tuple(parameters) == (
        "shuffled_purity",
        "shuffled_acc",
    )
    regression = completed_audit["summary"][
        "null_purity_acc_regression"
    ]
    assert not regression["correct_u_included_in_fit"]
    assert regression["fit_population"] == (
        "200 shuffled-U arms only"
    )


def test_existing_wq1a1_artifacts_are_byte_unchanged(
    completed_audit,
):
    assert (
        completed_audit["formal_before"]
        == completed_audit["formal_after"]
    )
    for filename, expected_hash in (
        audit.EXPECTED_FORMAL_FILE_SHA256.items()
    ):
        path = audit._resolve(audit.FORMAL_ROOT) / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            expected_hash
        )


def test_wq1a0_wq0_manifest_and_core_hashes_are_unchanged(
    completed_audit,
):
    assert (
        completed_audit["protected_before"]
        == completed_audit["protected_after"]
        == audit.EXPECTED_PROTECTED_FILE_SHA256
    )
    assert (
        completed_audit["wq1a0_before"]
        == completed_audit["wq1a0_after"]
        == audit.EXPECTED_WQ1A0_FILE_SHA256
    )


def test_required_artifacts_and_row_counts_are_written(
    completed_audit,
):
    root = completed_audit["output_root"]
    required = {
        "b6_wq1a2_seed20_summary.json",
        "sample_admission_diagnostic.csv",
        "margin_quartile_summary.csv",
        "formal_shuffled_admission_diagnostics.csv",
        "correlation_summary.json",
        "bootstrap_summary.json",
        "metadata.json",
    }
    assert required == {
        path.name for path in root.iterdir()
    }
    with open(
        root / "sample_admission_diagnostic.csv",
        newline="",
        encoding="utf-8",
    ) as input_file:
        assert len(list(csv.DictReader(input_file))) == 210
    with open(
        root / "margin_quartile_summary.csv",
        newline="",
        encoding="utf-8",
    ) as input_file:
        assert len(list(csv.DictReader(input_file))) == 4
    with open(
        root / "formal_shuffled_admission_diagnostics.csv",
        newline="",
        encoding="utf-8",
    ) as input_file:
        assert len(list(csv.DictReader(input_file))) == 200


def test_metadata_declares_read_only_diagnostic_boundaries(
    completed_audit,
):
    metadata = json.loads(
        (
            completed_audit["output_root"] / "metadata.json"
        ).read_text()
    )
    for key in (
        "labels_loaded",
        "clustering_labels_used",
        "model_loaded",
        "projector_training_performed",
        "optimizer_created",
        "backward_performed",
        "kmeans_fit_performed",
        "parameter_updates",
        "wq1a1_formal_conclusion_changed",
        "wq1a1_rescue_claimed",
        "multiseed_conclusion_claimed",
    ):
        assert metadata[key] is False
    assert metadata[
        "B6_WQ1A2_PROTECTED_INPUTS_UNCHANGED_PASS"
    ]


def test_decision_flags_follow_preregistered_formulas(
    completed_audit,
):
    summary = completed_audit["summary"]
    margin_expected = bool(
        summary["margin34_error_auc"] > 0.65
        and summary["median_margin34_error"]
        < summary["median_margin34_allclean"]
        and summary["margin34_error_auc_ci95"][0] > 0.5
    )
    purity_expected = bool(
        summary["rho_clean_fraction_acc"] > 0.0
        and summary["p_clean_fraction_acc"] < 0.05
    )
    assert (
        summary["B6_WQ1A2_MARGIN_INFORMATIVE"]
        == margin_expected
    )
    assert (
        summary["B6_WQ1A2_PURITY_PERFORMANCE_LINK"]
        == purity_expected
    )
    assert summary[
        "B6_WQ1A2_UNCERTAINTY_ACTION_ELIGIBLE"
    ] == bool(margin_expected and purity_expected)
    assert not summary["frozen_wq1a1_formal_null_pass"]
    assert not summary["frozen_wq1a1_formal_seed20_pass"]
    assert not summary[
        "frozen_wq1a1_multiseed_entry_eligible"
    ]


def test_only_seed20_is_supported(tmp_path):
    with pytest.raises(
        RuntimeError,
        match="supports only seed20",
    ):
        audit.run_audit(seed=30, output_dir=tmp_path)
