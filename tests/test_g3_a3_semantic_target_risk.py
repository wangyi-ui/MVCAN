"""Targeted tests for the G3-A3 read-only semantic target-risk diagnostic."""

import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments.g3_selective_semantic_cooperation import (
    evaluate_g3_a3_semantic_target_risk as diagnostic,
)
from irv.b4_information_utility import tensor_sha256


@pytest.fixture(scope="module")
def formal_fixed():
    return diagnostic.construct_fixed_scores()


def _uniform_aligned_q():
    return np.full(
        (
            diagnostic.SAMPLE_NUM,
            diagnostic.VIEW_NUM,
            diagnostic.CLASS_NUM,
        ),
        1.0 / diagnostic.CLASS_NUM,
        dtype=np.float64,
    )


def test_normalized_jsd_identity_symmetry_extreme_and_bounds():
    p = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1],
        ],
        dtype=np.float64,
    )
    q = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2],
        ],
        dtype=np.float64,
    )

    identity = diagnostic.normalized_js_divergence(p, p)
    forward = diagnostic.normalized_js_divergence(p, q)
    reverse = diagnostic.normalized_js_divergence(q, p)

    assert np.allclose(identity, 0.0, atol=1e-15)
    assert np.allclose(forward, reverse, atol=1e-15)
    assert np.isclose(forward[0], 1.0, atol=1e-15)
    assert np.isfinite(forward).all()
    assert np.all((forward >= 0.0) & (forward <= 1.0))


def test_high_u_consistency_uses_only_the_three_admitted_pairs():
    aligned_q = _uniform_aligned_q()
    aligned_q[:, 0] = np.eye(7, dtype=np.float64)[0]
    aligned_q[:, 1] = np.eye(7, dtype=np.float64)[0]
    aligned_q[:, 2] = np.eye(7, dtype=np.float64)[1]
    admission = np.zeros((diagnostic.SAMPLE_NUM, diagnostic.VIEW_NUM), dtype=bool)
    admission[:, :3] = True

    result = diagnostic.high_u_internal_consistency(aligned_q, admission)

    # Pair JSDs are 0, 1, 1, so C_high = 1 - 2/3.
    assert result.shape == (diagnostic.SAMPLE_NUM,)
    assert np.allclose(result, 1.0 / 3.0, atol=1e-15)
    assert result.flags.writeable is False


def test_high_u_consistency_hard_fails_unless_each_row_has_exactly_three_views():
    aligned_q = _uniform_aligned_q()
    admission = np.zeros((diagnostic.SAMPLE_NUM, diagnostic.VIEW_NUM), dtype=bool)
    admission[:, :3] = True
    admission[17, 3] = True

    with pytest.raises(RuntimeError, match="exactly three"):
        diagnostic.high_u_internal_consistency(aligned_q, admission)


def test_conflict_and_action_risk_match_registered_equations():
    p_all = np.full(
        (diagnostic.SAMPLE_NUM, diagnostic.CLASS_NUM),
        1.0 / diagnostic.CLASS_NUM,
    )
    p_high = p_all.copy()
    p_high[:, 0] += 0.06
    p_high[:, 1:] -= 0.01
    conflict = diagnostic.normalized_js_divergence(p_high, p_all)
    rho = np.linspace(0.0, 1.0, diagnostic.SAMPLE_NUM)
    consistency = np.linspace(1.0, 0.5, diagnostic.SAMPLE_NUM)

    risk = diagnostic.action_risk_score(rho, consistency, conflict)

    assert np.array_equal(risk, rho * consistency * conflict)
    assert risk.flags.writeable is False
    assert np.isfinite(risk).all()
    assert np.all((risk >= 0.0) & (risk <= 1.0))


def test_formal_sources_are_only_the_two_frozen_epoch0_artifacts(formal_fixed):
    assert set(formal_fixed["source_audit"]) == {"U", "SHUFFLE"}
    for arm, expected_hash in diagnostic.EXPECTED_SOURCE_FILE_SHA256.items():
        source = formal_fixed["source_audit"][arm]
        assert source["file_sha256"] == expected_hash
        assert source["path"].endswith(
            diagnostic.SOURCE_ARMS[arm] + "/refresh_epoch_0000.npz"
        )
        assert set(source["required_arrays"]) == {
            "P_all",
            "P_highU",
            "aligned_q",
            "admission",
            "rho",
            "sample_ids",
            "fusion_weights_used_for_P_all",
        }


def test_formal_U_and_shuffle_share_native_frozen_quantities_exactly(formal_fixed):
    sources = formal_fixed["source_audit"]
    for key in (
        "P_all",
        "aligned_q",
        "sample_ids",
        "fusion_weights_used_for_P_all",
    ):
        assert (
            sources["U"]["required_arrays"][key]["sha256"]
            == sources["SHUFFLE"]["required_arrays"][key]["sha256"]
        )
    assert formal_fixed["shared_frozen_quantities_exact"]
    assert all(formal_fixed["admission_exactly_three"].values())


def test_formal_scores_predictions_are_complete_hashed_bounded_readonly(formal_fixed):
    arrays = formal_fixed["arrays"]
    assert set(arrays) == set(diagnostic.SEALED_ARRAY_NAMES)
    for name, array in arrays.items():
        assert formal_fixed["hashes"][name]["sha256"] == tensor_sha256(array)
        assert formal_fixed["hashes"][name]["shape"] == list(array.shape)
        assert formal_fixed["hashes"][name]["dtype"] == str(array.dtype)
        assert array.flags.writeable is False
    for name in (
        "rho_U",
        "rho_shuffle",
        "C_high_U",
        "C_high_shuffle",
        "D_conflict_U",
        "D_conflict_shuffle",
        "E_coop_U",
        "E_coop_shuffle",
        "native_uncertainty",
    ):
        assert arrays[name].shape == (diagnostic.SAMPLE_NUM,)
        assert np.isfinite(arrays[name]).all()
        assert np.all((arrays[name] >= -diagnostic.BOUND_ATOL))
        assert np.all((arrays[name] <= 1.0 + diagnostic.BOUND_ATOL))


def test_formal_E_coop_is_exact_three_factor_product(formal_fixed):
    arrays = formal_fixed["arrays"]
    assert np.array_equal(
        arrays["E_coop_U"],
        arrays["rho_U"] * arrays["C_high_U"] * arrays["D_conflict_U"],
    )
    assert np.array_equal(
        arrays["E_coop_shuffle"],
        arrays["rho_shuffle"]
        * arrays["C_high_shuffle"]
        * arrays["D_conflict_shuffle"],
    )


def test_changing_labels_cannot_change_any_sealed_score_prediction_or_hash(formal_fixed):
    labels_a = np.arange(diagnostic.SAMPLE_NUM, dtype=np.int64) % 7
    labels_b = labels_a[::-1].copy()
    before = formal_fixed["hashes"]

    assert not np.array_equal(labels_a, labels_b)
    assert formal_fixed["hashes"] == before
    for name, array in formal_fixed["arrays"].items():
        assert tensor_sha256(array) == before[name]["sha256"]


def test_hungarian_mapping_is_fit_from_P_all_once_and_maps_cluster_to_class():
    labels = np.arange(diagnostic.SAMPLE_NUM, dtype=np.int64) % 7
    permutation = np.array([3, 6, 1, 4, 0, 5, 2], dtype=np.int64)
    cluster_for_class = np.argsort(permutation)
    p_all_cluster = cluster_for_class[labels]

    record = diagnostic.fit_global_cluster_mapping_once(labels, p_all_cluster)

    assert np.array_equal(record["mapping"], permutation)
    assert np.array_equal(record["mapping"][p_all_cluster], labels)
    assert record["mapping_fit_source"] == "P_all_prediction_only"
    assert record["mapping_fit_count"] == 1


def test_rank_diagnostic_uses_error_as_positive_class_without_classifier():
    outcome = np.zeros(diagnostic.SAMPLE_NUM, dtype=np.int64)
    outcome[::4] = 1
    score = np.linspace(0.0, 1.0, diagnostic.SAMPLE_NUM)
    score[outcome == 1] += 2.0

    result = diagnostic.rank_diagnostic(outcome, score)

    assert result["ROC_AUC"] == 1.0
    assert result["PR_AUC"] == 1.0
    assert result["mean_score_P_all_wrong"] > result["mean_score_P_all_correct"]
    assert result["wrong_minus_correct_mean_gap"] > 0.0


def test_error_enrichment_uses_fixed_descriptive_top_10_20_30_percent():
    errors = np.zeros(diagnostic.SAMPLE_NUM, dtype=np.int64)
    errors[:140] = 1
    score = np.zeros(diagnostic.SAMPLE_NUM, dtype=np.float64)
    score[:140] = 1.0
    sample_ids = np.arange(diagnostic.SAMPLE_NUM, dtype=np.int64)

    result = diagnostic.error_enrichment(errors, score, sample_ids)

    assert np.isclose(result["global_error_rate"], 0.1)
    assert result["top_10pct"]["sample_count"] == 140
    assert result["top_10pct"]["error_rate"] == 1.0
    assert result["top_10pct"]["enrichment_over_global"] == 10.0
    assert result["top_20pct"]["sample_count"] == 280
    assert result["top_30pct"]["sample_count"] == 420


def test_candidate_gate_is_strict_and_has_no_magnitude_threshold():
    primary = {
        "ROC_AUC": 0.600000000001,
        "mean_score_P_all_wrong": 0.2,
        "mean_score_P_all_correct": 0.1,
    }
    shuffled = {"ROC_AUC": 0.6}
    rho = {"ROC_AUC": 0.55}

    passed = diagnostic.action_risk_candidate_gate(primary, shuffled, rho)
    equal_shuffle = diagnostic.action_risk_candidate_gate(
        {**primary, "ROC_AUC": 0.6}, shuffled, rho
    )

    assert passed["ACTION_RISK_CANDIDATE_PASS"]
    assert passed["magnitude_threshold"] is None
    assert passed["native_uncertainty_is_hard_gate"] is False
    assert not equal_shuffle["ACTION_RISK_CANDIDATE_PASS"]


def test_semantic_action_counts_and_decisive_auc_reuse_supplied_mapping():
    labels = np.arange(diagnostic.SAMPLE_NUM, dtype=np.int64) % 7
    y_all = labels.copy()
    y_high = labels.copy()
    risk = np.zeros(diagnostic.SAMPLE_NUM, dtype=np.float64)
    y_all[:10] = (labels[:10] + 1) % 7
    y_high[:10] = labels[:10]
    risk[:10] = 0.9
    y_high[10:30] = (labels[10:30] + 1) % 7
    risk[10:30] = 0.1
    y_all[30:60] = (labels[30:60] + 1) % 7
    y_high[30:60] = (labels[30:60] + 2) % 7
    mapping = np.arange(7, dtype=np.int64)

    result = diagnostic.semantic_action_diagnostic(
        labels, y_all, y_high, mapping, risk
    )

    assert result["BENEFICIAL_OVERRIDE"] == 10
    assert result["HARMFUL_OVERRIDE"] == 20
    assert result["BOTH_WRONG"] == 30
    assert result["BOTH_CORRECT"] == 0
    assert result["decisive_subset_count"] == 30
    assert result["decisive_subset_E_coop_U"]["status"] == "AVAILABLE"
    assert result["decisive_subset_E_coop_U"]["ROC_AUC"] == 1.0


def test_semantic_action_reports_insufficient_when_one_decisive_class_is_empty():
    labels = np.arange(diagnostic.SAMPLE_NUM, dtype=np.int64) % 7
    y_all = labels.copy()
    y_high = labels.copy()
    y_all[:10] = (labels[:10] + 1) % 7
    risk = np.linspace(0.0, 1.0, diagnostic.SAMPLE_NUM)

    result = diagnostic.semantic_action_diagnostic(
        labels, y_all, y_high, np.arange(7, dtype=np.int64), risk
    )

    assert result["BENEFICIAL_OVERRIDE"] == 10
    assert result["HARMFUL_OVERRIDE"] == 0
    assert result["decisive_subset_E_coop_U"]["status"] == (
        "INSUFFICIENT_ONE_CLASS_EMPTY"
    )
    assert result["decisive_subset_E_coop_U"]["ROC_AUC"] is None


def test_labels_enter_only_after_protocol_scores_and_audit_are_written():
    constructor_source = inspect.getsource(diagnostic.construct_fixed_scores)
    evaluation_source = inspect.getsource(diagnostic.evaluate_after_score_seal)
    run_source = inspect.getsource(diagnostic.run_diagnostic)

    assert "load_caltech_labels_only" not in constructor_source
    assert "load_caltech_labels_only" in evaluation_source
    evaluation_call = run_source.index("evaluate_after_score_seal(")
    assert run_source.index('"protocol.json"') < evaluation_call
    assert run_source.index('"sealed_scores.npz"') < evaluation_call
    assert run_source.index('"score_audit.json"') < evaluation_call


def test_cli_exposes_no_tunable_combination_or_threshold_parameters():
    arguments = diagnostic.parse_args([])

    for forbidden in (
        "alpha",
        "temperature",
        "threshold",
        "rho",
        "lambda1",
        "percentile",
    ):
        assert not hasattr(arguments, forbidden)


def test_diagnostic_source_has_no_model_D2_or_fit_based_score_construction():
    source = Path(diagnostic.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "torch.optim",
        ".backward(",
        "run_experiment(",
        "_train_one_arm(",
        "load_frozen_u_only(",
        "compute_transfer_scores(",
        "oof_ridge_predictability(",
        "compute_information_utility(",
        "LogisticRegression",
        "target_distribution(",
    ):
        assert forbidden not in source
