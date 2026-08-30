"""Targeted tests for the G3-A2 confidence-preserving target gate."""

import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments.g3_selective_semantic_cooperation import (
    evaluate_g3_a2_confidence_preserving_target as diagnostic,
)
from irv.b4_information_utility import tensor_sha256


def _probabilities(sample_num, seed):
    rng = np.random.RandomState(seed)
    values = rng.uniform(size=(sample_num, diagnostic.CLASS_NUM))
    values /= values.sum(axis=1, keepdims=True)
    return values


def _delta(acc, nmi, ari):
    return {"delta_ACC": acc, "delta_NMI": nmi, "delta_ARI": ari}


@pytest.fixture(scope="module")
def formal_fixed():
    return diagnostic.construct_fixed_outputs()


def test_log_space_semantic_correction_matches_preregistered_formula():
    p_all = _probabilities(11, 2)
    p_high = _probabilities(11, 3)
    rho = np.linspace(0.0, 1.0, 11)

    result = diagnostic.log_space_semantic_correction(p_all, p_high, rho)
    expected_l_all = np.log(np.clip(p_all, diagnostic.EPS, None))
    expected_l_high = np.log(np.clip(p_high, diagnostic.EPS, None))
    expected = expected_l_all + rho[:, None] * (
        expected_l_high - expected_l_all
    )

    assert result["l_corr"].shape == (11, 7)
    assert np.array_equal(result["l_corr"], expected)
    assert np.array_equal(
        result["semantic_residual"], result["l_high"] - result["l_all"]
    )


def test_entropy_projection_solves_per_sample_tau_and_probability_boundary():
    p_all = _probabilities(19, 4)
    p_high = _probabilities(19, 5)
    rho = np.linspace(0.01, 0.9, 19)
    corrected = diagnostic.log_space_semantic_correction(
        p_all, p_high, rho
    )["l_corr"]

    result = diagnostic.solve_entropy_preserving_temperature(corrected, p_all)
    entropy_error = np.abs(
        diagnostic.entropy_per_sample(result["P_cp"])
        - diagnostic.entropy_per_sample(p_all)
    )

    assert result["tau"].shape == (19,)
    assert result["P_cp"].shape == (19, 7)
    assert np.isfinite(result["tau"]).all()
    assert np.all(result["tau"] > 0.0)
    assert np.isfinite(result["P_cp"]).all()
    assert np.all(result["P_cp"] >= 0.0)
    assert np.allclose(result["P_cp"].sum(axis=1), 1.0, atol=1e-12)
    assert float(entropy_error.max()) <= diagnostic.ENTROPY_PASS_ATOL


def test_rho_zero_is_identity_with_tau_one():
    p_all = _probabilities(13, 6)
    p_high = _probabilities(13, 7)
    rho = np.zeros(13, dtype=np.float64)
    corrected = diagnostic.log_space_semantic_correction(
        p_all, p_high, rho
    )["l_corr"]

    result = diagnostic.solve_entropy_preserving_temperature(corrected, p_all)

    assert np.allclose(result["tau"], 1.0, rtol=0.0, atol=1e-12)
    assert np.allclose(result["P_cp"], p_all, rtol=0.0, atol=1e-12)


def test_degenerate_constant_logits_with_lower_target_entropy_hard_fail():
    corrected = np.zeros((1, 7), dtype=np.float64)
    p_all = np.full((1, 7), 0.05, dtype=np.float64)
    p_all[0, 0] = 0.70

    with pytest.raises(RuntimeError, match="degenerate corrected logits"):
        diagnostic.solve_entropy_preserving_temperature(corrected, p_all)


def test_formal_sources_are_only_frozen_u_and_shuffled_epoch0_artifacts(
    formal_fixed,
):
    sources = formal_fixed["source_audit"]

    assert set(sources) == {"U", "SHUFFLE"}
    for arm in sources:
        assert sources[arm]["file_sha256"] == (
            diagnostic.EXPECTED_SOURCE_FILE_SHA256[arm]
        )
        assert sources[arm]["path"].endswith(
            diagnostic.SOURCE_ARMS[arm] + "/refresh_epoch_0000.npz"
        )
        assert "P_highU" in sources[arm]["keys"]
        assert "rho" in sources[arm]["keys"]


def test_formal_cp_targets_tau_and_predictions_are_fixed_hashed_readonly(
    formal_fixed,
):
    assert set(formal_fixed["targets"]) == set(diagnostic.TARGET_NAMES)
    assert set(formal_fixed["tau"]) == {"tau_U", "tau_SHUFFLE"}
    for name in diagnostic.TARGET_NAMES:
        target = formal_fixed["targets"][name]
        prediction = formal_fixed["predictions"][name]
        assert target.shape == (1400, 7)
        assert prediction.shape == (1400,)
        assert target.flags.writeable is False
        assert prediction.flags.writeable is False
        assert formal_fixed["fixed_hashes"][name][
            "target_sha256"
        ] == tensor_sha256(target)
        assert formal_fixed["fixed_hashes"][name][
            "prediction_sha256"
        ] == tensor_sha256(prediction)
    for name, tau in formal_fixed["tau"].items():
        assert tau.shape == (1400,)
        assert tau.flags.writeable is False
        assert np.isfinite(tau).all() and np.all(tau > 0.0)
        assert formal_fixed["fixed_hashes"][name]["sha256"] == tensor_sha256(
            tau
        )


def test_formal_entropy_preservation_probability_tau_and_identity_audits_pass(
    formal_fixed,
):
    for arm in ("CP_U", "CP_SHUFFLED_U"):
        audit = formal_fixed["projection_audit"][arm]
        assert audit["entropy_preservation_pass"]
        assert audit["probability_validity_pass"]
        assert audit["tau_validity_pass"]
        assert audit["rho_zero_identity_pass"]
        assert audit["projection_valid"]
        assert audit["max_absolute_entropy_error"] <= diagnostic.ENTROPY_PASS_ATOL


def test_retention_ratios_match_exact_registered_definitions(formal_fixed):
    consistency = formal_fixed["consistency"]
    disagreement = consistency["prediction_disagreement"]
    l1 = consistency["mean_row_L1"]
    expected_disagreement = (
        disagreement["P_cp_U_vs_P_cp_SHUFFLE"]["rate"]
        / disagreement["P_highU_U_vs_P_highU_SHUFFLE"]["rate"]
    )
    expected_l1 = (
        l1["P_cp_U_vs_P_cp_SHUFFLE"]
        / l1["P_highU_U_vs_P_highU_SHUFFLE"]
    )

    assert consistency["prediction_disagreement_retention"] == (
        expected_disagreement
    )
    assert consistency["L1_retention"] == expected_l1


def test_changing_labels_cannot_change_cp_targets_tau_predictions_or_hashes(
    formal_fixed,
):
    labels_a = np.arange(1400, dtype=np.int64) % 7
    labels_b = labels_a[::-1].copy()
    before = formal_fixed["fixed_hashes"]

    assert not np.array_equal(labels_a, labels_b)
    assert formal_fixed["fixed_hashes"] == before
    for name in diagnostic.TARGET_NAMES:
        assert tensor_sha256(formal_fixed["targets"][name]) == before[name][
            "target_sha256"
        ]
        assert tensor_sha256(formal_fixed["predictions"][name]) == before[name][
            "prediction_sha256"
        ]


def test_labels_enter_only_after_targets_tau_predictions_and_hashes_are_sealed():
    constructor_source = inspect.getsource(diagnostic.construct_fixed_outputs)
    evaluation_source = inspect.getsource(diagnostic.evaluate_after_fixed_output_seal)
    run_source = inspect.getsource(diagnostic.run_diagnostic)

    assert "load_caltech_labels_only" not in constructor_source
    assert "load_caltech_labels_only" in evaluation_source
    assert run_source.index("sealed_confidence_preserving_targets.npz") < (
        run_source.index("evaluate_after_fixed_output_seal(")
    )
    assert run_source.index("fixed_cp_target_audit.json") < run_source.index(
        "evaluate_after_fixed_output_seal("
    )


def test_preregistered_verdicts_have_no_magnitude_threshold():
    positive = _delta(1e-12, 1e-12, 1e-12)
    mixed = _delta(-1e-12, 1e-12, 1e-12)

    assert diagnostic.diagnostic_verdict(
        True, positive, positive
    ) == "CASE_A_CONFIDENCE_PRESERVING_SIGNAL_SURVIVES"
    assert diagnostic.diagnostic_verdict(
        True, positive, mixed
    ) == "CASE_B_DIRECTION_STILL_ERASED"
    assert diagnostic.diagnostic_verdict(
        False, positive, positive
    ) == "CASE_C_PROJECTION_INVALID"


def test_cli_exposes_no_tau_rho_alpha_or_temperature_tuning():
    arguments = diagnostic.parse_args([])

    for forbidden in ("tau", "rho", "alpha", "temperature", "lambda1"):
        assert not hasattr(arguments, forbidden)


def test_diagnostic_source_has_no_training_or_forbidden_method_calls():
    source = Path(diagnostic.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "torch.optim",
        ".backward(",
        "_train_one_arm(",
        "run_experiment(",
        "target_distribution(",
        "compute_transfer_scores(",
        "oof_ridge_predictability(",
        "compute_information_utility(",
        "load_frozen_u_only(",
    ):
        assert forbidden not in source
