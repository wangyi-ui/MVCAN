"""Targeted tests for the G3-A1 read-only target-interface diagnostic."""

import inspect
from pathlib import Path

import numpy as np

from experiments.g3_selective_semantic_cooperation import (
    evaluate_g3_a1_readonly_target_interface as diagnostic,
)
from irv.b4_information_utility import tensor_sha256


def _probability_target(seed=1):
    rng = np.random.RandomState(seed)
    values = rng.uniform(
        size=(diagnostic.SAMPLE_NUM, diagnostic.CLASS_NUM)
    ).astype(np.float32)
    values /= values.sum(axis=1, keepdims=True)
    return values


def _delta(acc, nmi, ari):
    return {"delta_ACC": acc, "delta_NMI": nmi, "delta_ARI": ari}


def test_formal_epoch0_archives_have_reported_keys_shapes_dtypes_and_hashes():
    expected_file_hashes = {
        "BASE": "b5271f5ea873518b72a192c9631d1bbc846bf4b4c49ca4e14299f470aa448a73",
        "U_CORRECTION": "be6763f6f6b916a25a44cb0e1d796f370c04587d4f8af94e0963326fdfce7d0c",
        "SHUFFLED_U_CORRECTION": "d09b5bf7e3104deeb10cc094d4e3771478792c75932fb74b22d7eee48853351a",
    }
    for arm, directory in diagnostic.ARM_DIRECTORIES.items():
        loaded = diagnostic.load_epoch0_archive(
            diagnostic.DEFAULT_G3_RUN_DIR
            / directory
            / "refresh_epoch_0000.npz"
        )

        assert set(loaded["keys"]) == set(diagnostic.EXPECTED_ARCHIVE_KEYS)
        assert loaded["file_sha256"] == expected_file_hashes[arm]
        for key, (shape, dtype) in diagnostic.EXPECTED_ARRAY_BOUNDARIES.items():
            assert loaded["arrays"][key].shape == shape
            assert loaded["arrays"][key].dtype == dtype
            assert loaded["schema"][key]["sha256"] == tensor_sha256(
                loaded["arrays"][key]
            )
            assert loaded["arrays"][key].flags.writeable is False


def test_fixed_target_set_is_complete_N7_hashed_and_readonly():
    fixed = diagnostic.construct_fixed_target_outputs()

    assert set(fixed["targets"]) == set(diagnostic.TARGET_NAMES)
    assert set(fixed["predictions"]) == set(diagnostic.TARGET_NAMES)
    assert not any(fixed["reconstruction_audit"].values())
    for name in diagnostic.TARGET_NAMES:
        target = fixed["targets"][name]
        prediction = fixed["predictions"][name]
        assert target.shape == (1400, 7)
        assert prediction.shape == (1400,)
        assert target.flags.writeable is False
        assert prediction.flags.writeable is False
        assert fixed["fixed_hashes"][name]["target_sha256"] == tensor_sha256(
            target
        )
        assert fixed["fixed_hashes"][name][
            "prediction_sha256"
        ] == tensor_sha256(prediction)


def test_native_epoch0_quantities_and_base_target_are_exactly_shared():
    fixed = diagnostic.construct_fixed_target_outputs()
    schemas = fixed["artifact_schema"]

    for key in (
        "sample_ids",
        "P_all",
        "M",
        "q_detached",
        "aligned_q",
        "fusion_weights_used_for_P_all",
        "next_fusion_weights",
    ):
        hashes = {schemas[arm]["arrays"][key]["sha256"] for arm in schemas}
        assert len(hashes) == 1
    assert tensor_sha256(fixed["targets"]["P_all"]) == (
        schemas["BASE"]["arrays"]["P_util"]["sha256"]
    )


def test_saved_p_util_is_exact_original_convex_formula():
    fixed = diagnostic.construct_fixed_target_outputs()
    for arm, high_name, util_name in (
        ("U_CORRECTION", "P_highU_U", "P_util_U"),
        (
            "SHUFFLED_U_CORRECTION",
            "P_highU_SHUFFLE",
            "P_util_SHUFFLE",
        ),
    ):
        directory = diagnostic.ARM_DIRECTORIES[arm]
        archive = diagnostic.load_epoch0_archive(
            diagnostic.DEFAULT_G3_RUN_DIR
            / directory
            / "refresh_epoch_0000.npz"
        )
        expected = diagnostic.reconstruct_p_util_from_saved_quantities(
            archive["arrays"], fixed["targets"][high_name]
        )

        assert np.array_equal(fixed["targets"][util_name], expected)


def test_high_u_fallback_uses_only_saved_alignment_admission_and_weights():
    aligned_q = np.zeros((1400, 6, 7), dtype=np.float32)
    for view_id in range(6):
        aligned_q[:, view_id, view_id] = 1.0
    admission = np.zeros((1400, 6), dtype=bool)
    admission[:, [0, 2, 5]] = True
    saved = {
        "aligned_q": aligned_q,
        "admission": admission,
        "fusion_weights_used_for_P_all": np.arange(1, 7, dtype=np.float64),
    }

    result = diagnostic.reconstruct_high_u_from_saved_quantities(saved)

    assert result.shape == (1400, 7)
    assert np.allclose(result.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(result[:, 0], 1.0 / 10.0)
    assert np.allclose(result[:, 2], 3.0 / 10.0)
    assert np.allclose(result[:, 5], 6.0 / 10.0)
    assert result.flags.writeable is False


def test_score_diagnostics_match_entropy_confidence_and_margin_definitions():
    target = np.zeros((1400, 7), dtype=np.float32)
    target[:, 0] = 0.6
    target[:, 1] = 0.4

    result = diagnostic.score_diagnostics(target)

    expected_entropy = -(0.6 * np.log(0.6) + 0.4 * np.log(0.4))
    assert np.isclose(result["entropy_mean"], expected_entropy)
    assert np.isclose(result["max_confidence_mean"], 0.6)
    assert np.isclose(result["top1_top2_margin_mean"], 0.2)


def test_disagreement_l1_and_argmax_change_definitions():
    first = np.arange(1400, dtype=np.int64) % 7
    second = first.copy()
    second[:140] = (second[:140] + 1) % 7
    target_a = np.eye(7, dtype=np.float32)[first]
    target_b = np.eye(7, dtype=np.float32)[second]

    disagreement = diagnostic.prediction_disagreement(first, second)

    assert disagreement["count"] == 140
    assert np.isclose(disagreement["rate"], 0.1)
    assert np.isclose(diagnostic.mean_row_l1(target_a, target_b), 0.2)


def test_changing_labels_cannot_change_sealed_targets_predictions_or_hashes():
    fixed = diagnostic.construct_fixed_target_outputs()
    before = fixed["fixed_hashes"]
    labels_a = np.arange(1400, dtype=np.int64) % 7
    labels_b = labels_a[::-1].copy()

    assert not np.array_equal(labels_a, labels_b)
    assert fixed["fixed_hashes"] == before
    for name in diagnostic.TARGET_NAMES:
        assert tensor_sha256(fixed["targets"][name]) == before[name][
            "target_sha256"
        ]
        assert tensor_sha256(fixed["predictions"][name]) == before[name][
            "prediction_sha256"
        ]


def test_labels_enter_only_in_post_seal_evaluation_function():
    constructor_source = inspect.getsource(diagnostic.construct_fixed_target_outputs)
    evaluation_source = inspect.getsource(
        diagnostic.evaluate_after_fixed_output_seal
    )
    run_source = inspect.getsource(diagnostic.run_diagnostic)

    assert "load_caltech_labels_only" not in constructor_source
    assert "load_caltech_labels_only" in evaluation_source
    assert run_source.index("sealed_targets_predictions.npz") < run_source.index(
        "evaluate_after_fixed_output_seal("
    )
    assert run_source.index("fixed_target_audit.json") < run_source.index(
        "evaluate_after_fixed_output_seal("
    )


def test_verdict_cases_use_strict_three_metric_dominance_without_threshold():
    positive = _delta(0.01, 0.02, 0.03)
    negative = _delta(-0.01, -0.02, -0.03)

    assert diagnostic.diagnostic_verdict(
        {
            "P_highU_U_minus_P_highU_SHUFFLE": positive,
            "P_util_U_minus_P_util_SHUFFLE": positive,
        },
        negative,
    ) == "CASE_A_TARGET_SIGNAL_SURVIVES_BUT_TRAINING_LOSES_IT"
    assert diagnostic.diagnostic_verdict(
        {
            "P_highU_U_minus_P_highU_SHUFFLE": positive,
            "P_util_U_minus_P_util_SHUFFLE": negative,
        },
        negative,
    ) == "CASE_B_INTERPOLATION_ERASES_UTILITY_SIGNAL"
    assert diagnostic.diagnostic_verdict(
        {
            "P_highU_U_minus_P_highU_SHUFFLE": negative,
            "P_util_U_minus_P_util_SHUFFLE": positive,
        },
        negative,
    ) == "CASE_C_G3_EPOCH0_HIGHU_SIGNAL_NOT_REPRODUCED"


def test_entropy_mismatch_audit_requires_all_three_desharpening_signs():
    diagnostics = {
        "P_all": {
            "entropy_mean": 1.0,
            "max_confidence_mean": 0.8,
            "top1_top2_margin_mean": 0.5,
        },
        "P_highU_U": {
            "entropy_mean": 1.2,
            "max_confidence_mean": 0.6,
            "top1_top2_margin_mean": 0.3,
        },
        "P_highU_SHUFFLE": {
            "entropy_mean": 1.3,
            "max_confidence_mean": 0.5,
            "top1_top2_margin_mean": 0.2,
        },
        "P_util_U": {
            "entropy_mean": 1.1,
            "max_confidence_mean": 0.7,
            "top1_top2_margin_mean": 0.4,
        },
        "P_util_SHUFFLE": {
            "entropy_mean": 1.15,
            "max_confidence_mean": 0.65,
            "top1_top2_margin_mean": 0.35,
        },
    }

    result = diagnostic._entropy_mismatch_audit(diagnostics)

    assert result["target_desharpening_observed"]
    assert result["delta_entropy_relative_to_P_all"]["P_util_U"] > 0.0


def test_diagnostic_source_has_no_training_or_method_modification_calls():
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
    ):
        assert forbidden not in source
