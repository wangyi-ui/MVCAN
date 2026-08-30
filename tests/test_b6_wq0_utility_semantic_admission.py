"""Tests for B6-WQ0 Utility-guided Native MVCAN view admission."""

import hashlib
import inspect
import json

import numpy as np
import pytest

import ClusteringTest
import experiments.b6_weak_quality.evaluate_b6_wq0_utility_semantic_admission as b6
from irv.b3_audit import hash_backbone


SUMMARY_PATH = (
    b6.REPOSITORY_ROOT
    / "outputs/b6_weak_quality/wq0_frozen_baseline_provenance"
    / "b6_wq0_frozen_baseline_summary.json"
)


@pytest.fixture(scope="module")
def canonical_candidates():
    with open(SUMMARY_PATH, "r", encoding="utf-8") as input_file:
        return json.load(input_file)["canonical_manifest_candidate_entries"]


@pytest.fixture(scope="module")
def canonical_manifest():
    manifest, path = b6.load_canonical_manifest()
    assert path == b6.CANONICAL_MANIFEST_PATH
    return manifest


@pytest.fixture(scope="module")
def seed20_prepared():
    return b6.prepare_seed(20)


@pytest.fixture(scope="module")
def seed20_base(seed20_prepared):
    return b6.run_base(seed20_prepared)


@pytest.fixture(scope="module")
def seed20_result(seed20_prepared, seed20_base):
    base, metrics, historical_audit = seed20_base
    return b6._base_result(
        seed20_prepared,
        base,
        metrics,
        historical_audit,
    )


def toy_inputs():
    attention = np.full((210, 5), 0.2, dtype=np.float64)
    utility = np.linspace(0.0, 1.0, 1050, dtype=np.float64).reshape(210, 5)
    return attention, utility


def test_architecture_adapter_obtains_exact_original_fusion_tensor(
    seed20_prepared, seed20_base
):
    base, _, _ = seed20_base
    scaled = seed20_prepared["fusion_inputs"]["scaled_z_views"]
    scales = base["native_fusion_scales"]
    expected = np.hstack([scaled[v] * scales[v] for v in range(5)])
    assert np.array_equal(base["fused_representation"], expected)
    assert base["fused_representation"].shape == (210, 50)
    audit = b6.architecture_audit()
    assert audit["sample_specific_native_attention_exists"] is False
    assert audit["native_view_fusion"]["exists"]


def test_base_path_equals_original_mvcan_path(seed20_prepared, seed20_base):
    first, _, _ = seed20_base
    inputs = seed20_prepared["fusion_inputs"]
    second = b6.original_mvcan_fusion_adapter(
        inputs["scaled_z_views"],
        inputs["view_assignments"],
        20,
        seed20_prepared["fusion_updates"],
    )
    assert np.array_equal(first["fused_representation"], second["fused_representation"])
    assert np.array_equal(first["predictions"], second["predictions"])
    assert np.array_equal(first["original_attention"], second["original_attention"])


def test_seed20_frozen_replay_infeasibility_is_reported_without_gate_relaxation(
    seed20_base,
):
    base, metrics, reproduction = seed20_base
    assert metrics != b6.SEED_PROVENANCE[20]["base_metrics"]
    assert reproduction["tolerance"] == 1e-12
    assert not reproduction["B6_WQ0_BASE_REPRO_PASS"]
    assert not base["B6_WQ0_FROZEN_B2_EXACT_REPLAY_FEASIBLE"]


def test_prefusion_assignment_divergence_and_temporary_q_verification(
    seed20_prepared,
):
    inputs = seed20_prepared["fusion_inputs"]
    rows = inputs["per_view_audit"]
    assert len(rows) == 5
    assert not inputs["B6_WQ0_PREFUSION_ASSIGNMENT_MATCH"]
    assert all(row["initialized_q_matches_kmeans"] for row in rows)
    assert all(not row["assignment_exact_match"] for row in rows)
    assert all(row["assignment_disagreement_fraction"] > 0.0 for row in rows)
    for actual, expected in zip(
        inputs["view_assignments"],
        inputs["original_style_view_assignments"],
    ):
        assert np.array_equal(actual, expected)


def test_global_and_fresh_minmax_scalers_match_exactly(seed20_prepared):
    rows = seed20_prepared["fusion_inputs"]["per_view_audit"]
    assert all(row["scaled_z_exact_match"] for row in rows)
    assert all(row["scaled_z_max_abs_error"] == 0.0 for row in rows)
    assert all(
        row["original_style_scaled_z_sha256"]
        == row["current_adapter_scaled_z_sha256"]
        for row in rows
    )


def test_config_t1_and_final_preupdate_www_are_replayed(seed20_prepared, seed20_base):
    base, _, _ = seed20_base
    assert seed20_prepared["fusion_updates"] == b6.get_default_config(
        b6.DATASET_NAME
    )["training"]["T_1"]
    updates = base["fusion_update_audit"]
    assert len(updates) == seed20_prepared["fusion_updates"]
    assert updates[0]["www_before"] == [1.0] * 5
    assert updates[1]["www_before"] == updates[0]["www_after"]
    assert base["final_www_used_for_historical_fusion"] == updates[-1]["www_before"]


def test_identity_admission_preserves_geometry_predictions_and_metrics(seed20_base):
    base, _, _ = seed20_base
    identity = base["identity_admission_audit"]
    assert identity["positive_global_scalar"] > 0.0
    assert identity["relative_view_scale_preserved"]
    assert (
        identity["geometry_max_abs_error_after_rescaling"]
        <= identity["geometry_atol"]
    )
    assert identity["geometry_within_tolerance"]
    assert identity["assignment_exact_match"]
    assert identity["metrics_exact_match"]
    assert identity["B6_WQ0_IDENTITY_ADMISSION_PASS"]


def test_identity_geometry_tolerance_comes_from_float32_epsilon(seed20_base):
    base, _, _ = seed20_base
    identity = base["identity_admission_audit"]
    expected_epsilon = float(np.finfo(np.float32).eps)
    expected_atol = (
        16.0 * expected_epsilon * identity["geometry_scale"]
    )
    assert identity["geometry_dtype"] == "float32"
    assert identity["geometry_machine_epsilon"] == expected_epsilon
    assert identity["geometry_atol"] == expected_atol
    assert identity["geometry_atol"] != 1e-6


def test_historical_reference_is_unchanged_and_separate_from_canonical_gate(
    seed20_result,
):
    expected = {
        "acc": 0.6904761904761905,
        "nmi": 0.6468195301776315,
        "ari": 0.5374688673991558,
    }
    assert b6.SEED_PROVENANCE[20]["base_metrics"] == expected
    assert seed20_result["historical_b2_metrics"] == expected
    assert (
        seed20_result["historical_reference_kind"]
        == "B2-training-loop-transient"
    )
    assert not seed20_result["B6_WQ0_HISTORICAL_B2_REPRO_PASS"]
    assert seed20_result["canonical_base_reproduction"][
        "canonical_baseline_registered"
    ]
    assert seed20_result["B6_WQ0_CANONICAL_BASE_REPRO_PASS"]
    assert seed20_result["B6_WQ0_IDENTITY_ADMISSION_PASS"]
    assert seed20_result["B6_WQ0_BACKBONE_FROZEN_PASS"]
    assert seed20_result["B6_WQ0_SCIENTIFIC_ACTION_GATE_PASS"]
    assert not seed20_result["B6_WQ0_SCIENTIFIC_ACTION_BLOCKED"]
    assert seed20_result["scientific_action_block_reason"] is None


def test_formal_manifest_exactly_registers_provenance_candidates(
    canonical_manifest,
    canonical_candidates,
):
    entries = canonical_manifest["canonical_entries"]
    assert canonical_manifest["registration_status"] == "registered"
    assert set(entries) == {"20", "30", "50"}
    assert entries == canonical_candidates
    required = {
        "model_seed",
        "backbone_dir",
        "expected_backbone_hash",
        "expected_normalized_z_hash",
        "expected_canonical_fused_hash",
        "expected_prediction_hash",
        "expected_canonical_metrics",
        "source_provenance",
    }
    for seed, entry in entries.items():
        assert set(entry) == required
        assert entry["model_seed"] == int(seed)
        assert set(entry["expected_canonical_metrics"]) == {
            "acc",
            "nmi",
            "ari",
        }
        assert all(
            len(entry[name]) == 64
            for name in (
                "expected_backbone_hash",
                "expected_normalized_z_hash",
                "expected_canonical_fused_hash",
                "expected_prediction_hash",
            )
        )
        assert b6._resolve(entry["source_provenance"]).is_file()


def canonical_audit_for_entry(entry, **overrides):
    actual = {
        "actual_metrics": dict(entry["expected_canonical_metrics"]),
        "actual_backbone_hash": entry["expected_backbone_hash"],
        "actual_normalized_z_hash": entry["expected_normalized_z_hash"],
        "actual_fused_hash": entry["expected_canonical_fused_hash"],
        "actual_prediction_hash": entry["expected_prediction_hash"],
    }
    actual.update(overrides)
    return b6.canonical_base_reproduction_audit(
        seed=entry["model_seed"],
        **actual,
    )


def test_scientific_gate_accepts_exact_registered_canonical_baseline(
    canonical_manifest,
):
    entry = canonical_manifest["canonical_entries"]["20"]
    audit = canonical_audit_for_entry(entry)
    assert audit["canonical_baseline_registered"]
    assert all(audit["checks"].values())
    assert audit["B6_WQ0_CANONICAL_BASE_REPRO_PASS"]
    assert b6.scientific_action_gate(
        audit["B6_WQ0_CANONICAL_BASE_REPRO_PASS"],
        identity_admission_pass=True,
        backbone_frozen_pass=True,
    )


def test_wrong_registered_fused_hash_blocks_scientific_gate(canonical_manifest):
    entry = canonical_manifest["canonical_entries"]["20"]
    audit = canonical_audit_for_entry(
        entry,
        actual_fused_hash="0" * 64,
    )
    assert not audit["checks"]["canonical_fused_hash_exact"]
    assert not audit["B6_WQ0_CANONICAL_BASE_REPRO_PASS"]
    assert not b6.scientific_action_gate(
        audit["B6_WQ0_CANONICAL_BASE_REPRO_PASS"],
        identity_admission_pass=True,
        backbone_frozen_pass=True,
    )


def test_wrong_registered_prediction_hash_blocks_scientific_gate(
    canonical_manifest,
):
    entry = canonical_manifest["canonical_entries"]["20"]
    audit = canonical_audit_for_entry(
        entry,
        actual_prediction_hash="0" * 64,
    )
    assert not audit["checks"]["prediction_hash_exact"]
    assert not audit["B6_WQ0_CANONICAL_BASE_REPRO_PASS"]
    assert not b6.scientific_action_gate(
        audit["B6_WQ0_CANONICAL_BASE_REPRO_PASS"],
        identity_admission_pass=True,
        backbone_frozen_pass=True,
    )


def test_historical_mismatch_is_not_an_input_to_scientific_gate(
    seed20_base,
    canonical_manifest,
):
    _, _, historical_audit = seed20_base
    entry = canonical_manifest["canonical_entries"]["20"]
    canonical_audit = canonical_audit_for_entry(entry)
    assert not historical_audit["B6_WQ0_BASE_REPRO_PASS"]
    assert canonical_audit["B6_WQ0_CANONICAL_BASE_REPRO_PASS"]
    assert "historical" not in inspect.signature(
        b6.scientific_action_gate
    ).parameters
    assert b6.scientific_action_gate(
        canonical_base_repro_pass=True,
        identity_admission_pass=True,
        backbone_frozen_pass=True,
    )


def test_first_divergence_and_historical_timing_are_explicit(seed20_prepared, seed20_base):
    base, _, _ = seed20_base
    assert base["first_divergence_point"] == (
        "checkpoint_native_assignment_vs_"
        "original_style_kmeans_initialized_assignment"
    )
    assert not seed20_prepared["fusion_inputs"][
        "B6_WQ0_PREFUSION_ASSIGNMENT_MATCH"
    ]


def test_source_utility_shape_is_210_by_5(seed20_prepared):
    assert seed20_prepared["utility"].shape == (210, 5)


def test_source_utility_is_in_unit_interval(seed20_prepared):
    utility = seed20_prepared["utility"]
    assert np.isfinite(utility).all()
    assert utility.min() >= 0.0
    assert utility.max() <= 1.0


def test_source_utility_hash_exactly_matches_a05(seed20_prepared):
    assert seed20_prepared["utility_hash"] == b6.SEED_PROVENANCE[20]["utility_hash"]


def test_correct_admission_shape_rowsum_and_values():
    attention, utility = toy_inputs()
    weights = b6.admission_weights(attention, utility)
    assert weights.shape == (210, 5)
    assert np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    assert np.isfinite(weights).all()
    assert weights.min() >= 0.0


def test_epsilon_is_exactly_one_e_minus_six():
    assert b6.EPSILON == 1e-6
    assert inspect.signature(b6.admission_weights).parameters["epsilon"].default == 1e-6


def test_admission_rule_has_no_unregistered_transform():
    source = inspect.getsource(b6.admission_weights).lower()
    for token in ("gamma", "temperature", "top_k", "topk", "threshold"):
        assert token not in source


def test_shuffle_only_sample_dimension_within_each_view():
    utility = np.arange(1050, dtype=np.float64).reshape(210, 5)
    bank = b6.generate_within_view_permutations(repeats=1)
    shuffled = b6.shuffle_utility_within_views(utility, bank[0])
    for view_id in range(5):
        assert np.array_equal(
            shuffled[:, view_id], utility[bank[0, view_id], view_id]
        )


def test_view_marginals_are_preserved_exactly():
    _, utility = toy_inputs()
    bank = b6.generate_within_view_permutations(repeats=1)
    shuffled = b6.shuffle_utility_within_views(utility, bank[0])
    for view_id in range(5):
        assert np.array_equal(
            np.sort(shuffled[:, view_id]), np.sort(utility[:, view_id])
        )


def test_fixed_permutation_bank_is_reproducible():
    first = b6.generate_within_view_permutations()
    second = b6.generate_within_view_permutations()
    assert first.shape == (200, 5, 210)
    assert np.array_equal(first, second)


def test_two_hundred_null_permutations_are_saved(tmp_path):
    values = np.linspace(0.0, 1.0, 200)
    b6.save_null_arrays(tmp_path, values, values + 1.0, values + 2.0)
    assert np.load(tmp_path / "shuffled_acc.npy").shape == (200,)
    assert np.load(tmp_path / "shuffled_nmi.npy").shape == (200,)
    assert np.load(tmp_path / "shuffled_ari.npy").shape == (200,)


def test_no_labels_or_corruption_mask_in_admission_api():
    functions = (
        b6.admission_weights,
        b6.fused_representation_from_admission,
        b6.shuffle_utility_within_views,
    )
    for function in functions:
        names = [name.lower() for name in inspect.signature(function).parameters]
        assert all("label" not in name for name in names)
        assert all("mask" not in name for name in names)


def test_no_optimizer_or_backward():
    source = inspect.getsource(b6)
    assert "torch.optim" not in source
    assert ".backward(" not in source


def test_backbone_before_after_is_exact(seed20_prepared, seed20_base):
    del seed20_base
    before = seed20_prepared["backbone_hash_before"]
    after = hash_backbone(seed20_prepared["models"].autoencoders)
    assert before == after


def test_mvcan_clustering_evaluator_is_reused():
    assert b6.mvcan_acc is ClusteringTest.acc
    assert b6.mvcan_nmi is ClusteringTest.nmi
    assert b6.mvcan_ari is ClusteringTest.ari
    source = inspect.getsource(b6._kmeans)
    assert "n_init=100" in source
    assert "random_state=int(seed)" in source


def gate_row(signal, acc, nmi, ari):
    return {
        "B6_WQ0_SEED_UTILITY_ADMISSION_SIGNAL": signal,
        "delta_vs_base": {"acc": acc, "nmi": nmi, "ari": ari},
    }


def test_multiseed_gate_logic_exact_pass():
    rows = [
        gate_row(True, 0.01, 0.02, 0.01),
        gate_row(True, 0.02, 0.01, 0.02),
        gate_row(False, -0.01, 0.01, 0.01),
    ]
    result = b6.multiseed_gate(rows)
    assert result["seed_signal_count"] == 2
    assert result["B6_WQ0_MULTISEED_ADMISSION_PASS"]


@pytest.mark.parametrize(
    "rows",
    [
        [
            gate_row(True, 0.01, 0.01, 0.01),
            gate_row(False, 0.01, 0.01, 0.01),
            gate_row(False, 0.01, 0.01, 0.01),
        ],
        [
            gate_row(True, 0.01, 0.01, 0.01),
            gate_row(True, 0.01, 0.01, 0.01),
            gate_row(False, -0.03, 0.01, 0.01),
        ],
        [
            gate_row(True, 0.01, -0.02, 0.01),
            gate_row(True, 0.01, 0.00, 0.01),
            gate_row(False, 0.01, 0.01, 0.01),
        ],
    ],
)
def test_multiseed_gate_rejects_each_failed_requirement(rows):
    assert not b6.multiseed_gate(rows)["B6_WQ0_MULTISEED_ADMISSION_PASS"]


def _mock_scientific_result(seed, signal=True, delta=0.01):
    return {
        "model_seed": int(seed),
        "B6_WQ0_SCIENTIFIC_ACTION_GATE_PASS": True,
        "B6_WQ0_SEED_UTILITY_ADMISSION_SIGNAL": bool(signal),
        "delta_vs_base": {
            "acc": float(delta),
            "nmi": float(delta),
            "ari": float(delta),
        },
    }


@pytest.mark.parametrize("seed", b6.SUPPORTED_SEEDS)
def test_single_supported_seed_full_scientific_mode_is_allowed(
    seed,
    monkeypatch,
    capsys,
):
    calls = []
    expected = _mock_scientific_result(seed)

    def fake_evaluate(
        model_seed,
        output_root,
        base_only=False,
        canonical_manifest_path=None,
    ):
        calls.append({
            "seed": model_seed,
            "output_root": output_root,
            "base_only": base_only,
            "canonical_manifest_path": canonical_manifest_path,
        })
        return expected

    monkeypatch.setattr(b6, "evaluate_seed", fake_evaluate)
    monkeypatch.setattr(b6, "_print_base_diagnostics", lambda result: None)
    monkeypatch.setattr(
        b6,
        "multiseed_gate",
        lambda results: pytest.fail("single-seed run aggregated results"),
    )
    result = b6.main(["--seeds", str(seed)])
    output = capsys.readouterr().out

    assert result is expected
    assert calls == [{
        "seed": seed,
        "output_root": b6.DEFAULT_OUTPUT_DIR,
        "base_only": False,
        "canonical_manifest_path": str(b6.CANONICAL_MANIFEST_PATH),
    }]
    assert "B6_WQ0_MULTISEED_ADMISSION_PASS" not in result
    assert "B6_WQ0_MULTISEED_ADMISSION_PASS=" not in output
    assert "B6_WQ0_SINGLE_SEED_SCIENTIFIC_COMPLETE=true" in output
    assert "model_seed=" + str(seed) in output


@pytest.mark.parametrize(
    "seeds",
    ((20, 30), (20, 50), (30, 50)),
)
def test_nonregistered_partial_multiseed_scientific_mode_is_rejected(
    seeds,
    monkeypatch,
):
    monkeypatch.setattr(
        b6,
        "evaluate_seed",
        lambda *args, **kwargs: pytest.fail(
            "invalid seed mode reached scientific evaluation"
        ),
    )
    with pytest.raises(
        RuntimeError,
        match="full evaluation requires one supported seed or seeds 20 30 50",
    ):
        b6.main(["--seeds", *(str(seed) for seed in seeds)])


def test_unsupported_scientific_seed_is_rejected_before_evaluation(
    monkeypatch,
):
    monkeypatch.setattr(
        b6,
        "evaluate_seed",
        lambda *args, **kwargs: pytest.fail(
            "unsupported seed reached scientific evaluation"
        ),
    )
    with pytest.raises(RuntimeError, match="unsupported seed"):
        b6.main(["--seeds", "40"])


def test_duplicate_scientific_seed_is_rejected_before_evaluation(
    monkeypatch,
):
    monkeypatch.setattr(
        b6,
        "evaluate_seed",
        lambda *args, **kwargs: pytest.fail(
            "duplicate seed reached scientific evaluation"
        ),
    )
    with pytest.raises(RuntimeError, match="duplicate seed"):
        b6.main(["--seeds", "20", "20"])


def test_full_three_seed_aggregation_control_flow_is_unchanged(
    monkeypatch,
    tmp_path,
    capsys,
):
    rows = {
        20: _mock_scientific_result(20, signal=True, delta=0.01),
        30: _mock_scientific_result(30, signal=True, delta=0.02),
        50: _mock_scientific_result(50, signal=False, delta=-0.01),
    }
    written = {}

    monkeypatch.setattr(
        b6,
        "evaluate_seed",
        lambda seed, *args, **kwargs: rows[seed],
    )
    monkeypatch.setattr(b6, "_print_base_diagnostics", lambda result: None)
    monkeypatch.setattr(
        b6,
        "_write_json",
        lambda path, value: written.update({"path": path, "value": value}),
    )
    summary = b6.main([
        "--seeds", "20", "30", "50",
        "--output-dir", str(tmp_path),
    ])
    output = capsys.readouterr().out
    expected_gate = b6.multiseed_gate([rows[20], rows[30], rows[50]])

    assert summary["seed_results"] == [rows[20], rows[30], rows[50]]
    assert all(summary[key] == value for key, value in expected_gate.items())
    assert written["path"] == tmp_path / "b6_wq0_multiseed_summary.json"
    assert written["value"] is summary
    assert "B6_WQ0_MULTISEED_ADMISSION_PASS=true" in output
    assert "B6_WQ0_SINGLE_SEED_SCIENTIFIC_COMPLETE" not in output


def test_base_only_supported_subset_behavior_is_unchanged(
    monkeypatch,
    capsys,
):
    calls = []

    def fake_evaluate(seed, *args, **kwargs):
        calls.append((seed, kwargs["base_only"]))
        return _mock_scientific_result(seed)

    monkeypatch.setattr(b6, "evaluate_seed", fake_evaluate)
    monkeypatch.setattr(b6, "_print_base_diagnostics", lambda result: None)
    monkeypatch.setattr(
        b6,
        "multiseed_gate",
        lambda results: pytest.fail("base-only run aggregated results"),
    )
    results = b6.main(["--base-only", "--seeds", "20", "30"])
    output = capsys.readouterr().out

    assert [result["model_seed"] for result in results] == [20, 30]
    assert calls == [(20, True), (30, True)]
    assert "B6_WQ0_BASE_ONLY_COMPLETE=true" in output
    assert "B6_WQ0_SINGLE_SEED_SCIENTIFIC_COMPLETE" not in output


def test_single_seed_scientific_path_still_uses_200_null_repeats():
    source = inspect.getsource(b6.evaluate_seed)
    assert b6.NULL_REPEATS == 200
    assert "for repeat_id in range(NULL_REPEATS):" in source
    assert '"shuffled_null_repeats": NULL_REPEATS' in source


def test_scientific_protocol_sources_are_unchanged():
    expected_hashes = {
        "_kmeans": "db4c596a90e754c084a79739ce9d47f0269740b8f83584e75d5252fca06533f2",
        "admission_weights": "5e292c93aca4160bec5de656d9f1c4ecd94be23dadf45e58d678fd539d796dbb",
        "fused_representation_from_admission": "09abc3474ffbcc47b41b7ba3ab0df0e7fdf44f2cbf3be4aa3203e974c3a35ff6",
        "generate_within_view_permutations": "ed8e08a0ac3d82ffb9583bf642747c612ab6ee319fed69784cb5b5d02a0d9e80",
        "shuffle_utility_within_views": "be0882f6b82ce93b769df7f454dfc517dacce177df07e48a048bbfb2994e5e91",
        "evaluate_fused_representation": "724365c0cd5366ee2d5648327aa05edec073128f6bcb59454e0cc8334e5351b0",
        "canonical_base_reproduction_audit": "bf8532f9140d91d2cf9772f74e9a7cdc046a006385dcef1a5bf2c74af77eff76",
        "scientific_action_gate": "8ba45a550728ed161aaf906748ee87d50d86b2c946d1add89122ad1fbc88e2a2",
        "load_source_utility": "292caecedb8509648effd1c0aad8938179a941ef692309140a5297b1f33dfd48",
        "summarize_null": "be1f05f08d1c9c3a562fbe4417fe6f37853f02d000c57a0a021fca860d176bbc",
        "multiseed_gate": "10ebd0ab34d0c70a0e166054fc50c749569e6f565d16fd0b5d52121484379275",
    }
    for name, expected_hash in expected_hashes.items():
        source = inspect.getsource(getattr(b6, name)).encode("utf-8")
        assert hashlib.sha256(source).hexdigest() == expected_hash
