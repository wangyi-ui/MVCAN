"""Tests for B6-WQ0 canonical frozen-baseline provenance."""

import copy
import inspect
import json

import numpy as np
import pytest

import experiments.b6_weak_quality.audit_b6_wq0_frozen_baseline_provenance as provenance
import experiments.b6_weak_quality.evaluate_b6_wq0_utility_semantic_admission as b6


@pytest.fixture(scope="module")
def seed20_provenance():
    return provenance.audit_seed(20)


@pytest.fixture(scope="module")
def blocked_seed20_evaluation(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("b6_blocked_evaluation")
    empty_manifest_path = output_dir / "empty_manifest.json"
    empty_manifest_path.write_text(
        json.dumps({
            "schema_version": "b6-wq0-frozen-baseline-v1",
            "registration_status": "test-unregistered",
            "canonical_entries": {},
        }),
        encoding="utf-8",
    )
    return b6.evaluate_seed(
        20,
        output_root=output_dir,
        base_only=False,
        canonical_manifest_path=empty_manifest_path,
    )


def test_provenance_source_has_no_training_actions():
    source = inspect.getsource(provenance)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "optimizer.step(" not in source


def test_seed20_backbone_is_exactly_frozen_and_no_updates(seed20_provenance):
    record = seed20_provenance
    assert record["backbone_hash_before"] == record["backbone_hash_after"]
    assert record["backbone_frozen_pass"]
    assert record["B6_WQ0_BACKBONE_FROZEN_PASS"]
    assert not record["optimizer_created"]
    assert not record["backward_performed"]
    assert not record["parameter_updates"]


def test_seed20_two_independent_replays_are_deterministic(seed20_provenance):
    record = seed20_provenance
    checks = record["determinism_audit"]["checks"]
    assert checks["backbone_hash_exact"]
    assert checks["raw_z_hashes_exact"]
    assert checks["normalized_z_hash_exact"]
    assert checks["scaled_z_hashes_exact"]
    assert checks["checkpoint_native_assignment_hashes_exact"]
    assert checks["kmeans_initialized_assignment_hashes_exact"]
    assert checks["www_sequence_exact"]
    assert checks["fusion_update_hashes_exact"]
    assert checks["final_fused_hash_exact"]
    assert checks["final_prediction_hash_exact"]
    assert checks["canonical_metrics_exact"]
    assert record["B6_WQ0_FROZEN_BASELINE_DETERMINISTIC_PASS"]


def test_seed20_required_hashes_and_finite_gates(seed20_provenance):
    record = seed20_provenance
    assert len(record["raw_z_sha256_per_view"]) == 5
    assert len(record["scaled_z_sha256_per_view"]) == 5
    assert len(
        record["checkpoint_native_assignment_sha256_per_view"]
    ) == 5
    assert len(
        record["kmeans_initialized_assignment_sha256_per_view"]
    ) == 5
    assert len(record["fusion_updates"]) == record["T1"]
    assert record["all_tensors_finite"]
    assert record["canonical_metric_finite"]
    assert record["all_artifacts_and_hashes_exist"]
    assert all(
        np.isfinite(value)
        for value in record["canonical_metrics"].values()
    )


def test_seed20_canonical_identity_gate_is_strict_and_passes(seed20_provenance):
    identity = seed20_provenance["identity_admission_audit"]
    assert identity["geometry_within_tolerance"]
    assert identity["relative_view_scale_preserved"]
    assert identity["assignment_exact_match"]
    assert identity["metrics_exact_match"]
    assert identity["B6_WQ0_IDENTITY_ADMISSION_PASS"]
    assert seed20_provenance["B6_WQ0_IDENTITY_ADMISSION_PASS"]


def test_historical_mismatch_does_not_invalidate_canonical_provenance(
    seed20_provenance,
):
    assert not seed20_provenance["historical_exact_replay_pass"]
    assert not seed20_provenance["B6_WQ0_HISTORICAL_B2_REPRO_PASS"]
    assert seed20_provenance[
        "B6_WQ0_FROZEN_BASELINE_DETERMINISTIC_PASS"
    ]
    assert seed20_provenance["B6_WQ0_FROZEN_BASELINE_SEED_PASS"]


def test_formal_manifest_has_exact_registered_seed_set():
    manifest, path = b6.load_canonical_manifest()
    assert path == b6.CANONICAL_MANIFEST_PATH
    assert set(manifest["canonical_entries"]) == {"20", "30", "50"}
    assert manifest["registration_status"] == "registered"


def test_unregistered_manifest_blocks_scientific_evaluator(
    blocked_seed20_evaluation,
):
    result = blocked_seed20_evaluation
    assert not result["B6_WQ0_CANONICAL_BASE_REPRO_PASS"]
    assert not result["B6_WQ0_SCIENTIFIC_ACTION_GATE_PASS"]
    assert result["B6_WQ0_SCIENTIFIC_ACTION_BLOCKED"]
    assert result["scientific_comparison_performed"] is False
    assert result["correct_u_metrics"] is None
    assert result["shuffled_primary_metrics"] is None
    assert result["scientific_action_block_reason"] == (
        "canonical frozen baseline is not formally registered"
    )


def _canonical_audit_from_record(record, manifest_path):
    return b6.canonical_base_reproduction_audit(
        seed=record["model_seed"],
        actual_metrics=record["canonical_metrics"],
        actual_backbone_hash=record["backbone_hash"],
        actual_normalized_z_hash=record["normalized_z_sha256"],
        actual_fused_hash=record[
            "canonical_fused_representation_sha256"
        ],
        actual_prediction_hash=record[
            "canonical_prediction_sha256"
        ],
        manifest_path=manifest_path,
    )


def test_registered_canonical_mismatch_blocks_action(seed20_provenance, tmp_path):
    entry = provenance._candidate_entry(
        seed20_provenance,
        tmp_path / "source_provenance.json",
    )
    entry["expected_prediction_hash"] = "0" * 64
    manifest_path = tmp_path / "wrong_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "schema_version": "b6-wq0-frozen-baseline-v1",
            "registration_status": "test-only",
            "canonical_entries": {"20": entry},
        }),
        encoding="utf-8",
    )
    audit = _canonical_audit_from_record(
        seed20_provenance, manifest_path
    )
    assert audit["canonical_baseline_registered"]
    assert not audit["checks"]["prediction_hash_exact"]
    assert not audit["B6_WQ0_CANONICAL_BASE_REPRO_PASS"]
    assert audit["scientific_action_block_reason"] == (
        "registered canonical baseline mismatch"
    )


def test_candidate_entry_can_validate_only_after_explicit_test_registration(
    seed20_provenance,
    tmp_path,
):
    entry = provenance._candidate_entry(
        seed20_provenance,
        tmp_path / "source_provenance.json",
    )
    manifest_path = tmp_path / "registered_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "schema_version": "b6-wq0-frozen-baseline-v1",
            "registration_status": "test-only",
            "canonical_entries": {"20": entry},
        }),
        encoding="utf-8",
    )
    audit = _canonical_audit_from_record(
        seed20_provenance, manifest_path
    )
    assert not seed20_provenance[
        "B6_WQ0_HISTORICAL_B2_REPRO_PASS"
    ]
    assert audit["B6_WQ0_CANONICAL_BASE_REPRO_PASS"]


def test_candidate_summary_is_produced_without_modifying_formal_manifest(
    seed20_provenance,
    tmp_path,
    monkeypatch,
):
    formal_before = b6.CANONICAL_MANIFEST_PATH.read_bytes()

    def fake_audit(seed):
        record = copy.deepcopy(seed20_provenance)
        record["model_seed"] = int(seed)
        return record

    monkeypatch.setattr(provenance, "audit_seed", fake_audit)
    summary, records = provenance.audit_provenance(
        [20, 30, 50],
        output_dir=tmp_path,
    )
    assert len(records) == 3
    assert set(summary["canonical_manifest_candidate_entries"]) == {
        "20",
        "30",
        "50",
    }
    assert summary["B6_WQ0_FROZEN_BASELINE_PROVENANCE_COMPLETE"]
    assert not summary["formal_manifest_modified"]
    assert b6.CANONICAL_MANIFEST_PATH.read_bytes() == formal_before
    for seed in (20, 30, 50):
        candidate = summary[
            "canonical_manifest_candidate_entries"
        ][str(seed)]
        assert candidate["model_seed"] == seed
        assert candidate["source_provenance"].endswith(
            "seed" + str(seed) + "_provenance.json"
        )
