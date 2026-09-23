import copy
import subprocess

import pytest

from experiments.paper.diagnostics import msrc_pre_r2_carrier_temporal_alignment as audit


def _valid_record():
    release_hashes = {"release_core/example.py": "frozen"}
    return {
        "schema": audit.P0_A4_R3_SCHEMA,
        "dataset": audit.P0_A4_R3_DATASET,
        "training_seed": 20,
        "historical_training_seed": 20,
        "release_code_modified": False,
        "release_core_source_sha256": release_hashes,
        "first_divergence": {
            "decision": "EXACT_INITIALIZATION_PARITY",
            "last_exact_stage": "S14_FINAL_PRE_R2",
            "first_divergent_stage": None,
            "first_divergent_components": [],
            "first_divergent_paths": [],
        },
    }, release_hashes


def test_valid_nested_first_divergence_schema_passes():
    record, release_hashes = _valid_record()
    assert audit._validate_p0_a4_r3_record(record, release_hashes) == (
        record["first_divergence"]
    )


def test_root_level_decision_is_not_a_compatibility_fallback():
    record, release_hashes = _valid_record()
    record["decision"] = record.pop("first_divergence")["decision"]
    with pytest.raises(RuntimeError, match="first_divergence"):
        audit._validate_p0_a4_r3_record(record, release_hashes)


@pytest.mark.parametrize("field, value", (
    ("decision", "NATIVE_INITIALIZATION_MIGRATION_DIVERGENCE"),
    ("last_exact_stage", "S13_POST_NATIVE_1000"),
    ("first_divergent_stage", "S14_FINAL_PRE_R2"),
    ("first_divergent_components", ["model"]),
    ("first_divergent_paths", ["model.aggregate_hash"]),
))
def test_non_exact_nested_first_divergence_fields_fail_closed(field, value):
    record, release_hashes = _valid_record()
    record["first_divergence"][field] = value
    with pytest.raises(RuntimeError, match="exact-initialization evidence mismatch"):
        audit._validate_p0_a4_r3_record(record, release_hashes)


def test_wrong_release_source_sha_fails_closed():
    record, release_hashes = _valid_record()
    record["release_core_source_sha256"] = {"release_core/example.py": "wrong"}
    with pytest.raises(RuntimeError, match="root evidence metadata mismatch"):
        audit._validate_p0_a4_r3_record(record, release_hashes)


def test_wrong_r3_whole_file_sha_fails_closed(monkeypatch):
    monkeypatch.setattr(audit, "P0_A4_R3_REPORT_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="report identity mismatch"):
        audit._verify_p0_a4_r3_evidence()


def test_release_core_remains_unmodified():
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", "release_core"],
        cwd="/root/autodl-tmp/UCRR-MVC-paper",
        check=True, capture_output=True, text=True,
    )
    assert result.stdout == ""
