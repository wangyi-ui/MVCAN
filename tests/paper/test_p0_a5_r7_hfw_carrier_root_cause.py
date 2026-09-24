import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.paper.diagnostics import msrc_hfw_carrier_root_cause_audit as audit


def _arm(offset=0.0):
    return {
        "M_v": np.full((1, 2, 2), offset, dtype=np.float32),
        "q_local": np.full((2, 2, 2), offset, dtype=np.float32),
        "q_aligned": np.full((2, 2, 2), offset, dtype=np.float32),
        "U_cycle": np.full((2, 1), offset, dtype=np.float32),
        "y_gen": np.array([0, int(offset != 0)], dtype=np.int64),
        "PredRelation_true": np.full((1, 1, 1), bool(offset), dtype=np.bool_),
        "relation_balance_weights_true": np.full((1, 1, 1), 1.0 + offset,
                                                   dtype=np.float64),
    }


def _r6_record(*, decision="FINAL_REFRESH_DEVICE_BACKEND_MISMATCH", exact=True):
    return {
        "schema": "paper-p0-a5-r6-final-refresh-replay-validity-v1",
        "full_gt_loaded": False,
        "validity_decision": decision,
        "h0_state_artifact": {"sha256": audit.R6_H0_ARTIFACT_SHA256},
        "H_vs_T_CUDA_refresh": {
            name: {"array_equal": exact}
            for name in ("P_all", "M_v", "prediction", "outgoing_weights")
        },
    }


def test_r6_success_evidence_gate_passes_actual_frozen_output():
    result = audit.validate_r6_evidence()
    assert result["record"]["validity_decision"] == "FINAL_REFRESH_DEVICE_BACKEND_MISMATCH"
    assert result["artifact_sha256"] == audit.R6_H0_ARTIFACT_SHA256
    assert np.array_equal(result["historical_weights"], audit.EXPECTED_H1_WEIGHTS)


@pytest.mark.parametrize("decision,exact", [
    ("FINAL_REFRESH_REPLAY_STILL_NOT_EXACT", True),
    ("FINAL_REFRESH_DEVICE_BACKEND_MISMATCH", False),
])
def test_r6_wrong_decision_or_cuda_refresh_difference_fails_closed(
        tmp_path, decision, exact):
    artifact = audit.R6_H0_ARTIFACT
    result = tmp_path / "r6.json"
    result.write_text(json.dumps(_r6_record(decision=decision, exact=exact)),
                      encoding="utf-8")
    with pytest.raises(RuntimeError, match="R6_REFRESH_VALIDITY_EVIDENCE_NOT_EXACT"):
        audit.validate_r6_evidence(result, artifact)


def test_historical_state_sha_gate_is_frozen():
    assert audit.HISTORICAL_H_STATE.is_file()
    assert audit.file_sha256(audit.HISTORICAL_H_STATE) == audit.HISTORICAL_H_STATE_SHA256


def test_historical_h_action_hash_gate_is_wired_without_training():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "_historical_h5_exact(h_arm)" in source
    assert "HISTORICAL_H_ACTION_CARRIER_NOT_EXACT" in source
    assert audit.EXPECTED_HISTORICAL_ACTION_HASHES == {
        "U_cycle": "3552d88de9f894276d0d1ccae8f4c55147e98aac9a8ae146eaa1cf54dcc9dd10",
        "PredRelation_true": "cbd2d2de1dfd51abba9b8e80fcc1ee82f4e902df6532ca2af388e1266ff0d2b8",
        "relation_balance_weights_true": "2d69642382947465851d69fe9f151df440979ce626481f9ad4c060909f92ff9a",
    }


def test_compare_contract_covers_exactly_all_seven_arrays():
    comparison = audit.compare_arm(_arm(), _arm())
    assert tuple(comparison) == audit.R7_ARRAYS
    assert audit.arm_exact(comparison) is True
    changed = _arm()
    changed["relation_balance_weights_true"][0, 0, 0] = 2.0
    assert audit.arm_exact(audit.compare_arm(_arm(), changed)) is False


@pytest.mark.parametrize("h,f,w,expected", [
    (0.0, 0.0, 1.0, "PRE_R2_CARRIER_EXACT_PARITY"),
    (0.0, 1.0, 0.0, "VIEW_WEIGHT_CARRIER_RESET_MISMATCH"),
    (1.0, 0.0, 0.0, "FINAL_REFRESH_TEMPORAL_STATE_MISMATCH"),
    (0.0, 1.0, 2.0, "COMBINED_FINAL_REFRESH_CARRIER_MISMATCH"),
])
def test_hfw_classifier_order_is_frozen(h, f, w, expected):
    assert audit.classify_hfw_carrier_root_cause(
        audit.compare_arm(_arm(h), _arm(f)),
        audit.compare_arm(_arm(h), _arm(w)),
        audit.compare_arm(_arm(w), _arm(f)),
    ) == expected



def test_hfw_classifier_fails_closed_for_inconsistent_comparison_records():
    def record(exact):
        return {name: {"array_equal": exact} for name in audit.R7_ARRAYS}

    assert audit.classify_hfw_carrier_root_cause(
        record(False), record(True), record(True)
    ) == "FAIL_CLOSED_UNCLASSIFIED"

def test_first_divergence_uses_frozen_causal_order():
    baseline, changed = _arm(), _arm()
    changed["q_aligned"][0, 0, 0] = 1.0
    assert audit.first_divergent_component(audit.compare_arm(baseline, changed)) == "q_aligned"
    changed = _arm()
    changed["M_v"][0, 0, 0] = 1.0
    assert audit.first_divergent_component(audit.compare_arm(baseline, changed)) == "M_v"


def test_action_diagnostics_have_no_threshold_gate():
    report = audit.action_diagnostics(_arm(), _arm(1.0), _arm(2.0))
    assert set(report) == {"U_cycle", "y_gen", "PredRelation_true"}
    assert report["U_cycle"]["pairs"]["H_vs_F"]["changed_entry_count"] > 0
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "threshold" not in source.lower()


def test_fresh_controls_receive_independent_weight_lists(monkeypatch):
    seen = []

    def fresh(model, views, weights):
        seen.append((id(weights), list(weights)))
        weights[0] = 99.0
        return np.zeros((1, 2, 2), dtype=np.float32)

    monkeypatch.setattr(audit, "_fresh_matches", fresh)
    monkeypatch.setattr(
        audit, "_carrier_from_aligned",
        lambda *args, **kwargs: _arm(),
    )
    legacy = SimpleNamespace(coordinate_snapshot=lambda *args: (
        np.zeros((2, 1, 2), dtype=np.float32), np.zeros((2, 1, 2), dtype=np.float32)
    ))
    weights = [1.0] * 5
    audit._fresh_arm(legacy, object(), [], weights, object(), device="cpu")
    audit._fresh_arm(legacy, object(), [], weights, object(), device="cpu")
    assert weights == [1.0] * 5
    assert seen[0][0] != seen[1][0]
    assert seen[0][1] == seen[1][1] == [1.0] * 5


def test_partial_is_atomic_gt_free_and_precedes_formal_output(tmp_path):
    output = tmp_path / "formal"
    partial = {
        "schema": audit.SCHEMA + "-partial", "stage_reached": "HFW_COMPARISONS_COMPLETED",
        "full_gt_loaded": False, "comparisons": {"H_vs_F": {}},
    }
    path = audit._write_partial_evidence(output, partial)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert output.exists() is False
    assert record["full_gt_loaded"] is False
    assert "ground_truth" not in path.read_text(encoding="utf-8").lower()
    assert not list(tmp_path.glob(path.name + ".*.tmp"))


def test_r7_is_cheap_and_does_not_repurpose_old_t_classifier():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "legacy.prepare_native_backbone(" not in source
    assert "classify_carrier_alignment" not in source
    assert "T_carrier" not in source
    assert "FINAL_REFRESH_DEVICE_BACKEND_MISMATCH" in source


def test_protected_sources_and_historical_archive_are_unchanged():
    root = Path(audit.__file__).resolve().parents[3]
    protected = [
        "release_core",
        "experiments/paper/diagnostics/msrc_pre_r2_carrier_temporal_alignment.py",
        "experiments/paper/diagnostics/msrc_final_refresh_replay_validity_audit.py",
    ]
    diff = subprocess.run(
        ["git", "diff", "--name-only", "--", *protected], cwd=root,
        check=True, text=True, stdout=subprocess.PIPE,
    )
    historical = subprocess.run(
        ["git", "-C", "/root/autodl-tmp/CVPR24-MVCAN", "status", "--short"],
        check=True, text=True, stdout=subprocess.PIPE,
    )
    assert diff.stdout == ""
    assert historical.stdout == ""
