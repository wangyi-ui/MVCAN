import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.paper.diagnostics import msrc_pre_r2_carrier_temporal_alignment as audit


def _capture_fixture():
    return {
        "H0_pre_refresh_model": {"aggregate_hash": "h0", "per_view": []},
        "H1_incoming_weights": [1.0] * 5,
        "H2_refresh_p": np.zeros((2, 2), dtype=np.float32),
        "H2_refresh_M_v": np.zeros((5, 2, 2), dtype=np.float32),
        "H2_refresh_prediction": np.zeros(2, dtype=np.int64),
        "H2_outgoing_weights": [1.0] * 5,
        "H3_post_update_model": {"aggregate_hash": "h3", "per_view": []},
        "H4_final_q_local": np.zeros((2, 5, 2), dtype=np.float32),
        "H4_final_q_aligned_retained_M": np.zeros((2, 5, 2), dtype=np.float32),
    }


def _stub_carrier_dependencies(monkeypatch, seen):
    def fake_cycle(q_tensor, actions):
        seen["device"] = str(q_tensor.device)
        return {
            "U_cycle": torch.zeros((4, 1), device=q_tensor.device),
            "y_gen": torch.zeros(4, dtype=torch.int64, device=q_tensor.device),
        }

    monkeypatch.setattr(audit, "compute_directional_cycle_utility", fake_cycle)
    monkeypatch.setattr(
        audit, "build_relation_semantics",
        lambda y_gen, split, actions: SimpleNamespace(
            pred_relation=np.zeros((1, 1, 1), dtype=np.bool_),
            balance_weights=np.ones((1, 1, 1), dtype=np.float64),
        ),
    )


@pytest.mark.parametrize("as_tensor", [False, True])
def test_carrier_accepts_numpy_and_cpu_tensor_and_serializes_outputs(
        monkeypatch, as_tensor):
    seen = {}
    _stub_carrier_dependencies(monkeypatch, seen)
    aligned = np.full((4, 5, 2), 0.5, dtype=np.float64)
    matrix = np.eye(2, dtype=np.float32)[None, ...].repeat(5, axis=0)
    if as_tensor:
        aligned = torch.from_numpy(aligned)
        matrix = torch.from_numpy(matrix)
    result = audit._carrier_from_aligned(
        aligned, aligned, matrix, object(), device="cpu"
    )
    assert seen["device"] == "cpu"
    assert isinstance(result["q_local"], np.ndarray)
    assert isinstance(result["q_aligned"], np.ndarray)
    assert isinstance(result["M_v"], np.ndarray)
    assert result["q_aligned"].dtype == np.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA test requires cuda:0")
def test_cuda_tensor_never_reenters_numpy_before_r2(monkeypatch):
    seen = {}
    _stub_carrier_dependencies(monkeypatch, seen)
    original = audit.np.ascontiguousarray

    def reject_tensor_numpy(value, *args, **kwargs):
        assert not isinstance(value, torch.Tensor)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(audit.np, "ascontiguousarray", reject_tensor_numpy)
    aligned = torch.full((4, 5, 2), 0.5, device="cuda:0")
    matrix = torch.eye(2, device="cuda:0")[None, ...].repeat(5, 1, 1)
    result = audit._carrier_from_aligned(
        aligned, aligned, matrix, object(), device="cuda:0"
    )
    assert seen["device"] == "cuda:0"
    assert all(isinstance(result[name], np.ndarray)
               for name in ("q_local", "q_aligned", "M_v"))


def test_all_diagnostic_arms_use_the_type_aware_carrier_wrapper():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert source.count("_carrier_with_partial_evidence(") == 5
    assert "F_PROVENANCE[\"q_local\"], F_PROVENANCE[\"q_aligned\"]" in source
    assert '"H_vs_F": compare_arm(H, F_ARM)' in source


def test_unexpected_boundary_error_persists_partial_and_reraises(tmp_path, monkeypatch):
    formal_output = tmp_path / "formal_output"
    evidence = audit._partial_evidence(_capture_fixture(), "cuda:0")
    message = "can't convert cuda:0 device type tensor to numpy"

    def fail(*args, **kwargs):
        raise TypeError(message)

    monkeypatch.setattr(audit, "_carrier_from_aligned", fail)
    with pytest.raises(TypeError, match="can't convert cuda:0"):
        audit._carrier_with_partial_evidence(
            np.zeros((1, 5, 2), dtype=np.float32),
            torch.zeros((1, 5, 2), device="cpu"),
            np.zeros((5, 2, 2), dtype=np.float32), object(), device="cuda:0",
            output_dir=formal_output, evidence=evidence,
            stage_reached="W_R2_R3_STARTED", completed_arms=["H", "F"],
        )
    partial = Path(str(formal_output) + ".partial.json")
    record = json.loads(partial.read_text(encoding="utf-8"))
    assert formal_output.exists() is False
    assert record["failure_class"] == "DIAGNOSTIC_TENSOR_NUMPY_BOUNDARY_MISMATCH"
    assert record["failure_exception_type"] == "TypeError"
    assert record["failure_exception_message"] == message
    assert record["stage_reached"] == "W_R2_R3_STARTED"
    assert record["completed_arms"] == ["H", "F"]
    assert record["full_gt_loaded"] is False
    assert not list(tmp_path.glob(partial.name + ".*.tmp"))


def test_r5_does_not_modify_release_core():
    root = Path(audit.__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", "release_core"],
        cwd=root, check=True, text=True, stdout=subprocess.PIPE,
    )
    assert result.stdout == ""
