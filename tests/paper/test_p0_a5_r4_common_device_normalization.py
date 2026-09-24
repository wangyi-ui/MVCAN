import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.paper.diagnostics import msrc_pre_r2_carrier_temporal_alignment as audit
from experiments.paper.diagnostics import p0_a3_protocol as protocol
from experiments.paper.transfer_audit.input_artifacts import load_materialized_inputs


def _split():
    _, _, _, split = load_materialized_inputs(protocol.MSRC_INPUT_DIR)
    return split


def test_carrier_builder_places_r2_input_on_requested_device(monkeypatch):
    seen = {}

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
    q = np.full((4, 5, 2), 0.5, dtype=np.float64)
    result = audit._carrier_from_aligned(
        q, q, np.eye(2, dtype=np.float32)[None, ...].repeat(5, axis=0),
        object(), device="cpu",
    )
    assert seen["device"] == "cpu"
    assert result["q_aligned"].dtype == np.float32


def test_f_provenance_is_separate_from_normalized_f_arm_and_all_arms_share_device():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "F_PROVENANCE, reconstructed_current_split" in source
    assert "validate_reconstruction_against_manifest(\n            F_PROVENANCE, manifest" in source
    assert "F_PROVENANCE[\"q_local\"], F_PROVENANCE[\"q_aligned\"]" in source
    assert '"H_vs_F": compare_arm(H, F_ARM)' in source
    assert '"H_vs_F": compare_arm(H, F_PROVENANCE)' not in source
    assert source.count("device=device") >= 4


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


def test_partial_evidence_is_atomic_gt_free_and_does_not_create_formal_output(tmp_path):
    formal_output = tmp_path / "formal_output"
    evidence = audit._partial_evidence(_capture_fixture(), "cuda:0")
    first = audit._write_partial_evidence(formal_output, evidence)
    evidence["stage_reached"] = "H_R2_R3_COMPLETED"
    second = audit._write_partial_evidence(formal_output, evidence)
    assert first == second == Path(str(formal_output) + ".partial.json")
    assert formal_output.exists() is False
    record = json.loads(second.read_text(encoding="utf-8"))
    assert record["full_gt_loaded"] is False
    assert record["stage_reached"] == "H_R2_R3_COMPLETED"
    assert "ground_truth" not in second.read_text(encoding="utf-8").lower()


_A6_LEGACY_STATE = Path(
    "outputs/paper/diagnostics/p0_a3_cross_dataset_seed20/"
    "msrc_native_initializer_parity_replay_r1/legacy_pre_r2_state.npz"
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA sentinel requires cuda:0")
def test_no_training_cpu_cuda_historical_r2_sentinel():
    with np.load(_A6_LEGACY_STATE, allow_pickle=False) as archive:
        q_local = np.ascontiguousarray(archive["q_local"], dtype=np.float32)
        q_aligned = np.ascontiguousarray(archive["q_aligned"], dtype=np.float32)
        matrices = np.ascontiguousarray(archive["M_v"], dtype=np.float32)
    split = _split()
    cpu = audit._carrier_from_aligned(
        q_local, q_aligned, matrices, split, device="cpu"
    )
    cuda = audit._carrier_from_aligned(
        q_local, q_aligned, matrices, split, device="cuda:0"
    )
    assert audit.ndarray_sha256(cuda["U_cycle"]) == (
        audit.EXPECTED_HISTORICAL_ACTION_HASHES["U_cycle"]
    )
    assert audit.ndarray_sha256(cpu["U_cycle"]) != (
        audit.EXPECTED_HISTORICAL_ACTION_HASHES["U_cycle"]
    )
    assert np.array_equal(cpu["y_gen"], cuda["y_gen"])
    assert np.array_equal(cpu["PredRelation_true"], cuda["PredRelation_true"])
    assert np.array_equal(
        cpu["relation_balance_weights_true"], cuda["relation_balance_weights_true"]
    )
    assert float(np.max(np.abs(cpu["U_cycle"] - cuda["U_cycle"]))) == pytest.approx(
        1.4901161193847656e-08, abs=1e-15
    )
