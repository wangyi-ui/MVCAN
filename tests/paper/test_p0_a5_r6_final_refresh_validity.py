import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.paper.diagnostics import msrc_final_refresh_replay_validity_audit as audit
from experiments.paper.diagnostics import msrc_pre_r2_carrier_temporal_alignment as old
from experiments.paper.transfer_audit.input_artifacts import file_sha256
from experiments.paper.diagnostics.state_hashing import model_state_snapshot


class _Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._encoder = nn.Linear(2, 2)
        self._decoder = nn.Linear(2, 2)
        self._cluster_layer = nn.Parameter(torch.eye(2))

    def encoder(self, values):
        return self._encoder(values)

    def clustering(self, latent):
        return torch.softmax(latent @ self._cluster_layer, dim=1)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoders = nn.ModuleList([_Autoencoder(), _Autoencoder()])

    def to_device(self, device):
        self.to(device)


def _model_and_state():
    torch.manual_seed(20)
    model = _Model()
    state = tuple(copy.deepcopy(item.state_dict()) for item in model.autoencoders)
    return model, state, model_state_snapshot(model)


def _refresh(seed=0):
    return {
        "P_all": np.full((3, 2), 0.5 + seed, dtype=np.float32),
        "M_v": np.eye(2, dtype=np.float32)[None, ...].repeat(2, axis=0),
        "prediction": np.array([0, 1, 0], dtype=np.int64),
        "outgoing_weights": np.array([1.0, 2.0], dtype=np.float64),
    }


def _capture():
    _, state, snapshot = _model_and_state()
    return {
        "H0_pre_refresh_state_dict_cpu": state,
        "H0_pre_refresh_model": snapshot,
        "H1_incoming_weights": [1.0, 2.0],
        "H2_refresh_p": _refresh()["P_all"],
        "H2_refresh_M_v": _refresh()["M_v"],
        "H2_refresh_prediction": _refresh()["prediction"],
        "H2_outgoing_weights": _refresh()["outgoing_weights"].tolist(),
        "H3_post_update_model": {"aggregate_hash": "post-update"},
    }


def test_h0_cpu_restore_is_exact_and_cpu():
    model, state, snapshot = _model_and_state()
    restored = audit._restore_h0_model(model, state, snapshot, device="cpu")
    assert audit._all_module_values_on(restored, "cpu")
    assert old._snapshot_exact(model_state_snapshot(restored), snapshot)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA test requires cuda:0")
def test_h0_cuda_restore_and_views_are_cuda_zero():
    model, state, snapshot = _model_and_state()
    restored = audit._restore_h0_model(model, state, snapshot, device="cuda:0")
    views = audit._views_on_device([np.zeros((3, 2), dtype=np.float32)] * 2, "cuda:0")
    assert audit._all_module_values_on(restored, "cuda:0")
    assert all(str(view.device) == "cuda:0" for view in views)
    assert old._snapshot_exact(model_state_snapshot(restored), snapshot)


def test_cpu_views_and_forward_representation_comparison_are_diagnostic_only():
    model, _, _ = _model_and_state()
    views = audit._views_on_device([np.zeros((3, 2), dtype=np.float32)] * 2, "cpu")
    representation = audit._forward_representations(model, views)
    comparisons = audit._representation_comparisons(representation, representation)
    assert all(str(view.device) == "cpu" for view in views)
    assert all(item["array_equal"] and item["argmax_exact"]
               for group in comparisons.values() for item in group)


def test_native_refresh_controls_use_independent_weight_copies_and_same_entrypoint():
    seen = []

    def native_refresh(model, views, weights, device, seed):
        seen.append((id(weights), list(weights), device, seed))
        weights[0] = 99.0
        return (
            torch.zeros((2, 2)), torch.eye(2)[None, ...],
            np.array([0, 1], dtype=np.int64), weights,
        )

    legacy = SimpleNamespace(native_refresh=native_refresh)
    incoming = [1.0]
    first = audit._run_native_refresh(legacy, object(), [], incoming, device="cpu")
    second = audit._run_native_refresh(legacy, object(), [], incoming, device="cuda:0")
    assert incoming == [1.0]
    assert seen[0][0] != seen[1][0]
    assert [item[2:] for item in seen] == [("cpu", audit.EXPECTED_SEED),
                                             ("cuda:0", audit.EXPECTED_SEED)]
    assert first["outgoing_weights"][0] == second["outgoing_weights"][0] == 99.0


def test_refresh_comparison_covers_all_outputs_and_requires_exact_matrices():
    reference = _refresh()
    comparisons = audit.compare_refresh(reference, _refresh())
    assert tuple(comparisons) == audit.REFRESH_OUTPUTS
    assert audit.refresh_exact(comparisons) is True
    changed = _refresh()
    changed["M_v"] = changed["M_v"].copy()
    changed["M_v"][0, 0, 0] = 0.0
    comparisons = audit.compare_refresh(reference, changed)
    assert comparisons["M_v"]["array_equal"] is False
    assert comparisons["M_v"]["max_abs_diff"] == 1.0
    assert audit.refresh_exact(comparisons) is False


@pytest.mark.parametrize(
    "cpu_exact,cuda_exact,expected",
    [
        (False, True, "FINAL_REFRESH_DEVICE_BACKEND_MISMATCH"),
        (True, True, "FINAL_REFRESH_REPLAY_EXACT_DEVICE_INVARIANT"),
        (False, False, "FINAL_REFRESH_REPLAY_STILL_NOT_EXACT"),
    ],
)
def test_three_refresh_validity_decisions(cpu_exact, cuda_exact, expected):
    reference = _refresh()
    cpu = audit.compare_refresh(reference, _refresh(0 if cpu_exact else 1))
    cuda = audit.compare_refresh(reference, _refresh(0 if cuda_exact else 1))
    assert audit.classify_refresh_validity(cpu, cuda) == expected


def test_partial_precedes_formal_output_and_contains_no_gt(tmp_path):
    output = tmp_path / "formal_output"
    artifact_payload = audit._h0_artifact_payload(
        _capture(), np.array([3, 4, 5], dtype=np.int64), {"adapter": "sha"}
    )
    artifact, digest = audit._write_h0_state_artifact(output, artifact_payload)
    partial = audit._partial_evidence(_capture(), "cuda:0", artifact, digest)
    reference = _refresh()
    partial.update({
        "T_CPU_REFRESH_hashes": audit._refresh_hashes(reference),
        "T_CUDA_REFRESH_hashes": audit._refresh_hashes(reference),
        "cpu_cuda_latent_comparisons": [],
        "cpu_cuda_q_local_comparisons": [],
        "H_vs_T_CPU_refresh": audit.compare_refresh(reference, reference),
        "H_vs_T_CUDA_refresh": audit.compare_refresh(reference, reference),
        "validity_decision": "FINAL_REFRESH_REPLAY_EXACT_DEVICE_INVARIANT",
    })
    partial_path = audit._write_partial_evidence(output, partial)
    record = json.loads(partial_path.read_text(encoding="utf-8"))
    stored = torch.load(artifact, map_location="cpu")
    assert output.exists() is False
    assert file_sha256(artifact) == digest
    assert record["full_gt_loaded"] is False
    assert record["H_vs_T_CUDA_refresh"]["M_v"]["array_equal"] is True
    assert stored["full_gt_loaded"] is False
    assert "ground_truth" not in json.dumps(record).lower()
    assert "ground_truth" not in stored


def test_r6_never_uses_full_t_carrier_as_its_validity_gate():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "classify_carrier_alignment" not in source
    assert "T_carrier_validity_gate" in source
    assert "compare_refresh(h_refresh, cuda_refresh)" in source


def test_r6_keeps_protected_sources_outside_its_diff():
    root = Path(audit.__file__).resolve().parents[3]
    protected = [
        "release_core",
        "experiments/paper/diagnostics/msrc_pre_r2_carrier_temporal_alignment.py",
    ]
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", *protected],
        cwd=root, check=True, text=True, stdout=subprocess.PIPE,
    )
    assert result.stdout == ""
