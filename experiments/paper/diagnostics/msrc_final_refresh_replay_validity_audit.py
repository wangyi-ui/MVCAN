"""P0-A5-R6: validate the final historical refresh, not a full T carrier.

The prior T-vs-H carrier comparison mixed a restored pre-update H0 model with
the post-update H3 carrier.  This module keeps that frozen evidence intact and
performs an independent, refresh-level CPU/CUDA replay control.
"""

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments.paper.transfer_audit.input_artifacts import file_sha256
from .msrc_g0b0_parity_audit import compare_arrays
from .msrc_native_initializer_parity_replay import (
    HISTORICAL_INPUT_DIR,
    load_frozen_legacy_adapter,
    verify_frozen_legacy_files,
)
from .msrc_pre_r2_carrier_temporal_alignment import (
    EXPECTED_SEED,
    HistoricalFinalRefreshCapture,
    _build_cpu_clone,
    _require,
    _snapshot_exact,
    _write_json_exclusive,
    _write_partial_evidence,
    instrument_historical_final_refresh,
)
from .state_hashing import model_state_snapshot


SCHEMA = "paper-p0-a5-r6-final-refresh-replay-validity-v1"
REFRESH_OUTPUTS = ("P_all", "M_v", "prediction", "outgoing_weights")


def _numpy(value, dtype):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _all_module_values_on(model, device):
    return all(
        str(value.device) == str(device)
        for autoencoder in model.autoencoders
        for value in tuple(autoencoder.parameters()) + tuple(autoencoder.buffers())
    )


def _restore_h0_model(final_model, h0_state, h0_snapshot, *, device):
    """Strictly restore H0 on CPU, then move the diagnostic clone if requested."""
    clone = _build_cpu_clone(final_model, h0_state, h0_snapshot)
    if str(device) != "cpu":
        if hasattr(clone, "to_device"):
            clone.to_device(device)
        else:
            clone.to(device)
        for autoencoder in clone.autoencoders:
            autoencoder.train()
    _require(_all_module_values_on(clone, device),
             "H0 restored model device contract mismatch")
    _require(_snapshot_exact(model_state_snapshot(clone), h0_snapshot),
             "RESTORED_H0_MODEL_NOT_EXACT")
    return clone


def _views_on_device(views, device):
    return [torch.from_numpy(np.ascontiguousarray(view)).to(device)
            for view in views]


@torch.no_grad()
def _forward_representations(model, full_views):
    latent, q_local = [], []
    for autoencoder, view in zip(model.autoencoders, full_views):
        view_latent = autoencoder.encoder(view)
        view_q = autoencoder.clustering(view_latent)
        latent.append(_numpy(view_latent, np.float32))
        q_local.append(_numpy(view_q, np.float32))
    return {"latent": latent, "q_local": q_local}


def _representation_comparisons(cpu, cuda):
    result = {"latent": [], "q_local": []}
    for name in result:
        _require(len(cpu[name]) == len(cuda[name]),
                 "CPU/CUDA view count mismatch")
        for cpu_value, cuda_value in zip(cpu[name], cuda[name]):
            comparison = compare_arrays(cuda_value, cpu_value)
            comparison["argmax_exact"] = bool(np.array_equal(
                np.asarray(cuda_value).argmax(axis=-1),
                np.asarray(cpu_value).argmax(axis=-1),
            ))
            result[name].append(comparison)
    return result


def _refresh_values(result):
    p_all, matrices, prediction, outgoing_weights = result
    return {
        "P_all": _numpy(p_all, np.float32),
        "M_v": _numpy(matrices, np.float32),
        "prediction": _numpy(prediction, np.int64),
        "outgoing_weights": np.ascontiguousarray(
            np.asarray(outgoing_weights, dtype=np.float64)
        ),
    }


def _historical_refresh_values(capture):
    return {
        "P_all": _numpy(capture["H2_refresh_p"], np.float32),
        "M_v": _numpy(capture["H2_refresh_M_v"], np.float32),
        "prediction": _numpy(capture["H2_refresh_prediction"], np.int64),
        "outgoing_weights": np.ascontiguousarray(
            np.asarray(capture["H2_outgoing_weights"], dtype=np.float64)
        ),
    }


def _refresh_hashes(values):
    return {
        name: compare_arrays(value, value)["logical_sha256"]["current"]
        for name, value in values.items()
    }


def compare_refresh(reference, observed):
    _require(set(reference) == set(REFRESH_OUTPUTS)
             and set(observed) == set(REFRESH_OUTPUTS),
             "refresh output schema mismatch")
    return {name: compare_arrays(observed[name], reference[name])
            for name in REFRESH_OUTPUTS}


def refresh_exact(comparisons):
    return all(comparisons[name]["array_equal"] for name in REFRESH_OUTPUTS)


def classify_refresh_validity(cpu_comparisons, cuda_comparisons):
    cuda_exact = refresh_exact(cuda_comparisons)
    cpu_exact = refresh_exact(cpu_comparisons)
    if cuda_exact and not cpu_exact:
        return "FINAL_REFRESH_DEVICE_BACKEND_MISMATCH"
    if cuda_exact and cpu_exact:
        return "FINAL_REFRESH_REPLAY_EXACT_DEVICE_INVARIANT"
    return "FINAL_REFRESH_REPLAY_STILL_NOT_EXACT"


def _run_native_refresh(legacy, model, views, incoming_weights, *, device):
    """Use the identical frozen legacy entrypoint with a non-shared weight list."""
    weights = list(incoming_weights)
    return _refresh_values(legacy.native_refresh(
        model, views, weights, device, EXPECTED_SEED
    ))


def _h0_artifact_payload(capture, sample_ids, source_hashes):
    states = tuple(
        {name: value.detach().cpu().clone() for name, value in state.items()}
        for state in capture["H0_pre_refresh_state_dict_cpu"]
    )
    return {
        "schema": "paper-p0-a5-r6-h0-diagnostic-state-v1",
        "full_gt_loaded": False,
        "H0_pre_refresh_state_dict_cpu": states,
        "H0_pre_refresh_model_hash": capture["H0_pre_refresh_model"]["aggregate_hash"],
        "H1_incoming_weights": [float(value) for value in capture["H1_incoming_weights"]],
        "sample_ids": _numpy(sample_ids, np.int64),
        "historical_adapter_sha256": source_hashes,
    }


def _write_h0_state_artifact(output_dir, payload):
    target = Path(str(Path(output_dir)) + ".h0_state.pt")
    _require(not target.exists(), "refusing to overwrite R6 H0 state artifact")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target, file_sha256(target)


def _partial_evidence(capture, device, artifact_path, artifact_sha256):
    return {
        "schema": SCHEMA + "-partial",
        "full_gt_loaded": False,
        "device_metadata": {"historical": str(device), "T_CPU_NATIVE": "cpu",
                            "T_CUDA_NATIVE": str(device)},
        "H0_exact_state_hash": capture["H0_pre_refresh_model"]["aggregate_hash"],
        "H3_post_update_state_hash": capture["H3_post_update_model"]["aggregate_hash"],
        "h0_state_artifact": {"path": str(artifact_path), "sha256": artifact_sha256},
        "H_REFRESH_hashes": _refresh_hashes(_historical_refresh_values(capture)),
        "T_CPU_REFRESH_hashes": None,
        "T_CUDA_REFRESH_hashes": None,
        "cpu_cuda_latent_comparisons": None,
        "cpu_cuda_q_local_comparisons": None,
        "H_vs_T_CPU_refresh": None,
        "H_vs_T_CUDA_refresh": None,
        "validity_decision": None,
        "failure_class": None,
    }


def _fail_with_partial(output_dir, partial, exc):
    partial.update({
        "failure_class": "R6_DIAGNOSTIC_EXECUTION_EXCEPTION",
        "failure_exception_type": type(exc).__name__,
        "failure_exception_message": str(exc),
    })
    _write_partial_evidence(output_dir, partial)
    raise exc


def run_audit(output_dir, device="cuda:0"):
    """Run only under explicit authorization; native replay is expensive."""
    _require(str(device) == "cuda:0", "R6 replay device is frozen at cuda:0")
    target = Path(output_dir)
    _require(not target.exists(), "refusing to overwrite R6 output")
    source_hashes = verify_frozen_legacy_files()
    legacy = load_frozen_legacy_adapter()
    historical = legacy.validate_materialized_inputs(
        HISTORICAL_INPUT_DIR, legacy.MSRC_RUNTIME_SPEC
    )
    capture = HistoricalFinalRefreshCapture()
    with instrument_historical_final_refresh(legacy, capture):
        native = legacy.prepare_native_backbone(
            historical["views"], historical["contract"], device, EXPECTED_SEED,
            legacy.MSRC_RUNTIME_SPEC,
        )
    h = capture.require_complete()
    artifact_path, artifact_sha256 = _write_h0_state_artifact(
        target, _h0_artifact_payload(h, historical["sample_ids"], source_hashes)
    )
    partial = _partial_evidence(h, device, artifact_path, artifact_sha256)
    _write_partial_evidence(target, partial)
    try:
        h_refresh = _historical_refresh_values(h)
        cpu_model = _restore_h0_model(
            native["model"], h["H0_pre_refresh_state_dict_cpu"],
            h["H0_pre_refresh_model"], device="cpu",
        )
        cuda_model = _restore_h0_model(
            native["model"], h["H0_pre_refresh_state_dict_cpu"],
            h["H0_pre_refresh_model"], device=device,
        )
        cpu_views = _views_on_device(historical["views"], "cpu")
        cuda_views = _views_on_device(historical["views"], device)
        cpu_representations = _forward_representations(cpu_model, cpu_views)
        cuda_representations = _forward_representations(cuda_model, cuda_views)
        representation_comparisons = _representation_comparisons(
            cpu_representations, cuda_representations
        )
        cpu_refresh = _run_native_refresh(
            legacy, cpu_model, cpu_views, h["H1_incoming_weights"], device="cpu"
        )
        cuda_refresh = _run_native_refresh(
            legacy, cuda_model, cuda_views, h["H1_incoming_weights"], device=device
        )
        cpu_comparisons = compare_refresh(h_refresh, cpu_refresh)
        cuda_comparisons = compare_refresh(h_refresh, cuda_refresh)
        decision = classify_refresh_validity(cpu_comparisons, cuda_comparisons)
        partial.update({
            "T_CPU_REFRESH_hashes": _refresh_hashes(cpu_refresh),
            "T_CUDA_REFRESH_hashes": _refresh_hashes(cuda_refresh),
            "cpu_cuda_latent_comparisons": representation_comparisons["latent"],
            "cpu_cuda_q_local_comparisons": representation_comparisons["q_local"],
            "H_vs_T_CPU_refresh": cpu_comparisons,
            "H_vs_T_CUDA_refresh": cuda_comparisons,
            "validity_decision": decision,
        })
        _write_partial_evidence(target, partial)
    except Exception as exc:
        _fail_with_partial(target, partial, exc)
    record = {
        "schema": SCHEMA,
        "training_seed": EXPECTED_SEED,
        "full_gt_loaded": False,
        "frozen_legacy_file_sha256": source_hashes,
        "H0_exact_state_hash": h["H0_pre_refresh_model"]["aggregate_hash"],
        "H3_post_update_state_hash": h["H3_post_update_model"]["aggregate_hash"],
        "H0_H3_state_exact": h["H0_pre_refresh_model"]["aggregate_hash"]
        == h["H3_post_update_model"]["aggregate_hash"],
        "h0_state_artifact": {"path": str(artifact_path), "sha256": artifact_sha256},
        "H_REFRESH_hashes": partial["H_REFRESH_hashes"],
        "T_CPU_REFRESH_hashes": partial["T_CPU_REFRESH_hashes"],
        "T_CUDA_REFRESH_hashes": partial["T_CUDA_REFRESH_hashes"],
        "cpu_cuda_latent_comparisons": partial["cpu_cuda_latent_comparisons"],
        "cpu_cuda_q_local_comparisons": partial["cpu_cuda_q_local_comparisons"],
        "H_vs_T_CPU_refresh": partial["H_vs_T_CPU_refresh"],
        "H_vs_T_CUDA_refresh": partial["H_vs_T_CUDA_refresh"],
        "validity_decision": partial["validity_decision"],
        "T_carrier_validity_gate": "NOT_USED_PRE_UPDATE_T_VS_POST_UPDATE_H",
        "device_metadata": partial["device_metadata"],
    }
    target.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(target / "msrc_final_refresh_replay_validity.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    run_audit(args.output_dir, args.device)


if __name__ == "__main__":
    main()
