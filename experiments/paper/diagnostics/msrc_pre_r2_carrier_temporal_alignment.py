"""P0-A5: diagnose only the MSRC pre-R2 refresh-carrier time semantics.

This wrapper is deliberately observational.  It invokes the hash-pinned legacy
adapter for the historical replay and release_core for each fresh refresh/R2/R3
operation; it does not alter either scientific implementation.
"""

import argparse
import copy
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.paper.transfer_audit.input_artifacts import load_materialized_inputs
from experiments.paper.transfer_diagnostics.materialize_msrc_current_condition_init import (
    verify_initialization,
)
from release_core.backbone.clustering import native_refresh_from_latents
from release_core.data.weak_quality import ndarray_sha256
from release_core.semantics import build_relation_semantics
from release_core.utility import build_directional_actions, compute_directional_cycle_utility

from . import p0_a3_protocol as protocol
from .msrc_g0b0_parity_audit import (
    compare_arrays,
    reconstruct_current_msrc,
    validate_reconstruction_against_manifest,
    verify_current_true_u,
)
from .msrc_native_initializer_parity_replay import (
    EXPECTED_HISTORICAL_ACTION_HASHES,
    HISTORICAL_INPUT_DIR,
    load_frozen_legacy_adapter,
    verify_frozen_legacy_files,
)
from .state_hashing import model_state_snapshot


P0_A4_R3_DIR = Path(
    "outputs/paper/diagnostics/p0_a4_msrc_native_stagewise_parity_seed20_r3"
)
P0_A4_R3_REPORT_SHA256 = (
    "efe21bd367a535a98413b8bfd0430b1d345a155ef221c69b5137de470e2e242c"
)
P0_A4_R3_LOG_SHA256 = (
    "3afd7a01fe2a96d1bbbb93d94b704616edd986c399c5b6a08dbe9491ef6d991e"
)
EXPECTED_SEED = 20
ARM_ARRAYS = ("q_local", "q_aligned", "M_v", "U_cycle", "y_gen",
              "PredRelation_true", "relation_balance_weights_true")


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _numpy(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _write_json_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _split(value):
    return SimpleNamespace(
        labeled_ids=np.asarray(value["labeled_ids"], dtype=np.int64),
        labeled_targets=np.asarray(value["labeled_targets"], dtype=np.int64),
        unlabeled_ids=np.asarray(value["unlabeled_ids"], dtype=np.int64),
    )


def _state_dict_cpu(model):
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


class HistoricalFinalRefreshCapture:
    """Records H0--H4 around the final legacy (epoch 1000) refresh."""

    def __init__(self, refresh_interval=100, final_epoch=1000):
        self.refresh_interval = int(refresh_interval)
        self.final_epoch = int(final_epoch)
        self.refresh_count = 0
        self.values = {}

    def before_refresh(self, model, view_weights):
        epoch = self.refresh_count * self.refresh_interval
        if epoch == self.final_epoch:
            self.values["H0_pre_refresh_model"] = model_state_snapshot(model)
            self.values["H0_pre_refresh_state_dict_cpu"] = _state_dict_cpu(model)
            self.values["H1_incoming_weights"] = [float(x) for x in view_weights]

    def after_refresh(self, result):
        epoch = self.refresh_count * self.refresh_interval
        if epoch == self.final_epoch:
            p_all, matches, prediction, weights = result
            self.values.update({
                "H2_refresh_p": _numpy(p_all, np.float32),
                "H2_refresh_M_v": _numpy(matches, np.float32),
                "H2_refresh_prediction": _numpy(prediction, np.int64),
                "H2_outgoing_weights": [float(x) for x in weights],
            })
        self.refresh_count += 1

    def after_coordinate(self, model, q_local, q_aligned):
        self.values.update({
            "H3_post_update_model": model_state_snapshot(model),
            "H4_final_q_local": _numpy(q_local, np.float32),
            "H4_final_q_aligned_retained_M": _numpy(q_aligned, np.float32),
        })

    def require_complete(self):
        required = (
            "H0_pre_refresh_model", "H0_pre_refresh_state_dict_cpu",
            "H1_incoming_weights", "H2_refresh_p", "H2_refresh_M_v",
            "H2_refresh_prediction", "H2_outgoing_weights",
            "H3_post_update_model", "H4_final_q_local",
            "H4_final_q_aligned_retained_M",
        )
        _require(all(name in self.values for name in required),
                 "historical epoch-1000 carrier capture is incomplete")
        return self.values


@contextmanager
def instrument_historical_final_refresh(legacy, capture):
    """Temporarily observe, then exactly restore, frozen adapter functions."""
    original_refresh = legacy.native_refresh
    original_coordinate = legacy.coordinate_snapshot

    def wrapped_refresh(model, full_views, view_weights, device, training_seed):
        capture.before_refresh(model, view_weights)
        result = original_refresh(model, full_views, view_weights, device, training_seed)
        capture.after_refresh(result)
        return result

    def wrapped_coordinate(model, full_views, matches):
        q_local, q_aligned = original_coordinate(model, full_views, matches)
        capture.after_coordinate(model, q_local, q_aligned)
        return q_local, q_aligned

    legacy.native_refresh = wrapped_refresh
    legacy.coordinate_snapshot = wrapped_coordinate
    try:
        yield capture
    finally:
        legacy.native_refresh = original_refresh
        legacy.coordinate_snapshot = original_coordinate


def _posteriors(model, full_views):
    latent_views, q_views = [], []
    with torch.no_grad():
        for autoencoder, view in zip(model.autoencoders, full_views):
            latent = autoencoder.encoder(view)
            q_local = autoencoder.clustering(latent)
            latent_views.append(_numpy(latent, np.float32))
            q_views.append(_numpy(q_local, np.float32))
    return latent_views, q_views


def _fresh_matches(model, full_views, weights):
    latent_views, q_views = _posteriors(model, full_views)
    _, matrices, _, _, _ = native_refresh_from_latents(
        latent_views, q_views, weights, n_clusters=7, random_state=EXPECTED_SEED
    )
    return np.ascontiguousarray(matrices, dtype=np.float32)


def _carrier_from_aligned(q_local, q_aligned, matrix, split):
    actions = build_directional_actions(5)
    cycle = compute_directional_cycle_utility(torch.from_numpy(q_aligned), actions)
    y_gen = _numpy(cycle["y_gen"], np.int64)
    semantics = build_relation_semantics(y_gen, split, actions)
    return {
        "q_local": _numpy(q_local, np.float32),
        "q_aligned": _numpy(q_aligned, np.float32),
        "M_v": _numpy(matrix, np.float32),
        "U_cycle": _numpy(cycle["U_cycle"], np.float32),
        "y_gen": y_gen,
        "PredRelation_true": _numpy(semantics.pred_relation, np.bool_),
        "relation_balance_weights_true": _numpy(semantics.balance_weights, np.float64),
    }


def compare_arm(left, right):
    return {name: compare_arrays(left[name], right[name]) for name in ARM_ARRAYS}


def arm_exact(comparison):
    return all(item["array_equal"] for item in comparison.values())


def classify_carrier_alignment(*, historical_exact, current_exact,
                               post_model_and_q_local_exact,
                               h_equals_f, t_equals_h, w_equals_h,
                               f_equals_w):
    """Frozen P0-A5 rule ordering; every unresolved branch fails closed."""
    if not historical_exact:
        return "HISTORICAL_CARRIER_REPLAY_NOT_EXACT"
    if not current_exact:
        return "CURRENT_RECONSTRUCTION_NOT_EXACT"
    if not post_model_and_q_local_exact:
        return "POST_NATIVE_MODEL_OR_QLOCAL_CONTRADICTION"
    if not t_equals_h:
        return "FINAL_REFRESH_REPLAY_NOT_EXACT"
    if h_equals_f:
        return "PRE_R2_CARRIER_EXACT_PARITY"
    if w_equals_h:
        return "VIEW_WEIGHT_CARRIER_RESET_MISMATCH"
    if f_equals_w:
        return "FINAL_REFRESH_TEMPORAL_STATE_MISMATCH"
    if not f_equals_w:
        return "COMBINED_FINAL_REFRESH_CARRIER_MISMATCH"
    return "FAIL_CLOSED_UNCLASSIFIED"


def _historical_h5_exact(carrier):
    observed = {name: ndarray_sha256(carrier[name])
                for name in EXPECTED_HISTORICAL_ACTION_HASHES}
    return observed == EXPECTED_HISTORICAL_ACTION_HASHES, observed



def _capture_report(capture):
    """JSON-safe H0--H4 evidence; the pre-update state dict remains in memory."""
    return {
        "H0_pre_refresh_model": capture["H0_pre_refresh_model"],
        "H1_incoming_weights": capture["H1_incoming_weights"],
        "H2_epoch1000_refresh": {
            "p_logical_sha256": ndarray_sha256(capture["H2_refresh_p"]),
            "M_v_logical_sha256": ndarray_sha256(capture["H2_refresh_M_v"]),
            "prediction_logical_sha256": ndarray_sha256(capture["H2_refresh_prediction"]),
            "outgoing_weights": capture["H2_outgoing_weights"],
        },
        "H3_post_update_model": capture["H3_post_update_model"],
        "H4_final_coordinate_snapshot": {
            "q_local_logical_sha256": ndarray_sha256(capture["H4_final_q_local"]),
            "q_aligned_retained_M_logical_sha256": ndarray_sha256(capture["H4_final_q_aligned_retained_M"]),
        },
    }

def _build_cpu_clone(final_model, state_dict):
    clone = copy.deepcopy(final_model).cpu()
    clone.load_state_dict(state_dict, strict=True)
    clone.train()
    return clone


def _verify_p0_a4_r3_evidence():
    from experiments.paper.transfer_audit.input_artifacts import file_sha256
    report = P0_A4_R3_DIR / "msrc_native_stagewise_parity.json"
    log = P0_A4_R3_DIR.parent / (P0_A4_R3_DIR.name + ".log")
    _require(file_sha256(report) == P0_A4_R3_REPORT_SHA256,
             "P0-A4 r3 report identity mismatch")
    _require(file_sha256(log) == P0_A4_R3_LOG_SHA256,
             "P0-A4 r3 log identity mismatch")
    with report.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    _require(record.get("decision") == "EXACT_INITIALIZATION_PARITY"
             and record.get("last_exact_stage") == "S14_FINAL_PRE_R2",
             "P0-A4 r3 exact-initialization evidence mismatch")
    return {"report_sha256": P0_A4_R3_REPORT_SHA256,
            "log_sha256": P0_A4_R3_LOG_SHA256,
            "decision": record["decision"],
            "last_exact_stage": record["last_exact_stage"]}


def run_audit(output_dir, device="cuda:0"):
    """Run only when explicitly authorized; full historical replay is expensive."""
    _require(str(device) == "cuda:0", "P0-A5 replay device is frozen at cuda:0")
    target = Path(output_dir)
    _require(not target.exists(), "refusing to overwrite P0-A5 output")
    r3_evidence = _verify_p0_a4_r3_evidence()
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
    historical_split = _split(historical["split"])
    H = _carrier_from_aligned(
        h["H4_final_q_local"], h["H4_final_q_aligned_retained_M"],
        h["H2_refresh_M_v"], historical_split,
    )
    historical_exact, historical_hashes = _historical_h5_exact(H)
    _require(historical_exact, "HISTORICAL_CARRIER_REPLAY_NOT_EXACT")

    manifest = verify_current_true_u()
    F, current_split = reconstruct_current_msrc(device)
    current_identity = validate_reconstruction_against_manifest(F, manifest)
    _require(all(item["exact"] for item in current_identity.values()),
             "CURRENT_RECONSTRUCTION_NOT_EXACT")
    initialization = verify_initialization(protocol.MSRC_INIT_DIR)
    current_model = initialization["model"]
    current_model.to_device(device)
    for autoencoder in current_model.autoencoders:
        autoencoder.train()
    _, current_views, _, _ = load_materialized_inputs(protocol.MSRC_INPUT_DIR)
    current_full_views = [torch.from_numpy(view).to(device) for view in current_views]
    current_snapshot = model_state_snapshot(current_model)
    q_current_local, _ = legacy.coordinate_snapshot(
        current_model, current_full_views,
        torch.from_numpy(h["H2_refresh_M_v"]).to(device),
    )
    post_comparisons = {
        "model": {"historical": h["H3_post_update_model"]["aggregate_hash"],
                  "current": current_snapshot["aggregate_hash"],
                  "exact": h["H3_post_update_model"]["aggregate_hash"] == current_snapshot["aggregate_hash"]},
        "q_local": compare_arrays(_numpy(q_current_local, np.float32), H["q_local"]),
    }
    post_exact = post_comparisons["model"]["exact"] and post_comparisons["q_local"]["array_equal"]
    _require(post_exact, "POST_NATIVE_MODEL_OR_QLOCAL_CONTRADICTION")

    W_matrix = _fresh_matches(current_model, current_full_views, h["H1_incoming_weights"])
    W_local, W_aligned = legacy.coordinate_snapshot(
        current_model, current_full_views, torch.from_numpy(W_matrix).to(device)
    )
    W = _carrier_from_aligned(W_local, W_aligned, W_matrix, current_split)
    pre_model = _build_cpu_clone(native["model"], h["H0_pre_refresh_state_dict_cpu"])
    historical_cpu_views = [torch.from_numpy(np.ascontiguousarray(view))
                            for view in historical["views"]]
    T_matrix = _fresh_matches(pre_model, historical_cpu_views, h["H1_incoming_weights"])
    T_local, T_aligned = legacy.coordinate_snapshot(
        pre_model, historical_cpu_views, torch.from_numpy(T_matrix)
    )
    T = _carrier_from_aligned(T_local, T_aligned, T_matrix, historical_split)
    comparisons = {"H_vs_F": compare_arm(H, F), "H_vs_W": compare_arm(H, W),
                   "H_vs_T": compare_arm(H, T), "F_vs_W": compare_arm(F, W)}
    decision = classify_carrier_alignment(
        historical_exact=historical_exact, current_exact=True,
        post_model_and_q_local_exact=post_exact,
        h_equals_f=arm_exact(comparisons["H_vs_F"]),
        t_equals_h=arm_exact(comparisons["H_vs_T"]),
        w_equals_h=arm_exact(comparisons["H_vs_W"]),
        f_equals_w=arm_exact(comparisons["F_vs_W"]),
    )
    _require(decision != "FINAL_REFRESH_REPLAY_NOT_EXACT",
             "FINAL_REFRESH_REPLAY_NOT_EXACT")
    record = {
        "schema": "paper-p0-a5-msrc-pre-r2-carrier-temporal-alignment-v1",
        "training_seed": EXPECTED_SEED, "full_gt_loaded": False,
        "p0_a4_r3_exact_evidence": r3_evidence,
        "frozen_legacy_file_sha256": source_hashes,
        "historical_H0_H4": _capture_report(h),
        "historical_h5_action_hashes": historical_hashes,
        "expected_historical_h5_action_hashes": EXPECTED_HISTORICAL_ACTION_HASHES,
        "historical_q_carrier_semantics": "post-final q_local aligned with retained epoch-1000 pre-update refresh M_v",
        "current_f_semantics": "post-final model fresh refresh with reset all-ones weights",
        "post_native_model_and_q_local": post_comparisons,
        "comparisons": comparisons, "decision": decision,
    }
    target.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(target / "msrc_pre_r2_carrier_temporal_alignment.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    run_audit(args.output_dir, args.device)


if __name__ == "__main__":
    main()
