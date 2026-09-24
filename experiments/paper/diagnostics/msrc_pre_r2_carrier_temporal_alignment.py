"""P0-A5: diagnose only the MSRC pre-R2 refresh-carrier time semantics.

This wrapper is deliberately observational.  It invokes the hash-pinned legacy
adapter for the historical replay and release_core for each fresh refresh/R2/R3
operation; it does not alter either scientific implementation.
"""

import argparse
import copy
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from experiments.paper.transfer_audit.input_artifacts import load_materialized_inputs
from experiments.paper.transfer_diagnostics import materialize_msrc_current_condition_init as current_initializer
from experiments.paper.transfer_diagnostics.materialize_msrc_current_condition_init import (
    verify_initialization,
)
from release_core.backbone.clustering import native_refresh_from_latents
from release_core.data.weak_quality import ndarray_sha256
from release_core.semantics import (
    SparseLabelSplit,
    build_relation_semantics,
    validate_sparse_label_split,
)
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
P0_A4_R3_SCHEMA = "paper-msrc-p0-a4-stagewise-parity-v1"
P0_A4_R3_DATASET = "MSRC"
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



def _write_partial_evidence(output_dir, evidence):
    """Atomically update evidence without creating the formal output directory."""
    target = Path(str(Path(output_dir)) + ".partial.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(evidence, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target

def _historical_sparse_split(historical, legacy):
    """Construct the release R3 contract from validated historical inputs."""
    split = historical["split"]
    result = SparseLabelSplit(
        sample_ids=np.asarray(historical["sample_ids"], dtype=np.int64),
        labeled_ids=np.asarray(split["labeled_ids"], dtype=np.int64),
        labeled_targets=np.asarray(split["labeled_targets"], dtype=np.int64),
        unlabeled_ids=np.asarray(split["unlabeled_ids"], dtype=np.int64),
        class_count=int(historical["contract"].K),
        labels_per_class=int(historical["contract"].labels_per_class),
        label_seed=EXPECTED_SEED,
        dataset_name=legacy.MSRC_RUNTIME_SPEC.dataset_name,
    )
    return validate_sparse_label_split(result)


def _sparse_split_exact(left, right):
    left = validate_sparse_label_split(left)
    right = validate_sparse_label_split(right)
    arrays = ("sample_ids", "labeled_ids", "labeled_targets", "unlabeled_ids")
    metadata = ("class_count", "labels_per_class", "label_seed", "dataset_name")
    return (
        all(np.array_equal(getattr(left, name), getattr(right, name))
            for name in arrays)
        and all(getattr(left, name) == getattr(right, name) for name in metadata)
    )


def _require_sparse_split_parity(historical_split, current_split):
    _require(_sparse_split_exact(historical_split, current_split),
             "HISTORICAL_CURRENT_SPARSE_SPLIT_CONTRACT_MISMATCH")

def _state_dict_cpu(model):
    autoencoders = tuple(model.autoencoders)
    states = tuple(
        {name: value.detach().cpu().clone()
         for name, value in autoencoder.state_dict().items()}
        for autoencoder in autoencoders
    )
    _require(len(states) == len(autoencoders) and all(states),
             "H0 per-view state capture is incomplete")
    return states


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


def _carrier_from_aligned(q_local, q_aligned, matrix, split, *, device):
    actions = build_directional_actions(5)
    if isinstance(q_aligned, torch.Tensor):
        q_tensor = q_aligned.detach().to(
            device=device, dtype=torch.float32
        ).contiguous()
    else:
        q_tensor = torch.from_numpy(
            np.ascontiguousarray(q_aligned, dtype=np.float32)
        ).to(device)
    cycle = compute_directional_cycle_utility(q_tensor, actions)
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



def _partial_evidence(capture, device):
    return {
        "schema": "paper-p0-a5-partial-evidence-v1",
        "stage_reached": "H0_H4_CAPTURED",
        "failure_class": None,
        "device": str(device),
        "full_gt_loaded": False,
        "historical_H0_H4": _capture_report(capture),
        "observed_historical_H_hashes": None,
        "expected_historical_H_hashes": EXPECTED_HISTORICAL_ACTION_HASHES,
        "current_provenance_gate": "NOT_RUN",
        "completed_arms": [],
    }


def _fail_with_partial(output_dir, evidence, failure_class):
    evidence["failure_class"] = failure_class
    _write_partial_evidence(output_dir, evidence)
    raise RuntimeError(failure_class)




def _diagnostic_exception_failure_class(exc):
    message = str(exc).lower()
    if "cuda" in message and "tensor" in message and "numpy" in message:
        return "DIAGNOSTIC_TENSOR_NUMPY_BOUNDARY_MISMATCH"
    return "DIAGNOSTIC_EXECUTION_EXCEPTION"


def _carrier_with_partial_evidence(
        q_local, q_aligned, matrix, split, *, device, output_dir, evidence,
        stage_reached, completed_arms):
    """Run one diagnostic R2/R3 carrier and persist unexpected failures."""
    try:
        return _carrier_from_aligned(
            q_local, q_aligned, matrix, split, device=device
        )
    except Exception as exc:
        if evidence.get("failure_class") is None:
            evidence.update({
                "failure_class": _diagnostic_exception_failure_class(exc),
                "failure_exception_type": type(exc).__name__,
                "failure_exception_message": str(exc),
                "stage_reached": stage_reached,
                "completed_arms": list(completed_arms),
            })
            _write_partial_evidence(output_dir, evidence)
        raise

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
            "q_aligned_retained_M_logical_sha256": ndarray_sha256(
                capture["H4_final_q_aligned_retained_M"]
            ),
        },
    }

def _snapshot_exact(actual, expected):
    required = (
        "model_hash", "encoder_parameter_hash", "decoder_parameter_hash",
        "cluster_centers_hash",
    )
    return (
        actual["aggregate_hash"] == expected["aggregate_hash"]
        and len(actual["per_view"]) == len(expected["per_view"])
        and all(
            current["view_id"] == frozen["view_id"]
            and all(current[name] == frozen[name] for name in required)
            for current, frozen in zip(actual["per_view"], expected["per_view"])
        )
    )

def _build_cpu_clone(final_model, per_view_state, expected_snapshot):
    clone = copy.deepcopy(final_model)
    autoencoders = tuple(clone.autoencoders)
    _require(len(per_view_state) == len(autoencoders),
             "H0 per-view state count mismatch")
    for autoencoder, state in zip(autoencoders, per_view_state):
        _require(isinstance(state, dict) and state,
                 "H0 per-view state is missing or empty")
        autoencoder.cpu()
        result = autoencoder.load_state_dict(state, strict=True)
        _require(not result.missing_keys and not result.unexpected_keys,
                 "H0 per-view strict restore mismatch")
        autoencoder.train()
        _require(
            all(value.device.type == "cpu" for value in (
                tuple(autoencoder.parameters()) + tuple(autoencoder.buffers())
            )) and autoencoder.training,
            "H0 restored autoencoder CPU/training contract mismatch",
        )
    restored = model_state_snapshot(clone)
    _require(_snapshot_exact(restored, expected_snapshot),
             "RESTORED_H0_MODEL_NOT_EXACT")
    return clone


def _validate_p0_a4_r3_record(record, release_source_sha256):
    """Accept only the frozen nested first-divergence schema."""
    _require(isinstance(record, dict), "P0-A4 r3 report must be a JSON object")
    _require(
        record["schema"] == P0_A4_R3_SCHEMA
        and record["dataset"] == P0_A4_R3_DATASET
        and record["training_seed"] == EXPECTED_SEED
        and record["historical_training_seed"] == EXPECTED_SEED
        and record["release_code_modified"] is False
        and record["release_core_source_sha256"] == release_source_sha256,
        "P0-A4 r3 root evidence metadata mismatch",
    )
    first = record.get("first_divergence")
    _require(isinstance(first, dict),
             "P0-A4 r3 first_divergence must be an object")
    _require(
        first["decision"] == "EXACT_INITIALIZATION_PARITY"
        and first["last_exact_stage"] == "S14_FINAL_PRE_R2"
        and first["first_divergent_stage"] is None
        and first["first_divergent_components"] == []
        and first["first_divergent_paths"] == [],
        "P0-A4 r3 exact-initialization evidence mismatch",
    )
    return first


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
    first = _validate_p0_a4_r3_record(
        record, current_initializer._release_source_hashes()
    )
    return {"report_sha256": P0_A4_R3_REPORT_SHA256,
            "log_sha256": P0_A4_R3_LOG_SHA256,
            "schema": record["schema"],
            "dataset": record["dataset"],
            "training_seed": record["training_seed"],
            "historical_training_seed": record["historical_training_seed"],
            "release_core_source_sha256": record["release_core_source_sha256"],
            "decision": first["decision"],
            "last_exact_stage": first["last_exact_stage"]}



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
    partial = _partial_evidence(h, device)
    _write_partial_evidence(target, partial)
    try:
        historical_split = _historical_sparse_split(historical, legacy)
        _, current_views, _, materialized_current_split = load_materialized_inputs(
            protocol.MSRC_INPUT_DIR
        )
        current_split = validate_sparse_label_split(materialized_current_split)
        if not _sparse_split_exact(historical_split, current_split):
            _fail_with_partial(
                target, partial, "HISTORICAL_CURRENT_SPARSE_SPLIT_CONTRACT_MISMATCH"
            )
        H = _carrier_with_partial_evidence(
            h["H4_final_q_local"], h["H4_final_q_aligned_retained_M"],
            h["H2_refresh_M_v"], historical_split, device=device, output_dir=target,
            evidence=partial, stage_reached="H_R2_R3_STARTED", completed_arms=[],
        )
        historical_exact, historical_hashes = _historical_h5_exact(H)
        partial.update({
            "stage_reached": "H_R2_R3_COMPLETED",
            "observed_historical_H_hashes": historical_hashes,
            "completed_arms": ["H"],
        })
        _write_partial_evidence(target, partial)
        if not historical_exact:
            _fail_with_partial(target, partial, "HISTORICAL_CARRIER_REPLAY_NOT_EXACT")

        manifest = verify_current_true_u()
        F_PROVENANCE, reconstructed_current_split = reconstruct_current_msrc(device)
        if not _sparse_split_exact(current_split, reconstructed_current_split):
            _fail_with_partial(
                target, partial, "HISTORICAL_CURRENT_SPARSE_SPLIT_CONTRACT_MISMATCH"
            )
        current_identity = validate_reconstruction_against_manifest(
            F_PROVENANCE, manifest
        )
        if not all(item["exact"] for item in current_identity.values()):
            partial["current_provenance_gate"] = {
                "state": "FAILED", "checks": current_identity,
            }
            _fail_with_partial(target, partial, "CURRENT_RECONSTRUCTION_NOT_EXACT")
        F_ARM = _carrier_with_partial_evidence(
            F_PROVENANCE["q_local"], F_PROVENANCE["q_aligned"],
            F_PROVENANCE["M_v"], current_split, device=device, output_dir=target,
            evidence=partial, stage_reached="F_PROVENANCE_AND_F_ARM_STARTED",
            completed_arms=["H"],
        )
        partial.update({
            "stage_reached": "F_PROVENANCE_AND_F_ARM_COMPLETED",
            "current_provenance_gate": {"state": "PASS", "checks": current_identity},
            "completed_arms": ["H", "F"],
        })
        _write_partial_evidence(target, partial)
        initialization = verify_initialization(protocol.MSRC_INIT_DIR)
        current_model = initialization["model"]
        current_model.to_device(device)
        for autoencoder in current_model.autoencoders:
            autoencoder.train()
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
        if not post_exact:
            _fail_with_partial(target, partial, "POST_NATIVE_MODEL_OR_QLOCAL_CONTRADICTION")

        W_matrix = _fresh_matches(current_model, current_full_views, h["H1_incoming_weights"])
        W_local, W_aligned = legacy.coordinate_snapshot(
            current_model, current_full_views, torch.from_numpy(W_matrix).to(device)
        )
        W = _carrier_with_partial_evidence(
            W_local, W_aligned, W_matrix, current_split, device=device,
            output_dir=target, evidence=partial, stage_reached="W_R2_R3_STARTED",
            completed_arms=["H", "F"],
        )
        pre_model = _build_cpu_clone(
            native["model"], h["H0_pre_refresh_state_dict_cpu"],
            h["H0_pre_refresh_model"],
        )
        historical_cpu_views = [torch.from_numpy(np.ascontiguousarray(view))
                                for view in historical["views"]]
        T_matrix = _fresh_matches(pre_model, historical_cpu_views, h["H1_incoming_weights"])
        T_local, T_aligned = legacy.coordinate_snapshot(
            pre_model, historical_cpu_views, torch.from_numpy(T_matrix)
        )
        T = _carrier_with_partial_evidence(
            T_local, T_aligned, T_matrix, historical_split, device=device,
            output_dir=target, evidence=partial, stage_reached="T_R2_R3_STARTED",
            completed_arms=["H", "F", "W"],
        )
        comparisons = {"H_vs_F": compare_arm(H, F_ARM), "H_vs_W": compare_arm(H, W),
                       "H_vs_T": compare_arm(H, T), "F_vs_W": compare_arm(F_ARM, W)}
        decision = classify_carrier_alignment(
            historical_exact=historical_exact, current_exact=True,
            post_model_and_q_local_exact=post_exact,
            h_equals_f=arm_exact(comparisons["H_vs_F"]),
            t_equals_h=arm_exact(comparisons["H_vs_T"]),
            w_equals_h=arm_exact(comparisons["H_vs_W"]),
            f_equals_w=arm_exact(comparisons["F_vs_W"]),
        )
        if decision == "FINAL_REFRESH_REPLAY_NOT_EXACT":
            _fail_with_partial(target, partial, "FINAL_REFRESH_REPLAY_NOT_EXACT")
        partial.update({"stage_reached": "H_F_W_T_COMPLETED",
                        "completed_arms": ["H", "F", "W", "T"]})
        _write_partial_evidence(target, partial)
    except Exception as exc:
        if partial.get("failure_class") is None:
            partial.update({
                "failure_class": _diagnostic_exception_failure_class(exc),
                "failure_exception_type": type(exc).__name__,
                "failure_exception_message": str(exc),
            })
            _write_partial_evidence(target, partial)
        raise

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
        "current_provenance_gate": current_identity,
        "F_ARM_semantics": "F_PROVENANCE q_local/q_aligned/M_v recomputed through common diagnostic device",
        "common_diagnostic_device": str(device),
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
