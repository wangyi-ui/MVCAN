"""P0-A5-R7: cheap H/F/W carrier root-cause audit after R6 validation.

This diagnostic intentionally never replays historical training.  It loads the
frozen pre-R2 H carrier and gates all work on the successful R6 CUDA refresh
validity evidence.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.paper.transfer_audit.input_artifacts import (
    file_sha256,
    load_materialized_inputs,
)
from experiments.paper.transfer_diagnostics.materialize_msrc_current_condition_init import (
    verify_initialization,
)
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
from .msrc_pre_r2_carrier_temporal_alignment import (
    ARM_ARRAYS,
    EXPECTED_SEED,
    _carrier_from_aligned,
    _historical_h5_exact,
    _historical_sparse_split,
    _numpy,
    _snapshot_exact,
    _write_json_exclusive,
    _write_partial_evidence,
    _fresh_matches,
)
from .state_hashing import model_state_snapshot


SCHEMA = "paper-p0-a5-r7-hfw-carrier-root-cause-v1"
R6_DIR = Path("outputs/paper/diagnostics/p0_a5_msrc_final_refresh_replay_validity_seed20_r6")
R6_RESULT = R6_DIR / "msrc_final_refresh_replay_validity.json"
R6_H0_ARTIFACT = Path(str(R6_DIR) + ".h0_state.pt")
R6_H0_ARTIFACT_SHA256 = "e042cc9144028cb9f5dc540fe88d2ce865c050486c8e22c7b17a278c4754196a"
HISTORICAL_H_STATE = Path(
    "outputs/paper/diagnostics/p0_a3_cross_dataset_seed20/"
    "msrc_native_initializer_parity_replay_r1/legacy_pre_r2_state.npz"
)
HISTORICAL_H_STATE_SHA256 = "6f2b433a3060f33775553292fce2eb89794f62c45a2cfabce8abc64cf83a3f14"
EXPECTED_H1_WEIGHTS = np.asarray([
    1.1124446711235916, 1.6307825636400173, 1.9014181452555536,
    1.2324449985302548, 1.2048197853115608,
], dtype=np.float64)
EXPECTED_H3_HASH = "05fe4712e0ce9f2ed62709265798a8ee62775283dfdde5417a833049863d6c73"
R7_ARRAYS = (
    "M_v", "q_local", "q_aligned", "U_cycle", "y_gen",
    "PredRelation_true", "relation_balance_weights_true",
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _progress(message):
    print("[R7] " + message, flush=True)


def _all_finite(value):
    return bool(np.isfinite(np.asarray(value)).all())


def _carrier_hashes(carrier):
    return {
        name: compare_arrays(carrier[name], carrier[name])["logical_sha256"]["current"]
        for name in R7_ARRAYS
    }


def compare_arm(left, right):
    return {name: compare_arrays(left[name], right[name]) for name in R7_ARRAYS}


def arm_exact(comparisons):
    return all(comparisons[name]["array_equal"] for name in R7_ARRAYS)


def first_divergent_component(comparisons):
    for name in R7_ARRAYS:
        if not comparisons[name]["array_equal"]:
            return name
    return None


def classify_hfw_carrier_root_cause(h_vs_f, h_vs_w, w_vs_f):
    h_f, h_w, w_f = (arm_exact(value) for value in (h_vs_f, h_vs_w, w_vs_f))
    if h_f:
        return "PRE_R2_CARRIER_EXACT_PARITY"
    if h_w and not w_f:
        return "VIEW_WEIGHT_CARRIER_RESET_MISMATCH"
    if w_f and not h_w:
        return "FINAL_REFRESH_TEMPORAL_STATE_MISMATCH"
    if not h_w and not w_f:
        return "COMBINED_FINAL_REFRESH_CARRIER_MISMATCH"
    return "FAIL_CLOSED_UNCLASSIFIED"


def _action_statistics(carrier):
    utility = np.asarray(carrier["U_cycle"], dtype=np.float64)
    return {
        "mean": float(utility.mean()) if utility.size else 0.0,
        "std": float(utility.std()) if utility.size else 0.0,
        "min": float(utility.min()) if utility.size else 0.0,
        "max": float(utility.max()) if utility.size else 0.0,
        "nonzero_fraction": float(np.count_nonzero(utility) / utility.size)
        if utility.size else 0.0,
    }


def _difference_statistics(left, right):
    left_array, right_array = np.asarray(left), np.asarray(right)
    _require(left_array.shape == right_array.shape, "action diagnostic shape mismatch")
    changed = left_array != right_array
    absolute = np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))
    return {
        "max_abs_diff": float(absolute.max()) if absolute.size else 0.0,
        "mean_abs_diff": float(absolute.mean()) if absolute.size else 0.0,
        "changed_entry_count": int(np.count_nonzero(changed)),
        "changed_entry_fraction": float(np.count_nonzero(changed) / changed.size)
        if changed.size else 0.0,
    }


def action_diagnostics(h_arm, f_arm, w_arm):
    pairs = {"H_vs_F": (h_arm, f_arm), "H_vs_W": (h_arm, w_arm),
             "W_vs_F": (w_arm, f_arm)}
    return {
        "U_cycle": {"H": _action_statistics(h_arm), "F": _action_statistics(f_arm),
                    "W": _action_statistics(w_arm),
                    "pairs": {name: _difference_statistics(left["U_cycle"], right["U_cycle"])
                              for name, (left, right) in pairs.items()}},
        "y_gen": {name: _difference_statistics(left["y_gen"], right["y_gen"])
                  for name, (left, right) in pairs.items()},
        "PredRelation_true": {name: _difference_statistics(
            left["PredRelation_true"], right["PredRelation_true"]
        ) for name, (left, right) in pairs.items()},
    }


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    _require(isinstance(value, dict), "R6 result must be a JSON object")
    return value


def validate_r6_evidence(result_path=R6_RESULT, artifact_path=R6_H0_ARTIFACT):
    result_path, artifact_path = Path(result_path), Path(artifact_path)
    _require(result_path.is_file() and artifact_path.is_file(),
             "R6_REFRESH_VALIDITY_EVIDENCE_NOT_EXACT")
    result_sha, artifact_sha = file_sha256(result_path), file_sha256(artifact_path)
    record = _read_json(result_path)
    _require(
        record.get("schema") == "paper-p0-a5-r6-final-refresh-replay-validity-v1"
        and record.get("full_gt_loaded") is False
        and record.get("validity_decision") == "FINAL_REFRESH_DEVICE_BACKEND_MISMATCH",
        "R6_REFRESH_VALIDITY_EVIDENCE_NOT_EXACT",
    )
    comparisons = record.get("H_vs_T_CUDA_refresh")
    _require(isinstance(comparisons, dict) and all(
        isinstance(comparisons.get(name), dict)
        and comparisons[name].get("array_equal") is True
        for name in ("P_all", "M_v", "prediction", "outgoing_weights")
    ), "R6_REFRESH_VALIDITY_EVIDENCE_NOT_EXACT")
    metadata = record.get("h0_state_artifact")
    _require(isinstance(metadata, dict) and metadata.get("sha256") == artifact_sha
             and artifact_sha == R6_H0_ARTIFACT_SHA256,
             "R6_REFRESH_VALIDITY_EVIDENCE_NOT_EXACT")
    artifact = torch.load(artifact_path, map_location="cpu")
    _require(isinstance(artifact, dict), "R6_REFRESH_VALIDITY_EVIDENCE_NOT_EXACT")
    required = {"schema", "full_gt_loaded", "H1_incoming_weights", "sample_ids",
                "H0_pre_refresh_state_dict_cpu", "historical_adapter_sha256"}
    _require(required.issubset(artifact) and artifact["full_gt_loaded"] is False,
             "R6_REFRESH_VALIDITY_EVIDENCE_NOT_EXACT")
    weights = np.asarray(artifact["H1_incoming_weights"], dtype=np.float64)
    _require(weights.shape == (5,) and _all_finite(weights)
             and np.array_equal(weights, EXPECTED_H1_WEIGHTS),
             "R6_REFRESH_VALIDITY_EVIDENCE_NOT_EXACT")
    return {"record": record, "artifact": artifact, "result_sha256": result_sha,
            "artifact_sha256": artifact_sha, "historical_weights": weights}


def load_historical_h_arm(legacy, *, device):
    _require(HISTORICAL_H_STATE.is_file()
             and file_sha256(HISTORICAL_H_STATE) == HISTORICAL_H_STATE_SHA256,
             "HISTORICAL_H_ACTION_CARRIER_NOT_EXACT")
    with np.load(HISTORICAL_H_STATE, allow_pickle=False) as archive:
        _require(set(archive.files) == {"q_local", "q_aligned", "M_v"},
                 "HISTORICAL_H_ACTION_CARRIER_NOT_EXACT")
        q_local = _numpy(archive["q_local"], np.float32)
        q_aligned = _numpy(archive["q_aligned"], np.float32)
        matrix = _numpy(archive["M_v"], np.float32)
    _require(q_local.shape == (210, 5, 7) and q_aligned.shape == (210, 5, 7)
             and matrix.shape == (5, 7, 7)
             and q_local.dtype == q_aligned.dtype == matrix.dtype == np.float32
             and _all_finite(q_local) and _all_finite(q_aligned) and _all_finite(matrix),
             "HISTORICAL_H_ACTION_CARRIER_NOT_EXACT")
    historical = legacy.validate_materialized_inputs(
        HISTORICAL_INPUT_DIR, legacy.MSRC_RUNTIME_SPEC
    )
    split = _historical_sparse_split(historical, legacy)
    h_arm = _carrier_from_aligned(q_local, q_aligned, matrix, split, device=device)
    exact, hashes = _historical_h5_exact(h_arm)
    _require(exact, "HISTORICAL_H_ACTION_CARRIER_NOT_EXACT")
    return h_arm, split, hashes


def _started_partial():
    return {
        "schema": SCHEMA + "-partial",
        "stage_reached": "R7_STARTED",
        "completed_arms": [],
        "full_gt_loaded": False,
        "failure_class": None,
        "R6_decision": None,
        "R6_result_sha256": None,
        "R6_h0_artifact_sha256": None,
        "historical_H_state_sha256": None,
        "historical_H_action_hashes": None,
        "F_provenance_gate": None,
        "F_replay_ones_parity": None,
        "post_final_parity": None,
        "carrier_hashes": {},
        "comparisons": None,
        "decision": None,
    }


def _partial_evidence(r6):
    partial = _started_partial()
    partial.update({
        "R6_decision": r6["record"]["validity_decision"],
        "R6_result_sha256": r6["result_sha256"],
        "R6_h0_artifact_sha256": r6["artifact_sha256"],
    })
    return partial


def _fail_with_partial(output_dir, partial, failure_class):
    partial["failure_class"] = failure_class
    _write_partial_evidence(output_dir, partial)
    raise RuntimeError(failure_class)


def _fresh_arm(legacy, model, full_views, weights, split, *, device):
    matrix = _fresh_matches(model, full_views, list(weights))
    local, aligned = legacy.coordinate_snapshot(
        model, full_views, torch.from_numpy(matrix).to(device)
    )
    return _carrier_from_aligned(local, aligned, matrix, split, device=device)


def run_audit(output_dir, device="cuda:0"):
    """Run only under explicit authorization; no historical training is invoked."""
    _require(str(device) == "cuda:0", "R7 device is frozen at cuda:0")
    target = Path(output_dir)
    _require(not target.exists(), "refusing to overwrite R7 output")
    _progress("START")
    partial = _started_partial()
    _write_partial_evidence(target, partial)
    try:
        r6 = validate_r6_evidence()
        partial = _partial_evidence(r6)
        partial["stage_reached"] = "R6_EVIDENCE_VALIDATED"
        _write_partial_evidence(target, partial)
        _progress("R6 evidence validated")
        source_hashes = verify_frozen_legacy_files()
        legacy = load_frozen_legacy_adapter()
        h_arm, historical_split, historical_hashes = load_historical_h_arm(
            legacy, device=device
        )
        partial.update({
            "stage_reached": "H_ARM_VALIDATED",
            "completed_arms": ["H"],
            "historical_H_state_sha256": HISTORICAL_H_STATE_SHA256,
            "historical_H_action_hashes": historical_hashes,
            "carrier_hashes": {"H": _carrier_hashes(h_arm)},
        })
        _write_partial_evidence(target, partial)
        _progress("Historical H validated")

        manifest = verify_current_true_u()
        f_provenance, current_split = reconstruct_current_msrc(device)
        f_gate = validate_reconstruction_against_manifest(f_provenance, manifest)
        if not all(item["exact"] for item in f_gate.values()):
            partial["F_provenance_gate"] = {"state": "FAILED", "checks": f_gate}
            _fail_with_partial(target, partial, "CURRENT_RECONSTRUCTION_NOT_EXACT")
        partial.update({"stage_reached": "F_PROVENANCE_VALIDATED",
                        "F_provenance_gate": {"state": "PASS", "checks": f_gate}})
        _write_partial_evidence(target, partial)
        _progress("Current F provenance validated")
        f_arm = _carrier_from_aligned(
            f_provenance["q_local"], f_provenance["q_aligned"],
            f_provenance["M_v"], current_split, device=device,
        )
        partial.update({"stage_reached": "F_ARM_VALIDATED",
                        "completed_arms": ["H", "F"],
                        "carrier_hashes": {**partial["carrier_hashes"],
                                           "F": _carrier_hashes(f_arm)}})
        _write_partial_evidence(target, partial)

        _, current_views, _, materialized_split = load_materialized_inputs(protocol.MSRC_INPUT_DIR)
        _require(np.array_equal(current_split.sample_ids, materialized_split.sample_ids),
                 "CURRENT_SPARSE_SPLIT_CONTRACT_MISMATCH")
        initialization = verify_initialization(protocol.MSRC_INIT_DIR)
        current_model = initialization["model"]
        current_model.to_device(device)
        for autoencoder in current_model.autoencoders:
            autoencoder.train()
        current_snapshot = model_state_snapshot(current_model)
        if current_snapshot["aggregate_hash"] != EXPECTED_H3_HASH:
            _fail_with_partial(target, partial, "POST_FINAL_MODEL_OR_H_CARRIER_PARITY_FAILED")
        current_views_cuda = [torch.from_numpy(view).to(device) for view in current_views]

        ones = [1.0] * 5
        f_replay_ones = _fresh_arm(
            legacy, current_model, current_views_cuda, ones, current_split, device=device
        )
        f_replay_comparison = compare_arm(f_replay_ones, f_arm)
        if not arm_exact(f_replay_comparison):
            partial["F_replay_ones_parity"] = f_replay_comparison
            _fail_with_partial(target, partial, "CURRENT_FRESH_REFRESH_REPLAY_NOT_EXACT")
        partial.update({"stage_reached": "F_REPLAY_ONES_VALIDATED",
                        "F_replay_ones_parity": f_replay_comparison})
        _write_partial_evidence(target, partial)
        _progress("F replay-ones parity validated")

        h_matrix = torch.from_numpy(h_arm["M_v"]).to(device)
        current_local_h, current_aligned_h = legacy.coordinate_snapshot(
            current_model, current_views_cuda, h_matrix
        )
        post_parity = {
            "model_h3_hash_exact": current_snapshot["aggregate_hash"] == EXPECTED_H3_HASH,
            "q_local": compare_arrays(_numpy(current_local_h, np.float32), h_arm["q_local"]),
            "q_aligned": compare_arrays(_numpy(current_aligned_h, np.float32), h_arm["q_aligned"]),
        }
        if not (post_parity["model_h3_hash_exact"] and post_parity["q_local"]["array_equal"]
                and post_parity["q_aligned"]["array_equal"]):
            partial["post_final_parity"] = post_parity
            _fail_with_partial(target, partial, "POST_FINAL_MODEL_OR_H_CARRIER_PARITY_FAILED")
        partial.update({"stage_reached": "POST_FINAL_PARITY_VALIDATED",
                        "post_final_parity": post_parity})
        _write_partial_evidence(target, partial)
        _progress("Post-final H parity validated")

        w_arm = _fresh_arm(
            legacy, current_model, current_views_cuda, r6["historical_weights"],
            current_split, device=device,
        )
        partial.update({"stage_reached": "W_ARM_BUILT",
                        "completed_arms": ["H", "F", "W"],
                        "carrier_hashes": {**partial["carrier_hashes"],
                                           "W": _carrier_hashes(w_arm)}})
        _write_partial_evidence(target, partial)
        _progress("W arm built")

        comparisons = {"H_vs_F": compare_arm(h_arm, f_arm),
                       "H_vs_W": compare_arm(h_arm, w_arm),
                       "W_vs_F": compare_arm(w_arm, f_arm)}
        partial.update({"stage_reached": "HFW_COMPARISONS_COMPLETED",
                        "comparisons": comparisons,
                        "first_divergent_component": {
                            name: first_divergent_component(value)
                            for name, value in comparisons.items()},
                        "action_diagnostics": action_diagnostics(h_arm, f_arm, w_arm)})
        _write_partial_evidence(target, partial)
        _progress("H/F/W comparisons completed")
        partial["decision"] = classify_hfw_carrier_root_cause(
            comparisons["H_vs_F"], comparisons["H_vs_W"], comparisons["W_vs_F"]
        )
        partial["stage_reached"] = "DECISION_COMPLETED"
        _write_partial_evidence(target, partial)
        _progress("DECISION = " + partial["decision"])
    except Exception as exc:
        if partial.get("failure_class") is None:
            partial.update({"failure_class": str(exc) if str(exc) in (
                "R6_REFRESH_VALIDITY_EVIDENCE_NOT_EXACT",
                "HISTORICAL_H_ACTION_CARRIER_NOT_EXACT"
            ) else "R7_DIAGNOSTIC_EXECUTION_EXCEPTION",
                            "failure_exception_type": type(exc).__name__,
                            "failure_exception_message": str(exc)})
            _write_partial_evidence(target, partial)
        raise
    record = {
        "schema": SCHEMA,
        "full_gt_loaded": False,
        "training_seed": EXPECTED_SEED,
        "R6_evidence": {"decision": partial["R6_decision"],
                        "result_sha256": partial["R6_result_sha256"],
                        "h0_artifact_sha256": partial["R6_h0_artifact_sha256"]},
        "historical_H_state_sha256": partial["historical_H_state_sha256"],
        "historical_H_action_hashes": partial["historical_H_action_hashes"],
        "F_provenance_gate": partial["F_provenance_gate"],
        "F_replay_ones_parity": partial["F_replay_ones_parity"],
        "post_final_parity": partial["post_final_parity"],
        "carrier_hashes": partial["carrier_hashes"],
        "comparisons": partial["comparisons"],
        "first_divergent_component": partial["first_divergent_component"],
        "action_diagnostics": partial["action_diagnostics"],
        "decision": partial["decision"],
        "frozen_legacy_file_sha256": source_hashes,
    }
    target.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(target / "msrc_hfw_carrier_root_cause.json", record)
    _progress("DONE")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    run_audit(args.output_dir, args.device)


if __name__ == "__main__":
    main()
