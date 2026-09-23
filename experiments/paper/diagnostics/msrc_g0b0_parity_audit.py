"""GT-free Gate A: localize MSRC P0-A2 versus historical G0-B0 parity."""

import argparse
import json
import os
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
from release_core.backbone.clustering import native_refresh_from_latents
from release_core.data.weak_quality import ndarray_sha256
from release_core.runtime import SealedPredictionPaths
from release_core.runtime.evaluation import _validate_sealed_prediction
from release_core.semantics import build_relation_semantics
from release_core.utility import build_directional_actions, compute_directional_cycle_utility

from . import p0_a3_protocol as protocol


CURRENT_TRUE_U_FILES = {
    "run_manifest.json": "2d407bbf4dc4c62b1dc0600b2f78fbc61a672eb5cbd459c19955961edb07bb46",
    "pre_gt_bundle.npz": "69887eb388318c72706769fc3676a97b05f6429e59844ca25119fba48019ab90",
    "pre_gt_audit.json": "7c73039fd0d813d2f130ee50232be747bbfda4e22be184bddeb86a472b96c299",
    "pre_gt_seal.json": "5d773ce39bb0cea8d8fc3d1639817137d933a37a666e6933461744bb672db9bd",
}
PROVENANCE_KEYS = {
    "q_local": "q_local_logical_sha256",
    "q_aligned": "q_aligned_logical_sha256",
    "M_v": "M_v_logical_sha256",
    "U_cycle": "true_u_cycle_logical_sha256",
    "y_gen": "y_gen_logical_sha256",
    "PredRelation_true": "pred_relation_logical_sha256",
    "relation_balance_weights_true": "balance_logical_sha256",
}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    _require(isinstance(value, dict), "JSON root must be an object")
    return value


def _write_json_exclusive(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_historical_msrc_files():
    """Verify all frozen whole-file identities before reading arrays."""
    result = {}
    for path, expected in protocol.HISTORICAL_MSRC_FILES.items():
        _require(path.is_file(), "historical MSRC file is missing: " + str(path))
        actual = file_sha256(path)
        _require(actual == expected, "historical MSRC SHA256 mismatch: " + str(path))
        result[str(path)] = actual
    return result


def verify_current_true_u():
    """Verify the preserved P0-A2 TRUE_U run without opening full GT."""
    for name, expected in CURRENT_TRUE_U_FILES.items():
        path = protocol.MSRC_TRUE_U_DIR / name
        _require(path.is_file(), "current TRUE_U file is missing: " + str(path))
        _require(file_sha256(path) == expected, "current TRUE_U SHA256 mismatch: " + name)
    sealed = SealedPredictionPaths(
        protocol.MSRC_TRUE_U_DIR / "pre_gt_bundle.npz",
        protocol.MSRC_TRUE_U_DIR / "pre_gt_audit.json",
        protocol.MSRC_TRUE_U_DIR / "pre_gt_seal.json",
    )
    _validate_sealed_prediction(sealed)
    return _read_json(protocol.MSRC_TRUE_U_DIR / "run_manifest.json")


def reconstruct_current_msrc(device="cpu"):
    """Reconstruct R2/R3 tensors through release_core; never load full GT."""
    _, views, _, split = load_materialized_inputs(protocol.MSRC_INPUT_DIR)
    initialization = verify_initialization(protocol.MSRC_INIT_DIR)
    model = initialization["model"]
    model.to_device(device)
    for autoencoder in model.autoencoders:
        autoencoder.train()
    latent_views, q_views = [], []
    with torch.no_grad():
        for autoencoder, view in zip(model.autoencoders, views):
            latent = autoencoder.encoder(torch.from_numpy(view).to(device))
            posterior = autoencoder.clustering(latent)
            latent_views.append(np.ascontiguousarray(latent.cpu().numpy()))
            q_views.append(np.ascontiguousarray(posterior.cpu().numpy()))
    _, matrices, _, _, _ = native_refresh_from_latents(
        latent_views, q_views, (1.0,) * 5, n_clusters=7, random_state=20
    )
    q_local = np.ascontiguousarray(np.stack(q_views, axis=1), dtype=np.float32)
    matrix = np.ascontiguousarray(matrices, dtype=np.float32)
    q_aligned = np.ascontiguousarray(np.stack([
        q_local[:, view_id, :] @ matrix[view_id].T for view_id in range(5)
    ], axis=1), dtype=np.float32)
    actions = build_directional_actions(5)
    cycle = compute_directional_cycle_utility(torch.from_numpy(q_aligned), actions)
    utility = np.ascontiguousarray(cycle["U_cycle"].numpy(), dtype=np.float64)
    y_gen = np.ascontiguousarray(cycle["y_gen"].numpy(), dtype=np.int64)
    semantics = build_relation_semantics(y_gen, split, actions)
    arrays = {
        "q_local": q_local,
        "q_aligned": q_aligned,
        "M_v": matrix,
        "U_cycle": utility,
        "y_gen": y_gen,
        "PredRelation_true": np.ascontiguousarray(semantics.pred_relation),
        "relation_balance_weights_true": np.ascontiguousarray(
            semantics.balance_weights
        ),
        "generator_membership": np.ascontiguousarray(
            cycle["generator_membership"], dtype=np.int64
        ),
        "verifier_membership": np.ascontiguousarray(
            cycle["verifier_membership"], dtype=np.int64
        ),
        "closure": np.ascontiguousarray(cycle["closure"].numpy(), dtype=np.bool_),
        "class_pred": np.ascontiguousarray(semantics.class_pred, dtype=np.int64),
    }
    return arrays, split


def validate_reconstruction_against_manifest(arrays, manifest):
    provenance = manifest.get("action_provenance", {})
    checks = {}
    for name, key in PROVENANCE_KEYS.items():
        actual = ndarray_sha256(arrays[name])
        expected = provenance.get(key)
        _require(actual == expected, "P0-A2 action provenance mismatch: " + name)
        checks[name] = {"expected": expected, "actual": actual, "exact": True}
    return checks


def load_historical_msrc():
    with np.load(protocol.HISTORICAL_MSRC_ARTIFACT, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    for name, expected in protocol.HISTORICAL_MSRC_LOGICAL.items():
        if name in arrays:
            _require(ndarray_sha256(arrays[name]) == expected,
                     "historical logical SHA256 mismatch: " + name)
    return arrays


def compare_arrays(current, historical):
    left, right = np.asarray(current), np.asarray(historical)
    same_shape = left.shape == right.shape
    exact_kind = left.dtype == np.bool_ or np.issubdtype(left.dtype, np.integer)
    equal = bool(same_shape and np.array_equal(left, right))
    result = {
        "shape": {"current": list(left.shape), "historical": list(right.shape)},
        "dtype": {"current": str(left.dtype), "historical": str(right.dtype)},
        "logical_sha256": {
            "current": ndarray_sha256(left), "historical": ndarray_sha256(right)
        },
        "array_equal": equal,
        "comparison_policy": "exact" if exact_kind else "floating",
        "allclose_1e-12": None,
        "allclose_1e-8": None,
        "max_abs_diff": None,
    }
    if same_shape:
        if exact_kind:
            result["max_abs_diff"] = float(np.max(np.abs(
                left.astype(np.int64) - right.astype(np.int64)
            ))) if left.size else 0.0
        else:
            a, b = left.astype(np.float64), right.astype(np.float64)
            result["allclose_1e-12"] = bool(np.allclose(a, b, rtol=0.0, atol=1e-12))
            result["allclose_1e-8"] = bool(np.allclose(a, b, rtol=0.0, atol=1e-8))
            result["max_abs_diff"] = float(np.max(np.abs(a - b))) if a.size else 0.0
    return result


def unavailable_historical_comparison(current, reason):
    """Represent a missing frozen carrier without fabricating a comparison."""
    value = np.asarray(current)
    return {
        "shape": {"current": list(value.shape), "historical": None},
        "dtype": {"current": str(value.dtype), "historical": None},
        "logical_sha256": {
            "current": ndarray_sha256(value), "historical": None,
        },
        "array_equal": None,
        "comparison_policy": "unavailable",
        "allclose_1e-12": None,
        "allclose_1e-8": None,
        "max_abs_diff": None,
        "reason": reason,
    }


def classify(comparisons):
    action_equal = all(comparisons[name]["array_equal"] for name in (
        "generator_membership", "verifier_membership"
    ))
    if not action_equal:
        return "FAIL_CLOSED_ACTION_MEMBERSHIP_OR_ORDER_MISMATCH"
    if comparisons["final_predictions"]["array_equal"]:
        return "PASS_HISTORICAL_G0_B0_BEHAVIOR_PARITY"
    r2_r3_exact = all(comparisons[name]["array_equal"] for name in (
        "U_cycle", "PredRelation_true", "relation_balance_weights_true"
    ))
    final_state_equal = all(comparisons[name]["array_equal"] for name in (
        "final_predictions", "final_q_local", "final_q_aligned", "final_M_v"
    ))
    if r2_r3_exact and not final_state_equal:
        return "R4_R5_OR_PAPER_TRAINING_ORCHESTRATION"
    if not comparisons["U_cycle"]["array_equal"]:
        if not comparisons["q_aligned"]["array_equal"]:
            return "CURRENT_CONDITION_NATIVE_INITIALIZATION_OR_STATE_TRAJECTORY"
        numerical_only = (
            comparisons["y_gen"]["array_equal"]
            and comparisons["PredRelation_true"]["array_equal"]
            and comparisons["relation_balance_weights_true"]["array_equal"]
            and comparisons["U_cycle"]["allclose_1e-8"]
        )
        if numerical_only:
            return "NUMERICAL_UTILITY_ONLY"
    return "FAIL_CLOSED_SCIENTIFIC_IMPLEMENTATION_MISMATCH"


def run_audit(output_path, device="cpu"):
    historical_files = verify_historical_msrc_files()
    manifest = verify_current_true_u()
    current, split = reconstruct_current_msrc(device)
    provenance = validate_reconstruction_against_manifest(current, manifest)
    historical = load_historical_msrc()
    with np.load(protocol.MSRC_TRUE_U_DIR / "pre_gt_bundle.npz",
                 allow_pickle=False) as archive:
        current_final = {name: np.ascontiguousarray(archive[name]) for name in (
            "final_predictions", "q_local", "q_aligned", "M_v"
        )}
    pairs = {
        "generator_membership": (current["generator_membership"], historical["generator_membership"]),
        "verifier_membership": (current["verifier_membership"], historical["verifier_membership"]),
        "U_cycle": (current["U_cycle"], historical["U_cycle"]),
        "PredRelation_true": (current["PredRelation_true"], historical["PredRelation_true"]),
        "relation_balance_weights_true": (
            current["relation_balance_weights_true"],
            historical["relation_balance_weights_true"],
        ),
        "final_predictions": (current_final["final_predictions"], historical["final_predictions"]),
        "final_q_local": (current_final["q_local"], historical["q_local"]),
        "final_q_aligned": (current_final["q_aligned"], historical["q_aligned"]),
        "final_M_v": (current_final["M_v"], historical["M_v"]),
    }
    comparisons = {name: compare_arrays(*values) for name, values in pairs.items()}
    missing_reason = (
        "historical artifact carries final post-R5 q/M only; it does not carry "
        "the pre-R2 state or y_gen, so no value is reconstructed from final q"
    )
    for name in ("q_local", "q_aligned", "M_v", "y_gen"):
        comparisons[name] = unavailable_historical_comparison(
            current[name], missing_reason
        )
    record = {
        "schema": "paper-p0-a3-msrc-g0b0-parity-v1",
        "gate": "A",
        "full_gt_loaded": False,
        "training_run": False,
        "historical_file_sha256": historical_files,
        "current_action_provenance_validation": provenance,
        "comparisons": comparisons,
        "decision": classify(comparisons),
        "decision_rules": ["A1", "A2", "A3", "A4", "A5"],
    }
    _write_json_exclusive(output_path, record)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_audit(args.output, args.device), sort_keys=True))


if __name__ == "__main__":
    main()
