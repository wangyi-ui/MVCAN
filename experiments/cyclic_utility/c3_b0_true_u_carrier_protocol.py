"""Fail-closed primitives for C3-B0 TRUE_U carrier materialization.

The carrier is observed after epoch-20 Phase B and the final native target
refresh, but before the frozen final predictions are sealed.  The native
mapping has orientation M_v[global, local], hence local posteriors are aligned
with ``q_local @ M_v.T``.  No ground-truth array is an allowed carrier field.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch

from irv.b4_information_utility import tensor_sha256


STAGE = "C3-B0-CARRIER-A0"
PURPOSE = "READ_ONLY_PARENT_CARRIER_MATERIALIZATION"
SCIENTIFIC_PARENT = "C3-B0"
CARRIER_ARM = "TRUE_U"
SEEDS = (20, 30, 50)
N = 1400
V = 6
K = 7
FORMAL_EPOCHS = 20
SNAPSHOT_STAGE = "C3-B0 TRUE_U final native target refresh"
SNAPSHOT_EPOCH = 20
SNAPSHOT_POSITION = (
    "after_epoch20_Phase_B_after_final_native_refresh_"
    "before_final_prediction_seal"
)
ARTIFACT_KEYS = ("sample_ids", "q_local", "q_aligned", "M_v")


def _require(condition, message="C3_B0_CARRIER_REPLAY_PARITY_FAIL_CLOSED"):
    if not condition:
        raise RuntimeError(message)


def validate_seed(seed):
    value = int(seed)
    if value not in SEEDS:
        raise ValueError("carrier seed must be one of " + str(SEEDS))
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with open(Path(path), "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_sha256(value):
    """Reuse the frozen dtype/shape/contiguous-byte logical hash."""
    return tensor_sha256(value)


def validate_sample_ids(sample_ids):
    ids = np.asarray(sample_ids, dtype=np.int64)
    _require(
        ids.shape == (N,) and np.array_equal(ids, np.arange(N, dtype=np.int64)),
        "C3_B0_CARRIER_SAMPLE_IDS_FAIL_CLOSED",
    )
    return np.ascontiguousarray(ids)


def validate_snapshot_metadata(stage, epoch, position):
    _require(
        stage == SNAPSHOT_STAGE
        and int(epoch) == SNAPSHOT_EPOCH
        and position == SNAPSHOT_POSITION,
        "C3_B0_CARRIER_SNAPSHOT_MISMATCH_FAIL_CLOSED",
    )
    return {
        "snapshot_stage": stage,
        "snapshot_epoch": int(epoch),
        "snapshot_position": position,
        "q_and_M_v_same_refresh_lifetime": True,
    }


def validate_scientific_parent(parent_stage):
    _require(
        parent_stage == SCIENTIFIC_PARENT,
        "C3_B0_CARRIER_PARENT_FAIL_CLOSED",
    )
    return {"scientific_parent": SCIENTIFIC_PARENT, "C4_used": False}


def validate_parent_hash_audits(parent_hashes, c3_source, c3_formal):
    _require(
        isinstance(parent_hashes, dict)
        and parent_hashes.get("all_parent_hashes_pass") is True
        and isinstance(c3_source, dict)
        and c3_source.get("all_entries_exact_match") is True
        and isinstance(c3_formal, dict)
        and c3_formal.get("all_entries_exact_match") is True,
        "C3_B0_CARRIER_PARENT_HASH_FAIL_CLOSED",
    )
    return {
        "all_parent_hashes_pass": True,
        "frozen_C3_B0_source_hashes_pass": True,
        "frozen_C3_B0_formal_output_hashes_pass": True,
    }


def validate_permutation_matrices(matches):
    """Validate M_v[global,local] as six exact 7x7 permutations."""
    values = (
        matches.detach().cpu().numpy() if torch.is_tensor(matches)
        else np.asarray(matches)
    )
    _require(
        values.shape == (V, K, K) and np.isfinite(values).all(),
        "C3_B0_CARRIER_COORDINATE_MAPPING_FAIL_CLOSED",
    )
    binary = np.logical_or(values == 0.0, values == 1.0)
    rows = values.sum(axis=2)
    columns = values.sum(axis=1)
    checks = {
        "shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
        "binary_entries": bool(binary.all()),
        "row_sums_one": bool(np.array_equal(rows, np.ones((V, K)))),
        "column_sums_one": bool(np.array_equal(columns, np.ones((V, K)))),
        "one_entry_per_row": bool(np.array_equal((values == 1.0).sum(2), np.ones((V, K)))),
        "one_entry_per_column": bool(np.array_equal((values == 1.0).sum(1), np.ones((V, K)))),
    }
    _require(
        all(value is True for key, value in checks.items() if key != "shape"),
        "C3_B0_CARRIER_COORDINATE_MAPPING_FAIL_CLOSED",
    )
    checks["all_permutation_checks_pass"] = True
    return checks


def align_q_readonly(q_local, matches):
    """Map q_local[N,V,K] to q_aligned[N,V,K] using M_v[global,local]."""
    _require(
        torch.is_tensor(q_local) and torch.is_tensor(matches),
        "C3_B0_CARRIER_COORDINATE_MAPPING_FAIL_CLOSED",
    )
    local = q_local.detach()
    mapping = matches.detach().to(device=local.device, dtype=local.dtype)
    validate_permutation_matrices(mapping)
    _require(
        tuple(local.shape) == (N, V, K)
        and bool(torch.isfinite(local).all().item())
        and bool((local >= 0).all().item()),
        "C3_B0_CARRIER_Q_SHAPE_OR_FINITE_FAIL_CLOSED",
    )
    with torch.no_grad():
        aligned = torch.stack(
            [local[:, view_id, :] @ mapping[view_id].T for view_id in range(V)],
            dim=1,
        ).detach()
    _require(
        tuple(aligned.shape) == (N, V, K)
        and bool(torch.isfinite(aligned).all().item())
        and bool((aligned >= 0).all().item())
        and torch.allclose(
            aligned.sum(dim=2), local.sum(dim=2), rtol=0.0, atol=1e-6
        ),
        "C3_B0_CARRIER_Q_ALIGNMENT_FAIL_CLOSED",
    )
    return aligned


def validate_coordinate_snapshot(q_local, q_aligned, matches):
    expected = align_q_readonly(q_local, matches)
    _require(
        torch.is_tensor(q_aligned)
        and tuple(q_aligned.shape) == (N, V, K)
        and not q_aligned.requires_grad
        and bool(torch.isfinite(q_aligned).all().item())
        and bool((q_aligned >= 0).all().item())
        and torch.equal(q_aligned.detach(), expected),
        "C3_B0_CARRIER_COORDINATE_MAPPING_FAIL_CLOSED",
    )
    return {
        "coordinate_mapping_pass": True,
        "mapping_formula": "q_aligned[:,v,:] = q_local[:,v,:] @ M_v[v].T",
        "M_v_orientation": "M_v[global,local]",
        "M_v": validate_permutation_matrices(matches),
        "q_local_shape": list(q_local.shape),
        "q_aligned_shape": list(q_aligned.shape),
        "q_aligned_finite": True,
        "q_aligned_non_negative": True,
        "row_mass_preserved": True,
    }


def validate_replay_parity(
    expected_model_hash,
    replayed_model_hash,
    expected_predictions,
    replayed_predictions,
    expected_sample_ids,
    replayed_sample_ids,
):
    expected_ids = np.asarray(expected_sample_ids, dtype=np.int64)
    replayed_ids = np.asarray(replayed_sample_ids, dtype=np.int64)
    expected_pred = np.asarray(expected_predictions, dtype=np.int64)
    replayed_pred = np.asarray(replayed_predictions, dtype=np.int64)
    model_equal = expected_model_hash == replayed_model_hash
    canonical = np.arange(N, dtype=np.int64)
    ids_equal = (
        expected_ids.shape == replayed_ids.shape == (N,)
        and np.array_equal(expected_ids, replayed_ids)
        and np.array_equal(expected_ids, canonical)
    )
    predictions_equal = (
        expected_pred.shape == replayed_pred.shape == (N,)
        and np.array_equal(expected_pred, replayed_pred)
    )
    _require(model_equal and ids_equal and predictions_equal)
    return {
        "final_model_hash_expected": expected_model_hash,
        "final_model_hash_replayed": replayed_model_hash,
        "final_model_hash_equal": True,
        "final_predictions_hash_expected": logical_sha256(expected_pred),
        "final_predictions_hash_replayed": logical_sha256(replayed_pred),
        "final_predictions_equal": True,
        "final_prediction_sample_ids_equal": True,
        "sample_ids_exact_arange_pass": True,
    }


def validate_artifact_keys(arrays):
    keys = tuple(arrays.keys())
    _require(
        keys == ARTIFACT_KEYS
        and not any(
            token in key.lower()
            for key in keys
            for token in ("gt", "label", "truth", "metric", "acc", "nmi", "ari")
        ),
        "C3_B0_CARRIER_ARTIFACT_WHITELIST_FAIL_CLOSED",
    )
    return True


def build_valid_seal(audit, artifact_path, audit_path):
    required = (
        audit.get("all_parent_hashes_pass") is True,
        audit.get("final_model_hash_equal") is True,
        audit.get("final_predictions_equal") is True,
        audit.get("final_prediction_sample_ids_equal") is True,
        audit.get("coordinate_mapping_pass") is True,
        audit.get("GT_loaded_for_carrier_materialization") is False,
        audit.get("C4_used") is False,
    )
    _require(all(required), "C3_B0_CARRIER_SEAL_FAIL_CLOSED")
    return {
        "stage": STAGE,
        "seed": validate_seed(audit["seed"]),
        "artifact_purpose": PURPOSE,
        "all_parent_hashes_pass": True,
        "final_model_hash_equal": True,
        "final_predictions_equal": True,
        "sample_ids_equal": True,
        "coordinate_mapping_pass": True,
        "GT_loaded_for_carrier_materialization": False,
        "C4_used": False,
        "carrier_valid_for_downstream_readonly_use": True,
        "artifact_path": str(artifact_path),
        "artifact_file_sha256": file_sha256(artifact_path),
        "audit_path": str(audit_path),
        "audit_file_sha256": file_sha256(audit_path),
        "artifact_keys": list(ARTIFACT_KEYS),
    }
