"""P1-A3 runtime adapters over frozen release primitives; no alternate science."""

import hashlib
import json
from pathlib import Path

import numpy as np

from release_core.runtime import ProvenanceConfig
from .p1_a1_base_runtime import BaseProvenanceConfig


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, record):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")


def build_initialization(**kwargs):
    from .p1_a3_real_execution import build_initialization as implementation
    return implementation(**kwargs)


def build_true_action(**kwargs):
    from .p1_a3_real_execution import build_action as implementation
    return implementation(**kwargs)


def materialize_ours_true_u_adapter(*, true_action, output_dir):
    """Derive release input files from immutable canonical true action."""
    target = Path(output_dir)
    if target.exists():
        raise RuntimeError("FORMAL_ARM_ADAPTER_OUTPUT_ALREADY_EXISTS")
    with np.load(true_action["artifact"], allow_pickle=False) as archive:
        utility = np.ascontiguousarray(archive["U_cycle"], dtype=np.float64)
        unlabeled_ids = np.ascontiguousarray(archive["unlabeled_ids"], dtype=np.int64)
        labeled_ids = np.ascontiguousarray(archive["labeled_ids"], dtype=np.int64)
        relation = np.ascontiguousarray(archive["PredRelation_true"], dtype=np.bool_)
        balance = np.ascontiguousarray(archive["relation_balance_weights_true"], dtype=np.float64)
    target.mkdir(parents=True, exist_ok=False)
    utility_path, semantic_path = target / "utility.npz", target / "semantics.npz"
    np.savez(utility_path, U_cycle=utility, unlabeled_ids=unlabeled_ids)
    np.savez(semantic_path, PredRelation_true=relation, relation_balance_weights_true=balance, labeled_ids=labeled_ids, unlabeled_ids=unlabeled_ids)
    utility_audit, semantic_audit = target / "utility_audit.json", target / "semantic_audit.json"
    _write(utility_audit, {"schema": "p1-a3-ours-true-u-utility-v1", "source_action_sha256": true_action["artifact_sha256"], "artifact_sha256": _sha(utility_path), "canonical_transform": "U = canonical true U_cycle", "full_gt_loaded": False})
    _write(semantic_audit, {"schema": "p1-a3-ours-true-u-semantics-v1", "source_action_sha256": true_action["artifact_sha256"], "artifact_sha256": _sha(semantic_path), "canonical_transform": "PredRelation/balance = canonical true relation/balance", "full_gt_loaded": False})
    return {"root": target, "utility": utility_path, "utility_audit": utility_audit, "semantic": semantic_path, "semantic_audit": semantic_audit}


def arm_provenance(*, feature, feature_audit, split, split_audit, utility, utility_audit, semantic, semantic_audit, initialization, output):
    required = (feature, feature_audit, split, split_audit, utility, utility_audit, semantic, semantic_audit, initialization["audit"], *initialization["checkpoint_paths"])
    expected = tuple((Path(path), _sha(path)) for path in required)
    return ProvenanceConfig(feature_artifact=feature, feature_audit=feature_audit, sparse_split_artifact=split, sparse_split_audit=split_audit, utility_artifact=utility, utility_audit=utility_audit, semantic_artifact=semantic, semantic_audit=semantic_audit, checkpoint_paths=tuple(initialization["checkpoint_paths"]), checkpoint_audit=initialization["audit"], output_root=output, strict_replay=True, expected_file_sha256=expected, expected_initial_model_sha256=initialization["initial_model_sha256"])


def base_provenance(*, feature, feature_audit, split, split_audit, initialization, output):
    """Construct strict native-only provenance for the BASE wrapper."""
    required = (feature, feature_audit, split, split_audit, initialization["audit"], *initialization["checkpoint_paths"])
    expected = tuple((Path(path), _sha(path)) for path in required)
    return BaseProvenanceConfig(feature_artifact=feature, feature_audit=feature_audit, sparse_split_artifact=split, sparse_split_audit=split_audit, checkpoint_paths=tuple(initialization["checkpoint_paths"]), checkpoint_audit=initialization["audit"], output_root=output, strict_replay=True, expected_file_sha256=expected, expected_initial_model_sha256=initialization["initial_model_sha256"])
