"""P1-A1 canonical true-action namespace; never recalculates an arm artifact."""

import hashlib
import json
from . import p1_a2_execution_contract as execution
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_paths(root):
    root = Path(root)
    return {"root": root, "artifact": root / "true_action_state.npz",
            "audit": root / "action_audit.json", "seal": root / "action_seal.json"}


def verify_true_action(root, *, dataset, training_seed, initial_model_sha256):
    """Return only an exact, detached canonical action state for this seed."""
    paths = action_paths(root)
    if not paths["root"].is_dir() or not all(path.is_file() for path in paths.values() if path != paths["root"]):
        raise RuntimeError("FORMAL_ACTION_STATE_INCOMPLETE")
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    seal = json.loads(paths["seal"].read_text(encoding="utf-8"))
    if not (seal.get("seal_valid") is True and audit.get("dataset") == dataset
            and audit.get("training_seed") == training_seed
            and audit.get("initial_model_sha256") == initial_model_sha256
            and audit.get("detached_frozen") is True
            and _sha256(paths["artifact"]) == seal.get("artifact_sha256")
            and _sha256(paths["audit"]) == seal.get("action_audit_sha256")):
        raise RuntimeError("FORMAL_ACTION_STATE_PROVENANCE_MISMATCH")
    return {**paths, "artifact_sha256": _sha256(paths["artifact"]), "audit": audit}


def build_true_action(*_args, **_kwargs):
    """Reject a fresh build until P1-A0 exposes refresh-carrier semantics."""
    raise RuntimeError("FORMAL_ACTION_MATERIALIZATION_BUILDER_NOT_IMPLEMENTED")


def select_arm_utility(true_action, arm, arm_protocol, *, dataset=None):
    """Permit no undeclared arm transform; SHUFFLE_U needs frozen seed/axis."""
    if dataset is not None:
        capability, reason = execution.arm_capability(dataset, arm)
        if capability != "AUTHORIZED":
            raise RuntimeError(reason)
    if arm == "OURS_TRUE_U":
        return {"kind": "true", "source": true_action["artifact"]}
    if arm == "TRUE_UNIFORM":
        return {"kind": "uniform", "source": true_action["artifact"]}
    if arm == "SHUFFLE_U" and dataset == "Caltech-6V":
        return {"kind": "shuffle_relation", "source": true_action["artifact"],
                "utility": "true U_cycle", "relation": "historical fixed shuffled sparse targets"}
    raise RuntimeError("FORMAL_ARM_SEMANTICS_UNRESOLVED")
    raise RuntimeError("FORMAL_ARM_SEMANTICS_UNRESOLVED")
