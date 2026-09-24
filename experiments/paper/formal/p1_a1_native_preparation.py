"""P1-A1 sealed native-initialization namespace; no scientific constants live here."""

import hashlib
import json
from pathlib import Path

from . import p1_a0_formal_protocol as protocol


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialization_paths(root):
    root = Path(root)
    return {
        "root": root,
        "audit": root / "initialization_audit.json",
        "manifest": root / "initialization_manifest.json",
        "seal": root / "initialization_seal.json",
    }


def verify_initialization(root, *, dataset, training_seed):
    """Verify/reuse only a complete sealed dataset x training-seed state."""
    paths = initialization_paths(root)
    if not paths["root"].is_dir() or not all(path.is_file() for path in paths.values() if path != paths["root"]):
        raise RuntimeError("FORMAL_INITIALIZATION_INCOMPLETE")
    records = {name: json.loads(path.read_text(encoding="utf-8"))
               for name, path in paths.items() if name != "root"}
    manifest, seal = records["manifest"], records["seal"]
    if not (seal.get("seal_valid") is True and manifest.get("dataset") == dataset
            and manifest.get("training_seed") == training_seed
            and manifest.get("native_config_seed") == next(item.native_config_seed for item in protocol.FORMAL_DATASETS if item.name == dataset)
            and manifest.get("full_gt_loaded") is False and manifest.get("sparse_labels_used") is False
            and _sha256(paths["audit"]) == seal.get("initialization_audit_sha256")
            and _sha256(paths["manifest"]) == seal.get("initialization_manifest_sha256")):
        raise RuntimeError("FORMAL_INITIALIZATION_PROVENANCE_MISMATCH")
    checkpoints = tuple(Path(path) for path in manifest.get("checkpoint_paths", ()))
    if not checkpoints or not all(path.is_file() for path in checkpoints):
        raise RuntimeError("FORMAL_INITIALIZATION_INCOMPLETE")
    hashes = tuple(_sha256(path) for path in checkpoints)
    if list(hashes) != manifest.get("checkpoint_sha256") or list(hashes) != seal.get("checkpoint_sha256"):
        raise RuntimeError("FORMAL_INITIALIZATION_CHECKPOINT_MISMATCH")
    return {**paths, "checkpoint_paths": checkpoints, "checkpoint_sha256": hashes,
            "initial_model_sha256": manifest.get("combined_model_sha256"), "manifest": manifest}


def build_initialization(*_args, **_kwargs):
    """Reserved real-build entrypoint: rejects missing P1-A0 generator semantics."""
    # P1-A0 deliberately owns science.  It does not yet expose the historical
    # DataLoader-generator continuity required by P1-A1 §8, so guessing would
    # silently create a new native-preparation method.
    raise RuntimeError("FORMAL_NATIVE_PREPARATION_BUILDER_NOT_IMPLEMENTED")
