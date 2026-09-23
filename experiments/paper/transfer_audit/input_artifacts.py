"""GT-free validation for sealed paper materialization artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np

from release_core.semantics import SparseLabelSplit, validate_sparse_label_split

from . import msrc_p0_protocol as protocol


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_materialized_inputs(input_dir):
    """Validate and load sealed inputs without importing a dataset/GT loader."""
    root = Path(input_dir)
    paths = {
        "feature": root / "msrc_trainable_features.npz",
        "split": root / "msrc_sparse_split.npz",
        "feature_audit": root / "feature_audit.json",
        "split_audit": root / "sparse_split_audit.json",
        "weak_audit": root / "weak_quality_audit.json",
        "seal": root / "materialization_seal.json",
    }
    _require(all(path.is_file() for path in paths.values()), "materialized input is missing")
    with paths["seal"].open("r", encoding="utf-8") as stream:
        seal = json.load(stream)
    _require(
        seal.get("seal_valid") is True
        and seal.get("dataset") == protocol.DATASET
        and seal.get("dataset_sha256") == protocol.DATASET_SHA256
        and seal.get("full_gt_persisted") is False
        and seal.get("full_gt_available_to_training_runner") is False
        and file_sha256(paths["feature"]) == seal["feature_artifact_sha256"]
        and file_sha256(paths["feature_audit"]) == seal["feature_audit_sha256"]
        and file_sha256(paths["split"]) == seal["sparse_split_artifact_sha256"]
        and file_sha256(paths["split_audit"]) == seal["sparse_split_audit_sha256"]
        and file_sha256(paths["weak_audit"]) == seal["weak_quality_audit_sha256"]
        and seal.get("split_sha256") == protocol.SPLIT_SHA256,
        "materialization seal mismatch",
    )
    with np.load(paths["feature"], allow_pickle=False) as archive:
        _require(tuple(archive.files) == protocol.FEATURE_FIELDS, "feature whitelist mismatch")
        _require(
            {name.lower() for name in archive.files}.isdisjoint(
                protocol.FORBIDDEN_FEATURE_FIELDS
            ),
            "feature artifact contains a GT field",
        )
        views = tuple(np.ascontiguousarray(archive[name]) for name in protocol.FEATURE_FIELDS[:-1])
        sample_ids = np.ascontiguousarray(archive["sample_ids"])
    with np.load(paths["split"], allow_pickle=False) as archive:
        _require(tuple(archive.files) == protocol.SPLIT_FIELDS, "split whitelist mismatch")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in protocol.SPLIT_FIELDS}
    _require(
        tuple(view.shape for view in views)
        == tuple((protocol.N, size) for size in protocol.VIEW_DIMS)
        and all(view.dtype == np.float32 for view in views),
        "feature tensor contract mismatch",
    )
    _require(
        sample_ids.dtype == np.int64
        and np.array_equal(sample_ids, np.arange(protocol.N, dtype=np.int64))
        and np.array_equal(sample_ids, arrays["sample_ids"]),
        "sample-ID mismatch",
    )
    split = validate_sparse_label_split(SparseLabelSplit(
        sample_ids=arrays["sample_ids"],
        labeled_ids=arrays["labeled_ids"],
        labeled_targets=arrays["labeled_targets"],
        unlabeled_ids=arrays["unlabeled_ids"],
        class_count=protocol.K,
        labels_per_class=protocol.LABELS_PER_CLASS,
        label_seed=protocol.LABEL_SEED,
        dataset_name=protocol.DATASET,
    ))
    _require(
        split.digest == protocol.SPLIT_SHA256
        and tuple(int(value) for value in split.labeled_ids) == protocol.LABELED_IDS
        and tuple(int(value) for value in split.labeled_targets) == protocol.LABELED_TARGETS,
        "sparse split frozen identity mismatch",
    )
    return paths, views, sample_ids, split
