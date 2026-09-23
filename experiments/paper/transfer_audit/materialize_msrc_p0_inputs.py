"""Materialize the frozen MSRC-v1 feature-only and sparse-split inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.generic_contract.sparse_label_contract import (
    materialize_hash_ranked_sparse_split,
)
from release_core.data import load_msrc_v1
from release_core.data.weak_quality import (
    apply_half_gaussian_corruption,
    generate_half_corruption_mask,
    ndarray_sha256,
)
from release_core.semantics import SparseLabelSplit, validate_sparse_label_split

from . import msrc_p0_protocol as protocol


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _array_record(value):
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "logical_sha256": ndarray_sha256(array),
        "finite": bool(np.isfinite(array).all()),
    }


def validate_dataset_path(dataset_path):
    """Require the one explicit authoritative dataset path and exact bytes."""
    path = Path(dataset_path)
    _require(path.is_file(), "explicit MSRC-v1 dataset path does not exist")
    _require(
        path.resolve() == protocol.DATASET_PATH.resolve(),
        "dataset path must be the frozen authoritative path; no fallback is allowed",
    )
    actual = file_sha256(path)
    _require(actual == protocol.DATASET_SHA256, "MSRC-v1 dataset SHA256 mismatch")
    return path, actual


def inspect_frozen_protocol(dataset_path):
    """Read-only verification used by materialization and protocol tests."""
    path, dataset_hash = validate_dataset_path(dataset_path)
    clean_views, label_container = load_msrc_v1(path)
    _require(len(label_container) == 1, "MSRC-v1 label-container mismatch")
    labels = np.ascontiguousarray(label_container[0], dtype=np.int64)
    _require(
        len(clean_views) == protocol.V
        and labels.shape == (protocol.N,)
        and tuple(view.shape for view in clean_views)
        == tuple((protocol.N, size) for size in protocol.VIEW_DIMS),
        "MSRC-v1 dataset contract mismatch",
    )
    split = materialize_hash_ranked_sparse_split(
        labels,
        dataset_name=protocol.DATASET,
        label_seed=protocol.LABEL_SEED,
        labels_per_class=protocol.LABELS_PER_CLASS,
    )
    sample_ids = np.arange(protocol.N, dtype=np.int64)
    validated = validate_sparse_label_split(SparseLabelSplit(
        sample_ids=sample_ids,
        labeled_ids=split.labeled_ids,
        labeled_targets=split.labeled_targets,
        unlabeled_ids=split.unlabeled_ids,
        class_count=protocol.K,
        labels_per_class=protocol.LABELS_PER_CLASS,
        label_seed=protocol.LABEL_SEED,
        dataset_name=protocol.DATASET,
    ))
    mask, mask_audit = generate_half_corruption_mask(
        protocol.N, protocol.V, protocol.WEAK_QUALITY_SEED
    )
    row_counts = mask.sum(axis=1, dtype=np.int64)
    view_counts = mask.sum(axis=0, dtype=np.int64)
    _require(
        int(mask.sum()) == protocol.CORRUPTED_PAIRS
        and int(np.count_nonzero(row_counts == 2)) == protocol.ROWS_WITH_TWO
        and int(np.count_nonzero(row_counts == 3)) == protocol.ROWS_WITH_THREE
        and tuple(int(value) for value in view_counts) == protocol.PER_VIEW_CORRUPTED
        and mask_audit["mask_sha256"] == protocol.CORRUPTION_MASK_SHA256,
        "MSRC-v1 frozen corruption-mask contract mismatch",
    )
    _require(
        tuple(int(value) for value in validated.labeled_ids) == protocol.LABELED_IDS
        and tuple(int(value) for value in validated.labeled_targets)
        == protocol.LABELED_TARGETS
        and validated.digest == protocol.SPLIT_SHA256
        and split.split_sha256 == protocol.SPLIT_SHA256,
        "MSRC-v1 frozen sparse-split contract mismatch",
    )
    return {
        "dataset_path": path,
        "dataset_sha256": dataset_hash,
        "clean_views": clean_views,
        "labels": labels,
        "sample_ids": sample_ids,
        "split": validated,
        "mask": mask,
    }


def materialize(*, dataset_path, output_dir):
    """Create one sealed input directory, refusing every overwrite."""
    target = Path(output_dir)
    _require(not target.exists(), "refusing to overwrite materialized inputs")
    inspected = inspect_frozen_protocol(dataset_path)
    corrupted_views, weak_audit = apply_half_gaussian_corruption(
        inspected["clean_views"], protocol.SNR_DB, protocol.WEAK_QUALITY_SEED
    )
    mask = np.ascontiguousarray(weak_audit.pop("mask"), dtype=np.bool_)
    _require(
        ndarray_sha256(mask) == protocol.CORRUPTION_MASK_SHA256
        and weak_audit["corrupted_pair_count"] == protocol.CORRUPTED_PAIRS
        and weak_audit["snr_target_pass"] is True
        and weak_audit["shape_preserved_pass"] is True
        and weak_audit["dtype_preserved_pass"] is True
        and weak_audit["all_finite_pass"] is True
        and weak_audit["unchanged_clean_pairs_max_abs_error"] == 0.0,
        "MSRC-v1 frozen weak-quality realization mismatch",
    )

    target.mkdir(parents=True, exist_ok=False)
    feature_path = target / "msrc_trainable_features.npz"
    split_path = target / "msrc_sparse_split.npz"
    feature_audit_path = target / "feature_audit.json"
    split_audit_path = target / "sparse_split_audit.json"
    weak_audit_path = target / "weak_quality_audit.json"
    seal_path = target / "materialization_seal.json"

    feature_payload = {
        "X" + str(index + 1): np.ascontiguousarray(view)
        for index, view in enumerate(corrupted_views)
    }
    feature_payload["sample_ids"] = inspected["sample_ids"]
    _require(tuple(feature_payload) == protocol.FEATURE_FIELDS, "feature whitelist mismatch")
    _require(
        {name.lower() for name in feature_payload}.isdisjoint(
            protocol.FORBIDDEN_FEATURE_FIELDS
        ),
        "feature-only artifact contains a GT field",
    )
    split = inspected["split"]
    split_payload = {
        "sample_ids": inspected["sample_ids"],
        "labeled_ids": np.ascontiguousarray(split.labeled_ids),
        "labeled_targets": np.ascontiguousarray(split.labeled_targets),
        "unlabeled_ids": np.ascontiguousarray(split.unlabeled_ids),
    }
    _require(tuple(split_payload) == protocol.SPLIT_FIELDS, "split whitelist mismatch")
    np.savez(feature_path, **feature_payload)
    np.savez(split_path, **split_payload)

    feature_hash = file_sha256(feature_path)
    split_hash = file_sha256(split_path)
    feature_audit = {
        "schema": "paper-msrc-p0-feature-audit-v1",
        "dataset": protocol.DATASET,
        "dataset_path": str(inspected["dataset_path"]),
        "dataset_sha256": inspected["dataset_sha256"],
        "artifact_path": str(feature_path),
        "artifact_sha256": feature_hash,
        "fields": list(protocol.FEATURE_FIELDS),
        "arrays": {name: _array_record(value) for name, value in feature_payload.items()},
        "mask_logical_sha256": protocol.CORRUPTION_MASK_SHA256,
        "per_sample_corrupted_count_unique": [2, 3],
        "per_view_corrupted_counts": list(protocol.PER_VIEW_CORRUPTED),
        "target_snr_db": protocol.SNR_DB,
        "trainable_artifact_forbidden_fields_absent": True,
        "full_gt_persisted": False,
        "full_gt_used_for_corruption": False,
    }
    split_audit = {
        "schema": "paper-msrc-p0-sparse-split-audit-v1",
        "dataset": protocol.DATASET,
        "artifact_path": str(split_path),
        "artifact_sha256": split_hash,
        "fields": list(protocol.SPLIT_FIELDS),
        "arrays": {name: _array_record(value) for name, value in split_payload.items()},
        "split_sha256": split.digest,
        "label_seed": protocol.LABEL_SEED,
        "labels_per_class": protocol.LABELS_PER_CLASS,
        "full_gt_loaded_only_during_dedicated_split_materialization": True,
        "full_gt_persisted": False,
        "unlabeled_gt_persisted": False,
    }
    _write_json(feature_audit_path, feature_audit)
    _write_json(split_audit_path, split_audit)
    _write_json(weak_audit_path, weak_audit)
    seal = {
        "schema": "paper-msrc-p0-materialization-seal-v1",
        "dataset": protocol.DATASET,
        "dataset_sha256": protocol.DATASET_SHA256,
        "feature_artifact_sha256": feature_hash,
        "feature_audit_sha256": file_sha256(feature_audit_path),
        "sparse_split_artifact_sha256": split_hash,
        "sparse_split_audit_sha256": file_sha256(split_audit_path),
        "weak_quality_audit_sha256": file_sha256(weak_audit_path),
        "split_sha256": protocol.SPLIT_SHA256,
        "full_gt_persisted": False,
        "full_gt_available_to_training_runner": False,
        "seal_valid": True,
    }
    _write_json(seal_path, seal)
    return seal


def load_materialized_inputs(input_dir):
    """Validate and load sealed paper inputs without any dataset/GT access."""
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
        views = tuple(np.ascontiguousarray(archive[name]) for name in protocol.FEATURE_FIELDS[:-1])
        sample_ids = np.ascontiguousarray(archive["sample_ids"])
    with np.load(paths["split"], allow_pickle=False) as archive:
        _require(tuple(archive.files) == protocol.SPLIT_FIELDS, "split whitelist mismatch")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in protocol.SPLIT_FIELDS}
    _require(np.array_equal(sample_ids, arrays["sample_ids"]), "sample-ID mismatch")
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
    _require(split.digest == protocol.SPLIT_SHA256, "sparse split logical hash mismatch")
    return paths, views, sample_ids, split


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = materialize(dataset_path=args.dataset_path, output_dir=args.output_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
