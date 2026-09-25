"""Formal input sealing boundary.  Raw full GT is confined to split creation."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from release_core.data.weak_quality import ndarray_sha256
from release_core.semantics import SparseLabelSplit, validate_sparse_label_split

from . import p1_a0_formal_protocol as protocol


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



def _runtime_split_from_generic(raw_split, *, sample_ids, class_count):
    """Cross the generic-contract to release-runtime split type boundary."""
    runtime_split = SparseLabelSplit(sample_ids=np.ascontiguousarray(sample_ids, dtype=np.int64), labeled_ids=raw_split.labeled_ids, labeled_targets=raw_split.labeled_targets, unlabeled_ids=raw_split.unlabeled_ids, class_count=class_count, labels_per_class=raw_split.labels_per_class, label_seed=raw_split.label_seed, dataset_name=raw_split.dataset_name)
    runtime_split = validate_sparse_label_split(runtime_split)
    if runtime_split.digest != raw_split.split_sha256:
        raise RuntimeError("FORMAL_SPARSE_SPLIT_DIGEST_CONVERSION_MISMATCH")
    if not (np.array_equal(runtime_split.labeled_ids, raw_split.labeled_ids) and np.array_equal(runtime_split.labeled_targets, raw_split.labeled_targets) and np.array_equal(runtime_split.unlabeled_ids, raw_split.unlabeled_ids)):
        raise RuntimeError("FORMAL_SPARSE_SPLIT_CONTENT_CONVERSION_MISMATCH")
    return runtime_split

def materialize_inputs(*, dataset, views, corruption_mask, sparse_split, output_dir, sparse_split_source="hash-ranked generic materialization", full_gt_loaded_at_input_boundary=False, full_gt_used_to_select_sparse_ids=True):
    """Seal feature-only views plus an explicit sparse split; never persist GT."""
    item = next(value for value in protocol.FORMAL_DATASETS if value.name == dataset)
    target = Path(output_dir)
    if target.exists():
        raise RuntimeError("FORMAL_INPUT_OUTPUT_ALREADY_EXISTS")
    arrays = tuple(np.ascontiguousarray(view, dtype=np.float32) for view in views)
    ids = np.arange(item.n_samples, dtype=np.int64)
    mask = np.ascontiguousarray(corruption_mask, dtype=np.bool_)
    split = validate_sparse_label_split(sparse_split)
    if not (len(arrays) == item.n_views and tuple(view.shape for view in arrays) == tuple((item.n_samples, dim) for dim in item.view_dims)
            and mask.shape == (item.n_samples, item.n_views) and ndarray_sha256(mask) == item.weak_quality_mask_sha256
            and split.dataset_name == dataset and split.label_seed == protocol.SPARSE_LABEL_PROTOCOL["label_seed"]
            and split.labels_per_class == protocol.SPARSE_LABEL_PROTOCOL["labels_per_class"]):
        raise RuntimeError("FORMAL_INPUT_CONTRACT_MISMATCH")
    target.mkdir(parents=True, exist_ok=False)
    features = target / "features.npz"
    split_path = target / "sparse_split.npz"
    np.savez(features, **{"view_" + str(index + 1): view for index, view in enumerate(arrays)}, sample_ids=ids)
    np.savez(split_path, sample_ids=ids, labeled_ids=split.labeled_ids, labeled_targets=split.labeled_targets, unlabeled_ids=split.unlabeled_ids)
    row_counts = mask.sum(axis=1, dtype=np.int64)
    view_counts = mask.sum(axis=0, dtype=np.int64)
    feature_audit = {"dataset": dataset, "feature_sha256": _sha(features), "sample_ids_exact": True,
                     "trainable_artifact_forbidden_fields_absent": True, "full_gt_persisted": False,
                     "weak_quality_seed": protocol.WEAK_QUALITY_PROTOCOL["realization_seed"],
                     "snr_db": protocol.WEAK_QUALITY_PROTOCOL["snr_db"], "per_sample_corrupted_count_unique": [int(value) for value in np.unique(row_counts)], "snr_audit": {"target_snr_db": protocol.WEAK_QUALITY_PROTOCOL["snr_db"]}, "mask_logical_sha256": ndarray_sha256(mask)}
    split_audit = {"dataset": dataset, "split_sha256": split.digest, "artifact_sha256": _sha(split_path), "label_seed": split.label_seed,
                   "labels_per_class": split.labels_per_class, "full_gt_persisted": False,
                   "unlabeled_gt_persisted": False, "full_gt_loaded_only_during_split_materialization": True, "sparse_split_source": sparse_split_source, "full_gt_loaded_at_input_boundary": bool(full_gt_loaded_at_input_boundary), "full_gt_used_to_select_sparse_ids": bool(full_gt_used_to_select_sparse_ids)}
    _write(target / "feature_audit.json", feature_audit)
    _write(target / "sparse_split_audit.json", split_audit)
    _write(target / "weak_quality_audit.json", {"mask_logical_sha256": ndarray_sha256(mask), "snr_db": protocol.WEAK_QUALITY_PROTOCOL["snr_db"], "corrupted_pair_count": int(mask.sum()), "per_sample_corrupted_count_unique": [int(value) for value in np.unique(row_counts)], "per_view_corrupted_counts": [int(value) for value in view_counts]})
    _write(target / "materialization_seal.json", {"seal_valid": True, "dataset": dataset,
           "feature_sha256": _sha(features), "split_sha256": _sha(split_path),
           "feature_audit_sha256": _sha(target / "feature_audit.json"),
           "split_audit_sha256": _sha(target / "sparse_split_audit.json"),
           "full_gt_persisted": False, "unlabeled_gt_persisted": False})
    return target

_CALTECH_LABELED_IDS = np.asarray([67, 82, 90, 111, 200, 365, 440, 513, 536, 983, 1027, 1250, 1316, 1385], dtype=np.int64)
_CALTECH_LABELED_TARGETS = np.asarray([4, 6, 2, 3, 0, 4, 1, 5, 6, 3, 0, 2, 5, 1], dtype=np.int64)


def _caltech_runtime_split():
    """Use the frozen backward-compatibility split, never a new selector."""
    from experiments.generic_contract.sparse_label_contract import frozen_caltech_sparse_split
    raw_split = frozen_caltech_sparse_split()
    if not (np.array_equal(raw_split.labeled_ids, _CALTECH_LABELED_IDS) and np.array_equal(raw_split.labeled_targets, _CALTECH_LABELED_TARGETS) and raw_split.label_seed == 20 and raw_split.labels_per_class == 2 and raw_split.labeled_ids.size == 14 and np.array_equal(np.bincount(raw_split.labeled_targets, minlength=7), np.full(7, 2, dtype=np.int64))):
        raise RuntimeError("FORMAL_CALTECH_FROZEN_SPARSE_SPLIT_MISMATCH")
    return _runtime_split_from_generic(raw_split, sample_ids=np.arange(1400, dtype=np.int64), class_count=7)


def materialize_caltech_inputs(output_dir):
    """Authorized Caltech input boundary with a frozen sparse split."""
    from release_core.data import load_caltech
    from release_core.data.weak_quality import apply_half_gaussian_corruption, generate_half_corruption_mask
    data_path = Path("data/Caltech.mat")
    expected = "72fa848269b663f819a8e9bd441628ece1955c654d98e2b10a85be1cd2613d5a"
    if not data_path.is_file() or _sha(data_path) != expected:
        raise RuntimeError("FORMAL_DATASET_SOURCE_UNRESOLVED")
    views, labels = load_caltech(data_path)
    corrupted, _ = apply_half_gaussian_corruption(views, 2.5, 20)
    mask, _ = generate_half_corruption_mask(1400, 6, 20)
    runtime_split = _caltech_runtime_split()
    return materialize_inputs(dataset="Caltech-6V", views=corrupted, corruption_mask=mask, sparse_split=runtime_split, output_dir=output_dir, sparse_split_source="frozen Caltech backward-compatibility split", full_gt_loaded_at_input_boundary=True, full_gt_used_to_select_sparse_ids=False)

def _msrc_source_path():
    candidates = (Path("data/MSRC-v1.mat"), Path("data/MSRC_v1.mat"))
    resolved = tuple(path for path in candidates if path.is_file())
    if len(resolved) != 1:
        raise RuntimeError("FORMAL_MSRC_DATASET_SOURCE_UNRESOLVED")
    return resolved[0]

def materialize_msrc_inputs(output_dir):
    from release_core.data import load_msrc_v1
    from release_core.data.weak_quality import apply_half_gaussian_corruption, generate_half_corruption_mask
    from experiments.generic_contract.sparse_label_contract import materialize_hash_ranked_sparse_split
    views, label_sets = load_msrc_v1(_msrc_source_path())
    labels = np.ascontiguousarray(label_sets[0], dtype=np.int64)
    mask, _ = generate_half_corruption_mask(210, 5, 20)
    row_histogram = np.bincount(mask.sum(axis=1), minlength=4)
    if int(mask.sum()) != 525 or not np.array_equal(row_histogram[2:4], np.array([105, 105])):
        raise RuntimeError("FORMAL_MSRC_WEAK_QUALITY_CONTRACT_MISMATCH")
    if not np.array_equal(mask.sum(axis=0), np.full(5, 105)):
        raise RuntimeError("FORMAL_MSRC_WEAK_QUALITY_CONTRACT_MISMATCH")
    if ndarray_sha256(mask) != "d64f5fbf95518e5a77e44a1fe18a15dee9924b8c2a341e930f86f20ce5ac9e9d":
        raise RuntimeError("FORMAL_MSRC_WEAK_QUALITY_CONTRACT_MISMATCH")
    corrupted, _ = apply_half_gaussian_corruption(views, 2.5, 20)
    raw_split = materialize_hash_ranked_sparse_split(labels, dataset_name="MSRC-v1", label_seed=20, labels_per_class=2)
    runtime_split = _runtime_split_from_generic(raw_split, sample_ids=np.arange(210, dtype=np.int64), class_count=7)
    if runtime_split.labeled_ids.size != 14 or not np.array_equal(np.bincount(runtime_split.labeled_targets, minlength=7), np.full(7, 2)):
        raise RuntimeError("FORMAL_MSRC_SPARSE_SPLIT_CONTRACT_MISMATCH")
    return materialize_inputs(dataset="MSRC-v1", views=corrupted, corruption_mask=mask, sparse_split=runtime_split, output_dir=output_dir, sparse_split_source="generic SHA256-ranked sparse split", full_gt_loaded_at_input_boundary=True, full_gt_used_to_select_sparse_ids=True)


def _bdgp_source_path():
    candidates = (Path("data/BDGP2V_N.mat"),)
    resolved = tuple(path for path in candidates if path.is_file())
    if len(resolved) != 1:
        raise RuntimeError("FORMAL_BDGP_DATASET_SOURCE_UNRESOLVED")
    return resolved[0]


def materialize_bdgp_inputs(output_dir):
    """Authorized BDGP generic input boundary; never use Caltech split IDs."""
    from release_core.data import load_bdgp
    from release_core.data.weak_quality import apply_half_gaussian_corruption, generate_half_corruption_mask
    from experiments.generic_contract.sparse_label_contract import materialize_hash_ranked_sparse_split
    item = next(value for value in protocol.FORMAL_DATASETS if value.name == "BDGP")
    views, label_sets = load_bdgp(_bdgp_source_path())
    labels = np.ascontiguousarray(label_sets[0], dtype=np.int64)
    mask, _ = generate_half_corruption_mask(item.n_samples, item.n_views, 20)
    if not (np.array_equal(mask.sum(axis=1), np.ones(item.n_samples, dtype=np.int64))
            and int(mask.sum()) == 2500
            and np.array_equal(mask.sum(axis=0), np.full(2, 1250, dtype=np.int64))
            and ndarray_sha256(mask) == item.weak_quality_mask_sha256):
        raise RuntimeError("FORMAL_BDGP_WEAK_QUALITY_CONTRACT_MISMATCH")
    corrupted, _ = apply_half_gaussian_corruption(views, 2.5, 20)
    raw_split = materialize_hash_ranked_sparse_split(labels, dataset_name="BDGP", label_seed=20, labels_per_class=2)
    runtime_split = _runtime_split_from_generic(raw_split, sample_ids=np.arange(item.n_samples, dtype=np.int64), class_count=item.n_clusters)
    if not (runtime_split.labeled_ids.size == 10
            and np.array_equal(np.bincount(runtime_split.labeled_targets, minlength=item.n_clusters), np.full(item.n_clusters, 2, dtype=np.int64))):
        raise RuntimeError("FORMAL_BDGP_SPARSE_SPLIT_CONTRACT_MISMATCH")
    return materialize_inputs(dataset="BDGP", views=corrupted, corruption_mask=mask, sparse_split=runtime_split, output_dir=output_dir, sparse_split_source="generic SHA256-ranked sparse split", full_gt_loaded_at_input_boundary=True, full_gt_used_to_select_sparse_ids=True)
