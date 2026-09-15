"""Immutable deterministic sparse-label split contracts."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np


DEFAULT_LABEL_SEED = 20
DEFAULT_LABELS_PER_CLASS = 2


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(name + " must be an integer")
    return int(value)


def _dataset_name(value):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("dataset_name must be a non-empty canonical name")
    return value


def _readonly_int64(value):
    array = np.array(value, dtype=np.int64, copy=True, order="C")
    array.setflags(write=False)
    return array


def canonical_split_payload(
    dataset_name,
    label_seed,
    labels_per_class,
    labeled_ids,
    labeled_targets,
    unlabeled_ids,
):
    """Return canonical JSON: UTF-8, sorted keys, compact separators."""
    record = {
        "dataset_name": _dataset_name(dataset_name),
        "label_seed": _integer(label_seed, "label_seed"),
        "labels_per_class": _integer(labels_per_class, "labels_per_class"),
        "labeled_ids": [int(value) for value in np.asarray(labeled_ids).tolist()],
        "labeled_targets": [
            int(value) for value in np.asarray(labeled_targets).tolist()
        ],
        "unlabeled_ids": [int(value) for value in np.asarray(unlabeled_ids).tolist()],
    }
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def split_sha256(
    dataset_name,
    label_seed,
    labels_per_class,
    labeled_ids,
    labeled_targets,
    unlabeled_ids,
):
    return hashlib.sha256(
        canonical_split_payload(
            dataset_name,
            label_seed,
            labels_per_class,
            labeled_ids,
            labeled_targets,
            unlabeled_ids,
        )
    ).hexdigest()


@dataclass(frozen=True)
class SparseLabelSplit:
    dataset_name: str
    label_seed: int
    labels_per_class: int
    labeled_ids: np.ndarray
    labeled_targets: np.ndarray
    unlabeled_ids: np.ndarray
    split_sha256: str

    def __post_init__(self):
        name = _dataset_name(self.dataset_name)
        seed = _integer(self.label_seed, "label_seed")
        budget = _integer(self.labels_per_class, "labels_per_class")
        if seed < 0:
            raise ValueError("label_seed must be non-negative")
        if budget < 1:
            raise ValueError("labels_per_class must be positive")
        labeled = _readonly_int64(self.labeled_ids)
        targets = _readonly_int64(self.labeled_targets)
        unlabeled = _readonly_int64(self.unlabeled_ids)
        if labeled.ndim != targets.ndim or labeled.ndim != unlabeled.ndim or labeled.ndim != 1:
            raise ValueError("sparse split arrays must be one-dimensional")
        if labeled.shape != targets.shape:
            raise ValueError("labeled_ids and labeled_targets must have equal shape")
        all_ids = np.concatenate((labeled, unlabeled))
        if (
            np.any(all_ids < 0)
            or np.unique(all_ids).size != all_ids.size
            or not np.array_equal(np.sort(all_ids), np.arange(all_ids.size))
        ):
            raise ValueError("sample IDs must form a unique zero-based partition")
        expected = split_sha256(name, seed, budget, labeled, targets, unlabeled)
        if self.split_sha256 != expected:
            raise ValueError("split_sha256 does not match canonical split payload")
        object.__setattr__(self, "dataset_name", name)
        object.__setattr__(self, "label_seed", seed)
        object.__setattr__(self, "labels_per_class", budget)
        object.__setattr__(self, "labeled_ids", labeled)
        object.__setattr__(self, "labeled_targets", targets)
        object.__setattr__(self, "unlabeled_ids", unlabeled)


def _build_split(dataset_name, label_seed, labels_per_class, labeled, targets, unlabeled):
    digest = split_sha256(
        dataset_name, label_seed, labels_per_class, labeled, targets, unlabeled
    )
    return SparseLabelSplit(
        dataset_name,
        label_seed,
        labels_per_class,
        labeled,
        targets,
        unlabeled,
        digest,
    )


def materialize_hash_ranked_sparse_split(
    labels,
    *,
    dataset_name,
    label_seed=DEFAULT_LABEL_SEED,
    labels_per_class=DEFAULT_LABELS_PER_CLASS,
):
    """SPLIT MATERIALIZATION ONLY — NOT TRAINING API.

    Consume canonical contiguous full labels only at this explicit boundary
    and select each class by the frozen SHA256 ranking payload.
    """
    dataset_name = _dataset_name(dataset_name)
    label_seed = _integer(label_seed, "label_seed")
    labels_per_class = _integer(labels_per_class, "labels_per_class")
    if label_seed < 0:
        raise ValueError("label_seed must be non-negative")
    if labels_per_class < 1:
        raise ValueError("labels_per_class must be positive")
    values = np.asarray(labels)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if values.dtype == np.bool_ or not np.issubdtype(values.dtype, np.integer):
        raise TypeError("labels must contain canonical integer class IDs")
    values = np.ascontiguousarray(values, dtype=np.int64)
    classes = np.unique(values)
    if classes.size < 2 or not np.array_equal(
        classes, np.arange(classes.size, dtype=np.int64)
    ):
        raise ValueError("labels must use canonical contiguous class IDs 0...K-1")

    selected = []
    for class_id in classes.tolist():
        candidates = np.flatnonzero(values == class_id)
        if candidates.size < labels_per_class:
            raise ValueError("every class must contain labels_per_class samples")
        ranked = []
        for sample_id in candidates.tolist():
            payload = (
                "G0_SPARSE_LABEL_V1|dataset="
                + dataset_name
                + "|label_seed="
                + str(label_seed)
                + "|class="
                + str(class_id)
                + "|sample_id="
                + str(sample_id)
            )
            ranked.append((hashlib.sha256(payload.encode("utf-8")).hexdigest(), sample_id))
        ranked.sort(key=lambda item: (item[0], item[1]))
        selected.extend(sample_id for _, sample_id in ranked[:labels_per_class])

    labeled = np.sort(np.asarray(selected, dtype=np.int64))
    targets = values[labeled]
    all_ids = np.arange(values.size, dtype=np.int64)
    unlabeled = np.setdiff1d(all_ids, labeled, assume_unique=True)
    return _build_split(
        dataset_name, label_seed, labels_per_class, labeled, targets, unlabeled
    )


def frozen_caltech_sparse_split():
    """Return the frozen Caltech-6V split without loading full ground truth."""
    from experiments.cyclic_utility import c2_a0_sparse_label_utility_protocol as c2

    labeled = c2.fixed_labeled_ids()
    targets = c2.fixed_labeled_targets()
    sample_ids = c2.canonical_sample_ids()
    unlabeled = np.setdiff1d(sample_ids, labeled, assume_unique=True)
    validated = c2.validate_fixed_sparse_split(labeled, unlabeled, targets)
    return _build_split(
        "Caltech-6V",
        DEFAULT_LABEL_SEED,
        DEFAULT_LABELS_PER_CLASS,
        validated["labeled_sample_ids"],
        validated["labeled_targets"],
        validated["unlabeled_ids"],
    )
