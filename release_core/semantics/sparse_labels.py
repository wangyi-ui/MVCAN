"""Immutable runtime contract for an explicit sparse-label split."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np


def _integer(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(name + " must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(name + " is outside its valid range")
    return result


def _readonly_integer_vector(value, name, allow_empty=False):
    array = np.asarray(value)
    if (
        array.ndim != 1
        or (array.size == 0 and not allow_empty)
        or array.dtype == np.bool_
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(name + " must be a one-dimensional integer array")
    result = np.array(array, dtype=np.int64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _dataset_name(value):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("dataset_name must be a non-empty canonical name")
    return value


def canonical_split_payload(
    dataset_name,
    label_seed,
    labels_per_class,
    labeled_ids,
    labeled_targets,
    unlabeled_ids,
):
    """Return the frozen compact canonical JSON split payload."""
    record = {
        "dataset_name": _dataset_name(dataset_name),
        "label_seed": _integer(label_seed, "label_seed", 0),
        "labels_per_class": _integer(
            labels_per_class, "labels_per_class", 1
        ),
        "labeled_ids": [
            int(value) for value in np.asarray(labeled_ids).tolist()
        ],
        "labeled_targets": [
            int(value) for value in np.asarray(labeled_targets).tolist()
        ],
        "unlabeled_ids": [
            int(value) for value in np.asarray(unlabeled_ids).tolist()
        ],
    }
    return json.dumps(
        record, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def split_sha256(
    dataset_name,
    label_seed,
    labels_per_class,
    labeled_ids,
    labeled_targets,
    unlabeled_ids,
):
    """Hash a split in the frozen generic-contract namespace."""
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
    """Explicit sparse supervision and canonical sample-axis partition."""

    sample_ids: np.ndarray
    labeled_ids: np.ndarray
    labeled_targets: np.ndarray
    unlabeled_ids: np.ndarray
    class_count: int
    labels_per_class: int
    label_seed: int
    dataset_name: str

    def __post_init__(self):
        sample_ids = _readonly_integer_vector(self.sample_ids, "sample_ids")
        labeled_ids = _readonly_integer_vector(self.labeled_ids, "labeled_ids")
        labeled_targets = _readonly_integer_vector(
            self.labeled_targets, "labeled_targets"
        )
        unlabeled_ids = _readonly_integer_vector(
            self.unlabeled_ids, "unlabeled_ids", allow_empty=True
        )
        class_count = _integer(self.class_count, "class_count", 2)
        labels_per_class = _integer(
            self.labels_per_class, "labels_per_class", 1
        )
        label_seed = _integer(self.label_seed, "label_seed", 0)
        dataset_name = _dataset_name(self.dataset_name)

        if labeled_ids.shape != labeled_targets.shape:
            raise ValueError("labeled_ids and labeled_targets must have equal shape")
        if np.unique(sample_ids).size != sample_ids.size:
            raise ValueError("sample_ids must be unique")
        if not np.array_equal(sample_ids, np.sort(sample_ids)):
            raise ValueError("sample_ids must be in canonical ascending order")
        if np.unique(labeled_ids).size != labeled_ids.size:
            raise ValueError("labeled_ids must be unique")
        if np.unique(unlabeled_ids).size != unlabeled_ids.size:
            raise ValueError("unlabeled_ids must be unique")
        if not np.isin(labeled_ids, sample_ids).all():
            raise ValueError("labeled_ids contain an unknown sample ID")
        if not np.isin(unlabeled_ids, sample_ids).all():
            raise ValueError("unlabeled_ids contain an unknown sample ID")
        if np.intersect1d(labeled_ids, unlabeled_ids).size:
            raise ValueError("labeled_ids and unlabeled_ids must be disjoint")

        expected_unlabeled = sample_ids[~np.isin(sample_ids, labeled_ids)]
        if not np.array_equal(unlabeled_ids, expected_unlabeled):
            raise ValueError(
                "unlabeled_ids must be the canonical ascending complement"
            )
        if labeled_ids.size + unlabeled_ids.size != sample_ids.size:
            raise ValueError("sparse split must cover the complete sample axis")
        if np.any((labeled_targets < 0) | (labeled_targets >= class_count)):
            raise ValueError("labeled_targets contain an invalid class index")
        expected_counts = np.full(class_count, labels_per_class, dtype=np.int64)
        counts = np.bincount(labeled_targets, minlength=class_count)
        if not np.array_equal(counts, expected_counts):
            raise ValueError("labeled_targets violate labels_per_class")

        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "labeled_ids", labeled_ids)
        object.__setattr__(self, "labeled_targets", labeled_targets)
        object.__setattr__(self, "unlabeled_ids", unlabeled_ids)
        object.__setattr__(self, "class_count", class_count)
        object.__setattr__(self, "labels_per_class", labels_per_class)
        object.__setattr__(self, "label_seed", label_seed)
        object.__setattr__(self, "dataset_name", dataset_name)

    @property
    def digest(self):
        """Canonical generic split digest."""
        return split_sha256(
            self.dataset_name,
            self.label_seed,
            self.labels_per_class,
            self.labeled_ids,
            self.labeled_targets,
            self.unlabeled_ids,
        )


def validate_sparse_label_split(split):
    """Validate and return an immutable explicit sparse-label split."""
    if not isinstance(split, SparseLabelSplit):
        raise ValueError("split must be a SparseLabelSplit")
    # Reconstruct so validation cannot be bypassed by post-construction mutation.
    return SparseLabelSplit(
        sample_ids=split.sample_ids,
        labeled_ids=split.labeled_ids,
        labeled_targets=split.labeled_targets,
        unlabeled_ids=split.unlabeled_ids,
        class_count=split.class_count,
        labels_per_class=split.labels_per_class,
        label_seed=split.label_seed,
        dataset_name=split.dataset_name,
    )
