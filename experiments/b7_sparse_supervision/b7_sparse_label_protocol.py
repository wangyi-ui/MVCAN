"""Frozen sparse-label and sample-view admission protocol for B7-A0.

This module contains no model training.  In particular, the normal admission
builder has no oracle/corruption-mask argument; the diagnostic oracle policy is
kept in a separate function.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.d2_caltech6v.evaluate_d2_a0_utility_transfer import (
    topk_admission,
)
from weak_quality import ndarray_sha256


DATASET_NAME = "Caltech-6V"
SAMPLE_NUM = 1400
VIEW_NUM = 6
CLUSTER_NUM = 7
LABELS_PER_CLASS = 2
LABEL_SPLIT_SEED = 20
TOP_K = 3
SHUFFLE_SEED = 20260818
ARMS = (
    "UNSUP",
    "LABEL_ONLY",
    "U_LABEL",
    "SHUFFLED_U_LABEL",
    "ORACLE_LABEL",
)
NORMAL_ARMS = ARMS[:-1]


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def json_sha256(value):
    """Hash a JSON-compatible value with canonical separators/key ordering."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_class_balanced_split(labels, labels_per_class=2, seed=20):
    """Select a deterministic, class-balanced sparse-label subset.

    A local RandomState is the only RNG used.  Global ``np.random`` state is
    neither read nor changed.
    """
    values = np.asarray(labels)
    _require(values.ndim == 1 and values.size > 0, "labels must have shape [N]")
    _require(np.issubdtype(values.dtype, np.integer), "labels must be integers")
    labels_per_class = int(labels_per_class)
    _require(labels_per_class > 0, "labels_per_class must be positive")
    seed = int(seed)
    classes = np.unique(values)
    rng = np.random.RandomState(seed)
    selected_by_class = {}
    selected = []
    for class_id in classes:
        class_ids = np.flatnonzero(values == class_id).astype(np.int64)
        _require(
            class_ids.size >= labels_per_class,
            "class has fewer samples than labels_per_class",
        )
        chosen = np.sort(
            rng.permutation(class_ids)[:labels_per_class].astype(np.int64)
        )
        selected_by_class[str(int(class_id))] = [
            int(sample_id) for sample_id in chosen
        ]
        selected.extend(chosen.tolist())

    labeled_ids = np.sort(np.asarray(selected, dtype=np.int64))
    all_ids = np.arange(values.size, dtype=np.int64)
    labeled_mask = np.zeros(values.size, dtype=bool)
    labeled_mask[labeled_ids] = True
    unlabeled_ids = all_ids[~labeled_mask]
    _require(
        labeled_ids.size == classes.size * labels_per_class,
        "balanced labeled count mismatch",
    )
    _require(
        np.intersect1d(labeled_ids, unlabeled_ids).size == 0
        and np.array_equal(
            np.sort(np.concatenate((labeled_ids, unlabeled_ids))), all_ids
        ),
        "labeled/unlabeled split is not a partition",
    )
    return {
        "labeled_sample_ids": labeled_ids,
        "unlabeled_sample_ids": unlabeled_ids,
        "labeled_mask": labeled_mask,
        "per_class_labeled_ids": selected_by_class,
        "labels_per_class": labels_per_class,
        "label_split_seed": seed,
        "labeled_ids_sha256": ndarray_sha256(labeled_ids),
    }


def save_label_split(split, output_dir, dataset=DATASET_NAME):
    """Persist the exact sparse-label partition and its provenance."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    labeled_ids = np.asarray(split["labeled_sample_ids"], dtype=np.int64)
    unlabeled_ids = np.asarray(split["unlabeled_sample_ids"], dtype=np.int64)
    np.save(root / "labeled_sample_ids.npy", labeled_ids)
    np.save(root / "unlabeled_sample_ids.npy", unlabeled_ids)
    record = {
        "dataset": str(dataset),
        "N": int(labeled_ids.size + unlabeled_ids.size),
        "K": int(len(split["per_class_labeled_ids"])),
        "labels_per_class": int(split["labels_per_class"]),
        "labeled_count": int(labeled_ids.size),
        "unlabeled_count": int(unlabeled_ids.size),
        "label_split_seed": int(split["label_split_seed"]),
        "per_class_labeled_ids": dict(split["per_class_labeled_ids"]),
        "labeled_ids_sha256": str(split["labeled_ids_sha256"]),
    }
    record["label_split_sha256"] = json_sha256(record)
    with open(root / "label_split.json", "w", encoding="utf-8") as output_file:
        json.dump(record, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
    return record


def frozen_u_topk_admission(utility, top_k=TOP_K):
    """Reuse D2's stable descending Top-k with low-view-ID tie breaking."""
    values = np.asarray(utility)
    _, admission = topk_admission(values, admitted_view_num=int(top_k))
    return admission


def deterministic_within_sample_view_shuffle(utility, shuffle_seed=SHUFFLE_SEED):
    """Permute each sample's Utility values across its view identities."""
    values = np.asarray(utility, dtype=np.float64)
    _require(values.ndim == 2, "utility must have shape [N,V]")
    _require(np.isfinite(values).all(), "utility must be finite")
    rng = np.random.RandomState(int(shuffle_seed))
    mapping = np.empty(values.shape, dtype=np.int64)
    shuffled = np.empty_like(values)
    for sample_id in range(values.shape[0]):
        mapping[sample_id] = rng.permutation(values.shape[1])
        shuffled[sample_id] = values[sample_id, mapping[sample_id]]
        _require(
            np.array_equal(
                np.sort(shuffled[sample_id]), np.sort(values[sample_id])
            ),
            "within-sample Utility multiset changed",
        )
    return shuffled, mapping


def _restrict_to_labeled(base_admission, labeled_sample_ids):
    base = np.asarray(base_admission, dtype=bool)
    _require(base.ndim == 2, "base admission must have shape [N,V]")
    labeled_ids = np.asarray(labeled_sample_ids, dtype=np.int64)
    _require(labeled_ids.ndim == 1, "labeled_sample_ids must have shape [L]")
    _require(
        labeled_ids.size == np.unique(labeled_ids).size
        and np.all((0 <= labeled_ids) & (labeled_ids < base.shape[0])),
        "labeled_sample_ids are invalid",
    )
    result = np.zeros_like(base, dtype=bool)
    result[labeled_ids] = base[labeled_ids]
    return result


def build_normal_supervised_admission(
    arm,
    utility,
    labeled_sample_ids,
    shuffle_seed=SHUFFLE_SEED,
):
    """Build a non-oracle supervision mask; no corruption mask can enter."""
    arm = str(arm).upper()
    _require(arm in NORMAL_ARMS, "unsupported normal B7 arm")
    values = np.asarray(utility)
    _require(values.ndim == 2, "utility must have shape [N,V]")
    shuffle_mapping = None
    if arm == "UNSUP":
        base = np.zeros(values.shape, dtype=bool)
    elif arm == "LABEL_ONLY":
        base = np.ones(values.shape, dtype=bool)
    elif arm == "U_LABEL":
        base = frozen_u_topk_admission(values)
    else:
        shuffled, shuffle_mapping = deterministic_within_sample_view_shuffle(
            values, shuffle_seed=shuffle_seed
        )
        base = frozen_u_topk_admission(shuffled)
    admission = _restrict_to_labeled(base, labeled_sample_ids)
    return admission, shuffle_mapping


def build_oracle_supervised_admission(corruption_mask, labeled_sample_ids):
    """Build the diagnostic-only clean-view supervision mask."""
    corrupt = np.asarray(corruption_mask, dtype=bool)
    _require(corrupt.ndim == 2, "corruption_mask must have shape [N,V]")
    clean = np.logical_not(corrupt)
    return _restrict_to_labeled(clean, labeled_sample_ids)


def supervised_channel_count(admission_mask):
    mask = np.asarray(admission_mask, dtype=bool)
    _require(mask.ndim == 2, "admission_mask must have shape [N,V]")
    return int(mask.sum(dtype=np.int64))
