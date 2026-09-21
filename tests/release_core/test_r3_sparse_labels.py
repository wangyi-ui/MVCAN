from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from release_core.semantics import SparseLabelSplit, validate_sparse_label_split


FROZEN_SPLITS = (
    (
        "Caltech-6V",
        1400,
        7,
        [67, 82, 90, 111, 200, 365, 440, 513, 536, 983, 1027, 1250, 1316, 1385],
        [4, 6, 2, 3, 0, 4, 1, 5, 6, 3, 0, 2, 5, 1],
        "954e571808fac8e7d92b7213dde1058410c3318b12cbc110c446f716482b64e6",
    ),
    (
        "MSRC-v1",
        210,
        7,
        [17, 20, 30, 42, 60, 81, 96, 107, 139, 142, 153, 175, 182, 201],
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        "4f16e8b1b8213591a7ea1d36115335ccee43bf568d32bbcb007f929bc9b15231",
    ),
    (
        "BDGP",
        2500,
        5,
        [340, 459, 556, 589, 593, 617, 634, 2012, 2037, 2226],
        [2, 4, 2, 3, 0, 0, 3, 4, 1, 1],
        "dca812ac480c97b3e85ff9e1c4331a39ff794a6b3cfa40e0b3b84f954df68154",
    ),
)


def _split(name="synthetic", sample_count=10, class_count=2):
    sample_ids = np.arange(sample_count, dtype=np.int64)
    labeled_ids = np.asarray([1, 3, 6, 8], dtype=np.int64)
    labeled_targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
    unlabeled_ids = sample_ids[~np.isin(sample_ids, labeled_ids)]
    return SparseLabelSplit(
        sample_ids=sample_ids,
        labeled_ids=labeled_ids,
        labeled_targets=labeled_targets,
        unlabeled_ids=unlabeled_ids,
        class_count=class_count,
        labels_per_class=2,
        label_seed=20,
        dataset_name=name,
    )


def _replace(split, **changes):
    values = {
        "sample_ids": split.sample_ids,
        "labeled_ids": split.labeled_ids,
        "labeled_targets": split.labeled_targets,
        "unlabeled_ids": split.unlabeled_ids,
        "class_count": split.class_count,
        "labels_per_class": split.labels_per_class,
        "label_seed": split.label_seed,
        "dataset_name": split.dataset_name,
    }
    values.update(changes)
    return SparseLabelSplit(**values)


@pytest.mark.parametrize("name,N,K,labeled,targets,digest", FROZEN_SPLITS)
def test_frozen_split_ids_targets_complement_and_digest(
    name, N, K, labeled, targets, digest
):
    sample_ids = np.arange(N, dtype=np.int64)
    labeled_ids = np.asarray(labeled, dtype=np.int64)
    unlabeled_ids = sample_ids[~np.isin(sample_ids, labeled_ids)]
    split = SparseLabelSplit(
        sample_ids,
        labeled_ids,
        np.asarray(targets, dtype=np.int64),
        unlabeled_ids,
        K,
        2,
        20,
        name,
    )
    assert np.array_equal(split.sample_ids, sample_ids)
    assert np.array_equal(split.labeled_ids, labeled_ids)
    assert np.array_equal(split.labeled_targets, targets)
    assert np.array_equal(split.unlabeled_ids, np.setdiff1d(sample_ids, labeled_ids))
    assert np.intersect1d(split.labeled_ids, split.unlabeled_ids).size == 0
    assert np.array_equal(
        np.sort(np.concatenate((split.labeled_ids, split.unlabeled_ids))),
        split.sample_ids,
    )
    assert split.digest == digest


def test_split_is_structurally_and_array_immutable():
    split = _split()
    with pytest.raises(FrozenInstanceError):
        split.label_seed = 30
    for value in (
        split.sample_ids,
        split.labeled_ids,
        split.labeled_targets,
        split.unlabeled_ids,
    ):
        assert value.flags.writeable is False
        with pytest.raises(ValueError):
            value[0] = 99


def test_validation_returns_defensive_readonly_copy_without_reordering():
    split = _split()
    checked = validate_sparse_label_split(split)
    assert checked is not split
    assert np.array_equal(checked.labeled_ids, [1, 3, 6, 8])
    assert np.array_equal(checked.labeled_targets, [0, 1, 0, 1])
    assert np.array_equal(checked.unlabeled_ids, [0, 2, 4, 5, 7, 9])


@pytest.mark.parametrize(
    "changes",
    (
        {"sample_ids": np.asarray([0, 1, 1, 3, 4, 5, 6, 7, 8, 9])},
        {"sample_ids": np.asarray([1, 0, 2, 3, 4, 5, 6, 7, 8, 9])},
        {"labeled_ids": np.asarray([1, 1, 6, 8])},
        {"unlabeled_ids": np.asarray([0, 2, 4, 5, 7, 7])},
        {"labeled_ids": np.asarray([1, 3, 6, 12])},
        {"unlabeled_ids": np.asarray([0, 2, 4, 5, 7, 12])},
        {"unlabeled_ids": np.asarray([0, 2, 4, 5, 7, 8])},
        {"unlabeled_ids": np.asarray([2, 0, 4, 5, 7, 9])},
        {"labeled_targets": np.asarray([0, 1, 0])},
        {"labeled_targets": np.asarray([0, 1, 0, 2])},
        {"labeled_targets": np.asarray([0, 0, 0, 1])},
        {"labels_per_class": 1},
    ),
)
def test_invalid_split_contracts_fail_closed(changes):
    with pytest.raises(ValueError):
        _replace(_split(), **changes)


def test_nonintegral_ids_and_targets_fail_closed():
    split = _split()
    with pytest.raises(ValueError):
        _replace(split, labeled_ids=split.labeled_ids.astype(np.float64))
    with pytest.raises(ValueError):
        _replace(split, labeled_targets=split.labeled_targets.astype(np.float64))

