import hashlib
import inspect
import json

import numpy as np
import pytest

from experiments.generic_contract.sparse_label_contract import (
    canonical_split_payload,
    materialize_hash_ranked_sparse_split,
)


def _labels():
    return np.arange(60, dtype=np.int64) % 3


def _expected_ids(labels, dataset_name, seed, count):
    chosen = []
    for class_id in range(3):
        ranked = []
        for sample_id in np.flatnonzero(labels == class_id):
            payload = (
                f"G0_SPARSE_LABEL_V1|dataset={dataset_name}|label_seed={seed}"
                f"|class={class_id}|sample_id={int(sample_id)}"
            )
            ranked.append((hashlib.sha256(payload.encode("utf-8")).hexdigest(), int(sample_id)))
        chosen.extend(sample_id for _, sample_id in sorted(ranked)[:count])
    return np.sort(np.asarray(chosen, dtype=np.int64))


def test_hash_ranked_selection_matches_exact_payload_and_is_deterministic():
    labels = _labels()
    first = materialize_hash_ranked_sparse_split(labels, dataset_name="Synthetic-3V")
    second = materialize_hash_ranked_sparse_split(labels, dataset_name="Synthetic-3V")
    assert np.array_equal(first.labeled_ids, _expected_ids(labels, "Synthetic-3V", 20, 2))
    assert np.array_equal(first.labeled_targets, labels[first.labeled_ids])
    assert first.split_sha256 == second.split_sha256
    assert np.array_equal(first.labeled_ids, second.labeled_ids)


def test_split_hash_uses_documented_canonical_json():
    split = materialize_hash_ranked_sparse_split(_labels(), dataset_name="Synthetic-3V")
    payload = canonical_split_payload(
        split.dataset_name,
        split.label_seed,
        split.labels_per_class,
        split.labeled_ids,
        split.labeled_targets,
        split.unlabeled_ids,
    )
    decoded = payload.decode("utf-8")
    assert decoded == json.dumps(json.loads(decoded), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(payload).hexdigest() == split.split_sha256


def test_one_two_five_label_budgets_are_nested():
    labels = _labels()
    splits = {
        count: materialize_hash_ranked_sparse_split(
            labels, dataset_name="Synthetic-3V", labels_per_class=count
        )
        for count in (1, 2, 5)
    }
    assert set(splits[1].labeled_ids) < set(splits[2].labeled_ids)
    assert set(splits[2].labeled_ids) < set(splits[5].labeled_ids)


def test_returned_arrays_are_read_only_and_form_zero_based_partition():
    split = materialize_hash_ranked_sparse_split(_labels(), dataset_name="Synthetic-3V")
    assert not split.labeled_ids.flags.writeable
    assert not split.labeled_targets.flags.writeable
    assert not split.unlabeled_ids.flags.writeable
    assert np.intersect1d(split.labeled_ids, split.unlabeled_ids).size == 0
    assert np.array_equal(
        np.sort(np.concatenate((split.labeled_ids, split.unlabeled_ids))), np.arange(60)
    )


@pytest.mark.parametrize(
    "labels",
    (
        np.array([1, 1, 2, 2]),
        np.array([0, 0, 2, 2]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([False, False, True, True]),
    ),
)
def test_noncanonical_labels_fail_closed(labels):
    with pytest.raises((TypeError, ValueError)):
        materialize_hash_ranked_sparse_split(labels, dataset_name="Invalid")


def test_insufficient_class_population_fails_closed():
    with pytest.raises(ValueError, match="every class"):
        materialize_hash_ranked_sparse_split(
            np.array([0, 0, 1], dtype=np.int64), dataset_name="Invalid", labels_per_class=2
        )


def test_split_materializer_is_explicitly_not_a_training_api():
    docstring = inspect.getdoc(materialize_hash_ranked_sparse_split)
    assert "SPLIT MATERIALIZATION ONLY" in docstring
    assert "NOT TRAINING API" in docstring
