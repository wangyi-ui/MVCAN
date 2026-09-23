from types import SimpleNamespace

import numpy as np
import pytest

from experiments.paper.diagnostics import msrc_pre_r2_carrier_temporal_alignment as audit
from release_core.semantics import SparseLabelSplit, validate_sparse_label_split


def _historical():
    sample_ids = np.arange(8, dtype=np.int64)
    return {
        "sample_ids": sample_ids,
        "split": {
            "labeled_ids": np.array([0, 3, 5, 7], dtype=np.int64),
            "labeled_targets": np.array([0, 0, 1, 1], dtype=np.int64),
            "unlabeled_ids": np.array([1, 2, 4, 6], dtype=np.int64),
        },
        "contract": SimpleNamespace(K=2, labels_per_class=2),
    }


def _legacy():
    return SimpleNamespace(MSRC_RUNTIME_SPEC=SimpleNamespace(dataset_name="MSRC"))


def _split(**changes):
    fields = {
        "sample_ids": np.arange(8, dtype=np.int64),
        "labeled_ids": np.array([0, 3, 5, 7], dtype=np.int64),
        "labeled_targets": np.array([0, 0, 1, 1], dtype=np.int64),
        "unlabeled_ids": np.array([1, 2, 4, 6], dtype=np.int64),
        "class_count": 2,
        "labels_per_class": 2,
        "label_seed": 20,
        "dataset_name": "MSRC",
    }
    fields.update(changes)
    return SparseLabelSplit(**fields)


def test_historical_dict_constructs_valid_exact_sparse_label_split():
    historical = _historical()
    split = audit._historical_sparse_split(historical, _legacy())
    assert isinstance(split, SparseLabelSplit)
    assert isinstance(validate_sparse_label_split(split), SparseLabelSplit)
    assert np.array_equal(split.sample_ids, historical["sample_ids"])
    assert np.array_equal(split.labeled_ids, historical["split"]["labeled_ids"])
    assert np.array_equal(split.labeled_targets, historical["split"]["labeled_targets"])
    assert np.array_equal(split.unlabeled_ids, historical["split"]["unlabeled_ids"])
    assert split.class_count == 2
    assert split.labels_per_class == 2
    assert split.label_seed == 20
    assert split.dataset_name == "MSRC"


def test_historical_current_sparse_split_parity_passes():
    historical = audit._historical_sparse_split(_historical(), _legacy())
    assert audit._sparse_split_exact(historical, _split()) is True
    assert audit._require_sparse_split_parity(historical, _split()) is None


@pytest.mark.parametrize("other", (
    _split(labeled_ids=np.array([0, 2, 5, 7]),
           unlabeled_ids=np.array([1, 3, 4, 6])),
    _split(labeled_targets=np.array([0, 1, 0, 1])),
    SparseLabelSplit(
        sample_ids=np.arange(10, 18, dtype=np.int64),
        labeled_ids=np.array([10, 13, 15, 17], dtype=np.int64),
        labeled_targets=np.array([0, 0, 1, 1], dtype=np.int64),
        unlabeled_ids=np.array([11, 12, 14, 16], dtype=np.int64),
        class_count=2, labels_per_class=2, label_seed=20, dataset_name="MSRC",
    ),
    _split(dataset_name="MSRC-alternate"),
))
def test_sparse_split_array_or_metadata_mismatch_fails_closed(other):
    with pytest.raises(RuntimeError, match="HISTORICAL_CURRENT_SPARSE_SPLIT_CONTRACT_MISMATCH"):
        audit._require_sparse_split_parity(_split(), other)


def test_simplenamespace_is_rejected_by_release_contract():
    invalid = SimpleNamespace(
        sample_ids=np.arange(8), labeled_ids=np.array([0]),
        labeled_targets=np.array([0]), unlabeled_ids=np.arange(1, 8),
        class_count=2, labels_per_class=1, label_seed=20, dataset_name="MSRC",
    )
    with pytest.raises(ValueError, match="SparseLabelSplit"):
        validate_sparse_label_split(invalid)


def test_split_adapter_retains_gt_firewall():
    text = open(audit.__file__, encoding="utf-8").read().lower()
    assert "load_full_gt" not in text
    assert "ground_truth" not in text
