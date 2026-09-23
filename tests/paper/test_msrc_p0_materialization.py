from pathlib import Path

import numpy as np
import pytest

from experiments.paper.transfer_audit import msrc_p0_protocol as p
from experiments.paper.transfer_audit import materialize_msrc_p0_inputs as materializer
from release_core.data.weak_quality import generate_half_corruption_mask, ndarray_sha256


def test_authoritative_dataset_sha_and_sparse_split_are_exact():
    inspected = materializer.inspect_frozen_protocol(p.DATASET_PATH)
    assert inspected["dataset_sha256"] == p.DATASET_SHA256
    assert np.array_equal(inspected["sample_ids"], np.arange(210, dtype=np.int64))
    assert tuple(inspected["split"].labeled_ids) == p.LABELED_IDS
    assert tuple(inspected["split"].labeled_targets) == p.LABELED_TARGETS
    assert inspected["split"].digest == p.SPLIT_SHA256


def test_corruption_contract_and_logical_sha_are_exact():
    mask, audit = generate_half_corruption_mask(210, 5, 20)
    rows = mask.sum(axis=1)
    assert mask.shape == (210, 5)
    assert int(mask.sum()) == 525
    assert np.count_nonzero(rows == 2) == 105
    assert np.count_nonzero(rows == 3) == 105
    assert tuple(mask.sum(axis=0)) == (105, 105, 105, 105, 105)
    assert ndarray_sha256(mask) == audit["mask_sha256"] == p.CORRUPTION_MASK_SHA256


def test_dataset_sha_mismatch_fails_closed_without_fallback(tmp_path, monkeypatch):
    fake = tmp_path / "MSRC_v1.mat"
    fake.write_bytes(b"not-the-authoritative-dataset")
    monkeypatch.setattr(p, "DATASET_PATH", fake)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        materializer.validate_dataset_path(fake)


def test_non_authoritative_same_name_has_no_fallback(tmp_path):
    fake = tmp_path / "MSRC_v1.mat"
    fake.write_bytes(p.DATASET_PATH.read_bytes())
    with pytest.raises(RuntimeError, match="no fallback"):
        materializer.validate_dataset_path(fake)


def test_materialization_refuses_existing_output_before_any_write(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="overwrite"):
        materializer.materialize(dataset_path=p.DATASET_PATH, output_dir=existing)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_feature_whitelist_excludes_all_gt_spellings():
    assert set(name.lower() for name in p.FEATURE_FIELDS).isdisjoint(
        p.FORBIDDEN_FEATURE_FIELDS
    )
