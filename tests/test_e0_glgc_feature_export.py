"""Targeted tests for the E0-A frozen Caltech feature exporter."""

import json

import numpy as np
import pytest

from experiments.e0_glgc_adapter import export_caltech6v_frozen_features as exporter
import weak_quality


@pytest.fixture(scope="module")
def frozen_sets():
    return exporter.build_frozen_feature_sets()


def test_replayed_clean_and_noisy_views_match_D1_hashes(frozen_sets):
    d1_audit = frozen_sets["d1"]["audit"]

    assert frozen_sets["clean_hashes"] == d1_audit["clean_view_sha256"]
    assert frozen_sets["noisy_hashes"] == d1_audit["corrupted_view_sha256"]
    assert frozen_sets["runtime_audit"]["mask_sha256"] == (
        exporter.EXPECTED_MASK_SHA256
    )


def test_replayed_protocol_is_exact_k3_column_balanced_and_2p5db(frozen_sets):
    audit = frozen_sets["runtime_audit"]

    assert audit["corruption_seed"] == 20
    assert audit["mask_seed"] == 20
    assert audit["noise_seed"] == 1000023
    assert audit["per_sample_corrupted_count_unique"] == [3]
    assert audit["per_view_corrupted_counts"] == [700] * 6
    assert audit["snr_target_pass"]
    assert np.isclose(audit["global_aggregate_achieved_snr_db"], 2.5)


def test_frozen_feature_shapes_dtypes_and_sample_ids(frozen_sets):
    for views in (frozen_sets["clean_views"], frozen_sets["noisy_views"]):
        assert len(views) == 6
        for view, dimension in zip(views, exporter.VIEW_DIMS):
            assert view.shape == (1400, dimension)
            assert view.dtype == np.float32
            assert np.isfinite(view).all()
            assert view.flags.writeable is False
    assert np.array_equal(frozen_sets["sample_ids"], np.arange(1400))
    assert frozen_sets["sample_ids"].dtype == np.int64


def test_replay_is_content_hash_reproducible(frozen_sets):
    replay = exporter.build_frozen_feature_sets()

    assert replay["sample_ids_hash"] == frozen_sets["sample_ids_hash"]
    assert replay["clean_hashes"] == frozen_sets["clean_hashes"]
    assert replay["noisy_hashes"] == frozen_sets["noisy_hashes"]


def test_exported_npz_has_only_features_and_sample_ids(tmp_path):
    result = exporter.export_frozen_features(tmp_path)

    for filename in (
        "caltech6v_clean_seed20.npz",
        "caltech6v_snr2p5_k3_seed20.npz",
    ):
        with np.load(tmp_path / filename, allow_pickle=False) as archive:
            assert tuple(archive.files) == exporter.NPZ_KEYS
            assert not (set(archive.files) & exporter.FORBIDDEN_TRAINABLE_KEYS)
            assert np.array_equal(archive["sample_ids"], np.arange(1400))
    assert result["audit"]["trainable_artifact_forbidden_fields_absent"]
    assert result["audit"]["d1_provenance"]["D1_PROVENANCE_MATCH"]


def test_export_protocol_and_audit_contain_provenance_not_oracle_matrix(tmp_path):
    exporter.export_frozen_features(tmp_path)
    protocol = json.loads((tmp_path / "export_protocol.json").read_text())
    audit = json.loads((tmp_path / "export_audit.json").read_text())

    assert protocol["source_files"]["datasets.py"] == exporter.file_sha256(
        exporter.REPOSITORY_ROOT / "datasets.py"
    )
    assert protocol["source_files"]["weak_quality.py"] == exporter.file_sha256(
        exporter.REPOSITORY_ROOT / "weak_quality.py"
    )
    assert protocol["oracle_mask_in_trainable_artifact"] is False
    assert audit["d1_provenance"]["logical_mask_sha256"] == (
        exporter.EXPECTED_MASK_SHA256
    )
    assert "corruption_mask" not in audit
    assert "mask" not in audit


def test_saved_view_content_hashes_equal_logical_hashes(tmp_path):
    result = exporter.export_frozen_features(tmp_path)

    for arm, filename in (
        ("clean_artifact", "caltech6v_clean_seed20.npz"),
        ("noisy_artifact", "caltech6v_snr2p5_k3_seed20.npz"),
    ):
        with np.load(tmp_path / filename, allow_pickle=False) as archive:
            hashes = [
                weak_quality.ndarray_sha256(archive["X" + str(index)])
                for index in range(1, 7)
            ]
        assert hashes == result["audit"][arm]["view_content_sha256"]
