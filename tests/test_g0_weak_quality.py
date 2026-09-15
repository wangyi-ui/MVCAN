import inspect

import numpy as np

import weak_quality as legacy
from experiments.generic_contract.generic_weak_quality import (
    apply_half_gaussian_corruption,
    generate_half_corruption_mask,
)


def _synthetic_views():
    rng = np.random.RandomState(20260915)
    return [
        (rng.normal(size=(210, width)) + 0.5).astype(np.float32)
        for width in (7, 11, 5, 13, 9)
    ]


def test_v6_mask_is_bitwise_identical_to_legacy():
    generic_mask, generic_metadata = generate_half_corruption_mask(1400, 6, 20)
    legacy_mask, legacy_metadata = legacy.generate_balanced_corruption_mask(1400, 6, 3, 20)
    assert np.array_equal(generic_mask, legacy_mask)
    assert generic_metadata["mask_sha256"] == legacy_metadata["mask_sha256"]
    assert np.array_equal(generic_mask.sum(axis=1), legacy_mask.sum(axis=1))
    assert np.array_equal(generic_mask.sum(axis=0), legacy_mask.sum(axis=0))


def test_v2_mask_is_bitwise_identical_to_legacy():
    generic_mask, _ = generate_half_corruption_mask(17, 2, 30)
    legacy_mask, _ = legacy.generate_balanced_corruption_mask(17, 2, 1, 30)
    assert np.array_equal(generic_mask, legacy_mask)


def test_v5_structural_half_corruption_statistics():
    mask, metadata = generate_half_corruption_mask(210, 5, 20)
    row_counts = mask.sum(axis=1)
    view_counts = mask.sum(axis=0)
    assert mask.shape == (210, 5)
    assert mask.dtype == np.bool_
    assert int(mask.sum()) == 525
    assert int(np.sum(row_counts == 2)) == 105
    assert int(np.sum(row_counts == 3)) == 105
    assert int(view_counts.max()) - int(view_counts.min()) <= 1
    assert metadata["balanced_mask_pass"] is True


def test_odd_mask_is_deterministic_and_does_not_consume_global_rng():
    first, _ = generate_half_corruption_mask(211, 5, 20)
    second, _ = generate_half_corruption_mask(211, 5, 20)
    assert np.array_equal(first, second)
    np.random.seed(731)
    expected = np.random.random(10)
    np.random.seed(731)
    generate_half_corruption_mask(211, 5, 20)
    assert np.array_equal(np.random.random(10), expected)


def test_even_gaussian_path_is_exact_legacy_delegation():
    rng = np.random.RandomState(9)
    views = [(rng.normal(size=(20, width)) + 1.0).astype(np.float32) for width in (3, 4)]
    generic, generic_audit = apply_half_gaussian_corruption(views, 2.5, 20)
    historical, historical_audit = legacy.apply_heterogeneous_gaussian_corruption(
        views, k=1, snr_db=2.5, corruption_seed=20
    )
    assert all(np.array_equal(left, right) for left, right in zip(generic, historical))
    assert np.array_equal(generic_audit["mask"], historical_audit["mask"])


def test_odd_gaussian_synthetic_audit():
    views = _synthetic_views()
    outputs, audit = apply_half_gaussian_corruption(views, 2.5, 20)
    mask = audit["mask"]
    assert mask.shape == (210, 5)
    assert audit["corrupted_pair_count"] == 525
    assert audit["per_sample_corrupted_count_unique"] == [2, 3]
    assert audit["per_sample_corrupted_count_min"] == 2
    assert audit["per_sample_corrupted_count_max"] == 3
    assert audit["per_view_count_max"] - audit["per_view_count_min"] <= 1
    for view_index, (clean, output) in enumerate(zip(views, outputs)):
        assert output.shape == clean.shape
        assert output.dtype == clean.dtype
        assert np.isfinite(output).all()
        assert np.array_equal(output[~mask[:, view_index]], clean[~mask[:, view_index]])
        assert np.all(np.any(output[mask[:, view_index]] != clean[mask[:, view_index]], axis=1))
    assert np.allclose(
        audit["per_view_aggregate_achieved_snr_db"], 2.5, atol=legacy.SNR_TOLERANCE_DB, rtol=0.0
    )
    assert abs(audit["global_aggregate_achieved_snr_db"] - 2.5) < legacy.SNR_TOLERANCE_DB
    assert audit["snr_target_pass"] is True


def test_odd_gaussian_is_bitwise_deterministic_and_global_rng_isolated():
    views = _synthetic_views()
    first, first_audit = apply_half_gaussian_corruption(views, 2.5, 20)
    second, second_audit = apply_half_gaussian_corruption(views, 2.5, 20)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert first_audit["mask_sha256"] == second_audit["mask_sha256"]
    np.random.seed(913)
    expected = np.random.random(10)
    np.random.seed(913)
    apply_half_gaussian_corruption(views, 2.5, 20)
    assert np.array_equal(np.random.random(10), expected)


def test_all_new_public_weak_quality_apis_are_label_isolated():
    forbidden = {"y", "Y", "label", "labels", "target", "targets", "gt", "GT"}
    for function in (generate_half_corruption_mask, apply_half_gaussian_corruption):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)
