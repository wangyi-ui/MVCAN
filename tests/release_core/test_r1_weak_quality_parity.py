import copy

import numpy as np
import pytest

import weak_quality as legacy
from experiments.generic_contract import generic_weak_quality as generic
from release_core.data import weak_quality as clean


@pytest.mark.parametrize(
    "n_samples,n_views,expected_hash",
    [
        (1400, 6, "e6250535db6e33b9263102cf335fcf9e8552a24cae4b70ec5a94887f4e58bf0b"),
        (210, 5, "d64f5fbf95518e5a77e44a1fe18a15dee9924b8c2a341e930f86f20ce5ac9e9d"),
        (2500, 2, "a3c882c29f2f5064d552e60bdc1a2d585764317e7ba78aa055cd1c3c22e5de07"),
    ],
)
def test_frozen_mask_hashes(n_samples, n_views, expected_hash):
    mask, metadata = clean.generate_half_corruption_mask(n_samples, n_views, 20)
    assert clean.ndarray_sha256(mask) == expected_hash
    assert metadata["mask_sha256"] == expected_hash
    if n_views == 6:
        assert np.all(mask.sum(1) == 3)
    elif n_views == 5:
        counts = mask.sum(1)
        assert np.count_nonzero(counts == 2) == 105
        assert np.count_nonzero(counts == 3) == 105
        assert int(mask.sum()) == 525
    else:
        assert np.all(mask.sum(1) == 1)
        assert mask.sum(0).tolist() == [1250, 1250]


def _views(n_views):
    return [
        (np.arange(7 * (index + 3), dtype=np.float32).reshape(7, index + 3) + 1.0)
        / (index + 2.0)
        for index in range(n_views)
    ]


@pytest.mark.parametrize("n_views", [4, 5])
def test_gaussian_corruption_exact_and_rng_isolated(n_views):
    views = _views(n_views)
    if n_views % 2 == 0:
        old_views, old_audit = legacy.apply_heterogeneous_gaussian_corruption(
            views, n_views // 2, 2.5, 20
        )
    else:
        old_views, old_audit = generic.apply_half_gaussian_corruption(views, 2.5, 20)
    np.random.seed(441)
    before = copy.deepcopy(np.random.get_state())
    new_views, new_audit = clean.apply_half_gaussian_corruption(views, 2.5, 20)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    for original, old, new in zip(views, old_views, new_views):
        assert new.shape == original.shape
        assert new.dtype == original.dtype
        assert np.isfinite(new).all()
        np.testing.assert_array_equal(old, new)
    np.testing.assert_array_equal(old_audit["mask"], new_audit["mask"])
    for key in (
        "mask_sha256", "corrupted_pair_count", "noise_seed",
        "per_view_aggregate_achieved_snr_db",
        "global_aggregate_achieved_snr_db", "snr_target_pass",
        "unchanged_clean_pairs_max_abs_error",
    ):
        assert old_audit[key] == new_audit[key]
    assert new_audit["noise_seed"] == 1000023
