import numpy as np

from experiments.cyclic_utility import c0_complementary_semantic_verification as c0
from experiments.cyclic_utility import c2_a0_sparse_label_utility_protocol as c2
from experiments.generic_contract.action_space import build_action_space
from experiments.generic_contract.generic_weak_quality import generate_half_corruption_mask
from experiments.generic_contract.sparse_label_contract import frozen_caltech_sparse_split
from weak_quality import generate_balanced_corruption_mask, ndarray_sha256


def test_caltech_action_arrays_match_frozen_c0_exactly():
    generic = build_action_space(6)
    historical_generator, historical_verifier = c0.enumerate_complementary_splits()
    assert np.array_equal(np.asarray(generic.generator_subsets), historical_generator)
    assert np.array_equal(np.asarray(generic.verifier_subsets), historical_verifier)


def test_caltech_half_mask_matches_frozen_legacy_exactly():
    generic, generic_metadata = generate_half_corruption_mask(1400, 6, 20)
    historical, historical_metadata = generate_balanced_corruption_mask(1400, 6, 3, 20)
    assert np.array_equal(generic, historical)
    assert generic_metadata["mask_sha256"] == historical_metadata["mask_sha256"]


def test_caltech_frozen_sparse_ids_targets_and_partition_are_preserved():
    split = frozen_caltech_sparse_split()
    assert tuple(split.labeled_ids) == c2.FIXED_LABELED_IDS
    assert tuple(split.labeled_targets) == c2.FIXED_LABELED_TARGETS
    assert split.labeled_ids.shape == (14,)
    assert split.unlabeled_ids.shape == (1386,)
    assert np.array_equal(np.bincount(split.labeled_targets, minlength=7), np.full(7, 2))


def test_caltech_canonical_sample_ids_and_logical_hash_are_preserved():
    sample_ids = c2.canonical_sample_ids()
    assert np.array_equal(sample_ids, np.arange(1400, dtype=np.int64))
    assert ndarray_sha256(sample_ids) == "887cb8dff9918e38f0b0c7f467d3b9aec75dcef9bf821b2307fd076ca5c42237"
