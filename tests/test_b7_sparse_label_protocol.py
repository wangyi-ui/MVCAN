"""Unit tests for the frozen B7-A0 sparse-label/admission protocol."""

import inspect

import numpy as np

from experiments.b7_sparse_supervision import b7_sparse_label_protocol as protocol
from weak_quality import ndarray_sha256


def _labels(sample_per_class=10):
    return np.repeat(np.arange(7, dtype=np.int64), sample_per_class)


def _utility(sample_num=70):
    rng = np.random.RandomState(9)
    return rng.uniform(size=(sample_num, 6)).astype(np.float64)


def _corruption_mask(sample_num=70):
    mask = np.zeros((sample_num, 6), dtype=bool)
    for sample_id in range(sample_num):
        mask[sample_id, (sample_id + np.arange(3)) % 6] = True
    return mask


def test_class_balanced_split_is_seven_times_two():
    labels = _labels()
    split = protocol.make_class_balanced_split(labels, 2, 20)

    assert split["labeled_sample_ids"].shape == (14,)
    assert split["unlabeled_sample_ids"].shape == (56,)
    assert all(
        len(sample_ids) == 2
        for sample_ids in split["per_class_labeled_ids"].values()
    )
    assert np.array_equal(
        np.bincount(labels[split["labeled_sample_ids"]], minlength=7),
        np.full(7, 2),
    )


def test_split_is_exactly_deterministic_for_the_same_seed():
    first = protocol.make_class_balanced_split(_labels(), 2, 20)
    second = protocol.make_class_balanced_split(_labels(), 2, 20)

    assert np.array_equal(
        first["labeled_sample_ids"], second["labeled_sample_ids"]
    )
    assert first["labeled_ids_sha256"] == second["labeled_ids_sha256"]


def test_split_changes_for_a_different_seed():
    first = protocol.make_class_balanced_split(_labels(), 2, 20)
    second = protocol.make_class_balanced_split(_labels(), 2, 21)

    assert not np.array_equal(
        first["labeled_sample_ids"], second["labeled_sample_ids"]
    )


def test_split_uses_a_local_rng_without_changing_global_numpy_state():
    np.random.seed(1234)
    expected = np.random.get_state()
    protocol.make_class_balanced_split(_labels(), 2, 20)
    actual = np.random.get_state()

    assert expected[0] == actual[0]
    assert np.array_equal(expected[1], actual[1])
    assert expected[2:] == actual[2:]


def test_all_arms_use_the_exact_same_labeled_sample_ids():
    split = protocol.make_class_balanced_split(_labels(), 2, 20)
    labeled_ids = split["labeled_sample_ids"]
    utility = _utility()
    masks = {}
    for arm in protocol.NORMAL_ARMS:
        masks[arm], _ = protocol.build_normal_supervised_admission(
            arm, utility, labeled_ids
        )
    masks["ORACLE_LABEL"] = protocol.build_oracle_supervised_admission(
        _corruption_mask(), labeled_ids
    )

    unlabeled = split["unlabeled_sample_ids"]
    assert all(not mask[unlabeled].any() for mask in masks.values())
    assert all(mask.shape == utility.shape for mask in masks.values())


def test_fixed_arm_supervised_channel_counts_are_exact():
    split = protocol.make_class_balanced_split(_labels(), 2, 20)
    labeled_ids = split["labeled_sample_ids"]
    utility = _utility()
    expected = {
        "UNSUP": 0,
        "LABEL_ONLY": 84,
        "U_LABEL": 42,
        "SHUFFLED_U_LABEL": 42,
    }
    for arm, count in expected.items():
        admission, _ = protocol.build_normal_supervised_admission(
            arm, utility, labeled_ids
        )
        assert protocol.supervised_channel_count(admission) == count
    oracle = protocol.build_oracle_supervised_admission(
        _corruption_mask(), labeled_ids
    )
    assert protocol.supervised_channel_count(oracle) == 42


def test_u_label_top3_ties_match_d2_stable_low_id_behavior():
    utility = np.array([
        [0.5, 0.5, 0.1, 0.5, 0.2, 0.0],
        [0.3, 0.9, 0.9, 0.9, 0.1, 0.0],
    ])
    admission = protocol.frozen_u_topk_admission(utility)

    assert np.array_equal(np.flatnonzero(admission[0]), [0, 1, 3])
    assert np.array_equal(np.flatnonzero(admission[1]), [1, 2, 3])


def test_all_admission_builders_leave_utility_byte_exact():
    utility = _utility()
    before = utility.copy()
    before_hash = ndarray_sha256(utility)
    labeled_ids = protocol.make_class_balanced_split(_labels(), 2, 20)[
        "labeled_sample_ids"
    ]
    for arm in protocol.NORMAL_ARMS:
        protocol.build_normal_supervised_admission(arm, utility, labeled_ids)

    assert np.array_equal(utility, before)
    assert ndarray_sha256(utility) == before_hash


def test_shuffled_control_is_deterministic_and_preserves_each_row_multiset():
    utility = _utility()
    first, first_mapping = protocol.deterministic_within_sample_view_shuffle(
        utility, protocol.SHUFFLE_SEED
    )
    second, second_mapping = protocol.deterministic_within_sample_view_shuffle(
        utility, protocol.SHUFFLE_SEED
    )

    assert np.array_equal(first, second)
    assert np.array_equal(first_mapping, second_mapping)
    assert np.array_equal(np.sort(first, axis=1), np.sort(utility, axis=1))
    assert all(
        np.array_equal(np.sort(row), np.arange(6)) for row in first_mapping
    )


def test_normal_admission_api_cannot_accept_an_oracle_mask():
    parameters = inspect.signature(
        protocol.build_normal_supervised_admission
    ).parameters

    assert set(parameters) == {
        "arm",
        "utility",
        "labeled_sample_ids",
        "shuffle_seed",
    }
    assert all(
        token not in name.lower()
        for name in parameters
        for token in ("oracle", "corrupt", "clean", "mask")
    )
