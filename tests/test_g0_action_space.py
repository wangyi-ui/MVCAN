import numpy as np
import pytest

from experiments.cyclic_utility import c0_complementary_semantic_verification as c0
from experiments.generic_contract.action_space import (
    action_count,
    build_action_space,
    historical_action_logical_sha256,
)


EXPECTED_GENERATOR_SHA = "1da78797e059f952bf6ae3749fc94ca2dd5bebf83d76b1e8eeb6a03583fe36da"
EXPECTED_VERIFIER_SHA = "aa726d58832d5eb17ff952049d6be1b3c6eb7295b6fcba6b656c3c1ead80baca"


@pytest.mark.parametrize("V,expected", ((2, 2), (3, 6), (5, 20), (6, 20)))
def test_frozen_action_counts(V, expected):
    assert action_count(V) == expected
    assert build_action_space(V).S == expected


@pytest.mark.parametrize("value", (True, 1, 2.5, "6"))
def test_invalid_action_view_counts_fail_closed(value):
    with pytest.raises((TypeError, ValueError)):
        action_count(value)


def test_v2_exact_directional_order():
    actions = build_action_space(2)
    assert actions.generator_subsets == ((0,), (1,))
    assert actions.verifier_subsets == ((1,), (0,))


def test_v3_first_low_then_high_native_lexicographic_blocks():
    actions = build_action_space(3)
    assert actions.generator_subsets == (
        (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)
    )
    assert actions.generator_sizes == (1, 1, 1, 2, 2, 2)
    assert actions.verifier_sizes == (2, 2, 2, 1, 1, 1)


def test_v5_has_two_ordered_ten_action_blocks():
    actions = build_action_space(5)
    assert actions.generator_sizes[:10] == (2,) * 10
    assert actions.generator_sizes[10:] == (3,) * 10
    assert actions.generator_subsets[0] == (0, 1)
    assert actions.generator_subsets[9] == (3, 4)
    assert actions.generator_subsets[10] == (0, 1, 2)
    assert actions.generator_subsets[-1] == (2, 3, 4)


@pytest.mark.parametrize("V", (2, 3, 5, 6))
def test_every_action_is_unique_disjoint_and_complete(V):
    actions = build_action_space(V)
    pairs = tuple(zip(actions.generator_subsets, actions.verifier_subsets))
    assert len(set(pairs)) == actions.S
    for generator, verifier in pairs:
        assert set(generator).isdisjoint(verifier)
        assert set(generator).union(verifier) == set(range(V))
        assert verifier == tuple(view for view in range(V) if view not in generator)


def test_v6_exact_historical_arrays_order_and_logical_hashes():
    actions = build_action_space(6)
    historical_generator, historical_verifier = c0.enumerate_complementary_splits()
    generator = np.asarray(actions.generator_subsets, dtype=np.int64)
    verifier = np.asarray(actions.verifier_subsets, dtype=np.int64)
    assert np.array_equal(generator, historical_generator)
    assert np.array_equal(verifier, historical_verifier)
    assert historical_action_logical_sha256(generator) == EXPECTED_GENERATOR_SHA
    assert historical_action_logical_sha256(verifier) == EXPECTED_VERIFIER_SHA
