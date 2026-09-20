from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from experiments.generic_contract.action_space import build_action_space
from release_core.utility.action_space import (
    action_count,
    build_directional_actions,
    canonical_action_hashes,
)


EXPECTED_GENERATORS = {
    2: ((0,), (1,)),
    5: (
        (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
        (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
        (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4),
        (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4),
    ),
    6: (
        (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 1, 5), (0, 2, 3),
        (0, 2, 4), (0, 2, 5), (0, 3, 4), (0, 3, 5), (0, 4, 5),
        (1, 2, 3), (1, 2, 4), (1, 2, 5), (1, 3, 4), (1, 3, 5),
        (1, 4, 5), (2, 3, 4), (2, 3, 5), (2, 4, 5), (3, 4, 5),
    ),
}

EXPECTED_HASHES = {
    2: (
        "8a779fd09d0f7a4ff666981de44701140fed6eb7ee405b7927b6813ce777be80",
        "2424701ad3aa3b1a8c2c8f8aefab6929471ee2bae87b885df1c074c307923105",
    ),
    5: (
        "79abb568021f7c2d96854ba442a0a9195a786223ed7f89931d1ff6e578e0c7d5",
        "5596a1b98b26c4e6b261b3fa0fded7dfb1d2a965aee8799e0f161f46710c517e",
    ),
    6: (
        "1da78797e059f952bf6ae3749fc94ca2dd5bebf83d76b1e8eeb6a03583fe36da",
        "aa726d58832d5eb17ff952049d6be1b3c6eb7295b6fcba6b656c3c1ead80baca",
    ),
}


@pytest.mark.parametrize("num_views,expected", ((2, 2), (5, 20), (6, 20)))
def test_action_count_contract(num_views, expected):
    assert action_count(num_views) == expected
    assert build_directional_actions(num_views).S == expected


@pytest.mark.parametrize("num_views", (2, 5, 6))
def test_exact_frozen_action_order(num_views):
    clean = build_directional_actions(num_views)
    reference = build_action_space(num_views)
    assert clean.generators == EXPECTED_GENERATORS[num_views]
    assert clean.generators == reference.generator_subsets
    assert clean.verifiers == reference.verifier_subsets


@pytest.mark.parametrize("num_views", (2, 5, 6))
def test_directional_complement_membership_and_indices(num_views):
    actions = build_directional_actions(num_views)
    pairs = tuple(zip(actions.generators, actions.verifiers))
    assert len(set(pairs)) == actions.S
    assert actions.generator_membership.shape == (actions.S, num_views)
    assert actions.verifier_membership.shape == (actions.S, num_views)
    assert np.array_equal(
        actions.generator_membership + actions.verifier_membership,
        np.ones((actions.S, num_views), dtype=np.int64),
    )
    assert actions.generator_membership.flags.writeable is False
    assert actions.verifier_membership.flags.writeable is False
    for generator, verifier in pairs:
        assert set(generator).isdisjoint(verifier)
        assert set(generator).union(verifier) == set(range(num_views))
        assert verifier == tuple(
            view for view in range(num_views) if view not in generator
        )
        assert all(0 <= index < num_views for index in generator + verifier)


@pytest.mark.parametrize("num_views", (2, 5, 6))
def test_frozen_canonical_action_hashes(num_views):
    hashes = canonical_action_hashes(build_directional_actions(num_views))
    assert (hashes["generator"], hashes["verifier"]) == EXPECTED_HASHES[num_views]


def test_complementary_directions_are_not_collapsed():
    for num_views in (2, 5):
        actions = build_directional_actions(num_views)
        pairs = set(zip(actions.generators, actions.verifiers))
        for generator, verifier in pairs:
            assert (verifier, generator) in pairs


def test_action_space_is_frozen():
    actions = build_directional_actions(2)
    with pytest.raises(FrozenInstanceError):
        actions.num_views = 3


@pytest.mark.parametrize("value", (-3, -1, 0, 1))
def test_num_views_below_two_is_rejected(value):
    with pytest.raises(ValueError, match="at least two"):
        build_directional_actions(value)


@pytest.mark.parametrize("value", (True, 2.0, "2", None))
def test_non_integer_num_views_is_rejected(value):
    with pytest.raises(TypeError, match="integer"):
        action_count(value)
