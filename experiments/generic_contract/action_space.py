"""Generic directional complementary-view action enumeration."""

import itertools
import math
from dataclasses import dataclass

import numpy as np


def _view_count(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("V must be an integer")
    value = int(value)
    if value < 2:
        raise ValueError("V must be at least two")
    return value


def action_count(V):
    """Return the frozen directional action count S(V)."""
    V = _view_count(V)
    half = V // 2
    count = math.comb(V, half)
    return count if V % 2 == 0 else 2 * count


@dataclass(frozen=True)
class ActionSpace:
    """Immutable variable-cardinality generator/complement actions."""

    V: int
    generator_subsets: tuple
    verifier_subsets: tuple

    def __post_init__(self):
        V = _view_count(self.V)
        generator = tuple(tuple(int(item) for item in row) for row in self.generator_subsets)
        verifier = tuple(tuple(int(item) for item in row) for row in self.verifier_subsets)
        object.__setattr__(self, "V", V)
        object.__setattr__(self, "generator_subsets", generator)
        object.__setattr__(self, "verifier_subsets", verifier)
        if len(generator) != len(verifier) or len(generator) != action_count(V):
            raise ValueError("action count mismatch")
        all_views = set(range(V))
        seen = set()
        for group, complement in zip(generator, verifier):
            if (
                set(group).intersection(complement)
                or set(group).union(complement) != all_views
                or complement != tuple(view for view in range(V) if view not in group)
            ):
                raise ValueError("generator/verifier complement invariant failed")
            action = (group, complement)
            if action in seen:
                raise ValueError("duplicate directional action")
            seen.add(action)

    @property
    def S(self):
        return len(self.generator_subsets)

    @property
    def generator_sizes(self):
        return tuple(len(group) for group in self.generator_subsets)

    @property
    def verifier_sizes(self):
        return tuple(len(group) for group in self.verifier_subsets)


def build_action_space(V):
    """Build actions in the exact preregistered native-combination order."""
    V = _view_count(V)
    low = V // 2
    sizes = (low,) if V % 2 == 0 else (low, low + 1)
    generator = tuple(
        group
        for size in sizes
        for group in itertools.combinations(range(V), size)
    )
    verifier = tuple(
        tuple(view for view in range(V) if view not in group)
        for group in generator
    )
    return ActionSpace(V, generator, verifier)


def historical_action_logical_sha256(subsets):
    """Reuse the historical canonical tensor serializer for rectangular actions."""
    from irv.b4_information_utility import tensor_sha256

    array = np.asarray(subsets, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("historical action logical SHA requires a rectangular array")
    return tensor_sha256(array)
