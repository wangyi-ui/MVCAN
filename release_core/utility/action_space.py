"""Generic directional complementary-view action enumeration."""

import hashlib
import itertools
import math
import struct
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


def _view_count(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("num_views must be an integer")
    value = int(value)
    if value < 2:
        raise ValueError("num_views must be at least two")
    return value


def action_count(num_views):
    """Return the directional complementary-action count for ``num_views``."""
    num_views = _view_count(num_views)
    half = num_views // 2
    count = math.comb(num_views, half)
    return count if num_views % 2 == 0 else 2 * count


@dataclass(frozen=True)
class ActionSpace:
    """Immutable ordered generator-to-complement action metadata."""

    num_views: int
    generators: tuple
    verifiers: tuple

    def __post_init__(self):
        num_views = _view_count(self.num_views)
        generators = tuple(
            tuple(int(index) for index in action) for action in self.generators
        )
        verifiers = tuple(
            tuple(int(index) for index in action) for action in self.verifiers
        )
        object.__setattr__(self, "num_views", num_views)
        object.__setattr__(self, "generators", generators)
        object.__setattr__(self, "verifiers", verifiers)

        if len(generators) != len(verifiers):
            raise ValueError("generator/verifier action counts differ")
        if len(generators) != action_count(num_views):
            raise ValueError("action count mismatch")

        all_views = set(range(num_views))
        seen = set()
        for generator, verifier in zip(generators, verifiers):
            if (
                len(generator) == 0
                or len(verifier) == 0
                or len(set(generator)) != len(generator)
                or len(set(verifier)) != len(verifier)
                or set(generator).intersection(verifier)
                or set(generator).union(verifier) != all_views
                or verifier
                != tuple(view for view in range(num_views) if view not in generator)
            ):
                raise ValueError("generator/verifier complement invariant failed")
            action = (generator, verifier)
            if action in seen:
                raise ValueError("duplicate directional action")
            seen.add(action)

    @property
    def V(self):
        """Compatibility spelling for the runtime view dimension."""
        return self.num_views

    @property
    def S(self):
        """Number of directed actions."""
        return len(self.generators)

    @property
    def generator_membership(self):
        return _membership_array(self.generators, self.num_views)

    @property
    def verifier_membership(self):
        return _membership_array(self.verifiers, self.num_views)


def build_directional_actions(num_views):
    """Build actions in the frozen native-combination ordering."""
    num_views = _view_count(num_views)
    low = num_views // 2
    sizes = (low,) if num_views % 2 == 0 else (low, low + 1)
    generators = tuple(
        action
        for size in sizes
        for action in itertools.combinations(range(num_views), size)
    )
    verifiers = tuple(
        tuple(view for view in range(num_views) if view not in generator)
        for generator in generators
    )
    return ActionSpace(num_views, generators, verifiers)


def _membership_array(actions, num_views):
    membership = np.zeros((len(actions), num_views), dtype=np.int64)
    for action_id, action in enumerate(actions):
        membership[action_id, list(action)] = 1
    membership.setflags(write=False)
    return membership


def _logical_tensor_sha256(value):
    """Frozen tensor hash: dtype, comma-shape, and C bytes, length framed."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    for component in (
        str(array.dtype).encode("ascii"),
        ",".join(str(int(size)) for size in array.shape).encode("ascii"),
        array.tobytes(order="C"),
    ):
        digest.update(struct.pack(">Q", len(component)))
        digest.update(component)
    return digest.hexdigest()


def _canonical_action_array(actions, membership):
    sizes = {len(action) for action in actions}
    if len(sizes) == 1:
        return np.asarray(actions, dtype=np.int64)
    return membership


def canonical_action_hashes(actions):
    """Return frozen logical hashes preserving action membership and order.

    Uniform-cardinality actions retain the historical subset-array carrier.
    Variable-cardinality actions use their rectangular membership carrier.
    """
    if not isinstance(actions, ActionSpace):
        raise TypeError("actions must be an ActionSpace")
    generator_membership = actions.generator_membership
    verifier_membership = actions.verifier_membership
    generator_carrier = _canonical_action_array(
        actions.generators, generator_membership
    )
    verifier_carrier = _canonical_action_array(
        actions.verifiers, verifier_membership
    )
    return MappingProxyType({
        "generator": _logical_tensor_sha256(generator_carrier),
        "verifier": _logical_tensor_sha256(verifier_carrier),
        "generator_membership": _logical_tensor_sha256(generator_membership),
        "verifier_membership": _logical_tensor_sha256(verifier_membership),
    })
