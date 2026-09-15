"""Immutable dataset-shape contracts with no ground-truth dependency.

The structural boundary is ``X_list[v]: [N, D_v]``.  Downstream scientific
shapes are derived without labels: ``U_cycle: [N, S]`` and
``PredRelation: [N_u, L, S]``.  The class count K is always explicit.
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .action_space import action_count


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(name + " must be an integer")
    return int(value)


def _dataset_name(value):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("dataset_name must be a non-empty canonical name")
    return value


@dataclass(frozen=True)
class DatasetContract:
    """Frozen generic dimensions and their derived scientific shapes."""

    dataset_name: str
    N: int
    V: int
    K: int
    view_dims: tuple
    labels_per_class: int = 2

    def __post_init__(self):
        object.__setattr__(self, "dataset_name", _dataset_name(self.dataset_name))
        for name in ("N", "V", "K", "labels_per_class"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if not isinstance(self.view_dims, (tuple, list)):
            raise TypeError("view_dims must be a tuple or list of integers")
        dimensions = tuple(
            _integer(value, "view_dims[" + str(index) + "]")
            for index, value in enumerate(self.view_dims)
        )
        object.__setattr__(self, "view_dims", dimensions)

        if self.N <= 0:
            raise ValueError("N must be positive")
        if self.V < 2:
            raise ValueError("V must be at least two")
        if self.K < 2:
            raise ValueError("K must be at least two")
        if len(dimensions) != self.V:
            raise ValueError("len(view_dims) must equal V")
        if any(value <= 0 for value in dimensions):
            raise ValueError("all view dimensions must be positive")
        if self.labels_per_class < 1:
            raise ValueError("labels_per_class must be positive")
        if self.L >= self.N:
            raise ValueError("labels_per_class * K must be smaller than N")

    @property
    def L(self):
        return self.labels_per_class * self.K

    @property
    def N_u(self):
        return self.N - self.L

    @property
    def S(self):
        return action_count(self.V)

    @property
    def tensor_shapes(self):
        """Return immutable names-to-shapes for the frozen interfaces."""
        values = {
            "q_local": (self.N, self.V, self.K),
            "q_aligned": (self.N, self.V, self.K),
            "U_cycle": (self.N, self.S),
            "labeled_ids": (self.L,),
            "unlabeled_ids": (self.N_u,),
            "PredRelation": (self.N_u, self.L, self.S),
            "relation_balance_weights": (self.N_u, self.L, self.S),
            "final_predictions": (self.N,),
        }
        return MappingProxyType(values)


def infer_dataset_contract(
    X_list,
    *,
    dataset_name,
    K,
    labels_per_class=2,
):
    """Infer only N, V and D_v from ``X_list[v]: [N, D_v]``.

    K remains an explicit caller/configuration value.  This API deliberately
    accepts no labels and performs no ground-truth inference.
    """
    if not isinstance(X_list, (list, tuple)) or not X_list:
        raise ValueError("X_list must be a non-empty list or tuple")
    sample_count = None
    dimensions = []
    for view_index, view in enumerate(X_list):
        array = np.asarray(view)
        if array.ndim != 2:
            raise ValueError(
                "view " + str(view_index) + " must have shape [N, D_v]"
            )
        if sample_count is None:
            sample_count = array.shape[0]
        elif array.shape[0] != sample_count:
            raise ValueError("all views must have the same N")
        dimensions.append(array.shape[1])
    return DatasetContract(
        dataset_name=dataset_name,
        N=sample_count,
        V=len(X_list),
        K=K,
        view_dims=tuple(dimensions),
        labels_per_class=labels_per_class,
    )
