"""Generic data/backbone boundary validation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetContract:
    dataset_name: str
    n_samples: int
    n_views: int
    view_dims: tuple
    n_clusters: int


def infer_dataset_contract(views, dataset_name, n_clusters, expected_views=None):
    if not isinstance(views, (list, tuple)) or not views:
        raise ValueError("views must be a non-empty list or tuple")
    if expected_views is not None and len(views) != int(expected_views):
        raise ValueError("unexpected number of views")
    n_samples = None
    dimensions = []
    for index, value in enumerate(views):
        array = np.asarray(value)
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
            raise ValueError("view %d must be a floating [N,D] matrix" % index)
        if not np.isfinite(array).all():
            raise ValueError("view %d contains NaN or Inf" % index)
        if n_samples is None:
            n_samples = int(array.shape[0])
        elif array.shape[0] != n_samples:
            raise ValueError("all views must share the sample count")
        if array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ValueError("views must be non-empty")
        dimensions.append(int(array.shape[1]))
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError("dataset_name must be non-empty")
    if isinstance(n_clusters, bool) or int(n_clusters) <= 0:
        raise ValueError("n_clusters must be positive")
    return DatasetContract(
        dataset_name, n_samples, len(views), tuple(dimensions), int(n_clusters)
    )

