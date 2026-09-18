"""Explicit-path loaders preserving the frozen preprocessing semantics."""

from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler


def _load(path, required):
    mat = sio.loadmat(str(Path(path)))
    missing = set(required) - set(mat)
    if missing:
        raise KeyError("missing MATLAB keys: " + ", ".join(sorted(missing)))
    return mat


def load_caltech(path):
    mat = _load(path, ("X1", "X2", "X3", "X4", "X5", "X6", "Y"))
    views = [mat["X1"].astype("float32"), mat["X2"].astype("float32")]
    views.append(MinMaxScaler().fit_transform(mat["X3"].astype("float32")))
    views.extend(mat[name].astype("float32") for name in ("X4", "X5", "X6"))
    y = np.squeeze(mat["Y"]).astype("int")
    for index in range(len(y)):
        if y[index] == 95:
            y[index] = 5
    return views, [y]


def load_msrc_v1(path):
    mat = _load(path, ("fea", "gt"))
    raw_views = mat["fea"]
    if not isinstance(raw_views, np.ndarray) or raw_views.dtype != object:
        raise ValueError("MSRC-v1 key 'fea' must be a MATLAB cell array")
    if raw_views.size != 5:
        raise ValueError("MSRC-v1 must contain 5 views")
    raw_y = np.squeeze(np.asarray(mat["gt"]))
    if raw_y.ndim != 1:
        raise ValueError("MSRC-v1 labels must squeeze to shape [N]")
    if not np.isfinite(raw_y).all():
        raise ValueError("MSRC-v1 labels contain NaN or Inf")
    _, y = np.unique(raw_y, return_inverse=True)
    y = y.astype(np.int64, copy=False)
    views = []
    for index, raw_view in enumerate(raw_views.ravel(), start=1):
        view = np.asarray(raw_view)
        if view.ndim != 2:
            raise ValueError("MSRC-v1 view %d must be a matrix" % index)
        if view.shape[0] == y.shape[0]:
            oriented = view
        elif view.shape[1] == y.shape[0] and view.shape[0] != y.shape[0]:
            oriented = view.T
        else:
            raise ValueError("MSRC-v1 view %d has no unambiguous sample axis" % index)
        oriented = oriented.astype(np.float32, copy=False)
        if not np.isfinite(oriented).all():
            raise ValueError("MSRC-v1 view %d contains NaN or Inf" % index)
        views.append(np.ascontiguousarray(oriented))
    return views, [y]


def load_bdgp(path):
    mat = _load(path, ("X1", "X2", "Y"))
    views = [
        np.ascontiguousarray(mat["X1"].astype("float32")),
        np.ascontiguousarray(mat["X2"].astype("float32")),
    ]
    y = np.squeeze(mat["Y"]).astype("int")
    return views, [y]


def load_dataset(data_name, path):
    loaders = {"Caltech-6V": load_caltech, "MSRC-v1": load_msrc_v1, "BDGP": load_bdgp}
    if data_name not in loaders:
        raise ValueError("unsupported dataset: " + str(data_name))
    return loaders[data_name](path)
