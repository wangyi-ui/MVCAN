"""Canonical row-identity helpers."""

import numpy as np


def canonical_sample_ids(n_samples):
    if isinstance(n_samples, (bool, np.bool_)) or not isinstance(
        n_samples, (int, np.integer)
    ):
        raise TypeError("n_samples must be an integer")
    if int(n_samples) < 0:
        raise ValueError("n_samples must be nonnegative")
    return np.arange(int(n_samples), dtype=np.int64)


def validate_sample_ids(sample_ids, n_samples, require_identity=True):
    values = np.asarray(sample_ids)
    if values.shape != (int(n_samples),):
        raise ValueError("sample_ids must have shape [N]")
    if values.dtype != np.dtype(np.int64):
        raise TypeError("sample_ids must have dtype int64")
    if np.unique(values).size != values.size:
        raise ValueError("sample_ids must be unique")
    if require_identity and not np.array_equal(values, canonical_sample_ids(n_samples)):
        raise ValueError("sample_ids must preserve canonical identity order")
    return values
