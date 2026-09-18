import numpy as np
import pytest

from release_core.data.sample_ids import canonical_sample_ids, validate_sample_ids


@pytest.mark.parametrize("size", [0, 1, 7, 31])
def test_canonical_ids(size):
    expected = np.arange(size, dtype=np.int64)
    assert np.array_equal(canonical_sample_ids(size), expected)
    assert validate_sample_ids(expected, size) is expected


def test_invalid_ids_rejected():
    with pytest.raises(TypeError):
        validate_sample_ids(np.arange(4, dtype=np.int32), 4)
    with pytest.raises(ValueError):
        validate_sample_ids(np.array([0, 1, 1, 3], dtype=np.int64), 4)
    with pytest.raises(ValueError):
        validate_sample_ids(np.array([1, 0, 2, 3], dtype=np.int64), 4)
    with pytest.raises(ValueError):
        validate_sample_ids(np.arange(3, dtype=np.int64), 4)

