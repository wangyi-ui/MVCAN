"""Regression test for the frozen E1 reliability NumPy-to-Tensor boundary."""

import numpy as np

from experiments.e1_pairwise_utility.train_e1_pairwise_utility import (
    frozen_reliability_tensor,
)


def test_reliability_tensor_is_exact_independent_and_requires_no_grad():
    reliability = np.arange(24, dtype=np.float64).reshape(4, 6) / 23.0
    reliability.setflags(write=False)
    expected = reliability.copy()

    tensor = frozen_reliability_tensor(reliability)

    assert np.array_equal(tensor.numpy(), expected)
    assert not np.shares_memory(tensor.numpy(), reliability)
    assert tensor.numpy().flags.writeable is True
    assert tensor.requires_grad is False

    tensor[0, 0] = -1.0
    assert np.array_equal(reliability, expected)
