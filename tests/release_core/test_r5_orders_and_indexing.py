import numpy as np
import pytest
import torch

from release_core.training import precompute_training_orders


@pytest.mark.parametrize("size,epochs", [(7, 1), (13, 3), (31, 4)])
def test_order_pairs_cover_runtime_ids_and_are_separate(size, epochs):
    ids = np.arange(size, dtype=np.int64) * 3 + 11
    orders = precompute_training_orders(ids, epochs, seed=20)
    assert len(orders.semantic_orders) == len(orders.native_orders) == epochs
    assert orders.generator_objects_distinct
    assert orders.corresponding_values_identical
    for semantic, native in zip(orders.semantic_orders, orders.native_orders):
        assert semantic is not native
        assert semantic.data_ptr() != native.data_ptr()
        assert torch.equal(semantic, native)
        assert np.array_equal(np.sort(semantic.numpy()), np.sort(ids))


def test_orders_are_deterministic_and_seed_sensitive():
    ids = np.asarray([31, 7, 44, 2, 19, 61, 5, 28], dtype=np.int64)
    first = precompute_training_orders(ids, 3, seed=20)
    replay = precompute_training_orders(ids, 3, seed=20)
    changed = precompute_training_orders(ids, 3, seed=30)
    assert all(torch.equal(a, b) for a, b in zip(first.semantic_orders, replay.semantic_orders))
    assert any(not torch.equal(a, b) for a, b in zip(first.semantic_orders, changed.semantic_orders))


def test_semantic_filtering_occurs_after_full_order_batching():
    ids = np.arange(9, dtype=np.int64)
    labeled = np.asarray([1, 4], dtype=np.int64)
    order = np.asarray([4, 8, 0, 5, 1, 2, 7, 3, 6], dtype=np.int64)
    batches = [order[start:start + 4] for start in range(0, order.size, 4)]
    queries = [batch[~np.isin(batch, labeled)] for batch in batches]
    visited = np.concatenate(queries)
    assert [value.tolist() for value in queries] == [[8, 0, 5], [2, 7, 3], [6]]
    assert np.array_equal(np.sort(visited), np.asarray([0, 2, 3, 5, 6, 7, 8]))


@pytest.mark.parametrize(
    "ids,epochs",
    [
        (np.asarray([0, 0, 1], dtype=np.int64), 1),
        (np.arange(3, dtype=np.int64), 0),
    ],
)
def test_invalid_order_inputs_fail_closed(ids, epochs):
    with pytest.raises((ValueError, TypeError)):
        precompute_training_orders(ids, epochs, seed=20)


def test_ids_must_be_int64():
    with pytest.raises(TypeError):
        precompute_training_orders(np.arange(5, dtype=np.int32), 1, seed=20)
