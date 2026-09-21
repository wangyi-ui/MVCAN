import numpy as np
import pytest
import torch
import torch.nn.functional as F

from release_core.action import utility_conditioned_relation_loss
from release_core.backbone.native_objective import native_objective
from release_core.training import build_decoupled_optimizers
from release_core.training import precompute_training_orders
from tests.release_core.test_r5_optimizer_topology import make_case


def test_order_generation_matches_frozen_two_generator_algorithm():
    ids = np.arange(17, dtype=np.int64)
    clean = precompute_training_orders(ids, 4, seed=20)

    def historical_orders():
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20)
        return tuple(
            torch.from_numpy(ids)[torch.randperm(ids.size, generator=generator)]
            for _ in range(4)
        )

    semantic_reference = historical_orders()
    native_reference = historical_orders()
    assert all(torch.equal(a, b) for a, b in zip(clean.semantic_orders, semantic_reference))
    assert all(torch.equal(a, b) for a, b in zip(clean.native_orders, native_reference))


def test_relation_scalar_matches_authoritative_small_state_formula():
    torch.manual_seed(5)
    V, B, L, K, S = 3, 4, 2, 3, 2
    q_query = [torch.softmax(torch.randn(B, K), dim=1) for _ in range(V)]
    q_anchor = [torch.softmax(torch.randn(L, K), dim=1) for _ in range(V)]
    target = torch.randint(0, 2, (B, L, S), dtype=torch.float32)
    cycle = torch.rand(B, S) + 0.2
    balance = torch.rand(B, L, S) + 0.5
    clean, _ = utility_conditioned_relation_loss(
        q_query, q_anchor, target, cycle, balance
    )
    weight = (cycle[:, None, :] * balance).detach()
    denominator = weight.sum()
    reference_views = []
    for query, anchor in zip(q_query, q_anchor):
        probability = query @ anchor.detach().T
        epsilon = torch.finfo(probability.dtype).eps
        probability = probability.clamp(epsilon, 1.0 - epsilon)[:, :, None]
        bce = -target * torch.log(probability) - (1.0 - target) * torch.log(1.0 - probability)
        reference_views.append(torch.sum(weight * bce) / denominator)
    reference = torch.stack(reference_views).mean()
    assert torch.equal(clean, reference)


def test_native_scalar_matches_authoritative_sum_across_views():
    torch.manual_seed(7)
    values = []
    reference = []
    for _ in range(5):
        x = torch.randn(4, 3)
        reconstruction = torch.randn(4, 3)
        q = torch.softmax(torch.randn(4, 2), dim=1)
        p = torch.softmax(torch.randn(4, 2), dim=1).detach()
        value, _, _ = native_objective(reconstruction, x, q, p, 0.01)
        values.append(value)
        reference.append(F.mse_loss(reconstruction, x) + 0.01 * F.mse_loss(q, p))
    assert torch.equal(torch.stack(values).sum(), torch.stack(reference).sum())


def test_optimizer_defaults_and_full_batch_filter_match_history():
    case = make_case(6)
    semantic, native, audit = build_decoupled_optimizers(case["model"], 1e-4)
    assert audit.view_count == 6
    assert len(semantic) == len(native) == 6
    order = precompute_training_orders(case["sample_ids"], 1, 20).semantic_orders[0].numpy()
    batches = [order[start:start + 4] for start in range(0, order.size, 4)]
    query_batches = [batch[np.isin(batch, case["unlabeled_ids"])] for batch in batches]
    assert np.array_equal(
        np.sort(np.concatenate(query_batches)), np.sort(case["unlabeled_ids"])
    )
    assert all(optimizer.defaults["lr"] == 1e-4 for optimizer in semantic + native)


@pytest.mark.parametrize(
    "epoch,interval,expected",
    [(0, 100, True), (1, 100, False), (99, 100, False), (100, 100, True)],
)
def test_zero_based_refresh_decision(epoch, interval, expected):
    assert (epoch % interval == 0) is expected


def test_historical_step_count_formula():
    N, B, V, epochs = 1400, 256, 6, 20
    batch_count = (N + B - 1) // B
    assert batch_count == 6
    assert V * batch_count == 36
    assert V * batch_count * epochs == 720
