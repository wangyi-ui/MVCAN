"""Targeted E1 tensor, gradient, provenance, and identity tests."""

import inspect

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    EXPECTED_RELIABILITY_SHA256,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    align_semantic_probabilities,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    glgc_positive_pair_contrastive_loss,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    positive_pair_weights,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    reliability_pair_factor,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    robust_inter_affinity,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    whole_row_reliability_shuffle,
)
from experiments.e1_pairwise_utility.train_e1_pairwise_utility import (
    LAMBDA1,
)
from experiments.e1_pairwise_utility.train_e1_pairwise_utility import (
    load_frozen_feature_artifact,
)
from experiments.e1_pairwise_utility.train_e1_pairwise_utility import (
    load_frozen_reliability,
)
from experiments.e1_pairwise_utility.train_e1_pairwise_utility import (
    native_mvcan_losses,
)
from experiments.e1_pairwise_utility.train_e1_pairwise_utility import parse_args


def _permutation_matrices(view_num, cluster_num):
    matrices = []
    for view_id in range(view_num):
        order = torch.roll(torch.arange(cluster_num), shifts=view_id)
        matrices.append(torch.eye(cluster_num)[order])
    return torch.stack(matrices)


def _released_noise_inter_graph_reference(h_sem, temperature=0.07, eta=0.2):
    """Literal compact transcription of released util.py for comparison."""
    batch_size, view_num, _ = h_sem.shape
    views = [F.normalize(h_sem[:, view_id, :]) for view_id in range(view_num)]
    result = torch.zeros(
        (view_num, view_num, batch_size, batch_size), dtype=h_sem.dtype
    )
    diagonal = torch.eye(batch_size, dtype=torch.bool)
    for left in range(view_num):
        for right in range(view_num):
            graph = (
                2.0 - 2.0 * (views[left] @ views[right].t())
            ).clamp(min=0.0)
            graph = torch.exp(-graph / temperature)
            if left == right:
                graph[diagonal] = 1.0
            else:
                graph[diagonal] = (
                    graph[diagonal]
                    / graph.diag().max().clamp_min(1e-7).detach()
                )
            graph = graph / graph.sum(1, keepdim=True).clamp_min(1e-7)
            result[left, right] = graph
    identity = torch.eye(batch_size)
    for left in range(view_num):
        for right in range(view_num):
            if left != right:
                result[left, right] = result[left, right].mm(
                    result[right, left].t()
                )
                result[left, right] += eta * identity
    return result


def test_hungarian_alignment_has_exact_shape_value_and_gradient_boundary():
    torch.manual_seed(3)
    q_local = torch.softmax(torch.randn(5, 3, 4), dim=-1).requires_grad_(True)
    matches = _permutation_matrices(3, 4).requires_grad_(True)

    q_aligned, h_sem = align_semantic_probabilities(q_local, matches)
    expected = torch.stack(
        [q_local[:, view_id] @ matches[view_id].detach().t()
         for view_id in range(3)],
        dim=1,
    )
    h_sem.square().sum().backward()

    assert q_aligned.shape == (5, 3, 4)
    assert h_sem.shape == (5, 3, 4)
    assert torch.allclose(q_aligned, expected)
    assert q_local.grad is not None
    assert matches.grad is None


def test_graph_is_exact_released_noise_logic_and_always_stop_gradient():
    torch.manual_seed(4)
    h_sem = F.normalize(torch.randn(6, 3, 5), dim=-1).requires_grad_(True)

    graph = robust_inter_affinity(h_sem)
    reference = _released_noise_inter_graph_reference(h_sem.detach())

    assert graph.shape == (3, 3, 6, 6)
    assert torch.allclose(graph, reference)
    assert graph.requires_grad is False
    assert graph.grad_fn is None


def test_reliability_factor_and_arm_weights_are_exact_and_frozen():
    reliability = torch.tensor([
        [1.0, 0.25, 0.4],
        [0.0, 0.8, 0.9],
        [1.2, -0.1, 0.5],
    ], requires_grad=True)
    graph = torch.zeros(3, 3, 3, 3)
    graph[0, 2].diagonal().copy_(torch.tensor([0.2, 0.4, 0.6]))
    graph[2, 0].diagonal().copy_(torch.tensor([0.4, 0.6, 0.8]))

    factor = reliability_pair_factor(reliability, 0, 2)
    lwc, g_pair, lwc_factor = positive_pair_weights(
        graph, reliability, 0, 2, "LWC"
    )
    calibrated, _, calibrated_factor = positive_pair_weights(
        graph, reliability, 0, 2, "R_LWC"
    )

    expected_factor = torch.sqrt(torch.tensor([0.4, 0.0, 0.5]))
    assert torch.allclose(factor, expected_factor)
    assert torch.allclose(g_pair, torch.tensor([0.3, 0.5, 0.7]))
    assert torch.equal(lwc, g_pair)
    assert lwc_factor is None
    assert torch.allclose(calibrated_factor, expected_factor)
    assert torch.allclose(calibrated, g_pair * expected_factor)
    assert not calibrated.requires_grad


def test_positive_weighted_contrastive_matches_released_logit_rule_and_grads():
    torch.manual_seed(7)
    left = F.normalize(torch.randn(4, 5), dim=-1).requires_grad_(True)
    right = F.normalize(torch.randn(4, 5), dim=-1).requires_grad_(True)
    weights = torch.tensor([0.2, 0.4, 0.6, 0.8], requires_grad=True)

    loss = glgc_positive_pair_contrastive_loss(left, right, weights)

    features = torch.cat((left, right), dim=0)
    similarity = features @ features.t() / 0.5
    positive = torch.cat(
        (torch.diag(similarity, 4), torch.diag(similarity, -4))
    ).reshape(8, 1)
    positive = positive * torch.cat((weights.detach(), weights.detach())).view(8, 1)
    mask = torch.ones((8, 8), dtype=torch.bool)
    mask.fill_diagonal_(False)
    rows = torch.arange(4)
    mask[rows, 4 + rows] = False
    mask[4 + rows, rows] = False
    negatives = similarity[mask].reshape(8, -1)
    manual = F.cross_entropy(
        torch.cat((positive, negatives), dim=1),
        torch.zeros(8, dtype=torch.long),
        reduction="sum",
    ) / 8
    loss.backward()

    assert torch.allclose(loss, manual)
    assert left.grad is not None and torch.count_nonzero(left.grad) > 0
    assert right.grad is not None and torch.count_nonzero(right.grad) > 0
    assert weights.grad is None


def test_shuffle_is_deterministic_whole_row_and_preserves_all_distributions():
    reliability = np.arange(60, dtype=np.float64).reshape(10, 6) / 59.0

    first, permutation, first_audit = whole_row_reliability_shuffle(
        reliability, seed=20
    )
    second, second_permutation, second_audit = whole_row_reliability_shuffle(
        reliability, seed=20
    )

    assert np.array_equal(permutation, second_permutation)
    assert np.array_equal(first, second)
    assert np.array_equal(first, reliability[permutation])
    assert all(first_audit[key] for key in (
        "permutation_is_bijection_pass",
        "permutation_nonidentity_pass",
        "R_multiset_audit_pass",
        "per_view_distribution_audit_pass",
    ))
    assert first_audit["permutation_sha256"] == second_audit["permutation_sha256"]


def test_native_loss_is_exact_mvcam_rec_plus_point_zero_one_clu():
    torch.manual_seed(9)
    x_views = [torch.randn(4, 3), torch.randn(4, 2)]
    reconstructions = [torch.randn(4, 3), torch.randn(4, 2)]
    q_views = [torch.softmax(torch.randn(4, 3), -1) for _ in range(2)]
    p_all = torch.softmax(torch.randn(4, 3), -1)
    matches = torch.stack((torch.eye(3), torch.eye(3)[[1, 2, 0]]))

    losses, diagnostics = native_mvcan_losses(
        x_views, reconstructions, q_views, p_all, matches
    )

    for view_id in range(2):
        expected = F.mse_loss(reconstructions[view_id], x_views[view_id])
        expected = expected + LAMBDA1 * F.mse_loss(
            q_views[view_id], p_all @ matches[view_id]
        )
        assert torch.equal(losses[view_id], expected)
        assert len(diagnostics[view_id]) == 2


def test_real_frozen_R_and_sample_id_provenance_without_loading_mask():
    _, sample_ids, feature_audit = load_frozen_feature_artifact()
    reliability, reliability_audit = load_frozen_reliability(
        sample_ids, feature_provenance=feature_audit
    )

    assert sample_ids.shape == (1400,)
    assert np.array_equal(sample_ids, np.arange(1400))
    assert reliability.shape == (1400, 6)
    assert reliability_audit["R_logical_sha256"] == EXPECTED_RELIABILITY_SHA256
    assert reliability_audit["loaded_fields"] == ["U"]
    assert reliability_audit["corruption_mask_loaded"] is False
    assert reliability.flags.writeable is False


def test_cli_freezes_seed_schedule_and_exposes_no_tuning_arguments():
    smoke = parse_args(["--arm", "R_LWC", "--epochs", "2"])
    formal = parse_args(["--arm", "BASE", "--epochs", "100"])
    parameters = inspect.signature(load_frozen_reliability).parameters

    assert smoke.epochs == 2 and formal.epochs == 100
    assert not hasattr(smoke, "seed")
    assert not hasattr(smoke, "lambda1")
    assert not hasattr(smoke, "temperature")
    assert not hasattr(smoke, "alpha")
    assert not hasattr(smoke, "topk")
    assert "labels" not in parameters and "corruption_mask" not in parameters
    with pytest.raises(SystemExit):
        parse_args(["--arm", "LWC", "--epochs", "3"])
