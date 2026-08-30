"""End-to-end E1 pair-branch gradient audit on native MVCAN modules."""

import torch

from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    align_semantic_probabilities,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    pairwise_semantic_cooperation_loss,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    robust_inter_affinity,
)
from model import Autoencoder


def test_pair_branch_reaches_mvcam_cluster_layers_and_encoders():
    torch.manual_seed(8)
    encoders = [
        Autoencoder([3, 4], batchnorm=False, n_clusters=3)
        for _ in range(2)
    ]
    inputs = [torch.randn(5, 3), torch.randn(5, 3)]
    q_views = []
    for encoder, values in zip(encoders, inputs):
        latent = encoder.encoder(values)
        q_views.append(encoder.clustering(latent))
    # q_local: [B=5,V=2,K=3], differentiable.
    q_local = torch.stack(q_views, dim=1)
    # M: [V=2,K=3,K=3], constant/no grad.
    matches = torch.stack((torch.eye(3), torch.eye(3)))
    _, h_sem = align_semantic_probabilities(q_local, matches)
    # G_inter: [V,V,B,B], constructed from h_sem.detach().
    graph = robust_inter_affinity(h_sem.detach())
    # R_batch: [B,V], frozen/no grad.
    reliability = torch.ones(5, 2)

    loss, _ = pairwise_semantic_cooperation_loss(
        h_sem, graph, reliability, "R_LWC"
    )
    loss.backward()

    assert graph.requires_grad is False and graph.grad_fn is None
    for encoder in encoders:
        assert encoder._cluster_layer.grad is not None
        assert torch.count_nonzero(encoder._cluster_layer.grad) > 0
        first_weight = next(encoder._encoder.parameters())
        assert first_weight.grad is not None
        assert torch.count_nonzero(first_weight.grad) > 0
