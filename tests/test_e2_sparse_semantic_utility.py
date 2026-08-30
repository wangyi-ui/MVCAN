"""Targeted tests for E2 sparse semantic interaction utility."""

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    robust_inter_affinity,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    whole_row_reliability_shuffle,
)
from experiments.e2_sparse_semantic_utility.semantic_interaction_utility import (
    build_sparse_class_prototypes,
)
from experiments.e2_sparse_semantic_utility.semantic_interaction_utility import (
    normalized_js_similarity,
)
from experiments.e2_sparse_semantic_utility.semantic_interaction_utility import (
    semantic_interaction_weights,
)
from experiments.e2_sparse_semantic_utility.semantic_interaction_utility import (
    semantic_pairwise_cooperation_loss,
)
from experiments.e2_sparse_semantic_utility.semantic_interaction_utility import (
    sparse_semantic_probabilities,
)
from experiments.e2_sparse_semantic_utility.train_e2_sparse_semantic_utility import (
    EXPECTED_LABELED_IDS,
)
from experiments.e2_sparse_semantic_utility.train_e2_sparse_semantic_utility import (
    EXPECTED_LABELED_IDS_SHA256,
)
from experiments.e2_sparse_semantic_utility.train_e2_sparse_semantic_utility import (
    EXPECTED_LABEL_SPLIT_SHA256,
)
from experiments.e2_sparse_semantic_utility.train_e2_sparse_semantic_utility import (
    REPOSITORY_ROOT,
)
from experiments.e2_sparse_semantic_utility.train_e2_sparse_semantic_utility import (
    audit_lwc_replay_identity,
)
from experiments.e2_sparse_semantic_utility.train_e2_sparse_semantic_utility import (
    load_sparse_training_labels,
)
from model import Autoencoder


def _balanced_targets():
    return np.repeat(np.arange(7, dtype=np.int64), 2)


def _normalized_semantics(sample_num=14, view_num=2, semantic_dim=3):
    torch.manual_seed(31)
    return F.normalize(
        torch.randn(sample_num, view_num, semantic_dim), dim=-1
    )


def test_frozen_B7_split_provides_only_exact_14_training_labels():
    labeled_ids, targets, unlabeled_ids, audit = load_sparse_training_labels()

    assert np.array_equal(labeled_ids, EXPECTED_LABELED_IDS)
    assert labeled_ids.shape == (14,)
    assert targets.shape == (14,)
    assert unlabeled_ids.shape == (1386,)
    assert np.array_equal(np.bincount(targets, minlength=7), np.full(7, 2))
    assert audit["labeled_ids_sha256"] == EXPECTED_LABELED_IDS_SHA256
    assert audit["label_split_sha256"] == EXPECTED_LABEL_SPLIT_SHA256
    assert audit["full_label_file_opened_before_training"] is False
    assert audit["split_regenerated"] is False


def test_unweighted_prototype_is_normalized_sum_of_two_anchors():
    h_sem = _normalized_semantics()
    ids = np.arange(14, dtype=np.int64)
    targets = _balanced_targets()

    prototypes, audit = build_sparse_class_prototypes(
        h_sem, ids, targets, class_num=7
    )

    expected = torch.empty_like(prototypes)
    for view_id in range(2):
        for class_id in range(7):
            members = torch.tensor([2 * class_id, 2 * class_id + 1])
            expected[view_id, class_id] = F.normalize(
                h_sem[members, view_id].sum(dim=0), dim=0
            )
    assert torch.allclose(prototypes, expected)
    assert prototypes.requires_grad is False
    assert audit["weighted_by_reliability"] is False
    assert audit["anchor_count_two_every_view_class_pass"] is True
    assert audit["prototype_unit_norm_pass"] is True
    assert audit["zero_fill_used"] is False


def test_R_weighted_prototype_matches_frozen_equation_without_direct_pair_R():
    h_sem = _normalized_semantics()
    ids = np.arange(14, dtype=np.int64)
    targets = _balanced_targets()
    reliability = torch.linspace(0.1, 1.0, 28).reshape(14, 2)
    reliability.requires_grad_(True)

    prototypes, audit = build_sparse_class_prototypes(
        h_sem.requires_grad_(True),
        ids,
        targets,
        class_num=7,
        reliability=reliability,
    )

    members = torch.tensor([0, 1])
    weights = reliability.detach()[members, 0]
    source = (
        (weights[:, None] * h_sem.detach()[members, 0]).sum(dim=0)
        / (weights.sum() + 1e-12)
    )
    assert torch.allclose(prototypes[0, 0], F.normalize(source, dim=0))
    assert prototypes.requires_grad is False
    assert reliability.grad is None
    assert h_sem.grad is None
    assert audit["weighted_by_reliability"] is True
    assert audit["denominator_positive_every_view_class_pass"] is True


def test_shuffled_R_changes_only_complete_row_correspondence_for_prototypes():
    h_sem = _normalized_semantics()
    ids = np.arange(14, dtype=np.int64)
    targets = _balanced_targets()
    reliability = np.arange(84, dtype=np.float64).reshape(14, 6) / 83.0
    reliability += 0.01

    shuffled, permutation, shuffle_audit = whole_row_reliability_shuffle(
        reliability, seed=20
    )
    prototypes, prototype_audit = build_sparse_class_prototypes(
        h_sem,
        ids,
        targets,
        class_num=7,
        reliability=shuffled[:, :2],
    )

    assert np.array_equal(shuffled, reliability[permutation])
    assert shuffle_audit["R_multiset_audit_pass"] is True
    assert shuffle_audit["per_view_distribution_audit_pass"] is True
    assert prototypes.shape == (2, 7, 3)
    assert prototype_audit["prototype_unit_norm_pass"] is True


def test_zero_reliability_denominator_is_hard_failure_not_zero_fill():
    h_sem = _normalized_semantics()
    reliability = torch.ones(14, 2)
    reliability[0:2, 0] = 0.0

    with pytest.raises(RuntimeError, match="denominator"):
        build_sparse_class_prototypes(
            h_sem,
            np.arange(14, dtype=np.int64),
            _balanced_targets(),
            class_num=7,
            reliability=reliability,
        )


def test_semantic_probability_is_fixed_affinity_normalized_and_detached():
    h_sem = _normalized_semantics(sample_num=5, semantic_dim=7)
    prototypes = F.normalize(torch.randn(2, 7, 7), dim=-1)
    h_sem.requires_grad_(True)
    prototypes.requires_grad_(True)

    semantic_prob, audit = sparse_semantic_probabilities(h_sem, prototypes)
    cosine = torch.einsum(
        "bvd,vcd->bvc", h_sem.detach(), prototypes.detach()
    ).clamp(-1.0, 1.0)
    affinity = (1.0 + cosine) / 2.0 + 1e-12
    expected = affinity / affinity.sum(dim=-1, keepdim=True)

    assert semantic_prob.shape == (5, 2, 7)
    assert torch.allclose(semantic_prob, expected)
    assert torch.all(semantic_prob >= 0.0)
    assert torch.allclose(semantic_prob.sum(-1), torch.ones(5, 2))
    assert semantic_prob.requires_grad is False
    assert audit["temperature_used"] is False


def test_normalized_JS_similarity_has_identity_and_disjoint_boundaries():
    probabilities = torch.tensor([
        [[0.5, 0.5], [0.5, 0.5]],
        [[1.0, 0.0], [0.0, 1.0]],
    ])

    similarity = normalized_js_similarity(probabilities, 0, 1)

    assert similarity.shape == (2,)
    assert similarity[0].item() == pytest.approx(1.0)
    assert similarity[1].item() == pytest.approx(0.0, abs=1e-6)
    assert torch.all((similarity >= 0.0) & (similarity <= 1.0))
    assert similarity.requires_grad is False


def test_U_sem_is_exactly_G_times_S_and_API_cannot_accept_R():
    batch_size = 3
    graph = torch.zeros(2, 2, batch_size, batch_size)
    graph[0, 1].diagonal().copy_(torch.tensor([0.2, 0.4, 0.6]))
    graph[1, 0].diagonal().copy_(torch.tensor([0.4, 0.6, 0.8]))
    probabilities = torch.tensor([
        [[0.7, 0.3], [0.7, 0.3]],
        [[0.9, 0.1], [0.1, 0.9]],
        [[0.6, 0.4], [0.5, 0.5]],
    ])

    utility, graph_evidence, similarity = semantic_interaction_weights(
        graph, probabilities, 0, 1
    )

    assert torch.equal(utility, graph_evidence * similarity)
    assert utility.requires_grad is False
    parameters = inspect.signature(semantic_interaction_weights).parameters
    assert set(parameters) == {"graph_inter", "semantic_prob", "left", "right"}


def test_pair_loss_keeps_encoder_gradient_while_semantic_evidence_is_frozen():
    torch.manual_seed(37)
    encoders = [
        Autoencoder([4, 3], batchnorm=False, n_clusters=3)
        for _ in range(2)
    ]
    q_views = []
    for encoder in encoders:
        latent = encoder.encoder(torch.randn(5, 4))
        q_views.append(encoder.clustering(latent))
    q_local = torch.stack(q_views, dim=1)
    h_sem = F.normalize(q_local, dim=-1)
    graph = robust_inter_affinity(h_sem.detach())
    prototypes = F.normalize(torch.randn(2, 3, 3), dim=-1)
    semantic_prob, _ = sparse_semantic_probabilities(
        h_sem.detach(), prototypes
    )

    loss, diagnostics = semantic_pairwise_cooperation_loss(
        h_sem, graph, semantic_prob
    )
    loss.backward()

    assert graph.requires_grad is False
    assert semantic_prob.requires_grad is False
    assert diagnostics[0]["U_sem_stop_gradient_pass"] is True
    for encoder in encoders:
        assert torch.count_nonzero(encoder._cluster_layer.grad) > 0
        assert torch.count_nonzero(next(encoder._encoder.parameters()).grad) > 0


def test_LWC_replay_audit_accepts_the_frozen_E1_smoke_record_exactly():
    reference_dir = (
        Path(REPOSITORY_ROOT)
        / "outputs/e1_pairwise_utility/lwc_2ep_seed20"
    )
    import json

    with open(reference_dir / "e1_audit.json", "r", encoding="utf-8") as handle:
        reference = json.load(handle)
    audit = audit_lwc_replay_identity(
        reference["initial_model_hash"],
        reference["final_model_hash"],
        reference["prediction_audit"],
        reference["metrics"],
        reference_dir,
    )

    assert audit["LWC_REPLAY_EXACT_MATCH_PASS"] is True
    assert all(audit["checks"].values())
