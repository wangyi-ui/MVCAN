"""Targeted E3-A0 sparse semantic relation expansion tests."""

import inspect

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.e2_sparse_semantic_utility.train_e2_sparse_semantic_utility import (
    EXPECTED_LABELED_IDS,
)
from experiments.e3_sparse_relation_expansion.semantic_relation_expansion import (
    RELATION_TEMPERATURE,
)
from experiments.e3_sparse_relation_expansion.semantic_relation_expansion import (
    anchor_induced_class_evidence,
)
from experiments.e3_sparse_relation_expansion.semantic_relation_expansion import (
    build_anchor_batch_graph,
)
from experiments.e3_sparse_relation_expansion.semantic_relation_expansion import (
    build_relation_targets,
)
from experiments.e3_sparse_relation_expansion.semantic_relation_expansion import (
    cross_sample_relation_loss,
)
from experiments.e3_sparse_relation_expansion.semantic_relation_expansion import (
    cross_sample_semantic_similarity,
)
from experiments.e3_sparse_relation_expansion.semantic_relation_expansion import (
    deterministic_anchor_label_shuffle,
)
from experiments.e3_sparse_relation_expansion import (
    train_e3_sparse_relation_expansion as e3_train,
)
from model import Autoencoder


def _anchor_labels():
    return np.repeat(np.arange(7, dtype=np.int64), 2)


def _semantic_tensor(sample_num, requires_grad=False):
    torch.manual_seed(41 + sample_num)
    tensor = F.normalize(torch.randn(sample_num, 6, 7), dim=-1)
    return tensor.requires_grad_(requires_grad)


def test_anchor_batch_graph_shapes_and_stop_gradient_are_exact():
    anchor_h = _semantic_tensor(14, requires_grad=True)
    h_batch = _semantic_tensor(5, requires_grad=True)

    graph_all, graph_anchor_batch, graph_batch_batch = (
        build_anchor_batch_graph(anchor_h, h_batch)
    )

    assert graph_all.shape == (6, 6, 19, 19)
    assert graph_anchor_batch.shape == (6, 6, 14, 5)
    assert graph_batch_batch.shape == (6, 6, 5, 5)
    assert torch.equal(graph_anchor_batch, graph_all[:, :, :14, 14:])
    assert torch.equal(graph_batch_batch, graph_all[:, :, 14:, 14:])
    assert not graph_all.requires_grad
    assert not graph_anchor_batch.requires_grad
    assert not graph_batch_batch.requires_grad


def test_anchor_class_evidence_is_mean_over_source_views_and_two_anchors():
    torch.manual_seed(43)
    graph = torch.rand(6, 6, 14, 4)
    labels = _anchor_labels()

    anchor_prob, evidence, audit = anchor_induced_class_evidence(
        graph, labels
    )

    expected = torch.empty(4, 6, 7)
    for target_view in range(6):
        for class_id in range(7):
            class_mask = torch.from_numpy(labels == class_id)
            expected[:, target_view, class_id] = graph[
                :, target_view, class_mask, :
            ].mean(dim=(0, 1))
    expected_prob = (expected + 1e-12) / (
        expected + 1e-12
    ).sum(dim=-1, keepdim=True)

    assert torch.allclose(evidence, expected)
    assert torch.allclose(anchor_prob, expected_prob)
    assert anchor_prob.shape == (4, 6, 7)
    assert torch.allclose(anchor_prob.sum(-1), torch.ones(4, 6))
    assert anchor_prob.requires_grad is False
    assert audit["class_counts"] == [2, 2, 2, 2, 2, 2, 2]
    assert audit["argmax_used"] is False
    assert audit["temperature_used"] is False


def test_cross_sample_S_is_probability_dot_product_for_cross_views():
    torch.manual_seed(47)
    probabilities = torch.rand(3, 6, 7)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

    similarity = cross_sample_semantic_similarity(probabilities)

    expected_01 = probabilities[:, 0] @ probabilities[:, 1].t()
    assert similarity.shape == (6, 6, 3, 3)
    assert torch.allclose(similarity[0, 1], expected_01)
    assert torch.count_nonzero(similarity[0, 0]) == 0
    assert torch.all((similarity >= 0.0) & (similarity <= 1.0))
    assert similarity.requires_grad is False


def test_U_and_T_are_unlabeled_cross_sample_only_without_threshold_or_topk():
    batch_size = 4
    graph = torch.ones(6, 6, batch_size, batch_size)
    probabilities = torch.full((batch_size, 6, 7), 1.0 / 7.0)
    similarity = cross_sample_semantic_similarity(probabilities)
    unlabeled = torch.tensor([False, True, True, True])

    utility, target, valid_rows, relation_mask, audit = build_relation_targets(
        graph, similarity, unlabeled
    )

    assert utility.shape == (6, 6, batch_size, batch_size)
    assert target.shape == utility.shape
    assert torch.count_nonzero(utility.diagonal(dim1=-2, dim2=-1)) == 0
    assert torch.count_nonzero(utility[:, :, 0, :]) == 0
    assert torch.count_nonzero(utility[:, :, :, 0]) == 0
    assert torch.allclose(
        target.sum(-1)[valid_rows],
        torch.ones_like(target.sum(-1)[valid_rows]),
    )
    assert not bool(relation_mask[:, :, 0, :].any().item())
    assert audit["valid_row_count"] > 0
    assert audit["valid_off_diagonal_relation_count"] > 0
    assert audit["threshold_used"] is False
    assert audit["topk_used"] is False
    assert audit["reliability_used"] is False
    assert not utility.requires_grad and not target.requires_grad


def test_relation_loss_has_nonzero_cluster_and_encoder_gradient_only_on_student():
    torch.manual_seed(53)
    encoders = [
        Autoencoder([4, 7], batchnorm=False, n_clusters=7)
        for _ in range(6)
    ]
    q_views = []
    for encoder in encoders:
        latent = encoder.encoder(torch.randn(5, 4))
        q_views.append(encoder.clustering(latent))
    q_local = torch.stack(q_views, dim=1)
    h_sem = F.normalize(q_local, dim=-1)
    anchor_h = _semantic_tensor(14)
    _, graph_anchor_batch, graph_batch_batch = build_anchor_batch_graph(
        anchor_h, h_sem
    )
    anchor_prob, _, _ = anchor_induced_class_evidence(
        graph_anchor_batch, _anchor_labels()
    )
    semantic_similarity = cross_sample_semantic_similarity(anchor_prob)
    unlabeled = torch.ones(5, dtype=torch.bool)
    _, relation_target, valid_rows, _, _ = build_relation_targets(
        graph_batch_batch, semantic_similarity, unlabeled
    )

    loss, audit = cross_sample_relation_loss(
        h_sem, relation_target, valid_rows, unlabeled
    )
    loss.backward()

    assert loss.item() > 0.0
    assert audit["temperature"] == RELATION_TEMPERATURE == 0.07
    assert audit["target_stop_gradient_pass"] is True
    assert relation_target.requires_grad is False
    for encoder in encoders:
        assert encoder._cluster_layer.grad is not None
        assert torch.count_nonzero(encoder._cluster_layer.grad) > 0
        first_weight = next(encoder._encoder.parameters())
        assert first_weight.grad is not None
        assert torch.count_nonzero(first_weight.grad) > 0


def test_shuffled_label_control_preserves_counts_and_breaks_global_permutation():
    true_labels = np.asarray(
        [4, 6, 2, 3, 0, 4, 1, 5, 6, 3, 0, 2, 5, 1],
        dtype=np.int64,
    )

    first, first_permutation, audit = deterministic_anchor_label_shuffle(
        true_labels, seed=20
    )
    second, second_permutation, second_audit = (
        deterministic_anchor_label_shuffle(true_labels, seed=20)
    )

    assert np.array_equal(first, second)
    assert np.array_equal(first_permutation, second_permutation)
    assert np.array_equal(np.bincount(first, minlength=7), np.full(7, 2))
    assert audit["changed_count"] >= 10
    assert audit["class_counts_preserved_pass"] is True
    assert audit["not_global_class_permutation_pass"] is True
    assert audit["permutation_sha256"] == second_audit["permutation_sha256"]
    assert audit["labeled_ids_shuffled"] is False
    assert audit["anchor_representations_shuffled"] is False


def test_training_boundary_has_exact_14_labels_and_no_reliability_interface():
    labeled_ids, labels, unlabeled_ids, audit = (
        e3_train.load_sparse_training_labels()
    )
    parser_args = e3_train.parse_args(["--arm", "SEM_REL", "--epochs", "2"])
    main_source = inspect.getsource(e3_train.main)

    assert np.array_equal(labeled_ids, EXPECTED_LABELED_IDS)
    assert labels.shape == (14,)
    assert unlabeled_ids.shape == (1386,)
    assert audit["full_label_file_opened_before_training"] is False
    assert not hasattr(parser_args, "reliability_path")
    assert "load_frozen_reliability" not in main_source
    assert "D2" not in inspect.getsource(e3_train.parse_args)


def test_LWC_replay_is_direct_frozen_E1_call_and_has_no_relation_branch():
    source = inspect.getsource(e3_train.main)

    assert "e1_train.train_continuation" in source
    assert 'arm="LWC"' in source
    assert '"E3_relation_branch_entered": False' in source
    assert "unused_lwc_api_placeholder()" in source


def test_total_objective_reuses_only_native_lambda1_for_both_branches():
    source = inspect.getsource(e3_train.train_relation_continuation)

    assert "native_total + LAMBDA1 * lwc_loss + LAMBDA1 * relation_loss" in source
    assert e3_train.LAMBDA1 == 0.01
    assert "relation_coefficient" not in source
    assert "temperature_override" not in source
    assert "topk" not in inspect.getsource(e3_train.parse_args).lower()
