"""Targeted E3-B0 first-order label-conditioned anchor utility tests."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    GLGC_CONTRASTIVE_TEMPERATURE,
)
from experiments.e3_sparse_relation_expansion.semantic_relation_expansion import (
    deterministic_anchor_label_shuffle,
)
from experiments.e3b_first_order_anchor_utility import (
    first_order_anchor_utility as fo,
)
from experiments.e3b_first_order_anchor_utility import (
    train_e3b_first_order_anchor_utility as e3b_train,
)
from model import Autoencoder


def _labels():
    return np.repeat(np.arange(7, dtype=np.int64), 2)


def _semantics(sample_num, requires_grad=False):
    torch.manual_seed(101 + sample_num)
    return F.normalize(
        torch.rand(sample_num, 6, 7), dim=-1
    ).requires_grad_(requires_grad)


def test_anchor_refresh_is_14x6x7_detached_and_restores_batchnorm():
    torch.manual_seed(103)
    encoders = [
        Autoencoder([4, 7], batchnorm=True, n_clusters=7)
        for _ in range(6)
    ]
    model = SimpleNamespace(autoencoders=encoders)
    for encoder in encoders:
        encoder.train()
    anchor_views = [torch.randn(14, 4) for _ in range(6)]
    matches = torch.stack([torch.eye(7) for _ in range(6)])
    before = [
        {name: value.clone() for name, value in encoder.named_buffers()}
        for encoder in encoders
    ]

    anchor_h, audit = fo.refresh_anchor_semantics(
        model, anchor_views, matches, torch.device("cpu")
    )

    assert anchor_h.shape == (14, 6, 7)
    assert not anchor_h.requires_grad and anchor_h.grad_fn is None
    assert audit["BN_running_state_unchanged_pass"] is True
    assert audit["BN_training_mode_restored_pass"] is True
    assert all(encoder.training for encoder in encoders)
    for encoder, snapshot in zip(encoders, before):
        assert all(
            torch.equal(dict(encoder.named_buffers())[name], value)
            for name, value in snapshot.items()
        )


def test_FO_graph_contains_14_anchors_plus_only_unlabeled_targets(monkeypatch):
    captured = {}

    def fake_graph(combined):
        captured["shape"] = tuple(combined.shape)
        size = combined.shape[0]
        return torch.ones(6, 6, size, size)

    monkeypatch.setattr(fo, "robust_inter_affinity", fake_graph)
    h_batch = _semantics(5, requires_grad=True)
    batch_ids = torch.tensor([2, 7, 11, 17, 23])
    labeled_ids = np.asarray([7] + list(range(100, 113)), dtype=np.int64)
    h_unlab, mask, subset_audit = fo.unlabeled_target_subset(
        h_batch, batch_ids, labeled_ids
    )
    _, _, graph_audit = fo.build_symmetric_anchor_graph(
        _semantics(14), h_unlab
    )

    assert mask.tolist() == [True, False, True, True, True]
    assert h_unlab.shape == (4, 6, 7)
    assert captured["shape"] == (18, 6, 7)
    assert graph_audit["combined_shape"] == [18, 6, 7]
    assert subset_audit["labeled_duplicate_in_graph_target"] is False


def test_symmetric_anchor_graph_forward_reverse_indices_are_exact(monkeypatch):
    size = 17
    graph = torch.arange(
        6 * 6 * size * size, dtype=torch.float64
    ).reshape(6, 6, size, size)

    def fake_graph(_):
        return graph

    monkeypatch.setattr(fo, "robust_inter_affinity", fake_graph)
    graph_all, graph_symmetric, audit = fo.build_symmetric_anchor_graph(
        _semantics(14).double(), _semantics(3).double()
    )
    for source_view in range(6):
        for target_view in range(6):
            for anchor_id in (0, 9, 13):
                for target_id in (0, 2):
                    expected = 0.0 if source_view == target_view else 0.5 * (
                        graph[source_view, target_view, anchor_id, 14 + target_id]
                        + graph[target_view, source_view, 14 + target_id, anchor_id]
                    )
                    assert graph_symmetric[
                        source_view, target_view, anchor_id, target_id
                    ].item() == expected
    assert torch.equal(graph_all, graph)
    assert audit["symmetric_extraction_exact_pass"] is True


def test_class_evidence_excludes_same_view_and_averages_two_anchors():
    graph = torch.arange(6 * 6 * 14 * 3, dtype=torch.float64).reshape(
        6, 6, 14, 3
    )
    labels = _labels()
    probability, evidence, audit = fo.anchor_induced_class_evidence(
        graph, labels
    )
    expected = torch.empty(3, 6, 7, dtype=torch.float64)
    for target_view in range(6):
        source_views = [value for value in range(6) if value != target_view]
        for class_id in range(7):
            anchors = torch.tensor([2 * class_id, 2 * class_id + 1])
            expected[:, target_view, class_id] = graph[
                source_views, target_view
            ][:, anchors].mean(dim=(0, 1))
    expected_probability = (expected + 1e-12) / (
        expected + 1e-12
    ).sum(dim=-1, keepdim=True)

    assert torch.allclose(evidence, expected)
    assert torch.allclose(probability, expected_probability)
    assert audit["source_views_per_target"] == 5
    assert audit["class_counts"] == [2] * 7


def test_first_order_factor_is_exact_direct_probability_gather():
    torch.manual_seed(107)
    probability = torch.rand(4, 6, 7)
    probability /= probability.sum(dim=-1, keepdim=True)
    labels = np.asarray([4, 6, 2, 3, 0, 4, 1, 5, 6, 3, 0, 2, 5, 1])

    factor, audit = fo.first_order_label_factor(probability, labels)

    assert factor.shape == (6, 6, 14, 4)
    for source_view in range(6):
        for target_view in range(6):
            expected = torch.zeros(14, 4) if source_view == target_view else torch.stack([
                probability[:, target_view, label]
                for label in labels
            ])
            assert torch.equal(factor[source_view, target_view], expected)
    assert audit["gather_exact_pass"] is True
    assert audit["no_pairwise_probability_dot_product"] is True


def test_source_has_no_cross_sample_probability_similarity_operator():
    source = inspect.getsource(fo)

    assert "cross_sample_semantic_similarity" not in source
    assert "P_i dot P_j" not in source
    assert "JS semantic similarity" not in source
    assert 'torch.gather(' in source


def test_information_utility_is_exact_G_times_first_order_factor():
    torch.manual_seed(109)
    graph = torch.rand(6, 6, 14, 3)
    factor = torch.rand(6, 6, 14, 3)
    diagonal = torch.eye(6, dtype=torch.bool)
    graph[diagonal] = 0.0
    factor[diagonal] = 0.0

    utility, _, _, audit = fo.build_first_order_targets(graph, factor)

    assert torch.equal(utility, graph * factor)
    assert audit["U_FO_finite_pass"] is True
    assert audit["reliability_used"] is False


def test_soft_target_normalizes_only_over_14_anchor_dimension():
    graph = torch.ones(6, 6, 14, 3)
    factor = torch.arange(1, 15, dtype=torch.float32).view(1, 1, 14, 1)
    factor = factor.expand_as(graph).clone()
    diagonal = torch.eye(6, dtype=torch.bool)
    graph[diagonal] = 0.0
    factor[diagonal] = 0.0

    _, target, valid_rows, audit = fo.build_first_order_targets(graph, factor)

    assert target.shape == (6, 6, 3, 14)
    assert torch.allclose(
        target.sum(-1)[valid_rows],
        torch.ones_like(target.sum(-1)[valid_rows]),
    )
    assert audit["T_valid_row_sum_one_pass"] is True
    assert not target.requires_grad


def test_student_temperature_is_exact_E1_contrastive_temperature():
    assert fo.STUDENT_TEMPERATURE == GLGC_CONTRASTIVE_TEMPERATURE == 0.5
    with pytest.raises(RuntimeError, match="0.5"):
        fo.first_order_anchor_relation_loss(
            _semantics(3, requires_grad=True),
            _semantics(14),
            torch.full((6, 6, 3, 14), 1.0 / 14.0),
            (~torch.eye(6, dtype=torch.bool))[:, :, None].expand(6, 6, 3),
            temperature=0.07,
        )


def test_loss_is_sum_of_15_pair_averages_with_30_directions():
    torch.manual_seed(113)
    h_unlab = _semantics(4, requires_grad=True)
    anchor_h = _semantics(14)
    target = torch.rand(6, 6, 4, 14)
    target /= target.sum(dim=-1, keepdim=True)
    valid = (~torch.eye(6, dtype=torch.bool))[:, :, None].expand(6, 6, 4)

    loss, audit = fo.first_order_anchor_relation_loss(
        h_unlab, anchor_h, target.detach(), valid
    )
    student = F.normalize(h_unlab, dim=-1)
    teacher = F.normalize(anchor_h, dim=-1)
    expected_pairs = []
    for left in range(6):
        for right in range(left + 1, 6):
            directions = []
            for source, target_view in ((left, right), (right, left)):
                log_q = F.log_softmax(
                    student[:, target_view] @ teacher[:, source].t() / 0.5,
                    dim=-1,
                )
                directions.append(
                    -(target[source, target_view] * log_q).sum(-1).mean()
                )
            expected_pairs.append(0.5 * (directions[0] + directions[1]))

    assert torch.allclose(loss, torch.stack(expected_pairs).sum())
    assert audit["unordered_pair_count"] == 15
    assert audit["direction_count"] == 30
    assert audit["aggregation"] == "sum_15_unordered_pairs"
    assert audit["direction_pairing"] == "mean_two_directions"


def test_teacher_target_detached_but_student_reaches_all_cluster_and_encoders():
    torch.manual_seed(127)
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
    anchor_h = _semantics(14)
    _, graph, _ = fo.build_symmetric_anchor_graph(anchor_h, h_sem)
    probability, _, _ = fo.anchor_induced_class_evidence(graph, _labels())
    factor, _ = fo.first_order_label_factor(probability, _labels())
    _, target, valid, _ = fo.build_first_order_targets(graph, factor)
    loss, audit = fo.first_order_anchor_relation_loss(
        h_sem, anchor_h, target, valid
    )
    loss.backward()

    assert audit["teacher_stop_gradient_pass"] is True
    assert audit["target_stop_gradient_pass"] is True
    assert not graph.requires_grad and not probability.requires_grad
    assert not factor.requires_grad and not target.requires_grad
    assert q_local.grad_fn is not None
    for encoder in encoders:
        assert encoder._cluster_layer.grad is not None
        assert torch.count_nonzero(encoder._cluster_layer.grad) > 0
        first_weight = next(encoder._encoder.parameters())
        assert first_weight.grad is not None
        assert torch.count_nonzero(first_weight.grad) > 0


def test_shuffle_control_preserves_counts_changes_10_and_is_not_class_permutation():
    labels = np.asarray([4, 6, 2, 3, 0, 4, 1, 5, 6, 3, 0, 2, 5, 1])
    shuffled, _, audit = deterministic_anchor_label_shuffle(labels, seed=20)

    assert np.array_equal(np.bincount(shuffled, minlength=7), np.full(7, 2))
    assert audit["changed_count"] >= 10
    assert audit["not_global_class_permutation_pass"] is True
    assert audit["labeled_ids_shuffled"] is False
    assert audit["anchor_representations_shuffled"] is False


def test_no_reliability_interface_or_argument_is_exposed():
    parser_args = e3b_train.parse_args([
        "--arm", "FO_ANCHOR", "--epochs", "2"
    ])
    core_signatures = " ".join(
        str(inspect.signature(function))
        for function in (
            fo.anchor_induced_class_evidence,
            fo.first_order_label_factor,
            fo.build_first_order_targets,
            fo.first_order_anchor_relation_loss,
        )
    )

    assert not hasattr(parser_args, "reliability_path")
    assert "reliability" not in core_signatures.lower()
    assert "load_frozen_reliability" not in inspect.getsource(e3b_train.main)


def test_full_labels_are_loaded_only_after_predictions_and_checkpoint_are_fixed():
    source = inspect.getsource(e3b_train.main)
    prediction_position = source.index("save_and_hash_predictions")
    checkpoint_position = source.index("save_final_models")
    label_position = source.index("load_labels_after_predictions")

    assert prediction_position < label_position
    assert checkpoint_position < label_position
    assert "load_labels_after_predictions" not in inspect.getsource(
        e3b_train.run_diagnostic_only
    )


def test_LWC_replay_directly_reuses_frozen_E1_arm():
    source = inspect.getsource(e3b_train.main)

    assert "e1_train.train_continuation" in source
    assert 'arm="LWC"' in source
    assert '"E3B_FO_branch_entered": False' in source
    assert "unused_lwc_api_placeholder()" in source


def test_objective_reuses_lambda1_without_new_coefficient():
    source = inspect.getsource(e3b_train.train_first_order_continuation)

    assert "native_total + LAMBDA1 * lwc_loss + LAMBDA1 * fo_loss" in source
    assert e3b_train.LAMBDA1 == 0.01
    assert "fo_coefficient" not in source


def test_diagnostic_parser_mode_is_exclusive_and_has_all_required_fields():
    args = e3b_train.parse_args(["--diagnostic-only"])
    source = inspect.getsource(e3b_train.run_diagnostic_only)
    required = (
        "P_true_vs_shuffle_L1_mean",
        "F_true_vs_shuffle_abs_mean",
        "F_true_vs_shuffle_abs_max",
        "U_true_vs_shuffle_relative_L1",
        "T_true_vs_shuffle_row_L1_mean",
        "T_true_vs_shuffle_row_L1_p50",
        "T_true_vs_shuffle_row_L1_max",
        "L_FO_true",
        "L_FO_shuffle",
        "grad_FO_true_norm",
        "grad_FO_shuffle_norm",
        "grad_LWC_norm",
        "grad_native_norm",
        "cos_grad_true_shuffle",
        "cos_grad_true_LWC",
        "cos_grad_true_native",
        "relative_true_shuffle_grad_difference",
        "effective_FO_over_LWC_gradient_ratio",
        "signal_retention_vs_E3A0_pass",
    )

    assert args.diagnostic_only is True
    assert args.arm is None and args.epochs is None
    assert all(field in source for field in required)
    with pytest.raises(SystemExit):
        e3b_train.parse_args([
            "--diagnostic-only", "--arm", "FO_ANCHOR", "--epochs", "2"
        ])
