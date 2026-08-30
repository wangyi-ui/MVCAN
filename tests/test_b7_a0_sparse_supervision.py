"""Synthetic isolation tests for B7-A0 semantic supervision."""

import copy
import inspect

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ClusteringTest import acc as mvcan_acc
from ClusteringTest import ari as mvcan_ari
from ClusteringTest import nmi as mvcan_nmi
from experiments.b7_sparse_supervision import b7_sparse_label_protocol as protocol
from experiments.b7_sparse_supervision import (
    train_b7_a0_sparse_supervision_admission as train,
)
from irv.b3_audit import hash_backbone
from irv.b3_audit import hash_state_dict


def _toy_semantic_inputs(sample_num=21):
    generator = torch.Generator().manual_seed(7)
    z_stack = torch.randn(sample_num, 6, 10, generator=generator).detach()
    unsupervised = np.zeros((sample_num, 6), dtype=bool)
    unsupervised[:, :3] = True
    labeled_ids = np.arange(7, dtype=np.int64)
    labeled_targets = np.arange(7, dtype=np.int64)
    supervised = np.zeros((sample_num, 6), dtype=bool)
    supervised[labeled_ids] = True
    return z_stack, unsupervised, supervised, labeled_ids, labeled_targets


@pytest.fixture(scope="module")
def frozen_seed20_inputs():
    return train.load_frozen_b7_inputs()


def test_real_frozen_seed20_z_shape_and_detach(frozen_seed20_inputs):
    z_stack = frozen_seed20_inputs["z_stack"]

    assert z_stack.shape == (1400, 6, 10)
    assert z_stack.requires_grad is False
    assert torch.isfinite(z_stack).all()


def test_real_frozen_seed20_u_is_read_only_and_hash_registered(
    frozen_seed20_inputs,
):
    utility = frozen_seed20_inputs["utility"]

    assert utility.shape == (1400, 6)
    assert utility.flags.writeable is False
    assert frozen_seed20_inputs["utility_sha256"] == (
        "3458e099f8c6e7317e3edbcef15ff629f368735e115fa6aea30e4516be8ac2cc"
    )


def test_real_frozen_seed20_backbone_requires_no_gradient(frozen_seed20_inputs):
    assert frozen_seed20_inputs["backbone_requires_grad_pass"] is True


def test_carrier_produces_required_h_and_logits_shapes():
    model = train.B7SemanticCarrier()
    z_stack = torch.randn(19, 6, 10)

    h_stack, logits = model(z_stack)

    assert h_stack.shape == (19, 6, 10)
    assert logits.shape == (19, 6, 7)
    assert torch.allclose(
        torch.linalg.vector_norm(h_stack, dim=-1),
        torch.ones(19, 6),
        atol=1e-6,
    )


def test_hard_admission_supervised_loss_formula_is_exact():
    logits = torch.tensor([
        [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]],
        [[0.2, 0.1, 0.0], [0.3, 0.1, -0.1]],
        [[-1.0, 0.0, 2.0], [2.0, 0.0, -1.0]],
    ])
    admission = np.array([
        [True, False],
        [False, False],
        [True, True],
    ])
    labeled_ids = np.array([0, 2], dtype=np.int64)
    labeled_targets = np.array([0, 2], dtype=np.int64)

    actual, count = train.sparse_supervised_cross_entropy(
        logits, admission, labeled_ids, labeled_targets
    )
    expected = torch.stack((
        F.cross_entropy(logits[0, 0][None, :], torch.tensor([0])),
        F.cross_entropy(logits[2, 0][None, :], torch.tensor([2])),
        F.cross_entropy(logits[2, 1][None, :], torch.tensor([2])),
    )).mean()

    assert count == 3
    assert actual.item() == pytest.approx(expected.item(), rel=1e-7)


def test_unlabeled_label_permutation_cannot_change_training_loss_inputs():
    z_stack, unsup, supervised, labeled_ids, _ = _toy_semantic_inputs()
    full_targets = np.arange(21, dtype=np.int64) % 7
    sparse_targets = full_targets[labeled_ids].copy()
    permuted = full_targets.copy()
    unlabeled_ids = np.setdiff1d(np.arange(21), labeled_ids)
    permuted[unlabeled_ids] = permuted[unlabeled_ids][::-1]
    model = train.B7SemanticCarrier()
    _, logits = model(z_stack)

    first, _ = train.sparse_supervised_cross_entropy(
        logits, supervised, labeled_ids, sparse_targets
    )
    second, _ = train.sparse_supervised_cross_entropy(
        logits, supervised, labeled_ids, permuted[labeled_ids]
    )

    assert np.array_equal(sparse_targets, permuted[labeled_ids])
    assert torch.equal(first, second)


def test_training_apis_cannot_receive_full_or_unlabeled_labels():
    for function in (train.compute_b7_losses, train.train_one_arm):
        parameters = inspect.signature(function).parameters
        assert "evaluation_targets" not in parameters
        assert "full_labels" not in parameters
        assert "unlabeled_labels" not in parameters
        assert "corruption_mask" not in parameters
        assert "utility" not in parameters
        assert "labeled_targets" in parameters


def test_backbone_parameters_are_exactly_unchanged_after_optimizer_step():
    torch.manual_seed(5)
    backbones = [nn.Linear(4, 10) for _ in range(6)]
    for module in backbones:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    inputs = [torch.randn(21, 4) for _ in range(6)]
    with torch.no_grad():
        z_stack = torch.stack(
            [F.normalize(module(value), dim=1) for module, value in zip(backbones, inputs)],
            dim=1,
        ).detach()
    before = hash_backbone(backbones)
    _, unsup, supervised, labeled_ids, labeled_targets = _toy_semantic_inputs()
    model = train.B7SemanticCarrier()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train.LEARNING_RATE,
        weight_decay=train.WEIGHT_DECAY,
    )

    _, finite = train.semantic_optimizer_step(
        model,
        optimizer,
        z_stack,
        unsup,
        supervised,
        labeled_ids,
        labeled_targets,
    )
    after = hash_backbone(backbones)

    assert finite is True
    assert before == after
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for module in backbones
        for parameter in module.parameters()
    )


def test_five_arms_load_one_exact_semantic_initial_state():
    initial_state, expected_hash = train.initial_semantic_state(20)
    hashes = []
    for _ in protocol.ARMS:
        model = train.B7SemanticCarrier()
        model.load_state_dict(copy.deepcopy(initial_state), strict=True)
        hashes.append(hash_state_dict(model.state_dict()))

    assert hashes == [expected_hash] * 5


def test_all_unlabeled_and_labeled_metrics_use_the_correct_indices():
    targets = np.repeat(np.arange(7, dtype=np.int64), 4)
    prediction = targets.copy()
    labeled_ids = np.arange(0, 28, 4, dtype=np.int64)
    unlabeled_ids = np.setdiff1d(np.arange(28), labeled_ids)
    prediction[labeled_ids] = np.roll(prediction[labeled_ids], 1)

    metrics = train.cluster_metrics_by_partition(
        targets, prediction, labeled_ids, unlabeled_ids
    )

    assert metrics["ACC_all"] == pytest.approx(mvcan_acc(targets, prediction))
    assert metrics["NMI_all"] == pytest.approx(mvcan_nmi(targets, prediction))
    assert metrics["ARI_all"] == pytest.approx(mvcan_ari(targets, prediction))
    assert metrics["ACC_unlabeled"] == pytest.approx(
        mvcan_acc(targets[unlabeled_ids], prediction[unlabeled_ids])
    )
    assert metrics["NMI_unlabeled"] == pytest.approx(
        mvcan_nmi(targets[unlabeled_ids], prediction[unlabeled_ids])
    )
    assert metrics["ARI_unlabeled"] == pytest.approx(
        mvcan_ari(targets[unlabeled_ids], prediction[unlabeled_ids])
    )
    assert metrics["ACC_labeled"] == pytest.approx(
        mvcan_acc(targets[labeled_ids], prediction[labeled_ids])
    )


def test_all_arms_share_one_unsupervised_admission_policy():
    utility = np.random.RandomState(4).uniform(size=(70, 6))
    corruption = np.zeros((70, 6), dtype=bool)
    corruption[:, :3] = True
    labeled_ids = protocol.make_class_balanced_split(
        np.repeat(np.arange(7), 10), 2, 20
    )["labeled_sample_ids"]

    unsupervised, supervised, mapping = train.build_all_arm_admissions(
        utility, corruption, labeled_ids
    )

    assert unsupervised.shape == (70, 6)
    assert np.all(unsupervised.sum(axis=1) == 3)
    assert set(supervised) == set(protocol.ARMS)
    assert mapping.shape == (70, 6)


def test_utility_cannot_enter_feature_or_loss_api():
    loss_parameters = inspect.signature(train.compute_b7_losses).parameters
    readout_parameters = inspect.signature(train.shared_semantic_readout).parameters

    assert "utility" not in loss_parameters
    assert "utility" not in readout_parameters
    assert set(readout_parameters) == {"h_stack", "admission"}
