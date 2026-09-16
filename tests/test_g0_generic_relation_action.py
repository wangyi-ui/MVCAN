import numpy as np
import torch

from experiments.cyclic_utility import c3_b0_relation_action_protocol as frozen
from experiments.generic_contract.generic_relation_action import (
    build_relation_balance_weights,
    build_relation_semantics,
    relation_semantic_loss,
)
from irv.b4_information_utility import tensor_sha256
from weak_quality import ndarray_sha256


def _small_semantics():
    N, K, S = 15, 3, 5
    y_gen = np.asarray(
        [[(sample + direction) % K for direction in range(S)]
         for sample in range(N)],
        dtype=np.int64,
    )
    labeled = np.arange(6, dtype=np.int64)
    targets = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    unlabeled = np.arange(6, N, dtype=np.int64)
    return build_relation_semantics(
        y_gen, labeled, targets, unlabeled,
        class_count=K, labels_per_class=2,
    )


def test_runtime_relation_shapes_and_balance():
    result = _small_semantics()
    assert result["PredRelation_true"].shape == (9, 6, 5)
    assert result["relation_balance_weights_true"].shape == (9, 6, 5)
    for direction in range(5):
        predicted = result["PredRelation_true"][:, :, direction]
        weights = result["relation_balance_weights_true"][:, :, direction]
        assert np.isclose(weights[predicted].sum(), 0.5, rtol=0, atol=1e-15)
        assert np.isclose(weights[~predicted].sum(), 0.5, rtol=0, atol=1e-15)


def test_relation_loss_detach_and_five_view_mean():
    generator = torch.Generator().manual_seed(20)
    queries = []
    anchors = []
    for _ in range(5):
        query_logits = torch.rand((4, 3), generator=generator)
        query = (query_logits / query_logits.sum(1, keepdim=True)).requires_grad_()
        anchor_logits = torch.rand((6, 3), generator=generator)
        anchor = (anchor_logits / anchor_logits.sum(1, keepdim=True)).requires_grad_()
        queries.append(query)
        anchors.append(anchor)
    target = torch.tensor(
        np.indices((4, 6, 5)).sum(axis=0) % 2, dtype=torch.float32
    )
    utility = torch.full((4, 5), 0.5, requires_grad=True)
    balance = torch.ones((4, 6, 5), requires_grad=True)
    loss, audit = relation_semantic_loss(
        queries, anchors, target, utility, balance
    )
    loss.backward()
    assert audit["view_count"] == 5
    assert loss.detach().item() == torch.tensor(audit["view_losses"]).mean().item()
    assert all(query.grad is not None for query in queries)
    assert all(anchor.grad is None for anchor in anchors)
    assert utility.grad is None
    assert balance.grad is None


def test_balance_rejects_empty_group():
    predicted = np.ones((3, 2, 4), dtype=np.bool_)
    try:
        build_relation_balance_weights(predicted)
    except RuntimeError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty relation group must fail closed")


def test_caltech_frozen_relation_exact_parity():
    with np.load(
        "outputs/cyclic_utility/c3_a0_utility_conditioned_action_granularity_seed20/"
        "c3_a0_action_pre_gt.npz",
        allow_pickle=False,
    ) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in (
            "y_gen", "labeled_ids", "labeled_targets", "unlabeled_ids",
            "PredRelation_true", "relation_balance_weights_true",
        )}
    generic = build_relation_semantics(
        arrays["y_gen"],
        arrays["labeled_ids"],
        arrays["labeled_targets"],
        arrays["unlabeled_ids"],
        class_count=7,
        labels_per_class=2,
    )
    prediction = generic["PredRelation_true"]
    balance = generic["relation_balance_weights_true"]
    assert np.array_equal(prediction, arrays["PredRelation_true"])
    assert np.array_equal(balance, arrays["relation_balance_weights_true"])
    assert ndarray_sha256(prediction) == (
        "a61460618339930b577c0b53e9fe6f15f4e40650e3522fdab34fa9a43d7bfca5"
    )
    assert tensor_sha256(prediction) == (
        "4e72d2cdf118391caa8b586f1f8220d5cbf52a2e6bdf5bb5afa6a8a43ec19d5a"
    )
    assert ndarray_sha256(balance) == (
        "1d7f1dcc65f49b37fdd6d0d8947456cdf828dff83936e10ca507cf1093d300aa"
    )
    assert tensor_sha256(balance) == (
        "af9402d1adbe4cc7313277064360a68ff2e7b6a2e2ff88f57f3061b07df227f9"
    )


def test_caltech_frozen_relation_loss_parity():
    generator = torch.Generator().manual_seed(2042)
    queries, anchors = [], []
    for _ in range(6):
        q = torch.rand((5, 7), generator=generator)
        a = torch.rand((14, 7), generator=generator)
        queries.append(q / q.sum(1, keepdim=True))
        anchors.append((a / a.sum(1, keepdim=True)).detach())
    target = torch.tensor(
        np.indices((5, 14, 20)).sum(axis=0) % 2, dtype=torch.float32
    )
    utility = torch.rand((5, 20), generator=generator)
    balance = torch.rand((5, 14, 20), generator=generator) + 0.1
    generic_loss, _ = relation_semantic_loss(
        queries, anchors, target, utility, balance
    )
    frozen_loss, _ = frozen.relation_semantic_loss(
        queries, anchors, target, utility, balance, "TRUE_U"
    )
    assert torch.equal(generic_loss, frozen_loss)
