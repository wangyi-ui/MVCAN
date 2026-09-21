import numpy as np
import pytest
import torch

from release_core.backbone.autoencoder import Autoencoder
from release_core.training import build_decoupled_optimizers


class TinyBackbone:
    def __init__(self, view_count=2, input_dim=4, latent_dim=3, clusters=3):
        self.n_clusters = clusters
        self.autoencoders = [
            Autoencoder(
                [input_dim, 5, latent_dim],
                activation="relu",
                batchnorm=False,
                n_clusters=clusters,
            )
            for _ in range(view_count)
        ]


def make_case(view_count=2, *, noncontiguous=False, labeled_count=1):
    torch.manual_seed(17 + view_count)
    N, D, K, S = 8, 4, 3, 2
    model = TinyBackbone(view_count, D, 3, K)
    views = tuple(torch.randn(N, D) for _ in range(view_count))
    sample_ids = np.arange(N, dtype=np.int64)
    if noncontiguous:
        sample_ids = np.asarray([31, 7, 44, 2, 19, 61, 5, 28], dtype=np.int64)
    labeled = sample_ids[:labeled_count].copy()
    unlabeled = sample_ids[labeled_count:].copy()
    generator = np.random.default_rng(29 + view_count)
    cycle = generator.uniform(0.2, 1.0, size=(unlabeled.size, S)).astype(np.float32)
    target = generator.integers(
        0, 2, size=(unlabeled.size, labeled.size, S), dtype=np.int64
    ).astype(np.bool_)
    balance = generator.uniform(
        0.5, 1.5, size=target.shape
    ).astype(np.float32)
    return {
        "model": model,
        "views": views,
        "sample_ids": sample_ids,
        "labeled_ids": labeled,
        "unlabeled_ids": unlabeled,
        "u_cycle": cycle,
        "pred_relation": target,
        "balance": balance,
        "device": torch.device("cpu"),
    }


def make_native_state(case):
    from release_core.training import NativeTargetState

    N = case["sample_ids"].size
    V = len(case["views"])
    K = case["model"].n_clusters
    torch.manual_seed(211 + V)
    p_global = torch.softmax(torch.randn(N, K), dim=1).detach()
    matches = torch.stack([torch.eye(K) for _ in range(V)]).detach()
    return NativeTargetState(
        p_global=p_global,
        matches=matches,
        view_weights=tuple(1.0 for _ in range(V)),
        refresh_count=1,
        last_refresh_epoch=0,
    )


@pytest.mark.parametrize("view_count", [2, 5, 6])
def test_optimizer_topology_is_generic_and_decoupled(view_count):
    case = make_case(view_count)
    semantic, native, audit = build_decoupled_optimizers(case["model"], 1e-4)
    assert len(semantic) == len(native) == audit.view_count == view_count
    assert audit.separate_optimizer_objects
    assert audit.separate_state_mappings
    assert audit.identical_parameter_objects_per_view
    assert audit.exact_adam_configuration
    for autoencoder, left, right in zip(case["model"].autoencoders, semantic, native):
        left_ids = [id(p) for group in left.param_groups for p in group["params"]]
        right_ids = [id(p) for group in right.param_groups for p in group["params"]]
        assert left is not right
        assert left.state is not right.state
        assert left_ids == right_ids == [id(p) for p in autoencoder.parameters()]
        assert id(autoencoder._cluster_layer) in left_ids
        assert {id(p) for p in autoencoder._encoder.parameters()} <= set(left_ids)
        assert {id(p) for p in autoencoder._decoder.parameters()} <= set(left_ids)
        assert left.defaults == right.defaults
        assert left.defaults["lr"] == 1e-4
        assert left.defaults["betas"] == (0.9, 0.999)
        assert left.defaults["eps"] == 1e-8
        assert left.defaults["weight_decay"] == 0
        assert left.defaults["amsgrad"] is False
        assert not left.state and not right.state


def test_optimizer_states_evolve_independently_on_shared_parameters():
    case = make_case(2)
    semantic, native, _ = build_decoupled_optimizers(case["model"], 1e-4)
    for optimizer in semantic:
        optimizer.zero_grad(set_to_none=True)
    loss = sum(
        autoencoder.clustering(autoencoder.encoder(view))[:, 0].mean()
        for autoencoder, view in zip(case["model"].autoencoders, case["views"])
    )
    loss.backward()
    for optimizer in semantic:
        optimizer.step()
    assert all(optimizer.state for optimizer in semantic)
    assert all(not optimizer.state for optimizer in native)

    for optimizer in native:
        optimizer.zero_grad(set_to_none=True)
    native_loss = sum(
        output.square().mean() + q[:, 0].mean()
        for autoencoder, view in zip(case["model"].autoencoders, case["views"])
        for output, _, q in [autoencoder(view)]
    )
    native_loss.backward()
    for optimizer in native:
        optimizer.step()
    assert all(optimizer.state for optimizer in native)
    assert all(
        left.state is not right.state
        for left, right in zip(semantic, native)
    )


def test_caltech_parameter_tensor_count_is_compatibility_only():
    model = TinyBackbone(view_count=1, input_dim=4, latent_dim=10, clusters=3)
    model.autoencoders[0] = Autoencoder(
        [4, 500, 500, 2000, 10],
        activation="relu",
        batchnorm=False,
        n_clusters=3,
    )
    semantic, native, audit = build_decoupled_optimizers(model, 1e-4)
    assert audit.parameter_counts == (17,)
    assert len(semantic[0].param_groups[0]["params"]) == 17
    assert len(native[0].param_groups[0]["params"]) == 17


@pytest.mark.parametrize("lr", [0.0, -1.0, np.inf, np.nan])
def test_invalid_learning_rate_fails_closed(lr):
    with pytest.raises((TypeError, ValueError)):
        build_decoupled_optimizers(make_case()["model"], lr)
