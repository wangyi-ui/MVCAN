from types import SimpleNamespace

import torch

from experiments.paper.diagnostics.state_hashing import stage_snapshot
from release_core.backbone.autoencoder import Autoencoder


def _fixture():
    torch.manual_seed(20)
    model = SimpleNamespace(autoencoders=[
        Autoencoder([3, 2], activation="relu", batchnorm=False, n_clusters=2)
        for _ in range(5)
    ])
    optimizers = tuple(
        torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
        for autoencoder in model.autoencoders
    )
    views = tuple(torch.arange(12, dtype=torch.float32).reshape(4, 3) for _ in range(5))
    generator = torch.Generator(device="cpu").manual_seed(20)
    return model, optimizers, views, generator


def test_snapshot_contains_all_preregistered_logical_hash_families():
    model, optimizers, views, generator = _fixture()
    snapshot = stage_snapshot("S0_CONSTRUCTION", model, optimizers, generator, views)
    assert snapshot["stage"] == "S0_CONSTRUCTION"
    assert len(snapshot["model"]["per_view"]) == 5
    assert len(snapshot["optimizer"]["per_view"]) == 5
    assert len(snapshot["representation"]["latent_per_view_hash"]) == 5
    assert len(snapshot["representation"]["q_local_per_view_hash"]) == 5
    assert set(snapshot["rng"]) == {
        "python_random_hash", "numpy_rng_hash", "torch_cpu_rng_hash",
        "torch_cuda_rng_hash", "dataloader_generator_state_hash",
    }
    per_view = snapshot["model"]["per_view"][0]
    assert per_view["encoder_parameter_hash"]
    assert per_view["decoder_parameter_hash"]
    assert per_view["cluster_centers_hash"]


def test_snapshot_hash_is_stable_without_a_state_update():
    model, optimizers, views, generator = _fixture()
    first = stage_snapshot("S", model, optimizers, generator, views)
    second = stage_snapshot("S", model, optimizers, generator, views)
    assert first["model"] == second["model"]
    assert first["optimizer"] == second["optimizer"]
    assert first["representation"] == second["representation"]
    assert first["rng"] == second["rng"]
