from types import SimpleNamespace

import torch

from experiments.paper.diagnostics.state_hashing import optimizer_state_snapshot
from release_core.backbone.autoencoder import Autoencoder


def test_adam_snapshot_records_step_exp_avg_and_exp_avg_sq_per_parameter():
    torch.manual_seed(20)
    autoencoder = Autoencoder(
        [3, 2], activation="relu", batchnorm=False, n_clusters=2
    )
    model = SimpleNamespace(autoencoders=[autoencoder])
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    before = optimizer_state_snapshot((optimizer,), model)

    values = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    latent = autoencoder.encoder(values)
    loss = torch.nn.functional.mse_loss(autoencoder.decoder(latent), values)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    after = optimizer_state_snapshot((optimizer,), model)
    assert before["aggregate_hash"] != after["aggregate_hash"]
    active = 0
    inactive = 0
    for record in after["per_view"][0]["parameters"]:
        fields = record["state_tensor_hashes"]
        if fields:
            active += 1
            assert {"step", "exp_avg", "exp_avg_sq"}.issubset(fields)
        else:
            inactive += 1
    assert active > 0
    assert inactive == 1
