"""Minimal independent-per-view backbone container."""

import torch

from .autoencoder import Autoencoder


class MultiViewBackbone:
    def __init__(self, config, view_num, view_size, n_clusters=20, seed=0):
        self._config = config
        self.view_num = int(view_num)
        self._latent_dim = config["Autoencoder"]["arch"][-1]
        self.autoencoders = []
        self.n_clusters = int(n_clusters)
        self.seed = int(seed)
        for view_index in range(self.view_num):
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            self.autoencoders.append(Autoencoder(
                [view_size[view_index], 500, 500, 2000, self._latent_dim],
                config["Autoencoder"]["activations"],
                config["Autoencoder"]["batchnorm"],
                n_clusters=self.n_clusters,
                FCN=config["Autoencoder"]["FCN"],
                channal=config["Autoencoder"]["channal"],
            ))

    def to_device(self, device):
        for autoencoder in self.autoencoders:
            autoencoder.to(device)
        return self

    def forward_view(self, view_index, values):
        return self.autoencoders[int(view_index)](values)

    def per_view_outputs(self, values):
        if len(values) != self.view_num:
            raise ValueError("input view count mismatch")
        return [model(value) for model, value in zip(self.autoencoders, values)]

    def state_dicts(self):
        return [model.state_dict() for model in self.autoencoders]

    def load_state_dicts(self, state_dicts, strict=True):
        if len(state_dicts) != self.view_num:
            raise ValueError("state view count mismatch")
        return [
            model.load_state_dict(state, strict=strict)
            for model, state in zip(self.autoencoders, state_dicts)
        ]
