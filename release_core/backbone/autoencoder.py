"""Frozen fully connected Autoencoder numerical primitive."""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Autoencoder(nn.Module):
    def __init__(self, encoder_dim, activation="relu", batchnorm=True,
                 n_clusters=10, FCN=True, channal=1):
        super(Autoencoder, self).__init__()
        self.FCN = FCN
        encoder_layers = []
        self._activation = activation
        self._batchnorm = batchnorm
        if FCN:
            self._dim = len(encoder_dim) - 1
            for i in range(self._dim):
                encoder_layers.append(nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
                if i < self._dim - 1:
                    if self._batchnorm:
                        encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                    if self._activation == "sigmoid":
                        encoder_layers.append(nn.Sigmoid())
                    elif self._activation == "leakyrelu":
                        encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                    elif self._activation == "tanh":
                        encoder_layers.append(nn.Tanh())
                    elif self._activation == "relu":
                        if i < self._dim - 1:
                            encoder_layers.append(nn.ReLU())
                    else:
                        raise ValueError("Unknown activation type %s" % self._activation)
        else:
            encoder_layers = [
                nn.Conv2d(channal[0], 32, (4, 4), stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, (4, 4), stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, (4, 4), stride=2, padding=1), nn.ReLU(),
                nn.Flatten(start_dim=1), nn.Linear(64 * 4 * 4, encoder_dim[-1]),
            ]
        self._encoder = nn.Sequential(*encoder_layers)
        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        if FCN:
            for i in range(self._dim):
                decoder_layers.append(nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
                if self._batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
                if self._activation == "sigmoid":
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == "leakyrelu":
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == "tanh":
                    decoder_layers.append(nn.Tanh())
                elif self._activation == "relu":
                    if i < self._dim - 1:
                        decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError("Unknown activation type %s" % self._activation)
        else:
            decoder_layers = [
                nn.Linear(encoder_dim[-1], 64 * 4 * 4), nn.ReLU(),
                nn.Unflatten(dim=1, unflattened_size=(64, 4, 4)),
                nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1), nn.ReLU(),
                nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1), nn.ReLU(),
                nn.ConvTranspose2d(32, channal[0], (4, 4), stride=2, padding=1),
            ]
        self._decoder = nn.Sequential(*decoder_layers)
        self.alpha = 1.0
        self._cluster_layer = Parameter(torch.Tensor(n_clusters, encoder_dim[-1]))
        torch.nn.init.xavier_normal_(self._cluster_layer.data)

    def encoder(self, x):
        return self._encoder(x)

    def decoder(self, latent):
        return self._decoder(latent)

    def clustering(self, latent):
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(latent.unsqueeze(1) - self._cluster_layer, 2), 2
        ) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        return (q.t() / torch.sum(q, 1)).t()

    def forward(self, x):
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(latent.unsqueeze(1) - self._cluster_layer, 2), 2
        ) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_hat, latent, q
