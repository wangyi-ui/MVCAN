"""Frozen per-view native objective."""

import torch.nn.functional as F


def native_objective(reconstruction, inputs, q_local, p_local, lambda1):
    reconstruction_loss = F.mse_loss(reconstruction, inputs)
    clustering_loss = F.mse_loss(q_local, p_local)
    total = reconstruction_loss + lambda1 * clustering_loss
    return total, reconstruction_loss, clustering_loss

