"""Detached per-view semantic projections for B3-A0."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetachedSemanticHeadBank(nn.Module):
    """A bank of independently initialized, per-view linear projections."""

    def __init__(self, view_num, latent_dim, semantic_dim, semantic_seed):
        super(DetachedSemanticHeadBank, self).__init__()
        self.view_num = int(view_num)
        self.latent_dim = int(latent_dim)
        self.semantic_dim = int(semantic_dim)
        self.semantic_seed = int(semantic_seed)

        if self.view_num <= 0:
            raise ValueError("view_num must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.semantic_dim <= 0:
            raise ValueError("semantic_dim must be positive")

        # The heads are created on CPU. fork_rng restores the Native torch RNG
        # state exactly when this context exits and does not touch CUDA RNGs.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.semantic_seed)
            heads = [
                nn.Linear(self.latent_dim, self.semantic_dim)
                for _ in range(self.view_num)
            ]
        self.heads = nn.ModuleList(heads)
        self.last_forward_audit = None

    def forward_one(self, z_v, view_idx):
        """Project one view without allowing gradients into its latent input.

        Args:
            z_v: [batch_size, latent_dim] Native encoder latent tensor.
            view_idx: Integer index selecting one per-view semantic head.

        Returns:
            s_v: [batch_size, semantic_dim] L2-normalized semantic tensor.
        """
        view_idx = int(view_idx)
        if not 0 <= view_idx < self.view_num:
            raise IndexError("view_idx is outside the semantic head bank")
        if z_v.ndim != 2 or z_v.shape[1] != self.latent_dim:
            raise ValueError(
                "z_v must have shape [batch_size, "
                + str(self.latent_dim)
                + "]"
            )

        # z_v:        [batch_size, latent_dim]
        # z_detached: [batch_size, latent_dim]
        z_detached = z_v.detach()
        # s_raw_v:    [batch_size, semantic_dim]
        s_raw_v = self.heads[view_idx](z_detached)
        # s_v:        [batch_size, semantic_dim]
        s_v = F.normalize(s_raw_v, p=2, dim=1, eps=1e-12)
        return s_v

    def forward_views(self, z_views):
        """Project every Native view through its isolated semantic head.

        Args:
            z_views: List of view_num tensors shaped
                [batch_size, latent_dim]. This is the future semantic-loss
                interface; B3-A0 only probes it under no_grad at the caller.

        Returns:
            s_views: List of view_num tensors shaped
                [batch_size, semantic_dim].
        """
        if len(z_views) != self.view_num:
            raise ValueError("z_views length must equal view_num")

        s_views = []
        detached_inputs_pass = True
        for view_idx, z_v in enumerate(z_views):
            # z_v:        [batch_size, latent_dim]
            # z_detached: [batch_size, latent_dim]
            z_detached = z_v.detach()
            detached_inputs_pass = detached_inputs_pass and (
                not z_detached.requires_grad and z_detached.grad_fn is None
            )
            # forward_one explicitly detaches again at the semantic boundary.
            # s_v: [batch_size, semantic_dim]
            s_v = self.forward_one(z_detached, view_idx)
            s_views.append(s_v)

        self.last_forward_audit = {
            "semantic_input_detached_pass": bool(detached_inputs_pass),
        }
        return s_views

