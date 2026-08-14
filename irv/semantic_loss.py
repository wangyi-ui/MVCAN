"""Uniform cross-view semantic objective for B3-A1."""

import math

import torch
import torch.nn.functional as F


def uniform_cross_view_infonce(semantic_views, temperature):
    """Compute symmetric InfoNCE averaged uniformly over unordered view pairs.

    Args:
        semantic_views: List of V tensors, where semantic_views[v] has shape
            [batch_size, semantic_dim].
        temperature: Positive finite logit temperature.

    Returns:
        loss: Scalar tensor containing the uniform mean over all V choose 2
            symmetric pair losses.
        diagnostics: Scalar metadata only; pairwise logits are not retained.
    """
    if not isinstance(semantic_views, (list, tuple)):
        raise TypeError("semantic_views must be a list or tuple")
    if len(semantic_views) < 2:
        raise ValueError("uniform cross-view InfoNCE requires at least 2 views")

    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        raise TypeError("temperature must be a real scalar")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")

    first = semantic_views[0]
    if not torch.is_tensor(first) or first.ndim != 2:
        raise ValueError(
            "semantic_views[0] must have shape [batch_size, semantic_dim]"
        )
    batch_size = int(first.shape[0])
    semantic_dim = int(first.shape[1])
    if batch_size <= 0 or semantic_dim <= 0:
        raise ValueError("batch_size and semantic_dim must be positive")

    for view_idx, semantic_view in enumerate(semantic_views):
        if not torch.is_tensor(semantic_view) or semantic_view.ndim != 2:
            raise ValueError(
                "semantic_views["
                + str(view_idx)
                + "] must have shape [batch_size, semantic_dim]"
            )
        if tuple(semantic_view.shape) != (batch_size, semantic_dim):
            raise ValueError("all semantic views must have the same shape")
        if semantic_view.device != first.device:
            raise ValueError("all semantic views must be on the same device")
        if not bool(torch.isfinite(semantic_view).all().item()):
            raise ValueError("all semantic views must be finite")

    # target: [batch_size]
    target = torch.arange(
        batch_size,
        device=first.device,
        dtype=torch.long,
    )
    pair_losses = []
    for view_v in range(len(semantic_views)):
        for view_w in range(view_v + 1, len(semantic_views)):
            # s_v: [batch_size, semantic_dim]
            # s_w: [batch_size, semantic_dim]
            s_v = semantic_views[view_v]
            s_w = semantic_views[view_w]
            # logits_vw: [batch_size, batch_size]
            logits_vw = torch.matmul(s_v, s_w.T) / temperature
            # logits_wv: [batch_size, batch_size]
            logits_wv = logits_vw.T
            loss_vw = 0.5 * (
                F.cross_entropy(logits_vw, target)
                + F.cross_entropy(logits_wv, target)
            )
            pair_losses.append(loss_vw)

    loss = torch.stack(pair_losses).mean()
    diagnostics = {
        "pair_count": len(pair_losses),
        "batch_size": batch_size,
        "semantic_dim": semantic_dim,
        "temperature": temperature,
        "loss_finite": bool(torch.isfinite(loss).item()),
    }
    return loss, diagnostics
