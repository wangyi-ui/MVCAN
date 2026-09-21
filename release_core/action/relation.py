"""Pure utility-conditioned sparse-relation objective."""

from dataclasses import dataclass

import torch

from release_core.semantics.relations import compute_view_relations


@dataclass(frozen=True)
class RelationActionAudit:
    """Detached metadata describing one pure relation-action objective."""

    view_count: int
    view_losses: tuple
    denominator: float
    view_arithmetic_mean: bool
    query_gradient_capable: bool
    anchor_stop_gradient_enforced: bool
    u_cycle_detached: bool
    pred_relation_detached: bool
    balance_detached: bool
    action_weight_detached: bool
    numerical_epsilon_only: bool


def _reference_tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise ValueError(name + " must be a torch tensor")
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(name + " must be a non-empty rank-2 tensor")
    if not value.is_floating_point():
        raise ValueError(name + " must have floating dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(name + " must be finite")
    return value


def build_action_weight(u_cycle, balance_weight, *, like):
    """Build detached ``[B,L,S]`` weights from frozen utility and balance."""
    reference = _reference_tensor(like, "like")
    cycle = torch.as_tensor(
        u_cycle, device=reference.device, dtype=reference.dtype
    ).detach()
    balance = torch.as_tensor(
        balance_weight, device=reference.device, dtype=reference.dtype
    ).detach()
    if cycle.ndim != 2 or cycle.shape[0] == 0 or cycle.shape[1] == 0:
        raise ValueError("u_cycle must have non-empty shape [B,S]")
    if (
        balance.ndim != 3
        or balance.shape[0] != cycle.shape[0]
        or balance.shape[1] == 0
        or balance.shape[2] != cycle.shape[1]
    ):
        raise ValueError("balance_weight must have compatible shape [B,L,S]")
    if not bool(torch.isfinite(cycle).all().item()) or bool(
        torch.any(cycle < 0).item()
    ):
        raise ValueError("u_cycle must be finite and nonnegative")
    if not bool(torch.isfinite(balance).all().item()) or bool(
        torch.any(balance <= 0).item()
    ):
        raise ValueError("balance_weight must be finite and positive")
    return (cycle[:, None, :] * balance).detach()


def relation_bce(relation_probability, pred_relation):
    """Return the broadcast manual BCE tensor with dtype-derived clamping."""
    probability = _reference_tensor(relation_probability, "relation_probability")
    target = torch.as_tensor(
        pred_relation, device=probability.device, dtype=probability.dtype
    ).detach()
    if (
        target.ndim != 3
        or target.shape[0] != probability.shape[0]
        or target.shape[1] != probability.shape[1]
        or target.shape[2] == 0
    ):
        raise ValueError("pred_relation must have compatible shape [B,L,S]")
    if not bool(torch.isfinite(target).all().item()) or not bool(
        torch.all((target == 0) | (target == 1)).item()
    ):
        raise ValueError("pred_relation must contain finite binary values")
    epsilon = torch.finfo(probability.dtype).eps
    clamped = probability.clamp(
        min=epsilon, max=1.0 - epsilon
    ).unsqueeze(-1)
    return -target * torch.log(clamped) - (1.0 - target) * torch.log(
        1.0 - clamped
    )


def utility_conditioned_relation_loss(
    q_query_views,
    q_anchor_views,
    pred_relation,
    u_cycle,
    balance_weight,
):
    """Compute independently normalized view losses and their arithmetic mean."""
    if not isinstance(q_query_views, (tuple, list)) or not isinstance(
        q_anchor_views, (tuple, list)
    ):
        raise ValueError("query and anchor views must be sequences")
    if len(q_query_views) != len(q_anchor_views) or len(q_query_views) < 2:
        raise ValueError("query and anchor views must have the same V >= 2")

    reference = _reference_tensor(q_query_views[0], "q_query_views[0]")
    batch_size, class_count = reference.shape
    anchor_count = None
    for view_id, (query, anchor) in enumerate(zip(q_query_views, q_anchor_views)):
        query = _reference_tensor(query, "q_query_views[" + str(view_id) + "]")
        anchor = _reference_tensor(anchor, "q_anchor_views[" + str(view_id) + "]")
        if (
            query.shape != (batch_size, class_count)
            or query.dtype != reference.dtype
            or query.device != reference.device
        ):
            raise ValueError("all query views must share [B,K], dtype, and device")
        if anchor_count is None:
            anchor_count = anchor.shape[0]
        if (
            anchor.shape != (anchor_count, class_count)
            or anchor.dtype != reference.dtype
            or anchor.device != reference.device
        ):
            raise ValueError("all anchor views must share [L,K], dtype, and device")

    target = torch.as_tensor(
        pred_relation, device=reference.device, dtype=reference.dtype
    ).detach()
    if target.ndim != 3 or target.shape[:2] != (batch_size, anchor_count):
        raise ValueError("pred_relation must have compatible shape [B,L,S]")
    if not bool(torch.isfinite(target).all().item()) or not bool(
        torch.all((target == 0) | (target == 1)).item()
    ):
        raise ValueError("pred_relation must contain finite binary values")

    action_weight = build_action_weight(
        u_cycle, balance_weight, like=reference
    )
    if action_weight.shape != target.shape:
        raise ValueError("action tensors must share shape [B,L,S]")
    denominator = action_weight.sum()
    epsilon = torch.finfo(reference.dtype).eps
    if not bool(torch.isfinite(denominator).item()) or not bool(
        (denominator > epsilon).item()
    ):
        raise ValueError("relation denominator is non-finite or too small")

    view_losses = []
    for query, anchor in zip(q_query_views, q_anchor_views):
        probability = compute_view_relations(query, anchor)
        bce = relation_bce(probability, target)
        view_loss = torch.sum(action_weight * bce) / denominator
        if not bool(torch.isfinite(view_loss).item()):
            raise ValueError("relation view loss is non-finite")
        view_losses.append(view_loss)
    loss = torch.stack(view_losses).mean()
    audit = RelationActionAudit(
        view_count=len(view_losses),
        view_losses=tuple(float(value.detach().item()) for value in view_losses),
        denominator=float(denominator.detach().item()),
        view_arithmetic_mean=True,
        query_gradient_capable=bool(loss.requires_grad),
        anchor_stop_gradient_enforced=True,
        u_cycle_detached=True,
        pred_relation_detached=True,
        balance_detached=True,
        action_weight_detached=(
            not action_weight.requires_grad and action_weight.grad_fn is None
        ),
        numerical_epsilon_only=True,
    )
    return loss, audit
