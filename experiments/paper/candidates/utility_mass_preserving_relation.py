"""Unwired paper-local R4-v2 candidate: preserve absolute utility mass."""
from dataclasses import dataclass
import torch
from release_core.action.relation import build_action_weight, relation_bce
from release_core.semantics.relations import compute_view_relations

@dataclass(frozen=True)
class UtilityMassAudit:
    view_count: int; view_losses: tuple; balance_denominator: float; action_weight_sum: float; utility_mass: float
    query_gradient_capable: bool; anchor_stop_gradient_enforced: bool; u_cycle_detached: bool; pred_relation_detached: bool; balance_detached: bool; action_weight_detached: bool
    no_new_semantic_lambda: bool; denominator_semantics: str

def utility_mass_preserving_relation_loss(q_query_views, q_anchor_views, pred_relation, u_cycle, balance_weight):
    """Mean_v sum(U_cycle[:,None,:]*balance*BCE_v)/sum(balance).

    # action_weight:[B,L,S] = U_cycle[:,None,:] * balance_weight
    # utility_mass:[scalar], detached = action_weight.sum() / balance_weight.sum()
    # L_v_UMP = sum(action_weight * BCE_v) / sum(balance_weight)
    # therefore L_v_UMP = utility_mass * L_v_old (when old denominator is nonzero).
    """
    if not isinstance(q_query_views, (tuple, list)) or not isinstance(q_anchor_views, (tuple, list)) or len(q_query_views) < 2 or len(q_query_views) != len(q_anchor_views):
        raise ValueError("query and anchor views must be same-length V >= 2 sequences")
    reference = q_query_views[0]
    if not isinstance(reference, torch.Tensor) or reference.ndim != 2 or not reference.is_floating_point() or reference.shape[0] == 0 or reference.shape[1] == 0:
        raise ValueError("q_query_views[0] must be finite [B,K] floating tensor")
    B, K = reference.shape; L = None
    for query, anchor in zip(q_query_views, q_anchor_views):
        if not isinstance(query, torch.Tensor) or not isinstance(anchor, torch.Tensor) or query.shape != (B, K) or query.dtype != reference.dtype or query.device != reference.device or anchor.ndim != 2 or anchor.shape[1] != K or anchor.dtype != reference.dtype or anchor.device != reference.device:
            raise ValueError("view tensor shape, dtype, or device mismatch")
        L = anchor.shape[0] if L is None else L
        if anchor.shape[0] != L or L == 0 or not bool(torch.isfinite(query).all()) or not bool(torch.isfinite(anchor).all()):
            raise ValueError("view tensor must be finite and nonempty")
    target = torch.as_tensor(pred_relation, device=reference.device, dtype=reference.dtype).detach()
    if target.ndim != 3 or target.shape[:2] != (B, L) or target.shape[2] == 0 or not bool(torch.all((target == 0) | (target == 1))):
        raise ValueError("pred_relation must be binary [B,L,S]")
    action_weight = build_action_weight(u_cycle, balance_weight, like=reference)
    balance = torch.as_tensor(balance_weight, device=reference.device, dtype=reference.dtype).detach()
    if action_weight.shape != target.shape or balance.shape != target.shape:
        raise ValueError("action tensors must share [B,L,S]")
    denominator = balance.sum()
    if not bool(torch.isfinite(denominator)) or not bool(denominator > 0):
        raise ValueError("balance denominator must be finite and positive")
    losses = [torch.sum(action_weight * relation_bce(compute_view_relations(query, anchor), target)) / denominator for query, anchor in zip(q_query_views, q_anchor_views)]
    loss = torch.stack(losses).mean()
    mass = (action_weight.sum() / denominator).detach()
    return loss, UtilityMassAudit(len(losses), tuple(float(x.detach()) for x in losses), float(denominator.detach()), float(action_weight.sum().detach()), float(mass), bool(loss.requires_grad), True, True, True, True, not action_weight.requires_grad and action_weight.grad_fn is None, True, "balance-mass denominator preserves absolute U_cycle action strength")
