"""Frozen sparse-prototype semantic evidence for the E1 LWC carrier.

Reliability is used only while constructing sparse prototypes.  It never
multiplies graph evidence or the final pair weight directly.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    glgc_positive_pair_contrastive_loss,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    symmetric_pair_graph_evidence,
)


SEMANTIC_EPS = 1e-12
E2_ARMS = (
    "LWC_REPLAY",
    "S_LWC",
    "RS_LWC",
    "SHUFFLED_RS_LWC",
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _frozen_tensor(value, device=None, dtype=None):
    if torch.is_tensor(value):
        tensor = value.detach()
        if device is not None or dtype is not None:
            tensor = tensor.to(
                device=tensor.device if device is None else device,
                dtype=tensor.dtype if dtype is None else dtype,
            )
        return tensor.detach()
    array = np.array(value, copy=True, order="C")
    tensor = torch.from_numpy(array)
    if device is not None or dtype is not None:
        tensor = tensor.to(
            device=tensor.device if device is None else device,
            dtype=tensor.dtype if dtype is None else dtype,
        )
    return tensor.detach()


@torch.no_grad()
def build_sparse_class_prototypes(
    h_sem_full,
    labeled_sample_ids,
    labeled_targets,
    class_num=7,
    reliability=None,
    eps=SEMANTIC_EPS,
):
    """Build frozen [V,K,D] prototypes from exactly two anchors per class.

    With ``reliability=None`` this implements
    Normalize(sum_l h_lvc).  Otherwise it implements
    Normalize(sum_l R_lv*h_lvc / (sum_l R_lv + eps)).
    """
    if not torch.is_tensor(h_sem_full) or h_sem_full.ndim != 3:
        raise ValueError("h_sem_full must be a tensor with shape [N,V,D]")
    sample_num, view_num, semantic_dim = h_sem_full.shape
    class_num = int(class_num)
    eps = float(eps)
    if class_num <= 1 or not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("class_num and eps are invalid")
    # Prototype inputs are forcibly detached: this branch cannot update MVCAN.
    frozen_h = h_sem_full.detach()
    ids = _frozen_tensor(
        labeled_sample_ids, device=frozen_h.device, dtype=torch.long
    )
    targets = _frozen_tensor(
        labeled_targets, device=frozen_h.device, dtype=torch.long
    )
    _require(ids.shape == (14,), "labeled_sample_ids must have shape [14]")
    _require(targets.shape == (14,), "labeled_targets must have shape [14]")
    _require(
        torch.unique(ids).numel() == 14
        and torch.all((ids >= 0) & (ids < sample_num)).item(),
        "labeled sample IDs are invalid",
    )
    _require(
        torch.all((targets >= 0) & (targets < class_num)).item(),
        "sparse targets are outside [0,K)",
    )
    class_counts = torch.bincount(targets, minlength=class_num)
    _require(
        torch.equal(
            class_counts,
            torch.full_like(class_counts, 2),
        ),
        "every class must have exactly two labeled anchors",
    )

    frozen_r = None
    if reliability is not None:
        frozen_r = _frozen_tensor(
            reliability, device=frozen_h.device, dtype=frozen_h.dtype
        )
        _require(
            frozen_r.shape == (sample_num, view_num),
            "R must have shape [N,V]",
        )
        _require(
            torch.isfinite(frozen_r).all().item()
            and torch.all((frozen_r >= 0.0) & (frozen_r <= 1.0)).item(),
            "R must be finite in [0,1]",
        )

    # prototypes: [V,K,D], fully assigned below; no zero-fill fallback exists.
    prototypes = torch.empty(
        (view_num, class_num, semantic_dim),
        device=frozen_h.device,
        dtype=frozen_h.dtype,
    )
    anchor_counts = torch.empty(
        (view_num, class_num), device=frozen_h.device, dtype=torch.long
    )
    denominators = torch.empty(
        (view_num, class_num), device=frozen_h.device, dtype=frozen_h.dtype
    )
    for view_id in range(view_num):
        for class_id in range(class_num):
            selected = ids[targets == class_id]
            count = int(selected.numel())
            _require(count == 2, "view/class anchor count is not exactly two")
            anchors = frozen_h[selected, view_id, :]  # [2,D]
            anchor_counts[view_id, class_id] = count
            if frozen_r is None:
                denominator = torch.tensor(
                    float(count), device=frozen_h.device, dtype=frozen_h.dtype
                )
                prototype_source = anchors.sum(dim=0)
            else:
                weights = frozen_r[selected, view_id]  # [2]
                denominator = weights.sum()
                _require(
                    torch.isfinite(denominator).item()
                    and float(denominator.item()) > 0.0,
                    "reliability prototype denominator must be positive",
                )
                prototype_source = (
                    (weights[:, None] * anchors).sum(dim=0)
                    / (denominator + eps)
                )
            denominators[view_id, class_id] = denominator
            source_norm = torch.linalg.vector_norm(
                prototype_source, ord=2, dim=0
            )
            _require(
                torch.isfinite(source_norm).item()
                and float(source_norm.item()) > 0.0,
                "prototype source has zero or non-finite norm",
            )
            prototypes[view_id, class_id] = F.normalize(
                prototype_source, p=2, dim=0, eps=eps
            )

    prototype_norms = torch.linalg.vector_norm(prototypes, ord=2, dim=-1)
    validity = {
        "shape_pass": tuple(prototypes.shape)
        == (view_num, class_num, semantic_dim),
        "anchor_count_two_every_view_class_pass": bool(
            torch.all(anchor_counts == 2).item()
        ),
        "denominator_positive_every_view_class_pass": bool(
            torch.all(denominators > 0.0).item()
        ),
        "prototype_finite_pass": bool(torch.isfinite(prototypes).all().item()),
        "prototype_unit_norm_pass": bool(
            torch.allclose(
                prototype_norms,
                torch.ones_like(prototype_norms),
                atol=1e-6,
                rtol=1e-6,
            )
        ),
        "no_missing_class_pass": bool(
            torch.all(anchor_counts.sum(dim=0) == 2 * view_num).item()
        ),
        "prototype_stop_gradient_pass": bool(
            not prototypes.requires_grad and prototypes.grad_fn is None
        ),
    }
    _require(all(validity.values()), "sparse prototype validity audit failed")
    audit = {
        **validity,
        "weighted_by_reliability": frozen_r is not None,
        "shape": list(prototypes.shape),
        "anchor_counts": anchor_counts.detach().cpu().tolist(),
        "class_counts": class_counts.detach().cpu().tolist(),
        "denominator_min": float(denominators.min().item()),
        "denominator_max": float(denominators.max().item()),
        "prototype_norm_min": float(prototype_norms.min().item()),
        "prototype_norm_max": float(prototype_norms.max().item()),
        "zero_fill_used": False,
    }
    return prototypes.detach(), audit


@torch.no_grad()
def sparse_semantic_probabilities(
    h_sem_detached,
    prototypes,
    eps=SEMANTIC_EPS,
):
    """Return frozen view-specific semantic probabilities [B,V,K]."""
    if not torch.is_tensor(h_sem_detached) or h_sem_detached.ndim != 3:
        raise ValueError("h_sem_detached must have shape [B,V,D]")
    if not torch.is_tensor(prototypes) or prototypes.ndim != 3:
        raise ValueError("prototypes must have shape [V,K,D]")
    batch_size, view_num, semantic_dim = h_sem_detached.shape
    prototype_view_num, class_num, prototype_dim = prototypes.shape
    if (prototype_view_num, prototype_dim) != (view_num, semantic_dim):
        raise ValueError("prototype shape does not match h_sem")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    frozen_h = F.normalize(h_sem_detached.detach(), p=2, dim=-1, eps=eps)
    frozen_p = F.normalize(prototypes.detach(), p=2, dim=-1, eps=eps)
    # cosine: [B,V,K], no learnable or tunable temperature.
    cosine = torch.einsum("bvd,vcd->bvc", frozen_h, frozen_p).clamp(-1.0, 1.0)
    affinity = (1.0 + cosine) / 2.0 + eps
    semantic_prob = affinity / affinity.sum(dim=-1, keepdim=True)
    row_sums = semantic_prob.sum(dim=-1)
    validity = {
        "shape_pass": tuple(semantic_prob.shape)
        == (batch_size, view_num, class_num),
        "finite_pass": bool(torch.isfinite(semantic_prob).all().item()),
        "nonnegative_pass": bool(torch.all(semantic_prob >= 0.0).item()),
        "row_sum_one_pass": bool(
            torch.allclose(
                row_sums,
                torch.ones_like(row_sums),
                atol=1e-6,
                rtol=1e-6,
            )
        ),
        "stop_gradient_pass": bool(
            not semantic_prob.requires_grad and semantic_prob.grad_fn is None
        ),
    }
    _require(all(validity.values()), "semantic probability validity failed")
    audit = {
        **validity,
        "shape": list(semantic_prob.shape),
        "minimum": float(semantic_prob.min().item()),
        "maximum": float(semantic_prob.max().item()),
        "max_row_sum_error": float((row_sums - 1.0).abs().max().item()),
        "temperature_used": False,
        "affinity_formula": "(1 + cosine) / 2 + eps",
    }
    return semantic_prob.detach(), audit


@torch.no_grad()
def normalized_js_similarity(semantic_prob, left, right, eps=SEMANTIC_EPS):
    """Return S_i^{uv}=1-JS(p_u,p_v)/log(2), shape [B]."""
    if not torch.is_tensor(semantic_prob) or semantic_prob.ndim != 3:
        raise ValueError("semantic_prob must have shape [B,V,K]")
    _, view_num, _ = semantic_prob.shape
    left = int(left)
    right = int(right)
    if not (0 <= left < right < view_num):
        raise ValueError("view pair must satisfy 0 <= left < right < V")
    eps = float(eps)
    frozen = semantic_prob.detach()
    p_left = frozen[:, left, :].clamp_min(eps)
    p_right = frozen[:, right, :].clamp_min(eps)
    mixture = 0.5 * (p_left + p_right)
    js = 0.5 * torch.sum(
        p_left * (torch.log(p_left) - torch.log(mixture)), dim=-1
    ) + 0.5 * torch.sum(
        p_right * (torch.log(p_right) - torch.log(mixture)), dim=-1
    )
    similarity = (1.0 - js / math.log(2.0)).clamp(0.0, 1.0).detach()
    _require(
        torch.isfinite(similarity).all().item()
        and torch.all((similarity >= 0.0) & (similarity <= 1.0)).item(),
        "normalized JS similarity must be finite in [0,1]",
    )
    _require(
        not similarity.requires_grad and similarity.grad_fn is None,
        "S must be stop-gradient",
    )
    return similarity


def semantic_interaction_weights(graph_inter, semantic_prob, left, right):
    """Return frozen U_sem=G*S [B]; R is intentionally absent from this API."""
    g_pair = symmetric_pair_graph_evidence(graph_inter, left, right)
    similarity = normalized_js_similarity(semantic_prob, left, right)
    if g_pair.shape != similarity.shape:
        raise ValueError("G and S pair shapes differ")
    # U_sem: [B], structural G times sparse semantic compatibility S only.
    interaction_utility = (g_pair * similarity).detach()
    _require(
        torch.isfinite(interaction_utility).all().item()
        and torch.all(interaction_utility >= 0.0).item(),
        "U_sem must be finite and non-negative",
    )
    _require(
        not interaction_utility.requires_grad
        and interaction_utility.grad_fn is None,
        "U_sem must be stop-gradient",
    )
    return interaction_utility, g_pair, similarity


def semantic_pairwise_cooperation_loss(h_sem, graph_inter, semantic_prob):
    """Sum E1 contrastive losses weighted by U_sem=G*S over all u<v."""
    if not torch.is_tensor(h_sem) or h_sem.ndim != 3:
        raise ValueError("h_sem must have shape [B,V,D]")
    batch_size, view_num, semantic_dim = h_sem.shape
    if tuple(graph_inter.shape) != (
        view_num,
        view_num,
        batch_size,
        batch_size,
    ):
        raise ValueError("G_inter shape mismatch")
    if tuple(semantic_prob.shape) != (batch_size, view_num, semantic_dim):
        raise ValueError("semantic_prob shape mismatch")
    losses = []
    diagnostics = []
    for left in range(view_num):
        for right in range(left + 1, view_num):
            weights, g_pair, similarity = semantic_interaction_weights(
                graph_inter, semantic_prob, left, right
            )
            # h_sem is deliberately non-detached here, preserving
            # L_pair -> q_local -> MVCAN cluster layer -> encoder.
            loss = glgc_positive_pair_contrastive_loss(
                h_sem[:, left, :], h_sem[:, right, :], weights
            )
            losses.append(loss)
            diagnostics.append({
                "left": left,
                "right": right,
                "G_min": float(g_pair.min().item()),
                "G_max": float(g_pair.max().item()),
                "S_min": float(similarity.min().item()),
                "S_max": float(similarity.max().item()),
                "U_sem_min": float(weights.min().item()),
                "U_sem_max": float(weights.max().item()),
                "S_range_pass": bool(
                    torch.all((similarity >= 0.0) & (similarity <= 1.0)).item()
                ),
                "U_sem_stop_gradient_pass": bool(
                    not weights.requires_grad and weights.grad_fn is None
                ),
            })
    total = torch.stack(losses).sum()
    _require(torch.isfinite(total).item(), "semantic pair loss is not finite")
    return total, diagnostics
