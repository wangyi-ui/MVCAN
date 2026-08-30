"""Isolated shared-semantic projector, loss, readout, and diagnostics."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from irv.b3_semantic_diagnostics import effective_rank


EXPECTED_VIEW_NUM = 5
EXPECTED_PRIVATE_DIM = 10
EXPECTED_HIDDEN_DIM = 32
EXPECTED_SEMANTIC_DIM = 10
EXPECTED_TEMPERATURE = 0.2


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


class ViewProjectors(nn.Module):
    """Five independent minimal view-specific semantic projectors."""

    def __init__(
        self,
        view_num=EXPECTED_VIEW_NUM,
        private_dim=EXPECTED_PRIVATE_DIM,
        hidden_dim=EXPECTED_HIDDEN_DIM,
        semantic_dim=EXPECTED_SEMANTIC_DIM,
    ):
        super().__init__()
        _require(view_num == EXPECTED_VIEW_NUM, "view_num must be 5")
        _require(private_dim == EXPECTED_PRIVATE_DIM, "private_dim must be 10")
        _require(hidden_dim == EXPECTED_HIDDEN_DIM, "hidden_dim must be 32")
        _require(
            semantic_dim == EXPECTED_SEMANTIC_DIM,
            "semantic_dim must be 10",
        )
        self.view_num = int(view_num)
        self.private_dim = int(private_dim)
        self.hidden_dim = int(hidden_dim)
        self.semantic_dim = int(semantic_dim)
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.private_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.semantic_dim),
            )
            for _ in range(self.view_num)
        ])

    def forward(self, z_stack):
        # z_stack: [N,V,Dz]
        _require(
            torch.is_tensor(z_stack) and z_stack.ndim == 3,
            "z_stack must have shape [N,V,Dz]",
        )
        _require(
            tuple(z_stack.shape[1:])
            == (self.view_num, self.private_dim),
            "z_stack view/private shape mismatch",
        )
        outputs = []
        for view_id, projector in enumerate(self.projectors):
            # projector_input: [N,Dz]
            projector_input = z_stack[:, view_id, :]
            # h_v_raw: [N,Ds]
            h_v_raw = projector(projector_input)
            # h_v: [N,Ds]
            h_v = F.normalize(h_v_raw, p=2, dim=-1)
            outputs.append(h_v)
        # h_stack: [N,V,Ds]
        h_stack = torch.stack(outputs, dim=1)
        _require(
            tuple(h_stack.shape)
            == (
                int(z_stack.shape[0]),
                self.view_num,
                self.semantic_dim,
            ),
            "h_stack shape mismatch",
        )
        return h_stack


def validate_admission_mask(admission, sample_num=None):
    # admission: [N,V]
    mask = torch.as_tensor(admission, dtype=torch.bool)
    _require(mask.ndim == 2, "admission must have shape [N,V]")
    _require(
        int(mask.shape[1]) == EXPECTED_VIEW_NUM,
        "admission must contain five views",
    )
    if sample_num is not None:
        _require(
            int(mask.shape[0]) == int(sample_num),
            "admission sample count mismatch",
        )
    _require(
        bool(torch.all(mask.sum(dim=1) > 0).item()),
        "every sample must admit at least one view",
    )
    return mask


def admitted_pair_indices(admission, view_v, view_w):
    mask = validate_admission_mask(admission)
    view_v = int(view_v)
    view_w = int(view_w)
    _require(
        0 <= view_v < view_w < EXPECTED_VIEW_NUM,
        "pair must satisfy 0 <= v < w < 5",
    )
    selected = torch.logical_and(
        mask[:, view_v],
        mask[:, view_w],
    )
    # idx_vw: [M]
    return torch.nonzero(selected, as_tuple=False).flatten()


def symmetric_pair_infonce(pair_h_v, pair_h_w, temperature):
    # pair_h_v: [M,Ds]
    # pair_h_w: [M,Ds]
    _require(
        torch.is_tensor(pair_h_v)
        and torch.is_tensor(pair_h_w)
        and pair_h_v.ndim == 2
        and pair_h_w.ndim == 2,
        "pair tensors must have shape [M,Ds]",
    )
    _require(
        tuple(pair_h_v.shape) == tuple(pair_h_w.shape),
        "pair tensors must align",
    )
    sample_count = int(pair_h_v.shape[0])
    _require(sample_count >= 2, "InfoNCE pair requires M >= 2")
    temperature = float(temperature)
    _require(
        math.isfinite(temperature) and temperature == EXPECTED_TEMPERATURE,
        "temperature must be exactly 0.2",
    )
    # pair_logits: [M,M]
    pair_logits = torch.matmul(pair_h_v, pair_h_w.T) / temperature
    # targets: [M]
    targets = torch.arange(
        sample_count,
        dtype=torch.long,
        device=pair_logits.device,
    )
    loss = 0.5 * (
        F.cross_entropy(pair_logits, targets)
        + F.cross_entropy(pair_logits.T, targets)
    )
    _require(bool(torch.isfinite(loss).item()), "pair loss non-finite")
    return loss, pair_logits


def admitted_symmetric_infonce(h_stack, admission, temperature):
    # h_stack: [N,V,Ds]
    _require(
        torch.is_tensor(h_stack) and h_stack.ndim == 3,
        "h_stack must have shape [N,V,Ds]",
    )
    _require(
        tuple(h_stack.shape[1:])
        == (EXPECTED_VIEW_NUM, EXPECTED_SEMANTIC_DIM),
        "h_stack view/semantic shape mismatch",
    )
    mask = validate_admission_mask(
        admission,
        sample_num=h_stack.shape[0],
    ).to(h_stack.device)
    weighted_loss = None
    weighted_sample_count = 0
    pair_sample_counts = {}
    pair_losses = {}
    for view_v in range(EXPECTED_VIEW_NUM):
        for view_w in range(view_v + 1, EXPECTED_VIEW_NUM):
            pair_name = str(view_v) + "-" + str(view_w)
            idx_vw = admitted_pair_indices(
                mask,
                view_v,
                view_w,
            ).to(h_stack.device)
            sample_count = int(idx_vw.numel())
            pair_sample_counts[pair_name] = sample_count
            if sample_count < 2:
                pair_losses[pair_name] = None
                continue
            # pair_h_v: [M,Ds]
            pair_h_v = h_stack[idx_vw, view_v, :]
            # pair_h_w: [M,Ds]
            pair_h_w = h_stack[idx_vw, view_w, :]
            pair_loss, _ = symmetric_pair_infonce(
                pair_h_v,
                pair_h_w,
                temperature,
            )
            pair_losses[pair_name] = float(pair_loss.detach().item())
            contribution = sample_count * pair_loss
            weighted_loss = (
                contribution
                if weighted_loss is None
                else weighted_loss + contribution
            )
            weighted_sample_count += sample_count
    _require(
        weighted_loss is not None and weighted_sample_count > 0,
        "no eligible admitted view pair",
    )
    semantic_loss = weighted_loss / weighted_sample_count
    _require(
        bool(torch.isfinite(semantic_loss).item()),
        "semantic loss non-finite",
    )
    return semantic_loss, {
        "pair_sample_counts": pair_sample_counts,
        "pair_losses": pair_losses,
        "weighted_sample_count": int(weighted_sample_count),
        "temperature": float(temperature),
    }


def shared_semantic_readout(h_stack, admission):
    # h_stack: [N,V,Ds]
    _require(
        torch.is_tensor(h_stack) and h_stack.ndim == 3,
        "h_stack must have shape [N,V,Ds]",
    )
    mask = validate_admission_mask(
        admission,
        sample_num=h_stack.shape[0],
    ).to(h_stack.device)
    # admission_float: [N,V,1]
    admission_float = mask.to(h_stack.dtype).unsqueeze(-1)
    # masked_sum: [N,Ds]
    masked_sum = torch.sum(h_stack * admission_float, dim=1)
    # denominator: [N,1]
    denominator = torch.sum(admission_float, dim=1)
    # semantic: [N,Ds]
    semantic = F.normalize(
        masked_sum / denominator,
        p=2,
        dim=-1,
    )
    _require(
        tuple(semantic.shape)
        == (int(h_stack.shape[0]), EXPECTED_SEMANTIC_DIM),
        "semantic shape mismatch",
    )
    return semantic


def admission_purity_diagnostic(admission, oracle_clean_mask):
    admitted = np.asarray(admission, dtype=bool)
    oracle = np.asarray(oracle_clean_mask, dtype=bool)
    _require(
        admitted.shape == oracle.shape and admitted.ndim == 2,
        "admission/oracle shape mismatch",
    )
    _require(
        np.all(admitted.sum(axis=1) > 0),
        "admission rows must be nonempty",
    )
    intersection = np.logical_and(admitted, oracle).sum(
        axis=1,
        dtype=np.int64,
    )
    union = np.logical_or(admitted, oracle).sum(
        axis=1,
        dtype=np.int64,
    )
    corrupt_admitted = np.logical_and(
        admitted,
        np.logical_not(oracle),
    ).sum(axis=1, dtype=np.int64)
    return {
        "admitted_clean_fraction": float(
            np.sum(intersection, dtype=np.int64)
            / np.sum(admitted, dtype=np.int64)
        ),
        "corrupted_views_admitted_mean": float(
            np.mean(corrupt_admitted)
        ),
        "all_admitted_clean_fraction": float(
            np.mean(corrupt_admitted == 0)
        ),
        "oracle_set_jaccard_mean": float(
            np.mean(intersection / union)
        ),
    }


def offdiag_cosine_values(semantic):
    _require(
        torch.is_tensor(semantic) and semantic.ndim == 2,
        "semantic must have shape [N,Ds]",
    )
    normalized = F.normalize(semantic, p=2, dim=-1)
    cosine = torch.matmul(normalized, normalized.T)
    offdiag_mask = ~torch.eye(
        semantic.shape[0],
        dtype=torch.bool,
        device=semantic.device,
    )
    return cosine[offdiag_mask]


def collapse_diagnostics(semantic, prediction):
    _require(
        torch.is_tensor(semantic)
        and tuple(semantic.shape[1:]) == (EXPECTED_SEMANTIC_DIM,),
        "semantic must have shape [N,10]",
    )
    all_finite = bool(torch.isfinite(semantic).all().item())
    _require(all_finite, "semantic contains NaN or Inf")
    prediction_array = np.asarray(prediction)
    _require(
        prediction_array.shape == (int(semantic.shape[0]),),
        "prediction shape mismatch",
    )
    norms = torch.linalg.vector_norm(semantic, ord=2, dim=1)
    offdiag = offdiag_cosine_values(semantic)
    unique, counts = np.unique(
        prediction_array,
        return_counts=True,
    )
    effective_rank_value = float(effective_rank(semantic))
    cluster_size_min = int(counts.min())
    cluster_size_max = int(counts.max())
    unique_cluster_count = int(unique.size)
    no_collapse = bool(
        all_finite
        and effective_rank_value > 1.5
        and unique_cluster_count >= 4
        and cluster_size_max < 0.8 * int(semantic.shape[0])
    )
    return {
        "semantic_all_finite": all_finite,
        "semantic_l2_norm_mean": float(norms.mean().item()),
        "semantic_l2_norm_std": float(
            norms.std(unbiased=False).item()
        ),
        "semantic_effective_rank": effective_rank_value,
        "offdiag_cosine_mean": float(offdiag.mean().item()),
        "offdiag_cosine_std": float(
            offdiag.std(unbiased=False).item()
        ),
        "offdiag_cosine_p95": float(
            torch.quantile(offdiag, 0.95).item()
        ),
        "cluster_size_min": cluster_size_min,
        "cluster_size_max": cluster_size_max,
        "cluster_size_count": [
            int(value) for value in sorted(counts.tolist())
        ],
        "unique_cluster_count": unique_cluster_count,
        "B6_WQ1A1_NO_COLLAPSE_PASS": no_collapse,
    }
