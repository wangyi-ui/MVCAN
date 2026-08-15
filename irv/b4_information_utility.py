"""Frozen Information Utility and weighted semantic admission for B4-A1a."""

import hashlib
import math
import struct

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from irv.semantic_loss import uniform_cross_view_infonce


UTILITY_EPS = 1e-12


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compute_information_utility(predictability_scores):
    """Compute the preregistered per-view average-rank utility.

    Args:
        predictability_scores: Finite scores T with shape [N,V].

    Returns:
        U with shape [N,V], where U[:,v] = (rank(T[:,v])-1)/(N-1).
        Tensor inputs produce a detached tensor on the original device;
        array inputs produce a float64 NumPy array.
    """
    values = _as_numpy(predictability_scores).astype(np.float64, copy=False)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("predictability_scores must have shape [N>=2,V>=2]")
    if not np.isfinite(values).all():
        raise ValueError("predictability_scores must contain only finite values")
    sample_num, view_num = values.shape
    utility = np.empty_like(values)
    for view_id in range(view_num):
        ranks = rankdata(values[:, view_id], method="average")
        utility[:, view_id] = (ranks - 1.0) / float(sample_num - 1)
    if torch.is_tensor(predictability_scores):
        output_dtype = (
            predictability_scores.dtype
            if predictability_scores.is_floating_point()
            else torch.float64
        )
        return torch.from_numpy(utility).to(
            device=predictability_scores.device,
            dtype=output_dtype,
        ).detach()
    return utility


def unordered_view_pairs(view_num):
    """Return the fixed unordered pair enumeration for V views."""
    view_num = int(view_num)
    if view_num < 2:
        raise ValueError("view_num must be at least two")
    return [
        (view_v, view_w)
        for view_v in range(view_num)
        for view_w in range(view_v + 1, view_num)
    ]


def pair_admission_weight(utility, view_v, view_w):
    """Return detached omega_i^(vw)=U_i^v U_i^w with shape [N]."""
    if not torch.is_tensor(utility) or utility.ndim != 2:
        raise ValueError("utility must be a tensor with shape [N,V]")
    view_v = int(view_v)
    view_w = int(view_w)
    if view_v == view_w or not (
        0 <= view_v < utility.shape[1] and 0 <= view_w < utility.shape[1]
    ):
        raise ValueError("view IDs must be distinct and valid")
    # utility: [N,V]
    # pair_weight: [N]
    pair_weight = (utility[:, view_v] * utility[:, view_w]).detach()
    if not bool(torch.isfinite(pair_weight).all().item()):
        raise ValueError("pair weights must be finite")
    if bool((pair_weight < 0).any().item()):
        raise ValueError("pair weights must be non-negative")
    return pair_weight


def normalized_weighted_mean(per_sample_loss, weight, eps=UTILITY_EPS):
    """Compute sum(weight*loss)/(sum(weight)+eps) without scale confounding."""
    if (
        not torch.is_tensor(per_sample_loss)
        or not torch.is_tensor(weight)
        or per_sample_loss.ndim != 1
        or weight.ndim != 1
        or per_sample_loss.shape != weight.shape
    ):
        raise ValueError("per_sample_loss and weight must have equal shape [N]")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    detached_weight = weight.detach().to(
        device=per_sample_loss.device,
        dtype=per_sample_loss.dtype,
    )
    if not bool(torch.isfinite(per_sample_loss).all().item()):
        raise ValueError("per-sample losses must be finite")
    if not bool(torch.isfinite(detached_weight).all().item()):
        raise ValueError("weights must be finite")
    if bool((detached_weight < 0).any().item()):
        raise ValueError("weights must be non-negative")
    return torch.sum(detached_weight * per_sample_loss) / (
        torch.sum(detached_weight) + eps
    )


def weighted_symmetric_semantic_infonce(
    semantic_views,
    utility,
    temperature,
    eps=UTILITY_EPS,
):
    """Apply normalized utility admission to B3's symmetric InfoNCE.

    semantic_views is List[V], each [N,D]. utility is [N,V]. For each
    unordered pair, logits and reverse logits are [N,N], CE is [N], and
    pair_weight is [N].
    """
    if not isinstance(semantic_views, (list, tuple)) or len(semantic_views) < 2:
        raise ValueError("semantic_views must contain at least two views")
    first = semantic_views[0]
    if not torch.is_tensor(first) or first.ndim != 2:
        raise ValueError("each semantic view must have shape [N,D]")
    sample_num, semantic_dim = first.shape
    for semantic_view in semantic_views:
        if (
            not torch.is_tensor(semantic_view)
            or semantic_view.shape != first.shape
            or semantic_view.device != first.device
            or not bool(torch.isfinite(semantic_view).all().item())
        ):
            raise ValueError("all semantic views must be equal finite [N,D] tensors")
    if not torch.is_tensor(utility) or utility.shape != (
        sample_num,
        len(semantic_views),
    ):
        raise ValueError("utility must have shape [N,V]")
    if utility.device != first.device:
        raise ValueError("utility and semantic views must share a device")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")

    # target: [N]
    target = torch.arange(sample_num, device=first.device, dtype=torch.long)
    pair_losses = []
    all_pair_weights = []
    for view_v, view_w in unordered_view_pairs(len(semantic_views)):
        # S_v/S_w: [N,D]
        s_v = semantic_views[view_v]
        s_w = semantic_views[view_w]
        # logits_vw/logits_wv: [N,N]
        logits_vw = torch.matmul(s_v, s_w.T) / temperature
        logits_wv = logits_vw.T
        # CE forward/backward: [N]
        ce_vw = F.cross_entropy(logits_vw, target, reduction="none")
        ce_wv = F.cross_entropy(logits_wv, target, reduction="none")
        # pair_weight: [N]
        pair_weight = pair_admission_weight(utility, view_v, view_w)
        forward_loss = normalized_weighted_mean(ce_vw, pair_weight, eps=eps)
        backward_loss = normalized_weighted_mean(ce_wv, pair_weight, eps=eps)
        pair_losses.append(0.5 * (forward_loss + backward_loss))
        all_pair_weights.append(pair_weight)
    loss = torch.stack(pair_losses).mean()
    stacked_weights = torch.stack(all_pair_weights)
    diagnostics = {
        "pair_count": len(pair_losses),
        "batch_size": int(sample_num),
        "semantic_dim": int(semantic_dim),
        "temperature": temperature,
        "eps": float(eps),
        "weight_normalization": "sum(weight*CE)/(sum(weight)+eps)",
        "pair_weight_mean": float(stacked_weights.mean().item()),
        "pair_weight_min": float(stacked_weights.min().item()),
        "pair_weight_max": float(stacked_weights.max().item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
    }
    return loss, diagnostics


def uniform_equivalence_max_abs_error(semantic_views, temperature):
    """Compare all-one weighted loss with frozen B3 uniform InfoNCE."""
    first = semantic_views[0]
    utility = torch.ones(
        (first.shape[0], len(semantic_views)),
        device=first.device,
        dtype=first.dtype,
    )
    weighted, _ = weighted_symmetric_semantic_infonce(
        semantic_views,
        utility,
        temperature,
    )
    original, _ = uniform_cross_view_infonce(semantic_views, temperature)
    return float(torch.abs(weighted - original).detach().item())


def _summary(values, percentiles):
    values = np.asarray(values, dtype=np.float64)
    result = {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }
    for percentile in percentiles:
        result["p" + str(int(percentile))] = float(
            np.percentile(values, percentile)
        )
    return result


def information_utility_diagnostics(predictability_scores, utility):
    """Summarize T, U, and all 10 pair-weight distributions."""
    t_values = _as_numpy(predictability_scores).astype(np.float64, copy=False)
    u_values = _as_numpy(utility).astype(np.float64, copy=False)
    if t_values.shape != u_values.shape or t_values.ndim != 2:
        raise ValueError("T and U must have equal shape [N,V]")
    t_per_view = []
    u_per_view = []
    for view_id in range(t_values.shape[1]):
        t_per_view.append({"view_id": view_id, **_summary(t_values[:, view_id], ())})
        u_per_view.append({
            "view_id": view_id,
            **_summary(u_values[:, view_id], (10, 25, 50, 75, 90)),
        })
    pair_values = []
    for view_v, view_w in unordered_view_pairs(t_values.shape[1]):
        pair_values.append(u_values[:, view_v] * u_values[:, view_w])
    pair_values = np.concatenate(pair_values)
    return {
        "t_shape": list(t_values.shape),
        "u_shape": list(u_values.shape),
        "t_per_view": t_per_view,
        "u_per_view": u_per_view,
        "u_global": _summary(u_values.reshape(-1), (10, 25, 50, 75, 90)),
        "pair_weights": _summary(pair_values, (10, 50, 90)),
        "pair_count": len(unordered_view_pairs(t_values.shape[1])),
    }


def tensor_sha256(value):
    """Hash one tensor/array including dtype, shape, and contiguous bytes."""
    array = np.ascontiguousarray(_as_numpy(value))
    digest = hashlib.sha256()
    for component in (
        str(array.dtype).encode("ascii"),
        ",".join(str(int(size)) for size in array.shape).encode("ascii"),
        array.tobytes(order="C"),
    ):
        digest.update(struct.pack(">Q", len(component)))
        digest.update(component)
    return digest.hexdigest()


def tensor_view_list_sha256(values):
    """Hash an ordered list of view tensors without conflating view order."""
    digest = hashlib.sha256()
    for view_id, value in enumerate(values):
        component = (str(view_id) + ":" + tensor_sha256(value)).encode("ascii")
        digest.update(struct.pack(">Q", len(component)))
        digest.update(component)
    return digest.hexdigest()


def utility_reliability_audit(utility, corruption_mask):
    """Post-hoc noisy-condition audit; never used to construct utility."""
    values = _as_numpy(utility).astype(np.float64, copy=False)
    corrupted = np.asarray(corruption_mask, dtype=bool)
    if values.shape != corrupted.shape or values.ndim != 2:
        raise ValueError("utility and corruption_mask must have equal [N,V] shape")
    clean = ~corrupted
    clean_indicator = clean.astype(np.int64).reshape(-1)
    flat = values.reshape(-1)
    result = {
        "clean_mean": float(np.mean(values[clean])),
        "corrupted_mean": float(np.mean(values[corrupted])),
        "gap": float(np.mean(values[clean]) - np.mean(values[corrupted])),
        "auc": float(roc_auc_score(clean_indicator, flat)),
    }
    order = np.argsort(-flat, kind="mergesort")
    for percentage in (20, 40, 60):
        selected_count = int(round(flat.size * percentage / 100.0))
        selected = order[:selected_count]
        clean_count = int(clean_indicator[selected].sum())
        result["p" + str(percentage)] = {
            "precision": float(clean_count / selected_count),
            "selected_count": selected_count,
            "clean_count": clean_count,
            "corrupted_count": selected_count - clean_count,
        }
    return result
