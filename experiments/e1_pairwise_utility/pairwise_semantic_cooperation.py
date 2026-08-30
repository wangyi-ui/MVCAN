"""E1 Information-Utility Guided Local Pair Cooperation.

This module ports only the released GLGC local cross-view graph construction
and positive-pair weighted contrastive mechanism.  MVCAN's native target is
owned by the training entry and is deliberately not represented here.
"""

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F


GLGC_RELEASED_COMMIT = "7550e0c38e422538af03a62542df4bd632fd34d7"
GLGC_GRAPH_TEMPERATURE = 0.07
GLGC_CONTRASTIVE_TEMPERATURE = 0.5
GLGC_ETA = 0.2
EXPECTED_RELIABILITY_SHA256 = (
    "3458e099f8c6e7317e3edbcef15ff629f368735e115fa6aea30e4516be8ac2cc"
)

ARMS = ("BASE", "LWC", "R_LWC", "SHUFFLED_R_LWC")


def _require_tensor(value, name):
    if not torch.is_tensor(value):
        raise TypeError(name + " must be a torch tensor")


def logical_ndarray_sha256(value):
    """Hash dtype, shape, and C-order logical array content."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(size) for size in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def align_semantic_probabilities(q_local, match_matrices):
    """Map MVCAN local cluster probabilities into global coordinates.

    Args:
        q_local: [B,V,K], differentiable local cluster probabilities.
        match_matrices: [V,K,K], frozen Hungarian matrices M.

    Returns:
        q_aligned: [B,V,K], q_local[b,v] @ M[v].T.
        h_sem: [B,V,K], unit-normalized semantic representation.

    M is forcibly detached.  q_aligned and h_sem retain the path back to
    q_local, MVCAN cluster centers, and the encoders.
    """
    _require_tensor(q_local, "q_local")
    _require_tensor(match_matrices, "match_matrices")
    if q_local.ndim != 3:
        raise ValueError("q_local must have shape [B,V,K]")
    if match_matrices.ndim != 3:
        raise ValueError("match_matrices must have shape [V,K,K]")
    batch_size, view_num, cluster_num = q_local.shape
    if tuple(match_matrices.shape) != (view_num, cluster_num, cluster_num):
        raise ValueError("M shape must equal [V,K,K] from q_local")
    if batch_size <= 0 or view_num < 2 or cluster_num < 2:
        raise ValueError("q_local requires B>0, V>=2, and K>=2")
    if not bool(torch.isfinite(q_local).all().item()):
        raise ValueError("q_local must be finite")
    if not bool(torch.isfinite(match_matrices).all().item()):
        raise ValueError("M must be finite")

    # M: [V,K,K], constant/no grad. q_aligned: [B,V,K].
    frozen_m = match_matrices.detach().to(
        device=q_local.device, dtype=q_local.dtype
    )
    q_aligned = torch.einsum("bvk,vjk->bvj", q_local, frozen_m)
    # h_sem: [B,V,K], differentiable through q_aligned -> q_local.
    h_sem = F.normalize(q_aligned, dim=-1)
    if not bool(torch.isfinite(h_sem).all().item()):
        raise RuntimeError("h_sem contains NaN or Inf")
    return q_aligned, h_sem


@torch.no_grad()
def robust_inter_affinity(
    h_sem_detached,
    graph_temperature=GLGC_GRAPH_TEMPERATURE,
    eta=GLGC_ETA,
):
    """Port the released GLGC ``robust_affinity`` noise-path inter graph.

    Args:
        h_sem_detached: [B,V,K], explicitly detached semantic coordinates.

    Returns:
        G_inter: [V,V,B,B], frozen local cross-view graph.

    The loop order intentionally matches released ``util.py``: its second
    stage updates directional entries in place.  No mask branch is migrated.
    """
    _require_tensor(h_sem_detached, "h_sem_detached")
    if h_sem_detached.ndim != 3:
        raise ValueError("h_sem_detached must have shape [B,V,K]")
    batch_size, view_num, cluster_num = h_sem_detached.shape
    if batch_size <= 0 or view_num < 2 or cluster_num < 2:
        raise ValueError("h_sem_detached requires B>0, V>=2, and K>=2")
    graph_temperature = float(graph_temperature)
    eta = float(eta)
    if not math.isfinite(graph_temperature) or graph_temperature <= 0.0:
        raise ValueError("graph_temperature must be positive and finite")
    if not math.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be finite and non-negative")

    # Graph construction is always stop-gradient even if a caller forgets
    # detach(). h_views[v]: [B,K].
    frozen = h_sem_detached.detach()
    h_views = [F.normalize(frozen[:, view_id, :], dim=-1)
               for view_id in range(view_num)]
    # G_inter: [V,V,B,B], no grad.
    graph_inter = torch.zeros(
        (view_num, view_num, batch_size, batch_size),
        device=frozen.device,
        dtype=frozen.dtype,
    )
    diagonal = torch.eye(batch_size, device=frozen.device, dtype=torch.bool)
    for left in range(view_num):
        for right in range(view_num):
            distance = (
                2.0 - 2.0 * (h_views[left] @ h_views[right].t())
            ).clamp(min=0.0)
            graph = torch.exp(-distance / graph_temperature)
            if left == right:
                graph[diagonal] = 1.0
            else:
                graph[diagonal] = (
                    graph[diagonal]
                    / graph.diag().max().clamp_min(1e-7).detach()
                )
            graph = graph / graph.sum(1, keepdim=True).clamp_min(1e-7)
            graph_inter[left, right] = graph

    identity = torch.eye(
        batch_size, device=frozen.device, dtype=frozen.dtype
    )
    for left in range(view_num):
        for right in range(view_num):
            if left != right:
                graph_inter[left, right] = graph_inter[left, right].mm(
                    graph_inter[right, left].t()
                )
                graph_inter[left, right] += eta * identity

    if graph_inter.requires_grad or graph_inter.grad_fn is not None:
        raise RuntimeError("G_inter must be stop-gradient")
    if not bool(torch.isfinite(graph_inter).all().item()):
        raise RuntimeError("G_inter contains NaN or Inf")
    return graph_inter.detach()


def symmetric_pair_graph_evidence(graph_inter, left, right):
    """Return G_i^{uv}: [B] from both released directional graph diagonals."""
    _require_tensor(graph_inter, "graph_inter")
    if graph_inter.ndim != 4:
        raise ValueError("G_inter must have shape [V,V,B,B]")
    view_num_left, view_num_right, batch_left, batch_right = graph_inter.shape
    if view_num_left != view_num_right or batch_left != batch_right:
        raise ValueError("G_inter must have shape [V,V,B,B]")
    left = int(left)
    right = int(right)
    if not (0 <= left < right < view_num_left):
        raise ValueError("pair must satisfy 0 <= left < right < V")
    # g_pair: [B], symmetric evidence for the unordered pair u<->v.
    g_pair = 0.5 * (
        graph_inter[left, right].diag()
        + graph_inter[right, left].diag()
    )
    g_pair = g_pair.detach()
    if not bool(torch.isfinite(g_pair).all().item()):
        raise RuntimeError("g_pair contains NaN or Inf")
    if bool((g_pair < 0.0).any().item()):
        raise RuntimeError("g_pair must be non-negative")
    return g_pair


def reliability_pair_factor(reliability_batch, left, right):
    """Return sqrt(clamp(R_u)*clamp(R_v)): [B], frozen/no grad."""
    _require_tensor(reliability_batch, "reliability_batch")
    if reliability_batch.ndim != 2:
        raise ValueError("R_batch must have shape [B,V]")
    left = int(left)
    right = int(right)
    if not (0 <= left < right < reliability_batch.shape[1]):
        raise ValueError("pair must satisfy 0 <= left < right < V")
    # R_batch: [B,V], frozen/no grad. r_pair: [B], frozen/no grad.
    frozen = reliability_batch.detach().clamp(0.0, 1.0)
    r_pair = torch.sqrt(frozen[:, left] * frozen[:, right]).detach()
    if not bool(torch.isfinite(r_pair).all().item()):
        raise RuntimeError("r_pair contains NaN or Inf")
    return r_pair


def positive_pair_weights(graph_inter, reliability_batch, left, right, arm):
    """Construct frozen positive weights [B] for one preregistered arm."""
    if arm not in ARMS:
        raise ValueError("unknown E1 arm: " + str(arm))
    if arm == "BASE":
        raise ValueError("BASE has no pair branch")
    g_pair = symmetric_pair_graph_evidence(graph_inter, left, right)
    if arm == "LWC":
        positive_weights = g_pair
        r_pair = None
    else:
        r_pair = reliability_pair_factor(reliability_batch, left, right)
        # positive_weights: [B], frozen/no grad.
        positive_weights = (g_pair * r_pair).detach()
    return positive_weights, g_pair, r_pair


def glgc_positive_pair_contrastive_loss(
    h_left,
    h_right,
    positive_weights,
    temperature=GLGC_CONTRASTIVE_TEMPERATURE,
):
    """Released GLGC positive-logit weighting for one unordered view pair.

    h_left/h_right are non-detached [B,K].  The same unordered positive weight
    is duplicated for the two contrastive directions.  Negatives and the
    exclusion mask exactly follow ``Loss.forward_inter_noise``.
    """
    _require_tensor(h_left, "h_left")
    _require_tensor(h_right, "h_right")
    _require_tensor(positive_weights, "positive_weights")
    if h_left.ndim != 2 or h_left.shape != h_right.shape:
        raise ValueError("h_left and h_right must have equal shape [B,K]")
    batch_size = int(h_left.shape[0])
    if batch_size < 2:
        raise ValueError("pair contrastive loss requires B>=2")
    if positive_weights.shape != (batch_size,):
        raise ValueError("positive weights must have shape [B]")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")

    frozen_weights = positive_weights.detach().to(
        device=h_left.device, dtype=h_left.dtype
    )
    if not bool(torch.isfinite(frozen_weights).all().item()):
        raise ValueError("positive weights must be finite")
    if bool((frozen_weights < 0.0).any().item()):
        raise ValueError("positive weights must be non-negative")

    sample_twice = 2 * batch_size
    features = torch.cat((h_left, h_right), dim=0)  # [2B,K]
    similarity = features @ features.t() / temperature  # [2B,2B]
    positive = torch.cat(
        (
            torch.diag(similarity, batch_size),
            torch.diag(similarity, -batch_size),
        ),
        dim=0,
    ).reshape(sample_twice, 1)  # [2B,1]
    # GLGC weights the positive logit itself, not the per-sample CE value.
    positive = positive * torch.cat(
        (frozen_weights, frozen_weights), dim=0
    ).reshape(sample_twice, 1)

    negative_mask = torch.ones(
        (sample_twice, sample_twice),
        dtype=torch.bool,
        device=h_left.device,
    )
    negative_mask.fill_diagonal_(False)
    row_ids = torch.arange(batch_size, device=h_left.device)
    negative_mask[row_ids, batch_size + row_ids] = False
    negative_mask[batch_size + row_ids, row_ids] = False
    negatives = similarity[negative_mask].reshape(sample_twice, -1)
    logits = torch.cat((positive, negatives), dim=1)
    labels = torch.zeros(sample_twice, dtype=torch.long, device=h_left.device)
    loss = F.cross_entropy(logits, labels, reduction="sum") / sample_twice
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("pair contrastive loss is not finite")
    return loss


def pairwise_semantic_cooperation_loss(h_sem, graph_inter, reliability_batch, arm):
    """Sum GLGC-style positive-pair losses over all u<v view pairs."""
    _require_tensor(h_sem, "h_sem")
    if arm not in ARMS or arm == "BASE":
        raise ValueError("pair loss requires LWC, R_LWC, or SHUFFLED_R_LWC")
    if h_sem.ndim != 3:
        raise ValueError("h_sem must have shape [B,V,K]")
    batch_size, view_num, _ = h_sem.shape
    if graph_inter.shape != (view_num, view_num, batch_size, batch_size):
        raise ValueError("G_inter shape mismatch")
    if reliability_batch.shape != (batch_size, view_num):
        raise ValueError("R_batch shape mismatch")

    losses = []
    pair_diagnostics = []
    for left in range(view_num):
        for right in range(left + 1, view_num):
            weights, g_pair, r_pair = positive_pair_weights(
                graph_inter, reliability_batch, left, right, arm
            )
            # h_sem below is intentionally not detached: L_pair -> q_local ->
            # MVCAN cluster layer/encoder. positive weights: [B], no grad.
            loss = glgc_positive_pair_contrastive_loss(
                h_sem[:, left, :], h_sem[:, right, :], weights
            )
            losses.append(loss)
            pair_diagnostics.append({
                "left": left,
                "right": right,
                "g_mean": float(g_pair.mean().item()),
                "r_mean": (
                    None if r_pair is None else float(r_pair.mean().item())
                ),
                "positive_weight_mean": float(weights.mean().item()),
            })
    total = torch.stack(losses).sum()
    if not bool(torch.isfinite(total).item()):
        raise RuntimeError("summed pair loss is not finite")
    return total, pair_diagnostics


def whole_row_reliability_shuffle(reliability, seed=20):
    """Deterministically permute complete R rows; never shuffle views alone."""
    values = np.asarray(reliability)
    if values.ndim != 2 or values.shape[0] <= 1 or values.shape[1] < 2:
        raise ValueError("reliability must have shape [N>=2,V>=2]")
    if not np.isfinite(values).all():
        raise ValueError("reliability must be finite")
    permutation = np.random.RandomState(int(seed)).permutation(values.shape[0])
    shuffled = np.ascontiguousarray(values[permutation])
    per_view_distribution_pass = bool(
        np.array_equal(np.sort(values, axis=0), np.sort(shuffled, axis=0))
    )
    row_multiset_pass = bool(
        sorted(map(bytes, np.ascontiguousarray(values).view(np.uint8).reshape(values.shape[0], -1)))
        == sorted(map(bytes, shuffled.view(np.uint8).reshape(shuffled.shape[0], -1)))
    )
    audit = {
        "seed": int(seed),
        "whole_row_permutation": True,
        "permutation_sha256": logical_ndarray_sha256(permutation),
        "permutation_is_bijection_pass": bool(
            np.array_equal(np.sort(permutation), np.arange(values.shape[0]))
        ),
        "permutation_nonidentity_pass": bool(
            not np.array_equal(permutation, np.arange(values.shape[0]))
        ),
        "R_multiset_audit_pass": row_multiset_pass,
        "per_view_distribution_audit_pass": per_view_distribution_pass,
    }
    if not all(
        audit[key]
        for key in (
            "permutation_is_bijection_pass",
            "permutation_nonidentity_pass",
            "R_multiset_audit_pass",
            "per_view_distribution_audit_pass",
        )
    ):
        raise RuntimeError("whole-row reliability shuffle audit failed")
    return shuffled, permutation, audit
