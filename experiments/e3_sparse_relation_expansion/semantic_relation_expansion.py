"""Sparse-anchor expansion of cross-sample, cross-view semantic relations."""

import hashlib
import math
import struct

import numpy as np
import torch
import torch.nn.functional as F

from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    GLGC_GRAPH_TEMPERATURE,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    robust_inter_affinity,
)


E3_ARMS = ("LWC_REPLAY", "SEM_REL", "SHUFFLED_LABEL_REL")
ANCHOR_NUM = 14
VIEW_NUM = 6
CLASS_NUM = 7
RELATION_EPS = 1e-12
RELATION_TEMPERATURE = GLGC_GRAPH_TEMPERATURE


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def logical_array_sha256(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    for component in (
        str(array.dtype).encode("ascii"),
        ",".join(str(int(size)) for size in array.shape).encode("ascii"),
        array.tobytes(order="C"),
    ):
        digest.update(struct.pack(">Q", len(component)))
        digest.update(component)
    return digest.hexdigest()


@torch.no_grad()
def build_anchor_batch_graph(anchor_h, h_batch):
    """Run frozen E1 robust affinity on anchors concatenated with a batch.

    Args:
        anchor_h: detached [L=14,V=6,D=7].
        h_batch: [B,V=6,D=7]; it is forcibly detached for this branch.

    Returns:
        G_all [V,V,L+B,L+B], G_anchor_batch [V,V,L,B], and
        G_batch_batch [V,V,B,B], all stop-gradient.
    """
    if not torch.is_tensor(anchor_h) or anchor_h.ndim != 3:
        raise ValueError("anchor_h must have shape [14,6,7]")
    if not torch.is_tensor(h_batch) or h_batch.ndim != 3:
        raise ValueError("h_batch must have shape [B,6,7]")
    anchor_num, view_num, semantic_dim = anchor_h.shape
    batch_size, batch_view_num, batch_semantic_dim = h_batch.shape
    _require(
        (anchor_num, view_num, semantic_dim)
        == (ANCHOR_NUM, VIEW_NUM, CLASS_NUM),
        "anchor_h boundary must be [14,6,7]",
    )
    _require(
        batch_size > 1
        and (batch_view_num, batch_semantic_dim) == (view_num, semantic_dim),
        "h_batch boundary mismatch",
    )
    frozen_anchor = anchor_h.detach()
    frozen_batch = h_batch.detach()
    # E1 accepts [sample,view,dim]. Conceptual h_graph is the transpose
    # [V,14+B,7]; combined below is its E1-native [14+B,V,7] layout.
    combined = torch.cat((frozen_anchor, frozen_batch), dim=0)
    graph_all = robust_inter_affinity(combined)  # [V,V,14+B,14+B]
    graph_anchor_batch = graph_all[
        :, :, :anchor_num, anchor_num:
    ].detach()
    graph_batch_batch = graph_all[
        :, :, anchor_num:, anchor_num:
    ].detach()
    _require(
        tuple(graph_all.shape)
        == (view_num, view_num, anchor_num + batch_size, anchor_num + batch_size)
        and tuple(graph_anchor_batch.shape)
        == (view_num, view_num, anchor_num, batch_size)
        and tuple(graph_batch_batch.shape)
        == (view_num, view_num, batch_size, batch_size),
        "anchor/batch graph extraction shape mismatch",
    )
    _require(
        not graph_all.requires_grad
        and graph_all.grad_fn is None
        and not graph_anchor_batch.requires_grad
        and graph_anchor_batch.grad_fn is None
        and not graph_batch_batch.requires_grad
        and graph_batch_batch.grad_fn is None,
        "all structural graph tensors must be stop-gradient",
    )
    return graph_all.detach(), graph_anchor_batch, graph_batch_batch


@torch.no_grad()
def anchor_induced_class_evidence(
    graph_anchor_batch,
    anchor_labels,
    class_num=CLASS_NUM,
    eps=RELATION_EPS,
):
    """Compute P_anchor[i,v,c] from all source views and class-c anchors."""
    if not torch.is_tensor(graph_anchor_batch) or graph_anchor_batch.ndim != 4:
        raise ValueError("G_anchor_batch must have shape [V,V,14,B]")
    source_views, target_views, anchor_num, batch_size = (
        graph_anchor_batch.shape
    )
    class_num = int(class_num)
    eps = float(eps)
    _require(
        (source_views, target_views, anchor_num)
        == (VIEW_NUM, VIEW_NUM, ANCHOR_NUM),
        "G_anchor_batch boundary mismatch",
    )
    _require(
        class_num == CLASS_NUM and math.isfinite(eps) and eps > 0.0,
        "class_num or eps is invalid",
    )
    labels = torch.as_tensor(
        np.array(anchor_labels, copy=True, dtype=np.int64),
        device=graph_anchor_batch.device,
        dtype=torch.long,
    ).detach()
    _require(labels.shape == (ANCHOR_NUM,), "anchor labels must be [14]")
    counts = torch.bincount(labels, minlength=class_num)
    _require(
        torch.equal(counts, torch.full_like(counts, 2)),
        "anchor class counts must be exactly two per class",
    )
    frozen_graph = graph_anchor_batch.detach()
    # evidence: [B,V,K]. Each entry averages U source views x 2 anchors.
    evidence = torch.empty(
        (batch_size, target_views, class_num),
        device=frozen_graph.device,
        dtype=frozen_graph.dtype,
    )
    for target_view in range(target_views):
        for class_id in range(class_num):
            class_anchor_mask = labels == class_id
            selected = frozen_graph[
                :, target_view, class_anchor_mask, :
            ]  # [source V,2,B]
            _require(
                tuple(selected.shape) == (source_views, 2, batch_size),
                "class anchor evidence selection mismatch",
            )
            evidence[:, target_view, class_id] = selected.mean(dim=(0, 1))
    anchor_prob = (evidence + eps) / (
        evidence + eps
    ).sum(dim=-1, keepdim=True)
    row_sums = anchor_prob.sum(dim=-1)
    validity = {
        "shape_pass": tuple(anchor_prob.shape)
        == (batch_size, target_views, class_num),
        "finite_pass": bool(torch.isfinite(anchor_prob).all().item()),
        "nonnegative_pass": bool(torch.all(anchor_prob >= 0.0).item()),
        "row_sum_one_pass": bool(
            torch.allclose(
                row_sums,
                torch.ones_like(row_sums),
                atol=1e-6,
                rtol=1e-6,
            )
        ),
        "stop_gradient_pass": bool(
            not anchor_prob.requires_grad and anchor_prob.grad_fn is None
        ),
    }
    _require(all(validity.values()), "P_anchor validity audit failed")
    audit = {
        **validity,
        "shape": list(anchor_prob.shape),
        "class_counts": counts.cpu().tolist(),
        "minimum": float(anchor_prob.min().item()),
        "maximum": float(anchor_prob.max().item()),
        "max_row_sum_error": float((row_sums - 1.0).abs().max().item()),
        "argmax_used": False,
        "temperature_used": False,
        "threshold_used": False,
        "sharpening_used": False,
    }
    return anchor_prob.detach(), evidence.detach(), audit


@torch.no_grad()
def cross_sample_semantic_similarity(anchor_prob):
    """Compute S_sem[u,v,i,j]=dot(P[i,u],P[j,v]) for u != v."""
    if not torch.is_tensor(anchor_prob) or anchor_prob.ndim != 3:
        raise ValueError("P_anchor must have shape [B,6,7]")
    batch_size, view_num, class_num = anchor_prob.shape
    _require(
        (view_num, class_num) == (VIEW_NUM, CLASS_NUM),
        "P_anchor boundary mismatch",
    )
    frozen_prob = anchor_prob.detach()
    # semantic_similarity: [V,V,B,B].
    semantic_similarity = torch.einsum(
        "iuc,jvc->uvij", frozen_prob, frozen_prob
    )
    view_off_diagonal = ~torch.eye(
        view_num, device=frozen_prob.device, dtype=torch.bool
    )
    semantic_similarity = semantic_similarity * view_off_diagonal[:, :, None, None]
    _require(
        torch.isfinite(semantic_similarity).all().item()
        and torch.all(
            (semantic_similarity >= 0.0) & (semantic_similarity <= 1.0 + 1e-6)
        ).item(),
        "S_sem must be finite in [0,1]",
    )
    semantic_similarity = semantic_similarity.clamp(0.0, 1.0).detach()
    _require(
        not semantic_similarity.requires_grad
        and semantic_similarity.grad_fn is None,
        "S_sem must be stop-gradient",
    )
    return semantic_similarity


@torch.no_grad()
def build_relation_targets(
    graph_batch_batch,
    semantic_similarity,
    unlabeled_batch_mask,
    eps=RELATION_EPS,
):
    """Build frozen U_rel and row-normalized T_rel on unlabeled i!=j only."""
    if not torch.is_tensor(graph_batch_batch) or graph_batch_batch.ndim != 4:
        raise ValueError("G_batch_batch must have shape [V,V,B,B]")
    if not torch.is_tensor(semantic_similarity):
        raise ValueError("S_sem must be a tensor")
    view_num_left, view_num_right, batch_rows, batch_columns = (
        graph_batch_batch.shape
    )
    _require(
        view_num_left == view_num_right == VIEW_NUM
        and batch_rows == batch_columns,
        "G_batch_batch boundary mismatch",
    )
    _require(
        tuple(semantic_similarity.shape)
        == (VIEW_NUM, VIEW_NUM, batch_rows, batch_rows),
        "S_sem shape mismatch",
    )
    eps = float(eps)
    _require(math.isfinite(eps) and eps > 0.0, "eps must be positive")
    unlabeled = torch.as_tensor(
        unlabeled_batch_mask,
        device=graph_batch_batch.device,
        dtype=torch.bool,
    ).detach()
    _require(unlabeled.shape == (batch_rows,), "unlabeled mask must be [B]")
    view_off_diagonal = ~torch.eye(
        VIEW_NUM, device=unlabeled.device, dtype=torch.bool
    )
    sample_off_diagonal = ~torch.eye(
        batch_rows, device=unlabeled.device, dtype=torch.bool
    )
    relation_mask = (
        view_off_diagonal[:, :, None, None]
        & sample_off_diagonal[None, None, :, :]
        & unlabeled[None, None, :, None]
        & unlabeled[None, None, None, :]
    )
    # U_rel contains no reliability term: structural G times semantic S only.
    relation_utility = (
        graph_batch_batch.detach() * semantic_similarity.detach()
    )
    relation_utility = torch.where(
        relation_mask, relation_utility, torch.zeros_like(relation_utility)
    ).detach()
    row_mass = relation_utility.sum(dim=-1)  # [V,V,B]
    valid_rows = (
        view_off_diagonal[:, :, None]
        & unlabeled[None, None, :]
        & (row_mass > eps)
    )
    relation_target = torch.zeros_like(relation_utility)
    relation_target[valid_rows] = (
        relation_utility[valid_rows]
        / row_mass[valid_rows].unsqueeze(-1)
    )
    relation_target = relation_target.detach()
    valid_row_sums = relation_target.sum(dim=-1)[valid_rows]
    valid_relation_count = int(relation_mask.sum().item())
    validity = {
        "U_rel_finite_pass": bool(torch.isfinite(relation_utility).all().item()),
        "U_rel_stop_gradient_pass": bool(
            not relation_utility.requires_grad and relation_utility.grad_fn is None
        ),
        "T_rel_stop_gradient_pass": bool(
            not relation_target.requires_grad and relation_target.grad_fn is None
        ),
        "sample_diagonal_excluded_pass": bool(
            torch.count_nonzero(
                relation_utility.diagonal(dim1=-2, dim2=-1)
            ).item()
            == 0
        ),
        "labeled_target_rows_excluded_pass": bool(
            torch.count_nonzero(relation_utility[:, :, ~unlabeled, :]).item()
            == 0
        ),
        "labeled_target_columns_excluded_pass": bool(
            torch.count_nonzero(relation_utility[:, :, :, ~unlabeled]).item()
            == 0
        ),
        "T_valid_row_sum_one_pass": bool(
            valid_row_sums.numel() > 0
            and torch.allclose(
                valid_row_sums,
                torch.ones_like(valid_row_sums),
                atol=1e-6,
                rtol=1e-6,
            )
        ),
        "valid_row_count_positive_pass": int(valid_rows.sum().item()) > 0,
        "valid_off_diagonal_relation_count_positive_pass": valid_relation_count > 0,
    }
    _require(all(validity.values()), "cross-sample relation target audit failed")
    audit = {
        **validity,
        "valid_row_count": int(valid_rows.sum().item()),
        "valid_off_diagonal_relation_count": valid_relation_count,
        "unlabeled_batch_count": int(unlabeled.sum().item()),
        "labeled_batch_count": int((~unlabeled).sum().item()),
        "T_valid_row_sum_max_error": float(
            (valid_row_sums - 1.0).abs().max().item()
        ),
        "U_rel_min": float(relation_utility.min().item()),
        "U_rel_max": float(relation_utility.max().item()),
        "threshold_used": False,
        "topk_used": False,
        "reliability_used": False,
    }
    return (
        relation_utility.detach(),
        relation_target,
        valid_rows.detach(),
        relation_mask.detach(),
        audit,
    )


def cross_sample_relation_loss(
    h_sem,
    relation_target,
    valid_rows,
    unlabeled_batch_mask,
    temperature=RELATION_TEMPERATURE,
):
    """Soft-target CE from frozen T_rel to non-detached student relations."""
    if not torch.is_tensor(h_sem) or h_sem.ndim != 3:
        raise ValueError("h_sem must have shape [B,6,7]")
    batch_size, view_num, semantic_dim = h_sem.shape
    _require(
        (view_num, semantic_dim) == (VIEW_NUM, CLASS_NUM),
        "student h_sem boundary mismatch",
    )
    _require(
        tuple(relation_target.shape)
        == (view_num, view_num, batch_size, batch_size)
        and tuple(valid_rows.shape) == (view_num, view_num, batch_size),
        "relation target/valid-row shape mismatch",
    )
    temperature = float(temperature)
    _require(
        temperature == RELATION_TEMPERATURE,
        "E3 relation temperature is frozen at E1 graph temperature 0.07",
    )
    unlabeled = torch.as_tensor(
        unlabeled_batch_mask,
        device=h_sem.device,
        dtype=torch.bool,
    ).detach()
    _require(unlabeled.shape == (batch_size,), "unlabeled mask must be [B]")
    normalized_h = F.normalize(h_sem, p=2, dim=-1)
    row_losses = []
    consumed_rows = 0
    for source_view in range(view_num):
        for target_view in range(view_num):
            if source_view == target_view:
                continue
            row_ids = torch.nonzero(
                valid_rows[source_view, target_view], as_tuple=False
            ).flatten()
            if row_ids.numel() == 0:
                continue
            logits = (
                normalized_h[row_ids, source_view]
                @ normalized_h[:, target_view].t()
            ) / temperature  # [valid rows,B]
            candidate_mask = unlabeled[None, :].expand(row_ids.numel(), -1).clone()
            candidate_mask[
                torch.arange(row_ids.numel(), device=h_sem.device), row_ids
            ] = False
            _require(
                torch.all(candidate_mask.sum(dim=1) > 0).item(),
                "valid relation row has no unlabeled off-diagonal candidate",
            )
            masked_logits = logits.masked_fill(~candidate_mask, -torch.inf)
            log_probability = F.log_softmax(masked_logits, dim=-1)
            safe_log_probability = torch.where(
                candidate_mask,
                log_probability,
                torch.zeros_like(log_probability),
            )
            frozen_target = relation_target[
                source_view, target_view, row_ids
            ].detach()
            row_losses.append(
                -(frozen_target * safe_log_probability).sum(dim=-1)
            )
            consumed_rows += int(row_ids.numel())
    _require(row_losses, "no valid E3 relation rows reached student loss")
    per_row_loss = torch.cat(row_losses, dim=0)
    loss = per_row_loss.mean()
    _require(
        torch.isfinite(loss).item() and float(loss.detach().item()) > 0.0,
        "L_rel must be finite and positive",
    )
    audit = {
        "valid_row_count_consumed": consumed_rows,
        "per_row_loss_count": int(per_row_loss.numel()),
        "loss_finite_pass": bool(torch.isfinite(loss).item()),
        "loss_positive_pass": float(loss.detach().item()) > 0.0,
        "student_requires_grad_pass": bool(loss.requires_grad),
        "target_stop_gradient_pass": bool(
            not relation_target.requires_grad and relation_target.grad_fn is None
        ),
        "temperature": temperature,
        "hard_pseudo_label_used": False,
        "class_CE_used": False,
    }
    return loss, audit


def deterministic_anchor_label_shuffle(anchor_labels, seed=20):
    """Shuffle only anchor->class assignments with preregistered hard audits."""
    labels = np.asarray(anchor_labels, dtype=np.int64)
    if labels.shape != (ANCHOR_NUM,):
        raise ValueError("anchor labels must have shape [14]")
    original_counts = np.bincount(labels, minlength=CLASS_NUM)
    _require(
        np.array_equal(original_counts, np.full(CLASS_NUM, 2, dtype=np.int64)),
        "true anchor labels must have exactly two samples per class",
    )
    permutation = np.random.RandomState(int(seed)).permutation(ANCHOR_NUM)
    shuffled = np.ascontiguousarray(labels[permutation], dtype=np.int64)
    shuffled_counts = np.bincount(shuffled, minlength=CLASS_NUM)
    changed_count = int(np.count_nonzero(shuffled != labels))
    contingency = np.zeros((CLASS_NUM, CLASS_NUM), dtype=np.int64)
    for true_label, shuffled_label in zip(labels, shuffled):
        contingency[int(true_label), int(shuffled_label)] += 1
    split_original_class_pass = any(
        np.count_nonzero(contingency[class_id]) >= 2
        for class_id in range(CLASS_NUM)
    )
    audit = {
        "seed": int(seed),
        "true_anchor_labels": labels.tolist(),
        "shuffled_anchor_labels": shuffled.tolist(),
        "changed_count": changed_count,
        "class_counts_before": original_counts.tolist(),
        "class_counts_after": shuffled_counts.tolist(),
        "class_counts_preserved_pass": bool(
            np.array_equal(original_counts, shuffled_counts)
        ),
        "at_least_10_of_14_changed_pass": changed_count >= 10,
        "not_global_class_permutation_pass": bool(split_original_class_pass),
        "contingency_matrix": contingency.tolist(),
        "permutation": permutation.tolist(),
        "permutation_sha256": logical_array_sha256(permutation),
        "labels_sha256": logical_array_sha256(shuffled),
        "labeled_ids_shuffled": False,
        "anchor_representations_shuffled": False,
        "unlabeled_GT_used": False,
    }
    _require(
        audit["class_counts_preserved_pass"]
        and audit["at_least_10_of_14_changed_pass"]
        and audit["not_global_class_permutation_pass"],
        "deterministic shuffled-label control failed hard audits",
    )
    return shuffled, permutation, audit
