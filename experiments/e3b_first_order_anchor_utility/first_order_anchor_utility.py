"""First-order label-conditioned anchor relations for E3-B0.

The teacher is built only from the 14 real labeled anchors and the current
batch's unlabeled targets.  Every teacher tensor is frozen; the student
semantic representation remains differentiable.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    GLGC_CONTRASTIVE_TEMPERATURE,
    align_semantic_probabilities,
    robust_inter_affinity,
)


E3B_ARMS = ("LWC_REPLAY", "FO_ANCHOR", "SHUFFLED_FO_ANCHOR")
ANCHOR_NUM = 14
VIEW_NUM = 6
CLASS_NUM = 7
FIRST_ORDER_EPS = 1e-12
STUDENT_TEMPERATURE = GLGC_CONTRASTIVE_TEMPERATURE


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _balanced_anchor_labels(anchor_labels, device):
    labels = torch.as_tensor(
        np.array(anchor_labels, copy=True, dtype=np.int64),
        device=device,
        dtype=torch.long,
    ).detach()
    _require(labels.shape == (ANCHOR_NUM,), "anchor labels must have shape [14]")
    _require(
        torch.equal(
            torch.bincount(labels, minlength=CLASS_NUM),
            torch.full((CLASS_NUM,), 2, device=device, dtype=torch.long),
        ),
        "anchor class counts must be exactly two per class",
    )
    return labels


@torch.no_grad()
def refresh_anchor_semantics(model, anchor_x_views, match_matrices, device):
    """Refresh detached [14,6,7] anchor semantics without changing BN state."""
    if len(anchor_x_views) != VIEW_NUM:
        raise ValueError("anchor_x_views must contain six views")
    if not torch.is_tensor(match_matrices) or tuple(match_matrices.shape) != (
        VIEW_NUM,
        CLASS_NUM,
        CLASS_NUM,
    ):
        raise ValueError("match_matrices must have shape [6,7,7]")

    autoencoders = list(model.autoencoders)
    _require(len(autoencoders) == VIEW_NUM, "MVCAN must contain six autoencoders")
    training_states = [bool(autoencoder.training) for autoencoder in autoencoders]
    bn_before = [
        {
            name: value.detach().cpu().clone()
            for name, value in autoencoder.named_buffers()
            if name.endswith("running_mean")
            or name.endswith("running_var")
            or name.endswith("num_batches_tracked")
        }
        for autoencoder in autoencoders
    ]
    q_views = []
    try:
        for autoencoder in autoencoders:
            autoencoder.eval()
        for view_id, autoencoder in enumerate(autoencoders):
            x_anchor = torch.as_tensor(anchor_x_views[view_id]).to(device)
            _require(
                x_anchor.ndim == 2 and int(x_anchor.shape[0]) == ANCHOR_NUM,
                "every anchor feature view must have shape [14,D_v]",
            )
            _, _, q_local = autoencoder(x_anchor)
            q_views.append(q_local)
    finally:
        for autoencoder, was_training in zip(autoencoders, training_states):
            autoencoder.train(was_training)

    q_anchor_local = torch.stack(q_views, dim=1).detach()
    _, anchor_h = align_semantic_probabilities(
        q_anchor_local, match_matrices.detach()
    )
    anchor_h = F.normalize(anchor_h, p=2, dim=-1).detach()
    bn_after = [
        {
            name: value.detach().cpu().clone()
            for name, value in autoencoder.named_buffers()
            if name.endswith("running_mean")
            or name.endswith("running_var")
            or name.endswith("num_batches_tracked")
        }
        for autoencoder in autoencoders
    ]
    bn_unchanged = all(
        before.keys() == after.keys()
        and all(torch.equal(before[name], after[name]) for name in before)
        for before, after in zip(bn_before, bn_after)
    )
    mode_restored = all(
        bool(autoencoder.training) == was_training
        for autoencoder, was_training in zip(autoencoders, training_states)
    )
    audit = {
        "shape": list(anchor_h.shape),
        "shape_pass": tuple(anchor_h.shape) == (ANCHOR_NUM, VIEW_NUM, CLASS_NUM),
        "finite_pass": bool(torch.isfinite(anchor_h).all().item()),
        "stop_gradient_pass": bool(
            not anchor_h.requires_grad and anchor_h.grad_fn is None
        ),
        "BN_running_state_unchanged_pass": bool(bn_unchanged),
        "BN_training_mode_restored_pass": bool(mode_restored),
        "temporary_eval_used": True,
    }
    _require(all(value for key, value in audit.items() if key.endswith("_pass")),
             "anchor semantic refresh audit failed")
    return anchor_h, audit


def unlabeled_target_subset(h_sem, batch_ids, labeled_ids):
    """Return only current-batch unlabeled student rows and their mask."""
    if not torch.is_tensor(h_sem) or h_sem.ndim != 3:
        raise ValueError("h_sem must have shape [B,6,7]")
    ids = torch.as_tensor(batch_ids, device=h_sem.device, dtype=torch.long)
    anchors = torch.as_tensor(
        np.array(labeled_ids, copy=True, dtype=np.int64),
        device=h_sem.device,
        dtype=torch.long,
    )
    _require(ids.shape == (h_sem.shape[0],), "batch IDs must have shape [B]")
    _require(anchors.shape == (ANCHOR_NUM,), "labeled IDs must have shape [14]")
    unlabeled_mask = torch.all(ids[:, None] != anchors[None, :], dim=1).detach()
    h_unlab = h_sem[unlabeled_mask]
    _require(int(h_unlab.shape[0]) > 0, "FO branch has no unlabeled target")
    audit = {
        "batch_size": int(h_sem.shape[0]),
        "unlabeled_target_count": int(unlabeled_mask.sum().item()),
        "labeled_batch_count": int((~unlabeled_mask).sum().item()),
        "only_unlabeled_targets_pass": bool(
            torch.all(
                ids[unlabeled_mask, None] != anchors[None, :]
            ).item()
        ),
        "labeled_duplicate_in_graph_target": False,
        "student_non_detached_pass": bool(h_unlab.requires_grad),
    }
    return h_unlab, unlabeled_mask, audit


@torch.no_grad()
def build_symmetric_anchor_graph(anchor_h, h_unlab):
    """Build symmetric anchor/target evidence from E1's directional graph."""
    if not torch.is_tensor(anchor_h) or tuple(anchor_h.shape) != (
        ANCHOR_NUM,
        VIEW_NUM,
        CLASS_NUM,
    ):
        raise ValueError("anchor_h must have shape [14,6,7]")
    if not torch.is_tensor(h_unlab) or h_unlab.ndim != 3:
        raise ValueError("h_unlab must have shape [B_u,6,7]")
    unlabeled_num, view_num, class_num = h_unlab.shape
    _require(
        unlabeled_num > 0 and (view_num, class_num) == (VIEW_NUM, CLASS_NUM),
        "h_unlab boundary mismatch",
    )
    combined = torch.cat((anchor_h.detach(), h_unlab.detach()), dim=0)
    graph_all = robust_inter_affinity(combined).detach()
    # Forward: G[u,v,l,14+i]. Reverse: G[v,u,14+i,l].
    graph_forward = graph_all[:, :, :ANCHOR_NUM, ANCHOR_NUM:]
    graph_reverse = graph_all.transpose(0, 1)[
        :, :, ANCHOR_NUM:, :ANCHOR_NUM
    ].permute(0, 1, 3, 2)
    graph_symmetric = (0.5 * (graph_forward + graph_reverse)).detach()
    off_diagonal = ~torch.eye(VIEW_NUM, device=graph_symmetric.device,
                              dtype=torch.bool)
    graph_symmetric = torch.where(
        off_diagonal[:, :, None, None],
        graph_symmetric,
        torch.zeros_like(graph_symmetric),
    ).detach()
    expected_shape = (VIEW_NUM, VIEW_NUM, ANCHOR_NUM, unlabeled_num)
    _require(tuple(graph_symmetric.shape) == expected_shape,
             "symmetric anchor graph shape mismatch")
    _require(torch.isfinite(graph_symmetric).all().item(),
             "symmetric anchor graph is not finite")
    _require(torch.all(graph_symmetric >= 0.0).item(),
             "symmetric anchor graph must be nonnegative")
    audit = {
        "combined_shape": list(combined.shape),
        "G_all_shape": list(graph_all.shape),
        "G_anchor_sym_shape": list(graph_symmetric.shape),
        "combined_contains_only_14_anchors_plus_unlabeled_targets": True,
        "symmetric_extraction_exact_pass": bool(
            torch.equal(
                graph_symmetric[off_diagonal],
                (0.5 * (graph_forward + graph_reverse))[off_diagonal],
            )
        ),
        "same_view_zero_pass": bool(
            torch.count_nonzero(graph_symmetric[~off_diagonal]).item() == 0
        ),
        "finite_pass": True,
        "nonnegative_pass": True,
        "stop_gradient_pass": bool(
            not graph_symmetric.requires_grad and graph_symmetric.grad_fn is None
        ),
    }
    return graph_all, graph_symmetric, audit


@torch.no_grad()
def anchor_induced_class_evidence(
    graph_anchor_sym,
    active_anchor_labels,
    eps=FIRST_ORDER_EPS,
):
    """Average symmetric G over u!=v and the two anchors of each class."""
    if not torch.is_tensor(graph_anchor_sym) or graph_anchor_sym.ndim != 4:
        raise ValueError("G_anchor_sym must have shape [6,6,14,B_u]")
    source_views, target_views, anchor_num, unlabeled_num = graph_anchor_sym.shape
    _require(
        (source_views, target_views, anchor_num)
        == (VIEW_NUM, VIEW_NUM, ANCHOR_NUM),
        "G_anchor_sym boundary mismatch",
    )
    eps = float(eps)
    _require(math.isfinite(eps) and eps > 0.0, "eps must be positive")
    labels = _balanced_anchor_labels(
        active_anchor_labels, graph_anchor_sym.device
    )
    frozen_graph = graph_anchor_sym.detach()
    evidence = torch.empty(
        (unlabeled_num, VIEW_NUM, CLASS_NUM),
        device=frozen_graph.device,
        dtype=frozen_graph.dtype,
    )
    for target_view in range(VIEW_NUM):
        source_mask = torch.arange(
            VIEW_NUM, device=frozen_graph.device
        ) != target_view
        for class_id in range(CLASS_NUM):
            class_mask = labels == class_id
            selected = frozen_graph[
                source_mask, target_view, :, :
            ][:, class_mask, :]  # [V-1,2,B_u]
            _require(tuple(selected.shape) == (VIEW_NUM - 1, 2, unlabeled_num),
                     "cross-view class evidence selection mismatch")
            evidence[:, target_view, class_id] = selected.mean(dim=(0, 1))
    anchor_probability = (evidence + eps) / (
        evidence + eps
    ).sum(dim=-1, keepdim=True)
    row_sums = anchor_probability.sum(dim=-1)
    audit = {
        "shape": list(anchor_probability.shape),
        "shape_pass": tuple(anchor_probability.shape)
        == (unlabeled_num, VIEW_NUM, CLASS_NUM),
        "finite_pass": bool(torch.isfinite(anchor_probability).all().item()),
        "nonnegative_pass": bool(torch.all(anchor_probability >= 0.0).item()),
        "row_sum_one_pass": bool(torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=1e-6
        )),
        "stop_gradient_pass": bool(
            not anchor_probability.requires_grad
            and anchor_probability.grad_fn is None
        ),
        "class_counts": torch.bincount(labels, minlength=CLASS_NUM).cpu().tolist(),
        "source_views_per_target": VIEW_NUM - 1,
        "same_view_source_excluded": True,
        "argmax_used": False,
        "temperature_used": False,
        "sharpening_used": False,
        "threshold_used": False,
    }
    _require(all(value for key, value in audit.items() if key.endswith("_pass")),
             "P_anchor validity audit failed")
    return anchor_probability.detach(), evidence.detach(), audit


@torch.no_grad()
def first_order_label_factor(anchor_probability, active_anchor_labels):
    """Gather P_anchor[i,v,y_l] directly for every real labeled anchor."""
    if not torch.is_tensor(anchor_probability) or anchor_probability.ndim != 3:
        raise ValueError("P_anchor must have shape [B_u,6,7]")
    unlabeled_num, view_num, class_num = anchor_probability.shape
    _require((view_num, class_num) == (VIEW_NUM, CLASS_NUM),
             "P_anchor boundary mismatch")
    labels = _balanced_anchor_labels(
        active_anchor_labels, anchor_probability.device
    )
    # [B_u,V,L] is a direct class-index lookup; no probability products exist.
    gather_index = labels.view(1, 1, ANCHOR_NUM).expand(
        unlabeled_num, VIEW_NUM, ANCHOR_NUM
    )
    gathered = torch.gather(
        anchor_probability.detach(), dim=2, index=gather_index
    )  # [B_u,V,L]
    factor_target_view = gathered.permute(1, 2, 0).contiguous()  # [V,L,B_u]
    factor = factor_target_view.unsqueeze(0).expand(
        VIEW_NUM, -1, -1, -1
    ).clone()
    off_diagonal = ~torch.eye(VIEW_NUM, device=factor.device, dtype=torch.bool)
    factor = torch.where(
        off_diagonal[:, :, None, None], factor, torch.zeros_like(factor)
    ).detach()
    active_values = factor[off_diagonal]
    audit = {
        "F_first_shape": list(factor.shape),
        "F_first_min": float(active_values.min().item()),
        "F_first_max": float(active_values.max().item()),
        "F_first_mean": float(active_values.mean().item()),
        "F_first_std": float(active_values.std(unbiased=False).item()),
        "finite_pass": bool(torch.isfinite(factor).all().item()),
        "range_zero_one_pass": bool(
            torch.all((factor >= 0.0) & (factor <= 1.0)).item()
        ),
        "gather_exact_pass": bool(torch.equal(
            factor_target_view, gathered.permute(1, 2, 0)
        )),
        "stop_gradient_pass": bool(
            not factor.requires_grad and factor.grad_fn is None
        ),
        "no_pairwise_probability_dot_product": True,
    }
    _require(all(value for key, value in audit.items() if key.endswith("_pass")),
             "first-order factor audit failed")
    return factor, audit


@torch.no_grad()
def build_first_order_targets(
    graph_anchor_sym,
    first_order_factor,
    eps=FIRST_ORDER_EPS,
):
    """Construct U_FO=G_sym*F_first and normalize over 14 anchors."""
    if not torch.is_tensor(graph_anchor_sym) or graph_anchor_sym.ndim != 4:
        raise ValueError("G_anchor_sym must have shape [6,6,14,B_u]")
    if not torch.is_tensor(first_order_factor):
        raise ValueError("F_first must be a tensor")
    _require(first_order_factor.shape == graph_anchor_sym.shape,
             "G_anchor_sym and F_first shape mismatch")
    eps = float(eps)
    _require(math.isfinite(eps) and eps > 0.0, "eps must be positive")
    utility = (
        graph_anchor_sym.detach() * first_order_factor.detach()
    ).detach()  # [V,V,L,B_u]
    target_layout = utility.permute(0, 1, 3, 2).contiguous()  # [V,V,B_u,L]
    row_mass = target_layout.sum(dim=-1)
    off_diagonal = ~torch.eye(VIEW_NUM, device=utility.device, dtype=torch.bool)
    valid_rows = (off_diagonal[:, :, None] & (row_mass > eps)).detach()
    target = torch.zeros_like(target_layout)
    target[valid_rows] = (
        target_layout[valid_rows]
        / row_mass[valid_rows].unsqueeze(-1)
    )
    target = target.detach()
    valid_sums = target.sum(dim=-1)[valid_rows]
    audit = {
        "U_FO_shape": list(utility.shape),
        "T_anchor_shape": list(target.shape),
        "U_FO_finite_pass": bool(torch.isfinite(utility).all().item()),
        "U_FO_nonnegative_pass": bool(torch.all(utility >= 0.0).item()),
        "U_FO_stop_gradient_pass": bool(
            not utility.requires_grad and utility.grad_fn is None
        ),
        "T_anchor_stop_gradient_pass": bool(
            not target.requires_grad and target.grad_fn is None
        ),
        "T_valid_row_sum_one_pass": bool(
            valid_sums.numel() > 0 and torch.allclose(
                valid_sums, torch.ones_like(valid_sums), atol=1e-6, rtol=1e-6
            )
        ),
        "valid_row_count": int(valid_rows.sum().item()),
        "row_mass_skip_enabled": True,
        "threshold_used": False,
        "topk_used": False,
        "reliability_used": False,
    }
    _require(
        audit["U_FO_finite_pass"]
        and audit["U_FO_nonnegative_pass"]
        and audit["U_FO_stop_gradient_pass"]
        and audit["T_anchor_stop_gradient_pass"]
        and audit["T_valid_row_sum_one_pass"],
        "first-order utility/target audit failed",
    )
    return utility, target, valid_rows, audit


def first_order_anchor_relation_loss(
    h_unlab,
    anchor_h,
    anchor_target,
    valid_rows,
    temperature=STUDENT_TEMPERATURE,
):
    """Sum 15 unordered-pair losses, averaging the two directions per pair."""
    if not torch.is_tensor(h_unlab) or h_unlab.ndim != 3:
        raise ValueError("h_unlab must have shape [B_u,6,7]")
    if not torch.is_tensor(anchor_h) or tuple(anchor_h.shape) != (
        ANCHOR_NUM,
        VIEW_NUM,
        CLASS_NUM,
    ):
        raise ValueError("anchor_h must have shape [14,6,7]")
    unlabeled_num, view_num, class_num = h_unlab.shape
    _require((view_num, class_num) == (VIEW_NUM, CLASS_NUM),
             "h_unlab boundary mismatch")
    _require(
        tuple(anchor_target.shape)
        == (VIEW_NUM, VIEW_NUM, unlabeled_num, ANCHOR_NUM)
        and tuple(valid_rows.shape) == (VIEW_NUM, VIEW_NUM, unlabeled_num),
        "T_anchor or valid row shape mismatch",
    )
    temperature = float(temperature)
    _require(
        temperature == GLGC_CONTRASTIVE_TEMPERATURE == 0.5,
        "E3-B0 student temperature must equal E1 contrastive temperature 0.5",
    )
    student = F.normalize(h_unlab, p=2, dim=-1)
    teacher = F.normalize(anchor_h.detach(), p=2, dim=-1).detach()
    pair_losses = []
    direction_records = []

    def directional_loss(source_view, target_view):
        row_ids = torch.nonzero(
            valid_rows[source_view, target_view], as_tuple=False
        ).flatten()
        _require(row_ids.numel() > 0,
                 "every E3-B0 ordered view direction needs a valid row")
        logits = (
            student[row_ids, target_view]
            @ teacher[:, source_view].t()
        ) / temperature
        log_probability = F.log_softmax(logits, dim=-1)
        frozen_target = anchor_target[
            source_view, target_view, row_ids
        ].detach()
        value = -(frozen_target * log_probability).sum(dim=-1).mean()
        _require(torch.isfinite(value).item() and float(value.detach()) > 0.0,
                 "directional FO loss must be finite and positive")
        direction_records.append({
            "source_view": int(source_view),
            "target_view": int(target_view),
            "valid_row_count": int(row_ids.numel()),
            "loss": float(value.detach().item()),
        })
        return value

    for left in range(VIEW_NUM):
        for right in range(left + 1, VIEW_NUM):
            forward = directional_loss(left, right)
            reverse = directional_loss(right, left)
            pair_losses.append(0.5 * (forward + reverse))
    loss = torch.stack(pair_losses).sum()
    audit = {
        "unordered_pair_count": len(pair_losses),
        "direction_count": len(direction_records),
        "aggregation": "sum_15_unordered_pairs",
        "direction_pairing": "mean_two_directions",
        "directional_row_aggregation": "mean_valid_target_rows_once",
        "temperature": temperature,
        "loss_finite_pass": bool(torch.isfinite(loss).item()),
        "loss_positive_pass": float(loss.detach().item()) > 0.0,
        "student_requires_grad_pass": bool(loss.requires_grad),
        "teacher_stop_gradient_pass": bool(
            not teacher.requires_grad and teacher.grad_fn is None
        ),
        "target_stop_gradient_pass": bool(
            not anchor_target.requires_grad and anchor_target.grad_fn is None
        ),
        "direction_records": direction_records,
        "hard_label_CE_used": False,
        "class_CE_used": False,
    }
    _require(
        audit["unordered_pair_count"] == 15
        and audit["direction_count"] == 30
        and audit["loss_finite_pass"]
        and audit["loss_positive_pass"],
        "E3-B0 loss aggregation audit failed",
    )
    return loss, audit

