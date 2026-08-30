"""E3-B0 controlled continuation and read-only admission diagnostics."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.e1_pairwise_utility import train_e1_pairwise_utility as e1_train
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    align_semantic_probabilities,
    pairwise_semantic_cooperation_loss,
    robust_inter_affinity,
)
from experiments.e2_sparse_semantic_utility.train_e2_sparse_semantic_utility import (
    audit_lwc_replay_identity,
    load_sparse_training_labels,
    refresh_native_target_and_semantics,
)
from experiments.e3_sparse_relation_expansion.semantic_relation_expansion import (
    deterministic_anchor_label_shuffle,
)
from experiments.e3b_first_order_anchor_utility.first_order_anchor_utility import (
    E3B_ARMS,
    anchor_induced_class_evidence,
    build_first_order_targets,
    build_symmetric_anchor_graph,
    first_order_anchor_relation_loss,
    first_order_label_factor,
    refresh_anchor_semantics,
    unlabeled_target_subset,
)
from irv.b3_audit import hash_backbone


STAGE = "E3-B0"
DATASET = e1_train.DATASET
SEED = e1_train.SEED
SAMPLE_NUM = e1_train.SAMPLE_NUM
VIEW_NUM = e1_train.VIEW_NUM
CLASS_NUM = e1_train.CLUSTER_NUM
LAMBDA1 = e1_train.LAMBDA1
TARGET_REFRESH_INTERVAL = e1_train.TARGET_REFRESH_INTERVAL
EXPECTED_INITIAL_MODEL_HASH = (
    "299916ecb6d62b97b4e75f6a2ec98fd4f309766f0dfa30d98c958c2ca2058c9b"
)
OLD_E3A0_T_ROW_L1_MEAN = 0.02538328245282173
OLD_E3A0_COS_TRUE_SHUFFLE = 0.9996820688247681
OLD_E3A0_RELATIVE_GRAD_DIFFERENCE = 0.031482522521154924
OLD_E3A0_COS_REL_LWC = -0.9555885195732117
DIAGNOSTIC_UNLABELED_NUM = 256

DEFAULT_SPLIT_DIR = REPOSITORY_ROOT / "outputs/b7_sparse_supervision/b7a0_seed20"
DEFAULT_E1_LWC_SMOKE_DIR = (
    REPOSITORY_ROOT / "outputs/e1_pairwise_utility/lwc_2ep_seed20"
)
DEFAULT_E1_LWC_FORMAL_DIR = (
    REPOSITORY_ROOT / "outputs/e1_pairwise_utility/lwc_100ep_seed20"
)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/e3b_first_order_anchor_utility"
DEFAULT_DIAGNOSTIC_AUDIT = DEFAULT_OUTPUT_ROOT / "diagnostic_seed20/e3b0_diagnostic.json"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _display(path):
    value = Path(path).resolve()
    try:
        return str(value.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(value)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def unused_lwc_api_placeholder():
    """Shape-only E1 LWC input; the LWC arm never reads its values."""
    placeholder = np.zeros((SAMPLE_NUM, VIEW_NUM), dtype=np.float64)
    placeholder.setflags(write=False)
    return placeholder


def _anchor_raw_features(views, labeled_ids):
    anchor_views = [
        np.ascontiguousarray(view[np.asarray(labeled_ids, dtype=np.int64)])
        for view in views
    ]
    _require(
        all(values.shape[0] == 14 for values in anchor_views),
        "anchor raw feature extraction did not yield 14 samples per view",
    )
    return anchor_views


def _active_labels(arm, true_anchor_labels):
    if arm == "SHUFFLED_FO_ANCHOR":
        return deterministic_anchor_label_shuffle(
            true_anchor_labels, seed=SEED
        )[::2]
    labels = np.array(true_anchor_labels, copy=True, dtype=np.int64)
    return labels, {
        "seed": None,
        "true_anchor_labels": labels.tolist(),
        "shuffled_anchor_labels": None,
        "changed_count": 0,
        "control_active": False,
        "class_counts_after": np.bincount(labels, minlength=CLASS_NUM).tolist(),
        "unlabeled_GT_used": False,
    }


def _first_order_branch(h_sem, batch_ids, labeled_ids, anchor_h, active_labels):
    h_unlab, unlabeled_mask, subset_audit = unlabeled_target_subset(
        h_sem, batch_ids, labeled_ids
    )
    graph_all, graph_symmetric, graph_audit = build_symmetric_anchor_graph(
        anchor_h, h_unlab
    )
    anchor_probability, class_evidence, probability_audit = (
        anchor_induced_class_evidence(graph_symmetric, active_labels)
    )
    first_order_factor, factor_audit = first_order_label_factor(
        anchor_probability, active_labels
    )
    utility, anchor_target, valid_rows, target_audit = build_first_order_targets(
        graph_symmetric, first_order_factor
    )
    loss, loss_audit = first_order_anchor_relation_loss(
        h_unlab, anchor_h, anchor_target, valid_rows
    )
    tensors = {
        "h_unlab": h_unlab,
        "unlabeled_mask": unlabeled_mask,
        "G_all": graph_all,
        "G_anchor_sym": graph_symmetric,
        "A": class_evidence,
        "P_anchor": anchor_probability,
        "F_first": first_order_factor,
        "U_FO": utility,
        "T_anchor": anchor_target,
        "valid_rows": valid_rows,
    }
    audit = {
        "unlabeled_subset": subset_audit,
        "symmetric_graph": graph_audit,
        "anchor_probability": probability_audit,
        "first_order_factor": factor_audit,
        "utility_target": target_audit,
        "loss": loss_audit,
    }
    return loss, tensors, audit


def _fo_gradient_audit(loss, q_local, model):
    targets = [q_local]
    names = ["q_local"]
    for view_id, autoencoder in enumerate(model.autoencoders):
        targets.extend([
            autoencoder._cluster_layer,
            next(autoencoder._encoder.parameters()),
        ])
        names.extend([
            "cluster_layer_view" + str(view_id),
            "encoder_view" + str(view_id),
        ])
    gradients = torch.autograd.grad(
        loss, targets, retain_graph=True, allow_unused=True
    )
    audit = {}
    for name, gradient in zip(names, gradients):
        finite = gradient is not None and bool(torch.isfinite(gradient).all())
        nonzero = finite and bool(torch.count_nonzero(gradient).item() > 0)
        audit[name + "_gradient_finite_pass"] = bool(finite)
        audit[name + "_gradient_nonzero_pass"] = bool(nonzero)
    audit["FO_branch_gradient_exists_pass"] = bool(all(audit.values()))
    return audit


def _new_fo_aggregate():
    return {
        "batch_count": 0,
        "only_unlabeled_targets_pass": True,
        "no_labeled_duplicate_in_graph_target_pass": True,
        "G_symmetric_extraction_exact_pass": True,
        "G_stop_gradient_pass": True,
        "P_anchor_validity_pass": True,
        "F_first_gather_exact_pass": True,
        "F_first_range_pass": True,
        "U_FO_validity_pass": True,
        "T_anchor_validity_pass": True,
        "loss_aggregation_pass": True,
        "L_FO_finite_positive_pass": True,
        "valid_row_count_total": 0,
        "F_first_min": None,
        "F_first_max": None,
        "F_first_mean_sum": 0.0,
        "F_first_std_sum": 0.0,
    }


def _update_fo_aggregate(aggregate, audit):
    subset = audit["unlabeled_subset"]
    graph = audit["symmetric_graph"]
    probability = audit["anchor_probability"]
    factor = audit["first_order_factor"]
    target = audit["utility_target"]
    loss = audit["loss"]
    aggregate["batch_count"] += 1
    aggregate["only_unlabeled_targets_pass"] &= subset[
        "only_unlabeled_targets_pass"
    ]
    aggregate["no_labeled_duplicate_in_graph_target_pass"] &= not subset[
        "labeled_duplicate_in_graph_target"
    ]
    aggregate["G_symmetric_extraction_exact_pass"] &= graph[
        "symmetric_extraction_exact_pass"
    ]
    aggregate["G_stop_gradient_pass"] &= graph["stop_gradient_pass"]
    aggregate["P_anchor_validity_pass"] &= all(
        probability[key]
        for key in (
            "shape_pass", "finite_pass", "nonnegative_pass",
            "row_sum_one_pass", "stop_gradient_pass",
        )
    )
    aggregate["F_first_gather_exact_pass"] &= factor["gather_exact_pass"]
    aggregate["F_first_range_pass"] &= (
        factor["finite_pass"] and factor["range_zero_one_pass"]
    )
    aggregate["U_FO_validity_pass"] &= (
        target["U_FO_finite_pass"]
        and target["U_FO_nonnegative_pass"]
        and target["U_FO_stop_gradient_pass"]
    )
    aggregate["T_anchor_validity_pass"] &= (
        target["T_valid_row_sum_one_pass"]
        and target["T_anchor_stop_gradient_pass"]
    )
    aggregate["loss_aggregation_pass"] &= (
        loss["unordered_pair_count"] == 15
        and loss["direction_count"] == 30
        and loss["aggregation"] == "sum_15_unordered_pairs"
        and loss["direction_pairing"] == "mean_two_directions"
        and loss["temperature"] == 0.5
    )
    aggregate["L_FO_finite_positive_pass"] &= (
        loss["loss_finite_pass"] and loss["loss_positive_pass"]
    )
    aggregate["valid_row_count_total"] += target["valid_row_count"]
    for key, operation in (("F_first_min", min), ("F_first_max", max)):
        current = factor[key]
        aggregate[key] = current if aggregate[key] is None else operation(
            aggregate[key], current
        )
    aggregate["F_first_mean_sum"] += factor["F_first_mean"]
    aggregate["F_first_std_sum"] += factor["F_first_std"]


def train_first_order_continuation(
    arm,
    epochs,
    batch_size,
    model,
    optimizers,
    views,
    sample_ids,
    labeled_ids,
    true_anchor_labels,
    device,
):
    """Train FO arms while preserving MVCAN and the frozen E1 LWC carrier."""
    _require(arm in ("FO_ANCHOR", "SHUFFLED_FO_ANCHOR"),
             "first-order continuation received an invalid arm")
    _require(int(epochs) in (2, 100), "E3-B0 permits only 2 or 100 epochs")
    _require(int(batch_size) == 256, "E3-B0 batch size must remain 256")
    active_labels, shuffle_audit = _active_labels(arm, true_anchor_labels)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    view_weights = [1.0] * VIEW_NUM
    full_views = [torch.from_numpy(view) for view in views]
    id_tensor = torch.from_numpy(np.asarray(sample_ids, dtype=np.int64))
    anchor_x_views = _anchor_raw_features(views, labeled_ids)
    lwc_placeholder = torch.zeros((batch_size, VIEW_NUM), device=device)
    p_all = None
    matches = None
    native_target_refresh_count = 0
    anchor_refresh_audits = []
    fo_aggregate = _new_fo_aggregate()
    fo_gradient_audit = None
    lwc_gradient_audit = None
    epoch_records = []
    loss_finite_pass = True

    for epoch in range(int(epochs)):
        if epoch % TARGET_REFRESH_INTERVAL == 0:
            p_all, matches, _, view_weights, _ = refresh_native_target_and_semantics(
                model, full_views, view_weights, device
            )
            native_target_refresh_count += 1
        _require(p_all is not None and matches is not None,
                 "native target or Match matrices are missing")
        # Independent teacher freshness: exactly once at every epoch start.
        anchor_h, refresh_audit = refresh_anchor_semantics(
            model, anchor_x_views, matches, device
        )
        refresh_audit["epoch"] = epoch + 1
        anchor_refresh_audits.append(refresh_audit)

        dataset = torch.utils.data.TensorDataset(*full_views, p_all, id_tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=True,
            drop_last=False,
            generator=generator,
            num_workers=0,
        )
        visited = []
        native_sum = lwc_sum = fo_sum = total_sum = 0.0
        batch_count = 0
        for packed in loader:
            x_views = [packed[index].to(device) for index in range(VIEW_NUM)]
            p_batch = packed[VIEW_NUM].to(device)
            batch_ids = packed[VIEW_NUM + 1]
            visited.append(batch_ids.numpy().astype(np.int64, copy=False))
            for optimizer in optimizers:
                optimizer.zero_grad()
            reconstructions = []
            q_views = []
            for view_id in range(VIEW_NUM):
                x_hat, _, q_local_view = model.autoencoders[view_id](
                    x_views[view_id]
                )
                reconstructions.append(x_hat)
                q_views.append(q_local_view)
            q_local = torch.stack(q_views, dim=1)
            native_losses, _ = e1_train.native_mvcan_losses(
                x_views, reconstructions, q_views, p_batch, matches
            )
            native_total = torch.stack(native_losses).sum()
            q_aligned, h_sem = align_semantic_probabilities(q_local, matches)
            _require(q_aligned.requires_grad, "aligned q lost its student gradient")
            lwc_graph = robust_inter_affinity(h_sem.detach())
            lwc_loss, _ = pairwise_semantic_cooperation_loss(
                h_sem,
                lwc_graph,
                lwc_placeholder[: int(q_local.shape[0])],
                "LWC",
            )
            fo_loss, fo_tensors, fo_audit = _first_order_branch(
                h_sem, batch_ids, labeled_ids, anchor_h, active_labels
            )
            if fo_gradient_audit is None:
                fo_gradient_audit = _fo_gradient_audit(fo_loss, q_local, model)
            if lwc_gradient_audit is None:
                lwc_gradient_audit = e1_train._pair_gradient_audit(
                    lwc_loss, q_local, model
                )
            _update_fo_aggregate(fo_aggregate, fo_audit)
            for name in (
                "G_all", "G_anchor_sym", "A", "P_anchor",
                "F_first", "U_FO", "T_anchor",
            ):
                value = fo_tensors[name]
                _require(not value.requires_grad and value.grad_fn is None,
                         name + " must remain detached")
            total_loss = native_total + LAMBDA1 * lwc_loss + LAMBDA1 * fo_loss
            finite = all(bool(torch.isfinite(value).item()) for value in (
                native_total, lwc_loss, fo_loss, total_loss
            ))
            loss_finite_pass = bool(loss_finite_pass and finite)
            _require(finite, "non-finite E3-B0 total loss")
            total_loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            native_sum += float(native_total.detach().item())
            lwc_sum += float(lwc_loss.detach().item())
            fo_sum += float(fo_loss.detach().item())
            total_sum += float(total_loss.detach().item())
            batch_count += 1

        coverage = e1_train._coverage_audit(visited)
        _require(coverage["all_1400_sample_ids_exactly_once_pass"],
                 "epoch did not cover all 1400 native/LWC sample IDs")
        epoch_records.append({
            "epoch": epoch + 1,
            "batch_count": batch_count,
            "native_loss_mean": native_sum / batch_count,
            "L_LWC_mean": lwc_sum / batch_count,
            "L_FO_mean": fo_sum / batch_count,
            "total_loss_mean": total_sum / batch_count,
            **coverage,
        })

    _, final_matches, final_predictions, _ = e1_train.refresh_native_target(
        model, full_views, view_weights, device
    )
    fo_aggregate["F_first_mean"] = (
        fo_aggregate.pop("F_first_mean_sum") / fo_aggregate["batch_count"]
    )
    fo_aggregate["F_first_std_mean"] = (
        fo_aggregate.pop("F_first_std_sum") / fo_aggregate["batch_count"]
    )
    return final_predictions, {
        "epoch_records": epoch_records,
        "native_target_refresh_count": native_target_refresh_count,
        "anchor_refresh_count": len(anchor_refresh_audits),
        "anchor_refresh_count_equals_epochs_pass": len(anchor_refresh_audits)
        == int(epochs),
        "anchor_refresh_audits": anchor_refresh_audits,
        "BN_state_restored_all_epochs_pass": all(
            item["BN_running_state_unchanged_pass"]
            and item["BN_training_mode_restored_pass"]
            for item in anchor_refresh_audits
        ),
        "FO_audit": fo_aggregate,
        "FO_gradient_audit": fo_gradient_audit,
        "LWC_gradient_audit": lwc_gradient_audit,
        "label_shuffle_audit": shuffle_audit,
        "active_anchor_labels": active_labels.tolist(),
        "P_all_definition": "MVCAN target_distribution(new_P(latent_fusion, centers))",
        "P_all_rewritten": False,
        "M_no_grad_pass": bool(
            not final_matches.requires_grad and final_matches.grad_fn is None
        ),
        "loss_finite_pass": loss_finite_pass,
        "all_epochs_exact_sample_coverage_pass": all(
            record["all_1400_sample_ids_exactly_once_pass"]
            for record in epoch_records
        ),
        "sample_order_sha256_per_epoch": [
            record["sample_order_sha256"] for record in epoch_records
        ],
        "D2_reliability_loaded": False,
        "R_used": False,
        "direct_CE_used": False,
        "pseudo_label_used": False,
        "no_pairwise_probability_dot_product": True,
        "threshold_used": False,
        "topk_used": False,
        "epoch_selection_used": False,
    }


def _gradient_vector(loss, model):
    parameters = [
        parameter
        for autoencoder in model.autoencoders
        for parameter in autoencoder.parameters()
        if parameter.requires_grad
    ]
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    return torch.cat([
        torch.zeros_like(parameter).reshape(-1)
        if gradient is None else gradient.reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]).detach()


def _cosine(left, right):
    denominator = left.norm() * right.norm()
    _require(float(denominator.item()) > 0.0, "diagnostic gradient norm is zero")
    return float(torch.dot(left, right).div(denominator).item())


def _row_l1_statistics(true_target, shuffled_target, true_valid, shuffled_valid):
    common = true_valid & shuffled_valid
    _require(bool(common.any().item()), "no common valid diagnostic target rows")
    values = (true_target - shuffled_target).abs().sum(dim=-1)[common]
    return {
        "mean": float(values.mean().item()),
        "p50": float(torch.quantile(values, 0.5).item()),
        "max": float(values.max().item()),
        "common_valid_row_count": int(common.sum().item()),
    }


def run_diagnostic_only(
    model,
    views,
    labeled_ids,
    true_anchor_labels,
    unlabeled_ids,
    device,
):
    """Read-only initial-model diagnostic on the first 256 unlabeled IDs."""
    full_views = [torch.from_numpy(view) for view in views]
    view_weights = [1.0] * VIEW_NUM
    p_all, matches, _, _, _ = refresh_native_target_and_semantics(
        model, full_views, view_weights, device
    )
    anchor_h, anchor_refresh_audit = refresh_anchor_semantics(
        model, _anchor_raw_features(views, labeled_ids), matches, device
    )
    diagnostic_ids = np.asarray(
        unlabeled_ids[:DIAGNOSTIC_UNLABELED_NUM], dtype=np.int64
    )
    _require(diagnostic_ids.shape == (DIAGNOSTIC_UNLABELED_NUM,),
             "diagnostic requires the first 256 unlabeled IDs")
    x_views = [full_views[index][diagnostic_ids].to(device)
               for index in range(VIEW_NUM)]
    p_batch = p_all[diagnostic_ids].to(device)
    batch_ids = torch.from_numpy(diagnostic_ids)
    reconstructions = []
    q_views = []
    for view_id in range(VIEW_NUM):
        x_hat, _, q_view = model.autoencoders[view_id](x_views[view_id])
        reconstructions.append(x_hat)
        q_views.append(q_view)
    q_local = torch.stack(q_views, dim=1)
    native_losses, _ = e1_train.native_mvcan_losses(
        x_views, reconstructions, q_views, p_batch, matches
    )
    native_loss = torch.stack(native_losses).sum()
    _, h_sem = align_semantic_probabilities(q_local, matches)
    lwc_graph = robust_inter_affinity(h_sem.detach())
    lwc_loss, _ = pairwise_semantic_cooperation_loss(
        h_sem, lwc_graph,
        torch.zeros((DIAGNOSTIC_UNLABELED_NUM, VIEW_NUM), device=device),
        "LWC",
    )
    shuffled_labels, _, shuffle_audit = deterministic_anchor_label_shuffle(
        true_anchor_labels, seed=SEED
    )
    true_loss, true_tensors, true_audit = _first_order_branch(
        h_sem, batch_ids, labeled_ids, anchor_h, true_anchor_labels
    )
    shuffled_loss, shuffled_tensors, shuffled_audit = _first_order_branch(
        h_sem, batch_ids, labeled_ids, anchor_h, shuffled_labels
    )
    grad_true = _gradient_vector(true_loss, model)
    grad_shuffle = _gradient_vector(shuffled_loss, model)
    grad_lwc = _gradient_vector(lwc_loss, model)
    grad_native = _gradient_vector(native_loss, model)
    true_norm = float(grad_true.norm().item())
    shuffled_norm = float(grad_shuffle.norm().item())
    lwc_norm = float(grad_lwc.norm().item())
    native_norm = float(grad_native.norm().item())
    _require(all(math.isfinite(value) and value > 0.0 for value in (
        true_norm, shuffled_norm, lwc_norm, native_norm
    )), "diagnostic gradients must be finite and nonzero")
    row_l1 = _row_l1_statistics(
        true_tensors["T_anchor"], shuffled_tensors["T_anchor"],
        true_tensors["valid_rows"], shuffled_tensors["valid_rows"],
    )
    relative_gradient_difference = float(
        (grad_true - grad_shuffle).norm().div(grad_true.norm()).item()
    )
    cos_true_shuffle = _cosine(grad_true, grad_shuffle)
    result = {
        "stage": STAGE,
        "mode": "diagnostic-only",
        "seed": SEED,
        "initial_D1_model_only": True,
        "optimizer_step_called": False,
        "training_performed": False,
        "model_saved": False,
        "full_unlabeled_GT_loaded": False,
        "diagnostic_unlabeled_ids": diagnostic_ids.tolist(),
        "diagnostic_unlabeled_count": DIAGNOSTIC_UNLABELED_NUM,
        "anchor_refresh_audit": anchor_refresh_audit,
        "label_shuffle_audit": shuffle_audit,
        "P_true_vs_shuffle_L1_mean": float(
            (true_tensors["P_anchor"] - shuffled_tensors["P_anchor"])
            .abs().sum(dim=-1).mean().item()
        ),
        "F_true_vs_shuffle_abs_mean": float(
            (true_tensors["F_first"] - shuffled_tensors["F_first"])
            .abs()[~torch.eye(VIEW_NUM, device=device, dtype=torch.bool)]
            .mean().item()
        ),
        "F_true_vs_shuffle_abs_max": float(
            (true_tensors["F_first"] - shuffled_tensors["F_first"])
            .abs()[~torch.eye(VIEW_NUM, device=device, dtype=torch.bool)]
            .max().item()
        ),
        "U_true_vs_shuffle_relative_L1": float(
            (true_tensors["U_FO"] - shuffled_tensors["U_FO"]).abs().sum()
            .div(true_tensors["U_FO"].abs().sum().clamp_min(1e-12)).item()
        ),
        "T_true_vs_shuffle_row_L1_mean": row_l1["mean"],
        "T_true_vs_shuffle_row_L1_p50": row_l1["p50"],
        "T_true_vs_shuffle_row_L1_max": row_l1["max"],
        "T_common_valid_row_count": row_l1["common_valid_row_count"],
        "L_FO_true": float(true_loss.detach().item()),
        "L_FO_shuffle": float(shuffled_loss.detach().item()),
        "grad_FO_true_norm": true_norm,
        "grad_FO_shuffle_norm": shuffled_norm,
        "grad_LWC_norm": lwc_norm,
        "grad_native_norm": native_norm,
        "cos_grad_true_shuffle": cos_true_shuffle,
        "cos_grad_true_LWC": _cosine(grad_true, grad_lwc),
        "cos_grad_true_native": _cosine(grad_true, grad_native),
        "relative_true_shuffle_grad_difference": relative_gradient_difference,
        "effective_FO_over_LWC_gradient_ratio": true_norm / lwc_norm,
        "raw_branch_gradient_comparison": True,
        "true_branch_audit": true_audit,
        "shuffled_branch_audit": shuffled_audit,
        "no_pairwise_probability_dot_product": True,
        "old_E3A0_reference": {
            "OLD_E3A0_T_ROW_L1_MEAN": OLD_E3A0_T_ROW_L1_MEAN,
            "OLD_E3A0_COS_TRUE_SHUFFLE": OLD_E3A0_COS_TRUE_SHUFFLE,
            "OLD_E3A0_RELATIVE_GRAD_DIFFERENCE": (
                OLD_E3A0_RELATIVE_GRAD_DIFFERENCE
            ),
            "OLD_E3A0_COS_REL_LWC": OLD_E3A0_COS_REL_LWC,
        },
    }
    result["signal_retention_vs_E3A0_pass"] = bool(
        result["T_true_vs_shuffle_row_L1_mean"] > OLD_E3A0_T_ROW_L1_MEAN
        and result["cos_grad_true_shuffle"] < OLD_E3A0_COS_TRUE_SHUFFLE
        and result["relative_true_shuffle_grad_difference"]
        > OLD_E3A0_RELATIVE_GRAD_DIFFERENCE
    )
    return result


def _formal_gate(args):
    if args.epochs != 100:
        return
    _require(args.arm in ("FO_ANCHOR", "SHUFFLED_FO_ANCHOR"),
             "formal E3-B0 runs only FO_ANCHOR and SHUFFLED_FO_ANCHOR")
    diagnostic = _read_json(_resolve(args.diagnostic_audit_path))
    _require(diagnostic.get("signal_retention_vs_E3A0_pass") is True,
             "formal blocked: diagnostic signal-retention gate did not pass")
    for name, path in (
        ("LWC_REPLAY", args.lwc_smoke_audit_path),
        ("FO_ANCHOR", args.fo_smoke_audit_path),
        ("SHUFFLED_FO_ANCHOR", args.shuffled_smoke_audit_path),
    ):
        audit = _read_json(_resolve(path))
        _require(audit.get("E3B0_ENGINEERING_PASS") is True,
                 "formal blocked: " + name + " 2ep engineering smoke failed")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="E3-B0 first-order anchor utility; seed20 only"
    )
    parser.add_argument("--arm", choices=E3B_ARMS)
    parser.add_argument("--epochs", type=int, choices=(2, 100))
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=str(e1_train.DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--feature-path", default=str(e1_train.DEFAULT_FEATURE_PATH))
    parser.add_argument("--feature-audit-path", default=str(e1_train.DEFAULT_FEATURE_AUDIT_PATH))
    parser.add_argument("--glgc-repository", default=str(e1_train.DEFAULT_GLGC_REPOSITORY))
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--label-path", default=str(e1_train.DEFAULT_LABEL_PATH))
    parser.add_argument("--e1-lwc-smoke-reference-dir", default=str(DEFAULT_E1_LWC_SMOKE_DIR))
    parser.add_argument("--diagnostic-audit-path", default=str(DEFAULT_DIAGNOSTIC_AUDIT))
    parser.add_argument("--lwc-smoke-audit-path", default=str(
        DEFAULT_OUTPUT_ROOT / "lwc_replay_2ep_seed20/e3b0_audit.json"
    ))
    parser.add_argument("--fo-smoke-audit-path", default=str(
        DEFAULT_OUTPUT_ROOT / "fo_anchor_2ep_seed20/e3b0_audit.json"
    ))
    parser.add_argument("--shuffled-smoke-audit-path", default=str(
        DEFAULT_OUTPUT_ROOT / "shuffled_fo_anchor_2ep_seed20/e3b0_audit.json"
    ))
    args = parser.parse_args(argv)
    if args.diagnostic_only:
        parser.error("--diagnostic-only cannot be combined with --arm or --epochs") \
            if args.arm is not None or args.epochs is not None else None
    elif args.arm is None or args.epochs is None:
        parser.error("training mode requires both --arm and --epochs")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.diagnostic_only:
        _require(not (args.arm == "LWC_REPLAY" and args.epochs != 2),
                 "LWC_REPLAY is permitted only for the 2-epoch smoke")
        _formal_gate(args)
    if args.output_dir is None:
        output_dir = (
            DEFAULT_OUTPUT_ROOT / "diagnostic_seed20"
            if args.diagnostic_only else DEFAULT_OUTPUT_ROOT / (
                args.arm.lower() + "_" + str(args.epochs) + "ep_seed20"
            )
        )
    else:
        output_dir = _resolve(args.output_dir)
    _require(not output_dir.exists(), "refusing to overwrite E3-B0 output")
    device = torch.device(args.device)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "requested CUDA device unavailable")
        torch.cuda.set_device(device)

    e1_train.set_global_seed(SEED)
    glgc_provenance = e1_train.verify_glgc_released_logic(args.glgc_repository)
    views, sample_ids, feature_provenance = e1_train.load_frozen_feature_artifact(
        args.feature_path, args.feature_audit_path
    )
    labeled_ids, true_anchor_labels, unlabeled_ids, label_split_audit = (
        load_sparse_training_labels(args.split_dir)
    )
    model, config, checkpoint_provenance = e1_train.build_model_from_frozen_d1(
        args.checkpoint_dir, device
    )
    _require(
        checkpoint_provenance["initial_backbone_hash"]["aggregate"]
        == EXPECTED_INITIAL_MODEL_HASH,
        "initial aggregate model hash differs from frozen D1",
    )
    output_dir.mkdir(parents=True)

    if args.diagnostic_only:
        diagnostic = run_diagnostic_only(
            model, views, labeled_ids, true_anchor_labels, unlabeled_ids, device
        )
        diagnostic.update({
            "initial_model_hash": checkpoint_provenance["initial_backbone_hash"],
            "feature_provenance": feature_provenance,
            "B7_label_split_provenance": label_split_audit,
            "GLGC_released_logic_provenance": glgc_provenance,
        })
        path = output_dir / "e3b0_diagnostic.json"
        _write_json(path, diagnostic)
        print("signal_retention_vs_E3A0_pass=" + str(
            diagnostic["signal_retention_vs_E3A0_pass"]
        ))
        print("Saved: " + _display(path))
        return 0

    optimizers = e1_train.build_fresh_optimizers(model)
    optimizer_audit = {
        "policy": "fresh independent Adam per MVCAN view from model-only D1 checkpoint",
        "optimizer_count": len(optimizers),
        "initial_state_empty_pass": all(
            len(optimizer.state) == 0 for optimizer in optimizers
        ),
        "learning_rate": e1_train.LEARNING_RATE,
        "same_policy_all_arms": True,
    }
    if args.arm == "LWC_REPLAY":
        predictions, runtime_audit = e1_train.train_continuation(
            arm="LWC",
            epochs=args.epochs,
            batch_size=int(config["training"]["batch_size"]),
            model=model,
            optimizers=optimizers,
            views=views,
            sample_ids=sample_ids,
            reliability=unused_lwc_api_placeholder(),
            device=device,
        )
        runtime_audit.update({
            "E3B_FO_branch_entered": False,
            "replay_source": "E1 train_continuation with exact arm='LWC'",
            "D2_reliability_loaded": False,
            "R_used": False,
            "epoch_selection_used": False,
        })
    else:
        predictions, runtime_audit = train_first_order_continuation(
            arm=args.arm,
            epochs=args.epochs,
            batch_size=int(config["training"]["batch_size"]),
            model=model,
            optimizers=optimizers,
            views=views,
            sample_ids=sample_ids,
            labeled_ids=labeled_ids,
            true_anchor_labels=true_anchor_labels,
            device=device,
        )
        runtime_audit["E3B_FO_branch_entered"] = True

    prediction_path, prediction_audit = e1_train.save_and_hash_predictions(
        predictions, output_dir
    )
    final_model_hash = hash_backbone(model.autoencoders)
    model_outputs = e1_train.save_final_models(model, output_dir)
    full_labels = e1_train.load_labels_after_predictions(
        args.label_path, prediction_path
    )
    _require(np.array_equal(full_labels[labeled_ids], true_anchor_labels),
             "post-training sparse/full label consistency failed")
    primary_metrics = e1_train.evaluate_predictions(
        full_labels[unlabeled_ids], predictions[unlabeled_ids]
    )
    secondary_metrics = e1_train.evaluate_predictions(full_labels, predictions)
    replay_audit = None
    if args.arm == "LWC_REPLAY":
        replay_audit = audit_lwc_replay_identity(
            checkpoint_provenance["initial_backbone_hash"],
            final_model_hash,
            prediction_audit,
            secondary_metrics,
            args.e1_lwc_smoke_reference_dir,
        )
    fo_arm = args.arm != "LWC_REPLAY"
    fo_audit = runtime_audit.get("FO_audit", {})
    engineering_checks = {
        "initial_model_hash_pass": checkpoint_provenance[
            "initial_backbone_hash"
        ]["aggregate"] == EXPECTED_INITIAL_MODEL_HASH,
        "same_initial_optimizer_policy_pass": optimizer_audit[
            "initial_state_empty_pass"
        ],
        "exact_14_sparse_labels_pass": label_split_audit[
            "two_labels_per_class_pass"
        ],
        "unlabeled_1386_GT_training_isolation_pass": not label_split_audit[
            "full_label_file_opened_before_training"
        ],
        "prediction_and_checkpoint_fixed_before_full_labels_pass": True,
        "D2_reliability_not_loaded_pass": not runtime_audit[
            "D2_reliability_loaded"
        ],
        "R_not_used_pass": not runtime_audit["R_used"],
        "no_target_rewriting_pass": not runtime_audit["P_all_rewritten"],
        "M_no_grad_pass": runtime_audit["M_no_grad_pass"],
        "loss_finite_pass": runtime_audit["loss_finite_pass"],
        "all_1400_native_LWC_coverage_pass": runtime_audit[
            "all_epochs_exact_sample_coverage_pass"
        ],
        "anchor_refresh_count_equals_epochs_pass": bool(
            not fo_arm or runtime_audit["anchor_refresh_count_equals_epochs_pass"]
        ),
        "BN_state_restored_pass": bool(
            not fo_arm or runtime_audit["BN_state_restored_all_epochs_pass"]
        ),
        "only_unlabeled_FO_targets_pass": bool(
            not fo_arm or fo_audit["only_unlabeled_targets_pass"]
        ),
        "no_labeled_duplicate_in_FO_graph_pass": bool(
            not fo_arm or fo_audit["no_labeled_duplicate_in_graph_target_pass"]
        ),
        "G_symmetric_extraction_exact_pass": bool(
            not fo_arm or fo_audit["G_symmetric_extraction_exact_pass"]
        ),
        "P_anchor_validity_pass": bool(
            not fo_arm or fo_audit["P_anchor_validity_pass"]
        ),
        "F_first_order_gather_exact_pass": bool(
            not fo_arm or fo_audit["F_first_gather_exact_pass"]
        ),
        "no_probability_pair_dot_pass": bool(
            not fo_arm or runtime_audit["no_pairwise_probability_dot_product"]
        ),
        "U_FO_validity_pass": bool(not fo_arm or fo_audit["U_FO_validity_pass"]),
        "T_anchor_validity_pass": bool(
            not fo_arm or fo_audit["T_anchor_validity_pass"]
        ),
        "loss_15_pairs_30_directions_temperature_pass": bool(
            not fo_arm or fo_audit["loss_aggregation_pass"]
        ),
        "L_FO_finite_positive_pass": bool(
            not fo_arm or fo_audit["L_FO_finite_positive_pass"]
        ),
        "FO_student_gradient_pass": bool(
            not fo_arm
            or runtime_audit["FO_gradient_audit"]["FO_branch_gradient_exists_pass"]
        ),
        "FO_teacher_target_detached_pass": bool(
            not fo_arm
            or (
                fo_audit["G_stop_gradient_pass"]
                and fo_audit["U_FO_validity_pass"]
                and fo_audit["T_anchor_validity_pass"]
            )
        ),
        "LWC_replay_exact_match_pass": bool(
            fo_arm or replay_audit["LWC_REPLAY_EXACT_MATCH_PASS"]
        ),
        "fixed_final_epoch_no_selection_pass": not runtime_audit[
            "epoch_selection_used"
        ],
    }
    engineering_pass = bool(all(engineering_checks.values()))
    result = {
        "stage": STAGE,
        "arm": args.arm,
        "epochs": int(args.epochs),
        "seed": SEED,
        "dataset": DATASET,
        "N": SAMPLE_NUM,
        "V": VIEW_NUM,
        "K": CLASS_NUM,
        "lambda1": LAMBDA1,
        "objective": (
            "L_MVCAN + 0.01*L_LWC" if args.arm == "LWC_REPLAY"
            else "L_MVCAN + 0.01*L_LWC + 0.01*L_FO"
        ),
        "first_order_utility_equation": "U_FO[u,v,l,i] = G_anchor_sym[u,v,l,i] * P_anchor[i,v,active_label_l]",
        "initial_model_hash": checkpoint_provenance["initial_backbone_hash"],
        "final_model_hash": final_model_hash,
        "checkpoint_provenance": checkpoint_provenance,
        "optimizer_audit": optimizer_audit,
        "feature_provenance": feature_provenance,
        "B7_label_split_provenance": label_split_audit,
        "GLGC_released_logic_provenance": glgc_provenance,
        "runtime": runtime_audit,
        "prediction_audit": prediction_audit,
        "final_model_outputs": model_outputs,
        "label_protocol": {
            "training_label_count": 14,
            "unlabeled_training_GT_count": 0,
            "full_labels_loaded_only_after_prediction_and_checkpoint_fixed": True,
            "full_labels_used_only_for_final_metrics": True,
        },
        "primary_unlabeled_1386_metrics": primary_metrics,
        "secondary_all_1400_metrics": secondary_metrics,
        "LWC_replay_audit": replay_audit,
        "frozen_formal_LWC_reference": _display(DEFAULT_E1_LWC_FORMAL_DIR),
        "engineering_checks": engineering_checks,
        "E3B0_ENGINEERING_PASS": engineering_pass,
    }
    _write_json(output_dir / "e3b0_audit.json", result)
    print("E3B0_ENGINEERING_PASS=" + str(engineering_pass))
    print("ARM=" + args.arm)
    print("PRIMARY_UNLABELED_ACC={:.10f}".format(primary_metrics["ACC"]))
    print("PRIMARY_UNLABELED_NMI={:.10f}".format(primary_metrics["NMI"]))
    print("PRIMARY_UNLABELED_ARI={:.10f}".format(primary_metrics["ARI"]))
    print("SECONDARY_ALL_ACC={:.10f}".format(secondary_metrics["ACC"]))
    print("SECONDARY_ALL_NMI={:.10f}".format(secondary_metrics["NMI"]))
    print("SECONDARY_ALL_ARI={:.10f}".format(secondary_metrics["ARI"]))
    print("Saved: " + _display(output_dir / "e3b0_audit.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
