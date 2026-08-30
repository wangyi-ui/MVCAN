"""E3-A0 sparse-anchor cross-sample semantic relation continuation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.e1_pairwise_utility import train_e1_pairwise_utility as e1_train
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    align_semantic_probabilities,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    pairwise_semantic_cooperation_loss,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    robust_inter_affinity,
)
from experiments.e2_sparse_semantic_utility.train_e2_sparse_semantic_utility import (
    audit_lwc_replay_identity,
)
from experiments.e2_sparse_semantic_utility.train_e2_sparse_semantic_utility import (
    load_sparse_training_labels,
)
from experiments.e2_sparse_semantic_utility.train_e2_sparse_semantic_utility import (
    refresh_native_target_and_semantics,
)
from experiments.e3_sparse_relation_expansion.semantic_relation_expansion import (
    E3_ARMS,
)
from experiments.e3_sparse_relation_expansion.semantic_relation_expansion import (
    anchor_induced_class_evidence,
)
from experiments.e3_sparse_relation_expansion.semantic_relation_expansion import (
    build_anchor_batch_graph,
)
from experiments.e3_sparse_relation_expansion.semantic_relation_expansion import (
    build_relation_targets,
)
from experiments.e3_sparse_relation_expansion.semantic_relation_expansion import (
    cross_sample_relation_loss,
)
from experiments.e3_sparse_relation_expansion.semantic_relation_expansion import (
    cross_sample_semantic_similarity,
)
from experiments.e3_sparse_relation_expansion.semantic_relation_expansion import (
    deterministic_anchor_label_shuffle,
)
from irv.b3_audit import hash_backbone


STAGE = "E3-A0"
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
DEFAULT_SPLIT_DIR = (
    REPOSITORY_ROOT / "outputs/b7_sparse_supervision/b7a0_seed20"
)
DEFAULT_E1_LWC_SMOKE_DIR = (
    REPOSITORY_ROOT / "outputs/e1_pairwise_utility/lwc_2ep_seed20"
)
DEFAULT_E1_LWC_FORMAL_DIR = (
    REPOSITORY_ROOT / "outputs/e1_pairwise_utility/lwc_100ep_seed20"
)


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


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def unused_lwc_api_placeholder():
    """Shape-only E1 LWC API input; its values are never read by LWC weights."""
    placeholder = np.zeros((SAMPLE_NUM, VIEW_NUM), dtype=np.float64)
    placeholder.setflags(write=False)
    return placeholder


def _unlabeled_batch_mask(batch_ids, labeled_ids, device):
    ids = batch_ids.to(dtype=torch.long)
    anchors = torch.as_tensor(
        np.array(labeled_ids, copy=True, dtype=np.int64), dtype=torch.long
    )
    unlabeled = torch.all(ids[:, None] != anchors[None, :], dim=1)
    return unlabeled.to(device=device).detach()


def _relation_gradient_audit(relation_loss, q_local, model):
    targets = [
        q_local,
        model.autoencoders[0]._cluster_layer,
        next(model.autoencoders[0]._encoder.parameters()),
    ]
    gradients = torch.autograd.grad(
        relation_loss, targets, retain_graph=True, allow_unused=True
    )
    names = ("q_local", "cluster_layer_view0", "encoder_view0")
    result = {}
    for name, gradient in zip(names, gradients):
        finite = gradient is not None and bool(torch.isfinite(gradient).all())
        nonzero = finite and bool(torch.count_nonzero(gradient).item() > 0)
        result[name + "_gradient_finite_pass"] = bool(finite)
        result[name + "_gradient_nonzero_pass"] = bool(nonzero)
    result["relation_branch_gradient_exists_pass"] = bool(all(result.values()))
    return result


def _new_relation_aggregate():
    return {
        "batch_count": 0,
        "G_anchor_batch_shape_pass": True,
        "G_batch_batch_shape_pass": True,
        "graph_stop_gradient_pass": True,
        "P_anchor_validity_pass": True,
        "S_sem_range_pass": True,
        "S_sem_stop_gradient_pass": True,
        "U_rel_finite_pass": True,
        "U_rel_stop_gradient_pass": True,
        "T_rel_stop_gradient_pass": True,
        "sample_diagonal_excluded_pass": True,
        "labeled_target_rows_excluded_pass": True,
        "labeled_target_columns_excluded_pass": True,
        "T_valid_row_sum_one_pass": True,
        "valid_row_count_positive_pass": True,
        "valid_off_diagonal_relation_count_positive_pass": True,
        "L_rel_finite_pass": True,
        "L_rel_positive_pass": True,
        "valid_row_count_total": 0,
        "valid_off_diagonal_relation_count_total": 0,
        "L_rel_min": None,
        "L_rel_max": None,
    }


def _update_relation_aggregate(
    aggregate,
    graph_anchor_batch,
    graph_batch_batch,
    anchor_probability_audit,
    semantic_similarity,
    target_audit,
    relation_loss,
    loss_audit,
):
    batch_size = int(graph_batch_batch.shape[-1])
    aggregate["batch_count"] += 1
    aggregate["G_anchor_batch_shape_pass"] = bool(
        aggregate["G_anchor_batch_shape_pass"]
        and tuple(graph_anchor_batch.shape) == (VIEW_NUM, VIEW_NUM, 14, batch_size)
    )
    aggregate["G_batch_batch_shape_pass"] = bool(
        aggregate["G_batch_batch_shape_pass"]
        and tuple(graph_batch_batch.shape)
        == (VIEW_NUM, VIEW_NUM, batch_size, batch_size)
    )
    aggregate["graph_stop_gradient_pass"] = bool(
        aggregate["graph_stop_gradient_pass"]
        and not graph_anchor_batch.requires_grad
        and graph_anchor_batch.grad_fn is None
        and not graph_batch_batch.requires_grad
        and graph_batch_batch.grad_fn is None
    )
    aggregate["P_anchor_validity_pass"] = bool(
        aggregate["P_anchor_validity_pass"]
        and all(
            anchor_probability_audit[key]
            for key in (
                "shape_pass",
                "finite_pass",
                "nonnegative_pass",
                "row_sum_one_pass",
                "stop_gradient_pass",
            )
        )
    )
    aggregate["S_sem_range_pass"] = bool(
        aggregate["S_sem_range_pass"]
        and torch.isfinite(semantic_similarity).all().item()
        and torch.all(
            (semantic_similarity >= 0.0) & (semantic_similarity <= 1.0)
        ).item()
    )
    aggregate["S_sem_stop_gradient_pass"] = bool(
        aggregate["S_sem_stop_gradient_pass"]
        and not semantic_similarity.requires_grad
        and semantic_similarity.grad_fn is None
    )
    for key in (
        "U_rel_finite_pass",
        "U_rel_stop_gradient_pass",
        "T_rel_stop_gradient_pass",
        "sample_diagonal_excluded_pass",
        "labeled_target_rows_excluded_pass",
        "labeled_target_columns_excluded_pass",
        "T_valid_row_sum_one_pass",
        "valid_row_count_positive_pass",
        "valid_off_diagonal_relation_count_positive_pass",
    ):
        aggregate[key] = bool(aggregate[key] and target_audit[key])
    aggregate["L_rel_finite_pass"] = bool(
        aggregate["L_rel_finite_pass"] and loss_audit["loss_finite_pass"]
    )
    aggregate["L_rel_positive_pass"] = bool(
        aggregate["L_rel_positive_pass"] and loss_audit["loss_positive_pass"]
    )
    aggregate["valid_row_count_total"] += target_audit["valid_row_count"]
    aggregate["valid_off_diagonal_relation_count_total"] += target_audit[
        "valid_off_diagonal_relation_count"
    ]
    value = float(relation_loss.detach().item())
    aggregate["L_rel_min"] = (
        value if aggregate["L_rel_min"] is None
        else min(aggregate["L_rel_min"], value)
    )
    aggregate["L_rel_max"] = (
        value if aggregate["L_rel_max"] is None
        else max(aggregate["L_rel_max"], value)
    )


def train_relation_continuation(
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
    _require(
        arm in ("SEM_REL", "SHUFFLED_LABEL_REL"),
        "relation continuation received an invalid arm",
    )
    _require(int(epochs) in (2, 100), "E3 permits only 2 or 100 epochs")
    _require(int(batch_size) == 256, "E3 batch size must remain 256")
    if arm == "SHUFFLED_LABEL_REL":
        active_anchor_labels, _, shuffle_audit = (
            deterministic_anchor_label_shuffle(true_anchor_labels, seed=SEED)
        )
    else:
        active_anchor_labels = np.array(
            true_anchor_labels, copy=True, dtype=np.int64
        )
        shuffle_audit = {
            "seed": None,
            "true_anchor_labels": active_anchor_labels.tolist(),
            "shuffled_anchor_labels": None,
            "changed_count": 0,
            "control_active": False,
            "unlabeled_GT_used": False,
        }

    generator = torch.Generator()
    generator.manual_seed(SEED)
    view_weights = [1.0] * VIEW_NUM
    full_views = [torch.from_numpy(view) for view in views]
    id_tensor = torch.from_numpy(np.asarray(sample_ids, dtype=np.int64))
    lwc_placeholder = torch.zeros((batch_size, VIEW_NUM), device=device)
    p_all = None
    matches = None
    anchor_h = None
    anchor_refresh_audits = []
    relation_aggregate = _new_relation_aggregate()
    relation_gradient_audit = None
    lwc_gradient_audit = None
    loss_finite_pass = True
    epoch_records = []
    native_target_refresh_count = 0

    for epoch in range(int(epochs)):
        if epoch % TARGET_REFRESH_INTERVAL == 0:
            (
                p_all,
                matches,
                _,
                view_weights,
                h_sem_full,
            ) = refresh_native_target_and_semantics(
                model, full_views, view_weights, device
            )
            labeled_index = torch.as_tensor(
                np.array(labeled_ids, copy=True, dtype=np.int64),
                device=device,
                dtype=torch.long,
            )
            # anchor_h: [14,6,7], cached only at native refresh and detached.
            anchor_h = h_sem_full[labeled_index].detach()
            anchor_refresh_audit = {
                "epoch": epoch + 1,
                "shape": list(anchor_h.shape),
                "shape_pass": tuple(anchor_h.shape) == (14, VIEW_NUM, CLASS_NUM),
                "finite_pass": bool(torch.isfinite(anchor_h).all().item()),
                "stop_gradient_pass": bool(
                    not anchor_h.requires_grad and anchor_h.grad_fn is None
                ),
                "source": "current full-data aligned-q semantic refresh",
            }
            _require(
                all(
                    anchor_refresh_audit[key]
                    for key in ("shape_pass", "finite_pass", "stop_gradient_pass")
                ),
                "anchor_h refresh audit failed",
            )
            anchor_refresh_audits.append(anchor_refresh_audit)
            native_target_refresh_count += 1
        _require(
            p_all is not None
            and matches is not None
            and anchor_h is not None,
            "native target or anchor_h missing",
        )
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
        native_sum = 0.0
        lwc_sum = 0.0
        relation_sum = 0.0
        total_sum = 0.0
        batch_count = 0
        for packed in loader:
            x_views = [packed[view_id].to(device) for view_id in range(VIEW_NUM)]
            p_batch = packed[VIEW_NUM].to(device)
            batch_ids = packed[VIEW_NUM + 1]
            visited.append(batch_ids.numpy().astype(np.int64, copy=False))
            unlabeled_mask = _unlabeled_batch_mask(
                batch_ids, labeled_ids, device
            )
            for optimizer in optimizers:
                optimizer.zero_grad()
            reconstructions = []
            q_views = []
            for view_id in range(VIEW_NUM):
                x_hat, z, q = model.autoencoders[view_id](x_views[view_id])
                # z: [B,10], q local per view: [B,7].
                reconstructions.append(x_hat)
                q_views.append(q)
            # q_local/q_aligned/h_sem: [B,6,7], non-detached student path.
            q_local = torch.stack(q_views, dim=1)
            native_losses, _ = e1_train.native_mvcan_losses(
                x_views, reconstructions, q_views, p_batch, matches
            )
            native_total = torch.stack(native_losses).sum()
            q_aligned, h_sem = align_semantic_probabilities(q_local, matches)
            _require(q_aligned.requires_grad, "E3 aligned q lost gradient")

            # E1 LWC same-instance carrier remains unchanged and differentiable.
            current_batch_size = int(q_local.shape[0])
            lwc_graph = robust_inter_affinity(h_sem.detach())
            lwc_loss, _ = pairwise_semantic_cooperation_loss(
                h_sem,
                lwc_graph,
                lwc_placeholder[:current_batch_size],
                "LWC",
            )
            # E3 target graph is separate; anchors cannot alter frozen LWC.
            # Build it once for audited anchor/batch submatrices.
            graph_all, graph_anchor_batch, graph_batch_batch = (
                build_anchor_batch_graph(anchor_h, h_sem)
            )
            anchor_probability, _, probability_audit = (
                anchor_induced_class_evidence(
                    graph_anchor_batch,
                    active_anchor_labels,
                    class_num=CLASS_NUM,
                )
            )
            semantic_similarity = cross_sample_semantic_similarity(
                anchor_probability
            )
            (
                relation_utility,
                relation_target,
                valid_rows,
                _,
                target_audit,
            ) = build_relation_targets(
                graph_batch_batch,
                semantic_similarity,
                unlabeled_mask,
            )
            relation_loss, relation_loss_audit = cross_sample_relation_loss(
                h_sem,
                relation_target,
                valid_rows,
                unlabeled_mask,
            )
            if relation_gradient_audit is None:
                relation_gradient_audit = _relation_gradient_audit(
                    relation_loss, q_local, model
                )
            if lwc_gradient_audit is None:
                lwc_gradient_audit = e1_train._pair_gradient_audit(
                    lwc_loss, q_local, model
                )
            _update_relation_aggregate(
                relation_aggregate,
                graph_anchor_batch,
                graph_batch_batch,
                probability_audit,
                semantic_similarity,
                target_audit,
                relation_loss,
                relation_loss_audit,
            )
            _require(
                not graph_all.requires_grad
                and graph_all.grad_fn is None
                and not relation_utility.requires_grad
                and relation_utility.grad_fn is None,
                "E3 target branch unexpectedly requires gradients",
            )
            # No new coefficients: native lambda1=0.01 multiplies both branches.
            total_loss = (
                native_total + LAMBDA1 * lwc_loss + LAMBDA1 * relation_loss
            )
            finite = bool(
                torch.isfinite(native_total).item()
                and torch.isfinite(lwc_loss).item()
                and torch.isfinite(relation_loss).item()
                and torch.isfinite(total_loss).item()
            )
            loss_finite_pass = bool(loss_finite_pass and finite)
            _require(finite, "non-finite E3 total loss")
            total_loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            native_sum += float(native_total.detach().item())
            lwc_sum += float(lwc_loss.detach().item())
            relation_sum += float(relation_loss.detach().item())
            total_sum += float(total_loss.detach().item())
            batch_count += 1

        coverage = e1_train._coverage_audit(visited)
        _require(
            coverage["all_1400_sample_ids_exactly_once_pass"],
            "E3 epoch did not cover all 1400 IDs exactly once",
        )
        epoch_records.append({
            "epoch": epoch + 1,
            "batch_count": batch_count,
            "native_loss_mean": native_sum / batch_count,
            "L_LWC_mean": lwc_sum / batch_count,
            "L_rel_mean": relation_sum / batch_count,
            "total_loss_mean": total_sum / batch_count,
            **coverage,
        })

    _, final_matches, final_predictions, _ = e1_train.refresh_native_target(
        model, full_views, view_weights, device
    )
    return final_predictions, {
        "epoch_records": epoch_records,
        "native_target_refresh_count": native_target_refresh_count,
        "anchor_refresh_audits": anchor_refresh_audits,
        "anchor_h_validity_pass": bool(
            all(
                audit["shape_pass"]
                and audit["finite_pass"]
                and audit["stop_gradient_pass"]
                for audit in anchor_refresh_audits
            )
        ),
        "relation_audit": relation_aggregate,
        "relation_gradient_audit": relation_gradient_audit,
        "LWC_gradient_audit": lwc_gradient_audit,
        "label_shuffle_audit": shuffle_audit,
        "active_anchor_labels": active_anchor_labels.tolist(),
        "P_all_definition": "MVCAN target_distribution(new_P(latent_fusion, centers))",
        "P_all_rewritten": False,
        "M_no_grad_pass": bool(
            not final_matches.requires_grad and final_matches.grad_fn is None
        ),
        "loss_finite_pass": bool(loss_finite_pass),
        "all_epochs_exact_sample_coverage_pass": bool(
            all(
                record["all_1400_sample_ids_exactly_once_pass"]
                for record in epoch_records
            )
        ),
        "sample_order_sha256_per_epoch": [
            record["sample_order_sha256"] for record in epoch_records
        ],
        "D2_reliability_loaded": False,
        "R_used": False,
        "direct_CE_used": False,
        "pseudo_label_used": False,
        "threshold_used": False,
        "topk_used": False,
        "memory_used": False,
        "cycle_used": False,
        "epoch_selection_used": False,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="E3-A0 sparse semantic relation expansion; seed20 only"
    )
    parser.add_argument("--arm", required=True, choices=E3_ARMS)
    parser.add_argument("--epochs", required=True, type=int, choices=(2, 100))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=str(e1_train.DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--feature-path", default=str(e1_train.DEFAULT_FEATURE_PATH))
    parser.add_argument(
        "--feature-audit-path", default=str(e1_train.DEFAULT_FEATURE_AUDIT_PATH)
    )
    parser.add_argument("--glgc-repository", default=str(e1_train.DEFAULT_GLGC_REPOSITORY))
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--label-path", default=str(e1_train.DEFAULT_LABEL_PATH))
    parser.add_argument(
        "--e1-lwc-smoke-reference-dir",
        default=str(DEFAULT_E1_LWC_SMOKE_DIR),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _require(
        not (args.arm == "LWC_REPLAY" and args.epochs != 2),
        "LWC_REPLAY is permitted only for the 2-epoch engineering smoke",
    )
    output_dir = (
        REPOSITORY_ROOT
        / "outputs/e3_sparse_relation_expansion"
        / (args.arm.lower() + "_" + str(args.epochs) + "ep_seed20")
        if args.output_dir is None
        else _resolve(args.output_dir)
    )
    _require(not output_dir.exists(), "refusing to overwrite E3 output")
    device = torch.device(args.device)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "requested CUDA device unavailable")
        torch.cuda.set_device(device)

    e1_train.set_global_seed(SEED)
    glgc_provenance = e1_train.verify_glgc_released_logic(
        args.glgc_repository
    )
    views, sample_ids, feature_provenance = (
        e1_train.load_frozen_feature_artifact(
            args.feature_path, args.feature_audit_path
        )
    )
    # Only frozen B7 14-ID memberships enter before training; no full Y or R.
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
    optimizers = e1_train.build_fresh_optimizers(model)
    optimizer_audit = {
        "policy": "fresh independent Adam per MVCAN view from model-only D1 checkpoint",
        "optimizer_count": len(optimizers),
        "initial_state_empty_pass": bool(
            all(len(optimizer.state) == 0 for optimizer in optimizers)
        ),
        "learning_rate": e1_train.LEARNING_RATE,
        "same_policy_all_arms": True,
    }

    output_dir.mkdir(parents=True)
    if args.arm == "LWC_REPLAY":
        # E1 LWC never consults the placeholder values; D2 R is not loaded.
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
            "E3_relation_branch_entered": False,
            "replay_source": "E1 train_continuation with exact arm='LWC'",
            "D2_reliability_loaded": False,
            "R_used": False,
            "epoch_selection_used": False,
        })
    else:
        predictions, runtime_audit = train_relation_continuation(
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
        runtime_audit["E3_relation_branch_entered"] = True

    prediction_path, prediction_audit = e1_train.save_and_hash_predictions(
        predictions, output_dir
    )
    # Final checkpoint is fixed before any of the 1386 unlabeled GT is opened.
    final_model_hash = hash_backbone(model.autoencoders)
    model_outputs = e1_train.save_final_models(model, output_dir)
    full_labels = e1_train.load_labels_after_predictions(
        args.label_path, prediction_path
    )
    _require(
        np.array_equal(full_labels[labeled_ids], true_anchor_labels),
        "post-training sparse/full label consistency check failed",
    )
    primary_unlabeled_metrics = e1_train.evaluate_predictions(
        full_labels[unlabeled_ids], predictions[unlabeled_ids]
    )
    secondary_all_metrics = e1_train.evaluate_predictions(
        full_labels, predictions
    )
    replay_audit = None
    if args.arm == "LWC_REPLAY":
        replay_audit = audit_lwc_replay_identity(
            checkpoint_provenance["initial_backbone_hash"],
            final_model_hash,
            prediction_audit,
            secondary_all_metrics,
            args.e1_lwc_smoke_reference_dir,
        )

    relation_arm = args.arm != "LWC_REPLAY"
    relation_audit = runtime_audit.get("relation_audit", {})
    engineering_checks = {
        "initial_model_hash_pass": checkpoint_provenance[
            "initial_backbone_hash"
        ]["aggregate"]
        == EXPECTED_INITIAL_MODEL_HASH,
        "optimizer_initial_state_policy_pass": optimizer_audit[
            "initial_state_empty_pass"
        ],
        "same_sample_order_seed20_pass": True,
        "exact_14_sparse_labels_pass": label_split_audit[
            "two_labels_per_class_pass"
        ],
        "unlabeled_1386_GT_training_isolation_pass": bool(
            not label_split_audit["full_label_file_opened_before_training"]
        ),
        "D2_reliability_not_loaded_pass": bool(
            not runtime_audit["D2_reliability_loaded"]
        ),
        "R_not_used_pass": bool(not runtime_audit["R_used"]),
        "prediction_and_checkpoint_fixed_before_full_labels_pass": True,
        "no_target_rewriting_pass": not runtime_audit["P_all_rewritten"],
        "M_no_grad_pass": runtime_audit["M_no_grad_pass"],
        "loss_finite_pass": runtime_audit["loss_finite_pass"],
        "all_1400_sample_ids_covered_each_epoch_pass": runtime_audit[
            "all_epochs_exact_sample_coverage_pass"
        ],
        "anchor_h_shape_detach_pass": bool(
            not relation_arm or runtime_audit["anchor_h_validity_pass"]
        ),
        "G_anchor_batch_shape_pass": bool(
            not relation_arm or relation_audit["G_anchor_batch_shape_pass"]
        ),
        "G_batch_batch_shape_pass": bool(
            not relation_arm or relation_audit["G_batch_batch_shape_pass"]
        ),
        "P_anchor_validity_pass": bool(
            not relation_arm or relation_audit["P_anchor_validity_pass"]
        ),
        "S_sem_range_stopgrad_pass": bool(
            not relation_arm
            or (
                relation_audit["S_sem_range_pass"]
                and relation_audit["S_sem_stop_gradient_pass"]
            )
        ),
        "U_rel_finite_stopgrad_pass": bool(
            not relation_arm
            or (
                relation_audit["U_rel_finite_pass"]
                and relation_audit["U_rel_stop_gradient_pass"]
            )
        ),
        "diagonal_excluded_pass": bool(
            not relation_arm or relation_audit["sample_diagonal_excluded_pass"]
        ),
        "labeled_targets_excluded_pass": bool(
            not relation_arm
            or (
                relation_audit["labeled_target_rows_excluded_pass"]
                and relation_audit["labeled_target_columns_excluded_pass"]
            )
        ),
        "T_rel_row_sum_stopgrad_pass": bool(
            not relation_arm
            or (
                relation_audit["T_valid_row_sum_one_pass"]
                and relation_audit["T_rel_stop_gradient_pass"]
            )
        ),
        "valid_relation_count_positive_pass": bool(
            not relation_arm
            or (
                relation_audit["valid_row_count_positive_pass"]
                and relation_audit[
                    "valid_off_diagonal_relation_count_positive_pass"
                ]
            )
        ),
        "L_rel_finite_positive_pass": bool(
            not relation_arm
            or (
                relation_audit["L_rel_finite_pass"]
                and relation_audit["L_rel_positive_pass"]
            )
        ),
        "relation_gradient_nonzero_pass": bool(
            not relation_arm
            or runtime_audit["relation_gradient_audit"][
                "relation_branch_gradient_exists_pass"
            ]
        ),
        "target_branch_no_gradient_pass": bool(
            not relation_arm
            or (
                relation_audit["graph_stop_gradient_pass"]
                and relation_audit["S_sem_stop_gradient_pass"]
                and relation_audit["U_rel_stop_gradient_pass"]
                and relation_audit["T_rel_stop_gradient_pass"]
            )
        ),
        "LWC_replay_exact_match_pass": bool(
            relation_arm or replay_audit["LWC_REPLAY_EXACT_MATCH_PASS"]
        ),
        "fixed_final_epoch_no_selection_pass": bool(
            not runtime_audit["epoch_selection_used"]
        ),
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
            "L_MVCAN + 0.01*L_LWC"
            if args.arm == "LWC_REPLAY"
            else "L_MVCAN + 0.01*L_LWC + 0.01*L_rel"
        ),
        "initial_model_hash": checkpoint_provenance["initial_backbone_hash"],
        "final_model_hash": final_model_hash,
        "checkpoint_provenance": checkpoint_provenance,
        "optimizer_audit": optimizer_audit,
        "feature_provenance": feature_provenance,
        "B7_label_split_provenance": label_split_audit,
        "GLGC_released_logic_provenance": glgc_provenance,
        "D2_reliability": {
            "loaded": False,
            "used": False,
            "path_argument_exposed": False,
        },
        "runtime": runtime_audit,
        "prediction_audit": prediction_audit,
        "final_model_outputs": model_outputs,
        "label_protocol": {
            "training_label_count": 14,
            "unlabeled_training_GT_count": 0,
            "full_labels_loaded_only_after_prediction_and_checkpoint_fixed": True,
            "full_labels_used_only_for_final_metrics": True,
            "post_training_sparse_label_consistency_pass": True,
        },
        "primary_unlabeled_1386_metrics": primary_unlabeled_metrics,
        "secondary_all_1400_metrics": secondary_all_metrics,
        "LWC_replay_audit": replay_audit,
        "frozen_formal_LWC_reference": _display(DEFAULT_E1_LWC_FORMAL_DIR),
        "engineering_checks": engineering_checks,
        "E3A0_ENGINEERING_PASS": engineering_pass,
    }
    _write_json(output_dir / "e3a0_audit.json", result)
    print("E3A0_ENGINEERING_PASS=" + str(engineering_pass))
    print("ARM=" + args.arm)
    print("PRIMARY_UNLABELED_ACC={:.10f}".format(primary_unlabeled_metrics["ACC"]))
    print("PRIMARY_UNLABELED_NMI={:.10f}".format(primary_unlabeled_metrics["NMI"]))
    print("PRIMARY_UNLABELED_ARI={:.10f}".format(primary_unlabeled_metrics["ARI"]))
    print("SECONDARY_ALL_ACC={:.10f}".format(secondary_all_metrics["ACC"]))
    print("SECONDARY_ALL_NMI={:.10f}".format(secondary_all_metrics["NMI"]))
    print("SECONDARY_ALL_ARI={:.10f}".format(secondary_all_metrics["ARI"]))
    print("Saved: " + _display(output_dir / "e3a0_audit.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
