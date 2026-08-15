"""Offline B3-A2 diagnostics using one shared final backbone per condition."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from configure import get_default_config
from datasets import load_data
from irv.b3_audit import hash_backbone, hash_semantic_heads
from irv.b3_semantic_diagnostics import agreement_diagnostic
from irv.b3_semantic_diagnostics import all_finite_nested
from irv.b3_semantic_diagnostics import clustering_diagnostic
from irv.b3_semantic_diagnostics import collapse_diagnostic
from irv.b3_semantic_diagnostics import correspondence_diagnostic
from irv.b3_semantic_diagnostics import fuse_semantic_views
from irv.b3_semantic_diagnostics import noise_separation_diagnostic
from irv.b3_semantic_diagnostics import reconstruct_changed_row_mask
from irv.semantic_head import DetachedSemanticHeadBank
from model import MvCAN
from weak_quality import apply_weak_quality_protocol


DATASET_ID = 13
DATASET_NAME = "MSRC-v1"
EXPECTED_SAMPLE_NUM = 210
EXPECTED_VIEW_NUM = 5
EXPECTED_CLUSTER_NUM = 7
EXPECTED_LATENT_DIM = 10
EXPECTED_SEMANTIC_DIM = 10


def _load_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(path + " must contain a JSON object")
    return value


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _all_parameters_have_no_grad(modules):
    return all(
        parameter.grad is None
        for module in modules
        for parameter in module.parameters()
    )


def _evaluate_arm(semantic_views, labels, model_seed):
    # semantic_views[v]: [N, semantic_dim=10]
    # semantic_fused:   [N, semantic_dim=10]
    semantic_fused = fuse_semantic_views(semantic_views)
    per_view_clustering = []
    for view_idx, semantic_view in enumerate(semantic_views):
        metrics = clustering_diagnostic(
            semantic_view,
            labels,
            n_clusters=EXPECTED_CLUSTER_NUM,
            random_state=model_seed,
        )
        per_view_clustering.append({"view_id": view_idx, **metrics})
    fused_clustering = clustering_diagnostic(
        semantic_fused,
        labels,
        n_clusters=EXPECTED_CLUSTER_NUM,
        random_state=model_seed,
    )
    agreement_matrix, agreement_summary = agreement_diagnostic(semantic_views)
    agreement_record = {
        **agreement_summary,
        "shape": [int(size) for size in agreement_matrix.shape],
        "matrix": agreement_matrix.detach().cpu().tolist(),
    }
    correspondence = correspondence_diagnostic(semantic_views)
    per_view_collapse = []
    for view_idx, semantic_view in enumerate(semantic_views):
        per_view_collapse.append({
            "view_id": view_idx,
            **collapse_diagnostic(semantic_view),
        })
    collapse = {
        "per_view": per_view_collapse,
        "fused": collapse_diagnostic(semantic_fused),
    }
    collapse["all_finite_pass"] = all_finite_nested(collapse)
    return {
        "per_view_clustering": per_view_clustering,
        "fused_clustering": fused_clustering,
        "agreement": agreement_record,
        "correspondence": correspondence,
        "collapse": collapse,
    }, agreement_matrix, semantic_fused


def _write_metrics_csv(path, condition, arm_results, noise_diagnostic):
    fieldnames = [
        "condition",
        "arm",
        "fused_ACC",
        "fused_NMI",
        "fused_ARI",
        "agreement_mean",
        "same_sample_cosine_mean",
        "mismatched_cosine_mean",
        "correspondence_gap",
        "retrieval_top1",
        "fused_effective_rank",
        "fused_variance_mean",
        "fused_pairwise_cosine_mean",
        "fused_pairwise_cosine_std",
        "clean_agreement_mean",
        "corrupted_agreement_mean",
        "clean_corrupted_gap",
        "clean_corrupted_auc",
        "spearman_clean_indicator_vs_agreement",
    ]
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for arm_key, arm_name in (
            ("a0", "A0_untrained"),
            ("a1", "A1_uniform_trained"),
        ):
            arm = arm_results[arm_key]
            noise = (
                noise_diagnostic[arm_key]
                if noise_diagnostic is not None
                else {}
            )
            fused = arm["fused_clustering"]
            agreement = arm["agreement"]
            correspondence = arm["correspondence"]
            collapse = arm["collapse"]["fused"]
            writer.writerow({
                "condition": condition,
                "arm": arm_name,
                "fused_ACC": fused["acc"],
                "fused_NMI": fused["nmi"],
                "fused_ARI": fused["ari"],
                "agreement_mean": agreement["agreement_mean"],
                "same_sample_cosine_mean": correspondence[
                    "same_sample_cosine_mean"
                ],
                "mismatched_cosine_mean": correspondence[
                    "mismatched_cosine_mean"
                ],
                "correspondence_gap": correspondence["correspondence_gap"],
                "retrieval_top1": correspondence["retrieval_top1"],
                "fused_effective_rank": collapse["effective_rank"],
                "fused_variance_mean": collapse["variance_mean"],
                "fused_pairwise_cosine_mean": collapse[
                    "off_diagonal_pairwise_cosine_mean"
                ],
                "fused_pairwise_cosine_std": collapse[
                    "off_diagonal_pairwise_cosine_std"
                ],
                "clean_agreement_mean": noise.get("clean_agreement_mean", ""),
                "corrupted_agreement_mean": noise.get(
                    "corrupted_agreement_mean", ""
                ),
                "clean_corrupted_gap": noise.get(
                    "agreement_gap_clean_minus_corrupted", ""
                ),
                "clean_corrupted_auc": noise.get("clean_corrupted_auc", ""),
                "spearman_clean_indicator_vs_agreement": noise.get(
                    "spearman_clean_indicator_vs_agreement", ""
                ),
            })


def _write_sample_view_csv(path, corruption_mask, agreement_a0, agreement_a1):
    agreement_a0 = agreement_a0.detach().cpu().numpy()
    agreement_a1 = agreement_a1.detach().cpu().numpy()
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "sample_id",
                "view_id",
                "corrupted",
                "agreement_a0",
                "agreement_a1",
                "agreement_delta",
            ],
        )
        writer.writeheader()
        for sample_id in range(corruption_mask.shape[0]):
            for view_id in range(corruption_mask.shape[1]):
                a0_value = float(agreement_a0[sample_id, view_id])
                a1_value = float(agreement_a1[sample_id, view_id])
                writer.writerow({
                    "sample_id": sample_id,
                    "view_id": view_id,
                    "corrupted": bool(corruption_mask[sample_id, view_id]),
                    "agreement_a0": a0_value,
                    "agreement_a1": a1_value,
                    "agreement_delta": a1_value - a0_value,
                })


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, required=True)
    parser.add_argument("--condition", choices=("clean", "snr2p5_k2"), required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--semantic-seed", type=int, required=True)
    parser.add_argument("--backbone-dir", required=True)
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--a1-audit", required=True)
    parser.add_argument("--corruption-k", type=int, default=2)
    parser.add_argument("--snr-db", type=float, default=2.5)
    parser.add_argument("--corruption-seed", type=int, default=20)
    parser.add_argument("--expected-mask-sha256", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _require(args.dataset == DATASET_ID, "B3-A2 currently supports only dataset 13")
    _require(args.model_seed == 20, "B3-A2 seed20 evaluation requires model seed 20")
    _require(
        args.semantic_seed == 1020,
        "B3-A2 seed20 evaluation requires semantic seed 1020",
    )

    config = get_default_config(DATASET_NAME)
    config["dataset"] = DATASET_NAME
    clean_views, label_list = load_data(config)
    labels = np.asarray(label_list[0], dtype=np.int64)
    _require(len(clean_views) == EXPECTED_VIEW_NUM, "MSRC-v1 must have 5 views")
    _require(labels.shape == (EXPECTED_SAMPLE_NUM,), "MSRC-v1 labels must be [210]")
    _require(
        len(np.unique(labels)) == EXPECTED_CLUSTER_NUM,
        "MSRC-v1 must have 7 classes",
    )

    a1_audit = _load_json(args.a1_audit)
    _require(a1_audit.get("stage") == "B3-A1", "source audit must be B3-A1")
    _require(a1_audit.get("dataset") == DATASET_NAME, "audit dataset mismatch")
    _require(a1_audit.get("model_seed") == args.model_seed, "audit seed mismatch")
    _require(a1_audit.get("latent_dim") == EXPECTED_LATENT_DIM, "latent dim mismatch")
    _require(a1_audit.get("semantic_dim") == EXPECTED_SEMANTIC_DIM, "semantic dim mismatch")
    _require(a1_audit.get("view_num") == EXPECTED_VIEW_NUM, "view count mismatch")

    if args.condition == "clean":
        evaluation_views, corruption_audit = apply_weak_quality_protocol(
            clean_views,
            mode="none",
        )
        corruption_mask = reconstruct_changed_row_mask(
            clean_views,
            evaluation_views,
        )
        corruption_protocol_pass = bool(
            a1_audit.get("corruption_mode") == "none"
            and a1_audit.get("corruption_seed") is None
            and not corruption_mask.any()
            and corruption_audit["no_op_exact_pass"]
        )
    else:
        _require(
            args.expected_mask_sha256 is not None,
            "noisy condition requires --expected-mask-sha256",
        )
        evaluation_views, corruption_audit = apply_weak_quality_protocol(
            clean_views,
            mode="heterogeneous_gaussian",
            k=args.corruption_k,
            snr_db=args.snr_db,
            corruption_seed=args.corruption_seed,
        )
        corruption_mask = reconstruct_changed_row_mask(
            clean_views,
            evaluation_views,
        )
        per_sample_counts = corruption_mask.sum(axis=1)
        per_view_counts = corruption_mask.sum(axis=0)
        corruption_protocol_pass = bool(
            corruption_mask.shape == (EXPECTED_SAMPLE_NUM, EXPECTED_VIEW_NUM)
            and np.all(per_sample_counts == 2)
            and np.array_equal(per_view_counts, np.full(5, 84))
            and int(corruption_mask.sum()) == 420
            and np.array_equal(corruption_mask, corruption_audit["mask"])
            and corruption_audit["mask_sha256"]
            == args.expected_mask_sha256
            and a1_audit.get("corruption_mask_sha256")
            == args.expected_mask_sha256
            and a1_audit.get("corruption_mode")
            == "heterogeneous_gaussian"
            and a1_audit.get("corruption_k") == args.corruption_k
            and a1_audit.get("target_snr_db") == args.snr_db
            and a1_audit.get("corruption_seed") == args.corruption_seed
        )
    _require(corruption_protocol_pass, "corruption protocol integrity failed")

    view_sizes = [int(view.shape[1]) for view in evaluation_views]
    models = MvCAN(
        config,
        view_num=EXPECTED_VIEW_NUM,
        view_size=view_sizes,
        n_clusters=EXPECTED_CLUSTER_NUM,
        seed=args.model_seed,
        data_size=EXPECTED_SAMPLE_NUM,
        semantic_config=None,
    )
    for view_idx, autoencoder in enumerate(models.autoencoders):
        checkpoint_path = os.path.join(
            args.backbone_dir,
            DATASET_NAME + str(view_idx + 1) + "V.pth",
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        autoencoder.load_state_dict(state_dict, strict=True)
        autoencoder.eval()

    loaded_backbone_hash = hash_backbone(models.autoencoders)
    backbone_hash_pass = loaded_backbone_hash == a1_audit.get("backbone_hash")
    _require(backbone_hash_pass, "loaded backbone tensor hash mismatch")

    a0_heads = DetachedSemanticHeadBank(
        view_num=EXPECTED_VIEW_NUM,
        latent_dim=EXPECTED_LATENT_DIM,
        semantic_dim=EXPECTED_SEMANTIC_DIM,
        semantic_seed=args.semantic_seed,
    )
    a0_heads.eval()
    a0_hash = hash_semantic_heads(a0_heads)
    a0_initialization_hash_match_pass = bool(
        a0_hash == a1_audit.get("semantic_hash_initial")
    )
    _require(
        a0_initialization_hash_match_pass,
        "reconstructed A0 semantic initialization hash mismatch",
    )

    a1_heads = DetachedSemanticHeadBank(
        view_num=EXPECTED_VIEW_NUM,
        latent_dim=EXPECTED_LATENT_DIM,
        semantic_dim=EXPECTED_SEMANTIC_DIM,
        semantic_seed=args.semantic_seed,
    )
    a1_state_dict = torch.load(args.semantic_checkpoint, map_location="cpu")
    _require(
        bool(a1_state_dict)
        and all(key.startswith("heads.") for key in a1_state_dict.keys()),
        "semantic checkpoint contains non-semantic state",
    )
    a1_heads.load_state_dict(a1_state_dict, strict=True)
    a1_heads.eval()
    a1_hash = hash_semantic_heads(a1_heads)
    a1_semantic_checkpoint_hash_pass = bool(
        a1_hash == a1_audit.get("semantic_hash_final")
    )
    _require(
        a1_semantic_checkpoint_hash_pass,
        "loaded A1 semantic checkpoint tensor hash mismatch",
    )

    backbone_hash_before = hash_backbone(models.autoencoders)
    a0_hash_before = hash_semantic_heads(a0_heads)
    a1_hash_before = hash_semantic_heads(a1_heads)
    all_modules = list(models.autoencoders) + [a0_heads, a1_heads]
    eval_mode_pass = all(not module.training for module in all_modules)

    with torch.no_grad():
        latent_views = []
        for view_idx, autoencoder in enumerate(models.autoencoders):
            # X_v: [N=210, input_dim_v]
            X_v = torch.from_numpy(evaluation_views[view_idx]).float()
            # Z_v: [N=210, latent_dim=10]
            Z_v = autoencoder.encoder(X_v)
            latent_views.append(Z_v)
        # S_A0_v: [N=210, semantic_dim=10]
        semantic_a0_views = a0_heads.forward_views(latent_views)
        # S_A1_v: [N=210, semantic_dim=10]
        semantic_a1_views = a1_heads.forward_views(latent_views)

    expected_shapes_pass = bool(
        len(latent_views) == EXPECTED_VIEW_NUM
        and all(tuple(value.shape) == (EXPECTED_SAMPLE_NUM, 10) for value in latent_views)
        and all(
            tuple(value.shape) == (EXPECTED_SAMPLE_NUM, 10)
            for value in semantic_a0_views + semantic_a1_views
        )
    )
    _require(expected_shapes_pass, "offline feature shape check failed")

    arm_a0, agreement_a0, fused_a0 = _evaluate_arm(
        semantic_a0_views,
        labels,
        args.model_seed,
    )
    arm_a1, agreement_a1, fused_a1 = _evaluate_arm(
        semantic_a1_views,
        labels,
        args.model_seed,
    )
    expected_shapes_pass = bool(
        expected_shapes_pass
        and tuple(fused_a0.shape) == (EXPECTED_SAMPLE_NUM, 10)
        and tuple(fused_a1.shape) == (EXPECTED_SAMPLE_NUM, 10)
        and tuple(agreement_a0.shape) == (EXPECTED_SAMPLE_NUM, 5)
        and tuple(agreement_a1.shape) == (EXPECTED_SAMPLE_NUM, 5)
    )

    delta = {
        "fused_acc": (
            arm_a1["fused_clustering"]["acc"]
            - arm_a0["fused_clustering"]["acc"]
        ),
        "fused_nmi": (
            arm_a1["fused_clustering"]["nmi"]
            - arm_a0["fused_clustering"]["nmi"]
        ),
        "fused_ari": (
            arm_a1["fused_clustering"]["ari"]
            - arm_a0["fused_clustering"]["ari"]
        ),
        "agreement_mean": (
            arm_a1["agreement"]["agreement_mean"]
            - arm_a0["agreement"]["agreement_mean"]
        ),
        "correspondence_gap": (
            arm_a1["correspondence"]["correspondence_gap"]
            - arm_a0["correspondence"]["correspondence_gap"]
        ),
        "retrieval_top1": (
            arm_a1["correspondence"]["retrieval_top1"]
            - arm_a0["correspondence"]["retrieval_top1"]
        ),
        "fused_effective_rank": (
            arm_a1["collapse"]["fused"]["effective_rank"]
            - arm_a0["collapse"]["fused"]["effective_rank"]
        ),
    }

    noise_diagnostic = None
    if args.condition == "snr2p5_k2":
        noise_a0 = noise_separation_diagnostic(agreement_a0, corruption_mask)
        noise_a1 = noise_separation_diagnostic(agreement_a1, corruption_mask)
        noise_diagnostic = {
            "mask_shape": [int(size) for size in corruption_mask.shape],
            "mask_sha256": corruption_audit["mask_sha256"],
            "per_sample_corrupted_count_unique": [
                int(value) for value in np.unique(corruption_mask.sum(axis=1))
            ],
            "per_view_corrupted_counts": [
                int(value) for value in corruption_mask.sum(axis=0)
            ],
            "total_corrupted": int(corruption_mask.sum()),
            "a0": noise_a0,
            "a1": noise_a1,
            "delta_a1_minus_a0": {
                "clean_agreement_mean": (
                    noise_a1["clean_agreement_mean"]
                    - noise_a0["clean_agreement_mean"]
                ),
                "corrupted_agreement_mean": (
                    noise_a1["corrupted_agreement_mean"]
                    - noise_a0["corrupted_agreement_mean"]
                ),
                "agreement_gap_clean_minus_corrupted": (
                    noise_a1["agreement_gap_clean_minus_corrupted"]
                    - noise_a0["agreement_gap_clean_minus_corrupted"]
                ),
                "clean_corrupted_auc": (
                    noise_a1["clean_corrupted_auc"]
                    - noise_a0["clean_corrupted_auc"]
                ),
            },
        }

    all_outputs_finite_pass = bool(
        all_finite_nested(arm_a0)
        and all_finite_nested(arm_a1)
        and all_finite_nested(delta)
        and all_finite_nested(noise_diagnostic)
    )
    backbone_hash_after = hash_backbone(models.autoencoders)
    a0_hash_after = hash_semantic_heads(a0_heads)
    a1_hash_after = hash_semantic_heads(a1_heads)
    parameter_immutability_pass = bool(
        backbone_hash_before == backbone_hash_after
        and a0_hash_before == a0_hash_after
        and a1_hash_before == a1_hash_after
    )
    no_gradients_created_pass = _all_parameters_have_no_grad(all_modules)

    integrity = {
        "backbone_hash_pass": bool(backbone_hash_pass),
        "a0_initialization_hash_match_pass": bool(
            a0_initialization_hash_match_pass
        ),
        "a1_semantic_checkpoint_hash_pass": bool(
            a1_semantic_checkpoint_hash_pass
        ),
        "corruption_protocol_pass": bool(corruption_protocol_pass),
        "all_outputs_finite_pass": bool(all_outputs_finite_pass),
        "expected_shapes_pass": bool(expected_shapes_pass),
        "eval_mode_pass": bool(eval_mode_pass),
        "no_optimizer_pass": True,
        "no_backward_pass": bool(no_gradients_created_pass),
        "no_parameter_update_pass": bool(parameter_immutability_pass),
        "shared_final_backbone_pass": True,
        "loaded_backbone_hash": loaded_backbone_hash,
        "reconstructed_a0_semantic_hash": a0_hash,
        "loaded_a1_semantic_hash": a1_hash,
    }
    engineering_pass = all(
        value is True
        for key, value in integrity.items()
        if key.endswith("_pass")
    )
    integrity["b3_a2_engineering_pass"] = bool(engineering_pass)

    result = {
        "stage": "B3-A2",
        "condition": args.condition,
        "dataset": DATASET_NAME,
        "model_seed": args.model_seed,
        "semantic_seed": args.semantic_seed,
        "sample_num": EXPECTED_SAMPLE_NUM,
        "view_num": EXPECTED_VIEW_NUM,
        "cluster_num": EXPECTED_CLUSTER_NUM,
        "latent_dim": EXPECTED_LATENT_DIM,
        "semantic_dim": EXPECTED_SEMANTIC_DIM,
        "source": {
            "backbone_dir": args.backbone_dir,
            "semantic_checkpoint": args.semantic_checkpoint,
            "a1_audit": args.a1_audit,
        },
        "integrity": integrity,
        "a0": arm_a0,
        "a1": arm_a1,
        "delta_a1_minus_a0": delta,
        "noise_diagnostic": noise_diagnostic,
    }
    _require(all_finite_nested(result), "diagnostic JSON contains non-finite values")
    _require(engineering_pass, "B3-A2 engineering integrity failed")

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "b3_a2_diagnostics.json")
    with open(json_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
    _write_metrics_csv(
        os.path.join(args.output_dir, "b3_a2_metrics.csv"),
        args.condition,
        {"a0": arm_a0, "a1": arm_a1},
        noise_diagnostic,
    )
    if noise_diagnostic is not None:
        _write_sample_view_csv(
            os.path.join(args.output_dir, "sample_view_diagnostics.csv"),
            corruption_mask,
            agreement_a0,
            agreement_a1,
        )

    print("B3-A2 condition: " + args.condition)
    print("backbone_hash_pass=true")
    print("a0_initialization_hash_match_pass=true")
    print("a1_semantic_checkpoint_hash_pass=true")
    print("corruption_protocol_pass=true")
    print("A0 fused clustering: " + str(arm_a0["fused_clustering"]))
    print("A1 fused clustering: " + str(arm_a1["fused_clustering"]))
    print("A0 correspondence: " + str(arm_a0["correspondence"]))
    print("A1 correspondence: " + str(arm_a1["correspondence"]))
    if noise_diagnostic is not None:
        print("A1 noise diagnostic: " + str(noise_diagnostic["a1"]))
    print("B3_A2_ENGINEERING_PASS=" + str(engineering_pass).lower())
    print("B3-A2 diagnostics: " + json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
