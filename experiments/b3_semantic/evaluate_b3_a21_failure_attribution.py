"""Offline B3-A2.1 scientific-gate and reliability-failure diagnostics."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from configure import get_default_config
from datasets import load_data
from irv.b3_audit import hash_backbone, hash_semantic_heads
from irv.b3_predictability_diagnostics import bootstrap_reliability
from irv.b3_predictability_diagnostics import fold_assignment_csv_text
from irv.b3_predictability_diagnostics import fold_assignment_sha256
from irv.b3_predictability_diagnostics import make_fold_assignment
from irv.b3_predictability_diagnostics import null_distribution_summary
from irv.b3_predictability_diagnostics import oof_ridge_predictability
from irv.b3_predictability_diagnostics import score_summary
from irv.b3_predictability_diagnostics import value_position_in_null
from irv.b3_semantic_diagnostics import agreement_diagnostic
from irv.b3_semantic_diagnostics import all_finite_nested
from irv.b3_semantic_diagnostics import clustering_diagnostic
from irv.b3_semantic_diagnostics import correspondence_diagnostic
from irv.b3_semantic_diagnostics import effective_rank
from irv.b3_semantic_diagnostics import fuse_semantic_views
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
NULL_METRICS = (
    "fused_acc",
    "fused_nmi",
    "fused_ari",
    "correspondence_gap",
    "retrieval_top1",
    "effective_rank",
    "agreement_mean",
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(path + " must contain a JSON object")
    return value


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _all_parameters_have_no_grad(modules):
    return all(
        parameter.grad is None
        for module in modules
        for parameter in module.parameters()
    )


def _semantic_arm_metrics(semantic_views, labels, random_state):
    fused = fuse_semantic_views(semantic_views)
    clustering = clustering_diagnostic(
        fused,
        labels,
        n_clusters=EXPECTED_CLUSTER_NUM,
        random_state=random_state,
    )
    agreement, agreement_summary = agreement_diagnostic(semantic_views)
    correspondence = correspondence_diagnostic(semantic_views)
    return {
        "fused_acc": float(clustering["acc"]),
        "fused_nmi": float(clustering["nmi"]),
        "fused_ari": float(clustering["ari"]),
        "correspondence_gap": float(correspondence["correspondence_gap"]),
        "retrieval_top1": float(correspondence["retrieval_top1"]),
        "effective_rank": float(effective_rank(fused)),
        "agreement_mean": float(agreement_summary["agreement_mean"]),
    }, agreement


def _position_record(observed_metrics, null_rows):
    result = {}
    for metric in NULL_METRICS:
        position = value_position_in_null(
            observed_metrics[metric],
            [row[metric] for row in null_rows],
        )
        result[metric + "_percentile"] = position["percentile"]
        result[metric + "_z_score"] = position["z_score"]
        result[metric + "_value"] = position["value"]
    result["percentile_definition"] = (
        "100 * mean(null_value <= observed_value)"
    )
    return result


def _random_head_null(
    latent_views,
    labels,
    seed_start,
    seed_count,
    random_state,
):
    rows = []
    no_gradients_pass = True
    immutable_pass = True
    for semantic_seed in range(seed_start, seed_start + seed_count):
        heads = DetachedSemanticHeadBank(
            view_num=EXPECTED_VIEW_NUM,
            latent_dim=EXPECTED_LATENT_DIM,
            semantic_dim=EXPECTED_SEMANTIC_DIM,
            semantic_seed=semantic_seed,
        )
        heads.eval()
        hash_before = hash_semantic_heads(heads)
        with torch.no_grad():
            semantic_views = heads.forward_views(latent_views)
        metrics, _ = _semantic_arm_metrics(
            semantic_views,
            labels,
            random_state,
        )
        hash_after = hash_semantic_heads(heads)
        immutable_pass = bool(immutable_pass and hash_before == hash_after)
        no_gradients_pass = bool(
            no_gradients_pass
            and _all_parameters_have_no_grad([heads])
        )
        rows.append({
            "semantic_seed": int(semantic_seed),
            "semantic_head_hash": hash_before["aggregate"],
            **metrics,
        })
    summary = {
        metric: null_distribution_summary([row[metric] for row in rows])
        for metric in NULL_METRICS
    }
    return rows, summary, no_gradients_pass, immutable_pass


def _write_random_null_csv(path, condition, rows):
    fieldnames = [
        "condition",
        "semantic_seed",
        "fused_acc",
        "fused_nmi",
        "fused_ari",
        "correspondence_gap",
        "retrieval_top1",
        "effective_rank",
        "agreement_mean",
        "semantic_head_hash",
    ]
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({"condition": condition, **row})


def _write_sample_view_reliability(
    path,
    corruption_mask,
    agreement_a1,
    predictability,
):
    fieldnames = [
        "sample_id",
        "view_id",
        "corrupted",
        "agreement_a1",
        "predictability_z",
        "predictability_a0",
        "predictability_a1",
        "predictability_a1_minus_a0",
        "predictability_a1_minus_z",
        "secondary_pairwise_mean_a1",
        "secondary_pairwise_median_a1",
        "secondary_negative_mse_a1",
    ]
    scores_z = predictability["z_native"]["oof_consensus_cosine"]
    scores_a0 = predictability["a0_random"]["oof_consensus_cosine"]
    scores_a1 = predictability["a1_uniform"]["oof_consensus_cosine"]
    agreement = agreement_a1.detach().cpu().numpy()
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for sample_id in range(corruption_mask.shape[0]):
            for view_id in range(corruption_mask.shape[1]):
                writer.writerow({
                    "sample_id": sample_id,
                    "view_id": view_id,
                    "corrupted": bool(corruption_mask[sample_id, view_id]),
                    "agreement_a1": float(agreement[sample_id, view_id]),
                    "predictability_z": float(scores_z[sample_id, view_id]),
                    "predictability_a0": float(scores_a0[sample_id, view_id]),
                    "predictability_a1": float(scores_a1[sample_id, view_id]),
                    "predictability_a1_minus_a0": float(
                        scores_a1[sample_id, view_id]
                        - scores_a0[sample_id, view_id]
                    ),
                    "predictability_a1_minus_z": float(
                        scores_a1[sample_id, view_id]
                        - scores_z[sample_id, view_id]
                    ),
                    "secondary_pairwise_mean_a1": float(
                        predictability["a1_uniform"]["pairwise_cosine_mean"][
                            sample_id, view_id
                        ]
                    ),
                    "secondary_pairwise_median_a1": float(
                        predictability["a1_uniform"]["pairwise_cosine_median"][
                            sample_id, view_id
                        ]
                    ),
                    "secondary_negative_mse_a1": float(
                        predictability["a1_uniform"]["consensus_negative_mse"][
                            sample_id, view_id
                        ]
                    ),
                })


def _compact_predictability(record, reliability=None):
    compact = {
        "primary_score": "oof_consensus_cosine",
        "score_summary": score_summary(record["oof_consensus_cosine"]),
        "secondary_pairwise_mean_summary": score_summary(
            record["pairwise_cosine_mean"]
        ),
        "secondary_pairwise_median_summary": score_summary(
            record["pairwise_cosine_median"]
        ),
        "secondary_negative_mse_summary": score_summary(
            record["consensus_negative_mse"]
        ),
        "audit": record["audit"],
    }
    if reliability is not None:
        compact["reliability"] = reliability
    return compact


def _agreement_bootstrap_record(record):
    return {
        "clean_agreement_mean": record["clean_predictability_mean"],
        "corrupted_agreement_mean": record["corrupted_predictability_mean"],
        "agreement_gap": record["predictability_gap"],
        "clean_corrupted_auc": record["clean_corrupted_auc"],
        "spearman_clean_indicator_vs_agreement": record[
            "spearman_clean_indicator_vs_predictability"
        ],
        "agreement_gap_ci_low": record["gap_ci_low"],
        "agreement_gap_ci_high": record["gap_ci_high"],
        "agreement_auc_ci_low": record["auc_ci_low"],
        "agreement_auc_ci_high": record["auc_ci_high"],
        "agreement_spearman_ci_low": record["spearman_ci_low"],
        "agreement_spearman_ci_high": record["spearman_ci_high"],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, required=True)
    parser.add_argument(
        "--condition", choices=("clean", "snr2p5_k2"), required=True
    )
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--semantic-seed", type=int, required=True)
    parser.add_argument("--null-seed-start", type=int, required=True)
    parser.add_argument("--null-seed-count", type=int, required=True)
    parser.add_argument("--oof-folds", type=int, required=True)
    parser.add_argument("--ridge-alpha", type=float, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
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
    _require(args.dataset == DATASET_ID, "B3-A2.1 supports only dataset 13")
    _require(args.model_seed == 20, "B3-A2.1 requires model seed 20")
    _require(args.semantic_seed == 1020, "B3-A2.1 requires semantic seed 1020")
    _require(args.null_seed_start == 1020, "null seed start must be 1020")
    _require(args.null_seed_count == 100, "null seed count must be 100")
    _require(args.oof_folds == 5, "OOF fold count must be 5")
    _require(args.ridge_alpha == 1.0, "Ridge alpha must be 1.0")
    _require(args.bootstrap_repeats == 2000, "bootstrap repeats must be 2000")
    _require(args.bootstrap_seed == 20260815, "bootstrap seed mismatch")

    config = get_default_config(DATASET_NAME)
    config["dataset"] = DATASET_NAME
    clean_views, label_list = load_data(config)
    labels = np.asarray(label_list[0], dtype=np.int64)
    _require(len(clean_views) == EXPECTED_VIEW_NUM, "MSRC-v1 must have 5 views")
    _require(labels.shape == (EXPECTED_SAMPLE_NUM,), "labels must have shape [210]")
    _require(len(np.unique(labels)) == EXPECTED_CLUSTER_NUM, "expected 7 classes")

    a1_audit = _load_json(args.a1_audit)
    _require(a1_audit.get("stage") == "B3-A1", "source audit must be B3-A1")
    _require(a1_audit.get("dataset") == DATASET_NAME, "audit dataset mismatch")
    _require(a1_audit.get("model_seed") == args.model_seed, "audit seed mismatch")
    _require(a1_audit.get("semantic_seed") == args.semantic_seed, "semantic seed mismatch")
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
        corruption_protocol_pass = bool(
            corruption_mask.shape == (EXPECTED_SAMPLE_NUM, EXPECTED_VIEW_NUM)
            and np.all(corruption_mask.sum(axis=1) == 2)
            and np.array_equal(corruption_mask.sum(axis=0), np.full(5, 84))
            and int(corruption_mask.sum()) == 420
            and np.array_equal(corruption_mask, corruption_audit["mask"])
            and corruption_audit["mask_sha256"] == args.expected_mask_sha256
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
        autoencoder.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu"),
            strict=True,
        )
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
    a0_seed1020_hash_pass = a0_hash == a1_audit.get("semantic_hash_initial")
    _require(a0_seed1020_hash_pass, "A0 seed1020 hash mismatch")

    a1_heads = DetachedSemanticHeadBank(
        view_num=EXPECTED_VIEW_NUM,
        latent_dim=EXPECTED_LATENT_DIM,
        semantic_dim=EXPECTED_SEMANTIC_DIM,
        semantic_seed=args.semantic_seed,
    )
    a1_state = torch.load(args.semantic_checkpoint, map_location="cpu")
    _require(
        bool(a1_state) and all(key.startswith("heads.") for key in a1_state),
        "semantic checkpoint contains non-semantic state",
    )
    a1_heads.load_state_dict(a1_state, strict=True)
    a1_heads.eval()
    a1_hash = hash_semantic_heads(a1_heads)
    a1_semantic_hash_pass = a1_hash == a1_audit.get("semantic_hash_final")
    _require(a1_semantic_hash_pass, "A1 semantic checkpoint hash mismatch")

    backbone_hash_before = hash_backbone(models.autoencoders)
    a0_hash_before = hash_semantic_heads(a0_heads)
    a1_hash_before = hash_semantic_heads(a1_heads)
    fixed_modules = list(models.autoencoders) + [a0_heads, a1_heads]
    eval_mode_pass = all(not module.training for module in fixed_modules)

    with torch.no_grad():
        latent_views = []
        for view_idx, autoencoder in enumerate(models.autoencoders):
            features = torch.from_numpy(evaluation_views[view_idx]).float()
            latent_views.append(autoencoder.encoder(features))
        native_views = [
            F.normalize(latent, p=2, dim=1, eps=1e-12)
            for latent in latent_views
        ]
        semantic_a0_views = a0_heads.forward_views(latent_views)
        semantic_a1_views = a1_heads.forward_views(latent_views)

    expected_shapes_pass = bool(
        all(tuple(value.shape) == (EXPECTED_SAMPLE_NUM, 10) for value in latent_views)
        and all(
            tuple(value.shape) == (EXPECTED_SAMPLE_NUM, 10)
            for value in native_views + semantic_a0_views + semantic_a1_views
        )
    )
    _require(expected_shapes_pass, "offline representation shape check failed")

    native_per_view = []
    for view_id, native_view in enumerate(native_views):
        native_per_view.append({
            "view_id": view_id,
            **clustering_diagnostic(
                native_view,
                labels,
                n_clusters=EXPECTED_CLUSTER_NUM,
                random_state=args.model_seed,
            ),
        })
    native_concat = F.normalize(
        torch.cat(native_views, dim=1),
        p=2,
        dim=1,
        eps=1e-12,
    )
    native_z = {
        "per_view_clustering": native_per_view,
        "concat_clustering": clustering_diagnostic(
            native_concat,
            labels,
            n_clusters=EXPECTED_CLUSTER_NUM,
            random_state=args.model_seed,
        ),
        "effective_rank": effective_rank(native_concat),
        "concat_shape": [EXPECTED_SAMPLE_NUM, EXPECTED_VIEW_NUM * EXPECTED_LATENT_DIM],
        "diagnostic_only_not_native_mvcan_final_acc": True,
    }

    a0_metrics, agreement_a0 = _semantic_arm_metrics(
        semantic_a0_views,
        labels,
        args.model_seed,
    )
    a1_metrics, agreement_a1 = _semantic_arm_metrics(
        semantic_a1_views,
        labels,
        args.model_seed,
    )
    null_rows, null_summary, null_no_gradients_pass, null_immutable_pass = (
        _random_head_null(
            latent_views,
            labels,
            args.null_seed_start,
            args.null_seed_count,
            args.model_seed,
        )
    )
    seed1020_row = next(
        row for row in null_rows if row["semantic_seed"] == args.semantic_seed
    )
    for metric in NULL_METRICS:
        _require(
            seed1020_row[metric] == a0_metrics[metric],
            "seed1020 null metric mismatch for " + metric,
        )
    a0_seed1020_vs_random_null = _position_record(a0_metrics, null_rows)
    a1_vs_random_null = _position_record(a1_metrics, null_rows)

    fold_assignment = make_fold_assignment(
        EXPECTED_SAMPLE_NUM,
        n_splits=args.oof_folds,
        random_state=args.model_seed,
    )
    fold_hash = fold_assignment_sha256(fold_assignment)
    fold_deterministic_pass = bool(
        fold_hash
        == fold_assignment_sha256(
            make_fold_assignment(
                EXPECTED_SAMPLE_NUM,
                n_splits=args.oof_folds,
                random_state=args.model_seed,
            )
        )
    )
    predictability_raw = {
        "z_native": oof_ridge_predictability(
            native_views, fold_assignment, alpha=args.ridge_alpha
        ),
        "a0_random": oof_ridge_predictability(
            semantic_a0_views, fold_assignment, alpha=args.ridge_alpha
        ),
        "a1_uniform": oof_ridge_predictability(
            semantic_a1_views, fold_assignment, alpha=args.ridge_alpha
        ),
    }
    all_predictor_counts_pass = all(
        record["audit"]["ordered_view_pair_count"] == 20
        for record in predictability_raw.values()
    )
    oof_train_test_disjoint_pass = all(
        record["audit"]["train_test_disjoint_pass"] is True
        for record in predictability_raw.values()
    )
    oof_coverage_pass = all(
        record["audit"]["oof_coverage_pass"] is True
        and record["audit"]["oof_coverage_count"] == EXPECTED_SAMPLE_NUM
        for record in predictability_raw.values()
    )

    bootstrap = None
    agreement_bootstrap = None
    reliability_by_arm = None
    if args.condition == "snr2p5_k2":
        bootstrap = bootstrap_reliability(
            {
                "z_native": predictability_raw["z_native"][
                    "oof_consensus_cosine"
                ],
                "a0_random": predictability_raw["a0_random"][
                    "oof_consensus_cosine"
                ],
                "a1_uniform": predictability_raw["a1_uniform"][
                    "oof_consensus_cosine"
                ],
            },
            corruption_mask,
            repeats=args.bootstrap_repeats,
            seed=args.bootstrap_seed,
            paired_comparisons=(
                ("a1_uniform", "a0_random"),
                ("a1_uniform", "z_native"),
            ),
        )
        reliability_by_arm = bootstrap["arms"]
        agreement_result = bootstrap_reliability(
            {"a1_agreement": agreement_a1.detach().cpu().numpy()},
            corruption_mask,
            repeats=args.bootstrap_repeats,
            seed=args.bootstrap_seed,
        )
        agreement_bootstrap = {
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_seed": args.bootstrap_seed,
            "sampling_unit": "sample_id_with_all_views",
            **_agreement_bootstrap_record(
                agreement_result["arms"]["a1_agreement"]
            ),
        }

    predictability = {
        arm_name: _compact_predictability(
            record,
            None if reliability_by_arm is None else reliability_by_arm[arm_name],
        )
        for arm_name, record in predictability_raw.items()
    }

    backbone_hash_after = hash_backbone(models.autoencoders)
    a0_hash_after = hash_semantic_heads(a0_heads)
    a1_hash_after = hash_semantic_heads(a1_heads)
    no_parameter_update_pass = bool(
        backbone_hash_before == backbone_hash_after
        and a0_hash_before == a0_hash_after
        and a1_hash_before == a1_hash_after
        and null_immutable_pass
    )
    no_gradients_created_pass = bool(
        _all_parameters_have_no_grad(fixed_modules) and null_no_gradients_pass
    )
    all_outputs_finite_pass = bool(
        all_finite_nested(native_z)
        and all_finite_nested(a0_metrics)
        and all_finite_nested(a1_metrics)
        and all_finite_nested(null_rows)
        and all_finite_nested(null_summary)
        and all_finite_nested(predictability)
        and all_finite_nested(bootstrap)
        and all_finite_nested(agreement_bootstrap)
    )
    integrity = {
        "backbone_hash_pass": bool(backbone_hash_pass),
        "a1_semantic_hash_pass": bool(a1_semantic_hash_pass),
        "a0_seed1020_hash_pass": bool(a0_seed1020_hash_pass),
        "corruption_protocol_pass": bool(corruption_protocol_pass),
        "all_outputs_finite_pass": bool(all_outputs_finite_pass),
        "expected_shapes_pass": bool(expected_shapes_pass),
        "eval_mode_pass": bool(eval_mode_pass),
        "oof_train_test_disjoint_pass": bool(oof_train_test_disjoint_pass),
        "oof_coverage_pass": bool(oof_coverage_pass),
        "all_20_ordered_view_predictors_pass": bool(all_predictor_counts_pass),
        "fold_deterministic_pass": bool(fold_deterministic_pass),
        "no_optimizer_pass": True,
        "no_backward_pass": bool(no_gradients_created_pass),
        "no_parameter_update_pass": bool(no_parameter_update_pass),
        "shared_final_backbone_pass": True,
        "loaded_backbone_hash": loaded_backbone_hash,
        "reconstructed_a0_semantic_hash": a0_hash,
        "loaded_a1_semantic_hash": a1_hash,
        "fold_assignment_sha256": fold_hash,
    }
    engineering_pass = all(
        value is True
        for key, value in integrity.items()
        if key.endswith("_pass")
    )
    integrity["B3_A21_ENGINEERING_PASS"] = bool(engineering_pass)

    gates = {
        "B3_A21_ENGINEERING_PASS": bool(engineering_pass),
        "B3_A21_AGREEMENT_EVIDENCE_PASS": None,
        "B3_A21_PREDICTABILITY_ABSOLUTE_PASS": None,
        "B3_A21_PREDICTABILITY_GAIN_OVER_A0_PASS": None,
        "B3_A21_PREDICTABILITY_GAIN_OVER_Z_PASS": None,
        "B3_A21_UTILITY_EVIDENCE_CANDIDATE": None,
    }
    if args.condition == "snr2p5_k2":
        a1_reliability = bootstrap["arms"]["a1_uniform"]
        a1_minus_a0 = bootstrap["paired"]["a1_uniform_minus_a0_random"]
        a1_minus_z = bootstrap["paired"]["a1_uniform_minus_z_native"]
        agreement_pass = bool(
            agreement_bootstrap["agreement_auc_ci_low"] > 0.5
            and agreement_bootstrap["agreement_gap_ci_low"] > 0.0
        )
        absolute_pass = bool(
            a1_reliability["auc_ci_low"] > 0.5
            and a1_reliability["gap_ci_low"] > 0.0
        )
        gain_a0_pass = bool(
            a1_minus_a0["delta_auc_ci_low"] > 0.0
            or a1_minus_a0["delta_gap_ci_low"] > 0.0
        )
        gain_z_pass = bool(
            a1_minus_z["delta_auc_ci_low"] > 0.0
            or a1_minus_z["delta_gap_ci_low"] > 0.0
        )
        gates.update({
            "B3_A21_AGREEMENT_EVIDENCE_PASS": agreement_pass,
            "B3_A21_PREDICTABILITY_ABSOLUTE_PASS": absolute_pass,
            "B3_A21_PREDICTABILITY_GAIN_OVER_A0_PASS": gain_a0_pass,
            "B3_A21_PREDICTABILITY_GAIN_OVER_Z_PASS": gain_z_pass,
            "B3_A21_UTILITY_EVIDENCE_CANDIDATE": bool(
                engineering_pass and absolute_pass
            ),
        })

    result = {
        "stage": "B3-A2.1",
        "condition": args.condition,
        "dataset": DATASET_NAME,
        "model_seed": args.model_seed,
        "semantic_seed": args.semantic_seed,
        "sample_num": EXPECTED_SAMPLE_NUM,
        "view_num": EXPECTED_VIEW_NUM,
        "cluster_num": EXPECTED_CLUSTER_NUM,
        "latent_dim": EXPECTED_LATENT_DIM,
        "semantic_dim": EXPECTED_SEMANTIC_DIM,
        "protocol": {
            "primary_predictability_score": "oof_consensus_cosine",
            "oof_folds": args.oof_folds,
            "oof_shuffle": True,
            "oof_random_state": args.model_seed,
            "ridge_alpha": args.ridge_alpha,
            "ridge_fit_intercept": True,
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_unit": "sample_id_with_all_views",
            "mask_used_for_evaluation_only": True,
            "labels_used_for_clustering_only": True,
        },
        "source": {
            "backbone_dir": args.backbone_dir,
            "semantic_checkpoint": args.semantic_checkpoint,
            "a1_audit": args.a1_audit,
        },
        "integrity": integrity,
        "native_z": native_z,
        "a0_seed1020": a0_metrics,
        "a1_uniform": a1_metrics,
        "random_head_null": {
            "seed_start": args.null_seed_start,
            "seed_count": args.null_seed_count,
            **null_summary,
        },
        "a0_seed1020_vs_random_null": a0_seed1020_vs_random_null,
        "a1_vs_random_null": a1_vs_random_null,
        "predictability": predictability,
        "bootstrap": bootstrap,
        "agreement_bootstrap": agreement_bootstrap,
        "gates": gates,
        "scientific_scope_note": (
            "Offline seed20 evidence only; no final B3 PASS or B4 readiness is declared."
        ),
    }
    _require(all_finite_nested(result), "diagnostic JSON contains non-finite values")
    _require(engineering_pass, "B3-A2.1 engineering integrity failed")

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(
        args.output_dir,
        "b3_a21_failure_attribution.json",
    )
    _write_json(json_path, result)
    _write_random_null_csv(
        os.path.join(args.output_dir, "random_head_null.csv"),
        args.condition,
        null_rows,
    )
    _write_json(
        os.path.join(args.output_dir, "random_head_null_summary.json"),
        {
            "stage": "B3-A2.1",
            "condition": args.condition,
            "seed_start": args.null_seed_start,
            "seed_count": args.null_seed_count,
            "metrics": null_summary,
            "a0_seed1020_vs_random_null": a0_seed1020_vs_random_null,
            "a1_vs_random_null": a1_vs_random_null,
        },
    )
    with open(
        os.path.join(args.output_dir, "fold_assignment.csv"),
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        output_file.write(fold_assignment_csv_text(fold_assignment))
    if args.condition == "snr2p5_k2":
        _write_sample_view_reliability(
            os.path.join(args.output_dir, "sample_view_reliability.csv"),
            corruption_mask,
            agreement_a1,
            predictability_raw,
        )

    print("B3-A2.1 condition: " + args.condition)
    print("fold_assignment_sha256=" + fold_hash)
    print("Native z concat clustering: " + str(native_z["concat_clustering"]))
    print("A0 seed1020 metrics: " + str(a0_metrics))
    print("A1 uniform metrics: " + str(a1_metrics))
    print("Random-head fused ACC summary: " + str(null_summary["fused_acc"]))
    if bootstrap is not None:
        for arm_name in ("z_native", "a0_random", "a1_uniform"):
            print(arm_name + " reliability: " + str(bootstrap["arms"][arm_name]))
        print("paired reliability: " + str(bootstrap["paired"]))
        print("agreement bootstrap: " + str(agreement_bootstrap))
    for gate_name, gate_value in gates.items():
        print(gate_name + "=" + str(gate_value).lower())
    print("B3-A2.1 diagnostics: " + json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
