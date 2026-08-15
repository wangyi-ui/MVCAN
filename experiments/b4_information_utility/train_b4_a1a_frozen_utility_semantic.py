"""B4-A1a frozen-final-z utility-gated semantic mechanism probe."""

import argparse
import copy
import csv
import inspect
import json
import math
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
from irv.b3_predictability_diagnostics import fold_assignment_sha256
from irv.b3_predictability_diagnostics import make_fold_assignment
from irv.b3_predictability_diagnostics import oof_ridge_predictability
from irv.b3_semantic_diagnostics import agreement_diagnostic
from irv.b3_semantic_diagnostics import clustering_diagnostic
from irv.b3_semantic_diagnostics import collapse_diagnostic
from irv.b3_semantic_diagnostics import correspondence_diagnostic
from irv.b3_semantic_diagnostics import fuse_semantic_views
from irv.b3_semantic_diagnostics import reconstruct_changed_row_mask
from irv.b4_information_utility import compute_information_utility
from irv.b4_information_utility import information_utility_diagnostics
from irv.b4_information_utility import tensor_sha256
from irv.b4_information_utility import tensor_view_list_sha256
from irv.b4_information_utility import uniform_equivalence_max_abs_error
from irv.b4_information_utility import utility_reliability_audit
from irv.b4_information_utility import weighted_symmetric_semantic_infonce
from irv.semantic_head import DetachedSemanticHeadBank
from model import MvCAN
from weak_quality import apply_weak_quality_protocol


DATASET_NAME = "MSRC-v1"
EXPECTED_SAMPLE_NUM = 210
EXPECTED_VIEW_NUM = 5
EXPECTED_CLUSTER_NUM = 7
EXPECTED_LATENT_DIM = 10
EXPECTED_SEMANTIC_DIM = 10
SEMANTIC_SEED = 1020
EXPECTED_TEMPERATURE = 0.2
EXPECTED_SEMANTIC_LR = 1e-4
A23_REFERENCE = (
    REPOSITORY_ROOT
    / "outputs/b3_semantic/a23_reproduction/a23_scores_snr2p5_seed20.npz"
)
B4_A0_REFERENCE = (
    REPOSITORY_ROOT
    / "outputs/b4_information_utility/a0_offline/b4_a0_utility_benchmark.json"
)
TRACE_FIELDS = [
    "condition",
    "arm",
    "step",
    "semantic_loss",
    "semantic_grad_norm",
    "pair_weight_mean",
    "pair_weight_min",
    "pair_weight_max",
]
ENGINEERING_GATES = (
    "B4_A1A_CANONICAL_BACKBONE_PASS",
    "B4_A1A_BACKBONE_FROZEN_PASS",
    "B4_A1A_BACKBONE_HASH_UNCHANGED_PASS",
    "B4_A1A_NO_BACKBONE_GRAD_PASS",
    "B4_A1A_INITIAL_HEAD_MATCH_PASS",
    "B4_A1A_SEMANTIC_HEAD_UPDATE_PASS",
    "B4_A1A_OOF_INTEGRITY_PASS",
    "B4_A1A_T_DEFINITION_PASS",
    "B4_A1A_UTILITY_DEFINITION_PASS",
    "B4_A1A_UTILITY_FINITE_PASS",
    "B4_A1A_UTILITY_RANGE_PASS",
    "B4_A1A_NO_LABEL_LEAKAGE_PASS",
    "B4_A1A_NO_MASK_IN_UTILITY_PASS",
    "B4_A1A_UNIFORM_EQUIVALENCE_PASS",
    "B4_A1A_WEIGHT_NORMALIZATION_PASS",
    "B4_A1A_ALL_LOSS_FINITE_PASS",
    "B4_A1A_ALL_GRAD_FINITE_PASS",
    "B4_A1A_NONCOLLAPSE_PASS",
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _load_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    return value


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _write_trace(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=TRACE_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _optimizer_defaults(optimizer):
    result = {}
    for key, value in optimizer.defaults.items():
        if isinstance(value, tuple):
            result[key] = list(value)
        elif isinstance(value, (str, bool, int, float)) or value is None:
            result[key] = value
        else:
            result[key] = str(value)
    return result


def _semantic_diagnostics(semantic_views, labels, random_state):
    # semantic_views: List[V], each [N, semantic_dim]
    fused = fuse_semantic_views(semantic_views)
    agreement, agreement_summary = agreement_diagnostic(semantic_views)
    correspondence = correspondence_diagnostic(semantic_views)
    collapse = collapse_diagnostic(fused)
    return {
        "fused_clustering": clustering_diagnostic(
            fused,
            labels,
            n_clusters=EXPECTED_CLUSTER_NUM,
            random_state=random_state,
        ),
        "effective_rank": collapse["effective_rank"],
        "correspondence_gap": correspondence["correspondence_gap"],
        "retrieval_top1": correspondence["retrieval_top1"],
        "agreement": agreement_summary,
        "collapse": collapse,
        "semantic_shape": [EXPECTED_SAMPLE_NUM, EXPECTED_SEMANTIC_DIM],
        "agreement_shape": list(agreement.shape),
    }


def _load_and_validate_manifest(path):
    manifest = _load_json(path)
    _require(isinstance(manifest, list), "condition manifest must be a list")
    entries = {}
    for entry in manifest:
        condition = entry.get("condition")
        _require(condition in ("clean", "snr2p5_k2"), "unexpected condition")
        _require(condition not in entries, "duplicate condition")
        _require(entry.get("dataset") == DATASET_NAME, "dataset mismatch")
        _require(entry.get("model_seed") == 20, "model seed mismatch")
        checkpoints = [_resolve(value) for value in entry["checkpoint_paths"]]
        missing = [str(value) for value in checkpoints if not value.is_file()]
        if condition == "clean" and missing:
            raise RuntimeError("MISSING_CANONICAL_CLEAN_BACKBONE: " + str(missing))
        _require(not missing, "missing canonical checkpoint: " + str(missing))
        _require(len(checkpoints) == EXPECTED_VIEW_NUM, "expected 5 checkpoints")
        source_audit = _load_json(_resolve(entry["source_audit"]))
        source_pass = bool(
            source_audit.get("stage") == "B3-A1"
            and source_audit.get("dataset") == DATASET_NAME
            and source_audit.get("model_seed") == 20
            and source_audit.get("backbone_hash", {}).get("aggregate")
            == entry["expected_backbone_hash"]
        )
        if condition == "clean":
            source_pass = bool(
                source_pass
                and entry.get("corruption_mode") == "none"
                and entry.get("corruption_seed") is None
                and source_audit.get("corruption_mode") == "none"
            )
        else:
            corruption_audit = _load_json(_resolve(entry["corruption_audit"]))
            source_pass = bool(
                source_pass
                and entry.get("corruption_mode") == "heterogeneous_gaussian"
                and entry.get("corruption_seed") == 20
                and entry.get("corruption_k") == 2
                and entry.get("snr_db") == 2.5
                and corruption_audit.get("dataset") == DATASET_NAME
                and corruption_audit.get("model_seed") == 20
                and corruption_audit.get("corruption_seed") == 20
                and corruption_audit.get("corruption_k") == 2
                and corruption_audit.get("target_snr_db") == 2.5
                and corruption_audit.get("mask_sha256")
                == entry["expected_mask_sha256"]
            )
        _require(source_pass, "manifest source audit mismatch for " + condition)
        entries[condition] = {**entry, "checkpoint_paths": checkpoints}
    _require(set(entries) == {"clean", "snr2p5_k2"}, "manifest incomplete")
    return entries


def _prepare_data(entry, clean_views):
    condition = entry["condition"]
    if condition == "clean":
        evaluation_views, audit = apply_weak_quality_protocol(
            clean_views,
            mode="none",
        )
        mask = None
        protocol_pass = bool(
            audit["no_op_exact_pass"]
            and all(
                np.array_equal(clean, evaluated)
                for clean, evaluated in zip(clean_views, evaluation_views)
            )
        )
    else:
        evaluation_views, audit = apply_weak_quality_protocol(
            clean_views,
            mode="heterogeneous_gaussian",
            k=2,
            snr_db=2.5,
            corruption_seed=20,
        )
        reconstructed = reconstruct_changed_row_mask(clean_views, evaluation_views)
        stored = np.load(_resolve(entry["corruption_mask"]), allow_pickle=False)
        mask = stored.astype(bool, copy=False)
        protocol_pass = bool(
            np.array_equal(reconstructed, mask)
            and np.array_equal(reconstructed, audit["mask"])
            and audit["mask_sha256"] == entry["expected_mask_sha256"]
            and np.all(mask.sum(axis=1) == 2)
            and np.array_equal(mask.sum(axis=0), np.full(5, 84))
        )
    _require(protocol_pass, "condition data protocol failed for " + condition)
    return evaluation_views, mask, audit


def _load_frozen_backbone(entry, config, evaluation_views):
    view_sizes = [int(view.shape[1]) for view in evaluation_views]
    models = MvCAN(
        config,
        view_num=EXPECTED_VIEW_NUM,
        view_size=view_sizes,
        n_clusters=EXPECTED_CLUSTER_NUM,
        seed=20,
        data_size=EXPECTED_SAMPLE_NUM,
        semantic_config=None,
    )
    for view_id, autoencoder in enumerate(models.autoencoders):
        autoencoder.load_state_dict(
            torch.load(entry["checkpoint_paths"][view_id], map_location="cpu"),
            strict=True,
        )
        autoencoder.eval()
        autoencoder.requires_grad_(False)
    backbone_hash = hash_backbone(models.autoencoders)
    _require(
        backbone_hash["aggregate"] == entry["expected_backbone_hash"],
        "canonical backbone hash mismatch for " + entry["condition"],
    )
    return models, backbone_hash


def _extract_frozen_z(models, evaluation_views):
    # z_views: List[V], each [N,D]
    with torch.no_grad():
        z_views = []
        for view_id, autoencoder in enumerate(models.autoencoders):
            features = torch.from_numpy(evaluation_views[view_id]).float()
            z_v = autoencoder.encoder(features)
            z_views.append(F.normalize(z_v, p=2, dim=1, eps=1e-12).detach())
    _require(
        all(tuple(value.shape) == (EXPECTED_SAMPLE_NUM, EXPECTED_LATENT_DIM)
            for value in z_views),
        "frozen z shape mismatch",
    )
    _require(all(not value.requires_grad for value in z_views), "z must be frozen")
    return z_views


def _gradient_audit(heads):
    squared_norm = 0.0
    finite = True
    nonzero = False
    for parameter in heads.parameters():
        if parameter.grad is None:
            finite = False
            continue
        finite = bool(finite and torch.isfinite(parameter.grad).all().item())
        squared_norm += float(parameter.grad.detach().pow(2).sum().item())
        nonzero = bool(nonzero or parameter.grad.detach().abs().sum().item() > 0.0)
    return math.sqrt(squared_norm), finite, nonzero


def _train_arm(
    condition,
    arm,
    template_state,
    template_output_hash,
    z_views,
    utility,
    labels,
    steps,
    temperature,
    semantic_lr,
    output_dir,
    backbone_models,
    backbone_hash_before,
):
    heads = DetachedSemanticHeadBank(
        EXPECTED_VIEW_NUM,
        EXPECTED_LATENT_DIM,
        EXPECTED_SEMANTIC_DIM,
        SEMANTIC_SEED,
    )
    heads.load_state_dict(copy.deepcopy(template_state), strict=True)
    initial_hash = hash_semantic_heads(heads)
    heads.eval()
    with torch.no_grad():
        initial_outputs = heads.forward_views(z_views)
    initial_output_hash = tensor_view_list_sha256(initial_outputs)
    initial_output_match_pass = initial_output_hash == template_output_hash
    heads.train()

    optimizer = torch.optim.Adam(heads.parameters(), lr=semantic_lr)
    trace = []
    all_loss_finite = True
    all_grad_finite = True
    all_grad_nonzero = True
    for step in range(steps):
        # semantic_views: List[V], each [N,semantic_dim]
        semantic_views = heads.forward_views(z_views)
        semantic_loss, diagnostics = weighted_symmetric_semantic_infonce(
            semantic_views,
            utility,
            temperature=temperature,
        )
        optimizer.zero_grad()
        semantic_loss.backward()
        grad_norm, grad_finite, grad_nonzero = _gradient_audit(heads)
        all_loss_finite = bool(
            all_loss_finite and torch.isfinite(semantic_loss).item()
        )
        all_grad_finite = bool(all_grad_finite and grad_finite)
        all_grad_nonzero = bool(all_grad_nonzero and grad_nonzero)
        trace.append({
            "condition": condition,
            "arm": arm,
            "step": step,
            "semantic_loss": float(semantic_loss.detach().item()),
            "semantic_grad_norm": grad_norm,
            "pair_weight_mean": diagnostics["pair_weight_mean"],
            "pair_weight_min": diagnostics["pair_weight_min"],
            "pair_weight_max": diagnostics["pair_weight_max"],
        })
        optimizer.step()

    final_hash = hash_semantic_heads(heads)
    heads.eval()
    with torch.no_grad():
        final_outputs = heads.forward_views(z_views)
    final_diagnostics = _semantic_diagnostics(final_outputs, labels, 20)
    backbone_hash_after = hash_backbone(backbone_models.autoencoders)
    backbone_grad_none = all(
        parameter.grad is None
        for autoencoder in backbone_models.autoencoders
        for parameter in autoencoder.parameters()
    )
    z_grad_none = all(value.grad is None for value in z_views)

    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    _write_trace(arm_dir / "training_trace.csv", trace)
    torch.save(heads.state_dict(), arm_dir / "semantic_heads.pth")
    _write_json(
        arm_dir / "final_semantic_diagnostics.json",
        final_diagnostics,
    )
    arm_audit = {
        "condition": condition,
        "arm": arm,
        "semantic_steps": steps,
        "temperature": temperature,
        "semantic_lr": semantic_lr,
        "optimizer_class": "torch.optim.Adam",
        "optimizer_defaults": _optimizer_defaults(optimizer),
        "initial_semantic_hash": initial_hash,
        "initial_semantic_output_hash": initial_output_hash,
        "initial_semantic_output_match_pass": initial_output_match_pass,
        "final_semantic_hash": final_hash,
        "semantic_head_updated_pass": final_hash != initial_hash,
        "all_loss_finite_pass": all_loss_finite,
        "all_gradient_finite_pass": all_grad_finite,
        "all_gradient_nonzero_pass": all_grad_nonzero,
        "backbone_grad_none_pass": backbone_grad_none,
        "z_grad_none_pass": z_grad_none,
        "backbone_hash_before": backbone_hash_before,
        "backbone_hash_after": backbone_hash_after,
        "backbone_hash_unchanged_pass": (
            backbone_hash_before == backbone_hash_after
        ),
        "pair_weight_definition": "U[:,v] * U[:,w]",
        "weight_normalization": "sum(weight*CE)/(sum(weight)+1e-12)",
        "first_semantic_loss": trace[0]["semantic_loss"],
        "final_semantic_loss": trace[-1]["semantic_loss"],
        "final_semantic_diagnostics": final_diagnostics,
    }
    _write_json(arm_dir / "arm_audit.json", arm_audit)
    print(arm + " initial semantic hash=" + initial_hash["aggregate"])
    print(arm + " step0 loss=" + str(trace[0]["semantic_loss"]))
    print(arm + " step" + str(steps - 1) + " loss=" + str(trace[-1]["semantic_loss"]))
    print(arm + " final semantic hash=" + final_hash["aggregate"])
    print(arm + " semantic grad finite=" + str(all_grad_finite).lower())
    return arm_audit


def _external_t_audit(t_scores, folds, condition):
    if condition != "snr2p5_k2" or not A23_REFERENCE.is_file():
        return {
            "external_reference_audit_available": False,
            "exact_consistency_pass": None,
        }
    reference = np.load(A23_REFERENCE, allow_pickle=False)
    max_abs_error = float(np.max(np.abs(t_scores - reference["T"])))
    fold_exact = bool(np.array_equal(folds, reference["fold_assignment"]))
    return {
        "external_reference_audit_available": True,
        "reference_path": str(A23_REFERENCE.relative_to(REPOSITORY_ROOT)),
        "max_abs_error": max_abs_error,
        "tolerance": 1e-10,
        "fold_assignment_exact_match": fold_exact,
        "exact_consistency_pass": bool(max_abs_error < 1e-10 and fold_exact),
    }


def _external_utility_audit(reliability, condition):
    if condition != "snr2p5_k2" or not B4_A0_REFERENCE.is_file():
        return {
            "external_reference_audit_available": False,
            "exact_consistency_pass": None,
        }
    reference = _load_json(B4_A0_REFERENCE)["seeds"]["20"]["point"]["T_only"]
    errors = {
        "auc_abs_error": abs(reliability["auc"] - reference["auc"]),
        "p20_abs_error": abs(
            reliability["p20"]["precision"] - reference["p20"]["precision"]
        ),
        "p40_abs_error": abs(
            reliability["p40"]["precision"] - reference["p40"]["precision"]
        ),
        "p60_abs_error": abs(
            reliability["p60"]["precision"] - reference["p60"]["precision"]
        ),
    }
    return {
        "external_reference_audit_available": True,
        "reference_path": str(B4_A0_REFERENCE.relative_to(REPOSITORY_ROOT)),
        "reference": reference,
        **errors,
        "tolerance": 1e-12,
        "exact_consistency_pass": all(value < 1e-12 for value in errors.values()),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-manifest", required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("clean", "snr2p5_k2"),
        required=True,
    )
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("uniform", "t_utility"),
        required=True,
    )
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--semantic-lr", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _require(args.model_seed == 20, "B4-A1a pilot requires model seed 20")
    _require(args.arms == ["uniform", "t_utility"], "both fixed arms are required")
    _require(args.steps > 0, "steps must be positive")
    _require(args.temperature == EXPECTED_TEMPERATURE, "temperature must be 0.2")
    _require(args.semantic_lr == EXPECTED_SEMANTIC_LR, "semantic lr must be 1e-4")
    entries = _load_and_validate_manifest(args.condition_manifest)

    config = get_default_config(DATASET_NAME)
    config["dataset"] = DATASET_NAME
    clean_views, label_list = load_data(config)
    labels = np.asarray(label_list[0], dtype=np.int64)
    _require(labels.shape == (EXPECTED_SAMPLE_NUM,), "labels must have shape [210]")

    output_root = _resolve(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_conditions = {}
    for condition in args.conditions:
        entry = entries[condition]
        condition_dir = output_root / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        evaluation_views, corruption_mask, corruption_audit = _prepare_data(
            entry, clean_views
        )
        models, backbone_hash_before = _load_frozen_backbone(
            entry, config, evaluation_views
        )
        z_views = _extract_frozen_z(models, evaluation_views)
        z_hash = tensor_view_list_sha256(z_views)

        folds = make_fold_assignment(
            EXPECTED_SAMPLE_NUM,
            n_splits=5,
            random_state=20,
        )
        fold_hash = fold_assignment_sha256(folds)
        predictability = oof_ridge_predictability(z_views, folds, alpha=1.0)
        # T: [N,V], fixed B3 OOF consensus cosine.
        t_scores = predictability["oof_consensus_cosine"]
        t_hash = tensor_sha256(t_scores)
        # U: [N,V], computed once before semantic learning and frozen.
        utility_numpy = compute_information_utility(t_scores)
        utility = torch.from_numpy(utility_numpy).float().detach()
        utility_hash = tensor_sha256(utility_numpy)
        utility_diagnostics = information_utility_diagnostics(
            t_scores, utility_numpy
        )
        t_external = _external_t_audit(t_scores, folds, condition)

        template = DetachedSemanticHeadBank(
            EXPECTED_VIEW_NUM,
            EXPECTED_LATENT_DIM,
            EXPECTED_SEMANTIC_DIM,
            SEMANTIC_SEED,
        )
        template_state = copy.deepcopy(template.state_dict())
        template_hash = hash_semantic_heads(template)
        template.eval()
        with torch.no_grad():
            template_outputs = template.forward_views(z_views)
        template_output_hash = tensor_view_list_sha256(template_outputs)
        initial_diagnostics = _semantic_diagnostics(template_outputs, labels, 20)
        uniform_equivalence_error = uniform_equivalence_max_abs_error(
            template_outputs, args.temperature
        )
        _write_json(
            condition_dir / "initial_semantic_diagnostics.json",
            initial_diagnostics,
        )

        arm_weights = {
            "uniform": torch.ones(
                (EXPECTED_SAMPLE_NUM, EXPECTED_VIEW_NUM), dtype=torch.float32
            ),
            "t_utility": utility,
        }
        arm_audits = {}
        for arm in args.arms:
            arm_audits[arm] = _train_arm(
                condition,
                arm,
                template_state,
                template_output_hash,
                z_views,
                arm_weights[arm],
                labels,
                args.steps,
                args.temperature,
                args.semantic_lr,
                condition_dir,
                models,
                backbone_hash_before,
            )

        noisy_reliability = None
        utility_external = {
            "external_reference_audit_available": False,
            "exact_consistency_pass": None,
        }
        if corruption_mask is not None:
            noisy_reliability = utility_reliability_audit(
                utility_numpy, corruption_mask
            )
            utility_external = _external_utility_audit(
                noisy_reliability, condition
            )

        final_backbone_hash = hash_backbone(models.autoencoders)
        initial_hash_match = bool(
            arm_audits["uniform"]["initial_semantic_hash"]
            == arm_audits["t_utility"]["initial_semantic_hash"]
            == template_hash
        )
        initial_output_match = bool(
            arm_audits["uniform"]["initial_semantic_output_match_pass"]
            and arm_audits["t_utility"]["initial_semantic_output_match_pass"]
        )
        semantic_update_pass = all(
            arm_audits[arm]["semantic_head_updated_pass"] for arm in args.arms
        )
        loss_finite_pass = all(
            arm_audits[arm]["all_loss_finite_pass"] for arm in args.arms
        )
        grad_finite_pass = all(
            arm_audits[arm]["all_gradient_finite_pass"] for arm in args.arms
        )
        no_backbone_grad = all(
            arm_audits[arm]["backbone_grad_none_pass"]
            and arm_audits[arm]["z_grad_none_pass"]
            for arm in args.arms
        )
        noncollapse_pass = all(
            arm_audits[arm]["final_semantic_diagnostics"]["effective_rank"] > 1.5
            and arm_audits[arm]["final_semantic_diagnostics"]["collapse"][
                "variance_mean"
            ] > 1e-6
            for arm in args.arms
        )
        utility_signature = inspect.signature(compute_information_utility).parameters
        t_integrity = predictability["audit"]
        gates = {
            "B4_A1A_CANONICAL_BACKBONE_PASS": bool(
                backbone_hash_before["aggregate"] == entry["expected_backbone_hash"]
            ),
            "B4_A1A_BACKBONE_FROZEN_PASS": all(
                not parameter.requires_grad
                for autoencoder in models.autoencoders
                for parameter in autoencoder.parameters()
            ),
            "B4_A1A_BACKBONE_HASH_UNCHANGED_PASS": bool(
                backbone_hash_before == final_backbone_hash
                and all(
                    arm_audits[arm]["backbone_hash_unchanged_pass"]
                    for arm in args.arms
                )
            ),
            "B4_A1A_NO_BACKBONE_GRAD_PASS": no_backbone_grad,
            "B4_A1A_INITIAL_HEAD_MATCH_PASS": bool(
                initial_hash_match and initial_output_match
            ),
            "B4_A1A_SEMANTIC_HEAD_UPDATE_PASS": semantic_update_pass,
            "B4_A1A_OOF_INTEGRITY_PASS": bool(
                t_integrity["train_test_disjoint_pass"]
                and t_integrity["oof_coverage_pass"]
                and t_integrity["oof_coverage_count"] == EXPECTED_SAMPLE_NUM
                and t_integrity["ordered_view_pair_count"] == 20
            ),
            "B4_A1A_T_DEFINITION_PASS": bool(
                t_scores.shape == (EXPECTED_SAMPLE_NUM, EXPECTED_VIEW_NUM)
                and np.isfinite(t_scores).all()
                and (
                    not t_external["external_reference_audit_available"]
                    or t_external["exact_consistency_pass"]
                )
            ),
            "B4_A1A_UTILITY_DEFINITION_PASS": bool(
                np.array_equal(
                    utility_numpy,
                    compute_information_utility(t_scores),
                )
            ),
            "B4_A1A_UTILITY_FINITE_PASS": bool(np.isfinite(utility_numpy).all()),
            "B4_A1A_UTILITY_RANGE_PASS": bool(
                utility_numpy.min() >= 0.0 and utility_numpy.max() <= 1.0
            ),
            "B4_A1A_NO_LABEL_LEAKAGE_PASS": set(utility_signature)
            == {"predictability_scores"},
            "B4_A1A_NO_MASK_IN_UTILITY_PASS": set(utility_signature)
            == {"predictability_scores"},
            "B4_A1A_UNIFORM_EQUIVALENCE_PASS": (
                uniform_equivalence_error < 1e-7
            ),
            "B4_A1A_WEIGHT_NORMALIZATION_PASS": all(
                arm_audits[arm]["weight_normalization"]
                == "sum(weight*CE)/(sum(weight)+1e-12)"
                for arm in args.arms
            ),
            "B4_A1A_ALL_LOSS_FINITE_PASS": loss_finite_pass,
            "B4_A1A_ALL_GRAD_FINITE_PASS": grad_finite_pass,
            "B4_A1A_NONCOLLAPSE_PASS": noncollapse_pass,
        }
        engineering_pass = all(gates[name] is True for name in ENGINEERING_GATES)
        gates["B4_A1A_ENGINEERING_PASS"] = engineering_pass
        _require(engineering_pass, "B4-A1a engineering gate failed for " + condition)

        utility_audit = {
            "condition": condition,
            "primary_predictability_score": "oof_consensus_cosine",
            "information_utility_definition": (
                "per-view (average_rank(T)-1)/(N-1)"
            ),
            "pair_weight_definition": "U_i^v * U_i^w",
            "t_sha256": t_hash,
            "utility_sha256": utility_hash,
            "fold_sha256": fold_hash,
            "diagnostics": utility_diagnostics,
            "noisy_posthoc_reliability": noisy_reliability,
            "external_t_consistency": t_external,
            "external_b4_a0_consistency": utility_external,
            "corruption_mask_usage": "post_training_evaluation_only",
        }
        _write_json(condition_dir / "utility_audit.json", utility_audit)
        metadata = {
            "stage": "B4-A1a",
            "condition": condition,
            "dataset": DATASET_NAME,
            "model_seed": 20,
            "semantic_seed": SEMANTIC_SEED,
            "semantic_dim": EXPECTED_SEMANTIC_DIM,
            "temperature": args.temperature,
            "semantic_lr": args.semantic_lr,
            "semantic_steps": args.steps,
            "optimizer_class": "torch.optim.Adam",
            "optimizer_kwargs_explicit": {"lr": args.semantic_lr},
            "backbone_dir": entry["backbone_dir"],
            "checkpoint_paths": [
                str(path.relative_to(REPOSITORY_ROOT))
                for path in entry["checkpoint_paths"]
            ],
            "backbone_hash_before": backbone_hash_before,
            "backbone_hash_after": final_backbone_hash,
            "z_shape": [EXPECTED_SAMPLE_NUM, EXPECTED_LATENT_DIM],
            "z_view_count": EXPECTED_VIEW_NUM,
            "z_sha256": z_hash,
            "fold_sha256": fold_hash,
            "t_shape": list(t_scores.shape),
            "t_sha256": t_hash,
            "utility_shape": list(utility_numpy.shape),
            "utility_sha256": utility_hash,
            "initial_semantic_hash": template_hash,
            "initial_semantic_output_hash": template_output_hash,
            "uniform_equivalence_max_abs_error": uniform_equivalence_error,
            "utility_mode": "frozen_final_native_z_mechanism_probe",
            "mechanism_probe_only": True,
            "online_deployable_method": False,
            "final_method_claim_allowed": False,
            "utility_computed_once_before_semantic_training": True,
            "corruption_mask_used_for_training": False,
            "ground_truth_labels_used_for_training": False,
            "gates": gates,
        }
        _write_json(condition_dir / "metadata.json", metadata)
        run_conditions[condition] = {
            "metadata": metadata,
            "utility_audit": utility_audit,
            "arms": arm_audits,
        }

        print("condition=" + condition)
        print("backbone path=" + entry["backbone_dir"])
        print("backbone hash=" + backbone_hash_before["aggregate"])
        print("z shape=" + str([list(value.shape) for value in z_views]))
        print("z hash=" + z_hash)
        print("fold sha256=" + fold_hash)
        print("T shape=" + str(list(t_scores.shape)))
        print("T finite=" + str(np.isfinite(t_scores).all()).lower())
        print("U shape=" + str(list(utility_numpy.shape)))
        print("U min/max/mean=" + str((
            float(utility_numpy.min()),
            float(utility_numpy.max()),
            float(utility_numpy.mean()),
        )))
        print("utility hash=" + utility_hash)
        print("A23 T consistency=" + str(t_external).lower())
        print("initial hash exact match=" + str(initial_hash_match).lower())
        print("uniform equivalence max abs error=" + str(uniform_equivalence_error))
        print("backbone before/after exact=" + str(
            backbone_hash_before == final_backbone_hash
        ).lower())
        for gate_name, gate_value in gates.items():
            print(gate_name + "=" + str(gate_value).lower())

    overall_engineering_pass = all(
        record["metadata"]["gates"]["B4_A1A_ENGINEERING_PASS"]
        for record in run_conditions.values()
    )
    run_result = {
        "stage": "B4-A1a",
        "scope": "frozen_final_z_mechanism_probe",
        "conditions": run_conditions,
        "steps": args.steps,
        "B4_A1A_ENGINEERING_PASS": overall_engineering_pass,
        "B4_A1A_ACTION_SIGNAL_CANDIDATE": False,
        "action_signal_candidate_eligible": False,
        "action_signal_note": (
            "Smoke is engineering-only; scientific action gate requires the "
            "preregistered 1001-step two-condition pilot."
        ),
        "B4_FINAL_PASS_DECLARED": False,
        "B4_A1B_STARTED": False,
    }
    _write_json(output_root / "b4_a1a_run.json", run_result)
    print("B4_A1A_ENGINEERING_PASS=" + str(overall_engineering_pass).lower())
    print("B4_A1A_ACTION_SIGNAL_CANDIDATE=false (smoke ineligible)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
