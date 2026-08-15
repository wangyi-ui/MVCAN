"""B5-A0 protected shared-semantic Rate--Distortion mechanism probe."""

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


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from configure import get_default_config
from datasets import load_data
from irv.b3_audit import hash_backbone, hash_state_dict
from irv.b3_semantic_diagnostics import agreement_diagnostic
from irv.b3_semantic_diagnostics import all_finite_nested
from irv.b3_semantic_diagnostics import clustering_diagnostic
from irv.b3_semantic_diagnostics import collapse_diagnostic
from irv.b3_semantic_diagnostics import correspondence_diagnostic
from irv.b3_semantic_diagnostics import fuse_semantic_views
from irv.b3_semantic_diagnostics import reconstruct_changed_row_mask
from irv.b5_shared_semantic_rate import ConditionalPriorBank
from irv.b5_shared_semantic_rate import GaussianSemanticPosteriorBank
from irv.b5_shared_semantic_rate import analytic_kl_reference_check
from irv.b5_shared_semantic_rate import build_conditional_context
from irv.b5_shared_semantic_rate import canonical_view_tensor_sha256
from irv.b5_shared_semantic_rate import isotropic_directional_null_audit
from irv.b5_shared_semantic_rate import normalize_native_z
from irv.b5_shared_semantic_rate import z_l2_normalization_audit
from irv.b5_shared_semantic_rate import conditional_kl_decomposition
from irv.b5_shared_semantic_rate import conditional_rate_views
from irv.b5_shared_semantic_rate import deterministic_semantic
from irv.b5_shared_semantic_rate import gradient_audit
from irv.b5_shared_semantic_rate import isotropic_kl_decomposition
from irv.b5_shared_semantic_rate import isotropic_rate_views
from irv.b5_shared_semantic_rate import logvar_audit
from irv.b5_shared_semantic_rate import per_dimension_rate
from irv.b5_shared_semantic_rate import posterior_mu_radius_audit
from irv.b5_shared_semantic_rate import rate_distribution_audit
from irv.b5_shared_semantic_rate import sample_semantic_views
from irv.b5_shared_semantic_rate import shuffle_context_views
from irv.b5_shared_semantic_rate import tensor_list_sha256
from irv.b5_shared_semantic_rate import total_rate_distortion_loss
from irv.b5_shared_semantic_rate import uniform_semantic_distortion
from model import MvCAN
from weak_quality import apply_weak_quality_protocol


DATASET_NAME = "MSRC-v1"
EXPECTED_SAMPLE_NUM = 210
EXPECTED_VIEW_NUM = 5
EXPECTED_CLUSTER_NUM = 7
EXPECTED_LATENT_DIM = 10
EXPECTED_SEMANTIC_DIM = 10
SEMANTIC_SEED = 1020
PRIOR_SEED = 3020
EXPECTED_TEMPERATURE = 0.2
EXPECTED_SEMANTIC_LR = 1e-4
EXPECTED_BETA = 0.1
CONTEXT_SHUFFLE_SEED = 20260815
LOGVAR_MIN = -6.0
LOGVAR_MAX = 2.0
ARMS = ("semantic_only", "isotropic_rate", "conditional_rate")

TRACE_FIELDS = [
    "condition",
    "arm",
    "step",
    "distortion_loss",
    "rate_loss_per_dim",
    "beta",
    "beta_rate",
    "total_loss",
    "rate_to_distortion_ratio",
    "posterior_mu_l2_mean",
    "posterior_logvar_mean",
    "posterior_logvar_min",
    "posterior_logvar_max",
    "semantic_grad_norm",
    "prior_grad_norm",
]

ENGINEERING_GATES = (
    "B5_A0_CANONICAL_BACKBONE_PASS",
    "B5_A0_BACKBONE_FROZEN_PASS",
    "B5_A0_Z_NORMALIZATION_PASS",
    "B5_A0_CROSS_STAGE_Z_MATCH_PASS",
    "B5_A0_BACKBONE_HASH_UNCHANGED_PASS",
    "B5_A0_NO_BACKBONE_GRAD_PASS",
    "B5_A0_INITIAL_POSTERIOR_MATCH_PASS",
    "B5_A0_SAMPLING_RNG_MATCH_PASS",
    "B5_A0_FIXED_RADIUS_PASS",
    "B5_A0_LOGVAR_FINITE_PASS",
    "B5_A0_ANALYTIC_KL_PASS",
    "B5_A0_RATE_FINITE_PASS",
    "B5_A0_RATE_NONNEGATIVE_PASS",
    "B5_A0_DISTORTION_FINITE_PASS",
    "B5_A0_TOTAL_LOSS_FINITE_PASS",
    "B5_A0_POSTERIOR_GRAD_FINITE_PASS",
    "B5_A0_POSTERIOR_UPDATED_PASS",
    "B5_A0_CONDITIONAL_PRIOR_GRAD_PASS",
    "B5_A0_CONDITIONAL_CONTEXT_DETACH_PASS",
    "B5_A0_CONDITIONAL_CONTEXT_EXCLUDES_TARGET_PASS",
    "B5_A0_NO_UTILITY_USAGE_PASS",
    "B5_A0_NO_LABEL_LEAKAGE_PASS",
    "B5_A0_NO_MASK_TRAINING_USAGE_PASS",
    "B5_A0_NONCOLLAPSE_PASS",
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
            output_file, fieldnames=TRACE_FIELDS, lineterminator="\n"
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
        expected_z_hash = entry.get("expected_z_hash_b4_compatible")
        _require(
            isinstance(expected_z_hash, str) and len(expected_z_hash) == 64,
            "missing expected cross-stage z hash",
        )
        checkpoints = [_resolve(value) for value in entry["checkpoint_paths"]]
        missing = [str(value) for value in checkpoints if not value.is_file()]
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
                and corruption_audit.get("mask_sha256")
                == entry["expected_mask_sha256"]
            )
        _require(source_pass, "manifest source audit mismatch for " + condition)
        entries[condition] = {**entry, "checkpoint_paths": checkpoints}
    _require(set(entries) == {"clean", "snr2p5_k2"}, "manifest incomplete")
    return entries


def _prepare_data(entry, clean_views):
    """Reproduce the condition; returned mask is never passed to training."""
    condition = entry["condition"]
    if condition == "clean":
        evaluation_views, audit = apply_weak_quality_protocol(
            clean_views, mode="none"
        )
        corruption_mask = None
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
        corruption_mask = stored.astype(bool, copy=False)
        protocol_pass = bool(
            np.array_equal(reconstructed, corruption_mask)
            and np.array_equal(reconstructed, audit["mask"])
            and audit["mask_sha256"] == entry["expected_mask_sha256"]
            and np.all(corruption_mask.sum(axis=1) == 2)
            and np.array_equal(corruption_mask.sum(axis=0), np.full(5, 84))
        )
    _require(protocol_pass, "condition data protocol failed for " + condition)
    return evaluation_views, corruption_mask, audit


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
    for view_idx, autoencoder in enumerate(models.autoencoders):
        autoencoder.load_state_dict(
            torch.load(entry["checkpoint_paths"][view_idx], map_location="cpu"),
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
    # z_views: List[V=5], each [N=210, Dz=10]
    with torch.no_grad():
        z_views = []
        for view_idx, autoencoder in enumerate(models.autoencoders):
            features = torch.from_numpy(evaluation_views[view_idx]).float()
            # raw_z_v: [N=210, Dz=10]
            raw_z_v = autoencoder.encoder(features)
            # normalized_z_v: [N=210, Dz=10]
            normalized_z_v = normalize_native_z(raw_z_v)
            z_views.append(normalized_z_v.detach())
    _require(
        len(z_views) == EXPECTED_VIEW_NUM
        and all(
            tuple(value.shape) == (EXPECTED_SAMPLE_NUM, EXPECTED_LATENT_DIM)
            for value in z_views
        ),
        "frozen z shape mismatch",
    )
    _require(all(not value.requires_grad for value in z_views), "z must be frozen")
    return z_views


def _semantic_diagnostics(semantic_views, labels, random_state):
    """Frozen B3 descriptive diagnostics on deterministic posterior means."""
    fused = fuse_semantic_views(semantic_views)
    agreement, agreement_summary = agreement_diagnostic(semantic_views)
    correspondence = correspondence_diagnostic(semantic_views)
    collapse = collapse_diagnostic(fused)
    return {
        "evaluation_representation": "normalize(posterior_mu)",
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
        "correspondence": correspondence,
        "collapse": collapse,
        "semantic_shape": [EXPECTED_SAMPLE_NUM, EXPECTED_SEMANTIC_DIM],
        "agreement_shape": list(agreement.shape),
    }


def _conditional_structure_audit(prior_bank, radius_target):
    """Runtime audits for target exclusion and the detached prior path."""
    identifiable = [
        torch.full((4, EXPECTED_SEMANTIC_DIM), float(index + 1))
        for index in range(EXPECTED_VIEW_NUM)
    ]
    contexts = build_conditional_context(identifiable, radius_target)
    excludes_target = True
    for target in range(EXPECTED_VIEW_NUM):
        expected = torch.stack([
            identifiable[side] / radius_target
            for side in range(EXPECTED_VIEW_NUM)
            if side != target
        ]).mean(dim=0)
        excludes_target = bool(
            excludes_target and torch.equal(contexts[target], expected)
        )

    side_tensors = [
        torch.randn((4, EXPECTED_SEMANTIC_DIM), requires_grad=True)
        for _ in range(EXPECTED_VIEW_NUM)
    ]
    detached_contexts = build_conditional_context(side_tensors, radius_target)
    prior_mu, prior_logvar = prior_bank(detached_contexts)
    prior_path_loss = sum(
        value.sum() for value in prior_mu + prior_logvar
    )
    side_gradients = torch.autograd.grad(
        prior_path_loss,
        side_tensors,
        allow_unused=True,
    )
    detach_pass = all(value is None for value in side_gradients)
    return bool(detach_pass), bool(excludes_target)


def _decomposition_audit(arm, posterior_mu_views, posterior_logvar_views,
                         prior_mu_views=None, prior_logvar_views=None):
    if arm in ("semantic_only", "isotropic_rate"):
        records = [
            isotropic_kl_decomposition(mean, logvar)
            for mean, logvar in zip(
                posterior_mu_views, posterior_logvar_views
            )
        ]
        total = torch.stack([record["total"] for record in records], dim=1)
        mean_term = torch.stack([record["mean"] for record in records], dim=1)
        variance = torch.stack(
            [record["variance"] for record in records], dim=1
        )
        max_error = float(torch.max(torch.abs(total - mean_term - variance)).item())
        denominator = max(float(total.mean().item()), 1e-12)
        return {
            "rate_mu_fraction": float(mean_term.mean().item() / denominator),
            "rate_variance_fraction": float(variance.mean().item() / denominator),
            "rate_decomposition_max_abs_error": max_error,
        }

    records = [
        conditional_kl_decomposition(mean_q, logvar_q, mean_p, logvar_p)
        for mean_q, logvar_q, mean_p, logvar_p in zip(
            posterior_mu_views,
            posterior_logvar_views,
            prior_mu_views,
            prior_logvar_views,
        )
    ]
    total = torch.stack([record["total"] for record in records], dim=1)
    mismatch = torch.stack(
        [record["mean_mismatch"] for record in records], dim=1
    )
    variance = torch.stack([record["variance"] for record in records], dim=1)
    max_error = float(torch.max(torch.abs(total - mismatch - variance)).item())
    denominator = max(float(total.mean().item()), 1e-12)
    return {
        "rate_mean_mismatch_fraction": float(
            mismatch.mean().item() / denominator
        ),
        "rate_variance_fraction": float(variance.mean().item() / denominator),
        "rate_decomposition_max_abs_error": max_error,
    }


def _evaluate_final_rate(arm, posterior, prior, z_views, semantic_dim):
    posterior.eval()
    if prior is not None:
        prior.eval()
    with torch.no_grad():
        posterior_mu_views, posterior_logvar_views = posterior(z_views)
        if arm == "conditional_rate":
            contexts = build_conditional_context(
                posterior_mu_views, posterior.radius_target
            )
            prior_mu_views, prior_logvar_views = prior(contexts)
            raw_rate = conditional_rate_views(
                posterior_mu_views,
                posterior_logvar_views,
                prior_mu_views,
                prior_logvar_views,
            )
        else:
            prior_mu_views = None
            prior_logvar_views = None
            raw_rate = isotropic_rate_views(
                posterior_mu_views, posterior_logvar_views
            )
    rate_per_dim = raw_rate / semantic_dim
    return (
        posterior_mu_views,
        posterior_logvar_views,
        prior_mu_views,
        prior_logvar_views,
        rate_per_dim,
    )


def _train_arm(condition, arm, template_state, template_mu_hash,
               template_logvar_hash, z_views, steps, temperature, semantic_lr,
               beta, sampling_seed, output_dir, backbone_models,
               backbone_hash_before, analytic_audit):
    posterior = GaussianSemanticPosteriorBank(
        EXPECTED_VIEW_NUM,
        EXPECTED_LATENT_DIM,
        EXPECTED_SEMANTIC_DIM,
        SEMANTIC_SEED,
        LOGVAR_MIN,
        LOGVAR_MAX,
    )
    posterior.load_state_dict(copy.deepcopy(template_state), strict=True)
    initial_hash = hash_state_dict(posterior.state_dict())
    posterior.eval()
    with torch.no_grad():
        initial_mu, initial_logvar = posterior(z_views)
    initial_mu_hash = tensor_list_sha256(initial_mu)
    initial_logvar_hash = tensor_list_sha256(initial_logvar)
    initial_output_match_pass = bool(
        initial_mu_hash == template_mu_hash
        and initial_logvar_hash == template_logvar_hash
    )
    posterior.train()

    prior = None
    if arm == "conditional_rate":
        prior = ConditionalPriorBank(
            EXPECTED_VIEW_NUM,
            EXPECTED_SEMANTIC_DIM,
            PRIOR_SEED,
            LOGVAR_MIN,
            LOGVAR_MAX,
        )

    parameters = list(posterior.parameters())
    if prior is not None:
        parameters.extend(prior.parameters())
    optimizer = torch.optim.Adam(parameters, lr=semantic_lr)
    local_generator = torch.Generator(device="cpu")
    local_generator.manual_seed(int(sampling_seed))
    rate_training_enabled = arm != "semantic_only"
    effective_beta = float(beta) if rate_training_enabled else 0.0

    trace = []
    first_epsilon_hash = None
    all_distortion_finite = True
    all_rate_finite = True
    all_rate_nonnegative = True
    all_total_finite = True
    all_posterior_grad_finite = True
    all_posterior_grad_nonzero = True
    all_prior_grad_finite = True
    all_prior_grad_nonzero = True

    for step in range(steps):
        # posterior_mu/logvar views: List[V], each [N=210, Ds=10]
        posterior_mu_views, posterior_logvar_views = posterior(z_views)
        # semantic_views: stochastic normalized h, List[V], each [N, Ds]
        semantic_views, _, epsilon_views = sample_semantic_views(
            posterior_mu_views, posterior_logvar_views, local_generator
        )
        if first_epsilon_hash is None:
            first_epsilon_hash = tensor_list_sha256(epsilon_views)
        distortion_loss, distortion_diagnostics = uniform_semantic_distortion(
            semantic_views, temperature
        )
        _require(
            distortion_diagnostics["pair_count"] == 10,
            "semantic distortion must enumerate 10 unordered pairs",
        )

        if arm == "conditional_rate":
            # side_context: List[V], each [N,Ds], target excluded + detached.
            context_views = build_conditional_context(
                posterior_mu_views, posterior.radius_target
            )
            prior_mu_views, prior_logvar_views = prior(context_views)
            raw_rate = conditional_rate_views(
                posterior_mu_views,
                posterior_logvar_views,
                prior_mu_views,
                prior_logvar_views,
            )
        else:
            raw_rate = isotropic_rate_views(
                posterior_mu_views, posterior_logvar_views
            )
        # raw_rate: [N,V]; objective is the per-dimension mean.
        rate_loss_per_dim = per_dimension_rate(
            raw_rate.reshape(-1), EXPECTED_SEMANTIC_DIM
        ).mean()
        total_loss = total_rate_distortion_loss(
            distortion_loss,
            rate_loss_per_dim,
            effective_beta,
            rate_training_enabled,
        )

        optimizer.zero_grad()
        total_loss.backward()
        posterior_gradient = gradient_audit(posterior)
        prior_gradient = gradient_audit(prior) if prior is not None else None
        all_posterior_grad_finite = bool(
            all_posterior_grad_finite and posterior_gradient["finite"]
        )
        all_posterior_grad_nonzero = bool(
            all_posterior_grad_nonzero and posterior_gradient["nonzero"]
        )
        if prior_gradient is not None:
            all_prior_grad_finite = bool(
                all_prior_grad_finite and prior_gradient["finite"]
            )
            all_prior_grad_nonzero = bool(
                all_prior_grad_nonzero and prior_gradient["nonzero"]
            )

        radius_step = posterior_mu_radius_audit(
            posterior_mu_views, posterior.radius_target
        )
        logvar_step = logvar_audit(
            posterior_logvar_views, LOGVAR_MIN, LOGVAR_MAX
        )
        distortion_value = float(distortion_loss.detach().item())
        rate_value = float(rate_loss_per_dim.detach().item())
        beta_rate = effective_beta * rate_value
        ratio = beta_rate / max(abs(distortion_value), 1e-12)
        trace.append({
            "condition": condition,
            "arm": arm,
            "step": step,
            "distortion_loss": distortion_value,
            "rate_loss_per_dim": rate_value,
            "beta": effective_beta,
            "beta_rate": beta_rate,
            "total_loss": float(total_loss.detach().item()),
            "rate_to_distortion_ratio": ratio,
            "posterior_mu_l2_mean": radius_step["posterior_mu_l2_mean"],
            "posterior_logvar_mean": logvar_step["posterior_logvar_mean"],
            "posterior_logvar_min": logvar_step["posterior_logvar_min"],
            "posterior_logvar_max": logvar_step["posterior_logvar_max"],
            "semantic_grad_norm": posterior_gradient["norm"],
            "prior_grad_norm": (
                prior_gradient["norm"] if prior_gradient is not None else ""
            ),
        })
        all_distortion_finite = bool(
            all_distortion_finite and torch.isfinite(distortion_loss).item()
        )
        all_rate_finite = bool(
            all_rate_finite and torch.isfinite(raw_rate).all().item()
        )
        all_rate_nonnegative = bool(
            all_rate_nonnegative and raw_rate.min().detach().item() >= -1e-6
        )
        all_total_finite = bool(
            all_total_finite and torch.isfinite(total_loss).item()
        )
        optimizer.step()

    final_hash = hash_state_dict(posterior.state_dict())
    prior_final_hash = (
        hash_state_dict(prior.state_dict()) if prior is not None else None
    )
    (
        final_mu,
        final_logvar,
        final_prior_mu,
        final_prior_logvar,
        final_rate_per_dim,
    ) = _evaluate_final_rate(
        arm, posterior, prior, z_views, EXPECTED_SEMANTIC_DIM
    )
    final_semantic_views = [deterministic_semantic(value) for value in final_mu]
    radius_final = posterior_mu_radius_audit(final_mu, posterior.radius_target)
    logvar_final = logvar_audit(final_logvar, LOGVAR_MIN, LOGVAR_MAX)
    rate_audit = {
        **analytic_audit,
        **radius_final,
        **logvar_final,
        **rate_distribution_audit(final_rate_per_dim),
        **_decomposition_audit(
            arm,
            final_mu,
            final_logvar,
            final_prior_mu,
            final_prior_logvar,
        ),
        "rate_units": "nats_per_semantic_dimension",
        "rate_scale_warning": bool(
            rate_training_enabled
            and any(
                row["rate_to_distortion_ratio"] < 1e-4
                or row["rate_to_distortion_ratio"] > 1.0
                for row in trace
            )
        ),
    }

    conditional_context_detach_pass = None
    conditional_context_excludes_target_pass = None
    if prior is not None:
        (
            conditional_context_detach_pass,
            conditional_context_excludes_target_pass,
        ) = _conditional_structure_audit(prior, posterior.radius_target)
        with torch.no_grad():
            correct_context = build_conditional_context(
                final_mu, posterior.radius_target
            )
            shuffled_context = shuffle_context_views(
                correct_context, CONTEXT_SHUFFLE_SEED
            )
            correct_prior_mu, correct_prior_logvar = prior(correct_context)
            shuffled_prior_mu, shuffled_prior_logvar = prior(shuffled_context)
            correct_rate = conditional_rate_views(
                final_mu,
                final_logvar,
                correct_prior_mu,
                correct_prior_logvar,
            ).mean() / EXPECTED_SEMANTIC_DIM
            shuffled_rate = conditional_rate_views(
                final_mu,
                final_logvar,
                shuffled_prior_mu,
                shuffled_prior_logvar,
            ).mean() / EXPECTED_SEMANTIC_DIM
        correct_value = float(correct_rate.item())
        shuffled_value = float(shuffled_rate.item())
        delta = shuffled_value - correct_value
        rate_audit.update({
            "conditional_context_detach_pass": (
                conditional_context_detach_pass
            ),
            "conditional_context_correct_rate": correct_value,
            "conditional_context_shuffled_rate": shuffled_value,
            "conditional_context_shuffled_minus_correct": delta,
            "conditional_rate_correct_context": correct_value,
            "conditional_rate_shuffled_context": shuffled_value,
            "delta_shuffled_minus_correct": delta,
            "conditional_context_shuffle_seed": CONTEXT_SHUFFLE_SEED,
            "CONDITIONAL_CONTEXT_DEPENDENCE_CANDIDATE": bool(delta > 0.0),
        })

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
    torch.save(posterior.state_dict(), arm_dir / "posterior.pth")
    if prior is not None:
        torch.save(prior.state_dict(), arm_dir / "conditional_prior.pth")
    _write_json(arm_dir / "rate_audit.json", rate_audit)

    arm_audit = {
        "condition": condition,
        "arm": arm,
        "steps": steps,
        "temperature": temperature,
        "semantic_lr": semantic_lr,
        "beta": effective_beta,
        "rate_training_enabled": rate_training_enabled,
        "optimizer_class": "torch.optim.Adam",
        "optimizer_defaults": _optimizer_defaults(optimizer),
        "initial_posterior_hash": initial_hash,
        "initial_posterior_mu_output_hash": initial_mu_hash,
        "initial_posterior_logvar_output_hash": initial_logvar_hash,
        "initial_posterior_output_match_pass": initial_output_match_pass,
        "first_epsilon_hash": first_epsilon_hash,
        "final_posterior_hash": final_hash,
        "final_conditional_prior_hash": prior_final_hash,
        "posterior_updated_pass": final_hash != initial_hash,
        "distortion_finite_pass": all_distortion_finite,
        "rate_finite_pass": all_rate_finite,
        "rate_nonnegative_pass": all_rate_nonnegative,
        "total_loss_finite_pass": all_total_finite,
        "posterior_gradient_finite_pass": all_posterior_grad_finite,
        "posterior_gradient_nonzero_pass": all_posterior_grad_nonzero,
        "conditional_prior_gradient_finite_pass": (
            all_prior_grad_finite if prior is not None else None
        ),
        "conditional_prior_gradient_nonzero_pass": (
            all_prior_grad_nonzero if prior is not None else None
        ),
        "conditional_context_detach_pass": conditional_context_detach_pass,
        "conditional_context_excludes_target_pass": (
            conditional_context_excludes_target_pass
        ),
        "backbone_grad_none_pass": backbone_grad_none,
        "z_grad_none_pass": z_grad_none,
        "backbone_hash_before": backbone_hash_before,
        "backbone_hash_after": backbone_hash_after,
        "backbone_hash_unchanged_pass": (
            backbone_hash_before == backbone_hash_after
        ),
        "posterior_mu_radius_audit": radius_final,
        "posterior_logvar_audit": logvar_final,
        "rate_audit": rate_audit,
        "first_trace": trace[0],
        "final_trace": trace[-1],
    }
    _write_json(arm_dir / "arm_audit.json", arm_audit)
    return arm_audit, final_semantic_views


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
    parser.add_argument("--arms", nargs="+", choices=ARMS, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--semantic-lr", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--sampling-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _require(args.model_seed == 20, "B5-A0 currently requires model seed 20")
    _require(args.arms == list(ARMS), "all three preregistered arms are required")
    _require(args.steps > 0, "steps must be positive")
    _require(args.temperature == EXPECTED_TEMPERATURE, "temperature must be 0.2")
    _require(args.semantic_lr == EXPECTED_SEMANTIC_LR, "semantic lr must be 1e-4")
    isotropic_null_audit = isotropic_directional_null_audit(EXPECTED_SEMANTIC_DIM)
    _require(args.beta == EXPECTED_BETA, "B5-A0 preregisters beta=0.1")
    entries = _load_and_validate_manifest(args.condition_manifest)
    analytic_audit = analytic_kl_reference_check()
    _require(analytic_audit["analytic_kl_check_pass"], "analytic KL mismatch")

    config = get_default_config(DATASET_NAME)
    config["dataset"] = DATASET_NAME
    clean_views, label_list = load_data(config)
    labels = np.asarray(label_list[0], dtype=np.int64)
    _require(labels.shape == (EXPECTED_SAMPLE_NUM,), "labels must have shape [210]")
    _require(len(clean_views) == EXPECTED_VIEW_NUM, "MSRC-v1 must have 5 views")

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
        z_hash = tensor_list_sha256(z_views)

        z_norm_audit = z_l2_normalization_audit(z_views)
        z_hash_b4_compatible = canonical_view_tensor_sha256(z_views)
        expected_z_hash_b4_compatible = entry[
            "expected_z_hash_b4_compatible"
        ]
        cross_stage_z_exact_match = bool(
            z_hash_b4_compatible == expected_z_hash_b4_compatible
        )
        _require(z_norm_audit["z_normalization_pass"], "z normalization failed")
        _require(cross_stage_z_exact_match, "cross-stage normalized-z hash mismatch")
        template = GaussianSemanticPosteriorBank(
            EXPECTED_VIEW_NUM,
            EXPECTED_LATENT_DIM,
            EXPECTED_SEMANTIC_DIM,
            SEMANTIC_SEED,
            LOGVAR_MIN,
            LOGVAR_MAX,
        )
        template_state = copy.deepcopy(template.state_dict())
        template_hash = hash_state_dict(template_state)
        template.eval()
        with torch.no_grad():
            template_mu, template_logvar = template(z_views)
        template_mu_hash = tensor_list_sha256(template_mu)
        template_logvar_hash = tensor_list_sha256(template_logvar)
        template_radius = posterior_mu_radius_audit(
            template_mu, template.radius_target
        )

        arm_audits = {}
        final_semantics = {}
        for arm in args.arms:
            arm_audits[arm], final_semantics[arm] = _train_arm(
                condition,
                arm,
                template_state,
                template_mu_hash,
                template_logvar_hash,
                z_views,
                args.steps,
                args.temperature,
                args.semantic_lr,
                args.beta,
                args.sampling_seed,
                condition_dir,
                models,
                backbone_hash_before,
                analytic_audit,
            )
            diagnostics = _semantic_diagnostics(
                final_semantics[arm], labels, args.model_seed
            )
            arm_audits[arm]["final_semantic_diagnostics"] = diagnostics
            _write_json(
                condition_dir / arm / "final_semantic_diagnostics.json",
                diagnostics,
            )
            _write_json(
                condition_dir / arm / "arm_audit.json",
                arm_audits[arm],
            )

        initial_hashes = {
            arm: arm_audits[arm]["initial_posterior_hash"]
            for arm in args.arms
        }
        initial_mu_hashes = {
            arm: arm_audits[arm]["initial_posterior_mu_output_hash"]
            for arm in args.arms
        }
        initial_logvar_hashes = {
            arm: arm_audits[arm]["initial_posterior_logvar_output_hash"]
            for arm in args.arms
        }
        epsilon_hashes = {
            arm: arm_audits[arm]["first_epsilon_hash"] for arm in args.arms
        }
        initial_match = bool(
            len(set(initial_hashes.values())) == 1
            and next(iter(initial_hashes.values())) == template_hash
            and len(set(initial_mu_hashes.values())) == 1
            and next(iter(initial_mu_hashes.values())) == template_mu_hash
            and len(set(initial_logvar_hashes.values())) == 1
            and next(iter(initial_logvar_hashes.values())) == template_logvar_hash
            and all(
                arm_audits[arm]["initial_posterior_output_match_pass"]
                for arm in args.arms
            )
        )
        sampling_match = len(set(epsilon_hashes.values())) == 1
        initial_audit = {
            "template_posterior_hash": template_hash,
            "template_posterior_mu_output_hash": template_mu_hash,
            "template_posterior_logvar_output_hash": template_logvar_hash,
            "initial_posterior_hash_semantic_only": initial_hashes[
                "semantic_only"
            ],
            "initial_posterior_hash_isotropic": initial_hashes[
                "isotropic_rate"
            ],
            "initial_posterior_hash_conditional": initial_hashes[
                "conditional_rate"
            ],
            "initial_posterior_hashes": initial_hashes,
            "initial_posterior_mu_output_hashes": initial_mu_hashes,
            "initial_posterior_logvar_output_hashes": initial_logvar_hashes,
            "initial_posterior_match_pass": initial_match,
            "sampling_seed": args.sampling_seed,
            "first_epsilon_hashes": epsilon_hashes,
            "sampling_rng_match_pass": sampling_match,
            "initial_radius_audit": template_radius,
        }
        _write_json(condition_dir / "initial_posterior_audit.json", initial_audit)

        backbone_hash_after = hash_backbone(models.autoencoders)
        backbone_frozen = all(
            not parameter.requires_grad
            for autoencoder in models.autoencoders
            for parameter in autoencoder.parameters()
        )
        no_backbone_grad = all(
            arm_audits[arm]["backbone_grad_none_pass"]
            and arm_audits[arm]["z_grad_none_pass"]
            for arm in args.arms
        )
        fixed_radius_pass = all(
            arm_audits[arm]["posterior_mu_radius_audit"][
                "posterior_mu_fixed_radius_pass"
            ]
            for arm in args.arms
        )
        logvar_finite_pass = all(
            arm_audits[arm]["posterior_logvar_audit"][
                "posterior_logvar_finite_pass"
            ]
            for arm in args.arms
        )
        noncollapse_pass = all(
            arm_audits[arm]["final_semantic_diagnostics"]["effective_rank"]
            > 2.0
            and arm_audits[arm]["final_semantic_diagnostics"]["collapse"][
                "all_finite_pass"
            ]
            for arm in args.arms
        )
        posterior_api = inspect.signature(
            GaussianSemanticPosteriorBank.forward_one
        ).parameters
        rate_api = inspect.signature(conditional_rate_views).parameters
        module_source = (
            REPOSITORY_ROOT / "irv/b5_shared_semantic_rate.py"
        ).read_text(encoding="utf-8")
        no_external_weighting_import = "irv." + "b4_" not in module_source
        no_label_api = set(posterior_api) == {"self", "z_v", "view_idx"}
        no_mask_api = set(rate_api) == {
            "posterior_mu_views",
            "posterior_logvar_views",
            "prior_mu_views",
            "prior_logvar_views",
        }
        conditional_audit = arm_audits["conditional_rate"]
        gates = {
            "B5_A0_CANONICAL_BACKBONE_PASS": bool(
                backbone_hash_before["aggregate"]
                == entry["expected_backbone_hash"]
            ),
            "B5_A0_BACKBONE_FROZEN_PASS": bool(backbone_frozen),
            "B5_A0_BACKBONE_HASH_UNCHANGED_PASS": bool(
                backbone_hash_before == backbone_hash_after
                and all(
                    arm_audits[arm]["backbone_hash_unchanged_pass"]
                    for arm in args.arms
                )
            ),
            "B5_A0_NO_BACKBONE_GRAD_PASS": bool(no_backbone_grad),
            "B5_A0_Z_NORMALIZATION_PASS": bool(
                z_norm_audit["z_normalization_pass"]
            ),
            "B5_A0_CROSS_STAGE_Z_MATCH_PASS": bool(
                cross_stage_z_exact_match
            ),
            "B5_A0_INITIAL_POSTERIOR_MATCH_PASS": bool(initial_match),
            "B5_A0_SAMPLING_RNG_MATCH_PASS": bool(sampling_match),
            "B5_A0_FIXED_RADIUS_PASS": bool(fixed_radius_pass),
            "B5_A0_LOGVAR_FINITE_PASS": bool(logvar_finite_pass),
            "B5_A0_ANALYTIC_KL_PASS": bool(
                analytic_audit["analytic_kl_check_pass"]
            ),
            "B5_A0_RATE_FINITE_PASS": all(
                arm_audits[arm]["rate_finite_pass"] for arm in args.arms
            ),
            "B5_A0_RATE_NONNEGATIVE_PASS": all(
                arm_audits[arm]["rate_nonnegative_pass"] for arm in args.arms
            ),
            "B5_A0_DISTORTION_FINITE_PASS": all(
                arm_audits[arm]["distortion_finite_pass"] for arm in args.arms
            ),
            "B5_A0_TOTAL_LOSS_FINITE_PASS": all(
                arm_audits[arm]["total_loss_finite_pass"] for arm in args.arms
            ),
            "B5_A0_POSTERIOR_GRAD_FINITE_PASS": all(
                arm_audits[arm]["posterior_gradient_finite_pass"]
                and arm_audits[arm]["posterior_gradient_nonzero_pass"]
                for arm in args.arms
            ),
            "B5_A0_POSTERIOR_UPDATED_PASS": all(
                arm_audits[arm]["posterior_updated_pass"] for arm in args.arms
            ),
            "B5_A0_CONDITIONAL_PRIOR_GRAD_PASS": bool(
                conditional_audit["conditional_prior_gradient_finite_pass"]
                and conditional_audit["conditional_prior_gradient_nonzero_pass"]
            ),
            "B5_A0_CONDITIONAL_CONTEXT_DETACH_PASS": bool(
                conditional_audit["conditional_context_detach_pass"]
            ),
            "B5_A0_CONDITIONAL_CONTEXT_EXCLUDES_TARGET_PASS": bool(
                conditional_audit["conditional_context_excludes_target_pass"]
            ),
            "B5_A0_NO_UTILITY_USAGE_PASS": bool(no_external_weighting_import),
            "B5_A0_NO_LABEL_LEAKAGE_PASS": bool(no_label_api),
            "B5_A0_NO_MASK_TRAINING_USAGE_PASS": bool(no_mask_api),
            "B5_A0_NONCOLLAPSE_PASS": bool(noncollapse_pass),
        }
        engineering_pass = all(gates[name] is True for name in ENGINEERING_GATES)
        gates["B5_A0_ENGINEERING_PASS"] = bool(engineering_pass)

        rate_scale_warning = any(
            arm_audits[arm]["rate_audit"]["rate_scale_warning"]
            for arm in ("isotropic_rate", "conditional_rate")
        )
        logvar_saturation_warning = any(
            arm_audits[arm]["posterior_logvar_audit"][
                "POSTERIOR_LOGVAR_SATURATION_WARNING"
            ]
            for arm in args.arms
        )
        metadata = {
            "stage": "B5-A0",
            "mechanism_probe_only": True,
            "utility_used": False,
            "caip_used": False,
            "cycle_used": False,
            "memory_used": False,
            "pseudo_label_used": False,
            "backbone_feedback_enabled": False,
            "dataset": DATASET_NAME,
            "condition": condition,
            "model_seed": args.model_seed,
            "semantic_seed": SEMANTIC_SEED,
            "sampling_seed": args.sampling_seed,
            "prior_seed": PRIOR_SEED,
            "semantic_dim": EXPECTED_SEMANTIC_DIM,
            "radius_target": math.sqrt(EXPECTED_SEMANTIC_DIM),
            "temperature": args.temperature,
            "semantic_lr": args.semantic_lr,
            "steps": args.steps,
            "beta": args.beta,
            "posterior_logvar_min": LOGVAR_MIN,
            "posterior_logvar_max": LOGVAR_MAX,
            "backbone_path": entry["backbone_dir"],
            "backbone_hash_before": backbone_hash_before,
            "backbone_hash_after": backbone_hash_after,
            "z_hash": z_hash,
            "z_shapes": [list(value.shape) for value in z_views],
            "corruption_mask_used_for_training": False,
            "ground_truth_labels_used_for_training": False,
            "z_hash_native": z_hash,
            "z_hash_b5_native": z_hash,
            "z_hash_b4_compatible": z_hash_b4_compatible,
            "expected_z_hash_b4_compatible": expected_z_hash_b4_compatible,
            "cross_stage_z_exact_match": cross_stage_z_exact_match,
            **z_norm_audit,
            "isotropic_fixed_radius_directional_null": True,
            "isotropic_role": "fixed_radius_directional_null_control",
            "isotropic_directional_gradient_audit": isotropic_null_audit,
            "B5_A0_ISOTROPIC_DIRECTIONAL_NULL_CONFIRMED": (
                isotropic_null_audit[
                    "B5_A0_ISOTROPIC_DIRECTIONAL_NULL_CONFIRMED"
                ]
            ),
            "rate_scale_warning": rate_scale_warning,
            "posterior_logvar_saturation_warning": logvar_saturation_warning,
            "analytic_kl_audit": analytic_audit,
            "initial_posterior_audit": initial_audit,
            "gates": gates,
            "B5_A0_SCIENTIFIC_CANDIDATE": False,
            "scientific_candidate_note": "smoke ineligible",
            "B5_A1_STARTED": False,
        }
        _write_json(condition_dir / "metadata.json", metadata)
        run_conditions[condition] = {
            "metadata": metadata,
            "arms": arm_audits,
            "corruption_posthoc_available": corruption_mask is not None,
            "corruption_mask_sha256": corruption_audit.get("mask_sha256"),
        }

        print("condition=" + condition)
        print("backbone path=" + entry["backbone_dir"])
        print("backbone hash=" + backbone_hash_before["aggregate"])
        print("z shape=" + str([list(value.shape) for value in z_views]))
        print("z hash=" + z_hash)
        print(
            "z_l2_norm_max_abs_error_from_one="
            + str(z_norm_audit["z_l2_norm_max_abs_error_from_one"])
        )
        print("z_hash_b4_compatible=" + z_hash_b4_compatible)
        print(
            "expected_z_hash_b4_compatible="
            + expected_z_hash_b4_compatible
        )
        print(
            "B5_A0_CROSS_STAGE_Z_MATCH_PASS="
            + str(cross_stage_z_exact_match).lower()
        )
        print(
            "B5_A0_ISOTROPIC_DIRECTIONAL_NULL_CONFIRMED="
            + str(isotropic_null_audit[
                "B5_A0_ISOTROPIC_DIRECTIONAL_NULL_CONFIRMED"
            ]).lower()
        )
        print("semantic_dim=" + str(EXPECTED_SEMANTIC_DIM))
        print("radius_target=" + str(math.sqrt(EXPECTED_SEMANTIC_DIM)))
        for arm in args.arms:
            audit = arm_audits[arm]
            print(arm + " initial posterior hash=" + audit["initial_posterior_hash"])
        print("initial posterior hash exact match=" + str(initial_match).lower())
        for arm in args.arms:
            print(arm + " first epsilon hash=" + epsilon_hashes[arm])
        print("sampling RNG exact match=" + str(sampling_match).lower())
        print(
            "analytic KL max error="
            + str(analytic_audit["analytic_kl_reference_max_abs_error"])
        )
        for arm in args.arms:
            audit = arm_audits[arm]
            first = audit["first_trace"]
            final = audit["final_trace"]
            print(
                arm + " step0 distortion/rate/beta-rate/total/ratio="
                + str((
                    first["distortion_loss"],
                    first["rate_loss_per_dim"],
                    first["beta_rate"],
                    first["total_loss"],
                    first["rate_to_distortion_ratio"],
                ))
            )
            print(
                arm + " step" + str(args.steps - 1)
                + " distortion/rate/beta-rate/total/ratio="
                + str((
                    final["distortion_loss"],
                    final["rate_loss_per_dim"],
                    final["beta_rate"],
                    final["total_loss"],
                    final["rate_to_distortion_ratio"],
                ))
            )
            print(
                arm + " fixed-radius audit="
                + str(audit["posterior_mu_radius_audit"])
            )
            print(
                arm + " logvar audit="
                + str(audit["posterior_logvar_audit"])
            )
            print(
                arm + " posterior grad finite="
                + str(audit["posterior_gradient_finite_pass"]).lower()
            )
            print(arm + " final posterior hash=" + audit["final_posterior_hash"])
            print(
                arm + " posterior updated="
                + str(audit["posterior_updated_pass"]).lower()
            )
        print(
            "conditional prior grad finite="
            + str(conditional_audit["conditional_prior_gradient_finite_pass"]).lower()
        )
        print(
            "conditional context detach pass="
            + str(conditional_audit["conditional_context_detach_pass"]).lower()
        )
        print(
            "backbone before/after exact="
            + str(backbone_hash_before == backbone_hash_after).lower()
        )
        print("RATE_SCALE_WARNING=" + str(rate_scale_warning).lower())
        print(
            "POSTERIOR_LOGVAR_SATURATION_WARNING="
            + str(logvar_saturation_warning).lower()
        )
        for gate_name, gate_value in gates.items():
            print(gate_name + "=" + str(gate_value).lower())

        _require(engineering_pass, "B5-A0 engineering gate failed for " + condition)
        _require(not rate_scale_warning, "RATE_SCALE_WARNING requires STOP")
        _require(
            not logvar_saturation_warning,
            "POSTERIOR_LOGVAR_SATURATION_WARNING requires STOP",
        )

    overall_engineering_pass = all(
        record["metadata"]["gates"]["B5_A0_ENGINEERING_PASS"]
        for record in run_conditions.values()
    )
    run_result = {
        "stage": "B5-A0",
        "scope": "protected_shared_semantic_rate_mechanism_probe",
        "conditions": run_conditions,
        "steps": args.steps,
        "B5_A0_ENGINEERING_PASS": bool(overall_engineering_pass),
        "B5_A0_SCIENTIFIC_CANDIDATE": False,
        "scientific_candidate_note": "smoke ineligible",
        "B5_A0_RATE_CHANNEL_CANDIDATE": False,
        "B5_A1_STARTED": False,
    }
    _require(all_finite_nested(run_result), "run JSON contains non-finite values")
    _write_json(output_root / "b5_a0_run.json", run_result)
    print("B5_A0_ENGINEERING_PASS=" + str(overall_engineering_pass).lower())
    print("B5_A0_SCIENTIFIC_CANDIDATE=false (smoke ineligible)")
    print("B5_A1_STARTED=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
