"""Offline permutation audit of B5-A0 conditional-context specificity."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from configure import get_default_config
from irv.b3_audit import hash_backbone, hash_state_dict
from irv.b5_shared_semantic_rate import ConditionalPriorBank
from irv.b5_shared_semantic_rate import GaussianSemanticPosteriorBank
from irv.b5_shared_semantic_rate import build_conditional_context
from irv.b5_shared_semantic_rate import canonical_view_tensor_sha256
from irv.b5_shared_semantic_rate import conditional_rate_views
from irv.b5_shared_semantic_rate import normalize_native_z
from irv.b5_shared_semantic_rate import z_l2_normalization_audit
from model import MvCAN
from weak_quality import apply_weak_quality_protocol


DATASET_NAME = "MSRC-v1"
EXPECTED_SAMPLE_NUM = 210
EXPECTED_VIEW_NUM = 5
EXPECTED_CLUSTER_NUM = 7
EXPECTED_LATENT_DIM = 10
EXPECTED_SEMANTIC_DIM = 10
EXPECTED_MODEL_SEED = 20
SUPPORTED_MODEL_SEEDS = (20, 30, 50)
EXPECTED_PERMUTATIONS = 500
EXPECTED_PERMUTATION_SEED = 20260816
SUPPORTED_CONDITIONS = ("clean", "snr2p5_k2")
EXPECTED_NOISY_CANONICAL_Z_HASH = (
    "6dd6f0d4c9fea4a5b44d61a0984daa7137541dcb0762489d668967fb610bc894"
)
# Backward-compatible alias for the historical noisy evaluator/tests.
EXPECTED_CANONICAL_Z_HASH = EXPECTED_NOISY_CANONICAL_Z_HASH


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _load_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(str(path) + " must contain a JSON object")
    return value


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def load_msrc_feature_views():
    """Load only the MATLAB `fea` variable; ground-truth `gt` is not read."""
    candidates = (
        REPOSITORY_ROOT / "data/MSRC-v1.mat",
        REPOSITORY_ROOT / "data/MSRC_v1.mat",
    )
    data_path = next((path for path in candidates if path.is_file()), None)
    _require(data_path is not None, "missing MSRC-v1 feature file")
    matlab = sio.loadmat(str(data_path), variable_names=["fea"])
    _require("fea" in matlab and "gt" not in matlab, "feature-only load failed")
    raw_views = matlab["fea"]
    _require(
        isinstance(raw_views, np.ndarray)
        and raw_views.dtype == object
        and raw_views.size == EXPECTED_VIEW_NUM,
        "MSRC-v1 fea must contain five views",
    )
    feature_views = []
    for raw_view in raw_views.ravel():
        value = np.asarray(raw_view)
        _require(value.ndim == 2, "each feature view must be a matrix")
        if value.shape[0] == EXPECTED_SAMPLE_NUM:
            oriented = value
        elif value.shape[1] == EXPECTED_SAMPLE_NUM:
            oriented = value.T
        else:
            raise RuntimeError("feature view has no sample axis of length 210")
        oriented = np.ascontiguousarray(oriented.astype(np.float32, copy=False))
        _require(np.isfinite(oriented).all(), "feature view must be finite")
        feature_views.append(oriented)
    return feature_views


def expected_canonical_z_hash(metadata, condition):
    """Read the current condition's registered canonical hash from metadata."""
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError("unsupported B5-A0 evaluation condition")
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    expected = metadata.get("expected_z_hash_b4_compatible")
    if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)):
        raise ValueError("metadata canonical z hash must be lowercase sha256")
    _require(
        metadata.get("z_hash_b4_compatible") == expected,
        "metadata canonical z hash fields disagree",
    )
    model_seed = int(metadata.get("model_seed", EXPECTED_MODEL_SEED))
    _require(model_seed in SUPPORTED_MODEL_SEEDS, "unsupported metadata model seed")
    if condition == "snr2p5_k2" and model_seed == EXPECTED_MODEL_SEED:
        _require(
            expected == EXPECTED_NOISY_CANONICAL_Z_HASH,
            "seed20 noisy canonical z regression registration changed",
        )
    return expected


def condition_evaluation_views(clean_feature_views, condition,
                               model_seed=EXPECTED_MODEL_SEED):
    """Use one feature path, adding only the registered noisy perturbation."""
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError("unsupported B5-A0 evaluation condition")
    if not isinstance(clean_feature_views, (list, tuple)):
        raise TypeError("clean_feature_views must be a list or tuple")
    if len(clean_feature_views) != EXPECTED_VIEW_NUM:
        raise ValueError("MSRC-v1 must contain exactly five feature views")
    model_seed = int(model_seed)
    if model_seed not in SUPPORTED_MODEL_SEEDS:
        raise ValueError("unsupported model seed")
    if condition == "clean":
        return list(clean_feature_views)
    # No stored corruption state is loaded or returned. The noisy condition is
    # deterministically regenerated from the same feature-only input path.
    return apply_weak_quality_protocol(
        clean_feature_views,
        mode="heterogeneous_gaussian",
        k=2,
        snr_db=2.5,
        corruption_seed=model_seed,
    )[0]


def generate_sample_permutations(sample_num, permutation_count,
                                 permutation_seed):
    """Return local-RNG sample permutations with shape [B,N]."""
    sample_num = int(sample_num)
    permutation_count = int(permutation_count)
    if sample_num <= 1 or permutation_count <= 0:
        raise ValueError("sample_num and permutation_count must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(permutation_seed))
    return torch.stack([
        torch.randperm(sample_num, generator=generator)
        for _ in range(permutation_count)
    ], dim=0)


def permute_context_views(context_views, sample_permutation):
    """Apply one shared sample-axis permutation to every target context."""
    if not isinstance(context_views, (list, tuple)) or not context_views:
        raise ValueError("context_views must be a non-empty list or tuple")
    if not torch.is_tensor(sample_permutation) or sample_permutation.ndim != 1:
        raise ValueError("sample_permutation must have shape [N]")
    sample_num = int(context_views[0].shape[0])
    if sample_permutation.shape != (sample_num,):
        raise ValueError("sample_permutation length mismatch")
    canonical = torch.arange(sample_num, device=sample_permutation.device)
    if not torch.equal(torch.sort(sample_permutation).values, canonical):
        raise ValueError("sample_permutation must contain every sample once")
    shuffled = []
    for context in context_views:
        if not torch.is_tensor(context) or context.ndim != 2:
            raise ValueError("each context must have shape [N,D]")
        if int(context.shape[0]) != sample_num:
            raise ValueError("all contexts must share sample count")
        # Only dimension 0 changes. The identical permutation is used for all
        # five target-view contexts, preserving every context row exactly.
        shuffled.append(context[sample_permutation].detach())
    return shuffled


def conditional_rate_matrix(posterior_mu_views, posterior_logvar_views,
                            prior, context_views, semantic_dim):
    """Return KL/dimension for every sample-view as an [N,V] tensor."""
    prior_mu_views, prior_logvar_views = prior(context_views)
    raw_rate = conditional_rate_views(
        posterior_mu_views,
        posterior_logvar_views,
        prior_mu_views,
        prior_logvar_views,
    )
    return raw_rate / int(semantic_dim)


def compute_correct_context_rates(posterior_mu_views,
                                  posterior_logvar_views, prior,
                                  radius_target, semantic_dim):
    """Compute deterministic correct-context global and per-view rates."""
    context_views = build_conditional_context(
        posterior_mu_views, radius_target
    )
    with torch.no_grad():
        rate_matrix = conditional_rate_matrix(
            posterior_mu_views,
            posterior_logvar_views,
            prior,
            context_views,
            semantic_dim,
        )
    per_view = rate_matrix.mean(dim=0)
    return rate_matrix.mean(), per_view, context_views


def compute_permuted_context_rates(posterior_mu_views,
                                   posterior_logvar_views, prior,
                                   context_views, sample_permutations,
                                   semantic_dim):
    """Compute null rates using one permutation across all target views."""
    if not torch.is_tensor(sample_permutations) or sample_permutations.ndim != 2:
        raise ValueError("sample_permutations must have shape [B,N]")
    global_rates = []
    per_view_rates = []
    with torch.no_grad():
        for sample_permutation in sample_permutations:
            shuffled_contexts = permute_context_views(
                context_views, sample_permutation
            )
            rate_matrix = conditional_rate_matrix(
                posterior_mu_views,
                posterior_logvar_views,
                prior,
                shuffled_contexts,
                semantic_dim,
            )
            target_rates = rate_matrix.mean(dim=0)
            per_view_rates.append(target_rates)
            global_rates.append(target_rates.mean())
    # shuffle_rates: [B]; per_view_shuffle_rates: [B,V]
    return torch.stack(global_rates), torch.stack(per_view_rates)


def permutation_test_summary(correct_rate, shuffle_rates):
    """Summarize the preregistered one-sided R_correct < R_shuffle test."""
    correct_rate = float(correct_rate)
    values = np.asarray(shuffle_rates, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("shuffle_rates must be a finite non-empty [B] array")
    percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    less_or_equal_count = int(np.count_nonzero(values <= correct_rate))
    p_value = float((1 + less_or_equal_count) / (values.size + 1))
    delta_mean = float(np.mean(values) - correct_rate)
    result = {
        "correct_rate": correct_rate,
        "shuffle_rate_mean": float(np.mean(values)),
        "shuffle_rate_std": float(np.std(values)),
        "shuffle_rate_min": float(np.min(values)),
        "shuffle_rate_max": float(np.max(values)),
        "shuffle_rate_p01": float(percentiles[0]),
        "shuffle_rate_p05": float(percentiles[1]),
        "shuffle_rate_p25": float(percentiles[2]),
        "shuffle_rate_p50": float(percentiles[3]),
        "shuffle_rate_p75": float(percentiles[4]),
        "shuffle_rate_p95": float(percentiles[5]),
        "shuffle_rate_p99": float(percentiles[6]),
        "delta_mean": delta_mean,
        "relative_delta_mean": float(delta_mean / correct_rate),
        "shuffle_less_or_equal_correct_count": less_or_equal_count,
        "fraction_shuffle_gt_correct": float(np.mean(values > correct_rate)),
        "permutation_p_value": p_value,
    }
    result["B5_A0_CONDITIONAL_CONTEXT_SPECIFICITY_PASS"] = bool(
        correct_rate < result["shuffle_rate_p05"] and p_value < 0.05
    )
    return result


def per_target_view_summaries(correct_rates, per_view_shuffle_rates):
    """Return descriptive one-sided permutation results for each target view."""
    correct = np.asarray(correct_rates, dtype=np.float64)
    shuffled = np.asarray(per_view_shuffle_rates, dtype=np.float64)
    if shuffled.ndim != 2 or correct.shape != (shuffled.shape[1],):
        raise ValueError("correct [V] and shuffled [B,V] shapes must align")
    results = []
    for view_id in range(shuffled.shape[1]):
        values = shuffled[:, view_id]
        correct_value = float(correct[view_id])
        less_or_equal_count = int(np.count_nonzero(values <= correct_value))
        results.append({
            "target_view": int(view_id),
            "correct_rate_v": correct_value,
            "shuffle_mean_v": float(np.mean(values)),
            "delta_v": float(np.mean(values) - correct_value),
            "fraction_shuffle_gt_correct_v": float(
                np.mean(values > correct_value)
            ),
            "permutation_p_value_v": float(
                (1 + less_or_equal_count) / (values.size + 1)
            ),
        })
    return results


def correct_rate_reproduction_audit(correct_rate, correct_per_view,
                                    stored_correct_rate, tolerance=1e-6):
    """Check current-condition stored/global/per-view consistency."""
    correct_rate = float(correct_rate)
    stored_correct_rate = float(stored_correct_rate)
    tolerance = float(tolerance)
    per_view = np.asarray(correct_per_view, dtype=np.float64)
    if per_view.ndim != 1 or per_view.size == 0:
        raise ValueError("correct_per_view must be a non-empty [V] array")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    finite_pass = bool(
        math.isfinite(correct_rate)
        and math.isfinite(stored_correct_rate)
        and np.isfinite(per_view).all()
    )
    per_view_mean = float(np.mean(per_view))
    stored_error = abs(correct_rate - stored_correct_rate)
    mean_error = abs(per_view_mean - correct_rate)
    result = {
        "stored_current_condition_R_correct": stored_correct_rate,
        "computed_R_correct": correct_rate,
        "computed_R_correct_per_view": [float(value) for value in per_view],
        "computed_per_view_mean": per_view_mean,
        "stored_rate_abs_error": stored_error,
        "per_view_mean_abs_error": mean_error,
        "per_view_rates_finite_pass": finite_pass,
    }
    result["correct_rate_reproduction_pass"] = bool(
        finite_pass and stored_error < tolerance and mean_error < tolerance
    )
    return result


def _stored_correct_rate(arm_dir):
    rate_audit = _load_json(arm_dir / "rate_audit.json")
    if "conditional_context_correct_rate" in rate_audit:
        return float(rate_audit["conditional_context_correct_rate"])
    arm_audit = _load_json(arm_dir / "arm_audit.json")
    nested = arm_audit.get("rate_audit", {})
    _require(
        "conditional_context_correct_rate" in nested,
        "current-condition stored correct rate is missing",
    )
    return float(nested["conditional_context_correct_rate"])


def _load_and_validate_input(input_dir, condition):
    condition_dir = input_dir / condition
    metadata = _load_json(condition_dir / "metadata.json")
    gates = metadata.get("gates", {})
    _require(condition in SUPPORTED_CONDITIONS, "unsupported input condition")
    _require(metadata.get("stage") == "B5-A0", "input stage mismatch")
    _require(metadata.get("condition") == condition, "input condition mismatch")
    _require(gates.get("B5_A0_ENGINEERING_PASS") is True, "input engineering fail")
    _require(
        metadata.get("cross_stage_z_exact_match") is True,
        "input cross-stage z audit failed",
    )
    expected_canonical_z_hash(metadata, condition)
    arm_dir = condition_dir / "conditional_rate"
    required = (
        arm_dir / "posterior.pth",
        arm_dir / "conditional_prior.pth",
        arm_dir / "arm_audit.json",
        arm_dir / "rate_audit.json",
    )
    _require(all(path.is_file() for path in required), "missing conditional state")
    return condition_dir, arm_dir, metadata


def _reconstruct_frozen_z(metadata, condition):
    clean_views = load_msrc_feature_views()
    model_seed = int(metadata.get("model_seed", -1))
    _require(model_seed in SUPPORTED_MODEL_SEEDS, "unsupported metadata model seed")
    evaluation_views = condition_evaluation_views(
        clean_views, condition, model_seed=model_seed
    )
    expected_z_hash = expected_canonical_z_hash(metadata, condition)
    config = get_default_config(DATASET_NAME)
    models = MvCAN(
        config,
        view_num=EXPECTED_VIEW_NUM,
        view_size=[int(value.shape[1]) for value in evaluation_views],
        n_clusters=EXPECTED_CLUSTER_NUM,
        seed=model_seed,
        data_size=EXPECTED_SAMPLE_NUM,
        semantic_config=None,
    )
    backbone_dir = _resolve(metadata["backbone_path"])
    for view_id, autoencoder in enumerate(models.autoencoders):
        checkpoint = backbone_dir / (DATASET_NAME + str(view_id + 1) + "V.pth")
        autoencoder.load_state_dict(
            torch.load(str(checkpoint), map_location="cpu"), strict=True
        )
        autoencoder.eval()
        autoencoder.requires_grad_(False)
    backbone_hash = hash_backbone(models.autoencoders)
    _require(
        backbone_hash == metadata["backbone_hash_after"],
        "reconstructed backbone hash mismatch",
    )
    with torch.no_grad():
        z_views = []
        for view_id, autoencoder in enumerate(models.autoencoders):
            features = torch.from_numpy(evaluation_views[view_id]).float()
            raw_z_v = autoencoder.encoder(features)
            z_views.append(normalize_native_z(raw_z_v))
    z_audit = z_l2_normalization_audit(z_views)
    z_hash = canonical_view_tensor_sha256(z_views)
    _require(z_audit["z_normalization_pass"], "reconstructed z not normalized")
    _require(z_hash == expected_z_hash, "reconstructed z hash mismatch")
    return models, backbone_hash, z_views, z_hash, z_audit


def _load_final_conditional_modules(arm_dir, metadata):
    arm_audit = _load_json(arm_dir / "arm_audit.json")
    posterior = GaussianSemanticPosteriorBank(
        EXPECTED_VIEW_NUM,
        EXPECTED_LATENT_DIM,
        int(metadata["semantic_dim"]),
        int(metadata["semantic_seed"]),
        float(metadata["posterior_logvar_min"]),
        float(metadata["posterior_logvar_max"]),
    )
    posterior.load_state_dict(
        torch.load(str(arm_dir / "posterior.pth"), map_location="cpu"),
        strict=True,
    )
    prior = ConditionalPriorBank(
        EXPECTED_VIEW_NUM,
        int(metadata["semantic_dim"]),
        int(metadata["prior_seed"]),
        float(metadata["posterior_logvar_min"]),
        float(metadata["posterior_logvar_max"]),
    )
    prior.load_state_dict(
        torch.load(str(arm_dir / "conditional_prior.pth"), map_location="cpu"),
        strict=True,
    )
    posterior.eval().requires_grad_(False)
    prior.eval().requires_grad_(False)
    posterior_hash = hash_state_dict(posterior.state_dict())
    prior_hash = hash_state_dict(prior.state_dict())
    _require(
        posterior_hash == arm_audit["final_posterior_hash"],
        "final posterior state hash mismatch",
    )
    _require(
        prior_hash == arm_audit["final_conditional_prior_hash"],
        "final conditional prior state hash mismatch",
    )
    return posterior, prior, posterior_hash, prior_hash


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--condition", choices=SUPPORTED_CONDITIONS, required=True)
    parser.add_argument("--permutations", type=int, required=True)
    parser.add_argument("--permutation-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _require(
        args.permutations == EXPECTED_PERMUTATIONS,
        "specificity audit preregisters 500 permutations",
    )
    _require(
        args.permutation_seed == EXPECTED_PERMUTATION_SEED,
        "specificity audit permutation seed mismatch",
    )
    input_dir = _resolve(args.input_dir)
    condition_dir, arm_dir, metadata = _load_and_validate_input(
        input_dir, args.condition
    )
    expected_z_hash = expected_canonical_z_hash(metadata, args.condition)
    models, backbone_hash, z_views, z_hash, z_audit = _reconstruct_frozen_z(
        metadata, args.condition
    )
    posterior, prior, posterior_hash, prior_hash = (
        _load_final_conditional_modules(arm_dir, metadata)
    )

    with torch.no_grad():
        # posterior_mu/logvar: List[V=5], each [N=210,Ds=10].
        posterior_mu_views, posterior_logvar_views = posterior(z_views)
        correct_rate, correct_per_view, context_views = (
            compute_correct_context_rates(
                posterior_mu_views,
                posterior_logvar_views,
                prior,
                posterior.radius_target,
                posterior.semantic_dim,
            )
        )
    correct_value = float(correct_rate.item())
    correct_per_view_values = correct_per_view.cpu().numpy().astype(
        np.float64, copy=False
    )
    stored_correct = _stored_correct_rate(arm_dir)
    reproduction = correct_rate_reproduction_audit(
        correct_value, correct_per_view_values, stored_correct
    )
    correct_error = reproduction["stored_rate_abs_error"]
    correct_repro_pass = reproduction["correct_rate_reproduction_pass"]
    _require(
        correct_repro_pass,
        "current-condition correct-context rate not reproduced",
    )

    sample_permutations = generate_sample_permutations(
        EXPECTED_SAMPLE_NUM, args.permutations, args.permutation_seed
    )
    shuffle_rates_tensor, per_view_shuffle_tensor = (
        compute_permuted_context_rates(
            posterior_mu_views,
            posterior_logvar_views,
            prior,
            context_views,
            sample_permutations,
            posterior.semantic_dim,
        )
    )
    shuffle_rates = shuffle_rates_tensor.cpu().numpy().astype(
        np.float64, copy=False
    )
    per_view_shuffle_rates = per_view_shuffle_tensor.cpu().numpy().astype(
        np.float64, copy=False
    )
    _require(
        shuffle_rates.shape == (EXPECTED_PERMUTATIONS,)
        and per_view_shuffle_rates.shape
        == (EXPECTED_PERMUTATIONS, EXPECTED_VIEW_NUM),
        "permutation output shape mismatch",
    )
    summary = permutation_test_summary(correct_value, shuffle_rates)
    per_view = per_target_view_summaries(
        correct_per_view_values, per_view_shuffle_rates
    )
    backbone_after = hash_backbone(models.autoencoders)
    no_grad_pass = all(
        parameter.grad is None
        for module in list(models.autoencoders) + [posterior, prior]
        for parameter in module.parameters()
    )

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "shuffle_rates.npy", shuffle_rates)
    np.save(output_dir / "per_view_shuffle_rates.npy", per_view_shuffle_rates)
    result = {
        "stage": "B5-A0 Conditional Context Specificity Audit",
        "condition": args.condition,
        "model_seed": int(metadata["model_seed"]),
        "training_performed": False,
        "deterministic_posterior_evaluation": True,
        "posterior_sampling_used": False,
        "corruption_mask_used": False,
        "corruption_mask_loaded": False,
        "labels_used": False,
        "labels_loaded": False,
        "permutations": args.permutations,
        "permutation_seed": args.permutation_seed,
        "permutation_axis": "sample_dimension_only",
        "same_permutation_across_all_target_views": True,
        "side_context_marginal_rows_preserved": True,
        "input_dir": str(input_dir.relative_to(REPOSITORY_ROOT)),
        "condition_dir": str(condition_dir.relative_to(REPOSITORY_ROOT)),
        "posterior_checkpoint": str(
            (arm_dir / "posterior.pth").relative_to(REPOSITORY_ROOT)
        ),
        "conditional_prior_checkpoint": str(
            (arm_dir / "conditional_prior.pth").relative_to(REPOSITORY_ROOT)
        ),
        "input_engineering_pass": True,
        "input_cross_stage_z_exact_match": True,
        "canonical_z_hash": z_hash,
        "expected_canonical_z_hash": expected_z_hash,
        "canonical_z_exact_match": z_hash == expected_z_hash,
        "z_l2_normalization_audit": z_audit,
        "backbone_hash_before": backbone_hash,
        "backbone_hash_after": backbone_after,
        "backbone_hash_unchanged_pass": backbone_hash == backbone_after,
        "posterior_state_hash": posterior_hash,
        "conditional_prior_state_hash": prior_hash,
        "no_parameter_gradients_pass": no_grad_pass,
        "R_correct": correct_value,
        "stored_v2_R_correct": stored_correct,
        "stored_current_condition_R_correct": stored_correct,
        "R_correct_per_view": [float(value) for value in correct_per_view_values],
        "R_correct_per_view_mean": reproduction["computed_per_view_mean"],
        "correct_rate_max_abs_error": correct_error,
        "correct_rate_per_view_mean_abs_error": reproduction[
            "per_view_mean_abs_error"
        ],
        "correct_rate_per_view_finite_pass": reproduction[
            "per_view_rates_finite_pass"
        ],
        "B5_A0_SPECIFICITY_CORRECT_RATE_REPRO_PASS": correct_repro_pass,
        **summary,
        "per_target_view": per_view,
        "shuffle_rates_shape": list(shuffle_rates.shape),
        "per_view_shuffle_rates_shape": list(per_view_shuffle_rates.shape),
        "B5_A0_SCIENTIFIC_PASS_DECLARED": False,
        "B5_A1_STARTED": False,
    }
    _require(
        all(
            math.isfinite(value)
            for value in (
                result["R_correct"],
                result["shuffle_rate_mean"],
                result["shuffle_rate_std"],
                result["permutation_p_value"],
            )
        ),
        "specificity statistics must be finite",
    )
    _write_json(output_dir / "b5_a0_context_specificity.json", result)

    print("R_correct=" + str(correct_value))
    print("stored-current-condition R_correct=" + str(stored_correct))
    print("max abs error=" + str(correct_error))
    print(
        "shuffle mean/std="
        + str((summary["shuffle_rate_mean"], summary["shuffle_rate_std"]))
    )
    print(
        "shuffle p05/p50/p95="
        + str((
            summary["shuffle_rate_p05"],
            summary["shuffle_rate_p50"],
            summary["shuffle_rate_p95"],
        ))
    )
    print("delta mean=" + str(summary["delta_mean"]))
    print("relative delta mean=" + str(summary["relative_delta_mean"]))
    print(
        "fraction shuffle > correct="
        + str(summary["fraction_shuffle_gt_correct"])
    )
    print("one-sided p-value=" + str(summary["permutation_p_value"]))
    for record in per_view:
        print(
            "view " + str(record["target_view"])
            + " correct/shuffle-mean/delta/p-value="
            + str((
                record["correct_rate_v"],
                record["shuffle_mean_v"],
                record["delta_v"],
                record["permutation_p_value_v"],
            ))
        )
    print(
        "B5_A0_SPECIFICITY_CORRECT_RATE_REPRO_PASS="
        + str(correct_repro_pass).lower()
    )
    print(
        "B5_A0_CONDITIONAL_CONTEXT_SPECIFICITY_PASS="
        + str(summary[
            "B5_A0_CONDITIONAL_CONTEXT_SPECIFICITY_PASS"
        ]).lower()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
