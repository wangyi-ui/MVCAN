"""B5-A0.4 offline relation-level utility alignment audit.

This evaluator is diagnostic only.  It reconstructs the frozen B5-A0
representations and conditional-rate modules, reuses the exact B5-A0.2
permutation bank, and tests label-free B3/B4 evidence against sample-specific
conditional path value.  It performs no training, backward pass, routing, or
parameter update.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.b5_semantic_rate.evaluate_b5_a0_context_specificity import (
    _load_and_validate_input,
    _load_final_conditional_modules,
    _reconstruct_frozen_z,
    compute_correct_context_rates,
    expected_canonical_z_hash,
)
from experiments.b5_semantic_rate.evaluate_b5_a02_side_target_attribution import (
    file_sha256,
    ordered_source_target_relations,
    permutation_bank_sha256,
    single_source_shuffled_contexts,
    validate_permutation_bank,
)
from irv.b3_audit import hash_backbone, hash_state_dict
from irv.b3_predictability_diagnostics import (
    fold_assignment_sha256,
    make_fold_assignment,
    normalize_representation_views,
    oof_ridge_predictability,
)
from irv.b4_information_utility import (
    compute_information_utility,
    tensor_sha256,
)
from irv.b5_shared_semantic_rate import analytic_kl_diag_gaussian


STAGE = "B5-A0.4"
CONDITION = "snr2p5_k2"
MODEL_SEED = 20
SUPPORTED_MODEL_SEEDS = (20, 30, 50)
EXPECTED_SAMPLE_NUM = 210
EXPECTED_VIEW_NUM = 5
EXPECTED_RELATION_COUNT = 20
EXPECTED_INSTANCE_COUNT = 4200
EXPECTED_PERMUTATIONS = 500
RIDGE_ALPHA = 1.0
RIDGE_FIT_INTERCEPT = True
EXPECTED_TRAINING_STEPS = 100
# Backward-compatible seed20 regression aliases.
EXPECTED_CANONICAL_Z_HASH = (
    "6dd6f0d4c9fea4a5b44d61a0984daa7137541dcb0762489d668967fb610bc894"
)
EXPECTED_PERMUTATION_BANK_HASH = (
    "887e2c52c2ecc08f93f3351609e5ce499aa280423fb8c5c994aefb051d907635"
)
EXPECTED_B3_FOLD_HASH = (
    "40acb98215fa67831547a9f92ef6777ce14448c7a021be551a8a0ecee38fd2c8"
)
RELATION_BENEFIT_TOLERANCE = 1e-6
EVIDENCE_NULL_SEED = 20260816
DEFAULT_B3_FOLD_REFERENCE = (
    REPOSITORY_ROOT
    / "outputs/b3_semantic/a21_snr2p5_k2_seed20/fold_assignment.csv"
)
DEFAULT_B4_UTILITY_REFERENCE = (
    REPOSITORY_ROOT
    / "outputs/b4_information_utility/a1a_frozen_utility_seed20/"
    "snr2p5_k2/utility_audit.json"
)


def default_b3_score_reference(model_seed):
    model_seed = int(model_seed)
    if model_seed not in SUPPORTED_MODEL_SEEDS:
        raise ValueError("unsupported model seed")
    return (
        REPOSITORY_ROOT
        / "outputs/b3_semantic/a23_reproduction/"
        / ("a23_scores_snr2p5_seed" + str(model_seed) + ".npz")
    )


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_requested_model_seed(metadata, requested_model_seed):
    """Require the explicit evaluator seed to match checkpoint metadata."""
    requested_model_seed = int(requested_model_seed)
    if requested_model_seed not in SUPPORTED_MODEL_SEEDS:
        raise ValueError("unsupported model seed")
    actual_model_seed = int(metadata.get("model_seed", -1))
    _require(
        actual_model_seed == requested_model_seed,
        "requested model seed does not match checkpoint metadata",
    )
    return actual_model_seed


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _load_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(str(path) + " must contain a JSON object")
    return value


def _strict_json_value(value):
    """Recursively replace non-finite scalars with null for strict JSON."""
    if isinstance(value, dict):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json_value(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    return value


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(
            _strict_json_value(value),
            output_file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        output_file.write("\n")


def _off_diagonal(view_num):
    return ~np.eye(int(view_num), dtype=bool)


def load_exact_permutation_bank(
    a02_dir,
    sample_num=EXPECTED_SAMPLE_NUM,
    expected_hash=EXPECTED_PERMUTATION_BANK_HASH,
    expected_repeats=EXPECTED_PERMUTATIONS,
):
    """Load and validate the existing A0.2 bank; never generate a new bank."""
    bank_path = _resolve(a02_dir) / "permutation_bank.npy"
    _require(bank_path.is_file(), "A0.2 permutation_bank.npy is missing")
    array = np.load(bank_path, allow_pickle=False)
    _require(
        array.shape == (int(expected_repeats), int(sample_num)),
        "A0.2 permutation bank shape mismatch",
    )
    _require(array.dtype == np.int64, "A0.2 permutation bank dtype mismatch")
    bank = torch.from_numpy(np.ascontiguousarray(array)).to(dtype=torch.int64)
    validate_permutation_bank(bank, sample_num)
    actual_hash = permutation_bank_sha256(bank)
    _require(actual_hash == expected_hash, "A0.2 permutation bank hash mismatch")
    return bank, actual_hash, bank_path


def target_sample_rates_for_contexts(
    posterior_mu_views,
    posterior_logvar_views,
    prior,
    target_view,
    context_batches,
    semantic_dim,
):
    """Return conditional KL/dimension for every context and sample: [B,N]."""
    target_view = int(target_view)
    semantic_dim = int(semantic_dim)
    if not torch.is_tensor(context_batches) or context_batches.ndim != 3:
        raise ValueError("context_batches must have shape [B,N,Ds]")
    batch_num, sample_num, context_dim = context_batches.shape
    if context_dim != semantic_dim or semantic_dim <= 0:
        raise ValueError("context semantic dimension mismatch")
    if not (0 <= target_view < len(posterior_mu_views)):
        raise IndexError("target view is outside posterior view lists")
    mean_q = posterior_mu_views[target_view]
    logvar_q = posterior_logvar_views[target_view]
    if mean_q.shape != (sample_num, context_dim) or logvar_q.shape != mean_q.shape:
        raise ValueError("target posterior and context shapes do not align")
    flat_context = context_batches.reshape(batch_num * sample_num, context_dim)
    with torch.no_grad():
        mean_p, logvar_p = prior.forward_one(flat_context, target_view)
        expanded_mean_q = mean_q.unsqueeze(0).expand(
            batch_num, -1, -1
        ).reshape(batch_num * sample_num, context_dim)
        expanded_logvar_q = logvar_q.unsqueeze(0).expand(
            batch_num, -1, -1
        ).reshape(batch_num * sample_num, context_dim)
        rates = analytic_kl_diag_gaussian(
            expanded_mean_q,
            expanded_logvar_q,
            mean_p,
            logvar_p,
        ).reshape(batch_num, sample_num)
        rates = rates / float(semantic_dim)
    if not bool(torch.isfinite(rates).all().item()):
        raise RuntimeError("sample-level conditional rates are non-finite")
    return rates.detach()


def compute_relation_benefit(
    posterior_mu_views,
    posterior_logvar_views,
    prior,
    radius_target,
    semantic_dim,
    permutation_bank,
):
    """Compute G_i^(source->target) using the exact shared A0.2 bank."""
    view_num = len(posterior_mu_views)
    if view_num < 2 or len(posterior_logvar_views) != view_num:
        raise ValueError("posterior view lists must align and contain >=2 views")
    sample_num = int(posterior_mu_views[0].shape[0])
    validate_permutation_bank(permutation_bank, sample_num)
    correct_global, correct_per_view, correct_contexts = (
        compute_correct_context_rates(
            posterior_mu_views,
            posterior_logvar_views,
            prior,
            radius_target,
            semantic_dim,
        )
    )
    del correct_global, correct_per_view
    correct_sample_rates = np.empty((sample_num, view_num), dtype=np.float64)
    with torch.no_grad():
        for target_view in range(view_num):
            rates = target_sample_rates_for_contexts(
                posterior_mu_views,
                posterior_logvar_views,
                prior,
                target_view,
                correct_contexts[target_view].unsqueeze(0),
                semantic_dim,
            )
            correct_sample_rates[:, target_view] = (
                rates[0].cpu().numpy().astype(np.float64, copy=False)
            )

    side_semantic_views = [
        value.detach() / float(radius_target) for value in posterior_mu_views
    ]
    benefit = np.full(
        (sample_num, view_num, view_num), np.nan, dtype=np.float64
    )
    for source_view, target_view in ordered_source_target_relations(view_num):
        contexts = single_source_shuffled_contexts(
            side_semantic_views,
            target_view,
            source_view,
            permutation_bank,
        )
        shuffled_rates = target_sample_rates_for_contexts(
            posterior_mu_views,
            posterior_logvar_views,
            prior,
            target_view,
            contexts,
            semantic_dim,
        )
        shuffled_mean = shuffled_rates.mean(dim=0).cpu().numpy().astype(
            np.float64, copy=False
        )
        benefit[:, source_view, target_view] = (
            shuffled_mean - correct_sample_rates[:, target_view]
        )
    off_diagonal = _off_diagonal(view_num)
    _require(
        np.isnan(benefit[:, ~off_diagonal]).all(),
        "relation benefit diagonal must be NaN",
    )
    _require(
        np.isfinite(benefit[:, off_diagonal]).all(),
        "relation benefit off-diagonal must be finite",
    )
    return benefit, correct_sample_rates


def relation_benefit_reproduction_error(relation_benefit, aggregate_delta):
    """Return max off-diagonal error between mean_i G and A0.2 delta."""
    benefit = np.asarray(relation_benefit, dtype=np.float64)
    reference = np.asarray(aggregate_delta, dtype=np.float64)
    if benefit.ndim != 3 or reference.shape != benefit.shape[1:]:
        raise ValueError("relation benefit/reference shapes do not align")
    view_num = benefit.shape[1]
    if benefit.shape[2] != view_num:
        raise ValueError("relation benefit must have shape [N,V,V]")
    off_diagonal = _off_diagonal(view_num)
    reproduced = np.full(reference.shape, np.nan, dtype=np.float64)
    reproduced[off_diagonal] = benefit[:, off_diagonal].mean(axis=0)
    error = float(np.max(np.abs(reproduced[off_diagonal] - reference[off_diagonal])))
    return error, reproduced


def load_canonical_b3_folds(
    sample_num=EXPECTED_SAMPLE_NUM,
    model_seed=MODEL_SEED,
    reference_path=None,
):
    """Construct seed-specific B3 folds and exact-match the saved artifact."""
    model_seed = int(model_seed)
    if model_seed not in SUPPORTED_MODEL_SEEDS:
        raise ValueError("unsupported model seed")
    folds = make_fold_assignment(sample_num, n_splits=5, random_state=model_seed)
    fold_hash = fold_assignment_sha256(folds)
    reference_path = _resolve(
        default_b3_score_reference(model_seed)
        if reference_path is None
        else reference_path
    )
    _require(reference_path.is_file(), "canonical B3 fold reference is missing")
    if reference_path.suffix == ".npz":
        with np.load(reference_path, allow_pickle=False) as archive:
            _require(
                "fold_assignment" in archive.files,
                "B3 score artifact has no fold assignment",
            )
            reference_folds = np.asarray(
                archive["fold_assignment"], dtype=np.int64
            )
    else:
        table = np.genfromtxt(
            reference_path, delimiter=",", names=True, dtype=np.int64
        )
        reference_ids = np.asarray(table["sample_id"], dtype=np.int64)
        reference_folds = np.asarray(table["fold_id"], dtype=np.int64)
        _require(
            np.array_equal(
                reference_ids, np.arange(sample_num, dtype=np.int64)
            ),
            "canonical B3 fold reference sample IDs mismatch",
        )
    _require(
        reference_folds.shape == (sample_num,)
        and np.array_equal(reference_folds, folds),
        "canonical B3 fold assignment mismatch",
    )
    _require(
        fold_assignment_sha256(reference_folds) == fold_hash,
        "canonical B3 fold hash mismatch",
    )
    return folds, fold_hash, reference_path


def fold_integrity_audit(fold_assignment):
    """Audit disjoint train/test sets and exact one-fold OOF coverage."""
    folds = np.asarray(fold_assignment, dtype=np.int64)
    if folds.ndim != 1 or folds.size == 0:
        raise ValueError("fold_assignment must have shape [N]")
    coverage = np.zeros(folds.size, dtype=np.int64)
    disjoint = True
    for fold_id in np.unique(folds):
        test_ids = np.flatnonzero(folds == fold_id)
        train_ids = np.flatnonzero(folds != fold_id)
        disjoint = bool(
            disjoint
            and test_ids.size > 0
            and train_ids.size > 0
            and np.intersect1d(train_ids, test_ids).size == 0
        )
        coverage[test_ids] += 1
    return {
        "train_test_disjoint_pass": disjoint,
        "all_samples_covered_exactly_once_pass": bool(np.all(coverage == 1)),
        "coverage": coverage,
    }


def pairwise_predictability_from_oof(
    representation_views,
    fold_assignment,
    alpha=RIDGE_ALPHA,
):
    """Return directional OOF cosine T with orientation [N,source,target]."""
    if float(alpha) != RIDGE_ALPHA:
        raise ValueError("B5-A0.4 requires Ridge alpha=1.0")
    normalized = normalize_representation_views(representation_views)
    sample_num, view_num, _ = normalized.shape
    result = oof_ridge_predictability(
        representation_views,
        fold_assignment,
        alpha=float(alpha),
    )
    predictions = np.asarray(result["predicted_target_views"], dtype=np.float64)
    source_ids = np.asarray(result["source_view_ids"], dtype=np.int64)
    pairwise = np.full(
        (sample_num, view_num, view_num), np.nan, dtype=np.float64
    )
    for target_view in range(view_num):
        for source_position, source_view in enumerate(source_ids[target_view]):
            pairwise[:, source_view, target_view] = np.sum(
                predictions[:, target_view, source_position]
                * normalized[:, target_view],
                axis=1,
            )
    off_diagonal = _off_diagonal(view_num)
    _require(
        np.isnan(pairwise[:, ~off_diagonal]).all(),
        "pairwise predictability diagonal must be NaN",
    )
    _require(
        np.isfinite(pairwise[:, off_diagonal]).all(),
        "pairwise predictability must be finite",
    )
    _require(
        np.max(np.abs(pairwise[:, off_diagonal])) <= 1.0 + 1e-10,
        "pairwise cosine is outside [-1,1]",
    )
    return pairwise, result


def rank_percentile_relations(pairwise_predictability):
    """Average-rank percentile independently within every ordered relation."""
    values = np.asarray(pairwise_predictability, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("pairwise_predictability must have shape [N,V,V]")
    sample_num, view_num, _ = values.shape
    if sample_num < 2:
        raise ValueError("at least two samples are required")
    output = np.full_like(values, np.nan, dtype=np.float64)
    for source_view, target_view in ordered_source_target_relations(view_num):
        relation = values[:, source_view, target_view]
        if not np.isfinite(relation).all():
            raise ValueError("relation predictability must be finite")
        ranks = rankdata(relation, method="average")
        output[:, source_view, target_view] = (
            ranks - 1.0
        ) / float(sample_num - 1)
    return output


def reconstruct_source_utility(
    predictability_result,
    b4_reference_path=None,
    b3_score_reference=None,
    fold_hash=None,
):
    """Recompute frozen B4 U with seed-specific B3/B4 provenance."""
    consensus = np.asarray(
        predictability_result["oof_consensus_cosine"], dtype=np.float64
    )
    actual_t_hash = tensor_sha256(consensus)
    expected_utility_hash = None
    expected_fold_hash = None
    if b4_reference_path is not None:
        reference_path = _resolve(b4_reference_path)
        _require(
            reference_path.is_file(), "verifiable B4 Utility reference is missing"
        )
        reference = _load_json(reference_path)
        expected_utility_hash = reference.get("utility_sha256")
        expected_t_hash = reference.get("t_sha256")
        expected_fold_hash = reference.get("fold_sha256")
        _require(
            isinstance(expected_utility_hash, str)
            and len(expected_utility_hash) == 64,
            "B4 utility_sha256 reference is missing",
        )
        provenance = "historical_b4_utility_hash_exact_match"
    else:
        _require(
            b3_score_reference is not None,
            "seed-specific B3 score reference is required",
        )
        reference_path = _resolve(b3_score_reference)
        _require(reference_path.is_file(), "B3 score reference is missing")
        with np.load(reference_path, allow_pickle=False) as archive:
            _require(
                {"T", "fold_assignment"} <= set(archive.files),
                "B3 score reference is incomplete",
            )
            historical_t = np.asarray(archive["T"], dtype=np.float64)
            historical_folds = np.asarray(
                archive["fold_assignment"], dtype=np.int64
            )
        _require(
            np.array_equal(consensus, historical_t),
            "current consensus T does not exact-match historical B3",
        )
        expected_t_hash = tensor_sha256(historical_t)
        expected_fold_hash = fold_assignment_sha256(historical_folds)
        provenance = "deterministic_frozen_b4_formula_from_historical_b3_T"
    _require(actual_t_hash == expected_t_hash, "B3/B4 consensus T mismatch")
    if fold_hash is not None:
        _require(fold_hash == expected_fold_hash, "source U fold provenance mismatch")
    utility = compute_information_utility(consensus)
    utility_hash = tensor_sha256(utility)
    if expected_utility_hash is not None:
        _require(utility_hash == expected_utility_hash, "B4 source U hash mismatch")
    _require(np.isfinite(utility).all(), "source U is non-finite")
    _require(
        float(np.min(utility)) >= 0.0 and float(np.max(utility)) <= 1.0,
        "source U is outside [0,1]",
    )
    return utility, {
        "source_utility_provenance": provenance,
        "formula": "per-view (average_rank(T)-1)/(N-1)",
        "rank_method": "average",
        "reference_path": str(reference_path.relative_to(REPOSITORY_ROOT)),
        "historical_utility_hash_available": expected_utility_hash is not None,
        "expected_utility_sha256": expected_utility_hash,
        "utility_sha256": utility_hash,
        "expected_t_sha256": expected_t_hash,
        "t_sha256": actual_t_hash,
        "expected_fold_sha256": expected_fold_hash,
        "exact_reproduction_pass": True,
    }


def source_relation_evidence(source_utility):
    """Replicate source-only U_i^w over all four targets."""
    utility = np.asarray(source_utility, dtype=np.float64)
    if utility.ndim != 2 or utility.shape[1] < 2:
        raise ValueError("source_utility must have shape [N,V>=2]")
    sample_num, view_num = utility.shape
    evidence = np.full((sample_num, view_num, view_num), np.nan, dtype=np.float64)
    for source_view, target_view in ordered_source_target_relations(view_num):
        evidence[:, source_view, target_view] = utility[:, source_view]
    return evidence


def structural_relation_evidence(structural_matrix, sample_num):
    """Replicate clean relation-level structural scores over samples."""
    structural = np.asarray(structural_matrix, dtype=np.float64)
    if structural.ndim != 2 or structural.shape[0] != structural.shape[1]:
        raise ValueError("structural_matrix must have shape [V,V]")
    view_num = structural.shape[0]
    off_diagonal = _off_diagonal(view_num)
    if not np.isfinite(structural[off_diagonal]).all():
        raise ValueError("structural off-diagonal entries must be finite")
    result = np.repeat(structural[None, :, :], int(sample_num), axis=0)
    result[:, ~off_diagonal] = np.nan
    return result


def relation_fixed_effect(relation_benefit):
    """Remove the sample mean independently from each ordered relation."""
    benefit = np.asarray(relation_benefit, dtype=np.float64)
    if benefit.ndim != 3 or benefit.shape[1] != benefit.shape[2]:
        raise ValueError("relation_benefit must have shape [N,V,V]")
    output = np.full_like(benefit, np.nan, dtype=np.float64)
    for source_view, target_view in ordered_source_target_relations(benefit.shape[1]):
        values = benefit[:, source_view, target_view]
        output[:, source_view, target_view] = values - np.mean(values)
    return output


def flatten_relations(values):
    """Flatten all N x V(V-1) valid relation instances in sample-major order."""
    array = np.asarray(values)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError("values must have shape [N,V,V]")
    return array[:, _off_diagonal(array.shape[1])].reshape(-1)


def safe_auc(binary_outcome, evidence):
    """Return NaN for a single-class outcome instead of raising."""
    outcome = np.asarray(binary_outcome, dtype=np.int64).reshape(-1)
    scores = np.asarray(evidence, dtype=np.float64).reshape(-1)
    if outcome.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("AUC inputs must be equal finite vectors")
    if np.unique(outcome).size < 2:
        return float("nan")
    return float(roc_auc_score(outcome, scores))


def safe_spearman(continuous_outcome, evidence):
    """Return a finite rho when defined, otherwise NaN."""
    outcome = np.asarray(continuous_outcome, dtype=np.float64).reshape(-1)
    scores = np.asarray(evidence, dtype=np.float64).reshape(-1)
    if outcome.shape != scores.shape:
        raise ValueError("Spearman inputs must have equal shape")
    if not np.isfinite(outcome).all() or not np.isfinite(scores).all():
        raise ValueError("Spearman inputs must be finite")
    if np.unique(outcome).size < 2 or np.unique(scores).size < 2:
        return float("nan")
    return float(spearmanr(scores, outcome).statistic)


def alignment_metrics(evidence, continuous_outcome):
    """Compute binary positive-value AUC and continuous Spearman rho."""
    evidence_flat = flatten_relations(evidence)
    outcome_flat = flatten_relations(continuous_outcome).astype(np.float64)
    binary = outcome_flat > 0.0
    return {
        "auc": safe_auc(binary, evidence_flat),
        "spearman": safe_spearman(outcome_flat, evidence_flat),
        "positive_count": int(np.sum(binary)),
        "negative_count": int(np.sum(~binary)),
        "instance_count": int(binary.size),
    }


def relation_distribution_summaries(relation_benefit):
    benefit = np.asarray(relation_benefit, dtype=np.float64)
    rows = []
    for source_view, target_view in ordered_source_target_relations(benefit.shape[1]):
        values = benefit[:, source_view, target_view]
        rows.append({
            "source_view": source_view,
            "target_view": target_view,
            "relation": str(source_view) + "->" + str(target_view),
            "mean_g": float(np.mean(values)),
            "std_g": float(np.std(values)),
            "p25_g": float(np.percentile(values, 25)),
            "p50_g": float(np.percentile(values, 50)),
            "p75_g": float(np.percentile(values, 75)),
            "positive_fraction": float(np.mean(values > 0.0)),
        })
    return rows


def per_relation_diagnostics(
    relation_benefit,
    pairwise_predictability,
    pairwise_rank,
    source_utility,
):
    """Return the preregistered raw per-relation pair/source diagnostics."""
    benefit = np.asarray(relation_benefit, dtype=np.float64)
    raw_pair = np.asarray(pairwise_predictability, dtype=np.float64)
    rank_pair = np.asarray(pairwise_rank, dtype=np.float64)
    utility = np.asarray(source_utility, dtype=np.float64)
    rows = []
    for source_view, target_view in ordered_source_target_relations(benefit.shape[1]):
        g_values = benefit[:, source_view, target_view]
        binary = g_values > 0.0
        rows.append({
            "source_view": source_view,
            "target_view": target_view,
            "relation": str(source_view) + "->" + str(target_view),
            "mean_g": float(np.mean(g_values)),
            "std_g": float(np.std(g_values)),
            "p25_g": float(np.percentile(g_values, 25)),
            "p50_g": float(np.percentile(g_values, 50)),
            "p75_g": float(np.percentile(g_values, 75)),
            "positive_fraction": float(np.mean(binary)),
            "positive_count": int(np.sum(binary)),
            "negative_count": int(np.sum(~binary)),
            "spearman_raw_t_g": safe_spearman(
                g_values, raw_pair[:, source_view, target_view]
            ),
            "auc_rank_t_positive_g": safe_auc(
                binary, rank_pair[:, source_view, target_view]
            ),
            "spearman_source_u_g": safe_spearman(
                g_values, utility[:, source_view]
            ),
            "auc_source_u_positive_g": safe_auc(
                binary, utility[:, source_view]
            ),
        })
    return rows


def sample_bootstrap_indices(sample_num, repeats, seed):
    """Draw [repeats,N] sample IDs; each ID retains all ordered relations."""
    sample_num = int(sample_num)
    repeats = int(repeats)
    if sample_num <= 1 or repeats <= 0:
        raise ValueError("sample_num and repeats must be positive")
    generator = np.random.default_rng(int(seed))
    return generator.integers(
        0,
        sample_num,
        size=(repeats, sample_num),
        dtype=np.int64,
    )


def bootstrap_alignment_metrics(
    relation_benefit,
    pair_evidence,
    source_evidence,
    bootstrap_sample_ids,
):
    """Paired sample bootstrap; FE means are recomputed in every replicate."""
    benefit = np.asarray(relation_benefit, dtype=np.float64)
    pair = np.asarray(pair_evidence, dtype=np.float64)
    source = np.asarray(source_evidence, dtype=np.float64)
    sample_ids = np.asarray(bootstrap_sample_ids, dtype=np.int64)
    if benefit.shape != pair.shape or benefit.shape != source.shape:
        raise ValueError("bootstrap relation matrices must have equal shape")
    if sample_ids.ndim != 2 or sample_ids.shape[1] != benefit.shape[0]:
        raise ValueError("bootstrap_sample_ids must have shape [B,N]")
    if sample_ids.min() < 0 or sample_ids.max() >= benefit.shape[0]:
        raise ValueError("bootstrap sample ID outside valid range")
    repeats = sample_ids.shape[0]
    output = {
        "auc_fe_pair": np.empty(repeats, dtype=np.float64),
        "spearman_fe_pair": np.empty(repeats, dtype=np.float64),
        "auc_fe_source": np.empty(repeats, dtype=np.float64),
        "spearman_fe_source": np.empty(repeats, dtype=np.float64),
    }
    for repeat, selected_ids in enumerate(sample_ids):
        selected_benefit = benefit[selected_ids]
        selected_fe = relation_fixed_effect(selected_benefit)
        pair_metrics = alignment_metrics(pair[selected_ids], selected_fe)
        source_metrics = alignment_metrics(source[selected_ids], selected_fe)
        output["auc_fe_pair"][repeat] = pair_metrics["auc"]
        output["spearman_fe_pair"][repeat] = pair_metrics["spearman"]
        output["auc_fe_source"][repeat] = source_metrics["auc"]
        output["spearman_fe_source"][repeat] = source_metrics["spearman"]
    output["delta_auc_pair_minus_source"] = (
        output["auc_fe_pair"] - output["auc_fe_source"]
    )
    output["delta_spearman_pair_minus_source"] = (
        output["spearman_fe_pair"] - output["spearman_fe_source"]
    )
    return output


def percentile_ci(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap metric must be a non-empty vector")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"lower": None, "upper": None, "finite_repeats": 0}
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "finite_repeats": int(finite.size),
    }


def within_relation_shuffle_evidence(evidence, generator):
    """Shuffle only sample rows, independently inside each ordered relation."""
    values = np.asarray(evidence, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("evidence must have shape [N,V,V]")
    output = np.full_like(values, np.nan, dtype=np.float64)
    for source_view, target_view in ordered_source_target_relations(values.shape[1]):
        permutation = generator.permutation(values.shape[0])
        output[:, source_view, target_view] = values[
            permutation, source_view, target_view
        ]
    return output


def evidence_permutation_null(
    relation_benefit_fe,
    pair_evidence,
    repeats,
    seed=EVIDENCE_NULL_SEED,
):
    """Relation-preserving evidence null for the global FE pair metrics."""
    benefit_fe = np.asarray(relation_benefit_fe, dtype=np.float64)
    pair = np.asarray(pair_evidence, dtype=np.float64)
    if benefit_fe.shape != pair.shape:
        raise ValueError("null relation matrices must have equal shape")
    repeats = int(repeats)
    if repeats <= 0:
        raise ValueError("null repeats must be positive")
    generator = np.random.default_rng(int(seed))
    auc_values = np.empty(repeats, dtype=np.float64)
    rho_values = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        shuffled = within_relation_shuffle_evidence(pair, generator)
        metrics = alignment_metrics(shuffled, benefit_fe)
        auc_values[repeat] = metrics["auc"]
        rho_values[repeat] = metrics["spearman"]
    return {"auc_fe_pair": auc_values, "spearman_fe_pair": rho_values}


def summarize_null(null_values, observed):
    values = np.asarray(null_values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not math.isfinite(float(observed)):
        return {"mean": None, "p95": None, "empirical_p_value": None}
    return {
        "mean": float(np.mean(finite)),
        "p95": float(np.percentile(finite, 95)),
        "empirical_p_value": float(
            (1 + np.count_nonzero(finite >= float(observed)))
            / (finite.size + 1)
        ),
    }


def save_artifacts(
    output_dir,
    relation_benefit,
    relation_benefit_fe,
    pairwise_predictability,
    pairwise_rank,
    source_utility,
    per_relation_rows,
    bootstrap_values,
    bootstrap_sample_ids,
    null_values,
):
    """Save every preregistered A0.4 array/table artifact."""
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "relation_benefit.npy", relation_benefit)
    np.save(
        output_dir / "relation_benefit_positive_mask.npy",
        np.asarray(relation_benefit) > 0.0,
    )
    np.save(output_dir / "relation_benefit_fe.npy", relation_benefit_fe)
    np.save(output_dir / "pairwise_predictability.npy", pairwise_predictability)
    np.save(output_dir / "pairwise_predictability_rank.npy", pairwise_rank)
    np.save(output_dir / "source_utility.npy", source_utility)
    csv_path = output_dir / "per_relation_metrics.csv"
    fieldnames = list(per_relation_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in per_relation_rows:
            writer.writerow(_strict_json_value(row))
    np.savez(
        output_dir / "bootstrap_metrics.npz",
        sample_ids=np.asarray(bootstrap_sample_ids, dtype=np.int64),
        **bootstrap_values,
    )
    np.savez(output_dir / "evidence_null_metrics.npz", **null_values)
    return output_dir


def _format_ci(point, interval):
    return (
        f"{point:.9f} [{interval['lower']:.9f}, "
        f"{interval['upper']:.9f}]"
    )


def _print_report(result, per_relation_rows):
    print("canonical z hash=" + result["z_hash"])
    print("B3 fold hash=" + result["fold_hash"])
    print("permutation bank hash=" + result["permutation_bank_hash"])
    print(
        "relation benefit reproduction max error="
        + f"{result['relation_benefit_reproduction_error']:.12g}"
    )
    print(
        "source-U reproduction result/hash="
        + str(result["B5_A04_SOURCE_U_REPRO_PASS"])
        + "/"
        + result["source_utility"]["utility_sha256"]
    )
    print("N=" + str(result["sample_count"]))
    print("relation count=" + str(result["relation_count"]))
    print("instance count=" + str(result["instance_count"]))
    print("G positive fraction=" + f"{result['global_class_balance']['positive_fraction']:.9f}")
    raw = result["raw_metrics"]
    print("RAW:")
    for evidence in ("pair", "source", "struct"):
        print(f"AUC_raw_{evidence}={raw[evidence]['auc']:.9f}")
        print(f"Spearman_raw_{evidence}={raw[evidence]['spearman']:.9f}")
    adjusted = result["fe_adjusted_metrics"]
    intervals = result["bootstrap_cis"]
    print("FE:")
    print("AUC_FE_pair=" + _format_ci(adjusted["pair"]["auc"], intervals["auc_fe_pair"]))
    print(
        "Spearman_FE_pair="
        + _format_ci(adjusted["pair"]["spearman"], intervals["spearman_fe_pair"])
    )
    print(
        "AUC_FE_source="
        + _format_ci(adjusted["source"]["auc"], intervals["auc_fe_source"])
    )
    print(
        "Spearman_FE_source="
        + _format_ci(adjusted["source"]["spearman"], intervals["spearman_fe_source"])
    )
    print(
        "Delta_AUC pair-source="
        + _format_ci(
            result["paired_differences"]["delta_auc"],
            intervals["delta_auc_pair_minus_source"],
        )
    )
    print(
        "Delta_Spearman pair-source="
        + _format_ci(
            result["paired_differences"]["delta_spearman"],
            intervals["delta_spearman_pair_minus_source"],
        )
    )
    null = result["permutation_null_metrics"]
    print("Permutation null:")
    print(
        "pair AUC null mean/p95/p="
        + f"{null['auc_fe_pair']['mean']:.9f}/"
        + f"{null['auc_fe_pair']['p95']:.9f}/"
        + f"{null['auc_fe_pair']['empirical_p_value']:.9f}"
    )
    print(
        "pair Spearman null mean/p95/p="
        + f"{null['spearman_fe_pair']['mean']:.9f}/"
        + f"{null['spearman_fe_pair']['p95']:.9f}/"
        + f"{null['spearman_fe_pair']['empirical_p_value']:.9f}"
    )
    print("20 relation diagnostics:")
    for row in per_relation_rows:
        auc_pair = row["auc_rank_t_positive_g"]
        auc_source = row["auc_source_u_positive_g"]
        print(
            row["relation"]
            + f" mean_G={row['mean_g']:.9f} positive_fraction={row['positive_fraction']:.6f}"
            + f" pair_rho={row['spearman_raw_t_g']:.9f}"
            + " pair_auc="
            + ("null" if not math.isfinite(auc_pair) else f"{auc_pair:.9f}")
            + f" source_rho={row['spearman_source_u_g']:.9f}"
            + " source_auc="
            + ("null" if not math.isfinite(auc_source) else f"{auc_source:.9f}")
        )
    for flag_name in (
        "B5_A04_PERMUTATION_BANK_MATCH_PASS",
        "B5_A04_RELATION_BENEFIT_REPRO_PASS",
        "B5_A04_B3_FOLD_MATCH_PASS",
        "B5_A04_SOURCE_U_REPRO_PASS",
        "B5_A04_PAIRWISE_T_SIGNAL_PASS",
        "B5_A04_SOURCE_U_SIGNAL_PASS",
        "B5_A04_PAIRWISE_T_AUC_ADDS_OVER_SOURCE_U_PASS",
        "B5_A04_PAIRWISE_T_RHO_ADDS_OVER_SOURCE_U_PASS",
    ):
        print(flag_name + "=" + str(result[flag_name]))
    print("B5_A04_ALIGNMENT_AUDIT_COMPLETE=" + str(result["B5_A04_ALIGNMENT_AUDIT_COMPLETE"]).lower())
    print("B5_A1_STARTED=false")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--condition", choices=(CONDITION,), required=True)
    parser.add_argument(
        "--model-seed", type=int, choices=SUPPORTED_MODEL_SEEDS, required=True
    )
    parser.add_argument("--a02-dir", required=True)
    parser.add_argument("--clean-a02-dir", required=True)
    parser.add_argument("--specificity-json")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    parser.add_argument("--evidence-null-repeats", type=int, default=500)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--b3-fold-reference")
    parser.add_argument("--b4-utility-reference")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _require(args.condition == CONDITION, "B5-A0.4 primary condition mismatch")
    _require(args.model_seed in SUPPORTED_MODEL_SEEDS, "unsupported model seed")
    _require(args.bootstrap_repeats > 0, "bootstrap repeats must be positive")
    _require(args.evidence_null_repeats > 0, "null repeats must be positive")
    input_dir = _resolve(args.input_dir)
    condition_dir, arm_dir, metadata = _load_and_validate_input(
        input_dir, args.condition
    )
    validate_requested_model_seed(metadata, args.model_seed)
    _require(
        int(metadata.get("steps", -1)) == EXPECTED_TRAINING_STEPS,
        "B5-A0.4 requires the 100-step checkpoint",
    )
    expected_z_hash = expected_canonical_z_hash(metadata, args.condition)

    models, backbone_hash_before, z_views, z_hash, z_audit = (
        _reconstruct_frozen_z(metadata, args.condition)
    )
    _require(z_hash == expected_z_hash, "canonical Native-z hash mismatch")
    posterior, prior, posterior_hash_before, prior_hash_before = (
        _load_final_conditional_modules(arm_dir, metadata)
    )
    posterior_checkpoint = arm_dir / "posterior.pth"
    prior_checkpoint = arm_dir / "conditional_prior.pth"
    posterior_file_hash_before = file_sha256(posterior_checkpoint)
    prior_file_hash_before = file_sha256(prior_checkpoint)

    permutation_bank, bank_hash, bank_path = load_exact_permutation_bank(
        args.a02_dir
    )
    with torch.no_grad():
        posterior_mu_views, posterior_logvar_views = posterior(z_views)
        relation_benefit, correct_sample_rates = compute_relation_benefit(
            posterior_mu_views,
            posterior_logvar_views,
            prior,
            posterior.radius_target,
            posterior.semantic_dim,
            permutation_bank,
        )

    a02_result_path = (
        _resolve(args.a02_dir) / "b5_a02_side_target_attribution.json"
    )
    _require(a02_result_path.is_file(), "A0.2 provenance JSON is missing")
    a02_result = _load_json(a02_result_path)
    a02_seed = a02_result.get("model_seed")
    a02_seed_match = bool(
        a02_seed == args.model_seed
        or (args.model_seed == MODEL_SEED and a02_seed is None)
    )
    _require(a02_seed_match, "A0.2 model seed provenance mismatch")
    _require(
        a02_result.get("condition") == args.condition
        and a02_result.get("canonical_z_hash") == z_hash
        and a02_result.get("permutation_bank_hash") == bank_hash,
        "A0.2 condition/z/permutation provenance mismatch",
    )
    noisy_delta_path = _resolve(args.a02_dir) / "delta_matrix.npy"
    _require(
        noisy_delta_path.is_file(),
        "full-precision A0.2 delta_matrix.npy is missing",
    )
    noisy_delta = np.load(noisy_delta_path, allow_pickle=False).astype(
        np.float64, copy=False
    )
    reproduction_error, reproduced_delta = relation_benefit_reproduction_error(
        relation_benefit, noisy_delta
    )
    benefit_repro_pass = bool(reproduction_error < RELATION_BENEFIT_TOLERANCE)
    _require(benefit_repro_pass, "sample-level G does not reproduce A0.2 delta")

    b3_score_reference = _resolve(
        default_b3_score_reference(args.model_seed)
        if args.b3_fold_reference is None
        else args.b3_fold_reference
    )
    folds, fold_hash, fold_reference_path = load_canonical_b3_folds(
        model_seed=args.model_seed,
        reference_path=b3_score_reference,
    )
    fold_audit = fold_integrity_audit(folds)
    _require(
        fold_audit["train_test_disjoint_pass"]
        and fold_audit["all_samples_covered_exactly_once_pass"],
        "B3 OOF fold integrity failed",
    )
    pairwise, predictability_result = pairwise_predictability_from_oof(
        z_views, folds, alpha=RIDGE_ALPHA
    )
    pairwise_rank = rank_percentile_relations(pairwise)
    b4_reference = args.b4_utility_reference
    if b4_reference is None and args.model_seed == MODEL_SEED:
        b4_reference = DEFAULT_B4_UTILITY_REFERENCE
    source_utility, source_utility_audit = reconstruct_source_utility(
        predictability_result,
        b4_reference_path=b4_reference,
        b3_score_reference=b3_score_reference,
        fold_hash=fold_hash,
    )
    source_evidence = source_relation_evidence(source_utility)

    clean_delta_path = _resolve(args.clean_a02_dir) / "delta_matrix.npy"
    _require(clean_delta_path.is_file(), "clean A0.3 delta_matrix.npy is missing")
    clean_delta = np.load(clean_delta_path, allow_pickle=False).astype(
        np.float64, copy=False
    )
    structural_evidence = structural_relation_evidence(
        clean_delta, relation_benefit.shape[0]
    )
    relation_benefit_fe = relation_fixed_effect(relation_benefit)

    global_specificity = None
    if args.specificity_json is not None:
        specificity_path = _resolve(args.specificity_json)
        _require(specificity_path.is_file(), "specificity JSON is missing")
        specificity = _load_json(specificity_path)
        _require(
            specificity.get("condition") == args.condition
            and int(specificity.get("model_seed", -1)) == args.model_seed
            and specificity.get("canonical_z_hash") == z_hash,
            "specificity condition/seed/z provenance mismatch",
        )
        global_specificity = {
            "reference_path": str(
                specificity_path.relative_to(REPOSITORY_ROOT)
            ),
            "R_correct": float(specificity["R_correct"]),
            "shuffle_rate_mean": float(specificity["shuffle_rate_mean"]),
            "delta_mean": float(specificity["delta_mean"]),
            "relative_delta_mean": float(specificity["relative_delta_mean"]),
            "fraction_shuffle_gt_correct": float(
                specificity["fraction_shuffle_gt_correct"]
            ),
            "permutation_p_value": float(specificity["permutation_p_value"]),
            "specificity_pass": bool(
                specificity["B5_A0_CONDITIONAL_CONTEXT_SPECIFICITY_PASS"]
            ),
        }

    raw_metrics = {
        "pair": alignment_metrics(pairwise_rank, relation_benefit),
        "source": alignment_metrics(source_evidence, relation_benefit),
        "struct": alignment_metrics(structural_evidence, relation_benefit),
    }
    fe_metrics = {
        "pair": alignment_metrics(pairwise_rank, relation_benefit_fe),
        "source": alignment_metrics(source_evidence, relation_benefit_fe),
    }
    per_relation_rows = per_relation_diagnostics(
        relation_benefit,
        pairwise,
        pairwise_rank,
        source_utility,
    )

    bootstrap_ids = sample_bootstrap_indices(
        relation_benefit.shape[0],
        args.bootstrap_repeats,
        args.bootstrap_seed,
    )
    bootstrap_values = bootstrap_alignment_metrics(
        relation_benefit,
        pairwise_rank,
        source_evidence,
        bootstrap_ids,
    )
    bootstrap_cis = {
        name: percentile_ci(values) for name, values in bootstrap_values.items()
    }
    delta_auc = fe_metrics["pair"]["auc"] - fe_metrics["source"]["auc"]
    delta_spearman = (
        fe_metrics["pair"]["spearman"] - fe_metrics["source"]["spearman"]
    )

    null_values = evidence_permutation_null(
        relation_benefit_fe,
        pairwise_rank,
        args.evidence_null_repeats,
        seed=EVIDENCE_NULL_SEED,
    )
    null_summary = {
        "auc_fe_pair": summarize_null(
            null_values["auc_fe_pair"], fe_metrics["pair"]["auc"]
        ),
        "spearman_fe_pair": summarize_null(
            null_values["spearman_fe_pair"], fe_metrics["pair"]["spearman"]
        ),
    }

    backbone_hash_after = hash_backbone(models.autoencoders)
    posterior_hash_after = hash_state_dict(posterior.state_dict())
    prior_hash_after = hash_state_dict(prior.state_dict())
    posterior_file_hash_after = file_sha256(posterior_checkpoint)
    prior_file_hash_after = file_sha256(prior_checkpoint)
    no_parameter_gradients_pass = all(
        parameter.grad is None
        for module in list(models.autoencoders) + [posterior, prior]
        for parameter in module.parameters()
    )
    parameter_state_unchanged_pass = bool(
        backbone_hash_before == backbone_hash_after
        and posterior_hash_before == posterior_hash_after
        and prior_hash_before == prior_hash_after
        and posterior_file_hash_before == posterior_file_hash_after
        and prior_file_hash_before == prior_file_hash_after
    )
    _require(no_parameter_gradients_pass, "diagnostic created parameter gradients")
    _require(parameter_state_unchanged_pass, "frozen parameter state changed")

    off_diagonal = _off_diagonal(EXPECTED_VIEW_NUM)
    shape_pass = bool(
        relation_benefit.shape == (EXPECTED_SAMPLE_NUM, EXPECTED_VIEW_NUM, EXPECTED_VIEW_NUM)
        and relation_benefit_fe.shape == relation_benefit.shape
        and pairwise.shape == relation_benefit.shape
        and pairwise_rank.shape == relation_benefit.shape
        and source_utility.shape == (EXPECTED_SAMPLE_NUM, EXPECTED_VIEW_NUM)
        and np.isnan(relation_benefit[:, ~off_diagonal]).all()
        and np.isnan(pairwise[:, ~off_diagonal]).all()
    )
    relation_count = len(ordered_source_target_relations(EXPECTED_VIEW_NUM))
    instance_count = EXPECTED_SAMPLE_NUM * relation_count
    finite_pass = bool(
        np.isfinite(relation_benefit[:, off_diagonal]).all()
        and np.isfinite(relation_benefit_fe[:, off_diagonal]).all()
        and np.isfinite(pairwise[:, off_diagonal]).all()
        and np.isfinite(pairwise_rank[:, off_diagonal]).all()
        and np.isfinite(source_utility).all()
    )
    _require(shape_pass and finite_pass, "A0.4 output matrix audit failed")
    _require(relation_count == EXPECTED_RELATION_COUNT, "ordered relation count mismatch")
    _require(instance_count == EXPECTED_INSTANCE_COUNT, "relation instance count mismatch")

    pair_signal_pass = bool(
        bootstrap_cis["auc_fe_pair"]["lower"] > 0.5
        and bootstrap_cis["spearman_fe_pair"]["lower"] > 0.0
    )
    source_signal_pass = bool(
        bootstrap_cis["auc_fe_source"]["lower"] > 0.5
        and bootstrap_cis["spearman_fe_source"]["lower"] > 0.0
    )
    pair_auc_adds_pass = bool(
        bootstrap_cis["delta_auc_pair_minus_source"]["lower"] > 0.0
    )
    pair_rho_adds_pass = bool(
        bootstrap_cis["delta_spearman_pair_minus_source"]["lower"] > 0.0
    )
    positive_count = int(np.sum(relation_benefit[:, off_diagonal] > 0.0))
    negative_count = instance_count - positive_count
    engineering_complete = bool(
        bank_hash == EXPECTED_PERMUTATION_BANK_HASH
        and benefit_repro_pass
        and fold_hash == source_utility_audit["expected_fold_sha256"]
        and source_utility_audit["exact_reproduction_pass"]
        and z_hash == expected_z_hash
        and shape_pass
        and finite_pass
        and no_parameter_gradients_pass
        and parameter_state_unchanged_pass
    )
    _require(engineering_complete, "B5-A0.4 engineering completion failed")

    result = {
        "stage": STAGE,
        "condition": args.condition,
        "model_seed": args.model_seed,
        "training_performed": False,
        "backward_performed": False,
        "utility_used_for_training": False,
        "rate_modified": False,
        "context_modified": False,
        "B5_A1_STARTED": False,
        "labels_loaded": False,
        "labels_used": False,
        "corruption_mask_loaded": False,
        "corruption_mask_used": False,
        "input_dir": str(input_dir.relative_to(REPOSITORY_ROOT)),
        "condition_dir": str(condition_dir.relative_to(REPOSITORY_ROOT)),
        "a02_dir": str(_resolve(args.a02_dir).relative_to(REPOSITORY_ROOT)),
        "clean_a02_dir": str(_resolve(args.clean_a02_dir).relative_to(REPOSITORY_ROOT)),
        "sample_count": EXPECTED_SAMPLE_NUM,
        "view_count": EXPECTED_VIEW_NUM,
        "relation_count": relation_count,
        "instance_count": instance_count,
        "ridge": {
            "alpha": RIDGE_ALPHA,
            "fit_intercept": RIDGE_FIT_INTERCEPT,
            "fold_count": 5,
            "sample_level_oof": True,
            "ordered_pair_direction": "source->target",
        },
        "z_hash": z_hash,
        "z_sha256": z_hash,
        "expected_z_hash": expected_z_hash,
        "z_l2_normalization_audit": z_audit,
        "fold_hash": fold_hash,
        "fold_sha256": fold_hash,
        "expected_fold_hash": source_utility_audit["expected_fold_sha256"],
        "fold_reference": str(fold_reference_path.relative_to(REPOSITORY_ROOT)),
        "fold_integrity": {
            "train_test_disjoint_pass": fold_audit["train_test_disjoint_pass"],
            "all_samples_covered_exactly_once_pass": fold_audit[
                "all_samples_covered_exactly_once_pass"
            ],
            "oof_sample_count_per_ordered_relation": EXPECTED_SAMPLE_NUM,
            "ordered_relation_count": EXPECTED_RELATION_COUNT,
        },
        "permutation_bank_path": str(bank_path.relative_to(REPOSITORY_ROOT)),
        "permutation_bank_shape": list(permutation_bank.shape),
        "permutation_bank_hash": bank_hash,
        "expected_permutation_bank_hash": EXPECTED_PERMUTATION_BANK_HASH,
        "relation_benefit_definition": "mean_b(R_i,b_source-shuffled->target - R_correct_i,target)",
        "relation_benefit_shape": list(relation_benefit.shape),
        "relation_benefit_dtype": str(relation_benefit.dtype),
        "relation_benefit_reproduction_error": reproduction_error,
        "relation_benefit_reproduction_tolerance": RELATION_BENEFIT_TOLERANCE,
        "reproduced_delta_matrix": reproduced_delta,
        "a02_delta_matrix": noisy_delta,
        "correct_sample_rate_shape": list(correct_sample_rates.shape),
        "pairwise_predictability_shape": list(pairwise.shape),
        "pairwise_evidence_definition": "per-relation (average_rank(T)-1)/(N-1)",
        "source_utility": source_utility_audit,
        "source_utility_provenance": source_utility_audit[
            "source_utility_provenance"
        ],
        "source_utility_sha256": source_utility_audit["utility_sha256"],
        "source_evidence_definition": "E_source_i^(w->v)=U_i^w",
        "structural_reference_path": str(clean_delta_path.relative_to(REPOSITORY_ROOT)),
        "structural_evidence_definition": "E_struct_i^(w->v)=M_clean[w,v]",
        "structural_FE_not_applicable": True,
        "global_specificity": global_specificity,
        "global_class_balance": {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_fraction": float(positive_count / instance_count),
        },
        "relation_distribution_summaries": relation_distribution_summaries(
            relation_benefit
        ),
        "raw_metrics": raw_metrics,
        "fe_adjusted_metrics": fe_metrics,
        "bootstrap": {
            "unit": "sample_id_with_all_20_relations",
            "repeats": args.bootstrap_repeats,
            "seed": args.bootstrap_seed,
            "relation_means_recomputed_per_replicate": True,
            "paired_pair_source_resamples": True,
        },
        "bootstrap_cis": bootstrap_cis,
        "paired_differences": {
            "delta_auc": float(delta_auc),
            "delta_spearman": float(delta_spearman),
        },
        "permutation_null": {
            "unit": "evidence_shuffled_within_each_relation",
            "repeats": args.evidence_null_repeats,
            "seed": EVIDENCE_NULL_SEED,
        },
        "permutation_null_metrics": null_summary,
        "per_relation_metrics": per_relation_rows,
        "backbone_hash_before": backbone_hash_before,
        "backbone_hash_after": backbone_hash_after,
        "posterior_state_hash_before": posterior_hash_before,
        "posterior_state_hash_after": posterior_hash_after,
        "conditional_prior_state_hash_before": prior_hash_before,
        "conditional_prior_state_hash_after": prior_hash_after,
        "posterior_checkpoint_sha256_before": posterior_file_hash_before,
        "posterior_checkpoint_sha256_after": posterior_file_hash_after,
        "conditional_prior_checkpoint_sha256_before": prior_file_hash_before,
        "conditional_prior_checkpoint_sha256_after": prior_file_hash_after,
        "no_parameter_gradients_pass": no_parameter_gradients_pass,
        "parameter_state_unchanged_pass": parameter_state_unchanged_pass,
        "all_saved_matrix_shapes_pass": shape_pass,
        "all_off_diagonal_finite_pass": finite_pass,
        "B5_A04_PERMUTATION_BANK_MATCH_PASS": True,
        "B5_A04_RELATION_BENEFIT_REPRO_PASS": benefit_repro_pass,
        "B5_A04_B3_FOLD_MATCH_PASS": True,
        "B5_A04_SOURCE_U_REPRO_PASS": source_utility_audit[
            "exact_reproduction_pass"
        ],
        "B5_A04_PAIRWISE_T_SIGNAL_PASS": pair_signal_pass,
        "B5_A04_SOURCE_U_SIGNAL_PASS": source_signal_pass,
        "B5_A04_PAIRWISE_T_AUC_ADDS_OVER_SOURCE_U_PASS": pair_auc_adds_pass,
        "B5_A04_PAIRWISE_T_RHO_ADDS_OVER_SOURCE_U_PASS": pair_rho_adds_pass,
        "B5_A04_ALIGNMENT_AUDIT_COMPLETE": engineering_complete,
    }

    output_dir = save_artifacts(
        args.output_dir,
        relation_benefit,
        relation_benefit_fe,
        pairwise,
        pairwise_rank,
        source_utility,
        per_relation_rows,
        bootstrap_values,
        bootstrap_ids,
        null_values,
    )
    result_path = output_dir / "b5_a04_relation_utility_alignment.json"
    _write_json(result_path, result)
    _print_report(result, per_relation_rows)
    return result


if __name__ == "__main__":
    main()
