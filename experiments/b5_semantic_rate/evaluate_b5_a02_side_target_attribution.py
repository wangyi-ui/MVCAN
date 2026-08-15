"""Offline source-to-target attribution for the B5-A0 conditional context."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.b5_semantic_rate.evaluate_b5_a0_context_specificity import (
    EXPECTED_CANONICAL_Z_HASH,
)
from experiments.b5_semantic_rate.evaluate_b5_a0_context_specificity import (
    EXPECTED_MODEL_SEED,
)
from experiments.b5_semantic_rate.evaluate_b5_a0_context_specificity import (
    EXPECTED_SAMPLE_NUM,
)
from experiments.b5_semantic_rate.evaluate_b5_a0_context_specificity import (
    EXPECTED_VIEW_NUM,
)
from experiments.b5_semantic_rate.evaluate_b5_a0_context_specificity import (
    _load_and_validate_input,
)
from experiments.b5_semantic_rate.evaluate_b5_a0_context_specificity import (
    _load_final_conditional_modules,
)
from experiments.b5_semantic_rate.evaluate_b5_a0_context_specificity import (
    _reconstruct_frozen_z,
)
from experiments.b5_semantic_rate.evaluate_b5_a0_context_specificity import (
    compute_correct_context_rates,
)
from experiments.b5_semantic_rate.evaluate_b5_a0_context_specificity import (
    generate_sample_permutations,
)
from irv.b3_audit import hash_backbone, hash_state_dict
from irv.b5_shared_semantic_rate import analytic_kl_diag_gaussian


EXPECTED_PERMUTATIONS = 500
EXPECTED_PERMUTATION_SEED = 20260816
EXPECTED_TRAINING_STEPS = 100
EXPECTED_CORRECT_PER_VIEW = np.asarray(
    [
        0.8893569111824036,
        1.2556895017623901,
        1.158353328704834,
        0.89356929063797,
        1.4715262651443481,
    ],
    dtype=np.float64,
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
    if not isinstance(value, dict):
        raise ValueError(str(path) + " must contain a JSON object")
    return value


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def file_sha256(path):
    """Hash one checkpoint without mutating or deserializing it."""
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def ordered_source_target_relations(view_num):
    """Return every directed source-to-target relation, excluding diagonal."""
    view_num = int(view_num)
    if view_num < 2:
        raise ValueError("view_num must be at least two")
    return [
        (source_view, target_view)
        for source_view in range(view_num)
        for target_view in range(view_num)
        if source_view != target_view
    ]


def _validate_side_semantic_views(side_semantic_views):
    if not isinstance(side_semantic_views, (list, tuple)):
        raise TypeError("side_semantic_views must be a list or tuple")
    if len(side_semantic_views) < 2:
        raise ValueError("at least two semantic views are required")
    first = side_semantic_views[0]
    if not torch.is_tensor(first) or first.ndim != 2:
        raise ValueError("each side semantic view must have shape [N,Ds]")
    if int(first.shape[0]) <= 1 or int(first.shape[1]) <= 0:
        raise ValueError("side semantic dimensions must be positive")
    for value in side_semantic_views:
        if not torch.is_tensor(value) or value.shape != first.shape:
            raise ValueError("all side semantic views must share shape [N,Ds]")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("side semantic views must be finite")
    return len(side_semantic_views), int(first.shape[0]), int(first.shape[1])


def validate_permutation_bank(permutation_bank, sample_num):
    """Validate a shared [B,N] bank of exact sample permutations."""
    if not torch.is_tensor(permutation_bank) or permutation_bank.ndim != 2:
        raise ValueError("permutation_bank must have shape [B,N]")
    sample_num = int(sample_num)
    if int(permutation_bank.shape[0]) <= 0:
        raise ValueError("permutation_bank must contain at least one row")
    if int(permutation_bank.shape[1]) != sample_num:
        raise ValueError("permutation_bank sample dimension mismatch")
    if permutation_bank.dtype != torch.int64:
        raise ValueError("permutation_bank must use torch.int64 indices")
    expected = torch.arange(
        sample_num, dtype=torch.int64, device=permutation_bank.device
    ).expand_as(permutation_bank)
    if not torch.equal(torch.sort(permutation_bank, dim=1).values, expected):
        raise ValueError("every permutation bank row must contain 0..N-1")
    return permutation_bank


def permutation_bank_sha256(permutation_bank):
    """Hash bank dtype, shape, and exact row-major index bytes."""
    if not torch.is_tensor(permutation_bank) or permutation_bank.ndim != 2:
        raise ValueError("permutation_bank must have shape [B,N]")
    array = np.ascontiguousarray(permutation_bank.detach().cpu().numpy())
    digest = hashlib.sha256()
    for component in (
            str(array.dtype).encode("ascii"),
            ",".join(str(int(size)) for size in array.shape).encode("ascii"),
            array.tobytes(order="C")):
        digest.update(len(component).to_bytes(8, "big"))
        digest.update(component)
    return digest.hexdigest()


def single_source_shuffled_terms(side_semantic_views, target_view,
                                 source_view, sample_permutation):
    """Return the V-1 terms after permuting only one source's sample rows."""
    view_num, sample_num, _ = _validate_side_semantic_views(
        side_semantic_views
    )
    target_view = int(target_view)
    source_view = int(source_view)
    if not (0 <= target_view < view_num and 0 <= source_view < view_num):
        raise IndexError("source or target view is outside the view list")
    if source_view == target_view:
        raise ValueError("source and target views must differ")
    if not torch.is_tensor(sample_permutation):
        raise ValueError("sample_permutation must be a tensor")
    validate_permutation_bank(sample_permutation.reshape(1, -1), sample_num)
    if sample_permutation.device != side_semantic_views[0].device:
        raise ValueError("permutation and semantic views must share device")
    terms = []
    for side_view, value in enumerate(side_semantic_views):
        if side_view == target_view:
            continue
        term = value[sample_permutation] if side_view == source_view else value
        terms.append((side_view, term.detach()))
    return terms


def single_source_shuffled_context(side_semantic_views, target_view,
                                   source_view, sample_permutation):
    """Build one [N,Ds] four-side mean with one source correspondence broken."""
    terms = single_source_shuffled_terms(
        side_semantic_views, target_view, source_view, sample_permutation
    )
    return torch.stack([value for _, value in terms], dim=0).mean(dim=0).detach()


def single_source_shuffled_contexts(side_semantic_views, target_view,
                                    source_view, permutation_bank):
    """Vectorize one relation's B contexts as [B,N,Ds]."""
    view_num, sample_num, _ = _validate_side_semantic_views(
        side_semantic_views
    )
    target_view = int(target_view)
    source_view = int(source_view)
    if not (0 <= target_view < view_num and 0 <= source_view < view_num):
        raise IndexError("source or target view is outside the view list")
    if source_view == target_view:
        raise ValueError("source and target views must differ")
    validate_permutation_bank(permutation_bank, sample_num)
    if permutation_bank.device != side_semantic_views[0].device:
        raise ValueError("permutation bank and semantic views must share device")
    fixed_terms = [
        value
        for side_view, value in enumerate(side_semantic_views)
        if side_view not in (target_view, source_view)
    ]
    fixed_sum = torch.stack(fixed_terms, dim=0).sum(dim=0)
    shuffled_source = side_semantic_views[source_view][permutation_bank]
    contexts = (
        shuffled_source + fixed_sum.unsqueeze(0)
    ) / float(view_num - 1)
    return contexts.detach()


def target_rates_for_contexts(posterior_mu_views, posterior_logvar_views,
                              prior, target_view, context_batches,
                              semantic_dim):
    """Return mean target rate for each context batch as a [B] tensor."""
    target_view = int(target_view)
    semantic_dim = int(semantic_dim)
    if not torch.is_tensor(context_batches) or context_batches.ndim != 3:
        raise ValueError("context_batches must have shape [B,N,Ds]")
    batch_num, sample_num, context_dim = context_batches.shape
    if semantic_dim <= 0 or int(context_dim) != semantic_dim:
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
        flat_mean_q = mean_q.unsqueeze(0).expand(
            batch_num, -1, -1
        ).reshape(batch_num * sample_num, context_dim)
        flat_logvar_q = logvar_q.unsqueeze(0).expand(
            batch_num, -1, -1
        ).reshape(batch_num * sample_num, context_dim)
        raw_rates = analytic_kl_diag_gaussian(
            flat_mean_q, flat_logvar_q, mean_p, logvar_p
        )
        rates = raw_rates.reshape(batch_num, sample_num).mean(dim=1)
        rates = rates / float(semantic_dim)
    return rates.detach()


def relation_permutation_rates(posterior_mu_views, posterior_logvar_views,
                               prior, side_semantic_views, permutation_bank,
                               source_view, target_view, semantic_dim):
    """Evaluate B single-source correspondence permutations for one relation."""
    contexts = single_source_shuffled_contexts(
        side_semantic_views,
        target_view,
        source_view,
        permutation_bank,
    )
    return target_rates_for_contexts(
        posterior_mu_views,
        posterior_logvar_views,
        prior,
        target_view,
        contexts,
        semantic_dim,
    )


def summarize_relation(source_view, target_view, correct_rate, shuffle_rates,
                       permutation_bank_hash):
    """Summarize one preregistered one-sided source-to-target permutation test."""
    correct_rate = float(correct_rate)
    values = np.asarray(shuffle_rates, dtype=np.float64)
    if (
            values.ndim != 1
            or values.size == 0
            or not np.isfinite(values).all()):
        raise ValueError("shuffle_rates must be a finite non-empty [B] array")
    if not math.isfinite(correct_rate) or correct_rate == 0.0:
        raise ValueError("correct_rate must be finite and nonzero")
    percentiles = np.percentile(values, [5, 50, 95])
    less_or_equal_count = int(np.count_nonzero(values <= correct_rate))
    delta = float(np.mean(values) - correct_rate)
    return {
        "source_view": int(source_view),
        "target_view": int(target_view),
        "relation": str(int(source_view)) + "->" + str(int(target_view)),
        "correct_rate_v": correct_rate,
        "shuffle_rate_mean": float(np.mean(values)),
        "shuffle_rate_std": float(np.std(values)),
        "shuffle_rate_p05": float(percentiles[0]),
        "shuffle_rate_p50": float(percentiles[1]),
        "shuffle_rate_p95": float(percentiles[2]),
        "delta": delta,
        "relative_delta": float(delta / correct_rate),
        "shuffle_less_or_equal_correct_count": less_or_equal_count,
        "fraction_shuffle_gt_correct": float(np.mean(values > correct_rate)),
        "permutation_p_value": float(
            (1 + less_or_equal_count) / (values.size + 1)
        ),
        "permutation_bank_hash": str(permutation_bank_hash),
    }


def compute_side_target_attribution(posterior_mu_views,
                                    posterior_logvar_views, prior,
                                    radius_target, semantic_dim,
                                    permutation_bank):
    """Compute all V*(V-1) directed single-source attribution statistics."""
    view_num = len(posterior_mu_views)
    if len(posterior_logvar_views) != view_num:
        raise ValueError("posterior view lists must align")
    if view_num < 2:
        raise ValueError("attribution requires at least two views")
    sample_num = int(posterior_mu_views[0].shape[0])
    validate_permutation_bank(permutation_bank, sample_num)
    bank_hash = permutation_bank_sha256(permutation_bank)
    side_semantic_views = [
        value.detach() / float(radius_target) for value in posterior_mu_views
    ]
    correct_global, correct_per_view, _ = compute_correct_context_rates(
        posterior_mu_views,
        posterior_logvar_views,
        prior,
        radius_target,
        semantic_dim,
    )
    correct_values = correct_per_view.detach().cpu().numpy().astype(
        np.float64, copy=False
    )
    delta_matrix = np.full((view_num, view_num), np.nan, dtype=np.float64)
    pvalue_matrix = np.full((view_num, view_num), np.nan, dtype=np.float64)
    relative_matrix = np.full((view_num, view_num), np.nan, dtype=np.float64)
    fraction_matrix = np.full((view_num, view_num), np.nan, dtype=np.float64)
    records = []
    for source_view, target_view in ordered_source_target_relations(view_num):
        rates = relation_permutation_rates(
            posterior_mu_views,
            posterior_logvar_views,
            prior,
            side_semantic_views,
            permutation_bank,
            source_view,
            target_view,
            semantic_dim,
        )
        record = summarize_relation(
            source_view,
            target_view,
            correct_values[target_view],
            rates.cpu().numpy(),
            bank_hash,
        )
        records.append(record)
        delta_matrix[source_view, target_view] = record["delta"]
        pvalue_matrix[source_view, target_view] = record[
            "permutation_p_value"
        ]
        relative_matrix[source_view, target_view] = record["relative_delta"]
        fraction_matrix[source_view, target_view] = record[
            "fraction_shuffle_gt_correct"
        ]
    return {
        "correct_global_rate": float(correct_global.item()),
        "correct_per_view_rates": correct_values,
        "side_semantic_views": side_semantic_views,
        "relations": records,
        "relation_count": len(records),
        "permutation_bank_hash": bank_hash,
        "delta_matrix": delta_matrix,
        "pvalue_matrix": pvalue_matrix,
        "relative_delta_matrix": relative_matrix,
        "fraction_matrix": fraction_matrix,
    }


def leave_one_source_out_delta(posterior_mu_views, posterior_logvar_views,
                               prior, side_semantic_views,
                               correct_per_view_rates, semantic_dim):
    """Secondary diagnostic: remove one source and average the remaining V-2."""
    view_num, _, _ = _validate_side_semantic_views(side_semantic_views)
    if view_num < 3:
        raise ValueError("leave-one-source-out requires at least three views")
    correct = np.asarray(correct_per_view_rates, dtype=np.float64)
    if correct.shape != (view_num,):
        raise ValueError("correct_per_view_rates must have shape [V]")
    result = np.full((view_num, view_num), np.nan, dtype=np.float64)
    for source_view, target_view in ordered_source_target_relations(view_num):
        retained = [
            value
            for side_view, value in enumerate(side_semantic_views)
            if side_view not in (source_view, target_view)
        ]
        context = torch.stack(retained, dim=0).mean(dim=0).detach()
        rate = target_rates_for_contexts(
            posterior_mu_views,
            posterior_logvar_views,
            prior,
            target_view,
            context.unsqueeze(0),
            semantic_dim,
        )[0]
        result[source_view, target_view] = (
            float(rate.item()) - correct[target_view]
        )
    return result


def aggregate_targets(delta_matrix, pvalue_matrix):
    """Aggregate the four incoming source contributions for each target."""
    delta = np.asarray(delta_matrix, dtype=np.float64)
    pvalue = np.asarray(pvalue_matrix, dtype=np.float64)
    if delta.shape != pvalue.shape or delta.ndim != 2:
        raise ValueError("delta and p-value matrices must have equal [V,V] shape")
    view_num = int(delta.shape[0])
    if delta.shape[1] != view_num:
        raise ValueError("attribution matrices must be square")
    results = []
    for target_view in range(view_num):
        source_ids = [value for value in range(view_num) if value != target_view]
        values = np.asarray(
            [delta[source, target_view] for source in source_ids]
        )
        pvalues = np.asarray(
            [pvalue[source, target_view] for source in source_ids]
        )
        positive = [
            source for source in source_ids if delta[source, target_view] > 0.0
        ]
        negative = [
            source for source in source_ids if delta[source, target_view] < 0.0
        ]
        significant = [
            source for source in source_ids
            if delta[source, target_view] > 0.0
            and pvalue[source, target_view] < 0.05
        ]
        largest_positive = (
            max(positive, key=lambda source: delta[source, target_view])
            if positive else None
        )
        largest_negative = (
            min(negative, key=lambda source: delta[source, target_view])
            if negative else None
        )
        results.append({
            "target_view": target_view,
            "positive_source_count": len(positive),
            "negative_source_count": len(negative),
            "zero_source_count": len(source_ids) - len(positive) - len(negative),
            "significant_positive_source_count": len(significant),
            "significant_positive_sources": significant,
            "largest_positive_source": largest_positive,
            "largest_positive_delta": (
                float(delta[largest_positive, target_view])
                if largest_positive is not None else None
            ),
            "largest_negative_source": largest_negative,
            "largest_negative_delta": (
                float(delta[largest_negative, target_view])
                if largest_negative is not None else None
            ),
            "mean_source_delta": float(np.mean(values)),
            "all_incoming_finite": bool(
                np.isfinite(values).all() and np.isfinite(pvalues).all()
            ),
        })
    return results


def aggregate_sources(delta_matrix, pvalue_matrix):
    """Aggregate each source's contribution to its four other targets."""
    delta = np.asarray(delta_matrix, dtype=np.float64)
    pvalue = np.asarray(pvalue_matrix, dtype=np.float64)
    if delta.shape != pvalue.shape or delta.ndim != 2:
        raise ValueError("delta and p-value matrices must have equal [V,V] shape")
    view_num = int(delta.shape[0])
    if delta.shape[1] != view_num:
        raise ValueError("attribution matrices must be square")
    results = []
    for source_view in range(view_num):
        target_ids = [value for value in range(view_num) if value != source_view]
        values = np.asarray(
            [delta[source_view, target] for target in target_ids]
        )
        positive = [
            target for target in target_ids if delta[source_view, target] > 0.0
        ]
        significant = [
            target for target in target_ids
            if delta[source_view, target] > 0.0
            and pvalue[source_view, target] < 0.05
        ]
        results.append({
            "source_view": source_view,
            "mean_contribution_to_other_targets": float(np.mean(values)),
            "positive_target_count": len(positive),
            "significant_positive_target_count": len(significant),
            "significant_positive_targets": significant,
        })
    return results


def matrix_for_json(matrix):
    """Convert a directed matrix to strict JSON with null on its diagonal."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must have shape [V,V]")
    return [
        [
            None if row == column else float(values[row, column])
            for column in range(values.shape[1])
        ]
        for row in range(values.shape[0])
    ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--condition", choices=("snr2p5_k2",), required=True)
    parser.add_argument("--permutations", type=int, required=True)
    parser.add_argument("--permutation-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _require(
        args.permutations == EXPECTED_PERMUTATIONS,
        "B5-A0.2 preregisters exactly 500 permutations per relation",
    )
    _require(
        args.permutation_seed == EXPECTED_PERMUTATION_SEED,
        "B5-A0.2 permutation seed mismatch",
    )
    input_dir = _resolve(args.input_dir)
    condition_dir, arm_dir, metadata = _load_and_validate_input(
        input_dir, args.condition
    )
    _require(
        int(metadata.get("steps", -1)) == EXPECTED_TRAINING_STEPS,
        "B5-A0.2 requires the 100-step checkpoint",
    )
    _require(
        metadata.get("backbone_hash_before")
        == metadata.get("backbone_hash_after"),
        "input backbone changed during the frozen B5 run",
    )
    models, backbone_hash_before, z_views, z_hash, z_audit = (
        _reconstruct_frozen_z(metadata)
    )
    posterior, prior, posterior_hash_before, prior_hash_before = (
        _load_final_conditional_modules(arm_dir, metadata)
    )
    posterior_checkpoint = arm_dir / "posterior.pth"
    prior_checkpoint = arm_dir / "conditional_prior.pth"
    posterior_file_hash_before = file_sha256(posterior_checkpoint)
    prior_file_hash_before = file_sha256(prior_checkpoint)

    with torch.no_grad():
        posterior_mu_views, posterior_logvar_views = posterior(z_views)
    _require(
        len(posterior_mu_views) == EXPECTED_VIEW_NUM
        and all(
            value.shape == (EXPECTED_SAMPLE_NUM, int(metadata["semantic_dim"]))
            for value in posterior_mu_views
        ),
        "deterministic posterior semantic shape mismatch",
    )
    permutation_bank = generate_sample_permutations(
        EXPECTED_SAMPLE_NUM, args.permutations, args.permutation_seed
    )
    regenerated_bank = generate_sample_permutations(
        EXPECTED_SAMPLE_NUM, args.permutations, args.permutation_seed
    )
    permutation_deterministic_pass = bool(
        torch.equal(permutation_bank, regenerated_bank)
    )
    _require(permutation_deterministic_pass, "permutation bank is not deterministic")

    attribution = compute_side_target_attribution(
        posterior_mu_views,
        posterior_logvar_views,
        prior,
        posterior.radius_target,
        posterior.semantic_dim,
        permutation_bank,
    )
    correct_values = attribution["correct_per_view_rates"]
    correct_errors = np.abs(correct_values - EXPECTED_CORRECT_PER_VIEW)
    correct_max_abs_error = float(np.max(correct_errors))
    correct_repro_pass = bool(correct_max_abs_error < 1e-6)
    _require(correct_repro_pass, "100-step target correct rates not reproduced")

    leave_out_delta = leave_one_source_out_delta(
        posterior_mu_views,
        posterior_logvar_views,
        prior,
        attribution["side_semantic_views"],
        correct_values,
        posterior.semantic_dim,
    )
    delta_matrix = attribution["delta_matrix"]
    pvalue_matrix = attribution["pvalue_matrix"]
    relative_matrix = attribution["relative_delta_matrix"]
    fraction_matrix = attribution["fraction_matrix"]
    target_aggregation = aggregate_targets(delta_matrix, pvalue_matrix)
    source_aggregation = aggregate_sources(delta_matrix, pvalue_matrix)

    off_diagonal = ~np.eye(EXPECTED_VIEW_NUM, dtype=bool)
    matrices = (
        delta_matrix,
        pvalue_matrix,
        relative_matrix,
        fraction_matrix,
        leave_out_delta,
    )
    matrix_finite_pass = bool(
        all(
            np.isfinite(value[off_diagonal]).all()
            and np.isnan(np.diag(value)).all()
            for value in matrices
        )
    )
    relation_count_pass = bool(
        attribution["relation_count"]
        == EXPECTED_VIEW_NUM * (EXPECTED_VIEW_NUM - 1)
    )
    shared_bank_pass = bool(
        len({
            record["permutation_bank_hash"]
            for record in attribution["relations"]
        }) == 1
        and attribution["relations"][0]["permutation_bank_hash"]
        == attribution["permutation_bank_hash"]
    )

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
        posterior_hash_before == posterior_hash_after
        and prior_hash_before == prior_hash_after
        and posterior_file_hash_before == posterior_file_hash_after
        and prior_file_hash_before == prior_file_hash_after
    )
    backbone_unchanged_pass = bool(
        backbone_hash_before == backbone_hash_after
    )
    attribution_complete = bool(
        correct_repro_pass
        and matrix_finite_pass
        and relation_count_pass
        and permutation_deterministic_pass
        and shared_bank_pass
        and no_parameter_gradients_pass
        and parameter_state_unchanged_pass
        and backbone_unchanged_pass
        and z_hash == EXPECTED_CANONICAL_Z_HASH
    )
    _require(attribution_complete, "B5-A0.2 attribution completion checks failed")

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "delta_matrix.npy", delta_matrix)
    np.save(output_dir / "pvalue_matrix.npy", pvalue_matrix)
    np.save(output_dir / "relative_delta_matrix.npy", relative_matrix)
    np.save(output_dir / "fraction_matrix.npy", fraction_matrix)
    np.save(
        output_dir / "permutation_bank.npy",
        permutation_bank.cpu().numpy().astype(np.int64, copy=False),
    )
    np.save(output_dir / "leave_one_source_out_delta.npy", leave_out_delta)

    view2_relations = [
        record for record in attribution["relations"]
        if record["target_view"] == 2
    ]
    result = {
        "stage": "B5-A0.2 Side-to-Target Context Attribution",
        "condition": args.condition,
        "input_dir": str(input_dir.relative_to(REPOSITORY_ROOT)),
        "condition_dir": str(condition_dir.relative_to(REPOSITORY_ROOT)),
        "training_steps": EXPECTED_TRAINING_STEPS,
        "training_performed": False,
        "backward_performed": False,
        "posterior_sampling_used": False,
        "deterministic_posterior_evaluation": True,
        "semantic_state_definition": "posterior_mu / sqrt(semantic_dim)",
        "semantic_shapes": [list(value.shape) for value in posterior_mu_views],
        "labels_loaded": False,
        "labels_used": False,
        "corruption_mask_loaded": False,
        "corruption_mask_used": False,
        "primary_attribution": "single_source_sample_correspondence_permutation",
        "context_definition": "mean of V-1 side semantic terms",
        "matrix_orientation": "row=source_view,column=target_view",
        "matrix_diagonal": "NaN in npy; null in JSON",
        "leave_one_source_out_role": "secondary diagnostic only",
        "permutations_per_relation": args.permutations,
        "permutation_seed": args.permutation_seed,
        "same_permutation_bank_across_all_relations": shared_bank_pass,
        "permutation_bank_shape": list(permutation_bank.shape),
        "permutation_bank_hash": attribution["permutation_bank_hash"],
        "permutation_deterministic_pass": permutation_deterministic_pass,
        "posterior_checkpoint": str(
            posterior_checkpoint.relative_to(REPOSITORY_ROOT)
        ),
        "conditional_prior_checkpoint": str(
            prior_checkpoint.relative_to(REPOSITORY_ROOT)
        ),
        "posterior_checkpoint_sha256_before": posterior_file_hash_before,
        "posterior_checkpoint_sha256_after": posterior_file_hash_after,
        "conditional_prior_checkpoint_sha256_before": prior_file_hash_before,
        "conditional_prior_checkpoint_sha256_after": prior_file_hash_after,
        "posterior_state_hash_before": posterior_hash_before,
        "posterior_state_hash_after": posterior_hash_after,
        "conditional_prior_state_hash_before": prior_hash_before,
        "conditional_prior_state_hash_after": prior_hash_after,
        "parameter_state_unchanged_pass": parameter_state_unchanged_pass,
        "no_parameter_gradients_pass": no_parameter_gradients_pass,
        "canonical_z_hash": z_hash,
        "expected_canonical_z_hash": EXPECTED_CANONICAL_Z_HASH,
        "canonical_z_exact_match": z_hash == EXPECTED_CANONICAL_Z_HASH,
        "input_cross_stage_z_exact_match": bool(
            metadata.get("cross_stage_z_exact_match") is True
        ),
        "z_l2_normalization_audit": z_audit,
        "backbone_hash_before": backbone_hash_before,
        "backbone_hash_after": backbone_hash_after,
        "backbone_unchanged_pass": backbone_unchanged_pass,
        "R_correct": attribution["correct_global_rate"],
        "R_correct_per_view": [float(value) for value in correct_values],
        "expected_R_correct_per_view": [
            float(value) for value in EXPECTED_CORRECT_PER_VIEW
        ],
        "correct_rate_abs_errors": [float(value) for value in correct_errors],
        "correct_rate_reproduction_max_abs_error": correct_max_abs_error,
        "B5_A02_CORRECT_RATE_REPRO_PASS": correct_repro_pass,
        "relation_count": attribution["relation_count"],
        "expected_relation_count": EXPECTED_VIEW_NUM * (EXPECTED_VIEW_NUM - 1),
        "all_20_relations_evaluated_pass": relation_count_pass,
        "all_statistics_finite_pass": matrix_finite_pass,
        "relations": attribution["relations"],
        "delta_matrix": matrix_for_json(delta_matrix),
        "pvalue_matrix": matrix_for_json(pvalue_matrix),
        "relative_delta_matrix": matrix_for_json(relative_matrix),
        "fraction_matrix": matrix_for_json(fraction_matrix),
        "leave_one_source_out_delta": matrix_for_json(leave_out_delta),
        "target_aggregation": target_aggregation,
        "source_aggregation": source_aggregation,
        "target_view2_incoming_relations": view2_relations,
        "B5_A02_ATTRIBUTION_COMPLETE": attribution_complete,
        "B5_FINAL_SCIENTIFIC_PASS_DECLARED": False,
        "B5_A1_STARTED": False,
    }
    _write_json(output_dir / "b5_a02_side_target_attribution.json", result)

    print("correct-rate reproduction max abs error=" + str(
        correct_max_abs_error
    ))
    print("R_correct_per_view=" + str(result["R_correct_per_view"]))
    print("permutation_bank_hash=" + attribution["permutation_bank_hash"])
    print("delta matrix (row=source, column=target):")
    print(np.array2string(delta_matrix, precision=9, suppress_small=False))
    print("p-value matrix (row=source, column=target):")
    print(np.array2string(pvalue_matrix, precision=9, suppress_small=False))
    for record in target_aggregation:
        print(
            "target " + str(record["target_view"])
            + " positive/negative/significant-positive source count="
            + str((
                record["positive_source_count"],
                record["negative_source_count"],
                record["significant_positive_source_count"],
            ))
        )
    for record in source_aggregation:
        print(
            "source " + str(record["source_view"])
            + " mean contribution="
            + str(record["mean_contribution_to_other_targets"])
        )
    print("target view2 incoming relations:")
    for record in view2_relations:
        print(
            record["relation"]
            + " delta/p-value/fraction="
            + str((
                record["delta"],
                record["permutation_p_value"],
                record["fraction_shuffle_gt_correct"],
            ))
        )
    print(
        "B5_A02_CORRECT_RATE_REPRO_PASS="
        + str(correct_repro_pass).lower()
    )
    print(
        "B5_A02_ATTRIBUTION_COMPLETE="
        + str(attribution_complete).lower()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
