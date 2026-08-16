"""Synthetic tests for the B5-A0 conditional-context specificity audit."""

import inspect
import math

import numpy as np
import torch

import experiments.b5_semantic_rate.evaluate_b5_a0_context_specificity as specificity

from experiments.b5_semantic_rate.evaluate_b5_a0_context_specificity import (
    compute_correct_context_rates,
)
from experiments.b5_semantic_rate.evaluate_b5_a0_context_specificity import (
    compute_permuted_context_rates,
)
from experiments.b5_semantic_rate.evaluate_b5_a0_context_specificity import (
    generate_sample_permutations,
)
from experiments.b5_semantic_rate.evaluate_b5_a0_context_specificity import (
    permutation_test_summary,
)
from experiments.b5_semantic_rate.evaluate_b5_a0_context_specificity import (
    permute_context_views,
)
from irv.b5_shared_semantic_rate import ConditionalPriorBank
from irv.b5_shared_semantic_rate import build_conditional_context
from irv.b5_shared_semantic_rate import fixed_radius_mean


def _toy_gaussians(sample_num=8, view_num=5, semantic_dim=3):
    generator = torch.Generator().manual_seed(37)
    radius = math.sqrt(semantic_dim)
    posterior_mu = [
        fixed_radius_mean(
            torch.randn((sample_num, semantic_dim), generator=generator),
            radius,
        )
        for _ in range(view_num)
    ]
    posterior_logvar = [
        torch.zeros((sample_num, semantic_dim)) for _ in range(view_num)
    ]
    prior = ConditionalPriorBank(view_num, semantic_dim, prior_seed=3020)
    prior.eval().requires_grad_(False)
    return posterior_mu, posterior_logvar, prior, radius, semantic_dim


def _identifiable_contexts(sample_num=7, view_num=5, semantic_dim=3):
    row = torch.arange(sample_num, dtype=torch.float32).unsqueeze(1) * 10.0
    feature = torch.arange(semantic_dim, dtype=torch.float32).unsqueeze(0)
    return [row + feature + 100.0 * view for view in range(view_num)]


def test_same_context_correct_rate_is_deterministic():
    mean, logvar, prior, radius, semantic_dim = _toy_gaussians()
    first = compute_correct_context_rates(
        mean, logvar, prior, radius, semantic_dim
    )
    second = compute_correct_context_rates(
        mean, logvar, prior, radius, semantic_dim
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert all(torch.equal(a, b) for a, b in zip(first[2], second[2]))


def test_permutations_are_deterministic_with_fixed_seed():
    first = generate_sample_permutations(21, 12, 20260816)
    second = generate_sample_permutations(21, 12, 20260816)
    different = generate_sample_permutations(21, 12, 20260817)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_same_permutation_is_used_across_all_target_views():
    contexts = _identifiable_contexts()
    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    shuffled = permute_context_views(contexts, permutation)
    for target_view in range(5):
        assert torch.equal(shuffled[target_view], contexts[target_view][permutation])


def test_permutation_changes_only_sample_dimension():
    contexts = _identifiable_contexts()
    permutation = torch.tensor([6, 5, 4, 3, 2, 1, 0])
    shuffled = permute_context_views(contexts, permutation)
    for original, changed in zip(contexts, shuffled):
        assert changed.shape == original.shape
        assert torch.equal(changed, original[permutation])
        assert torch.equal(changed[:, 1] - changed[:, 0], torch.ones(7))
        assert torch.equal(changed[:, 2] - changed[:, 1], torch.ones(7))


def test_context_marginal_rows_are_preserved_exactly():
    contexts = _identifiable_contexts()
    permutation = torch.tensor([3, 6, 1, 5, 0, 2, 4])
    inverse = torch.argsort(permutation)
    shuffled = permute_context_views(contexts, permutation)
    for original, changed in zip(contexts, shuffled):
        assert torch.equal(changed[inverse], original)


def test_public_permutation_api_has_no_labels_or_mask():
    functions = (
        generate_sample_permutations,
        permute_context_views,
        compute_correct_context_rates,
        compute_permuted_context_rates,
        permutation_test_summary,
    )
    for function in functions:
        names = [name.lower() for name in inspect.signature(function).parameters]
        assert all("label" not in name for name in names)
        assert all("mask" not in name for name in names)


def test_one_sided_p_value_counts_shuffle_less_or_equal_correct():
    summary = permutation_test_summary(0.5, np.asarray([0.4, 0.6, 0.7]))
    assert summary["shuffle_less_or_equal_correct_count"] == 1
    assert summary["permutation_p_value"] == 0.5
    assert summary["fraction_shuffle_gt_correct"] == 2.0 / 3.0


def test_artificially_lower_correct_rate_passes_specificity_gate():
    shuffled = np.linspace(1.0, 2.0, 100)
    summary = permutation_test_summary(0.5, shuffled)
    assert summary["correct_rate"] < summary["shuffle_rate_p05"]
    assert summary["permutation_p_value"] == 1.0 / 101.0
    assert summary["B5_A0_CONDITIONAL_CONTEXT_SPECIFICITY_PASS"]


def test_correct_equal_to_null_median_fails_specificity_gate():
    shuffled = np.ones(100)
    summary = permutation_test_summary(1.0, shuffled)
    assert summary["correct_rate"] == summary["shuffle_rate_p50"]
    assert not summary["B5_A0_CONDITIONAL_CONTEXT_SPECIFICITY_PASS"]


def test_500_permutation_output_shapes_are_exact():
    mean, logvar, prior, radius, semantic_dim = _toy_gaussians(sample_num=6)
    contexts = build_conditional_context(mean, radius)
    permutations = generate_sample_permutations(6, 500, 20260816)
    shuffle_rates, per_view_shuffle_rates = compute_permuted_context_rates(
        mean,
        logvar,
        prior,
        contexts,
        permutations,
        semantic_dim,
    )
    assert shuffle_rates.shape == (500,)
    assert per_view_shuffle_rates.shape == (500, 5)
    assert torch.isfinite(shuffle_rates).all()
    assert torch.isfinite(per_view_shuffle_rates).all()


def _specificity_cli(condition):
    return [
        "--input-dir", "input",
        "--condition", condition,
        "--permutations", "500",
        "--permutation-seed", "20260816",
        "--output-dir", "output",
    ]


def test_specificity_cli_accepts_clean():
    args = specificity.parse_args(_specificity_cli("clean"))
    assert args.condition == "clean"


def test_specificity_cli_still_accepts_noisy():
    args = specificity.parse_args(_specificity_cli("snr2p5_k2"))
    assert args.condition == "snr2p5_k2"


def test_clean_canonical_hash_comes_from_current_metadata():
    clean_hash = "a" * 64
    metadata = {
        "expected_z_hash_b4_compatible": clean_hash,
        "z_hash_b4_compatible": clean_hash,
    }
    assert specificity.expected_canonical_z_hash(metadata, "clean") == clean_hash
    assert clean_hash != specificity.EXPECTED_NOISY_CANONICAL_Z_HASH


def test_clean_correct_rate_reproduces_stored_rate_and_per_view_mean():
    per_view = np.asarray([0.8, 1.0, 1.2, 1.4, 1.6])
    correct = float(np.mean(per_view))
    audit = specificity.correct_rate_reproduction_audit(
        correct, per_view, correct
    )
    assert audit["stored_rate_abs_error"] == 0.0
    assert audit["per_view_mean_abs_error"] == 0.0
    assert audit["per_view_rates_finite_pass"]
    assert audit["correct_rate_reproduction_pass"]


def test_clean_and_noisy_use_identical_permutation_construction():
    clean_bank = specificity.generate_sample_permutations(210, 500, 20260816)
    noisy_bank = specificity.generate_sample_permutations(210, 500, 20260816)
    assert torch.equal(clean_bank, noisy_bank)


def test_clean_feature_condition_preserves_feature_rows_exactly():
    views = [np.full((4, 3), view, dtype=np.float32) for view in range(5)]
    clean = specificity.condition_evaluation_views(views, "clean")
    assert all(np.array_equal(a, b) for a, b in zip(clean, views))
