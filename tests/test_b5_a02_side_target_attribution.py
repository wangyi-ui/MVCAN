"""Synthetic tests for B5-A0.2 directed side-to-target attribution."""

import inspect
import math

import numpy as np
import torch

import experiments.b5_semantic_rate.evaluate_b5_a02_side_target_attribution as a02
from irv.b5_shared_semantic_rate import ConditionalPriorBank
from irv.b5_shared_semantic_rate import fixed_radius_mean


def _identifiable_views(sample_num=7, view_num=5, semantic_dim=3):
    rows = torch.arange(sample_num, dtype=torch.float32).unsqueeze(1) * 10.0
    columns = torch.arange(semantic_dim, dtype=torch.float32).unsqueeze(0)
    return [rows + columns + 100.0 * view for view in range(view_num)]


def _toy_gaussians(sample_num=7, view_num=5, semantic_dim=3):
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


class _EchoPrior(torch.nn.Module):
    def forward_one(self, side_context, view_idx):
        del view_idx
        return side_context, torch.zeros_like(side_context)


def _nonidentity_cyclic_bank(sample_num):
    canonical = torch.arange(sample_num, dtype=torch.int64)
    return torch.stack([
        torch.roll(canonical, shifts=shift)
        for shift in range(1, sample_num)
    ])


def test_five_views_produce_20_ordered_relations_without_diagonal():
    relations = a02.ordered_source_target_relations(5)
    assert len(relations) == 20
    assert len(set(relations)) == 20
    assert all(source != target for source, target in relations)


def test_single_source_shuffle_only_changes_selected_source_term():
    views = _identifiable_views()
    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    terms = dict(a02.single_source_shuffled_terms(
        views, target_view=0, source_view=2, sample_permutation=permutation
    ))
    assert set(terms) == {1, 2, 3, 4}
    assert torch.equal(terms[2], views[2][permutation])
    assert not torch.equal(terms[2], views[2])
    for unchanged_source in (1, 3, 4):
        assert torch.equal(terms[unchanged_source], views[unchanged_source])


def test_context_remains_exact_mean_of_four_side_terms_and_has_n_by_ds_shape():
    views = _identifiable_views()
    permutation = torch.tensor([6, 5, 4, 3, 2, 1, 0])
    terms = a02.single_source_shuffled_terms(
        views, target_view=1, source_view=4, sample_permutation=permutation
    )
    expected = torch.stack([value for _, value in terms]).mean(dim=0)
    actual = a02.single_source_shuffled_context(
        views, target_view=1, source_view=4, sample_permutation=permutation
    )
    assert actual.shape == (7, 3)
    assert torch.equal(actual, expected)


def test_batched_context_matches_individual_four_term_contexts():
    views = _identifiable_views()
    bank = torch.stack([
        torch.tensor([4, 0, 6, 2, 1, 5, 3]),
        torch.tensor([6, 5, 4, 3, 2, 1, 0]),
    ])
    batched = a02.single_source_shuffled_contexts(
        views, target_view=3, source_view=0, permutation_bank=bank
    )
    assert batched.shape == (2, 7, 3)
    for index, permutation in enumerate(bank):
        expected = a02.single_source_shuffled_context(
            views, target_view=3, source_view=0,
            sample_permutation=permutation,
        )
        assert torch.equal(batched[index], expected)


def test_shuffled_source_row_marginal_is_preserved_exactly():
    views = _identifiable_views()
    permutation = torch.tensor([3, 6, 1, 5, 0, 2, 4])
    terms = dict(a02.single_source_shuffled_terms(
        views, target_view=4, source_view=1, sample_permutation=permutation
    ))
    inverse = torch.argsort(permutation)
    assert torch.equal(terms[1][inverse], views[1])


def test_same_permutation_bank_is_reused_across_all_relations():
    mean, logvar, prior, radius, semantic_dim = _toy_gaussians()
    bank = a02.generate_sample_permutations(7, 4, 20260816)
    result = a02.compute_side_target_attribution(
        mean, logvar, prior, radius, semantic_dim, bank
    )
    hashes = {record["permutation_bank_hash"] for record in result["relations"]}
    assert hashes == {a02.permutation_bank_sha256(bank)}
    assert result["relation_count"] == 20


def test_one_sided_p_value_counts_shuffle_less_or_equal_correct():
    summary = a02.summarize_relation(
        0, 1, 0.5, np.asarray([0.4, 0.6, 0.7]), "bank"
    )
    assert summary["shuffle_less_or_equal_correct_count"] == 1
    assert summary["permutation_p_value"] == 0.5
    assert summary["fraction_shuffle_gt_correct"] == 2.0 / 3.0


def test_positive_synthetic_relation_has_positive_delta():
    sample_num = 6
    target = torch.arange(sample_num, dtype=torch.float32).sub(2.5).unsqueeze(1)
    means = [target.clone() for _ in range(5)]
    logvars = [torch.zeros_like(target) for _ in range(5)]
    bank = _nonidentity_cyclic_bank(sample_num)
    shuffled_rates = a02.relation_permutation_rates(
        means,
        logvars,
        _EchoPrior(),
        means,
        bank,
        source_view=1,
        target_view=0,
        semantic_dim=1,
    )
    correct_context = torch.stack(means[1:]).mean(dim=0).unsqueeze(0)
    correct_rate = a02.target_rates_for_contexts(
        means, logvars, _EchoPrior(), 0, correct_context, 1
    )[0]
    assert float(shuffled_rates.mean() - correct_rate) > 0.0


def test_incompatible_synthetic_relation_can_have_negative_delta():
    sample_num = 6
    target = torch.arange(sample_num, dtype=torch.float32).sub(2.5).unsqueeze(1)
    means = [target.clone() for _ in range(5)]
    side_semantics = [target.clone() for _ in range(5)]
    side_semantics[1] = -target
    logvars = [torch.zeros_like(target) for _ in range(5)]
    bank = _nonidentity_cyclic_bank(sample_num)
    shuffled_rates = a02.relation_permutation_rates(
        means,
        logvars,
        _EchoPrior(),
        side_semantics,
        bank,
        source_view=1,
        target_view=0,
        semantic_dim=1,
    )
    correct_context = torch.stack([
        side_semantics[1], side_semantics[2],
        side_semantics[3], side_semantics[4],
    ]).mean(dim=0).unsqueeze(0)
    correct_rate = a02.target_rates_for_contexts(
        means, logvars, _EchoPrior(), 0, correct_context, 1
    )[0]
    assert float(shuffled_rates.mean() - correct_rate) < 0.0


def test_directed_matrices_are_five_by_five_with_nan_diagonal():
    mean, logvar, prior, radius, semantic_dim = _toy_gaussians()
    bank = a02.generate_sample_permutations(7, 3, 20260816)
    result = a02.compute_side_target_attribution(
        mean, logvar, prior, radius, semantic_dim, bank
    )
    for name in (
            "delta_matrix", "pvalue_matrix",
            "relative_delta_matrix", "fraction_matrix"):
        matrix = result[name]
        assert matrix.shape == (5, 5)
        assert np.isnan(np.diag(matrix)).all()
        assert np.isfinite(matrix[~np.eye(5, dtype=bool)]).all()
        strict_json = a02.matrix_for_json(matrix)
        assert all(strict_json[index][index] is None for index in range(5))


def test_public_attribution_api_has_no_labels_or_corruption_mask():
    functions = (
        a02.ordered_source_target_relations,
        a02.validate_permutation_bank,
        a02.permutation_bank_sha256,
        a02.single_source_shuffled_terms,
        a02.single_source_shuffled_context,
        a02.single_source_shuffled_contexts,
        a02.target_rates_for_contexts,
        a02.relation_permutation_rates,
        a02.summarize_relation,
        a02.compute_side_target_attribution,
        a02.leave_one_source_out_delta,
        a02.aggregate_targets,
        a02.aggregate_sources,
    )
    for function in functions:
        names = [name.lower() for name in inspect.signature(function).parameters]
        assert all("label" not in name for name in names)
        assert all("mask" not in name for name in names)


def test_attribution_has_no_backward_and_creates_no_parameter_gradients():
    mean, logvar, prior, radius, semantic_dim = _toy_gaussians()
    bank = a02.generate_sample_permutations(7, 2, 20260816)
    before = [value.detach().clone() for value in prior.parameters()]
    a02.compute_side_target_attribution(
        mean, logvar, prior, radius, semantic_dim, bank
    )
    after = list(prior.parameters())
    assert all(parameter.grad is None for parameter in after)
    assert all(torch.equal(a, b.detach()) for a, b in zip(before, after))
    assert ".backward(" not in inspect.getsource(a02)


def test_ordered_relations_explicitly_exclude_diagonal():
    relations = a02.ordered_source_target_relations(5)
    assert all(source != target for source, target in relations)


def test_other_three_sources_are_explicitly_unchanged():
    views = _identifiable_views()
    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    terms = dict(a02.single_source_shuffled_terms(
        views, target_view=0, source_view=2, sample_permutation=permutation
    ))
    for unchanged_source in (1, 3, 4):
        assert torch.equal(terms[unchanged_source], views[unchanged_source])


def test_single_relation_context_shape_is_explicitly_n_by_ds():
    views = _identifiable_views()
    permutation = torch.tensor([6, 5, 4, 3, 2, 1, 0])
    context = a02.single_source_shuffled_context(
        views, target_view=1, source_view=4, sample_permutation=permutation
    )
    assert context.shape == (7, 3)


def test_public_context_api_explicitly_has_no_corruption_mask():
    for function in (
            a02.single_source_shuffled_terms,
            a02.single_source_shuffled_context,
            a02.single_source_shuffled_contexts):
        names = [name.lower() for name in inspect.signature(function).parameters]
        assert all("mask" not in name for name in names)


def _a02_cli(condition):
    return [
        "--input-dir", "input",
        "--condition", condition,
        "--permutations", "500",
        "--permutation-seed", "20260816",
        "--output-dir", "output",
    ]


def test_a02_cli_accepts_clean():
    args = a02.parse_args(_a02_cli("clean"))
    assert args.condition == "clean"


def test_a02_cli_still_accepts_noisy():
    args = a02.parse_args(_a02_cli("snr2p5_k2"))
    assert args.condition == "snr2p5_k2"


def test_clean_reproduction_never_uses_noisy_historical_rates():
    clean_per_view = np.asarray([0.7, 0.9, 1.1, 1.3, 1.5])
    clean_global = float(np.mean(clean_per_view))
    audit = a02.condition_correct_rate_reproduction_audit(
        "clean", clean_global, clean_per_view, clean_global
    )
    assert not audit["noisy_historical_per_view_reference_applied"]
    assert audit["expected_noisy_R_correct_per_view"] is None
    assert audit["noisy_historical_per_view_abs_errors"] is None
    assert audit["noisy_historical_per_view_max_abs_error"] is None
    assert audit["B5_A02_CORRECT_RATE_REPRO_PASS"]


def test_noisy_historical_per_view_regression_remains_exact():
    noisy_per_view = a02.EXPECTED_NOISY_CORRECT_PER_VIEW.copy()
    noisy_global = float(np.mean(noisy_per_view))
    audit = a02.condition_correct_rate_reproduction_audit(
        "snr2p5_k2", noisy_global, noisy_per_view, noisy_global
    )
    assert audit["noisy_historical_per_view_reference_applied"]
    assert audit["noisy_historical_per_view_max_abs_error"] == 0.0
    assert audit["B5_A02_CORRECT_RATE_REPRO_PASS"]


def test_noisy_historical_per_view_regression_cannot_be_weakened():
    changed = a02.EXPECTED_NOISY_CORRECT_PER_VIEW.copy()
    changed[2] += 1e-3
    changed_global = float(np.mean(changed))
    audit = a02.condition_correct_rate_reproduction_audit(
        "snr2p5_k2", changed_global, changed, changed_global
    )
    assert not audit["noisy_historical_per_view_reproduction_pass"]
    assert not audit["B5_A02_CORRECT_RATE_REPRO_PASS"]


def test_same_n_and_seed_produce_expected_cross_condition_bank_hash():
    clean_bank = a02.generate_sample_permutations(210, 500, 20260816)
    noisy_bank = a02.generate_sample_permutations(210, 500, 20260816)
    assert torch.equal(clean_bank, noisy_bank)
    assert a02.permutation_bank_sha256(clean_bank) == (
        a02.EXPECTED_SHARED_PERMUTATION_BANK_HASH
    )


def test_clean_noisy_comparison_reports_effect_and_sign_flips():
    clean = np.asarray([
        [np.nan, 1.0, -1.0],
        [-1.0, np.nan, 1.0],
        [1.0, -1.0, np.nan],
    ])
    noisy = np.asarray([
        [np.nan, 2.0, -2.0],
        [1.0, np.nan, -1.0],
        [1.0, -1.0, np.nan],
    ])
    result = a02.compare_clean_noisy_delta_matrices(clean, noisy)
    assert np.allclose(
        result["noise_effect_matrix"][~np.eye(3, dtype=bool)],
        (noisy - clean)[~np.eye(3, dtype=bool)],
    )
    assert result["positive_in_both_count"] == 2
    assert result["negative_in_both_count"] == 2
    assert result["clean_positive_noisy_negative_count"] == 1
    assert result["clean_negative_noisy_positive_count"] == 1
    assert sum(record["sign_flip"] for record in result["relations"]) == 2


def test_new_generalization_api_has_no_labels_or_corruption_mask():
    for function in (
            a02.condition_correct_rate_reproduction_audit,
            a02.directed_matrix_from_json,
            a02.compare_clean_noisy_delta_matrices):
        names = [name.lower() for name in inspect.signature(function).parameters]
        assert all("label" not in name and "mask" not in name for name in names)
