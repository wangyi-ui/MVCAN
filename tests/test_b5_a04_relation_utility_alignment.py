"""Synthetic and provenance tests for the B5-A0.4 alignment audit."""

import inspect
import json
import math
from pathlib import Path

import numpy as np
import torch

import experiments.b5_semantic_rate.evaluate_b5_a02_side_target_attribution as a02
import experiments.b5_semantic_rate.evaluate_b5_a04_relation_utility_alignment as a04
from irv.b3_predictability_diagnostics import make_fold_assignment
from irv.b4_information_utility import compute_information_utility, tensor_sha256


ROOT = Path(__file__).resolve().parents[1]


class _EchoPrior(torch.nn.Module):
    def forward_one(self, side_context, view_idx):
        del view_idx
        return side_context, torch.zeros_like(side_context)

    def forward(self, side_context_views):
        outputs = [
            self.forward_one(value, view_idx)
            for view_idx, value in enumerate(side_context_views)
        ]
        return [value[0] for value in outputs], [value[1] for value in outputs]


class _AffinePrior(torch.nn.Module):
    def __init__(self, semantic_dim):
        super().__init__()
        self.linear = torch.nn.Linear(semantic_dim, semantic_dim)

    def forward_one(self, side_context, view_idx):
        del view_idx
        return self.linear(side_context), torch.zeros_like(side_context)

    def forward(self, side_context_views):
        outputs = [
            self.forward_one(value, view_idx)
            for view_idx, value in enumerate(side_context_views)
        ]
        return [value[0] for value in outputs], [value[1] for value in outputs]


def _toy_gaussians(sample_num=7, view_num=5, semantic_dim=3):
    generator = torch.Generator().manual_seed(404)
    radius = math.sqrt(semantic_dim)
    means = [
        torch.nn.functional.normalize(
            torch.randn((sample_num, semantic_dim), generator=generator),
            dim=1,
        )
        * radius
        for _ in range(view_num)
    ]
    logvars = [torch.zeros_like(value) for value in means]
    bank = torch.stack([
        torch.roll(torch.arange(sample_num), shifts=shift)
        for shift in range(1, sample_num)
    ]).to(torch.int64)
    return means, logvars, radius, semantic_dim, bank


def _random_relation_matrices(sample_num=18, seed=71):
    generator = np.random.default_rng(seed)
    benefit = np.full((sample_num, 5, 5), np.nan, dtype=np.float64)
    pair = np.full_like(benefit, np.nan)
    source_utility = generator.random((sample_num, 5))
    for source_view, target_view in a04.ordered_source_target_relations(5):
        benefit[:, source_view, target_view] = (
            generator.normal(size=sample_num)
            + 0.1 * source_view
            - 0.05 * target_view
        )
        pair[:, source_view, target_view] = generator.random(sample_num)
    source = a04.source_relation_evidence(source_utility)
    return benefit, pair, source, source_utility


def test_relation_benefit_has_n_by_five_by_five_shape_and_nan_diagonal():
    means, logvars, radius, semantic_dim, bank = _toy_gaussians()
    benefit, correct = a04.compute_relation_benefit(
        means,
        logvars,
        _EchoPrior(),
        radius,
        semantic_dim,
        bank,
    )
    assert benefit.shape == (7, 5, 5)
    assert benefit.dtype == np.float64
    assert correct.shape == (7, 5)
    assert np.isnan(np.diagonal(benefit, axis1=1, axis2=2)).all()
    assert np.isfinite(benefit[:, ~np.eye(5, dtype=bool)]).all()


def test_five_views_have_exactly_20_unique_ordered_relations():
    relations = a04.ordered_source_target_relations(5)
    assert len(relations) == 20
    assert len(set(relations)) == 20
    assert all(source != target for source, target in relations)


def test_sample_g_mean_reproduces_independent_a02_aggregate_delta():
    means, logvars, radius, semantic_dim, bank = _toy_gaussians()
    prior = _EchoPrior()
    benefit, _ = a04.compute_relation_benefit(
        means, logvars, prior, radius, semantic_dim, bank
    )
    aggregate = a02.compute_side_target_attribution(
        means, logvars, prior, radius, semantic_dim, bank
    )["delta_matrix"]
    error, reproduced = a04.relation_benefit_reproduction_error(
        benefit, aggregate
    )
    off_diagonal = ~np.eye(5, dtype=bool)
    assert error < 1e-6
    assert np.allclose(
        reproduced[off_diagonal], aggregate[off_diagonal], atol=1e-6, rtol=0.0
    )


def test_exact_existing_a02_permutation_bank_is_loaded_and_hash_matched():
    directory = ROOT / (
        "outputs/b5_semantic_rate/"
        "a02_side_target_attribution_step100_seed20"
    )
    bank, bank_hash, path = a04.load_exact_permutation_bank(directory)
    saved = np.load(path, allow_pickle=False)
    assert bank.shape == (500, 210)
    assert torch.equal(bank, torch.from_numpy(saved))
    assert bank_hash == a04.EXPECTED_PERMUTATION_BANK_HASH


def test_permutation_loader_does_not_contain_random_generation():
    source = inspect.getsource(a04.load_exact_permutation_bank)
    assert "np.load" in source
    assert "randperm" not in source
    assert "permutation(" not in source


def test_pairwise_t_shape_nan_diagonal_finite_range_and_oof_coverage():
    generator = np.random.default_rng(11)
    shared = generator.normal(size=(20, 4))
    views = [shared + 0.03 * generator.normal(size=shared.shape) for _ in range(5)]
    folds = make_fold_assignment(20, n_splits=5, random_state=20)
    pairwise, result = a04.pairwise_predictability_from_oof(views, folds)
    assert pairwise.shape == (20, 5, 5)
    assert np.isnan(np.diagonal(pairwise, axis1=1, axis2=2)).all()
    values = pairwise[:, ~np.eye(5, dtype=bool)]
    assert np.isfinite(values).all()
    assert values.min() >= -1.0 - 1e-10
    assert values.max() <= 1.0 + 1e-10
    assert result["audit"]["ordered_view_pair_count"] == 20
    assert result["audit"]["train_test_disjoint_pass"]
    assert result["audit"]["oof_coverage_pass"]
    assert result["audit"]["oof_coverage_count"] == 20


def test_oof_train_test_are_disjoint_and_every_sample_covered_once():
    folds = make_fold_assignment(210, n_splits=5, random_state=20)
    audit = a04.fold_integrity_audit(folds)
    assert audit["train_test_disjoint_pass"]
    assert audit["all_samples_covered_exactly_once_pass"]
    assert np.array_equal(audit["coverage"], np.ones(210, dtype=np.int64))


def test_ridge_definition_is_exactly_alpha_one_with_intercept():
    import irv.b3_predictability_diagnostics as b3

    source = inspect.getsource(b3._oof_ridge_predictability_core)
    assert "Ridge(alpha=float(alpha), fit_intercept=True)" in source
    assert a04.RIDGE_ALPHA == 1.0
    assert a04.RIDGE_FIT_INTERCEPT is True


def test_canonical_b3_fold_hash_and_saved_assignment_match():
    folds, fold_hash, reference = a04.load_canonical_b3_folds()
    assert folds.shape == (210,)
    assert reference.is_file()
    assert fold_hash == a04.EXPECTED_B3_FOLD_HASH


def test_relation_rank_is_in_unit_interval_and_handles_average_ties():
    values = np.full((4, 5, 5), np.nan, dtype=np.float64)
    for source_view, target_view in a04.ordered_source_target_relations(5):
        values[:, source_view, target_view] = [1.0, 1.0, 3.0, 4.0]
    ranks = a04.rank_percentile_relations(values)
    expected = np.asarray([0.5 / 3.0, 0.5 / 3.0, 2.0 / 3.0, 1.0])
    for source_view, target_view in a04.ordered_source_target_relations(5):
        assert np.allclose(ranks[:, source_view, target_view], expected)
    off_diagonal = ~np.eye(5, dtype=bool)
    assert ranks[:, off_diagonal].min() >= 0.0
    assert ranks[:, off_diagonal].max() <= 1.0


def test_source_u_shape_range_and_exact_b4_hash_reproduction():
    score_path = ROOT / (
        "outputs/b3_semantic/a23_reproduction/"
        "a23_scores_snr2p5_seed20.npz"
    )
    with np.load(score_path, allow_pickle=False) as archive:
        consensus_t = np.asarray(archive["T"], dtype=np.float64)
    utility, audit = a04.reconstruct_source_utility(
        {"oof_consensus_cosine": consensus_t},
        a04.DEFAULT_B4_UTILITY_REFERENCE,
    )
    assert utility.shape == (210, 5)
    assert utility.min() >= 0.0
    assert utility.max() <= 1.0
    assert audit["exact_reproduction_pass"]
    assert tensor_sha256(utility) == audit["expected_utility_sha256"]
    assert np.array_equal(utility, compute_information_utility(consensus_t))


def test_source_u_is_replicated_only_from_source_over_four_targets():
    utility = np.arange(30, dtype=np.float64).reshape(6, 5) / 29.0
    evidence = a04.source_relation_evidence(utility)
    assert evidence.shape == (6, 5, 5)
    assert np.isnan(np.diagonal(evidence, axis1=1, axis2=2)).all()
    for source_view, target_view in a04.ordered_source_target_relations(5):
        assert np.array_equal(
            evidence[:, source_view, target_view], utility[:, source_view]
        )


def test_relation_fixed_effect_has_zero_mean_for_every_relation():
    benefit, _, _, _ = _random_relation_matrices()
    adjusted = a04.relation_fixed_effect(benefit)
    for source_view, target_view in a04.ordered_source_target_relations(5):
        assert abs(adjusted[:, source_view, target_view].mean()) < 1e-12


def test_sample_bootstrap_draws_sample_ids_not_4200_relation_rows():
    sample_ids = a04.sample_bootstrap_indices(210, 30, 20260816)
    assert sample_ids.shape == (30, 210)
    assert sample_ids.dtype == np.int64
    assert sample_ids.min() >= 0
    assert sample_ids.max() < 210


def test_selected_bootstrap_sample_keeps_all_20_relations():
    benefit, pair, source, _ = _random_relation_matrices(sample_num=12)
    sample_ids = np.asarray([[3, 3, 8, 1, 10, 0, 6, 2, 7, 5, 4, 9]])
    output = a04.bootstrap_alignment_metrics(
        benefit, pair, source, sample_ids
    )
    selected = benefit[sample_ids[0]]
    assert selected[:, ~np.eye(5, dtype=bool)].shape == (12, 20)
    assert all(value.shape == (1,) for value in output.values())


def test_bootstrap_is_deterministic_and_paired_deltas_share_resamples():
    benefit, pair, source, _ = _random_relation_matrices()
    first_ids = a04.sample_bootstrap_indices(18, 12, 20260816)
    second_ids = a04.sample_bootstrap_indices(18, 12, 20260816)
    assert np.array_equal(first_ids, second_ids)
    first = a04.bootstrap_alignment_metrics(
        benefit, pair, source, first_ids
    )
    second = a04.bootstrap_alignment_metrics(
        benefit, pair, source, second_ids
    )
    for name in first:
        assert np.array_equal(first[name], second[name])
    assert np.allclose(
        first["delta_auc_pair_minus_source"],
        first["auc_fe_pair"] - first["auc_fe_source"],
    )
    assert np.allclose(
        first["delta_spearman_pair_minus_source"],
        first["spearman_fe_pair"] - first["spearman_fe_source"],
    )


def test_auc_single_class_relation_returns_nan_instead_of_error():
    assert math.isnan(a04.safe_auc(np.ones(9, dtype=np.int64), np.arange(9)))
    assert math.isnan(a04.safe_auc(np.zeros(9, dtype=np.int64), np.arange(9)))


def test_evidence_shuffle_is_strictly_within_each_relation():
    _, pair, _, _ = _random_relation_matrices(sample_num=25)
    shuffled = a04.within_relation_shuffle_evidence(
        pair, np.random.default_rng(20260816)
    )
    changed_count = 0
    for source_view, target_view in a04.ordered_source_target_relations(5):
        before = pair[:, source_view, target_view]
        after = shuffled[:, source_view, target_view]
        assert np.array_equal(np.sort(before), np.sort(after))
        changed_count += int(not np.array_equal(before, after))
    assert changed_count == 20
    assert np.isnan(np.diagonal(shuffled, axis1=1, axis2=2)).all()


def test_evidence_null_is_deterministic():
    benefit, pair, _, _ = _random_relation_matrices(sample_num=15)
    benefit_fe = a04.relation_fixed_effect(benefit)
    first = a04.evidence_permutation_null(benefit_fe, pair, 7, seed=99)
    second = a04.evidence_permutation_null(benefit_fe, pair, 7, seed=99)
    assert np.array_equal(first["auc_fe_pair"], second["auc_fe_pair"])
    assert np.array_equal(
        first["spearman_fe_pair"], second["spearman_fe_pair"]
    )


def test_structural_baseline_is_constant_within_relation():
    structural = np.arange(25, dtype=np.float64).reshape(5, 5)
    np.fill_diagonal(structural, np.nan)
    evidence = a04.structural_relation_evidence(structural, 13)
    for source_view, target_view in a04.ordered_source_target_relations(5):
        assert np.unique(evidence[:, source_view, target_view]).size == 1
        assert evidence[0, source_view, target_view] == structural[
            source_view, target_view
        ]


def test_public_diagnostic_apis_have_no_label_or_corruption_arguments():
    functions = (
        a04.load_exact_permutation_bank,
        a04.target_sample_rates_for_contexts,
        a04.compute_relation_benefit,
        a04.pairwise_predictability_from_oof,
        a04.rank_percentile_relations,
        a04.reconstruct_source_utility,
        a04.source_relation_evidence,
        a04.structural_relation_evidence,
        a04.relation_fixed_effect,
        a04.alignment_metrics,
        a04.bootstrap_alignment_metrics,
        a04.evidence_permutation_null,
    )
    for function in functions:
        parameters = [name.lower() for name in inspect.signature(function).parameters]
        assert all("label" not in name for name in parameters)
        assert all("corruption" not in name for name in parameters)
        assert all("mask" not in name for name in parameters)


def test_relation_benefit_has_no_backward_gradients_or_parameter_changes():
    means, logvars, radius, semantic_dim, bank = _toy_gaussians()
    prior = _AffinePrior(semantic_dim).eval().requires_grad_(False)
    before = [parameter.detach().clone() for parameter in prior.parameters()]
    a04.compute_relation_benefit(
        means, logvars, prior, radius, semantic_dim, bank
    )
    after = list(prior.parameters())
    assert all(parameter.grad is None for parameter in after)
    assert all(torch.equal(old, new.detach()) for old, new in zip(before, after))
    assert ".backward(" not in inspect.getsource(a04)


def test_saved_artifact_matrix_shapes_are_exact(tmp_path):
    benefit, pair, _, source_utility = _random_relation_matrices(sample_num=9)
    benefit_fe = a04.relation_fixed_effect(benefit)
    pair_rank = a04.rank_percentile_relations(pair)
    rows = a04.per_relation_diagnostics(
        benefit, pair, pair_rank, source_utility
    )
    bootstrap = {
        "auc_fe_pair": np.asarray([0.5]),
        "spearman_fe_pair": np.asarray([0.0]),
        "auc_fe_source": np.asarray([0.5]),
        "spearman_fe_source": np.asarray([0.0]),
        "delta_auc_pair_minus_source": np.asarray([0.0]),
        "delta_spearman_pair_minus_source": np.asarray([0.0]),
    }
    null = {
        "auc_fe_pair": np.asarray([0.5]),
        "spearman_fe_pair": np.asarray([0.0]),
    }
    output = a04.save_artifacts(
        tmp_path,
        benefit,
        benefit_fe,
        pair,
        pair_rank,
        source_utility,
        rows,
        bootstrap,
        np.arange(9, dtype=np.int64)[None, :],
        null,
    )
    assert np.load(output / "relation_benefit.npy").shape == (9, 5, 5)
    assert np.load(output / "relation_benefit_positive_mask.npy").shape == (
        9,
        5,
        5,
    )
    assert np.load(output / "relation_benefit_fe.npy").shape == (9, 5, 5)
    assert np.load(output / "pairwise_predictability.npy").shape == (9, 5, 5)
    assert np.load(output / "pairwise_predictability_rank.npy").shape == (
        9,
        5,
        5,
    )
    assert np.load(output / "source_utility.npy").shape == (9, 5)
    assert (output / "per_relation_metrics.csv").is_file()
    assert (output / "bootstrap_metrics.npz").is_file()
    assert (output / "evidence_null_metrics.npz").is_file()


def test_cli_matches_preregistered_invocation_surface():
    args = a04.parse_args([
        "--input-dir",
        "input",
        "--condition",
        "snr2p5_k2",
        "--model-seed",
        "20",
        "--a02-dir",
        "a02",
        "--clean-a02-dir",
        "clean-a02",
        "--bootstrap-repeats",
        "2000",
        "--bootstrap-seed",
        "20260816",
        "--evidence-null-repeats",
        "500",
        "--output-dir",
        "output",
    ])
    assert args.bootstrap_repeats == 2000
    assert args.bootstrap_seed == 20260816
    assert args.evidence_null_repeats == 500
    assert args.condition == "snr2p5_k2"
    assert args.model_seed == 20


def test_a04_cli_accepts_all_preregistered_model_seeds():
    for seed in (20, 30, 50):
        args = a04.parse_args([
            "--input-dir", "input",
            "--condition", "snr2p5_k2",
            "--model-seed", str(seed),
            "--a02-dir", "a02",
            "--clean-a02-dir", "clean-a02",
            "--output-dir", "output",
        ])
        assert args.model_seed == seed


def test_requested_seed_must_match_checkpoint_metadata():
    assert a04.validate_requested_model_seed({"model_seed": 30}, 30) == 30
    try:
        a04.validate_requested_model_seed({"model_seed": 30}, 50)
    except RuntimeError as error:
        assert "does not match checkpoint metadata" in str(error)
    else:
        raise AssertionError("mismatched checkpoint seed was accepted")


def test_seed_specific_z_hash_comes_from_matching_metadata():
    seed30_hash = "31bbf7426b0982ef301da938669a1ac482284fdf98e830663ca87b1ceaebbf27"
    metadata = {
        "model_seed": 30,
        "expected_z_hash_b4_compatible": seed30_hash,
        "z_hash_b4_compatible": seed30_hash,
    }
    assert a04.expected_canonical_z_hash(metadata, "snr2p5_k2") == seed30_hash
    assert seed30_hash != a04.EXPECTED_CANONICAL_Z_HASH


def test_seed_specific_fold_provenance_matches_historical_b3():
    expected = {
        20: a04.EXPECTED_B3_FOLD_HASH,
        30: "5780b9955895f03d53e22cfcba56b2608e87e986d647692faf258e2b7f97268f",
        50: "57e9426332aa3cec4b42eb14576b7e1dc33cf62185d42dbbb09e6e0f1f8e57ce",
    }
    for seed in (20, 30, 50):
        folds, fold_hash, reference = a04.load_canonical_b3_folds(
            model_seed=seed
        )
        assert folds.shape == (210,)
        assert reference == a04.default_b3_score_reference(seed)
        assert fold_hash == expected[seed]


def test_seed30_source_u_reconstruction_is_deterministic_and_frozen():
    reference = a04.default_b3_score_reference(30)
    with np.load(reference, allow_pickle=False) as archive:
        consensus_t = np.asarray(archive["T"], dtype=np.float64)
        folds = np.asarray(archive["fold_assignment"], dtype=np.int64)
    fold_hash = a04.fold_assignment_sha256(folds)
    first, first_audit = a04.reconstruct_source_utility(
        {"oof_consensus_cosine": consensus_t},
        b3_score_reference=reference,
        fold_hash=fold_hash,
    )
    second, second_audit = a04.reconstruct_source_utility(
        {"oof_consensus_cosine": consensus_t},
        b3_score_reference=reference,
        fold_hash=fold_hash,
    )
    assert np.array_equal(first, second)
    assert first_audit["utility_sha256"] == second_audit["utility_sha256"]
    assert first_audit["formula"] == "per-view (average_rank(T)-1)/(N-1)"
    assert first_audit["rank_method"] == "average"
    assert first.min() >= 0.0 and first.max() <= 1.0


def test_seed20_saved_alignment_regression_is_unchanged():
    path = ROOT / (
        "outputs/b5_semantic_rate/"
        "a04_relation_utility_alignment_step100_seed20/"
        "b5_a04_relation_utility_alignment.json"
    )
    result = json.loads(path.read_text())
    assert abs(result["fe_adjusted_metrics"]["pair"]["auc"] - 0.475573254) < 1e-9
    assert abs(result["fe_adjusted_metrics"]["pair"]["spearman"] + 0.070322753) < 1e-9
    assert abs(result["fe_adjusted_metrics"]["source"]["auc"] - 0.570364971) < 1e-9
    assert abs(result["fe_adjusted_metrics"]["source"]["spearman"] - 0.072709673) < 1e-9
    assert result["relation_benefit_reproduction_error"] < 1e-6


def test_b5_manifest_selects_seed_specific_checkpoint_provenance():
    from experiments.b5_semantic_rate import train_b5_a0_shared_semantic_rate as train

    manifest = ROOT / "experiments/b5_semantic_rate/b5_a0_condition_manifest.json"
    expected_z = {
        20: a04.EXPECTED_CANONICAL_Z_HASH,
        30: "31bbf7426b0982ef301da938669a1ac482284fdf98e830663ca87b1ceaebbf27",
        50: "da83e291d70a787f9da4b237fbe6beb555eac8955e6fca2d867dbd41eee1b606",
    }
    for seed in (20, 30, 50):
        entries = train._load_and_validate_manifest(
            manifest,
            model_seed=seed,
            requested_conditions=("snr2p5_k2",),
        )
        entry = entries["snr2p5_k2"]
        assert entry["model_seed"] == seed
        assert entry["corruption_seed"] == seed
        assert entry["expected_z_hash_b4_compatible"] == expected_z[seed]


def test_all_completed_seed_alignments_reproduce_their_own_a02_matrix():
    directories = {
        20: "a04_relation_utility_alignment_step100_seed20",
        30: "a05_utility_alignment_step100_seed30",
        50: "a05_utility_alignment_step100_seed50",
    }
    for seed, directory in directories.items():
        path = (
            ROOT
            / "outputs/b5_semantic_rate"
            / directory
            / "b5_a04_relation_utility_alignment.json"
        )
        result = json.loads(path.read_text())
        assert result["model_seed"] == seed
        assert result["relation_benefit_reproduction_error"] < 1e-6
        assert result["B5_A04_RELATION_BENEFIT_REPRO_PASS"]
        assert result["B5_A04_ALIGNMENT_AUDIT_COMPLETE"]
