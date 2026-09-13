import hashlib
import inspect
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.cyclic_utility import c2_a0_sparse_label_utility_protocol as c2
from experiments.cyclic_utility import c5_a0_utility_validated_pseudo_semantic_protocol as c5
from experiments.cyclic_utility import evaluate_c5_a0_utility_validated_pseudo_semantic as evaluator
from experiments.cyclic_utility import run_c5_a0_utility_validated_pseudo_semantic as runner
from experiments.cyclic_utility import summarize_c5_a0_utility_validated_pseudo_semantic as summarizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sparse_split():
    labeled = c2.fixed_labeled_ids()
    targets = c2.fixed_labeled_targets()
    unlabeled = np.setdiff1d(np.arange(c5.N, dtype=np.int64), labeled)
    return labeled, targets, unlabeled


@pytest.fixture(scope="module")
def q_aligned(sparse_split):
    labeled, targets, _ = sparse_split
    classes = np.arange(c5.N, dtype=np.int64) % c5.K
    classes[labeled] = targets
    q = torch.nn.functional.one_hot(
        torch.from_numpy(classes), num_classes=c5.K
    ).to(torch.float64)
    return q[:, None, :].repeat(1, c5.V, 1).detach()


@pytest.fixture(scope="module")
def candidates(q_aligned, sparse_split):
    generator, _ = c5.frozen_direction_definitions()
    return c5.build_directional_semantic_candidates(
        q_aligned, sparse_split[0], sparse_split[1], sparse_split[2], generator
    )


@pytest.fixture(scope="module")
def utility(sparse_split):
    values = np.arange(c5.N * c5.S, dtype=np.float64).reshape(c5.N, c5.S)
    return values, c5.extract_unlabeled_utility(
        values, np.arange(c5.N, dtype=np.int64), sparse_split[2]
    )


def test_01_directional_candidate_shapes(candidates):
    assert candidates["directional_evidence"].shape == (1386, 20, 7)
    assert candidates["directional_posterior"].shape == (1386, 20, 7)
    assert candidates["directional_pred"].shape == (1386, 20)
    assert candidates["directional_margin"].shape == (1386, 20)


def test_02_candidate_rows_sum_to_one(candidates):
    posterior = candidates["directional_posterior"]
    assert torch.allclose(
        posterior.sum(-1), torch.ones((c5.NU, c5.S), dtype=posterior.dtype),
        rtol=0.0, atol=c5.ROW_SUM_ATOL,
    )


def test_03_candidate_is_finite_and_nonnegative(candidates):
    for key in ("directional_evidence", "directional_posterior"):
        value = candidates[key]
        assert torch.isfinite(value).all()
        assert (value >= 0).all()


def test_04_verifier_only_q_change_does_not_change_direction_candidate(
    q_aligned, sparse_split,
):
    generator, verifier = c5.frozen_direction_definitions()
    baseline = c5.build_directional_semantic_candidates(
        q_aligned, *sparse_split[:2], sparse_split[2], generator
    )["directional_posterior"][:, 0]
    changed = q_aligned.clone()
    view_id = int(verifier[0, 0])
    changed[:, view_id] = torch.roll(changed[:, view_id], 1, dims=1)
    actual = c5.build_directional_semantic_candidates(
        changed, *sparse_split[:2], sparse_split[2], generator
    )["directional_posterior"][:, 0]
    assert torch.equal(actual, baseline)


def test_05_generator_q_change_changes_direction_candidate(q_aligned, sparse_split):
    generator, _ = c5.frozen_direction_definitions()
    baseline = c5.build_directional_semantic_candidates(
        q_aligned, *sparse_split[:2], sparse_split[2], generator
    )["directional_posterior"][:, 0]
    changed = q_aligned.clone()
    view_id = int(generator[0, 0])
    changed[sparse_split[2], view_id] = torch.roll(
        changed[sparse_split[2], view_id], 1, dims=1
    )
    actual = c5.build_directional_semantic_candidates(
        changed, *sparse_split[:2], sparse_split[2], generator
    )["directional_posterior"][:, 0]
    assert not torch.equal(actual, baseline)


def test_06_candidate_reuses_C3_relation_probability():
    source = inspect.getsource(c5.build_directional_semantic_candidates)
    assert "c3b0.relation_probability(" in source
    assert "cosine" not in source.lower()
    assert "temperature" not in source.lower()


def test_07_all_ones_aggregation_equals_direction_mean(candidates):
    directional = candidates["directional_posterior"]
    result = c5.aggregate_pseudo_semantics(
        directional, torch.ones((c5.NU, c5.S), dtype=directional.dtype)
    )
    assert torch.allclose(result["posterior"], directional.mean(1), rtol=0.0, atol=1e-12)


def test_08_true_aggregation_toy_correctness():
    directional = torch.zeros((c5.NU, c5.S, c5.K), dtype=torch.float64)
    directional[:, :, 0] = 1.0
    directional[:, 0, :] = 0.0
    directional[:, 0, 1] = 1.0
    weights = torch.ones((c5.NU, c5.S), dtype=torch.float64)
    weights[:, 0] = 20.0
    result = c5.aggregate_pseudo_semantics(directional, weights)
    assert torch.all(result["prediction"] == 1)
    assert torch.allclose(
        result["posterior"][:, :2],
        torch.tensor([19.0 / 39.0, 20.0 / 39.0], dtype=torch.float64),
    )


@pytest.mark.parametrize("kind", ["zero", "nan", "inf", "negative"])
def test_09_invalid_action_weights_fail_closed(candidates, kind):
    weights = torch.ones((c5.NU, c5.S), dtype=torch.float64)
    if kind == "zero":
        weights[0] = 0
    elif kind == "nan":
        weights[0, 0] = float("nan")
    elif kind == "inf":
        weights[0, 0] = float("inf")
    else:
        weights[0, 0] = -1
    with pytest.raises(RuntimeError, match="ACTION_WEIGHT_FAIL_CLOSED"):
        c5.aggregate_pseudo_semantics(candidates["directional_posterior"], weights)


@pytest.mark.parametrize("shape", [(c5.N, 6), (c5.N,)])
def test_10_wrong_U_shape_fails_closed(sparse_split, shape):
    with pytest.raises(RuntimeError, match="U_CYCLE_BOUNDARY_FAIL_CLOSED"):
        c5.extract_unlabeled_utility(
            np.ones(shape), np.arange(c5.N), sparse_split[2]
        )


def test_11_direction_count_not_20_fails_closed():
    generator, verifier = c5.frozen_direction_definitions()
    with pytest.raises(RuntimeError, match="DIRECTION_DEFINITION_FAIL_CLOSED"):
        c5.validate_direction_definitions(generator[:19], verifier[:19])


def test_12_generator_shape_not_20_by_3_fails_closed(q_aligned, sparse_split):
    generator, _ = c5.frozen_direction_definitions()
    with pytest.raises(RuntimeError, match="GENERATOR_DEFINITION_FAIL_CLOSED"):
        c5.build_directional_semantic_candidates(
            q_aligned, *sparse_split[:2], sparse_split[2], generator[:, :2]
        )


def test_13_sample_ID_mismatch_fails_closed(utility, sparse_split):
    sample_ids = np.arange(c5.N)
    sample_ids[[0, 1]] = sample_ids[[1, 0]]
    with pytest.raises(RuntimeError, match="U_CYCLE_BOUNDARY_FAIL_CLOSED"):
        c5.extract_unlabeled_utility(utility[0], sample_ids, sparse_split[2])


def test_14_labeled_unlabeled_overlap_fails_closed(sparse_split):
    unlabeled = sparse_split[2].copy()
    unlabeled[0] = sparse_split[0][0]
    with pytest.raises(RuntimeError):
        c5.validate_sparse_inputs(sparse_split[0], sparse_split[1], unlabeled)


def test_15_all_original_14_anchors_are_used(candidates, sparse_split):
    assert sparse_split[0].shape == (14,)
    assert candidates["all_14_sparse_anchors_used"] is True
    assert candidates["anchors_per_class"] == 2
    source = inspect.getsource(c5)
    assert "anchor_consistency" not in source
    assert "sample 200" not in source


def test_16_C4_is_not_parent_or_input():
    for module in (c5, runner, evaluator, summarizer):
        source = inspect.getsource(module)
        assert "import c4_" not in source
        assert "from experiments.cyclic_utility import c4_" not in source


def test_17_pre_GT_whitelist_forbids_oracle_fields():
    valid = OrderedDict((key, np.empty(0)) for key in c5.PRE_GT_ARRAY_KEYS)
    assert c5.validate_pre_gt_array_keys(valid)
    for forbidden in ("full_GT", "y_true", "correctness", "ACC"):
        invalid = OrderedDict(valid)
        invalid[forbidden] = np.empty(0)
        with pytest.raises(RuntimeError, match="WHITELIST_FAIL_CLOSED"):
            c5.validate_pre_gt_array_keys(invalid)


def test_18_runner_has_no_training_or_feedback_path():
    source = inspect.getsource(runner)
    for forbidden in (
        ".backward(", ".step(", "loss.backward(", "torch.optim",
        "relation_semantic_phase", "native_consolidation_phase",
    ):
        assert forbidden not in source
    assert '"training_performed": False' in source
    assert '"C5_output_fed_back_into_training": False' in source


def test_19_outputs_are_detached(candidates, utility):
    directional = candidates["directional_posterior"].clone().requires_grad_(True)
    result = c5.aggregate_pseudo_semantics(
        directional, torch.from_numpy(np.array(utility[1], copy=True))
    )
    assert all(not result[key].requires_grad for key in result)
    assert all(result[key].grad_fn is None for key in result)


def test_19b_candidate_outputs_are_detached(q_aligned, sparse_split):
    generator, _ = c5.frozen_direction_definitions()
    q = q_aligned.clone().requires_grad_(True)
    result = c5.build_directional_semantic_candidates(
        q, *sparse_split[:2], sparse_split[2], generator
    )
    for key in (
        "directional_evidence", "directional_posterior",
        "directional_pred", "directional_margin",
    ):
        assert result[key].requires_grad is False
        assert result[key].grad_fn is None


def test_20_coverage_tie_break_is_confidence_then_sample_ID(sparse_split):
    confidence = np.zeros(c5.NU)
    selected = c5.deterministic_coverage_selection(
        confidence, sparse_split[2][::-1], 0.10
    )
    assert np.array_equal(selected, np.sort(sparse_split[2])[:138])


def test_21_AUC_synthetic_sanity():
    correct = np.asarray([0, 0, 1, 1], dtype=np.int8)
    scores = np.asarray([0.0, 0.1, 0.9, 1.0])
    assert c5.binary_auc_fail_closed(correct, scores) == 1.0
    with pytest.raises(RuntimeError, match="AUC_UNDEFINED_FAIL_CLOSED"):
        c5.binary_auc_fail_closed(np.ones(4), scores)


def test_22_permuted_U_exact_fixed_mapping_and_provenance(utility, sparse_split):
    result = c5.build_permuted_utility(utility[1], sparse_split[2])
    ranked = sorted(
        map(int, sparse_split[2]),
        key=lambda sid: (
            hashlib.sha256((c5.PERMUTATION_NAMESPACE + str(sid)).encode()).hexdigest(), sid
        ),
    )
    source_for_first = ranked[1]
    first_row = int(np.flatnonzero(sparse_split[2] == ranked[0])[0])
    source_row = int(np.flatnonzero(sparse_split[2] == source_for_first)[0])
    assert result["permutation_source_ids"][first_row] == source_for_first
    assert np.array_equal(result["U_permuted"][first_row], utility[1][source_row])
    assert result["RNG_used"] is False and result["GT_used"] is False


def test_23_permutation_preserves_rows_and_direction_marginals(utility, sparse_split):
    result = c5.build_permuted_utility(utility[1], sparse_split[2])
    assert result["complete_row_multiset_preserved"] is True
    assert result["every_direction_marginal_preserved"] is True
    assert result["within_row_direction_structure_preserved"] is True
    assert not np.array_equal(result["U_permuted"], utility[1])
    for direction in range(c5.S):
        assert np.array_equal(
            np.sort(result["U_permuted"][:, direction]),
            np.sort(utility[1][:, direction]),
        )


def test_24_three_arm_output_shapes(candidates, utility, sparse_split):
    permuted = c5.build_permuted_utility(utility[1], sparse_split[2])["U_permuted"]
    arms = c5.build_three_arm_outputs(
        candidates["directional_posterior"], utility[1], permuted
    )
    assert tuple(arms) == c5.ARMS
    for arm in c5.ARMS:
        assert arms[arm]["posterior"].shape == (c5.NU, c5.K)
        assert arms[arm]["prediction"].shape == (c5.NU,)
        assert arms[arm]["confidence"].shape == (c5.NU,)


def test_25_TRUE_all_ones_equals_UNIFORM(candidates):
    ones = np.ones((c5.NU, c5.S))
    arms = c5.build_three_arm_outputs(candidates["directional_posterior"], ones, ones)
    assert torch.equal(arms["TRUE_U"]["posterior"], arms["UNIFORM"]["posterior"])
    assert torch.equal(arms["TRUE_U"]["prediction"], arms["UNIFORM"]["prediction"])


def test_26_evaluator_requires_seal_before_GT(tmp_path, monkeypatch):
    called = {"GT": False}
    def forbidden(*args, **kwargs):
        called["GT"] = True
        raise AssertionError("GT boundary crossed")
    monkeypatch.setattr(evaluator.e1_train, "load_labels_after_predictions", forbidden)
    with pytest.raises(RuntimeError, match="SEALED_PRE_GT_ARTIFACT_REQUIRED"):
        evaluator.evaluate(
            20, tmp_path / "bundle", tmp_path / "audit", tmp_path / "seal",
            tmp_path / "GT", tmp_path / "metrics",
        )
    assert called["GT"] is False


def test_27_fixed_coverage_precision_counts_and_hashes(sparse_split):
    ids = sparse_split[2]
    truth = ids % c5.K
    pred = truth.copy()
    confidence = np.linspace(0, 1, c5.NU)
    result = evaluator.fixed_coverage_precision(truth, pred, confidence, ids)
    assert result["selection_count@10"] == 138
    assert result["selection_count@20"] == 277
    assert result["selection_count@100"] == 1386
    assert result["Precision@20"] == 1.0
    assert len(result["selected_sample_ids_logical_sha256@20"]) == 64


def test_28_seed_specific_carrier_paths_and_valid_seals():
    for seed in c5.SEEDS:
        artifact, audit, seal = runner.seed_specific_carrier_paths(seed)
        assert artifact.parent.name == "c3_b0_true_u_carrier_seed" + str(seed)
        arrays, provenance = runner.load_sealed_carrier(seed)
        assert arrays["q_aligned"].shape == (1400, 6, 7)
        assert arrays["M_v"].shape == (6, 7, 7)
        assert provenance["seed_specific_carrier_provenance_pass"] is True
        assert provenance["C4_used"] is False


def test_29_original_U_cycle_is_N_by_20_and_ID_indexed(utility, sparse_split):
    full, expected = utility
    actual = c5.extract_unlabeled_utility(
        full, np.arange(c5.N, dtype=np.int64), sparse_split[2]
    )
    assert full.shape == (1400, 20)
    assert actual.shape == (1386, 20)
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual, full[sparse_split[2]])


def _synthetic_metric(seed, auc_true, auc_permuted, p_true, p_uniform, p_permuted):
    return {
        "stage": c5.STAGE,
        "seed": seed,
        "GT_loaded_only_after_pre_gt_seal": True,
        "training_performed": False,
        "Action_Utility_Predictiveness": {
            "AUC_TRUE": auc_true,
            "AUC_PERMUTED": auc_permuted,
            "Delta_AUC": auc_true - auc_permuted,
        },
        "Precision@20": {
            "TRUE_U": p_true,
            "UNIFORM": p_uniform,
            "PERMUTED_U": p_permuted,
            "Delta_TRUE_U_UNIFORM": p_true - p_uniform,
            "Delta_TRUE_U_PERMUTED_U": p_true - p_permuted,
        },
    }


def test_30_summarizer_exact_means_and_positive_counts():
    records = {
        20: _synthetic_metric(20, .8, .5, .7, .6, .5),
        30: _synthetic_metric(30, .4, .5, .4, .5, .5),
        50: _synthetic_metric(50, .7, .6, .8, .7, .7),
    }
    summary = summarizer.build_summary(records)
    assert summary["mean_AUC_TRUE"] == pytest.approx((.8 + .4 + .7) / 3)
    assert summary["AUC_TRUE_gt_AUC_PERMUTED_positive_seed_count"] == 2
    assert summary["P20_TRUE_U_gt_UNIFORM_positive_seed_count"] == 2
    assert summary["P20_TRUE_U_gt_PERMUTED_U_positive_seed_count"] == 2
    assert summary["formal_decision_made"] is False


def test_31_no_historical_shuffle_names_in_C5_outputs():
    assert "SHUFFLE_U" not in c5.ARMS
    assert not any("shuffle" in key.lower() for key in c5.PRE_GT_ARRAY_KEYS)
    assert "U_permuted" in c5.PRE_GT_ARRAY_KEYS


def _utility_with_zero_mass_rows(sparse_split):
    values = np.arange(c5.NU * c5.S, dtype=np.float64).reshape(c5.NU, c5.S) + 1.0
    values[[0, 7, 100]] = 0.0
    return values


def test_v2_01_zero_mass_TRUE_rows_are_abstained_not_crashed(sparse_split):
    values = _utility_with_zero_mass_rows(sparse_split)
    eligibility = c5.build_zero_mass_abstention(values, sparse_split[2])
    assert np.array_equal(eligibility["abstained_ids"], sparse_split[2][[0, 7, 100]])
    assert not eligibility["eligible_mask"][[0, 7, 100]].any()


def test_v2_02_zero_mass_rows_never_enter_aggregation(candidates, sparse_split):
    values = _utility_with_zero_mass_rows(sparse_split)
    eligibility = c5.build_zero_mass_abstention(values, sparse_split[2])
    directional = candidates["directional_posterior"][
        torch.from_numpy(np.array(eligibility["eligible_mask"], copy=True))
    ]
    permutation = c5.build_permuted_utility(
        eligibility["U_true_eligible"], eligibility["eligible_ids"]
    )
    arms = c5.build_three_arm_outputs(
        directional, eligibility["U_true_eligible"], permutation["U_permuted"]
    )
    assert all(arms[arm]["posterior"].shape[0] == c5.NU - 3 for arm in c5.ARMS)
    assert all(arms[arm]["prediction"].shape != (c5.NU,) for arm in c5.ARMS)


def test_v2_03_common_eligible_set_across_all_three_arms(candidates, sparse_split):
    values = _utility_with_zero_mass_rows(sparse_split)
    eligibility = c5.build_zero_mass_abstention(values, sparse_split[2])
    assert eligibility["same_eligible_set_for_all_arms"] is True
    source = inspect.getsource(runner.run_pre_gt)
    assert source.count("build_zero_mass_abstention(") == 1
    assert source.count("eligible_directional") >= 3


def test_v2_04_no_UNIFORM_fallback_for_abstained_rows(sparse_split):
    eligibility = c5.build_zero_mass_abstention(
        _utility_with_zero_mass_rows(sparse_split), sparse_split[2]
    )
    assert eligibility["uniform_fallback_for_abstained_rows"] is False
    source = inspect.getsource(runner.run_pre_gt)
    assert "uniform_fallback_for_abstained_rows" in source


def test_v2_05_eligible_permuted_rows_have_positive_mass(sparse_split):
    eligibility = c5.build_zero_mass_abstention(
        _utility_with_zero_mass_rows(sparse_split), sparse_split[2]
    )
    result = c5.build_permuted_utility(
        eligibility["U_true_eligible"], eligibility["eligible_ids"]
    )
    assert np.all(eligibility["U_true_eligible"].sum(1) > c5.EPSILON)
    assert np.all(result["U_permuted"].sum(1) > c5.EPSILON)
    assert result["every_permuted_row_has_positive_action_mass"] is True


def test_v2_06_eligible_complete_row_multiset_preserved(sparse_split):
    eligibility = c5.build_zero_mass_abstention(
        _utility_with_zero_mass_rows(sparse_split), sparse_split[2]
    )
    result = c5.build_permuted_utility(
        eligibility["U_true_eligible"], eligibility["eligible_ids"]
    )
    assert result["complete_row_multiset_preserved"] is True
    assert sorted(map(tuple, result["U_permuted"])) == sorted(
        map(tuple, eligibility["U_true_eligible"])
    )


def test_v2_07_eligible_directional_marginals_preserved(sparse_split):
    eligibility = c5.build_zero_mass_abstention(
        _utility_with_zero_mass_rows(sparse_split), sparse_split[2]
    )
    result = c5.build_permuted_utility(
        eligibility["U_true_eligible"], eligibility["eligible_ids"]
    )
    for direction in range(c5.S):
        assert np.array_equal(
            np.sort(result["U_permuted"][:, direction]),
            np.sort(eligibility["U_true_eligible"][:, direction]),
        )


def test_v2_08_permutation_is_deterministic_and_uses_V2_namespace(sparse_split):
    eligibility = c5.build_zero_mass_abstention(
        _utility_with_zero_mass_rows(sparse_split), sparse_split[2]
    )
    first = c5.build_permuted_utility(
        eligibility["U_true_eligible"], eligibility["eligible_ids"]
    )
    second = c5.build_permuted_utility(
        eligibility["U_true_eligible"], eligibility["eligible_ids"]
    )
    assert c5.PERMUTATION_NAMESPACE == "C5A0_UTILITY_PERMUTE_ELIGIBLE_V2:"
    assert np.array_equal(first["permutation_source_ids"], second["permutation_source_ids"])
    assert first["permutation_logical_sha256"] == second["permutation_logical_sha256"]
    assert np.all(first["permutation_source_ids"] != eligibility["eligible_ids"])


def test_v2_09_eligibility_and_abstention_rates_are_exact(sparse_split):
    eligibility = c5.build_zero_mass_abstention(
        _utility_with_zero_mass_rows(sparse_split), sparse_split[2]
    )
    assert eligibility["eligible_count"] == c5.NU - 3
    assert eligibility["abstention_count"] == 3
    assert eligibility["eligibility_rate"] == pytest.approx((c5.NU - 3) / c5.NU)
    assert eligibility["abstention_rate"] == pytest.approx(3 / c5.NU)


def test_v2_10_evaluator_uses_identical_eligible_IDs_for_all_arms():
    source = inspect.getsource(evaluator.evaluate)
    assert source.count("eligible_ids") >= 3
    assert "unlabeled_ids" not in source
    assert "identical_eligible_ids_used_for_all_arms" in source


def test_v2_11_coverage_denominator_is_eligible_count():
    eligible_count = 17
    ids = np.arange(100, 100 + eligible_count, dtype=np.int64)
    truth = ids % c5.K
    result = evaluator.fixed_coverage_precision(
        truth, truth, np.linspace(0.0, 1.0, eligible_count), ids
    )
    assert result["selection_count@20"] == int(np.floor(0.20 * eligible_count))
    assert result["selection_count@100"] == eligible_count


def test_v2_12_effective_coverage_denominator_is_all_unlabeled():
    eligible_count = 17
    ids = np.arange(100, 100 + eligible_count, dtype=np.int64)
    truth = ids % c5.K
    result = evaluator.fixed_coverage_precision(
        truth, truth, np.linspace(0.0, 1.0, eligible_count), ids
    )
    assert result["effective_coverage_vs_all_unlabeled@20"] == pytest.approx(
        result["selection_count@20"] / 1386
    )
    assert result["effective_coverage_vs_all_unlabeled@100"] == pytest.approx(
        eligible_count / 1386
    )


def test_v2_13_GT_remains_absent_before_seal():
    source = inspect.getsource(runner.run_pre_gt)
    assert "load_labels_after_predictions" not in source
    assert "GT_loaded" in source
    assert "GT_loaded_before_pre_gt_seal" in source


def test_32_frozen_parent_and_carrier_sources_unchanged():
    expected = {
        "experiments/cyclic_utility/c0_complementary_semantic_verification.py":
            "d52fbf0816557ba57a11fc490ead1a26598b35e68bac7808b7077c448bc7a4a9",
        "experiments/cyclic_utility/c3_a0_utility_conditioned_action_granularity_protocol.py":
            "8182370acd4cda507bfa425a3f40e05779945e49359d0b6c3c80b88472cb1115",
        "experiments/cyclic_utility/c3_b0_relation_action_protocol.py":
            "9ae55b93c2803d8dea594fed1cf3f26e32e8963779a87ee5027cdf9ae3340b1d",
        "experiments/cyclic_utility/train_c3_b0_relation_action_pilot.py":
            "0a657cb6e5faeaf21478c6bb7bd626bbd8385343408185d3730eb4b366aa8d50",
        "experiments/cyclic_utility/c3_b0_true_u_carrier_protocol.py":
            "2c2ffe044910b01f322b9dbb6bf390bb756d4a16d077ce7f19f83626960362e0",
        "experiments/cyclic_utility/materialize_c3_b0_true_u_carrier.py":
            "9811fd536a355cb28e3cd21da3feb8346109b1ea42359e5c3163292e7038d6ac",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest() == digest


def test_33_runner_and_evaluator_CLI_boundaries():
    run_args = runner.parse_args(["--seed", "20"])
    assert run_args.seed == 20 and not hasattr(run_args, "full_gt_path")
    eval_args = evaluator.parse_args(["--seed", "20"])
    assert eval_args.seed == 20 and hasattr(eval_args, "full_gt_path")
