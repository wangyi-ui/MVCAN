"""Tests for E4-CF0 counterfactual marginal information utility."""

import hashlib
import inspect
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.e4_semantic_memory_bank import (
    evaluate_e4cf0_counterfactual_utility as evaluator,
)
from experiments.e4_semantic_memory_bank.e4cf0_counterfactual_utility import (
    MATCHED_PAIR_NUM,
    PROXY_NAMES,
    WRITER_NUM,
    build_counterfactual_marginal_utility,
    build_e4cf0_decision,
    build_matched_proxy_shuffles,
    compare_proxies_to_delta,
    matched_pair_ranking,
    proxy_correlations,
    summarize_delta,
)
from experiments.e4_semantic_memory_bank.semantic_memory_bank import (
    validate_sparse_labels,
)
from irv.b4_information_utility import tensor_sha256


FROZEN_LABELED_IDS = np.array(
    [67, 82, 90, 111, 200, 365, 440, 513, 536, 983, 1027, 1250, 1316, 1385],
    dtype=np.int64,
)
FROZEN_LABELED_TARGETS = np.array(
    [4, 6, 2, 3, 0, 4, 1, 5, 6, 3, 0, 2, 5, 1],
    dtype=np.int64,
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inputs():
    generator = torch.Generator().manual_seed(411)
    h_labeled = F.normalize(
        torch.rand(14, 6, 7, generator=generator, dtype=torch.float64),
        dim=-1,
    ).detach()
    return h_labeled, FROZEN_LABELED_TARGETS.copy()


@pytest.fixture(scope="module")
def fixed_counterfactual():
    h_labeled, targets = _inputs()
    return (
        h_labeled,
        targets,
        build_counterfactual_marginal_utility(h_labeled, targets),
    )


def _class_rows(targets, class_id):
    return np.flatnonzero(np.asarray(targets) == class_id)


def _ordered_proxy_scores():
    scores = {}
    with np.load(
        evaluator.DEFAULT_E4A1_OUTPUT_DIR / "utility_components.npz",
        allow_pickle=False,
    ) as archive:
        for proxy_name in PROXY_NAMES:
            key = evaluator.FROZEN_COMPONENT_KEYS[proxy_name]
            scores[proxy_name] = torch.from_numpy(
                np.array(archive[key], copy=True, order="C")
            )
    return scores


def _comparison(
    R=(0.4, -0.1),
    S=(0.4, -0.1),
    U=(0.4, -0.1),
    shuffled_R=(0.3, -0.1),
    shuffled_S=(0.3, -0.1),
    shuffled_U=(0.3, -0.1),
):
    def record(pair):
        return {
            "Pearson_correlation": pair[1],
            "pairwise_rank_accuracy": pair[0],
            "Spearman_correlation": pair[1],
        }

    return {
        "comparisons": {
            "R": record(R),
            "SHUFFLED_R": record(shuffled_R),
            "S": record(S),
            "SHUFFLED_S": record(shuffled_S),
            "U_RS": record(U),
            "SHUFFLED_U_RS": record(shuffled_U),
        }
    }


def test_writer_count_shapes_finite_and_detached(fixed_counterfactual):
    _, _, result = fixed_counterfactual

    assert WRITER_NUM == 84
    assert result["audit"]["writer_count"] == 84
    for key in ("Delta_mem", "Q_plus", "Q_minus"):
        tensor = result[key]
        assert tensor.shape == (14, 6)
        assert torch.isfinite(tensor).all()
        assert not tensor.requires_grad and tensor.grad_fn is None


def test_exactly_two_labels_per_class():
    targets, counts = validate_sparse_labels(FROZEN_LABELED_TARGETS)

    assert targets.shape == (14,)
    assert counts.tolist() == [2] * 7


def test_heldout_peer_is_unique_different_and_same_class(
    fixed_counterfactual,
):
    _, targets, result = fixed_counterfactual
    peers = result["heldout_peer_rows"].cpu().numpy()

    assert result["audit"]["heldout_peer_count"] == [1] * 14
    for writer_row, peer_row in enumerate(peers):
        assert peer_row != writer_row
        assert targets[peer_row] == targets[writer_row]
    assert result["audit"]["heldout_peer_differs_from_writer_pass"]
    assert result["audit"]["heldout_peer_same_class_pass"]


def test_heldout_query_is_normalized_peer_mean(fixed_counterfactual):
    h_labeled, _, result = fixed_counterfactual
    peers = result["heldout_peer_rows"]
    expected = F.normalize(h_labeled[peers].mean(dim=1), dim=-1)

    assert result["heldout_query"].shape == (14, 7)
    assert torch.equal(result["heldout_query"], expected)


def test_true_class_memory_uses_writer_only_and_excludes_peer(
    fixed_counterfactual,
):
    h_labeled, _, result = fixed_counterfactual
    expected_plus = F.normalize(h_labeled.sum(dim=1), dim=-1)
    expected_minus = F.normalize(
        h_labeled.sum(dim=1)[:, None, :] - h_labeled,
        dim=-1,
    )

    assert torch.equal(result["true_prototype_plus"], expected_plus)
    assert torch.equal(result["true_prototype_minus"], expected_minus)
    assert result["audit"][
        "heldout_sample_excluded_from_true_class_memory"
    ]


def test_plus_has_six_minus_has_five_and_removes_exact_view(
    fixed_counterfactual,
):
    _, _, result = fixed_counterfactual
    plus_mask = result["plus_included_view_mask"]
    minus_mask = result["minus_included_view_mask"]

    assert torch.all(plus_mask.sum(dim=2) == 6)
    assert torch.all(minus_mask.sum(dim=2) == 5)
    for writer_row in range(14):
        for view_id in range(6):
            assert plus_mask[writer_row, view_id, view_id]
            assert not minus_mask[writer_row, view_id, view_id]
            retained = torch.nonzero(
                minus_mask[writer_row, view_id], as_tuple=False
            ).flatten()
            assert retained.tolist() == [
                other for other in range(6) if other != view_id
            ]
    assert result["audit"][
        "plus_has_exactly_6_writer_source_views_pass"
    ]
    assert result["audit"][
        "minus_has_exactly_5_writer_source_views_pass"
    ]
    assert result["audit"]["removed_view_is_exactly_writer_view_pass"]


def test_negative_prototypes_use_both_samples_all_views_and_are_shared(
    fixed_counterfactual,
):
    h_labeled, targets, result = fixed_counterfactual
    expected = []
    targets_tensor = torch.as_tensor(targets)
    for class_id in range(7):
        expected.append(
            F.normalize(
                h_labeled[targets_tensor == class_id].sum(dim=(0, 1)),
                dim=0,
            )
        )

    assert torch.equal(
        result["class_all_view_prototypes"], torch.stack(expected)
    )
    assert result["audit"][
        "negative_prototypes_shared_plus_minus_pass"
    ]


def test_counterfactual_formula_is_exact(fixed_counterfactual):
    h_labeled, targets, result = fixed_counterfactual
    writer_row, view_id = 3, 4
    peer_row = int(result["heldout_peer_rows"][writer_row])
    class_id = int(targets[writer_row])
    q_peer = F.normalize(h_labeled[peer_row].mean(dim=0), dim=0)
    m_plus = F.normalize(h_labeled[writer_row].sum(dim=0), dim=0)
    m_minus = F.normalize(
        h_labeled[writer_row].sum(dim=0) - h_labeled[writer_row, view_id],
        dim=0,
    )
    negatives = []
    target_tensor = torch.as_tensor(targets)
    for negative_class in range(7):
        if negative_class != class_id:
            negatives.append(
                F.normalize(
                    h_labeled[target_tensor == negative_class].sum(
                        dim=(0, 1)
                    ),
                    dim=0,
                )
            )
    hard_negative = torch.max(q_peer @ torch.stack(negatives).T)
    expected_plus = q_peer @ m_plus - hard_negative
    expected_minus = q_peer @ m_minus - hard_negative

    assert torch.equal(result["Q_plus"][writer_row, view_id], expected_plus)
    torch.testing.assert_close(
        result["Q_minus"][writer_row, view_id],
        expected_minus,
        rtol=1e-14,
        atol=1e-15,
    )
    torch.testing.assert_close(
        result["Delta_mem"][writer_row, view_id],
        expected_plus - expected_minus,
        rtol=1e-14,
        atol=1e-15,
    )


def test_delta_builder_has_only_sparse_semantic_inputs():
    parameters = tuple(
        inspect.signature(build_counterfactual_marginal_utility).parameters
    )
    calculation_source = inspect.getsource(
        build_counterfactual_marginal_utility
    ).split("return {", 1)[0]

    assert parameters == ("h_labeled", "labels_labeled")
    for forbidden in (
        "R_labeled",
        "S_mem",
        "U_RS",
        "oracle",
        "corruption",
        "full_GT",
    ):
        assert forbidden not in calculation_source


def test_delta_summary_uses_only_frozen_numerical_zero_tolerance():
    values = torch.linspace(-0.2, 0.2, 84, dtype=torch.float64).reshape(14, 6)
    values[0, 0] = 0.0
    values[0, 1] = 1e-12
    values[0, 2] = -1e-12
    values[0, 3] = 2e-12
    summary = summarize_delta(values)

    assert summary["writer_count"] == 84
    assert summary["delta_zero_fraction"] == pytest.approx(3 / 84)
    assert summary["zero_numerical_tolerance"] == 1e-12
    for key in (
        "delta_mean",
        "delta_std",
        "delta_min",
        "delta_max",
        "delta_positive_fraction",
        "delta_negative_fraction",
        "delta_abs_mean",
        "delta_p25",
        "delta_p50",
        "delta_p75",
    ):
        assert np.isfinite(summary[key])
    with pytest.raises(ValueError, match="frozen at 1e-12"):
        summarize_delta(values, zero_tolerance=1e-8)


def test_counterfactual_seal_saves_mapping_reloads_and_hashes(
    tmp_path, fixed_counterfactual
):
    _, _, result = fixed_counterfactual
    seal = evaluator.save_counterfactual_seal(
        result, FROZEN_LABELED_IDS, tmp_path
    )
    artifact_path = tmp_path / "counterfactual_utility.npz"

    assert artifact_path.is_file()
    assert (tmp_path / "counterfactual_seal.json").is_file()
    with np.load(artifact_path, allow_pickle=False) as archive:
        assert archive.files == [
            "Delta_mem",
            "Q_plus",
            "Q_minus",
            "writer_sample_ids",
            "writer_views",
            "heldout_peer_ids",
        ]
        assert archive["Delta_mem"].shape == (14, 6)
        assert archive["Q_plus"].shape == (14, 6)
        assert archive["Q_minus"].shape == (14, 6)
        assert archive["writer_sample_ids"].shape == (84,)
        assert archive["writer_views"].shape == (84,)
        assert archive["heldout_peer_ids"].shape == (84,)
        assert np.array_equal(
            archive["writer_sample_ids"],
            np.repeat(FROZEN_LABELED_IDS, 6),
        )
        assert np.array_equal(archive["writer_views"], np.tile(np.arange(6), 14))
        assert seal["Delta_mem_logical_sha256"] == tensor_sha256(
            archive["Delta_mem"]
        )
    with open(
        tmp_path / "counterfactual_seal.json", "r", encoding="utf-8"
    ) as input_file:
        reloaded_seal = json.load(input_file)
    assert reloaded_seal == seal
    assert seal["Delta_saved_before_posthoc_data"]
    assert seal["Delta_reloaded_before_posthoc_data"]
    assert seal["Delta_hashed_before_posthoc_data"]


@pytest.mark.parametrize("proxy_name", PROXY_NAMES)
def test_R_S_U_exact_E4A1_replay(proxy_name):
    _, audit = evaluator.verify_e4a1_proxy_replay(_ordered_proxy_scores())

    record = audit["components"][proxy_name]
    assert record["exact_array_replay_pass"]
    assert record["exact_hash_replay_pass"]
    assert record["frozen_logical_sha256"] == (
        evaluator.EXPECTED_PROXY_SHA256[proxy_name]
    )
    assert audit["E4A1_proxy_exact_replay_pass"]


@pytest.mark.parametrize("proxy_name", PROXY_NAMES)
def test_any_proxy_replay_mismatch_hard_fails(proxy_name):
    scores = _ordered_proxy_scores()
    scores[proxy_name] = scores[proxy_name].clone()
    scores[proxy_name][0, 0] += 1.0

    with pytest.raises(
        RuntimeError, match=evaluator.E4A1_PROXY_REPLAY_MISMATCH
    ):
        evaluator.verify_e4a1_proxy_replay(scores)


def test_matched_pair_ranking_has_exactly_42_pairs_and_correct_signs():
    delta = torch.zeros(14, 6, dtype=torch.float64)
    for class_id in range(7):
        first, second = _class_rows(FROZEN_LABELED_TARGETS, class_id)
        delta[first] = torch.arange(1, 7, dtype=torch.float64)
        delta[second] = 0.0

    aligned = matched_pair_ranking(
        delta, delta, FROZEN_LABELED_TARGETS
    )
    reversed_result = matched_pair_ranking(
        -delta, delta, FROZEN_LABELED_TARGETS
    )
    ties = matched_pair_ranking(
        torch.zeros_like(delta), delta, FROZEN_LABELED_TARGETS
    )

    assert MATCHED_PAIR_NUM == 42
    assert aligned["total_pair_count"] == 42
    assert aligned["valid_pair_count"] == 42
    assert aligned["tie_pair_count"] == 0
    assert aligned["pairwise_rank_accuracy"] == 1.0
    assert reversed_result["pairwise_rank_accuracy"] == 0.0
    assert ties["valid_pair_count"] == 0
    assert ties["tie_pair_count"] == 42
    assert ties["pairwise_rank_accuracy"] is None


def test_matched_proxy_shuffles_swap_each_proxy_without_rng():
    base = torch.linspace(0.01, 0.99, 84, dtype=torch.float64).reshape(14, 6)
    scores = {
        "R": base,
        "S": torch.flip(base, dims=(1,)),
        "U_RS": base.square(),
    }
    torch.manual_seed(73)
    rng_before = torch.random.get_rng_state().clone()
    first, audit = build_matched_proxy_shuffles(
        scores, FROZEN_LABELED_TARGETS
    )
    second, _ = build_matched_proxy_shuffles(
        scores, FROZEN_LABELED_TARGETS
    )

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert audit["matched_pair_count"] == 42
    assert audit["deterministic_no_RNG_pass"]
    for proxy_name in PROXY_NAMES:
        assert torch.equal(first[proxy_name], second[proxy_name])
        assert not torch.equal(first[proxy_name], scores[proxy_name])
        for class_id in range(7):
            first_row, second_row = _class_rows(
                FROZEN_LABELED_TARGETS, class_id
            )
            assert torch.equal(
                first[proxy_name][first_row], scores[proxy_name][second_row]
            )
            assert torch.equal(
                first[proxy_name][second_row], scores[proxy_name][first_row]
            )
            assert torch.equal(
                first[proxy_name][[first_row, second_row]].sum(dim=0),
                scores[proxy_name][[first_row, second_row]].sum(dim=0),
            )


def test_proxy_correlations_and_comparison_cover_direct_and_null():
    base = torch.linspace(0.01, 0.99, 84, dtype=torch.float64).reshape(14, 6)
    scores = {
        "R": base,
        "S": torch.flip(base, dims=(1,)),
        "U_RS": base.square(),
    }
    correlations = proxy_correlations(base, 3.0 * base + 1.0)
    comparison = compare_proxies_to_delta(
        scores, base, FROZEN_LABELED_TARGETS
    )

    assert correlations["Pearson_correlation"] == pytest.approx(1.0)
    assert correlations["Spearman_correlation"] == pytest.approx(1.0)
    assert tuple(comparison["comparisons"]) == (
        "R",
        "SHUFFLED_R",
        "S",
        "SHUFFLED_S",
        "U_RS",
        "SHUFFLED_U_RS",
    )
    for name in comparison["comparisons"]:
        assert "Pearson_correlation" in comparison["comparisons"][name]
        assert "Spearman_correlation" in comparison["comparisons"][name]
        assert comparison["comparisons"][name]["total_pair_count"] == 42


def test_decision_A_is_exact():
    decision = build_e4cf0_decision(
        {"delta_std": 1.1e-6},
        _comparison(U=(0.7, 0.2), shuffled_U=(0.4, -0.2)),
    )

    assert decision["Decision_A_condition"]
    assert not decision["Decision_B_condition"]
    assert not decision["Decision_C_condition"]
    assert not decision["Decision_D_condition"]
    assert decision["final_decision"] == "E4CF0_PROXY_HAS_ACTION_VALUE"


def test_decision_B_is_exact():
    decision = build_e4cf0_decision(
        {"delta_std": 1.1e-6},
        _comparison(
            R=(0.6, 0.0),
            S=(0.5, 0.3),
            U=(0.4, 0.2),
            shuffled_U=(0.3, -0.1),
        ),
    )

    assert not decision["Decision_A_condition"]
    assert decision["Decision_B_condition"]
    assert not decision["Decision_C_condition"]
    assert not decision["Decision_D_condition"]
    assert decision["final_decision"] == "E4CF0_HANDCRAFTED_PROXY_FAIL"


@pytest.mark.parametrize("delta_std", [0.0, 1e-6])
def test_decision_C_is_exact(delta_std):
    decision = build_e4cf0_decision(
        {"delta_std": delta_std},
        _comparison(U=(1.0, 1.0), shuffled_U=(0.0, -1.0)),
    )

    assert not decision["Decision_A_condition"]
    assert not decision["Decision_B_condition"]
    assert decision["Decision_C_condition"]
    assert not decision["Decision_D_condition"]
    assert decision["final_decision"] == (
        "E4CF0_MEMORY_WRITER_ACTION_UNIDENTIFIABLE"
    )


def test_mixed_proxy_gap_is_exhaustive_decision_D():
    decision = build_e4cf0_decision(
        {"delta_std": 0.1},
        _comparison(
            R=(0.7, 0.2),
            S=(0.4, -0.1),
            U=(0.4, 0.2),
            shuffled_U=(0.3, -0.1),
        ),
    )

    assert decision["delta_identifiable"]
    assert decision["R_proxy_joint_positive"]
    assert not decision["S_proxy_joint_positive"]
    assert not decision["U_proxy_joint_positive"]
    assert decision["any_proxy_joint_positive"]
    assert not decision["U_pairwise_gt_half"]
    assert decision["U_pairwise_gt_shuffle"]
    assert decision["U_spearman_positive"]
    assert not decision["Decision_A_condition"]
    assert not decision["Decision_B_condition"]
    assert not decision["Decision_C_condition"]
    assert decision["Decision_D_condition"]
    assert decision["final_decision"] == (
        "E4CF0_MIXED_PROXY_ACTION_VALUE_EVIDENCE"
    )
    assert decision["decision_explanation"] == (
        "Counterfactual memory-writer action values are identifiable, "
        "but the preregistered handcrafted proxy conclusions are mixed: "
        "at least one proxy exhibits non-null action-value association, "
        "while the primary RS proxy does not satisfy the full "
        "E4CF0_PROXY_HAS_ACTION_VALUE gate."
    )
    assert decision["Decision_D_guardrails"] == {
        "is_PASS": False,
        "allows_online_memory_training": False,
        "interpreted_as_RS_PASS": False,
        "interpreted_as_memory_FAIL": False,
        "automatically_enters_estimator_training": False,
    }


def test_decision_output_saves_direct_shuffled_components():
    decision = build_e4cf0_decision(
        {"delta_std": 0.1},
        _comparison(
            R=(0.7, 0.2),
            S=(0.6, 0.3),
            U=(0.8, 0.4),
            shuffled_R=(0.2, -0.2),
            shuffled_S=(0.3, -0.3),
            shuffled_U=(0.4, -0.4),
        ),
    )

    assert decision["proxy_component_metrics"] == {
        "R": {
            "pearson": 0.2,
            "spearman": 0.2,
            "pairwise_rank_accuracy": 0.7,
            "shuffled_pearson": -0.2,
            "shuffled_spearman": -0.2,
            "shuffled_pairwise_rank_accuracy": 0.2,
        },
        "S": {
            "pearson": 0.3,
            "spearman": 0.3,
            "pairwise_rank_accuracy": 0.6,
            "shuffled_pearson": -0.3,
            "shuffled_spearman": -0.3,
            "shuffled_pairwise_rank_accuracy": 0.3,
        },
        "U_RS": {
            "pearson": 0.4,
            "spearman": 0.4,
            "pairwise_rank_accuracy": 0.8,
            "shuffled_pearson": -0.4,
            "shuffled_spearman": -0.4,
            "shuffled_pairwise_rank_accuracy": 0.4,
        },
    }


def test_decision_tree_is_exhaustive_for_all_component_booleans():
    for flags in itertools.product((False, True), repeat=8):
        (
            delta_identifiable,
            R_pairwise_gt_half,
            R_spearman_positive,
            S_pairwise_gt_half,
            S_spearman_positive,
            U_pairwise_gt_half,
            U_pairwise_gt_shuffle,
            U_spearman_positive,
        ) = flags
        U_accuracy = 0.6 if U_pairwise_gt_half else 0.4
        shuffled_U_accuracy = (
            U_accuracy - 0.1
            if U_pairwise_gt_shuffle
            else U_accuracy + 0.1
        )
        decision = build_e4cf0_decision(
            {"delta_std": 1.1e-6 if delta_identifiable else 1e-6},
            _comparison(
                R=(
                    0.6 if R_pairwise_gt_half else 0.4,
                    0.2 if R_spearman_positive else -0.2,
                ),
                S=(
                    0.6 if S_pairwise_gt_half else 0.4,
                    0.2 if S_spearman_positive else -0.2,
                ),
                U=(
                    U_accuracy,
                    0.2 if U_spearman_positive else -0.2,
                ),
                shuffled_U=(shuffled_U_accuracy, -0.2),
            ),
        )

        active = [
            decision["Decision_A_condition"],
            decision["Decision_B_condition"],
            decision["Decision_C_condition"],
            decision["Decision_D_condition"],
        ]
        assert sum(active) == 1, flags
        assert decision["delta_identifiable"] is delta_identifiable
        assert (
            decision["U_pairwise_gt_half"] is U_pairwise_gt_half
        )
        assert (
            decision["U_pairwise_gt_shuffle"] is U_pairwise_gt_shuffle
        )
        assert (
            decision["U_spearman_positive"] is U_spearman_positive
        )


def test_seal_precedes_proxy_and_corruption_and_full_GT_is_never_loaded():
    run_source = inspect.getsource(evaluator.run_evaluation)

    assert run_source.index("build_counterfactual_marginal_utility") < (
        run_source.index("save_counterfactual_seal")
    )
    assert run_source.index("save_counterfactual_seal") < run_source.index(
        "recompute_and_replay_proxies"
    )
    assert run_source.index("save_counterfactual_seal") < run_source.index(
        "corruption_posthoc_analysis"
    )
    assert "load_full_ground_truth" not in run_source
    assert "full_GT" not in tuple(inspect.signature(evaluator.run_evaluation).parameters)


def test_no_training_optimizer_or_backward_in_CF0_sources():
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        (root / relative_path).read_text(encoding="utf-8")
        for relative_path in (
            "experiments/e4_semantic_memory_bank/e4cf0_counterfactual_utility.py",
            (
                "experiments/e4_semantic_memory_bank/"
                "evaluate_e4cf0_counterfactual_utility.py"
            ),
        )
    )

    for forbidden in ("torch.optim", ".backward(", ".train(", ".fit("):
        assert forbidden not in source


def test_frozen_representation_is_reused_without_new_alignment():
    source = inspect.getsource(evaluator.run_evaluation)

    assert "load_e1_lwc_semantic_representation" in source
    assert "align_q_to_global" not in source
    assert "h_sem[labeled_index].detach()" in source


def test_future_output_contract_and_formal_path_are_frozen():
    assert evaluator.DEFAULT_OUTPUT_DIR == (
        evaluator.REPOSITORY_ROOT
        / "outputs/e4_semantic_memory_bank"
        / "e4cf0_counterfactual_utility_seed20"
    )
    run_source = inspect.getsource(evaluator.run_evaluation)
    combined_source = run_source + inspect.getsource(
        evaluator.save_counterfactual_seal
    )
    for artifact in (
        "counterfactual_utility.npz",
        "proxy_comparison.json",
        "counterfactual_seal.json",
        "e4cf0_audit.json",
        "diagnostic_results.json",
    ):
        assert artifact in combined_source


def test_E4A0_and_E4A1_files_remain_bitwise_frozen():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "experiments/e4_semantic_memory_bank/semantic_memory_bank.py": (
            "8c4f99e1cf31ec03eb69183445076177362e6b4adc6e6dc56350c76f41a0afe6"
        ),
        (
            "experiments/e4_semantic_memory_bank/"
            "evaluate_e4_a0_memory_feasibility.py"
        ): "5f9e26d4ef3d446f22743dc4f7900369e9a0d82bde6f13a170eaa89278e99e4c",
        "tests/test_e4_a0_semantic_memory_bank.py": (
            "4ed15f1bb3f20e4788a07f820eb685bc75de1da5239161ed593db96d54a2e28c"
        ),
        (
            "experiments/e4_semantic_memory_bank/"
            "e4a1_memory_specific_utility.py"
        ): "ec0b6af7867ac19795abe7cafcc15c5e793421c1fc2d2f4c73920a8be76ac7de",
        (
            "experiments/e4_semantic_memory_bank/"
            "evaluate_e4_a1_memory_specific_utility.py"
        ): "123e04e616df758f38d383d9c5776ef7398f2078b50556919d7faa712e40f621",
        "tests/test_e4_a1_memory_specific_utility.py": (
            "8d5906b13b061cd163a2ff2db585026dfefbdf0a324d37073c0cf0a2fafd8d37"
        ),
    }

    for relative_path, expected_sha in expected.items():
        assert _file_sha256(root / relative_path) == expected_sha
