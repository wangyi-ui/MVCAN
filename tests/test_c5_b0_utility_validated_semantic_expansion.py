import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.cyclic_utility import c3_b0_relation_action_protocol as c3b0
from experiments.cyclic_utility import c5_b0_utility_validated_semantic_expansion_protocol as c5b0
from experiments.cyclic_utility import evaluate_c5_b0_utility_validated_semantic_expansion as evaluator
from experiments.cyclic_utility import run_c5_b0_utility_validated_semantic_expansion as runner
from experiments.cyclic_utility import summarize_c5_b0_utility_validated_semantic_expansion as summarizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sparse_split():
    labeled_ids = np.arange(c5b0.L, dtype=np.int64)
    labeled_labels = np.tile(np.arange(c5b0.K, dtype=np.int64), 2)
    unlabeled_ids = np.arange(c5b0.L, c5b0.N, dtype=np.int64)
    return labeled_ids, labeled_labels, unlabeled_ids


@pytest.fixture
def expansion_inputs(sparse_split):
    eligible_ids = sparse_split[2][:100]
    utility = np.arange(100 * c5b0.S, dtype=np.float64).reshape(100, c5b0.S) + 1.0
    directional_pred = (
        np.arange(100)[:, None] + np.arange(c5b0.S)[None, :]
    ) % c5b0.K
    return eligible_ids, utility, directional_pred


@pytest.fixture
def full_directional_labels():
    return (
        np.arange(c5b0.NU)[:, None] + np.arange(c5b0.S)[None, :]
    ) % c5b0.K


def _plan(arm, expansion_inputs, sparse_split):
    ids, utility, predicted = expansion_inputs
    return c5b0.build_anchor_plan(
        arm, ids, utility, predicted, sparse_split[0], sparse_split[1]
    )


@pytest.mark.parametrize("seed,eligible_count", ((20, 1285), (30, 1283), (50, 1288)))
def test_01_C5_A0_parent_seal_and_hash_verification(seed, eligible_count):
    arrays, audit = runner.load_frozen_c5a_parent(seed)
    assert audit["all_C5_A0_parent_hashes_pass"] is True
    assert audit["pre_gt_seal_verified_before_GT_load"] is True
    assert audit["GT_loaded"] is False
    assert arrays["eligible_ids"].shape == (eligible_count,)


def test_02_training_cannot_load_C5_A0_post_GT_metrics():
    source = inspect.getsource(runner.load_frozen_c5a_parent)
    assert "validate_pre_gt_seal(" in source
    assert "evaluate(" not in source
    assert "load_labels_after_predictions" not in inspect.getsource(runner)
    args = runner.parse_args(["--arm", "BASE", "--seed", "20"])
    assert not hasattr(args, "full_gt_path")


def test_03_common_eligible_ids_for_TRUE_and_PERM(expansion_inputs, sparse_split):
    ids, utility, predicted = expansion_inputs
    plans = {
        arm: c5b0.build_anchor_plan(
            arm, ids, utility, predicted, sparse_split[0], sparse_split[1]
        )
        for arm in ("TRUE_U_EXPAND", "PERMUTED_U_EXPAND")
    }
    assert np.array_equal(
        plans["TRUE_U_EXPAND"]["eligible_ids"],
        plans["PERMUTED_U_EXPAND"]["eligible_ids"],
    )


def test_04_TRUE_direction_argmax_and_smallest_tie(expansion_inputs):
    ids, utility, predicted = expansion_inputs
    utility = utility.copy()
    utility[0] = 1.0
    utility[0, 3] = utility[0, 7] = 10.0
    selected = c5b0.select_pseudo_semantic_anchors(
        "TRUE_U_EXPAND", ids, utility, predicted
    )
    assert selected["selected_direction"][0] == 3


def test_05_PERM_direction_argmax_and_smallest_tie(expansion_inputs):
    ids, utility, predicted = expansion_inputs
    utility = utility.copy()
    utility[1] = 2.0
    utility[1, 4] = utility[1, 9] = 20.0
    selected = c5b0.select_pseudo_semantic_anchors(
        "PERMUTED_U_EXPAND", ids, utility, predicted
    )
    assert selected["selected_direction"][1] == 4


def test_06_expansion_count_is_floor_point_20_Ne(expansion_inputs):
    selected = c5b0.select_pseudo_semantic_anchors(
        "TRUE_U_EXPAND", *expansion_inputs
    )
    assert selected["expansion_count"] == 20
    assert selected["selected_pseudo_anchor_ids"].shape == (20,)
    assert selected["expansion_mask"].shape == (100,)
    assert selected["expansion_mask"].sum() == 20


def test_07_ranking_is_negative_score_then_sample_ID():
    ids = np.array([40, 10, 30, 20, 50], dtype=np.int64)
    utility = np.ones((5, c5b0.S), dtype=np.float64)
    predicted = np.zeros((5, c5b0.S), dtype=np.int64)
    selected = c5b0.select_pseudo_semantic_anchors(
        "TRUE_U_EXPAND", ids, utility, predicted
    )
    assert selected["selected_pseudo_anchor_ids"].tolist() == [10]


def test_08_selected_labels_exact_directional_pred(expansion_inputs):
    selected = c5b0.select_pseudo_semantic_anchors(
        "TRUE_U_EXPAND", *expansion_inputs
    )
    _, _, predicted = expansion_inputs
    expected = predicted[
        selected["selected_rows"],
        selected["selected_direction"][selected["selected_rows"]],
    ]
    assert np.array_equal(selected["selected_pseudo_anchor_labels"], expected)


def test_09_posterior_TRUE_U_not_used_for_pseudo_label_creation():
    source = inspect.getsource(c5b0.select_pseudo_semantic_anchors)
    assert "directional_pred" in source
    assert "posterior_TRUE_U[" not in source


def test_10_no_continuous_utility_weighting(expansion_inputs):
    selected = c5b0.select_pseudo_semantic_anchors(
        "TRUE_U_EXPAND", *expansion_inputs
    )
    assert selected["continuous_U_weighting_used"] is False
    assert selected["posterior_TRUE_U_used"] is False


def test_11_no_new_scalar_utility():
    source = inspect.getsource(c5b0) + inspect.getsource(runner)
    for forbidden in ("U_tilde", "utility_new", "learned_utility"):
        assert forbidden not in source


def test_12_no_GT_before_seal():
    source = inspect.getsource(runner)
    assert "load_labels_after_predictions" not in source
    assert "full_gt_path" not in source
    assert '"GT_loaded_before_pre_gt_seal": False' in source


def test_13_no_C4_artifact_or_parent():
    source = inspect.getsource(runner)
    assert "c4_a0" not in source.lower()
    assert '"C4_used": False' in source


def test_14_no_memory_bank():
    source = (inspect.getsource(c5b0) + inspect.getsource(runner)).lower()
    assert "memory_bank" not in source
    assert '"memory_used": false' in source


def test_15_no_pseudo_CE():
    source = inspect.getsource(runner)
    assert "cross_entropy" not in source
    assert '"pseudo_CE_used": False' in source


def test_16_no_new_loss_coefficient():
    source = inspect.getsource(c5b0.relation_semantic_loss)
    assert "coefficient" not in source
    assert "pseudo_anchor_loss_weight" in source


def test_17_no_recursive_pseudo_label_refresh():
    source = inspect.getsource(runner.run_arm)
    assert source.count("build_all_anchor_plans(") == 1
    assert '"recursive_expansion_used": False' in source
    assert '"pseudo_anchor_refresh_used": False' in source


@pytest.mark.parametrize("seed", c5b0.SEEDS)
def test_18_original_14_anchor_targets_exact_frozen(seed):
    arrays, audit = runner.load_full_c3a0_action(seed)
    rebuilt = c5b0.build_relation_targets(
        arrays["class_pred_true"], arrays["labeled_targets"]
    )
    assert np.array_equal(rebuilt, arrays["PredRelation_true"])
    assert audit["equivalence_gate"]["PredRelation_true_bitwise_equal"] is True


@pytest.mark.parametrize("seed", c5b0.SEEDS)
def test_19_original_balance_weights_exact_frozen(seed):
    arrays, audit = runner.load_full_c3a0_action(seed)
    rebuilt = c5b0.build_relation_balance_weights(arrays["PredRelation_true"])
    assert np.array_equal(rebuilt, arrays["relation_balance_weights_true"])
    assert audit["equivalence_gate"]["relation_balance_weights_true_exact_equal"] is True


def test_20_pseudo_anchor_uses_same_relation_operator(full_directional_labels):
    labels = np.concatenate((np.tile(np.arange(7), 2), np.array([0, 2, 6]))).astype(np.int64)
    actual = c5b0.build_relation_targets(full_directional_labels, labels)
    expected = full_directional_labels[:, None, :] == labels[None, :, None]
    assert np.array_equal(actual, expected)


def test_21_self_relation_is_excluded(sparse_split):
    query = sparse_split[2]
    anchors = np.concatenate((sparse_split[0], query[:3]))
    pseudo = np.concatenate((np.zeros(14, dtype=bool), np.ones(3, dtype=bool)))
    mask, audit = c5b0.build_self_relation_mask(query, anchors, pseudo)
    for offset in range(3):
        assert not mask[offset, 14 + offset].any()
    assert audit["self_relation_pair_count_excluded"] == 3


def test_22_self_relation_count_used_is_zero(full_directional_labels, sparse_split):
    plan = c5b0.build_anchor_plan(
        "TRUE_U_EXPAND", sparse_split[2][:20],
        np.ones((20, c5b0.S)),
        np.zeros((20, c5b0.S), dtype=np.int64),
        sparse_split[0], sparse_split[1],
    )
    action = c5b0.build_training_action(
        np.ones((c5b0.NU, c5b0.S)), sparse_split[2],
        full_directional_labels, plan,
    )
    invalid = ~action["valid_relation_mask"]
    weights = action["U_cycle"][:, None, :] * action["balance_weight"]
    weights = weights * action["valid_relation_mask"]
    assert np.count_nonzero(weights[invalid]) == 0
    assert action["self_relation_count_used"] == 0


def test_23_BASE_replay_parity_gate_fail_closed():
    valid = {
        "stage": c5b0.STAGE,
        "arm": "BASE",
        "BASE_replays_frozen_C3_B0_TRUE_U": True,
        "final_model_hash_equal": True,
        "final_predictions_equal": True,
        "final_prediction_sample_ids_equal": True,
        "coordinate_mapping_pass": True,
        "all_parent_hashes_pass": True,
        "GT_loaded_before_pre_gt_seal": False,
    }
    assert runner.validate_base_parity_record(valid)
    invalid = dict(valid, final_predictions_equal=False)
    with pytest.raises(RuntimeError, match="BASE_REPLAY_PARITY"):
        runner.validate_base_parity_record(invalid)


def test_24_TRUE_and_PERM_training_budget_configuration_identical():
    assert c5b0.FORMAL_EPOCHS == c3b0.FORMAL_EPOCHS == 20
    assert c5b0.BATCH_SIZE == c3b0.BATCH_SIZE
    source = inspect.getsource(runner.train_expansion_arm)
    assert "arm" not in inspect.signature(runner.train_expansion_arm).parameters
    assert source.count("expanded_relation_semantic_phase(") == 1
    assert source.count("native_consolidation_phase(") == 1


def test_25_combined_anchor_count_is_14_plus_B(expansion_inputs, sparse_split):
    plan = _plan("TRUE_U_EXPAND", expansion_inputs, sparse_split)
    assert plan["combined_anchor_ids"].shape == (14 + 20,)
    assert plan["combined_anchor_labels"].shape == (14 + 20,)
    assert plan["combined_anchor_is_pseudo"].sum() == 20


def test_26_selected_pseudo_anchors_do_not_overlap_real(expansion_inputs, sparse_split):
    plan = _plan("TRUE_U_EXPAND", expansion_inputs, sparse_split)
    assert np.intersect1d(
        plan["selected_pseudo_anchor_ids"], plan["real_anchor_ids"]
    ).size == 0


def test_27_selected_ids_all_belong_to_frozen_eligible(expansion_inputs, sparse_split):
    plan = _plan("PERMUTED_U_EXPAND", expansion_inputs, sparse_split)
    assert np.isin(
        plan["selected_pseudo_anchor_ids"], plan["eligible_ids"]
    ).all()


def test_28_selection_is_deterministic(expansion_inputs):
    first = c5b0.select_pseudo_semantic_anchors(
        "TRUE_U_EXPAND", *expansion_inputs
    )
    second = c5b0.select_pseudo_semantic_anchors(
        "TRUE_U_EXPAND", *expansion_inputs
    )
    for key in (
        "selected_direction", "admission_score",
        "selected_pseudo_anchor_ids", "selected_pseudo_anchor_labels",
    ):
        assert np.array_equal(first[key], second[key])


def test_29_all_frozen_C5_A0_sources_unchanged():
    expected = {
        "c5_a0_utility_validated_pseudo_semantic_protocol.py": "62a39273c74d33f96ed277bdeb98ae94973f5afb1ddceabde2429dee0cc8970c",
        "run_c5_a0_utility_validated_pseudo_semantic.py": "6775b696fb2005fb0371558740a7319b66db3c6f970a7b741f78dcba75d7f1c5",
        "evaluate_c5_a0_utility_validated_pseudo_semantic.py": "817c188a8073c1b731ddb61ad0c168dba4d39db06a586d742223dd9bf062794e",
        "summarize_c5_a0_utility_validated_pseudo_semantic.py": "86fc85b9d2bcdf8ebf958201aa720e5c62744532b251b191711293c99222bf85",
        "test_c5_a0_utility_validated_pseudo_semantic.py": "2f532977e8f99787816d175231052f7c69ea9aa20c0524db578bc46026768c5e",
    }
    root = REPOSITORY_ROOT / "experiments/cyclic_utility"
    test_path = REPOSITORY_ROOT / "tests/test_c5_a0_utility_validated_pseudo_semantic.py"
    assert hashlib.sha256(test_path.read_bytes()).hexdigest() == expected.pop(
        "test_c5_a0_utility_validated_pseudo_semantic.py"
    )
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


def test_30_all_frozen_C3_B0_sources_unchanged():
    expected = {
        "c3_b0_relation_action_protocol.py": "9ae55b93c2803d8dea594fed1cf3f26e32e8963779a87ee5027cdf9ae3340b1d",
        "train_c3_b0_relation_action_pilot.py": "0a657cb6e5faeaf21478c6bb7bd626bbd8385343408185d3730eb4b366aa8d50",
        "c3_b0_true_u_carrier_protocol.py": "2c2ffe044910b01f322b9dbb6bf390bb756d4a16d077ce7f19f83626960362e0",
        "materialize_c3_b0_true_u_carrier.py": "9811fd536a355cb28e3cd21da3feb8346109b1ea42359e5c3163292e7038d6ac",
    }
    root = REPOSITORY_ROOT / "experiments/cyclic_utility"
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


def test_31_dynamic_relation_probability_equals_C3_for_14_anchors():
    sample = torch.softmax(torch.randn(3, c5b0.K, dtype=torch.float64), dim=1)
    anchor = torch.softmax(torch.randn(14, c5b0.K, dtype=torch.float64), dim=1)
    assert torch.equal(
        c5b0.relation_probability(sample, anchor),
        c3b0.relation_probability(sample, anchor),
    )


def test_32_dynamic_loss_equals_C3_with_14_real_anchors():
    torch.manual_seed(9)
    samples = [
        torch.softmax(torch.randn(3, 7, dtype=torch.float64), 1)
        for _ in range(6)
    ]
    anchors = [
        torch.softmax(torch.randn(14, 7, dtype=torch.float64), 1)
        for _ in range(6)
    ]
    target = np.zeros((3, 14, 20), dtype=bool)
    target[:, ::2] = True
    utility = np.arange(60, dtype=float).reshape(3, 20) + 1
    balance = np.arange(3 * 14 * 20, dtype=float).reshape(3, 14, 20) + 1
    valid = np.ones_like(target, dtype=bool)
    expected, _ = c3b0.relation_semantic_loss(
        samples, anchors, target, utility, balance, "TRUE_U"
    )
    actual, audit = c5b0.relation_semantic_loss(
        samples, anchors, target, utility, balance, valid
    )
    assert torch.equal(actual, expected)
    assert audit["self_relation_count_used"] == 0


def _passing_metrics():
    return {
        seed: {
            "BASE": {"ACC": 0.50, "NMI": 0.4, "ARI": 0.3, "Balanced_ACC": 0.5},
            "TRUE_U_EXPAND": {"ACC": 0.60, "NMI": 0.5, "ARI": 0.4, "Balanced_ACC": 0.6},
            "PERMUTED_U_EXPAND": {"ACC": 0.55, "NMI": 0.45, "ARI": 0.35, "Balanced_ACC": 0.55},
        }
        for seed in c5b0.SEEDS
    }


def test_33_primary_gate_is_exact_two_ACC_comparisons():
    summary = c5b0.summarize_multiseed_metrics(_passing_metrics())
    assert len(summary["gate_checks"]) == 4
    assert summary["C5_B0_PASS"] is True
    assert summary["positive_seed_count_vs_BASE"] == 3
    assert summary["positive_seed_count_vs_PERMUTED_U_EXPAND"] == 3


def test_34_secondary_metrics_do_not_change_primary_gate():
    metrics = _passing_metrics()
    for seed in c5b0.SEEDS:
        metrics[seed]["TRUE_U_EXPAND"]["NMI"] = -100.0
        metrics[seed]["TRUE_U_EXPAND"]["ARI"] = -100.0
    summary = summarizer.build_summary(metrics)
    assert summary["C5_B0_PASS"] is True
    assert set(summary["secondary_metrics_not_used_in_gate"]) == {
        "NMI", "ARI", "Balanced_ACC"
    }


def test_35_evaluator_requires_seal_before_GT(tmp_path, monkeypatch):
    called = {"GT": False}

    def forbidden(*args, **kwargs):
        called["GT"] = True
        raise AssertionError("GT boundary crossed")

    monkeypatch.setattr(evaluator.e1_train, "load_labels_after_predictions", forbidden)
    with pytest.raises(RuntimeError, match="SEALED_PRE_GT_OUTPUT_REQUIRED"):
        evaluator.evaluate(
            20, "BASE", tmp_path / "bundle", tmp_path / "audit",
            tmp_path / "seal", tmp_path / "GT", tmp_path / "metrics",
        )
    assert called["GT"] is False


def test_36_preregistered_protocol_hash_is_fixed():
    assert runner.verify_preregistration()["verified_before_training"] is True
    assert hashlib.sha256(runner.PREREGISTRATION.read_bytes()).hexdigest() == (
        c5b0.PREREGISTERED_PROTOCOL_SHA256
    )


def test_37_BASE_anchor_plan_has_no_pseudo_anchor(expansion_inputs, sparse_split):
    plan = _plan("BASE", expansion_inputs, sparse_split)
    assert plan["expansion_count"] == 0
    assert plan["selected_pseudo_anchor_ids"].size == 0
    assert plan["combined_anchor_ids"].shape == (14,)


def test_38_relation_target_axis_order_is_explicit(full_directional_labels, sparse_split):
    plan = c5b0.build_anchor_plan(
        "TRUE_U_EXPAND", sparse_split[2][:20], np.ones((20, 20)),
        np.zeros((20, 20), dtype=np.int64), sparse_split[0], sparse_split[1]
    )
    action = c5b0.build_training_action(
        np.ones((1386, 20)), sparse_split[2], full_directional_labels, plan
    )
    assert action["PredRelation"].shape == (1386, 18, 20)
    assert action["relation_target_axis_order"] == (
        "unlabeled_query", "combined_anchor", "direction"
    )


def test_39_real_and_pseudo_anchor_posteriors_share_stop_gradient_semantics():
    torch.manual_seed(21)
    samples = [
        torch.softmax(torch.randn(2, 7, dtype=torch.float64), 1)
        .detach().requires_grad_(True)
        for _ in range(6)
    ]
    anchors = [
        torch.softmax(torch.randn(15, 7, dtype=torch.float64), 1)
        .detach().requires_grad_(True)
        for _ in range(6)
    ]
    target = np.zeros((2, 15, 20), dtype=bool)
    target[:, ::2] = True
    utility = np.ones((2, 20), dtype=float)
    balance = np.ones((2, 15, 20), dtype=float)
    valid = np.ones((2, 15, 20), dtype=bool)
    loss, audit = c5b0.relation_semantic_loss(
        samples, anchors, target, utility, balance, valid
    )
    loss.backward()
    assert audit["anchor_posterior_detached"] is True
    assert all(sample.grad is not None for sample in samples)
    assert all(anchor.grad is None for anchor in anchors)


def test_40_selection_and_expansion_action_are_frozen_read_only(
    expansion_inputs, sparse_split, full_directional_labels,
):
    plan = _plan("TRUE_U_EXPAND", expansion_inputs, sparse_split)
    for key in (
        "selected_direction", "admission_score", "pseudo_semantic_label",
        "expansion_mask", "selected_pseudo_anchor_ids",
        "selected_pseudo_anchor_labels", "combined_anchor_ids",
    ):
        assert plan[key].flags.writeable is False
    action = c5b0.build_training_action(
        np.ones((1386, 20)), sparse_split[2], full_directional_labels, plan
    )
    for key in ("U_cycle", "PredRelation", "balance_weight", "valid_relation_mask"):
        assert action[key].flags.writeable is False


def test_41_both_expansion_arms_train_with_original_TRUE_U_cycle():
    source = inspect.getsource(runner.run_arm)
    build_call = source.split("training_action =", 1)[1].split(")", 1)[0]
    assert 'full_action["U_cycle"]' in build_call
    assert "U_permuted" not in inspect.getsource(c5b0.build_training_action)


def test_42_expansion_fails_closed_without_BASE_parity(tmp_path):
    with pytest.raises(RuntimeError, match="BASE_REPLAY_REQUIRED"):
        runner.load_base_parity_gate(20, tmp_path / "missing_base")
