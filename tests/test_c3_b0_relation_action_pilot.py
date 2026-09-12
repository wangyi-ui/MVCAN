import inspect

import numpy as np
import pytest
import torch

from experiments.cyclic_utility import c3_b0_relation_action_protocol as c3b0
from experiments.cyclic_utility import train_c3_b0_relation_action_pilot as train
from experiments.cyclic_utility import summarize_c3_b0_relation_action_pilot as summarize


@pytest.fixture(scope="module")
def action_arrays():
    labeled = np.arange(14, dtype=np.int64)
    unlabeled = np.arange(14, 1400, dtype=np.int64)[::-1].copy()
    rows = np.arange(1386, dtype=np.float64)[:, None]
    cycle = np.broadcast_to((rows + 1.0) / 1387.0, (1386, 20)).copy()
    true_relation = np.zeros((1386, 14, 20), dtype=np.bool_)
    true_relation[:, ::2, :] = True
    shuffled_relation = np.logical_not(true_relation)
    true_balance = np.full((1386, 14, 20), 0.25, dtype=np.float64)
    shuffled_balance = np.full((1386, 14, 20), 0.75, dtype=np.float64)
    return {
        "U_cycle": cycle,
        "unlabeled_ids": unlabeled,
        "labeled_ids": labeled,
        "PredRelation_true": true_relation,
        "PredRelation_shuffle": shuffled_relation,
        "relation_balance_weights_true": true_balance,
        "relation_balance_weights_shuffle": shuffled_balance,
    }


@pytest.fixture
def posterior_inputs():
    torch.manual_seed(17)
    samples = [
        torch.softmax(torch.randn(3, 7, dtype=torch.float64), -1)
        .detach().requires_grad_(True)
        for _ in range(6)
    ]
    anchors = [
        torch.softmax(torch.randn(14, 7, dtype=torch.float64), -1)
        .detach().requires_grad_(True)
        for _ in range(6)
    ]
    target = np.zeros((3, 14, 20), dtype=np.bool_)
    target[:, ::2, :] = True
    cycle = np.linspace(0.1, 1.0, 60).reshape(3, 20)
    balance = np.linspace(0.2, 1.2, 3 * 14 * 20).reshape(3, 14, 20)
    return samples, anchors, target, cycle, balance


@pytest.fixture
def tiny_optimizer_model():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.autoencoders = torch.nn.ModuleList(
                [torch.nn.Linear(3, 2) for _ in range(c3b0.VIEW_NUM)]
            )

    return TinyModel()


def _optimizer_parameter_ids(optimizers):
    return [
        [id(parameter) for group in optimizer.param_groups
         for parameter in group["params"]]
        for optimizer in optimizers
    ]


def _called_names(function):
    source = inspect.getsource(function)
    tree = compile(source, "<source>", "exec", flags=0, dont_inherit=True)
    del tree
    return source


# A / parent boundary
def test_A_C3_A0_and_parent_freeze_hashes_pass():
    audit = train.verify_frozen_parent_hashes()
    assert audit["all_parent_hashes_pass"]
    assert audit["C3_A0_multiseed_frozen_hashes_pass"]
    assert audit["C2_B0_frozen_hashes_pass"]
    assert audit["C2_C0_frozen_hashes_pass"]
    assert audit["C2_C1_frozen_hashes_pass"]
    assert audit["C2_C2_A0_frozen_hashes_pass"]


# B / sealed action boundary
def test_B_C3_A0_action_bundle_logical_hashes_pass():
    arrays, audit = train.load_frozen_c3a0_action_bundle(20)
    assert audit["all_required_logical_hashes_pass"]
    assert audit["GT_loaded"] is False
    assert audit["scientific_metric_loaded"] is False
    assert set(audit["logical_sha256"]) == set(train.ACTION_FIELDS)
    assert arrays["U_cycle"].shape == (1386, 20)


# C / ID alignment
def test_C_sample_ID_alignment_not_batch_position(action_arrays):
    selected = c3b0.action_batch_by_sample_ids(
        action_arrays, "TRUE_U", np.array([1399, 14, 100])
    )
    assert selected["source_rows"].tolist() == [0, 1385, 1299]
    assert selected["U_cycle"][:, 0].tolist() == pytest.approx(
        [1 / 1387, 1386 / 1387, 1300 / 1387]
    )


# D
def test_D_only_unlabeled_queries_survive(action_arrays):
    batch = np.array([0, 14, 7, 1399, 13, 100], dtype=np.int64)
    result = c3b0.unlabeled_query_ids(batch, action_arrays["unlabeled_ids"])
    assert result.tolist() == [14, 1399, 100]


# E-H
def test_E_anchor_count_exactly_14(action_arrays):
    assert c3b0.validate_action_arrays(action_arrays)["labeled_ids"].shape == (14,)


def test_F_pred_relation_shape(action_arrays):
    valid = c3b0.validate_action_arrays(action_arrays)
    assert valid["PredRelation_true"].shape == (1386, 14, 20)
    assert valid["PredRelation_shuffle"].shape == (1386, 14, 20)


def test_G_U_shape_aligned(action_arrays):
    assert c3b0.validate_action_arrays(action_arrays)["U_cycle"].shape == (1386, 20)


def test_H_relation_balance_weight_shape(action_arrays):
    valid = c3b0.validate_action_arrays(action_arrays)
    assert valid["relation_balance_weights_true"].shape == (1386, 14, 20)
    assert valid["relation_balance_weights_shuffle"].shape == (1386, 14, 20)


# I-K / arms
def test_I_TRUE_U_action_weight_is_exact_product():
    cycle = torch.tensor([[0.2, 0.8]], dtype=torch.float64)
    balance = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]], dtype=torch.float64)
    assert torch.equal(
        c3b0.action_weight_for_arm(cycle, balance, "TRUE_U"),
        cycle[:, None, :] * balance,
    )


def test_J_TRUE_UNIFORM_action_weight_is_balance_only():
    cycle = torch.tensor([[0.2, 0.8]], dtype=torch.float64)
    balance = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]], dtype=torch.float64)
    assert torch.equal(
        c3b0.action_weight_for_arm(cycle, balance, "TRUE_UNIFORM"), balance
    )


def test_K_SHUFFLE_U_selects_independent_action_and_weights(action_arrays):
    ids = np.array([1399, 14])
    selected = c3b0.action_batch_by_sample_ids(action_arrays, "SHUFFLE_U", ids)
    rows = selected["source_rows"]
    assert np.array_equal(
        selected["PredRelation"], action_arrays["PredRelation_shuffle"][rows]
    )
    assert np.array_equal(
        selected["balance_weight"],
        action_arrays["relation_balance_weights_shuffle"][rows],
    )
    assert not np.array_equal(
        selected["PredRelation"], action_arrays["PredRelation_true"][rows]
    )


# L
def test_L_BASE_has_no_Phase_A_call():
    source = inspect.getsource(train.train_relation_action_arm)
    assert 'if arm == "BASE"' in source
    assert '"executed": False' in source


def test_BASE_does_not_call_frozen_decoupled_builder(
    monkeypatch, tiny_optimizer_model
):
    def forbidden(*args, **kwargs):
        raise AssertionError("BASE called build_decoupled_optimizers")

    monkeypatch.setattr(train.vsa, "build_decoupled_optimizers", forbidden)
    semantic, native, audit = train.build_arm_optimizers(
        tiny_optimizer_model, "BASE"
    )
    assert semantic is None
    assert len(native) == c3b0.VIEW_NUM
    assert audit["BASE_native_only_optimizer_path"]


def test_BASE_semantic_optimizer_count_is_zero(tiny_optimizer_model):
    semantic, _, audit = train.build_arm_optimizers(
        tiny_optimizer_model, "BASE"
    )
    assert semantic is None
    assert audit["semantic_optimizer_count"] == 0
    assert audit["semantic_optimizer_created"] is False


def test_BASE_native_count_matches_frozen_native_half(tiny_optimizer_model):
    _, base_native, audit = train.build_arm_optimizers(
        tiny_optimizer_model, "BASE"
    )
    _, frozen_native, _ = train.vsa.build_decoupled_optimizers(
        tiny_optimizer_model, "BASE"
    )
    assert len(base_native) == len(frozen_native) == c3b0.VIEW_NUM
    assert audit["native_optimizer_count"] == c3b0.VIEW_NUM
    assert audit["total_optimizer_count"] == c3b0.VIEW_NUM


def test_BASE_native_parameter_IDs_match_frozen_native_half(
    tiny_optimizer_model,
):
    _, base_native, _ = train.build_arm_optimizers(
        tiny_optimizer_model, "BASE"
    )
    _, frozen_native, _ = train.vsa.build_decoupled_optimizers(
        tiny_optimizer_model, "BASE"
    )
    assert _optimizer_parameter_ids(base_native) == _optimizer_parameter_ids(
        frozen_native
    )


def test_BASE_native_config_matches_frozen_native_half(tiny_optimizer_model):
    _, base_native, _ = train.build_arm_optimizers(
        tiny_optimizer_model, "BASE"
    )
    _, frozen_native, frozen_audit = train.vsa.build_decoupled_optimizers(
        tiny_optimizer_model, "BASE"
    )
    assert train.vsa.optimizer_configuration(base_native) == (
        train.vsa.optimizer_configuration(frozen_native)
    )
    assert train.vsa.optimizer_configuration(base_native) == frozen_audit[
        "native_optimizer_config"
    ]


@pytest.mark.parametrize("arm", ("TRUE_U", "TRUE_UNIFORM", "SHUFFLE_U"))
def test_non_BASE_arms_keep_six_semantic_and_six_native_optimizers(
    tiny_optimizer_model, arm
):
    semantic, native, audit = train.build_arm_optimizers(
        tiny_optimizer_model, arm
    )
    assert len(semantic) == len(native) == c3b0.VIEW_NUM
    assert audit["semantic_optimizer_count"] == 6
    assert audit["native_optimizer_count"] == 6
    assert audit["total_optimizer_count"] == 12


@pytest.mark.parametrize("arm", ("TRUE_U", "TRUE_UNIFORM", "SHUFFLE_U"))
def test_non_BASE_arms_still_call_frozen_decoupled_builder(
    monkeypatch, tiny_optimizer_model, arm
):
    original = train.vsa.build_decoupled_optimizers
    calls = []

    def recording_builder(model, frozen_arm):
        calls.append(frozen_arm)
        return original(model, frozen_arm)

    monkeypatch.setattr(
        train.vsa, "build_decoupled_optimizers", recording_builder
    )
    train.build_arm_optimizers(tiny_optimizer_model, arm)
    assert calls == ["CYCLE"]


def test_BASE_epoch_has_no_semantic_activity_and_only_native_phase(
    monkeypatch,
):
    calls = {"native": 0, "refresh": 0}

    def forbidden(*args, **kwargs):
        raise AssertionError("BASE entered semantic Phase A")

    def fake_native(*args, **kwargs):
        calls["native"] += 1
        return {"executed": True}

    def fake_refresh(model, views, weights, device, training_seed):
        calls["refresh"] += 1
        return (
            torch.zeros((1400, 7)),
            torch.zeros((6, 7, 7)),
            np.zeros(1400, dtype=np.int64),
            weights,
        )

    monkeypatch.setattr(train, "relation_semantic_phase", forbidden)
    monkeypatch.setattr(train.vsa, "zero_optimizer_gradients", forbidden)
    monkeypatch.setattr(train, "native_consolidation_phase", fake_native)
    monkeypatch.setattr(train.e1_train, "refresh_native_target", fake_refresh)
    views = [np.zeros((1400, 1), dtype=np.float32) for _ in range(6)]
    orders = {
        "semantic_orders": [None] * c3b0.FORMAL_EPOCHS,
        "native_orders": [None] * c3b0.FORMAL_EPOCHS,
    }
    _, runtime = train.train_relation_action_arm(
        arm="BASE",
        epochs=c3b0.FORMAL_EPOCHS,
        model=object(),
        semantic_optimizers=None,
        native_optimizers=[object()] * c3b0.VIEW_NUM,
        views=views,
        sample_ids=np.arange(1400, dtype=np.int64),
        action_arrays=None,
        orders=orders,
        device=torch.device("cpu"),
        training_seed=20,
    )
    assert calls["native"] == c3b0.FORMAL_EPOCHS
    assert all(
        not item["phase_A"]["executed"] for item in runtime["epoch_records"]
    )
    assert all(
        item["phase_B"]["executed"] for item in runtime["epoch_records"]
    )


# M-N
def test_M_q_sample_shape_is_B_by_7(posterior_inputs):
    assert posterior_inputs[0][0].shape == (3, 7)


def test_N_q_anchor_shape_is_14_by_7(posterior_inputs):
    assert posterior_inputs[1][0].shape == (14, 7)


# O
def test_O_anchor_side_is_detached_and_query_side_gets_gradient(posterior_inputs):
    samples, anchors, target, cycle, balance = posterior_inputs
    loss, audit = c3b0.relation_semantic_loss(
        samples, anchors, target, cycle, balance, "TRUE_U"
    )
    loss.backward()
    assert audit["anchor_posterior_detached"]
    assert all(anchor.grad is None for anchor in anchors)
    assert all(sample.grad is not None for sample in samples)


# P
def test_P_relation_probability_is_exact_dot_product():
    sample = torch.softmax(torch.randn(2, 7, dtype=torch.float64), -1)
    anchor = torch.softmax(torch.randn(14, 7, dtype=torch.float64), -1)
    assert torch.equal(c3b0.relation_probability(sample, anchor), sample @ anchor.T)


# Q
def test_Q_BCE_is_exact_formula_with_dtype_epsilon():
    probability = torch.tensor([[0.25] * 14], dtype=torch.float64)
    target = torch.zeros((1, 14, 20), dtype=torch.float64)
    target[:, ::2, :] = 1.0
    actual = c3b0.relation_bce(probability, target)
    expanded = probability[:, :, None]
    expected = -target * torch.log(expanded) - (1 - target) * torch.log(1 - expanded)
    assert torch.equal(actual, expected)


# R-S
def test_R_six_view_result_is_strict_arithmetic_mean(posterior_inputs):
    samples, anchors, target, cycle, balance = posterior_inputs
    actual, audit = c3b0.relation_semantic_loss(
        samples, anchors, target, cycle, balance, "TRUE_U"
    )
    expected = torch.tensor(audit["view_losses"], dtype=actual.dtype).mean()
    assert actual.detach().item() == pytest.approx(expected.item(), abs=1e-15)
    assert audit["six_view_arithmetic_mean"]


def test_S_denominator_normalization_is_exact(posterior_inputs):
    samples, anchors, target, cycle, balance = posterior_inputs
    actual, audit = c3b0.relation_semantic_loss(
        samples, anchors, target, cycle, balance, "TRUE_U"
    )
    weights = torch.as_tensor(cycle)[:, None, :] * torch.as_tensor(balance)
    probability = samples[0] @ anchors[0].detach().T
    expected_view0 = torch.sum(
        weights * c3b0.relation_bce(probability, target)
    ) / torch.sum(weights)
    assert audit["view_losses"][0] == pytest.approx(expected_view0.item(), abs=1e-15)
    assert audit["denominator"] == pytest.approx(weights.sum().item(), abs=1e-15)


# T
def test_T_zero_denominator_hard_fails(posterior_inputs):
    samples, anchors, target, cycle, balance = posterior_inputs
    with pytest.raises(RuntimeError, match="denominator"):
        c3b0.relation_semantic_loss(
            samples, anchors, target, np.zeros_like(cycle), balance, "TRUE_U"
        )


# U-W
@pytest.mark.parametrize("forbidden", ("native_mvcan_losses", "mse_loss"))
def test_U_Phase_A_does_not_compute_REC_or_CLU(forbidden):
    assert forbidden not in inspect.getsource(train.relation_semantic_phase)


def test_V_Phase_B_does_not_compute_relation_loss():
    source = inspect.getsource(train.native_consolidation_phase)
    assert "relation_semantic_loss(" not in source
    assert "PredRelation" not in source


def test_W_no_additive_native_plus_relation_backward():
    source = inspect.getsource(train.train_relation_action_arm)
    assert "native_loss +" not in source
    assert "+ relation_loss" not in source
    assert source.count("relation_semantic_phase(") == 1
    assert source.count("native_consolidation_phase(") == 1


# X-Z
@pytest.mark.parametrize(
    "field",
    ("U_cycle_detached", "PredRelation_detached", "balance_weights_detached"),
)
def test_X_Y_Z_frozen_action_tensors_have_no_grad(posterior_inputs, field):
    samples, anchors, target, cycle, balance = posterior_inputs
    _, audit = c3b0.relation_semantic_loss(
        samples, anchors, target, cycle, balance, "TRUE_U"
    )
    assert audit[field]


# AA
def test_AA_native_P_global_M_and_P_local_detach_boundaries():
    native_source = inspect.getsource(train.native_consolidation_phase)
    helper_source = inspect.getsource(train.e1_train.native_mvcan_losses)
    refresh_source = inspect.getsource(train.e1_train.refresh_native_target)
    assert "native_p_all[ids].to(device).detach()" in native_source
    assert "native_matches.detach()" in native_source
    assert "matches[view_id].detach()" in helper_source
    assert "@torch.no_grad()" in refresh_source


# AB
def test_AB_no_N_by_N_action_tensor():
    source = inspect.getsource(train.relation_semantic_phase)
    assert "[N,N]" not in source.replace(" ", "")
    assert "[c3b0.SAMPLE_NUM, c3b0.SAMPLE_NUM" not in source
    assert "[c3b0.BATCH_SIZE, 14, 20]" in source


# AC-AE
def test_AC_no_memory_bank():
    assert "memory_bank" not in inspect.getsource(train).lower()
    assert "memory bank" not in inspect.getsource(c3b0).lower()


@pytest.mark.parametrize("forbidden", ("torch.topk", "np.percentile", "quantile("))
def test_AD_no_threshold_or_top_k_mechanism(forbidden):
    source = inspect.getsource(train) + inspect.getsource(c3b0)
    assert forbidden not in source


@pytest.mark.parametrize("forbidden", ("U_tilde", "utility_new", "reliability", "confidence"))
def test_AE_no_new_utility_or_selector(forbidden):
    source = inspect.getsource(train) + inspect.getsource(c3b0)
    assert forbidden not in source


# AF
def test_AF_exactly_four_fixed_arms():
    assert c3b0.ARMS == ("BASE", "TRUE_U", "TRUE_UNIFORM", "SHUFFLE_U")
    assert tuple(c3b0.ARM_DIRECTORY_NAMES) == c3b0.ARMS


def _passing_metrics():
    result = {}
    for seed, offset in zip(c3b0.SEEDS, (0.0, 0.01, -0.01)):
        result[seed] = {
            "BASE": {"ACC": 0.60 + offset, "NMI": 0.50, "ARI": 0.40},
            "TRUE_U": {"ACC": 0.90 + offset, "NMI": 0.70, "ARI": 0.60},
            "TRUE_UNIFORM": {"ACC": 0.70 + offset, "NMI": 0.60, "ARI": 0.50},
            "SHUFFLE_U": {"ACC": 0.80 + offset, "NMI": 0.65, "ARI": 0.55},
        }
    return result


# AG
def test_AG_ACC_NMI_ARI_comparisons_are_exact():
    result = c3b0.summarize_multiseed_metrics(_passing_metrics())
    assert result["comparisons"]["BASE"]["mean_delta"] == pytest.approx(
        {"ACC": 0.3, "NMI": 0.2, "ARI": 0.2}
    )
    assert result["comparisons"]["TRUE_UNIFORM"]["mean_delta"]["ACC"] \
        == pytest.approx(0.2)
    assert result["comparisons"]["SHUFFLE_U"]["mean_delta"]["ACC"] \
        == pytest.approx(0.1)


# AH
def test_AH_multiseed_gate_requires_all_eight_conditions():
    passed = c3b0.summarize_multiseed_metrics(_passing_metrics())
    assert len(passed["gate_checks"]) == 8
    assert all(passed["gate_checks"].values())
    assert passed["C3_B0_RELATION_ACTION_PILOT_PASS"]
    failing = _passing_metrics()
    for seed in c3b0.SEEDS:
        failing[seed]["TRUE_U"]["ARI"] = 0.30
    failed = c3b0.summarize_multiseed_metrics(failing)
    assert failed["C3_B0_RELATION_ACTION_PILOT_PASS"] is False


# AI
def test_AI_relation_FAIL_branches_to_class_pilot():
    metrics = _passing_metrics()
    for seed in c3b0.SEEDS:
        metrics[seed]["TRUE_U"]["ACC"] = 0.1
    result = c3b0.summarize_multiseed_metrics(metrics)
    assert result["decision"] == c3b0.FAIL_DECISION
    assert "C3-B1" in result["next_stage_if_fail"]


# AJ
def test_AJ_relation_PASS_only_makes_memory_eligible():
    result = c3b0.summarize_multiseed_metrics(_passing_metrics())
    assert result["memory_eligible"] is True
    assert result["memory_automatically_enabled"] is False


# AK-AL
def test_AK_no_GT_argument_in_training_core():
    signatures = (
        inspect.signature(train.relation_semantic_phase),
        inspect.signature(train.native_consolidation_phase),
        inspect.signature(train.train_relation_action_arm),
    )
    assert all("gt" not in str(signature).lower() for signature in signatures)


def test_AL_GT_is_loaded_only_after_final_prediction_seal():
    source = inspect.getsource(train.run_arm)
    seal_at = source.index("save_predictions_before_GT")
    labels_at = source.index("load_labels_after_predictions")
    metrics_at = source.index("evaluate_predictions")
    assert seal_at < labels_at < metrics_at
    assert '"GT_use": "final ACC/NMI/ARI evaluation only"' in source


def test_native_objective_is_exact_REC_plus_point01_CLU():
    assert c3b0.NATIVE_OBJECTIVE == "REC + 0.01 * CLU"
    source = inspect.getsource(train.native_consolidation_phase)
    assert "native_mvcan_losses(" in source
    assert "build_lwc_loss" not in source


def test_VSA_frozen_phase_order_and_optimizer_separation_are_reused():
    source = inspect.getsource(train.run_arm)
    optimizer_source = inspect.getsource(train.build_arm_optimizers)
    core = inspect.getsource(train.train_relation_action_arm)
    assert "build_arm_optimizers(" in source
    assert "vsa.build_decoupled_optimizers(" in optimizer_source
    assert "vsa.precompute_epoch_orders(" in source
    assert core.index("relation_semantic_phase(") < core.index(
        "native_consolidation_phase("
    )


def test_required_output_file_schema_is_implemented():
    source = inspect.getsource(train.run_arm)
    for filename in (
        "config.json", "audit.json", "metrics.json", "training_summary.json"
    ):
        assert filename in source


def test_summarizer_reads_only_C3_B0_records_not_GT():
    source = inspect.getsource(summarize.load_formal_records)
    assert "load_labels" not in source
    assert "final_predictions" not in source
    assert set(c3b0.METRICS) == {"ACC", "NMI", "ARI"}
