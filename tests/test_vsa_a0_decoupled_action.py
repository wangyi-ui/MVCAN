"""Protocol tests for Verified Semantic Action decoupling (no full run)."""

import ast
import copy
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.cyclic_utility import (
    c1_frozen_pseudo_supervision as c1,
)
from experiments.cyclic_utility import (
    summarize_vsa_a0_decoupled_action as summarize,
)
from experiments.cyclic_utility import (
    train_vsa_a0_decoupled_action as train,
)
from experiments.cyclic_utility import (
    vsa_decoupled_action_protocol as vsa,
)
from irv.b3_audit import hash_backbone


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def loaded_c0():
    return c1.load_frozen_c0_artifact(
        train.DEFAULT_C0_ARTIFACT_PATH,
        train.DEFAULT_C0_SEAL_PATH,
    )


@pytest.fixture(scope="session")
def frozen_model_and_M0(loaded_c0):
    c1.set_deterministic_seed(c1.SEED)
    views, sample_ids, _ = train.e1_train.load_frozen_feature_artifact(
        train.DEFAULT_FEATURE_PATH,
        train.DEFAULT_FEATURE_AUDIT_PATH,
    )
    model, _, checkpoint = train.c1_train.load_trainable_e1_lwc_model(
        train.DEFAULT_E1_LWC_MODEL_DIR,
        train.DEFAULT_E1_LWC_AUDIT_PATH,
        torch.device("cpu"),
    )
    M0, audit = c1.derive_frozen_M0(
        model,
        views,
        loaded_c0[0]["native_global_cluster"],
        torch.device("cpu"),
    )
    return model, M0, audit, checkpoint, sample_ids


@pytest.fixture(scope="session")
def synthetic_c0_arrays():
    rows = torch.arange(c1.SAMPLE_NUM)[:, None]
    directions = torch.arange(c1.DIRECTION_COUNT)[None, :]
    y_gen = ((rows + 2 * directions) % c1.CLASS_NUM).long()
    return {
        "y_gen": y_gen.detach(),
        "C_conf": torch.full((1400, 20), 0.75).detach(),
        "U_cycle": (
            ((rows + directions) % 3 != 0).float() * 0.5
        ).detach(),
        "U_cycle_shuffle": (
            ((rows + directions + 1) % 2 == 0).float() * 0.25
        ).detach(),
        "generator_subsets": torch.zeros(20, 3, dtype=torch.long),
        "verifier_subsets": torch.zeros(20, 3, dtype=torch.long),
    }


@pytest.fixture(scope="session")
def permutation_M0():
    return torch.stack(
        [
            torch.roll(torch.eye(7), shifts=view_id, dims=1)
            for view_id in range(6)
        ]
    ).detach()


@pytest.fixture(scope="session")
def cycle_targets(synthetic_c0_arrays, permutation_M0):
    old, old_audit = c1.build_frozen_pseudo_target(
        "CYCLE", synthetic_c0_arrays, permutation_M0
    )
    new, new_audit = vsa.build_frozen_semantic_target(
        "CYCLE", synthetic_c0_arrays, permutation_M0
    )
    return old, old_audit, new, new_audit


class TinyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value):
        return self.linear(value)


class TinyModel:
    def __init__(self, module_factory=TinyAutoencoder):
        self.autoencoders = [module_factory() for _ in range(vsa.VIEW_NUM)]


@pytest.fixture
def tiny_model():
    torch.manual_seed(123)
    return TinyModel()


def _called_names(callable_object):
    tree = ast.parse(inspect.getsource(callable_object))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def _metric_arms(base=0.1, uniform=0.2, conf=0.3, cycle=0.8, shuffled=0.4):
    def record(value):
        return {"ACC": value, "NMI": value, "ARI": value}

    return {
        "BASE": record(base),
        "UNIFORM": record(uniform),
        "CONF": record(conf),
        "CYCLE": record(cycle),
        "SHUFFLED_CYCLE": record(shuffled),
    }


def _identity_record(semantic_phase_used):
    model_hash = {
        "aggregate": c1.EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256,
        "per_view": ["same"] * vsa.VIEW_NUM,
    }
    return {
        "epochs": 1,
        "initial_model_hash": copy.deepcopy(model_hash),
        "final_model_hash": copy.deepcopy(model_hash),
        "native_order_sha256_per_epoch": ["native-order"],
        "optimizer_decoupling_audit": {
            "semantic_optimizer_learning_rate": (
                0.0 if semantic_phase_used else vsa.LEARNING_RATE
            )
        },
        "native_loss_history": {"epoch_records": [{"loss": 1.25}]},
        "prediction_logical_sha256": "prediction",
        "metrics": {"ACC": 0.7, "NMI": 0.6, "ARI": 0.5},
        "semantic_phase_used": semantic_phase_used,
        "phase_transition_audit": {
            "semantic_phase_model_hash_per_epoch": [
                {
                    "before": copy.deepcopy(model_hash),
                    "after": copy.deepcopy(model_hash),
                }
            ]
        },
    }


@pytest.mark.parametrize(
    "path,expected",
    (
        (train.DEFAULT_C0_ARTIFACT_PATH, c1.EXPECTED_C0_ARTIFACT_SHA256),
        (train.DEFAULT_C0_SEAL_PATH, c1.EXPECTED_C0_SEAL_SHA256),
    ),
)
def test_frozen_c0_artifact_and_seal_hashes(path, expected):
    assert c1.file_sha256(path) == expected


def test_frozen_c0_seal_boundary(loaded_c0):
    audit = loaded_c0[1]
    assert audit["loaded_pre_GT_arrays_only"]
    assert not audit["GT_derived_correctness_loaded"]
    assert audit["artifact_file_sha256_pass"]
    assert audit["seal_file_sha256_pass"]


@pytest.mark.parametrize(
    "name",
    ("y_gen", "C_conf", "U_cycle", "U_cycle_shuffle", "native_global_cluster"),
)
def test_frozen_c0_semantic_array_hashes(loaded_c0, name):
    assert c1.tensor_sha256(loaded_c0[0][name].numpy()) == (
        c1.EXPECTED_C0_ARRAY_SHA256[name]
    )


def test_frozen_M0_sealed_hash(frozen_model_and_M0):
    M0, audit = frozen_model_and_M0[1:3]
    assert audit["logical_sha256"] == c1.EXPECTED_M0_LOGICAL_SHA256
    assert c1.tensor_sha256(M0.long().numpy()) == (
        c1.EXPECTED_M0_LOGICAL_SHA256
    )


@pytest.mark.parametrize(
    "view_id,expected",
    tuple(enumerate(c1.EXPECTED_E1_CHECKPOINT_SHA256, start=1)),
)
def test_frozen_e1_checkpoint_hashes(view_id, expected):
    path = train.DEFAULT_E1_LWC_MODEL_DIR / (
        "Caltech-6V" + str(view_id) + "V.pth"
    )
    assert c1.file_sha256(path) == expected


def test_frozen_e1_initial_aggregate(frozen_model_and_M0):
    model = frozen_model_and_M0[0]
    assert hash_backbone(model.autoencoders)["aggregate"] == (
        c1.EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256
    )


@pytest.mark.parametrize(
    "relative_path,expected", tuple(train.FROZEN_C1_SOURCE_SHA256.items())
)
def test_frozen_c1_and_c1_p0_sources(relative_path, expected):
    assert c1.file_sha256(REPOSITORY_ROOT / relative_path) == expected


def test_exact_c1_target_builder_is_reused():
    source = inspect.getsource(vsa.build_frozen_semantic_target)
    assert "c1.build_frozen_pseudo_target" in source


@pytest.mark.parametrize(
    "field", ("evidence", "mass", "target_global", "sample_strength", "target_local")
)
def test_E_m_T_a_T_local_unchanged(cycle_targets, field):
    old, _, new, _ = cycle_targets
    assert torch.equal(old[field], new[field])


def test_semantic_only_loss_exact_formula():
    torch.manual_seed(7)
    q_local = torch.softmax(torch.randn(4, 6, 7), dim=-1)
    target = torch.softmax(torch.randn(4, 6, 7), dim=-1)
    strength = torch.tensor([0.2, 0.7, 1.3, 1.8])
    actual = vsa.semantic_only_loss(q_local, target, strength)
    expected = torch.mean(
        strength[:, None]
        * -torch.sum(target * torch.log(torch.clamp(q_local, min=1e-8)), dim=-1)
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("forbidden", ("mse_loss", "native_mvcan_losses", "build_lwc_loss"))
def test_semantic_phase_has_no_REC_CLU_or_LWC(forbidden):
    assert forbidden not in _called_names(train._semantic_phase)


def test_native_objective_string_is_exact():
    assert vsa.NATIVE_OBJECTIVE == "REC + 0.01 * CLU + 0.01 * LWC"


def test_native_phase_has_no_pseudo_or_semantic_objective():
    source = inspect.getsource(train._native_phase)
    assert "pseudo" not in source.lower()
    assert "semantic_only_loss" not in source
    assert "semantic_batch_by_sample_ids" not in source


def test_separate_semantic_and_native_optimizers(tiny_model):
    semantic, native, audit = vsa.build_decoupled_optimizers(tiny_model, "CYCLE")
    assert audit["separate_optimizer_instances"]
    assert all(left is not right for left, right in zip(semantic, native))


def test_optimizer_state_is_not_shared(tiny_model):
    semantic, native, audit = vsa.build_decoupled_optimizers(tiny_model, "CYCLE")
    assert not audit["optimizer_state_shared"]
    assert all(left.state is not right.state for left, right in zip(semantic, native))


def test_semantic_optimizer_same_class_as_native_C1(tiny_model):
    semantic, native, audit = vsa.build_decoupled_optimizers(tiny_model, "CYCLE")
    assert audit["semantic_optimizer_same_class_as_native_C1"]
    assert {item.__class__.__name__ for item in semantic + native} == {"Adam"}


def test_optimizer_hyperparameters_frozen(tiny_model):
    _, _, audit = vsa.build_decoupled_optimizers(tiny_model, "CYCLE")
    assert audit["semantic_optimizer_template_config"] == audit[
        "native_optimizer_config"
    ]
    assert audit["semantic_optimizer_learning_rate"] == vsa.LEARNING_RATE
    assert audit["native_optimizer_learning_rate"] == vsa.LEARNING_RATE


def test_optimizers_operate_on_same_parameter_objects(tiny_model):
    _, _, audit = vsa.build_decoupled_optimizers(tiny_model, "CYCLE")
    assert audit["same_model_parameter_objects"]


def test_no_additive_native_plus_pseudo_source():
    audit = vsa.assert_no_additive_gradient_source(
        (Path(train.__file__), Path(vsa.__file__))
    )
    assert not audit["additive_native_plus_pseudo_used"]
    assert not audit["closed_C1_combiner_called"]


def test_no_mixed_backward_source():
    audit = vsa.assert_no_additive_gradient_source((Path(train.__file__),))
    assert audit["backward_receivers"] == [["semantic_loss"], ["native_loss"]]
    assert not audit["single_backward_contains_native_and_pseudo"]
    assert not audit["semantic_native_gradient_mixed"]


def test_no_lambda_pseudo_training():
    audit = vsa.assert_no_additive_gradient_source((Path(train.__file__),))
    assert not audit["lambda_pseudo_training_used"]
    assert "effective_lambda_pseudo" not in inspect.getsource(train.train_vsa_arm)


def test_semantic_gradients_zeroed_correctly(tiny_model):
    semantic, _, _ = vsa.build_decoupled_optimizers(tiny_model, "CYCLE")
    sum(parameter.sum() for module in tiny_model.autoencoders for parameter in module.parameters()).backward()
    assert not vsa.all_parameter_gradients_cleared(tiny_model)
    vsa.zero_optimizer_gradients(semantic)
    assert vsa.all_parameter_gradients_cleared(tiny_model)


def test_native_gradients_zeroed_correctly(tiny_model):
    _, native, _ = vsa.build_decoupled_optimizers(tiny_model, "CYCLE")
    sum((parameter ** 2).sum() for module in tiny_model.autoencoders for parameter in module.parameters()).backward()
    vsa.zero_optimizer_gradients(native)
    assert vsa.all_parameter_gradients_cleared(tiny_model)


def test_no_cross_phase_gradient_accumulation_contract():
    source = inspect.getsource(train.train_vsa_arm)
    assert "no_cross_phase_gradient_accumulation" in source
    assert "zero_optimizer_gradients(semantic_optimizers)" in source
    assert "zero_optimizer_gradients(native_optimizers)" not in source
    assert "zero_optimizer_gradients(native_optimizers)" in inspect.getsource(
        train._native_phase
    )


def test_semantic_before_native_ordering():
    source = inspect.getsource(train.train_vsa_arm)
    assert source.index("_semantic_phase") < source.index("_native_phase")
    assert source.index('"phase": "semantic"') < source.index('"phase": "native"')


def test_native_uses_current_post_semantic_parameters():
    source = inspect.getsource(train.train_vsa_arm)
    assert "native_hash_before == semantic_hash_after" in source
    assert "current_post_semantic_params_used_by_native_phase" in source


def test_deterministic_semantic_orders():
    ids = np.arange(1400, dtype=np.int64)
    first = vsa.precompute_epoch_orders(3, ids)
    second = vsa.precompute_epoch_orders(3, ids)
    assert first["semantic_order_sha256_per_epoch"] == second[
        "semantic_order_sha256_per_epoch"
    ]


def test_deterministic_native_orders():
    ids = np.arange(1400, dtype=np.int64)
    first = vsa.precompute_epoch_orders(3, ids)
    second = vsa.precompute_epoch_orders(3, ids)
    assert first["native_order_sha256_per_epoch"] == second[
        "native_order_sha256_per_epoch"
    ]


def test_native_order_same_across_all_arms():
    ids = np.arange(1400, dtype=np.int64)
    hashes = {
        tuple(vsa.precompute_epoch_orders(2, ids)["native_order_sha256_per_epoch"])
        for _arm in vsa.ALL_ARMS
    }
    assert len(hashes) == 1


def test_semantic_order_same_across_pseudo_arms():
    ids = np.arange(1400, dtype=np.int64)
    hashes = {
        tuple(vsa.precompute_epoch_orders(2, ids)["semantic_order_sha256_per_epoch"])
        for _arm in vsa.PSEUDO_ARMS + (vsa.ENGINEERING_ARM,)
    }
    assert len(hashes) == 1


def test_semantic_and_native_generators_are_independent():
    audit = vsa.precompute_epoch_orders(2, np.arange(1400, dtype=np.int64))
    assert audit["independent_generator_instances"]
    assert not audit["semantic_can_advance_native_RNG"]
    assert not audit["global_DataLoader_RNG_used"]


def test_canonical_sample_ids_required():
    with pytest.raises(RuntimeError, match="canonical arange"):
        vsa.precompute_epoch_orders(1, np.arange(1400, dtype=np.int64)[::-1])


def test_sample_ID_target_indexing(cycle_targets):
    target = cycle_targets[2]
    ids = torch.tensor([1399, 0, 17], dtype=torch.long)
    local, strength = vsa.semantic_batch_by_sample_ids(target, ids)
    assert torch.equal(local, target["target_local"][ids])
    assert torch.equal(strength, target["sample_strength"][ids])


def test_batch_boundaries_are_256x5_plus_120():
    batches = vsa.ordered_batches(np.arange(1400, dtype=np.int64))
    assert tuple(len(batch) for batch in batches) == vsa.EXPECTED_BATCH_BOUNDARIES


def test_actual_frozen_model_has_no_stateful_layers(frozen_model_and_M0):
    audit = vsa.audit_model_stateful_layers(frozen_model_and_M0[0])
    assert audit["dropout_layer_count"] == 0
    assert audit["batchnorm_layer_count"] == 0
    assert audit["VSA_E0_forward_identity_safe_pass"]


def test_dropout_hard_fails():
    with pytest.raises(RuntimeError, match="stateful/stochastic"):
        vsa.audit_model_stateful_layers(TinyModel(lambda: nn.Dropout(0.5)))


def test_batchnorm_hard_fails():
    with pytest.raises(RuntimeError, match="stateful/stochastic"):
        vsa.audit_model_stateful_layers(TinyModel(lambda: nn.BatchNorm1d(3)))


def test_custom_running_buffer_hard_fails():
    class Buffered(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("running_value", torch.zeros(1))

    with pytest.raises(RuntimeError, match="registered_buffers"):
        vsa.audit_model_stateful_layers(TinyModel(Buffered))


def test_cycle_zero_exact_cycle_target(synthetic_c0_arrays, permutation_M0):
    cycle, _ = vsa.build_frozen_semantic_target(
        "CYCLE", synthetic_c0_arrays, permutation_M0
    )
    zero, _ = vsa.build_frozen_semantic_target(
        "CYCLE_ZERO", synthetic_c0_arrays, permutation_M0
    )
    for field in ("weights", "evidence", "mass", "target_global", "sample_strength", "target_local"):
        assert torch.equal(cycle[field], zero[field])


def test_cycle_zero_semantic_lr_is_exact_zero(tiny_model):
    semantic, native, audit = vsa.build_decoupled_optimizers(
        tiny_model, "CYCLE_ZERO"
    )
    assert audit["semantic_optimizer_learning_rate"] == 0.0
    assert {group["lr"] for optimizer in semantic for group in optimizer.param_groups} == {0.0}
    assert {group["lr"] for optimizer in native for group in optimizer.param_groups} == {1e-4}


def test_cycle_zero_semantic_step_leaves_model_hash_exact(tiny_model):
    semantic, _, _ = vsa.build_decoupled_optimizers(tiny_model, "CYCLE_ZERO")
    before = hash_backbone(tiny_model.autoencoders)
    loss = sum(
        (parameter ** 2).sum()
        for module in tiny_model.autoencoders
        for parameter in module.parameters()
    )
    loss.backward()
    assert vsa.parameter_gradient_audit(tiny_model)["gradient_nonzero"]
    for optimizer in semantic:
        optimizer.step()
    after = hash_backbone(tiny_model.autoencoders)
    assert before == after


def test_vsa_e0_identity_gate_passes_exact_records():
    result = vsa.validate_vsa_e0_identity(
        _identity_record(False), _identity_record(True)
    )
    assert result["VSA_E0_IDENTITY_PASS"]
    assert result["final_decision"] == "VSA_E0_IDENTITY_PASS"


@pytest.mark.parametrize(
    "mutation",
    (
        "native_order_sha256_per_epoch",
        "native_loss_history",
        "prediction_logical_sha256",
        "metrics",
    ),
)
def test_vsa_e0_identity_gate_fails_any_nonidentity(mutation):
    base = _identity_record(False)
    zero = _identity_record(True)
    if mutation == "native_loss_history":
        zero[mutation] = {"epoch_records": [{"loss": 9.0}]}
    elif mutation == "metrics":
        zero[mutation] = {"ACC": 0.0, "NMI": 0.0, "ARI": 0.0}
    else:
        zero[mutation] = "different"
    result = vsa.validate_vsa_e0_identity(base, zero)
    assert not result["VSA_E0_IDENTITY_PASS"]
    assert result["final_decision"] == "VSA_E0_IDENTITY_FAIL"


def test_base_has_no_semantic_action(synthetic_c0_arrays, permutation_M0):
    target, audit = vsa.build_frozen_semantic_target(
        "BASE", synthetic_c0_arrays, permutation_M0
    )
    assert target is None
    assert not audit["pseudo_supervision_used"]


@pytest.mark.parametrize(
    "arm,source",
    (
        ("UNIFORM", "exact_ones"),
        ("CONF", "C_conf"),
        ("CYCLE", "U_cycle"),
        ("SHUFFLED_CYCLE", "U_cycle_shuffle"),
    ),
)
def test_formal_arm_weight_contract(synthetic_c0_arrays, arm, source):
    weights, audit = c1.build_arm_weights(arm, synthetic_c0_arrays)
    assert audit["weight_source"] == source
    assert tuple(weights.shape) == (1400, 20)
    if arm == "UNIFORM":
        assert torch.equal(weights, torch.ones_like(weights))


def test_all_pseudo_arms_use_same_y_gen(synthetic_c0_arrays, permutation_M0):
    hashes = set()
    for arm in vsa.PSEUDO_ARMS:
        target, _ = vsa.build_frozen_semantic_target(
            arm, synthetic_c0_arrays, permutation_M0
        )
        hashes.add(c1.tensor_sha256(target["y_gen"].numpy()))
    assert len(hashes) == 1


def test_full_GT_loaded_only_after_prediction_seal():
    source = inspect.getsource(train.run_arm)
    assert source.index("save_predictions_before_GT") < source.index(
        "load_labels_after_predictions"
    )


@pytest.mark.parametrize(
    "forbidden_call",
    (
        "load_sparse_label_protocol",
        "load_frozen_reliability",
        "load_oracle_clean_labeled_weights",
    ),
)
def test_no_sparse_labels_R_or_oracle_loader(forbidden_call):
    assert forbidden_call not in _called_names(train.run_arm)


def test_feature_loader_excludes_GT_and_corruption_mask():
    source = inspect.getsource(train.run_arm)
    assert "load_frozen_feature_artifact" in source
    assert "load_frozen_reliability" not in source


@pytest.mark.parametrize(
    "forbidden",
    (
        "Memory",
        "P_corr",
        "P_util",
        "topk",
        "threshold",
        "PCGrad",
        "GradNorm",
        "gradient_normalize",
    ),
)
def test_forbidden_mechanisms_absent_from_training_runtime(forbidden):
    source = inspect.getsource(train.train_vsa_arm)
    assert forbidden not in source


def test_no_dynamic_cycle_refresh():
    source = inspect.getsource(train.run_arm)
    assert source.count("build_frozen_semantic_target") == 1
    assert source.index("build_frozen_semantic_target") < source.index("train_vsa_arm")


def test_final_prediction_sealed_with_canonical_ids():
    source = inspect.getsource(train.run_arm)
    assert "save_predictions_before_GT" in source
    payload = c1.prediction_seal_payload(
        np.arange(1400) % 7, np.arange(1400, dtype=np.int64)
    )
    assert payload["predictions"].shape == (1400,)


def test_model_parameter_hashes_in_train_audit():
    source = inspect.getsource(train.run_arm)
    assert '"initial_model_hash"' in source
    assert '"final_model_hash"' in source
    assert '"per_view"' not in source  # hash_backbone supplies per-view hashes.


def test_phase_hashes_in_phase_transition_audit():
    source = inspect.getsource(train.train_vsa_arm)
    assert "semantic_phase_model_hash_per_epoch" in source
    assert "native_phase_model_hash_per_epoch" in source


def test_required_per_arm_output_schema():
    assert summarize.REQUIRED_ARM_FILES == (
        "metrics.json",
        "train_audit.json",
        "semantic_target_audit.json",
        "semantic_loss_history.json",
        "native_loss_history.json",
        "phase_transition_audit.json",
        "final_predictions.npz",
    )


def test_required_summary_output_schema():
    assert summarize.SUMMARY_FILES == (
        "vsa_summary.json",
        "vsa_decision.json",
        "vsa_audit.json",
    )


def test_runner_refuses_overwrite_before_loading(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        train.run_arm("BASE", run_kind="E0", output_dir=output, device="cpu")


def test_summarizer_refuses_overwrite(tmp_path):
    root = tmp_path / "formal"
    root.mkdir()
    (root / "vsa_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        summarize.summarize(root)


def test_runner_creates_arm_directory_only_after_preflight():
    source = inspect.getsource(train.run_arm)
    mkdir_position = source.index("output_root.mkdir")
    assert source.index("verify_frozen_source_provenance") < mkdir_position
    assert source.index("precompute_epoch_orders") < mkdir_position


def test_five_formal_arms_are_exact():
    assert vsa.FORMAL_ARMS == (
        "BASE",
        "UNIFORM",
        "CONF",
        "CYCLE",
        "SHUFFLED_CYCLE",
    )


def test_cycle_zero_excluded_from_scientific_decision():
    assert vsa.ENGINEERING_ARM == "CYCLE_ZERO"
    assert vsa.ENGINEERING_ARM not in vsa.FORMAL_ARMS
    with pytest.raises(ValueError, match="formal five arms"):
        summarize.load_arm_bundle(Path("unused"), "CYCLE_ZERO")


@pytest.mark.parametrize(
    "candidate,comparator,expected",
    (
        (
            {"ACC": 2.0, "NMI": 2.0, "ARI": 0.0},
            {"ACC": 1.0, "NMI": 1.0, "ARI": 1.0},
            True,
        ),
        (
            {"ACC": 2.0, "NMI": 2.0, "ARI": -1.0},
            {"ACC": 1.0, "NMI": 1.0, "ARI": 1.0},
            False,
        ),
        (
            {"ACC": 2.0, "NMI": 1.0, "ARI": 1.0},
            {"ACC": 1.0, "NMI": 1.0, "ARI": 1.0},
            False,
        ),
    ),
)
def test_two_of_three_and_delta_sum_metric_gate(candidate, comparator, expected):
    assert vsa.metric_delta_record(candidate, comparator)["pass"] is expected


@pytest.mark.parametrize(
    "metrics,expected",
    (
        (_metric_arms(), "VSA_A0_DECOUPLED_ACTION_PASS"),
        (_metric_arms(base=0.9), "VSA_ACTION_NO_NET_GAIN"),
        (
            _metric_arms(shuffled=0.9),
            "VSA_ACTION_NOT_CORRESPONDENCE_SPECIFIC",
        ),
        (_metric_arms(conf=0.9), "VSA_UTILITY_ACTION_REDUNDANT"),
        (_metric_arms(uniform=0.9), "VSA_UTILITY_ACTION_REDUNDANT"),
    ),
)
def test_decision_priority_and_exact_labels(metrics, expected):
    decision = vsa.build_vsa_a0_decision(metrics)
    assert decision["final_decision"] == expected
    assert sum(decision["decision_conditions"].values()) == 1


def test_all_four_scientific_gate_names_are_exact():
    decision = vsa.build_vsa_a0_decision(_metric_arms())
    assert decision["NET_GAIN"]
    assert decision["CORRESPONDENCE_SPECIFICITY"]
    assert decision["BEATS_CONFIDENCE"]
    assert decision["BEATS_UNIFORM"]


def test_native_Match_is_separate_from_auxiliary_M0():
    source = inspect.getsource(train.train_vsa_arm)
    assert "refresh_native_target" in source
    tree = ast.parse(source)
    assert "M0" not in {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert "native_matches" in source


def test_native_P_all_not_rewritten():
    source = inspect.getsource(train.train_vsa_arm)
    assert "refresh_native_target" in source
    assert '"native_P_all_rewritten": False' in source


@pytest.mark.parametrize(
    "field",
    ("y_gen", "weights", "evidence", "mass", "target_global", "sample_strength", "target_local"),
)
def test_all_pseudo_tensors_are_detached(cycle_targets, field):
    tensor = cycle_targets[2][field]
    assert not tensor.requires_grad
    assert tensor.grad_fn is None


def test_M0_remains_frozen(cycle_targets):
    target = cycle_targets[2]
    assert not target["target_local"].requires_grad
    assert cycle_targets[3]["logical_sha256"]["M0"] == cycle_targets[1][
        "logical_sha256"
    ]["M0"]


def test_formal_and_engineering_epoch_contracts():
    train.validate_run_contract("E0", "BASE", 1)
    train.validate_run_contract("E0", "CYCLE_ZERO", 1)
    for arm in vsa.FORMAL_ARMS:
        train.validate_run_contract("A0", arm, 20)
    with pytest.raises(RuntimeError):
        train.validate_run_contract("A0", "CYCLE_ZERO", 20)
    with pytest.raises(RuntimeError):
        train.validate_run_contract("E0", "BASE", 20)


def test_cli_defaults_to_safe_engineering_mode():
    args = train.parse_args(["--arm", "BASE"])
    assert args.run_kind == "E0"
    assert args.epochs is None
    assert args.seed == 20


def test_formal_run_is_never_implicit():
    source = inspect.getsource(train.main)
    assert "args.run_kind" in source
    assert train.parse_args(["--arm", "CYCLE"]).run_kind == "E0"
    with pytest.raises(RuntimeError):
        train.validate_run_contract("E0", "CYCLE", 1)
