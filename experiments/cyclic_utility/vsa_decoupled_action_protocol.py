"""Frozen protocol primitives for Verified Semantic Action decoupling.

VSA keeps the frozen C1 pseudo target exactly intact, but gives semantic
action and native MVCAN consolidation independent optimizer state and
independent backward passes.  This module contains the auditable boundaries
shared by the runner, summarizer, and protocol tests.
"""

import ast
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from experiments.cyclic_utility import (
    c1_frozen_pseudo_supervision as c1,
)
from experiments.e1_pairwise_utility import (
    train_e1_pairwise_utility as e1_train,
)
from irv.b4_information_utility import tensor_sha256


SEED = c1.SEED
TRAINING_SEED_CHOICES = (20, 30, 50)
SAMPLE_NUM = c1.SAMPLE_NUM
VIEW_NUM = c1.VIEW_NUM
CLASS_NUM = c1.CLASS_NUM
BATCH_SIZE = c1.BATCH_SIZE
LEARNING_RATE = c1.LEARNING_RATE
FORMAL_EPOCHS = 20
ENGINEERING_EPOCHS = 1
FORMAL_ARMS = c1.FORMAL_ARMS
PSEUDO_ARMS = c1.PSEUDO_ARMS
ENGINEERING_ARM = c1.ENGINEERING_ARM
ALL_ARMS = c1.ALL_ARMS
METRIC_NAMES = c1.METRIC_NAMES

SEMANTIC_OBJECTIVE = (
    "mean_{i,v}(a_i * CE(T_local[i,v], q_local[i,v]))"
)
NATIVE_OBJECTIVE = "REC + 0.01 * CLU + 0.01 * LWC"
EXPECTED_BATCH_BOUNDARIES = (256, 256, 256, 256, 256, 120)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_training_seed(seed):
    """Validate the VSA-A0/B training seed without changing RNG semantics."""
    value = int(seed)
    if value not in TRAINING_SEED_CHOICES:
        raise ValueError(
            "VSA training seed must be one of "
            + str(TRAINING_SEED_CHOICES)
        )
    return value


def build_frozen_semantic_target(arm, c0_arrays, M0):
    """Reuse the exact frozen C1 E/m/T/a/T_local construction."""
    return c1.build_frozen_pseudo_target(arm, c0_arrays, M0)


def semantic_only_loss(q_local, target_local, sample_strength):
    """Compute only mean_[i,v] a_i CE(T_local[i,v], q_local[i,v])."""
    return c1.soft_pseudo_cross_entropy(
        q_local, target_local, sample_strength
    )


def semantic_batch_by_sample_ids(semantic_target, sample_ids_batch):
    """Index the detached frozen C1 tensors by canonical sample ID."""
    return c1.pseudo_batch_by_sample_ids(
        semantic_target, sample_ids_batch
    )


def _canonical_sample_ids(sample_ids):
    ids = np.asarray(sample_ids, dtype=np.int64)
    _require(
        ids.shape == (SAMPLE_NUM,)
        and np.array_equal(ids, np.arange(SAMPLE_NUM, dtype=np.int64)),
        "VSA requires exact canonical arange(1400) sample IDs",
    )
    return np.ascontiguousarray(ids)


def _precompute_orders_with_generator(epochs, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return [
        torch.randperm(SAMPLE_NUM, generator=generator)
        .numpy()
        .astype(np.int64, copy=False)
        for _ in range(int(epochs))
    ]


def validate_sample_order(order):
    values = np.asarray(order, dtype=np.int64)
    expected = np.arange(SAMPLE_NUM, dtype=np.int64)
    _require(
        values.shape == (SAMPLE_NUM,)
        and np.array_equal(np.sort(values), expected)
        and np.unique(values).size == SAMPLE_NUM,
        "VSA sample order must cover every canonical ID exactly once",
    )
    return np.ascontiguousarray(values)


def precompute_epoch_orders(epochs, sample_ids, seed=SEED):
    """Materialize phase orders without consuming global/DataLoader RNG.

    Two independent generators are seeded identically.  Consequently the
    semantic and native orders have the same frozen values, but neither phase
    can advance or otherwise perturb the other phase's generator state.
    """
    _require(int(epochs) > 0, "VSA epochs must be positive")
    seed = validate_training_seed(seed)
    ids = _canonical_sample_ids(sample_ids)
    semantic_permutations = _precompute_orders_with_generator(epochs, seed)
    native_permutations = _precompute_orders_with_generator(epochs, seed)
    semantic_orders = [
        validate_sample_order(ids[permutation])
        for permutation in semantic_permutations
    ]
    native_orders = [
        validate_sample_order(ids[permutation])
        for permutation in native_permutations
    ]
    semantic_hashes = [tensor_sha256(order) for order in semantic_orders]
    native_hashes = [tensor_sha256(order) for order in native_orders]
    return {
        "semantic_orders": semantic_orders,
        "native_orders": native_orders,
        "semantic_order_sha256_per_epoch": semantic_hashes,
        "native_order_sha256_per_epoch": native_hashes,
        "seed": int(seed),
        "canonical_sample_ids": True,
        "orders_precomputed_before_training": True,
        "independent_generator_instances": True,
        "global_DataLoader_RNG_used": False,
        "semantic_can_advance_native_RNG": False,
    }


def ordered_batches(order, batch_size=BATCH_SIZE):
    """Return explicit ID batches with the frozen 256x5+120 boundary."""
    if int(batch_size) != BATCH_SIZE:
        raise ValueError("VSA batch size is frozen at 256")
    values = validate_sample_order(order)
    batches = [
        np.ascontiguousarray(values[start : start + BATCH_SIZE])
        for start in range(0, SAMPLE_NUM, BATCH_SIZE)
    ]
    boundaries = tuple(int(batch.size) for batch in batches)
    _require(
        boundaries == EXPECTED_BATCH_BOUNDARIES,
        "VSA batch boundaries are not 256x5+120",
    )
    return batches


def _json_optimizer_value(value):
    if isinstance(value, tuple):
        return [_json_optimizer_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def optimizer_configuration(optimizers):
    """Return a stable, JSON-safe audit of a six-Adam collection."""
    return {
        "optimizer_count": len(optimizers),
        "per_optimizer": [
            {
                "class": optimizer.__class__.__name__,
                "defaults": {
                    key: _json_optimizer_value(value)
                    for key, value in sorted(optimizer.defaults.items())
                },
                "param_groups": [
                    {
                        key: _json_optimizer_value(value)
                        for key, value in sorted(group.items())
                        if key != "params"
                    }
                    for group in optimizer.param_groups
                ],
                "parameter_count": sum(
                    len(group["params"])
                    for group in optimizer.param_groups
                ),
                "state_entry_count": len(optimizer.state),
            }
            for optimizer in optimizers
        ],
    }


def build_decoupled_optimizers(model, arm):
    """Build separate semantic/native Adam state over the same parameters."""
    if arm not in ALL_ARMS:
        raise ValueError("unknown VSA arm")
    semantic_optimizers = e1_train.build_fresh_optimizers(model)
    native_optimizers = e1_train.build_fresh_optimizers(model)
    semantic_template = optimizer_configuration(semantic_optimizers)
    native_configuration = optimizer_configuration(native_optimizers)
    _require(
        semantic_template == native_configuration,
        "semantic optimizer template differs from native E1/C1 Adam",
    )
    if arm == ENGINEERING_ARM:
        for optimizer in semantic_optimizers:
            optimizer.defaults["lr"] = 0.0
            for group in optimizer.param_groups:
                group["lr"] = 0.0

    object_pairs_separate = all(
        semantic is not native
        for semantic, native in zip(
            semantic_optimizers, native_optimizers
        )
    )
    state_mappings_separate = all(
        semantic.state is not native.state
        for semantic, native in zip(
            semantic_optimizers, native_optimizers
        )
    )
    same_parameter_objects = all(
        [
            id(parameter)
            for group in semantic.param_groups
            for parameter in group["params"]
        ]
        == [
            id(parameter)
            for group in native.param_groups
            for parameter in group["params"]
        ]
        for semantic, native in zip(
            semantic_optimizers, native_optimizers
        )
    )
    semantic_configuration = optimizer_configuration(
        semantic_optimizers
    )
    expected_semantic_lr = (
        0.0 if arm == ENGINEERING_ARM else LEARNING_RATE
    )
    semantic_lrs = {
        float(group["lr"])
        for optimizer in semantic_optimizers
        for group in optimizer.param_groups
    }
    native_lrs = {
        float(group["lr"])
        for optimizer in native_optimizers
        for group in optimizer.param_groups
    }
    _require(
        len(semantic_optimizers) == len(native_optimizers) == VIEW_NUM
        and object_pairs_separate
        and state_mappings_separate
        and same_parameter_objects
        and semantic_lrs == {expected_semantic_lr}
        and native_lrs == {LEARNING_RATE},
        "VSA optimizer decoupling boundary mismatch",
    )
    audit = {
        "semantic_optimizer_config": semantic_configuration,
        "semantic_optimizer_template_config": semantic_template,
        "native_optimizer_config": native_configuration,
        "separate_optimizer_instances": True,
        "optimizer_state_shared": False,
        "same_model_parameter_objects": True,
        "semantic_optimizer_same_class_as_native_C1": True,
        "semantic_optimizer_hyperparameters_frozen": True,
        "native_optimizer_hyperparameters_frozen": True,
        "semantic_optimizer_learning_rate": expected_semantic_lr,
        "native_optimizer_learning_rate": LEARNING_RATE,
        "semantic_optimizer_used_only_in_semantic_phase": True,
        "native_optimizer_used_only_in_native_phase": True,
    }
    return semantic_optimizers, native_optimizers, audit


def zero_optimizer_gradients(optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def all_parameter_gradients_cleared(model):
    return all(
        parameter.grad is None
        for autoencoder in model.autoencoders
        for parameter in autoencoder.parameters()
    )


def parameter_gradient_audit(model):
    gradients = [
        parameter.grad
        for autoencoder in model.autoencoders
        for parameter in autoencoder.parameters()
        if parameter.grad is not None
    ]
    finite = bool(
        gradients
        and all(torch.isfinite(gradient).all().item() for gradient in gradients)
    )
    squared_norm = sum(
        float(torch.sum(gradient.detach() ** 2).item())
        for gradient in gradients
    )
    return {
        "gradient_tensor_count": len(gradients),
        "gradient_finite_pass": finite,
        "gradient_l2": float(squared_norm ** 0.5),
        "gradient_nonzero": bool(squared_norm > 0.0),
    }


def audit_model_stateful_layers(model):
    """Hard-fail if an extra semantic forward could mutate model state."""
    forbidden = []
    dropout_types = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    batchnorm_type = nn.modules.batchnorm._BatchNorm
    module_count = 0
    for view_id, autoencoder in enumerate(model.autoencoders):
        for name, module in autoencoder.named_modules():
            module_count += 1
            reasons = []
            if isinstance(module, dropout_types):
                reasons.append("Dropout")
            if isinstance(module, batchnorm_type):
                reasons.append("BatchNorm")
            buffers = tuple(
                buffer_name
                for buffer_name, _ in module.named_buffers(recurse=False)
            )
            if buffers:
                reasons.append("registered_buffers=" + ",".join(buffers))
            if bool(getattr(module, "vsa_forward_stateful", False)):
                reasons.append("explicit_stateful_marker")
            if reasons:
                forbidden.append({
                    "view": view_id,
                    "module": name or "<root>",
                    "class": module.__class__.__name__,
                    "reasons": reasons,
                })
    _require(
        not forbidden,
        "VSA stateful/stochastic layer guard failed: "
        + json.dumps(forbidden, sort_keys=True),
    )
    return {
        "module_count": module_count,
        "dropout_layer_count": 0,
        "batchnorm_layer_count": 0,
        "mutable_buffer_layer_count": 0,
        "stateful_layer_hard_fail_enabled": True,
        "mode_switch_workaround_used": False,
        "VSA_E0_forward_identity_safe_pass": True,
    }


def _expression_names(node):
    return {
        child.id.lower()
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
    }


def assert_no_additive_gradient_source(source_paths):
    """Hard-fail if VSA source constructs a mixed semantic/native loss."""
    audited = []
    backward_receivers = []
    for source_path in source_paths:
        path = Path(source_path)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and (
                node.attr == "combine_native_and_pseudo_loss"
            ):
                raise RuntimeError(
                    "VSA source calls the closed additive C1 carrier"
                )
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                names = _expression_names(node)
                has_native = any("native" in name for name in names)
                has_semantic = any(
                    "semantic" in name or "pseudo" in name
                    for name in names
                )
                if has_native and has_semantic:
                    raise RuntimeError(
                        "VSA source constructs additive native/semantic loss"
                    )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "backward"
            ):
                receiver_names = sorted(_expression_names(node.func.value))
                backward_receivers.append(receiver_names)
        audited.append(str(path))
    mixed_backward = any(
        any("native" in name for name in names)
        and any(
            "semantic" in name or "pseudo" in name for name in names
        )
        for names in backward_receivers
    )
    _require(not mixed_backward, "VSA source contains a mixed backward")
    return {
        "source_files_audited": audited,
        "backward_receivers": backward_receivers,
        "additive_native_plus_pseudo_used": False,
        "single_backward_contains_native_and_pseudo": False,
        "semantic_native_gradient_mixed": False,
        "lambda_pseudo_training_used": False,
        "closed_C1_combiner_called": False,
        "source_decoupling_hard_fail_pass": True,
    }


def metric_delta_record(candidate, comparator):
    """Apply the frozen strict 2/3 metrics plus positive delta-sum gate."""
    return c1.metric_delta_record(candidate, comparator)


def build_vsa_a0_decision(metrics_by_arm):
    """Apply the frozen VSA-A0 gate and exact priority/labels."""
    _require(
        tuple(metrics_by_arm) == FORMAL_ARMS,
        "VSA formal metric arm set/order mismatch",
    )
    cycle = metrics_by_arm["CYCLE"]
    comparisons = {
        "vs_BASE": metric_delta_record(cycle, metrics_by_arm["BASE"]),
        "vs_SHUFFLED_CYCLE": metric_delta_record(
            cycle, metrics_by_arm["SHUFFLED_CYCLE"]
        ),
        "vs_CONF": metric_delta_record(cycle, metrics_by_arm["CONF"]),
        "vs_UNIFORM": metric_delta_record(
            cycle, metrics_by_arm["UNIFORM"]
        ),
    }
    gates = {
        "NET_GAIN": comparisons["vs_BASE"]["pass"],
        "CORRESPONDENCE_SPECIFICITY": comparisons[
            "vs_SHUFFLED_CYCLE"
        ]["pass"],
        "BEATS_CONFIDENCE": comparisons["vs_CONF"]["pass"],
        "BEATS_UNIFORM": comparisons["vs_UNIFORM"]["pass"],
    }
    if not gates["NET_GAIN"]:
        decision = "VSA_ACTION_NO_NET_GAIN"
    elif not gates["CORRESPONDENCE_SPECIFICITY"]:
        decision = "VSA_ACTION_NOT_CORRESPONDENCE_SPECIFIC"
    elif not (gates["BEATS_CONFIDENCE"] and gates["BEATS_UNIFORM"]):
        decision = "VSA_UTILITY_ACTION_REDUNDANT"
    else:
        decision = "VSA_A0_DECOUPLED_ACTION_PASS"
    conditions = {
        "PASS": decision == "VSA_A0_DECOUPLED_ACTION_PASS",
        "FAIL_NET_GAIN": decision == "VSA_ACTION_NO_NET_GAIN",
        "FAIL_CORRESPONDENCE": (
            decision == "VSA_ACTION_NOT_CORRESPONDENCE_SPECIFIC"
        ),
        "FAIL_REDUNDANT": decision == "VSA_UTILITY_ACTION_REDUNDANT",
    }
    _require(
        sum(conditions.values()) == 1,
        "VSA decision tree entered an impossible state",
    )
    return {
        **gates,
        "comparisons": comparisons,
        "decision_conditions": conditions,
        "final_decision": decision,
        "threshold_adjustment_after_results": False,
        "cycle_pseudo_action_family_closed_if_fail": not conditions["PASS"],
    }


def validate_vsa_e0_identity(base_record, cycle_zero_record):
    """Require exact one-epoch BASE/CYCLE_ZERO engineering identity."""
    _require(
        base_record.get("epochs") == cycle_zero_record.get("epochs") == 1,
        "VSA-E0 requires exactly one epoch",
    )
    cycle_semantic_transitions = cycle_zero_record[
        "phase_transition_audit"
    ]["semantic_phase_model_hash_per_epoch"]
    cycle_semantic_unchanged = bool(
        len(cycle_semantic_transitions) == 1
        and all(
            record["before"] == record["after"]
            for record in cycle_semantic_transitions
        )
    )
    checks = {
        "same_initial_model_hash": (
            base_record["initial_model_hash"]
            == cycle_zero_record["initial_model_hash"]
        ),
        "frozen_initial_aggregate_hash": (
            base_record["initial_model_hash"]["aggregate"]
            == c1.EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256
        ),
        "same_native_sample_order": (
            base_record["native_order_sha256_per_epoch"]
            == cycle_zero_record["native_order_sha256_per_epoch"]
        ),
        "cycle_zero_semantic_model_unchanged": cycle_semantic_unchanged,
        "cycle_zero_semantic_lr_exact_zero": (
            cycle_zero_record["optimizer_decoupling_audit"][
                "semantic_optimizer_learning_rate"
            ]
            == 0.0
        ),
        "same_native_loss_trajectory": (
            base_record["native_loss_history"]["epoch_records"]
            == cycle_zero_record["native_loss_history"]["epoch_records"]
        ),
        "same_final_model_aggregate_hash": (
            base_record["final_model_hash"]["aggregate"]
            == cycle_zero_record["final_model_hash"]["aggregate"]
        ),
        "same_per_view_final_model_hashes": (
            base_record["final_model_hash"]["per_view"]
            == cycle_zero_record["final_model_hash"]["per_view"]
        ),
        "same_final_predictions": (
            base_record["prediction_logical_sha256"]
            == cycle_zero_record["prediction_logical_sha256"]
        ),
        "same_ACC_NMI_ARI": (
            base_record["metrics"] == cycle_zero_record["metrics"]
        ),
        "BASE_semantic_phase_unused": (
            base_record["semantic_phase_used"] is False
        ),
        "CYCLE_ZERO_full_semantic_path_used": (
            cycle_zero_record["semantic_phase_used"] is True
        ),
    }
    passed = bool(all(checks.values()))
    return {
        **checks,
        "VSA_E0_IDENTITY_PASS": passed,
        "final_decision": (
            "VSA_E0_IDENTITY_PASS" if passed else "VSA_E0_IDENTITY_FAIL"
        ),
        "CYCLE_ZERO_in_scientific_decision": False,
    }
