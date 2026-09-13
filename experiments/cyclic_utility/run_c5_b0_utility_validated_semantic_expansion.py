"""Run one C5-B0 arm and seal predictions before any full-GT access.

BASE exactly replays frozen C3-B0 TRUE_U. Expansion arms add one fixed set of
pseudo-semantic anchors while retaining the same optimizer, schedule, native
loss, TRUE_U relation action, and sample-ID order.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import c3_b0_relation_action_protocol as c3b0
from experiments.cyclic_utility import c3_b0_true_u_carrier_protocol as carrier
from experiments.cyclic_utility import c5_b0_utility_validated_semantic_expansion_protocol as c5b0
from experiments.cyclic_utility import evaluate_c5_a0_utility_validated_pseudo_semantic as c5a0_evaluator
from experiments.cyclic_utility import materialize_c3_b0_true_u_carrier as materializer
from experiments.cyclic_utility import run_c5_a0_utility_validated_pseudo_semantic as c5a0_runner
from experiments.cyclic_utility import train_c3_b0_relation_action_pilot as c3_train
from experiments.cyclic_utility import train_vsa_a0_decoupled_action as vsa_train
from experiments.cyclic_utility import vsa_decoupled_action_protocol as vsa
from experiments.e1_pairwise_utility import train_e1_pairwise_utility as e1_train
from irv.b3_audit import hash_backbone
from weak_quality import ndarray_sha256


OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/cyclic_utility"
PREREGISTRATION = REPOSITORY_ROOT / (
    "experiment_freeze/"
    "c5_b0_utility_validated_semantic_expansion_preregistered_20260913/"
    "PROTOCOL.txt"
)
C5A_FREEZE = REPOSITORY_ROOT / (
    "experiment_freeze/c5_a0_v2_multiseed_action_validity_positive_20260913"
)
C5A_PATHS = {
    20: "c5_a0_engineering_smoke_v2_seed20",
    30: "c5_a0_locked_validation_v2_seed30",
    50: "c5_a0_locked_validation_v2_seed50",
}
ARM_DIRECTORY_NAMES = OrderedDict((
    ("BASE", "base"),
    ("TRUE_U_EXPAND", "true_u_expand"),
    ("PERMUTED_U_EXPAND", "permuted_u_expand"),
))
ARTIFACT_NAME = "c5_b0_pre_gt_training.npz"
AUDIT_NAME = "c5_b0_pre_gt_audit.json"
SEAL_NAME = "c5_b0_pre_gt_seal.json"


def _require(condition, message="C5_B0_RUNNER_FAIL_CLOSED"):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _display(path):
    value = _resolve(path).resolve()
    try:
        return str(value.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(value)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, record):
    with open(path, "x", encoding="utf-8") as output_file:
        json.dump(record, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def _array_records(arrays):
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "logical_sha256": c5b0.logical_sha256(value),
        }
        for key, value in arrays.items()
    }


def default_output_dir(seed, arm):
    return OUTPUT_ROOT / (
        "c5_b0_utility_validated_semantic_expansion_seed"
        + str(c5b0.validate_seed(seed)) + "_"
        + ARM_DIRECTORY_NAMES[c5b0.validate_arm(arm)]
    )


def c5a_pre_gt_paths(seed):
    active_seed = c5b0.validate_seed(seed)
    root = OUTPUT_ROOT / C5A_PATHS[active_seed]
    return (
        root / "c5_pre_gt_bundle.npz",
        root / "c5_pre_gt_audit.json",
        root / "c5_pre_gt_seal.json",
    )


def _manifest_entries(path):
    entries = {}
    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if stripped:
                digest, relative = stripped.split(None, 1)
                entries[relative.strip()] = digest
    _require(entries, "C5_B0_EMPTY_PARENT_MANIFEST_FAIL_CLOSED")
    return entries


def _verify_manifest(path):
    return c3_train.verify_sha256_manifest(path, REPOSITORY_ROOT)


def verify_preregistration():
    _require(
        PREREGISTRATION.is_file()
        and carrier.file_sha256(PREREGISTRATION)
        == c5b0.PREREGISTERED_PROTOCOL_SHA256,
        "C5_B0_PREREGISTRATION_HASH_FAIL_CLOSED",
    )
    return {
        "path": _display(PREREGISTRATION),
        "file_sha256": carrier.file_sha256(PREREGISTRATION),
        "verified_before_training": True,
    }


def verify_frozen_c5a_inputs(seed):
    """Verify only C5-A0 sources/protocols and the three allowed pre-GT files."""
    active_seed = c5b0.validate_seed(seed)
    source = _verify_manifest(C5A_FREEZE / "c5_source_sha256.txt")
    protocols = _verify_manifest(C5A_FREEZE / "protocol_sha256.txt")
    artifact_manifest = _manifest_entries(
        C5A_FREEZE / "multiseed_artifacts_sha256.txt"
    )
    paths = c5a_pre_gt_paths(active_seed)
    verified = {}
    for path in paths:
        relative = _display(path)
        _require(
            relative in artifact_manifest
            and path.is_file()
            and carrier.file_sha256(path) == artifact_manifest[relative],
            "C5_B0_C5_A0_PARENT_HASH_FAIL_CLOSED",
        )
        verified[relative] = artifact_manifest[relative]
    return {
        "seed": active_seed,
        "source_manifest": source,
        "protocol_manifest": protocols,
        "allowed_pre_gt_artifacts": verified,
        "post_GT_metric_loaded": False,
        "all_C5_A0_parent_hashes_pass": True,
    }


def load_frozen_c5a_parent(seed):
    active_seed = c5b0.validate_seed(seed)
    frozen = verify_frozen_c5a_inputs(active_seed)
    artifact, audit, seal = c5a_pre_gt_paths(active_seed)
    arrays, provenance = c5a0_evaluator.validate_pre_gt_seal(
        active_seed, artifact, audit, seal
    )
    _require(
        arrays["eligible_ids"].ndim == 1
        and arrays["U_true_eligible"].shape
        == arrays["U_permuted"].shape
        == (arrays["eligible_ids"].size, c5b0.S)
        and arrays["directional_pred"].shape == (c5b0.NU, c5b0.S)
        and np.array_equal(
            arrays["eligible_ids"],
            arrays["unlabeled_ids"][arrays["eligible_mask"]],
        ),
        "C5_B0_C5_A0_PARENT_SCHEMA_FAIL_CLOSED",
    )
    return arrays, {**frozen, **provenance, "GT_loaded": False, "C4_used": False}


def load_full_c3a0_action(seed):
    """Load sealed C3-A0 semantic arrays, including exact class_pred_true."""
    active_seed = c5b0.validate_seed(seed)
    base_arrays, base_provenance = c3_train.load_frozen_c3a0_action_bundle(
        active_seed
    )
    npz_path, seal_path = c3_train.default_action_paths(active_seed)
    seal = _read_json(seal_path)
    required = (
        "sample_ids", "labeled_targets", "class_pred_true",
        "PredRelation_true", "relation_balance_weights_true",
    )
    with np.load(npz_path, allow_pickle=False) as archive:
        _require(
            all(key in archive.files for key in required),
            "C5_B0_C3_A0_SCHEMA_FAIL_CLOSED",
        )
        extra = {
            key: np.array(archive[key], copy=True, order="C")
            for key in required
        }
    for key, value in extra.items():
        record = seal.get("arrays", {}).get(key, {})
        _require(
            list(value.shape) == record.get("shape")
            and str(value.dtype) == record.get("dtype")
            and ndarray_sha256(value) == record.get("logical_sha256"),
            "C5_B0_C3_A0_LOGICAL_HASH_FAIL_CLOSED",
        )
        value.setflags(write=False)
    equivalence = c5b0.relation_equivalence_gate(
        extra["class_pred_true"], extra["labeled_targets"],
        extra["PredRelation_true"], extra["relation_balance_weights_true"],
    )
    merged = dict(base_arrays)
    merged.update(extra)
    return merged, {
        **base_provenance,
        "equivalence_gate": equivalence,
        "GT_loaded": False,
    }


def build_all_anchor_plans(c5a_arrays, action_arrays):
    eligible_ids = c5a_arrays["eligible_ids"]
    eligible_directional_pred = c5a_arrays["directional_pred"][
        c5a_arrays["eligible_mask"]
    ]
    common = {
        "eligible_ids": eligible_ids,
        "directional_pred": eligible_directional_pred,
        "real_anchor_ids": action_arrays["labeled_ids"],
        "real_anchor_labels": action_arrays["labeled_targets"],
    }
    plans = OrderedDict((
        ("BASE", c5b0.build_anchor_plan(
            "BASE", action_utility=None, **common
        )),
        ("TRUE_U_EXPAND", c5b0.build_anchor_plan(
            "TRUE_U_EXPAND",
            action_utility=c5a_arrays["U_true_eligible"], **common
        )),
        ("PERMUTED_U_EXPAND", c5b0.build_anchor_plan(
            "PERMUTED_U_EXPAND",
            action_utility=c5a_arrays["U_permuted"], **common
        )),
    ))
    _require(
        all(np.array_equal(plan["eligible_ids"], eligible_ids)
            for plan in plans.values()),
        "C5_B0_COMMON_ELIGIBLE_SET_FAIL_CLOSED",
    )
    return plans


def validate_base_parity_record(audit):
    _require(
        audit.get("stage") == c5b0.STAGE
        and audit.get("arm") == "BASE"
        and audit.get("BASE_replays_frozen_C3_B0_TRUE_U") is True
        and audit.get("final_model_hash_equal") is True
        and audit.get("final_predictions_equal") is True
        and audit.get("final_prediction_sample_ids_equal") is True
        and audit.get("coordinate_mapping_pass") is True
        and audit.get("all_parent_hashes_pass") is True
        and audit.get("GT_loaded_before_pre_gt_seal") is False,
        "C5_B0_BASE_REPLAY_PARITY_FAIL_CLOSED",
    )
    return True


def load_base_parity_gate(seed, output_dir=None):
    active_seed = c5b0.validate_seed(seed)
    root = default_output_dir(active_seed, "BASE") if output_dir is None else _resolve(output_dir)
    artifact = root / ARTIFACT_NAME
    audit_path = root / AUDIT_NAME
    seal_path = root / SEAL_NAME
    _require(
        artifact.is_file() and audit_path.is_file() and seal_path.is_file(),
        "C5_B0_BASE_REPLAY_REQUIRED_FAIL_CLOSED",
    )
    audit = _read_json(audit_path)
    seal = _read_json(seal_path)
    validate_base_parity_record(audit)
    _require(
        int(audit.get("seed", -1)) == active_seed
        and seal.get("pre_gt_seal_valid") is True
        and int(seal.get("seed", -1)) == active_seed
        and seal.get("arm") == "BASE"
        and seal.get("BASE_replay_parity_pass") is True
        and carrier.file_sha256(artifact) == seal.get("artifact_file_sha256")
        and carrier.file_sha256(audit_path) == seal.get("audit_file_sha256"),
        "C5_B0_BASE_REPLAY_SEAL_FAIL_CLOSED",
    )
    return {
        "artifact_path": _display(artifact),
        "artifact_file_sha256": carrier.file_sha256(artifact),
        "audit_path": _display(audit_path),
        "audit_file_sha256": carrier.file_sha256(audit_path),
        "seal_path": _display(seal_path),
        "seal_file_sha256": carrier.file_sha256(seal_path),
        "BASE_replay_parity_pass": True,
    }


def _tensor_ids(values):
    return torch.as_tensor(np.asarray(values, dtype=np.int64), dtype=torch.long)


def expanded_relation_semantic_phase(
    model, semantic_optimizers, full_views, training_action, order, device,
):
    """Phase A with q_sample[B,7], q_anchor[14+B,7], action[B,14+B,20]."""
    anchor_ids = _tensor_ids(training_action["anchor_ids"])
    visited = []
    records = []
    for batch_number, batch_sample_ids in enumerate(vsa.ordered_batches(order), 1):
        query_ids = c3b0.unlabeled_query_ids(
            batch_sample_ids, training_action["unlabeled_ids"]
        )
        _require(query_ids.size > 0, "C5_B0_EMPTY_PHASE_A_BATCH")
        visited.append(query_ids)
        selected = c5b0.action_batch_by_sample_ids(training_action, query_ids)
        query_tensor_ids = _tensor_ids(query_ids)
        vsa.zero_optimizer_gradients(semantic_optimizers)
        _require(
            vsa.all_parameter_gradients_cleared(model),
            "C5_B0_PHASE_A_GRADIENT_CLEAR_FAIL_CLOSED",
        )
        q_sample_views = []
        q_anchor_views = []
        for view_id in range(c5b0.V):
            query_input = full_views[view_id][query_tensor_ids].to(device)
            query_latent = model.autoencoders[view_id].encoder(query_input)
            q_sample_views.append(
                model.autoencoders[view_id].clustering(query_latent)
            )
            with torch.no_grad():
                anchor_input = full_views[view_id][anchor_ids].to(device)
                anchor_latent = model.autoencoders[view_id].encoder(anchor_input)
                anchor_q = model.autoencoders[view_id].clustering(anchor_latent)
            q_anchor_views.append(anchor_q.detach())
        loss, loss_audit = c5b0.relation_semantic_loss(
            q_sample_views, q_anchor_views,
            selected["PredRelation"], selected["U_cycle"],
            selected["balance_weight"], selected["valid_relation_mask"],
        )
        loss.backward()
        gradient = vsa.parameter_gradient_audit(model)
        _require(gradient["gradient_finite_pass"], "C5_B0_NONFINITE_GRADIENT")
        for optimizer in semantic_optimizers:
            optimizer.step()
        _require(loss_audit["self_relation_count_used"] == 0, "C5_B0_SELF_RELATION_USED")
        records.append({
            "batch": batch_number,
            "query_count": int(query_ids.size),
            "relation_loss": float(loss.detach().item()),
            "query_sample_ids_logical_sha256": ndarray_sha256(query_ids),
            "gradient": gradient,
            **loss_audit,
        })
    vsa.zero_optimizer_gradients(semantic_optimizers)
    _require(vsa.all_parameter_gradients_cleared(model), "C5_B0_PHASE_A_GRADIENT_LEAK")
    coverage = c3_train._coverage_record(visited, training_action["unlabeled_ids"])
    _require(coverage["all_expected_sample_ids_exactly_once_pass"], "C5_B0_COVERAGE_FAIL")
    return {
        "phase": "relation_semantic",
        "executed": True,
        "query_role": "unlabeled_only",
        "anchor_count": int(training_action["anchor_ids"].size),
        "real_anchor_count": c5b0.L,
        "pseudo_anchor_count": int(training_action["anchor_ids"].size - c5b0.L),
        "relation_target_axis_order": list(c5b0.RELATION_TARGET_AXIS_ORDER),
        "max_action_tensor_shape": [
            c5b0.BATCH_SIZE, int(training_action["anchor_ids"].size), c5b0.S
        ],
        "anchor_posterior_detached": True,
        "self_relation_count_used": int(sum(
            record["self_relation_count_used"] for record in records
        )),
        "batch_records": records,
        **coverage,
    }


def train_expansion_arm(
    model, semantic_optimizers, native_optimizers, views, sample_ids,
    training_action, orders, device, training_seed,
):
    """Exact C3-B0 20-epoch A->B schedule; only Phase-A anchor axis expands."""
    ids = carrier.validate_sample_ids(sample_ids)
    dataset = torch.utils.data.TensorDataset(
        *[torch.from_numpy(view) for view in views], torch.from_numpy(ids)
    )
    full_views = list(dataset.tensors[:c5b0.V])
    view_weights = [1.0] * c5b0.V
    native_p_all = None
    native_matches = None
    refresh_count = 0
    epoch_records = []
    for epoch in range(c5b0.FORMAL_EPOCHS):
        semantic_record = expanded_relation_semantic_phase(
            model, semantic_optimizers, full_views, training_action,
            orders["semantic_orders"][epoch], device,
        )
        if epoch % e1_train.TARGET_REFRESH_INTERVAL == 0:
            native_p_all, native_matches, _, view_weights = e1_train.refresh_native_target(
                model, full_views, view_weights, device,
                training_seed=training_seed,
            )
            refresh_count += 1
        native_record = c3_train.native_consolidation_phase(
            model, native_optimizers, full_views, native_p_all,
            native_matches, orders["native_orders"][epoch], device,
        )
        epoch_records.append({
            "epoch": epoch + 1,
            "phase_A": semantic_record,
            "phase_B": native_record,
        })
    _, final_matches, predictions, _ = e1_train.refresh_native_target(
        model, full_views, view_weights, device, training_seed=training_seed
    )
    _require(
        not final_matches.requires_grad and final_matches.grad_fn is None,
        "C5_B0_FINAL_MAPPING_DETACH_FAIL_CLOSED",
    )
    return np.ascontiguousarray(predictions, dtype=np.int64), {
        "epochs": c5b0.FORMAL_EPOCHS,
        "phase_A_before_phase_B_every_epoch": True,
        "separate_backward_passes": True,
        "additive_combined_loss_used": False,
        "native_target_refresh_count_inside_epochs": refresh_count,
        "final_native_prediction_refresh_count": 1,
        "native_refresh_interval": e1_train.TARGET_REFRESH_INTERVAL,
        "self_relation_count_used": int(sum(
            record["phase_A"]["self_relation_count_used"]
            for record in epoch_records
        )),
        "epoch_records": epoch_records,
    }


def prepare_training_context(seed, device):
    active_seed = c5b0.validate_seed(seed)
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        _require(torch.cuda.is_available(), "C5_B0_REQUESTED_CUDA_UNAVAILABLE")
        torch.cuda.set_device(torch_device)
    determinism = vsa_train.enable_strict_determinism(active_seed)
    parent_hashes, c3_source, c3_formal, c3_summary = (
        materializer.verify_frozen_c3_b0_lineage()
    )
    reference = materializer.load_frozen_true_u_reference(active_seed)
    action_arrays, action_provenance = c3_train.load_frozen_c3a0_action_bundle(
        active_seed
    )
    views, sample_ids, feature_provenance = e1_train.load_frozen_feature_artifact(
        c3_train.DEFAULT_FEATURE_PATH, c3_train.DEFAULT_FEATURE_AUDIT_PATH
    )
    fixed_feature_lineage = vsa_train.validate_fixed_feature_realization(
        c3_train.DEFAULT_FEATURE_PATH,
        c3_train.DEFAULT_FEATURE_AUDIT_PATH,
        feature_provenance,
    )
    model_dir, model_audit = materializer.seed_specific_e1_paths(active_seed)
    model, model_config, checkpoint_provenance = (
        vsa_train.load_lineage_aware_e1_lwc_model(
            model_dir, model_audit, torch_device, active_seed
        )
    )
    seed_lineage = materializer.validate_seed_lineage(
        active_seed, checkpoint_provenance, action_provenance
    )
    semantic_optimizers, native_optimizers, optimizer_audit = (
        c3_train.build_arm_optimizers(model, "TRUE_U")
    )
    orders = vsa.precompute_epoch_orders(
        c5b0.FORMAL_EPOCHS, sample_ids, seed=active_seed
    )
    return {
        "device": torch_device,
        "model": model,
        "model_config": model_config,
        "semantic_optimizers": semantic_optimizers,
        "native_optimizers": native_optimizers,
        "views": views,
        "sample_ids": sample_ids,
        "action_arrays": action_arrays,
        "orders": orders,
        "reference": reference,
        "provenance": {
            "determinism": determinism,
            "parent_hashes": parent_hashes,
            "C3_B0_source": c3_source,
            "C3_B0_formal": c3_formal,
            "C3_B0_hash_summary": c3_summary,
            "feature": feature_provenance,
            "fixed_feature": fixed_feature_lineage,
            "checkpoint": checkpoint_provenance,
            "seed_lineage": seed_lineage,
            "optimizer": optimizer_audit,
        },
    }


def _persist_pre_gt(target, arrays, audit):
    c5b0.validate_pre_gt_array_keys(arrays)
    target.mkdir(parents=True, exist_ok=False)
    artifact_path = target / ARTIFACT_NAME
    with open(artifact_path, "xb") as output_file:
        np.savez(output_file, **arrays)
        output_file.flush()
        os.fsync(output_file.fileno())
    records = _array_records(arrays)
    audit = dict(audit)
    audit.update({
        "artifact_path": _display(artifact_path),
        "artifact_file_sha256": carrier.file_sha256(artifact_path),
        "arrays": records,
    })
    audit_path = target / AUDIT_NAME
    _write_json(audit_path, audit)
    seal = {
        "stage": c5b0.STAGE,
        "seed": audit["seed"],
        "arm": audit["arm"],
        "GT_loaded_before_pre_gt_seal": False,
        "pre_gt_seal_valid": True,
        "all_parent_hashes_pass": True,
        "BASE_replay_parity_pass": audit["BASE_replay_parity_pass"],
        "artifact_path": _display(artifact_path),
        "artifact_file_sha256": carrier.file_sha256(artifact_path),
        "audit_path": _display(audit_path),
        "audit_file_sha256": carrier.file_sha256(audit_path),
        "array_whitelist": list(c5b0.PRE_GT_ARRAY_KEYS),
        "arrays": records,
    }
    seal_path = target / SEAL_NAME
    _write_json(seal_path, seal)
    return {
        "output_dir": target,
        "artifact_path": artifact_path,
        "audit_path": audit_path,
        "seal_path": seal_path,
        "audit": audit,
        "seal": seal,
    }


def run_arm(arm, seed, output_dir=None, device="cuda:0", base_output_dir=None):
    """Train one pre-registered arm; this function has no full-GT input path."""
    active_arm = c5b0.validate_arm(arm)
    active_seed = c5b0.validate_seed(seed)
    target = default_output_dir(active_seed, active_arm) if output_dir is None else _resolve(output_dir)
    _require(not target.exists(), "refusing to overwrite C5-B0 output")
    preregistration = verify_preregistration()
    c5a_arrays, c5a_provenance = load_frozen_c5a_parent(active_seed)
    full_action, full_action_provenance = load_full_c3a0_action(active_seed)
    plans = build_all_anchor_plans(c5a_arrays, full_action)
    plan = plans[active_arm]
    training_action = c5b0.build_training_action(
        full_action["U_cycle"], full_action["unlabeled_ids"],
        full_action["class_pred_true"], plan,
    )
    if active_arm == "BASE":
        base_gate = {"BASE_replay_parity_pass": True, "gate_source": "current replay"}
    else:
        base_gate = load_base_parity_gate(active_seed, base_output_dir)
    context = prepare_training_context(active_seed, device)
    initial_model_hash = hash_backbone(context["model"].autoencoders)
    if active_arm == "BASE":
        replay = materializer.replay_frozen_true_u(
            context["model"], context["semantic_optimizers"],
            context["native_optimizers"], context["views"],
            context["sample_ids"], context["action_arrays"], context["orders"],
            context["device"], active_seed,
        )
        predictions = replay["predictions"]
        runtime = replay["runtime"]
        runtime["self_relation_count_used"] = 0
        parity = carrier.validate_replay_parity(
            context["reference"]["final_model_hash"],
            hash_backbone(context["model"].autoencoders),
            context["reference"]["predictions"], predictions,
            context["reference"]["sample_ids"], replay["sample_ids"],
        )
        parity["coordinate_mapping_pass"] = replay["coordinate"]["coordinate_mapping_pass"]
    else:
        predictions, runtime = train_expansion_arm(
            context["model"], context["semantic_optimizers"],
            context["native_optimizers"], context["views"],
            context["sample_ids"], training_action, context["orders"],
            context["device"], active_seed,
        )
        _require(runtime["self_relation_count_used"] == 0, "C5_B0_SELF_RELATION_USED")
        parity = {
            "comparison_to_frozen_BASE_applicable": False,
            "final_prediction_sample_ids_equal": True,
            "BASE_coordinate_mapping_equality_verified": True,
        }
    final_model_hash = hash_backbone(context["model"].autoencoders)
    arrays = OrderedDict((
        ("sample_ids", np.ascontiguousarray(context["sample_ids"], dtype=np.int64)),
        ("final_predictions", np.ascontiguousarray(predictions, dtype=np.int64)),
        ("eligible_ids", np.ascontiguousarray(plan["eligible_ids"], dtype=np.int64)),
        ("selected_direction", np.ascontiguousarray(plan["selected_direction"], dtype=np.int64)),
        ("admission_score", np.ascontiguousarray(plan["admission_score"], dtype=np.float64)),
        ("expansion_mask", np.ascontiguousarray(plan["expansion_mask"], dtype=np.bool_)),
        ("pseudo_semantic_label", np.ascontiguousarray(plan["pseudo_semantic_label"], dtype=np.int64)),
        ("selected_pseudo_anchor_ids", np.ascontiguousarray(plan["selected_pseudo_anchor_ids"], dtype=np.int64)),
        ("selected_pseudo_anchor_labels", np.ascontiguousarray(plan["selected_pseudo_anchor_labels"], dtype=np.int64)),
        ("real_anchor_ids", np.ascontiguousarray(plan["real_anchor_ids"], dtype=np.int64)),
        ("real_anchor_labels", np.ascontiguousarray(plan["real_anchor_labels"], dtype=np.int64)),
        ("combined_anchor_ids", np.ascontiguousarray(plan["combined_anchor_ids"], dtype=np.int64)),
        ("combined_anchor_labels", np.ascontiguousarray(plan["combined_anchor_labels"], dtype=np.int64)),
        ("combined_anchor_is_pseudo", np.ascontiguousarray(plan["combined_anchor_is_pseudo"], dtype=np.bool_)),
    ))
    audit = {
        "stage": c5b0.STAGE,
        "seed": active_seed,
        "arm": active_arm,
        "preregistration": preregistration,
        "scientific_parent_lineage": [
            "C0", "C3-A0", "C3-B0 TRUE_U", "C3-B0 TRUE_U carrier",
            "C5-A0 V2 pre-GT pseudo-semantic artifact", "C5-B0",
        ],
        "C5_A0_parent": c5a_provenance,
        "C3_A0_parent": full_action_provenance,
        "training_provenance": context["provenance"],
        "base_parity_gate": base_gate,
        "BASE_replay_parity_pass": True,
        "BASE_replays_frozen_C3_B0_TRUE_U": active_arm == "BASE",
        "all_parent_hashes_pass": True,
        "initial_model_hash": initial_model_hash,
        "final_model_hash": final_model_hash,
        **parity,
        "eligible_count": int(plan["eligible_count"]),
        "expansion_budget_fraction": c5b0.EXPANSION_BUDGET_FRACTION,
        "expansion_count": int(plan["expansion_count"]),
        "selected_ID_logical_sha256": c5b0.logical_sha256(arrays["selected_pseudo_anchor_ids"]),
        "expansion_mask_logical_sha256": c5b0.logical_sha256(arrays["expansion_mask"]),
        "pseudo_semantic_label_all_eligible_logical_sha256": c5b0.logical_sha256(arrays["pseudo_semantic_label"]),
        "pseudo_label_logical_sha256": c5b0.logical_sha256(arrays["selected_pseudo_anchor_labels"]),
        "relation_target_logical_sha256": c5b0.logical_sha256(training_action["PredRelation"]),
        "relation_balance_weight_logical_sha256": c5b0.logical_sha256(training_action["balance_weight"]),
        "relation_target_shape": list(training_action["PredRelation"].shape),
        "relation_target_axis_order": list(c5b0.RELATION_TARGET_AXIS_ORDER),
        "self_relation_pair_count_excluded": training_action["self_relation_pair_count_excluded"],
        "self_relation_count_used": runtime["self_relation_count_used"],
        "training_performed": True,
        "GT_loaded_before_pre_gt_seal": False,
        "C4_used": False,
        "pseudo_CE_used": False,
        "continuous_U_weighting_used": False,
        "memory_used": False,
        "recursive_expansion_used": False,
        "pseudo_anchor_refresh_used": False,
        "new_loss_coefficient_used": False,
        "pseudo_anchor_loss_weight_used": False,
        "U_cycle_modified": False,
        "same_training_budget_and_configuration_all_arms": True,
        "training_configuration": {
            "epochs": c5b0.FORMAL_EPOCHS,
            "batch_size": c5b0.BATCH_SIZE,
            "phase_order": ["relation_semantic", "native_consolidation"],
            "relation_objective": c3b0.RELATION_OBJECTIVE,
            "native_objective": c3b0.NATIVE_OBJECTIVE,
            "optimizer": context["provenance"]["optimizer"],
        },
        "runtime": runtime,
    }
    validate_base_parity_record(audit) if active_arm == "BASE" else None
    return _persist_pre_gt(target, arrays, audit)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=c5b0.ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=c5b0.SEEDS)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_arm(
        args.arm, args.seed, args.output_dir, args.device,
        args.base_output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
