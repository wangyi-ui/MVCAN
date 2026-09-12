"""Engineering-only one-epoch smoke for the frozen C3-B0 training graph.

This wrapper deliberately stops before prediction export, label loading, and
scientific evaluation.  It reuses the frozen formal Phase-A and Phase-B
helpers; it does not define an alternative relation or native loss.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _thread_name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_name] = "1"

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import c3_b0_relation_action_protocol as c3b0
from experiments.cyclic_utility import train_c3_b0_relation_action_pilot as train


STAGE = "C3-B0-ENGINEERING-SMOKE"
REPAIR_FAILURE_CLASSIFICATION = (
    "ENGINEERING_SMOKE_AUDIT_CONTAINER_API_MISMATCH"
)
SMOKE_SEED = 20
SMOKE_EPOCH_COUNT = 1
SMOKE_ARMS = ("BASE", "TRUE_U", "TRUE_UNIFORM", "SHUFFLE_U")
PRE_FORMAL_FREEZE = (
    REPOSITORY_ROOT
    / "experiment_freeze/c3_b0_relation_action_preformal_20260912"
)
PRE_FORMAL_SOURCE_MANIFEST = PRE_FORMAL_FREEZE / "source_sha256.txt"
SMOKE_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/cyclic_utility"
SMOKE_DIRECTORY_NAMES = {
    "BASE": "c3_b0_relation_action_engineering_smoke_seed20_base",
    "TRUE_U": "c3_b0_relation_action_engineering_smoke_seed20_true_u",
    "TRUE_UNIFORM": (
        "c3_b0_relation_action_engineering_smoke_seed20_true_uniform"
    ),
    "SHUFFLE_U": "c3_b0_relation_action_engineering_smoke_seed20_shuffle_u",
}
OUTPUT_FILENAMES = ("smoke_audit.json", "smoke_runtime.json")


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def default_smoke_output_dir(arm):
    active_arm = c3b0.validate_arm(arm)
    _require(active_arm in SMOKE_ARMS, "unexpected C3-B0 smoke arm")
    return SMOKE_OUTPUT_ROOT / SMOKE_DIRECTORY_NAMES[active_arm]


def verify_preformal_source_hashes():
    """Verify the frozen formal source manifest against the repository."""
    return train.verify_sha256_manifest(
        PRE_FORMAL_SOURCE_MANIFEST, REPOSITORY_ROOT
    )


def _unique_optimizer_parameters(semantic_optimizers, native_optimizers):
    """Collect optimizer-managed tensors once by object identity."""
    parameters = []
    seen_ids = set()
    for optimizers in (semantic_optimizers, native_optimizers):
        if optimizers is None:
            continue
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    _require(
                        isinstance(parameter, torch.Tensor),
                        "optimizer contains a non-tensor parameter",
                    )
                    identity = id(parameter)
                    if identity not in seen_ids:
                        seen_ids.add(identity)
                        parameters.append(parameter)
    return tuple(parameters)


def _all_optimizer_parameters_finite(
    semantic_optimizers, native_optimizers,
):
    """Check the finite state of the exact union managed by both phases."""
    parameters = _unique_optimizer_parameters(
        semantic_optimizers, native_optimizers
    )
    return bool(
        parameters
        and all(torch.isfinite(parameter.data).all().item()
                for parameter in parameters)
    )


@torch.no_grad()
def audit_true_u_runtime_tensor_shapes(
    model, full_views, action_arrays, semantic_order, device,
):
    """Observe TRUE_U batch tensors without defining or evaluating a loss."""
    validated = c3b0.validate_action_arrays(action_arrays)
    first_query_ids = None
    for batch_sample_ids in train.vsa.ordered_batches(semantic_order):
        query_ids = c3b0.unlabeled_query_ids(
            batch_sample_ids, validated["unlabeled_ids"]
        )
        if query_ids.size:
            first_query_ids = query_ids
            break
    _require(first_query_ids is not None, "TRUE_U has no smoke query batch")
    selected = c3b0.action_batch_by_sample_ids(
        validated, "TRUE_U", first_query_ids
    )
    query_tensor_ids = train._tensor_ids(first_query_ids)
    anchor_tensor_ids = train._tensor_ids(validated["labeled_ids"])
    batch_size = int(first_query_ids.size)
    expected = {
        "U_batch": [batch_size, c3b0.DIRECTION_COUNT],
        "PredRelation": [batch_size, c3b0.LABEL_COUNT,
                         c3b0.DIRECTION_COUNT],
        "balance": [batch_size, c3b0.LABEL_COUNT,
                    c3b0.DIRECTION_COUNT],
        "q_query": [batch_size, c3b0.CLASS_NUM],
        "q_anchor": [c3b0.LABEL_COUNT, c3b0.CLASS_NUM],
        "relation_prob": [batch_size, c3b0.LABEL_COUNT],
    }
    observed = {
        "U_batch": list(selected["U_cycle"].shape),
        "PredRelation": list(selected["PredRelation"].shape),
        "balance": list(selected["balance_weight"].shape),
    }
    per_view = []
    for view_id in range(c3b0.VIEW_NUM):
        query_input = full_views[view_id][query_tensor_ids].to(device)
        anchor_input = full_views[view_id][anchor_tensor_ids].to(device)
        query_latent = model.autoencoders[view_id].encoder(query_input)
        anchor_latent = model.autoencoders[view_id].encoder(anchor_input)
        q_query = model.autoencoders[view_id].clustering(query_latent)
        q_anchor = model.autoencoders[view_id].clustering(anchor_latent)
        relation_prob = c3b0.relation_probability(q_query, q_anchor)
        shapes = {
            "view": view_id + 1,
            "q_query": list(q_query.shape),
            "q_anchor": list(q_anchor.shape),
            "relation_prob": list(relation_prob.shape),
        }
        _require(
            shapes["q_query"] == expected["q_query"]
            and shapes["q_anchor"] == expected["q_anchor"]
            and shapes["relation_prob"] == expected["relation_prob"],
            "TRUE_U posterior runtime tensor shape mismatch",
        )
        per_view.append(shapes)
    observed.update(per_view[0])
    observed.pop("view")
    shape_pass = all(observed[name] == value for name, value in expected.items())
    no_dense_pairwise = all(
        not (len(shape) >= 2 and shape[0] == c3b0.SAMPLE_NUM
             and shape[1] == c3b0.SAMPLE_NUM)
        for shape in observed.values()
    )
    _require(shape_pass, "TRUE_U action runtime tensor shape mismatch")
    _require(no_dense_pairwise, "forbidden all-sample pairwise tensor")
    source_rows = np.asarray(selected["source_rows"], dtype=np.int64)
    _require(
        source_rows.shape == first_query_ids.shape
        and np.array_equal(
            validated["unlabeled_ids"][source_rows], first_query_ids
        ),
        "TRUE_U sample-ID lookup mismatch",
    )
    return {
        "batch_unlabeled_count": batch_size,
        "expected_shapes": expected,
        "observed_shapes": observed,
        "per_view_posterior_shapes": per_view,
        "view_count": len(per_view),
        "sample_ID_lookup_pass": True,
        "runtime_tensor_shapes_pass": shape_pass,
        "forbidden_N_by_N_tensor_created": False,
        "forbidden_N_by_N_by_any_tensor_created": False,
    }


def _gradient_records_finite(records):
    return bool(
        records
        and all(record.get("gradient", {}).get("gradient_finite_pass") is True
                for record in records)
    )


def build_smoke_runtime(
    arm, optimizer_audit, phase_a, phase_b, semantic_optimizers,
    native_optimizers, true_u_shapes=None,
):
    """Reduce formal helper records to fail-closed engineering checks."""
    active_arm = c3b0.validate_arm(arm)
    phase_a_records = phase_a.get("batch_records", [])
    phase_b_records = phase_b.get("batch_records", [])
    phase_a_executed = phase_a.get("executed") is True
    phase_b_executed = phase_b.get("executed") is True
    relation_loss_finite = None if active_arm == "BASE" else bool(
        phase_a_records
        and np.isfinite(phase_a.get("relation_loss_mean", np.nan))
        and all(np.isfinite(record.get("relation_loss", np.nan))
                for record in phase_a_records)
    )
    denominator_finite_and_above_epsilon = None
    if active_arm != "BASE":
        denominator_finite_and_above_epsilon = bool(
            phase_a_records
            and all(
                np.isfinite(record.get("denominator", np.nan))
                and record.get("denominator", 0.0) > np.finfo(np.float32).eps
                for record in phase_a_records
            )
        )
    native_rec_finite = bool(
        phase_b_records
        and all(np.isfinite(record.get("reconstruction_loss_sum_views", np.nan))
                for record in phase_b_records)
    )
    native_clu_finite = bool(
        phase_b_records
        and all(np.isfinite(record.get("clustering_loss_sum_views", np.nan))
                for record in phase_b_records)
    )
    native_loss_finite = bool(
        phase_b_records
        and all(np.isfinite(record.get("native_loss", np.nan))
                for record in phase_b_records)
        and native_rec_finite and native_clu_finite
    )
    semantic_gradient_finite = None if active_arm == "BASE" else (
        _gradient_records_finite(phase_a_records)
    )
    native_gradient_finite = _gradient_records_finite(phase_b_records)
    gradient_finite = bool(
        native_gradient_finite
        and (active_arm == "BASE" or semantic_gradient_finite)
    )
    phase_a_view_pass = active_arm == "BASE" or bool(
        phase_a_records
        and all(record.get("view_count") == c3b0.VIEW_NUM
                for record in phase_a_records)
    )
    sample_id_alignment_pass = bool(
        phase_b.get("all_expected_sample_ids_exactly_once_pass") is True
        and (active_arm == "BASE" or
             phase_a.get("all_expected_sample_ids_exactly_once_pass") is True)
    )
    semantic_optimizer_count = int(
        optimizer_audit.get("semantic_optimizer_count", -1)
    )
    native_optimizer_count = int(
        optimizer_audit.get("native_optimizer_count", -1)
    )
    semantic_backward_count = len(phase_a_records) if phase_a_executed else 0
    native_backward_count = len(phase_b_records) if phase_b_executed else 0
    semantic_step_count = semantic_backward_count * semantic_optimizer_count
    native_step_count = native_backward_count * native_optimizer_count
    diagnostic_anchor_forwards = (
        c3b0.VIEW_NUM if active_arm == "TRUE_U" and true_u_shapes else 0
    )
    training_anchor_forwards = (
        semantic_backward_count * c3b0.VIEW_NUM
        if active_arm != "BASE" else 0
    )
    anchor_count = 0 if active_arm == "BASE" else int(
        phase_a.get("anchor_count", -1)
    )
    parameter_finite = _all_optimizer_parameters_finite(
        semantic_optimizers, native_optimizers
    )
    common_pass = bool(
        phase_b_executed
        and native_backward_count > 0
        and native_optimizer_count == c3b0.VIEW_NUM
        and native_loss_finite
        and gradient_finite
        and parameter_finite
        and sample_id_alignment_pass
        and phase_a_view_pass
    )
    if active_arm == "BASE":
        arm_pass = bool(
            not phase_a_executed
            and semantic_optimizer_count == 0
            and semantic_backward_count == 0
            and semantic_step_count == 0
            and training_anchor_forwards + diagnostic_anchor_forwards == 0
        )
    else:
        arm_pass = bool(
            phase_a_executed
            and semantic_optimizer_count == c3b0.VIEW_NUM
            and semantic_backward_count > 0
            and relation_loss_finite
            and denominator_finite_and_above_epsilon
            and anchor_count == c3b0.LABEL_COUNT
        )
        if active_arm == "TRUE_U":
            arm_pass = bool(
                arm_pass and true_u_shapes
                and true_u_shapes.get("runtime_tensor_shapes_pass") is True
                and true_u_shapes.get("sample_ID_lookup_pass") is True
                and true_u_shapes.get("view_count") == c3b0.VIEW_NUM
                and true_u_shapes.get("forbidden_N_by_N_tensor_created") is False
                and true_u_shapes.get(
                    "forbidden_N_by_N_by_any_tensor_created"
                ) is False
            )
    return {
        "arm": active_arm,
        "seed": SMOKE_SEED,
        "epoch_count": SMOKE_EPOCH_COUNT,
        "phase_a_executed": phase_a_executed,
        "phase_b_executed": phase_b_executed,
        "phase_a_execution_count": int(phase_a_executed),
        "phase_b_execution_count": int(phase_b_executed),
        "semantic_optimizer_count": semantic_optimizer_count,
        "native_optimizer_count": native_optimizer_count,
        "semantic_backward_count": semantic_backward_count,
        "native_backward_count": native_backward_count,
        "semantic_step_count": semantic_step_count,
        "native_step_count": native_step_count,
        "anchor_relation_forward_count": (
            training_anchor_forwards + diagnostic_anchor_forwards
        ),
        "relation_loss_applicable": active_arm != "BASE",
        "relation_loss_finite": relation_loss_finite,
        "native_REC_finite": native_rec_finite,
        "native_CLU_finite": native_clu_finite,
        "native_loss_finite": native_loss_finite,
        "semantic_gradient_finite": semantic_gradient_finite,
        "native_gradient_finite": native_gradient_finite,
        "gradient_finite": gradient_finite,
        "parameter_finite": parameter_finite,
        "parameter_finite_source": (
            "deduplicated union of semantic/native optimizer "
            "param_groups[*]['params']"
        ),
        "optimizer_parameter_count_after_id_deduplication": len(
            _unique_optimizer_parameters(
                semantic_optimizers, native_optimizers
            )
        ),
        "denominator_finite_and_above_epsilon": (
            denominator_finite_and_above_epsilon
        ),
        "sample_id_alignment_pass": sample_id_alignment_pass,
        "unlabeled_query_count_positive": (
            None if active_arm == "BASE" else
            bool(phase_a_records and all(
                record.get("query_count", 0) > 0
                for record in phase_a_records
            ))
        ),
        "anchor_count": anchor_count,
        "view_count": c3b0.VIEW_NUM if phase_a_view_pass else None,
        "TRUE_U_runtime_shapes": true_u_shapes,
        "formal_relation_phase_helper": (
            None if active_arm == "BASE" else
            "train_c3_b0_relation_action_pilot.relation_semantic_phase"
        ),
        "formal_native_phase_helper": (
            "train_c3_b0_relation_action_pilot.native_consolidation_phase"
        ),
        "formal_optimizer_helper": (
            "train_c3_b0_relation_action_pilot.build_arm_optimizers"
        ),
        "phase_A": phase_a,
        "phase_B": phase_b,
        "C3_B0_SMOKE_PASS": bool(common_pass and arm_pass),
    }


def run_one_epoch_training_core(
    arm, model, semantic_optimizers, native_optimizers, optimizer_audit,
    views, sample_ids, action_arrays, device,
):
    """Run one epoch through the exact frozen formal phase helpers."""
    active_arm = c3b0.validate_arm(arm)
    ids = np.asarray(sample_ids, dtype=np.int64)
    _require(
        ids.shape == (c3b0.SAMPLE_NUM,)
        and np.array_equal(ids, np.arange(c3b0.SAMPLE_NUM)),
        "smoke feature sample IDs are not canonical",
    )
    id_tensor = torch.from_numpy(ids)
    training_dataset = torch.utils.data.TensorDataset(
        *[torch.from_numpy(view) for view in views], id_tensor
    )
    _require(
        len(training_dataset.tensors) == c3b0.VIEW_NUM + 1
        and torch.equal(training_dataset.tensors[-1], id_tensor),
        "smoke TensorDataset must carry canonical sample IDs",
    )
    full_views = list(training_dataset.tensors[:c3b0.VIEW_NUM])
    orders = train.vsa.precompute_epoch_orders(
        SMOKE_EPOCH_COUNT, ids, seed=SMOKE_SEED
    )
    true_u_shapes = None
    if active_arm == "BASE":
        _require(semantic_optimizers is None,
                 "BASE smoke received semantic optimizers")
        phase_a = {
            "phase": "relation_semantic", "executed": False,
            "reason": "BASE has no Phase A",
        }
    else:
        _require(
            semantic_optimizers is not None
            and len(semantic_optimizers) == c3b0.VIEW_NUM,
            "non-BASE smoke semantic optimizer boundary mismatch",
        )
        if active_arm == "TRUE_U":
            true_u_shapes = audit_true_u_runtime_tensor_shapes(
                model, full_views, action_arrays,
                orders["semantic_orders"][0], device,
            )
        phase_a = train.relation_semantic_phase(
            model, semantic_optimizers, full_views, action_arrays, active_arm,
            orders["semantic_orders"][0], device,
        )
    view_weights = [1.0] * c3b0.VIEW_NUM
    native_p_all, native_matches, _, _ = train.e1_train.refresh_native_target(
        model, full_views, view_weights, device, training_seed=SMOKE_SEED
    )
    phase_b = train.native_consolidation_phase(
        model, native_optimizers, full_views, native_p_all, native_matches,
        orders["native_orders"][0], device,
    )
    runtime = build_smoke_runtime(
        active_arm, optimizer_audit, phase_a, phase_b,
        semantic_optimizers, native_optimizers, true_u_shapes
    )
    runtime["phase_sequence"] = [
        {"epoch": 1, "phase": "relation_semantic",
         "executed": active_arm != "BASE"},
        {"epoch": 1, "phase": "native_consolidation", "executed": True},
    ]
    runtime["order_audit"] = {
        "seed": orders["seed"],
        "semantic_order_sha256": orders[
            "semantic_order_sha256_per_epoch"
        ][0],
        "native_order_sha256": orders["native_order_sha256_per_epoch"][0],
        "orders_precomputed_before_training": orders[
            "orders_precomputed_before_training"
        ],
    }
    _require(runtime["C3_B0_SMOKE_PASS"],
             "C3-B0 engineering smoke runtime audit failed")
    return runtime


def _freeze_unchanged(before, after):
    return bool(
        before.get("all_entries_exact_match") is True
        and after.get("all_entries_exact_match") is True
        and before.get("manifest_file_sha256") == after.get(
            "manifest_file_sha256"
        )
        and before.get("entries") == after.get("entries")
    )


def run_smoke_arm(arm, device="cuda:0"):
    """Execute one engineering arm without any GT or scientific evaluation."""
    active_arm = c3b0.validate_arm(arm)
    _require(active_arm in SMOKE_ARMS, "unexpected smoke arm")
    target = default_smoke_output_dir(active_arm)
    _require(not target.exists(), "refusing to overwrite C3-B0 smoke output")
    before_hashes = verify_preformal_source_hashes()
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        _require(torch.cuda.is_available(), "requested CUDA unavailable")
        torch.cuda.set_device(torch_device)
    determinism = train.vsa_train.enable_strict_determinism(SMOKE_SEED)
    action_arrays, action_provenance = train.load_frozen_c3a0_action_bundle(
        SMOKE_SEED
    )
    views, sample_ids, feature_provenance = (
        train.e1_train.load_frozen_feature_artifact(
            train.DEFAULT_FEATURE_PATH, train.DEFAULT_FEATURE_AUDIT_PATH
        )
    )
    fixed_feature_lineage = train.vsa_train.validate_fixed_feature_realization(
        train.DEFAULT_FEATURE_PATH, train.DEFAULT_FEATURE_AUDIT_PATH,
        feature_provenance,
    )
    model, model_config, checkpoint_provenance = (
        train.vsa_train.load_lineage_aware_e1_lwc_model(
            train.DEFAULT_MODEL_DIR, train.DEFAULT_MODEL_AUDIT_PATH,
            torch_device, SMOKE_SEED,
        )
    )
    stateful_layer_audit = train.vsa.audit_model_stateful_layers(model)
    semantic_optimizers, native_optimizers, optimizer_audit = (
        train.build_arm_optimizers(model, active_arm)
    )
    target.mkdir(parents=True, exist_ok=False)
    runtime = run_one_epoch_training_core(
        active_arm, model, semantic_optimizers, native_optimizers,
        optimizer_audit, views, sample_ids, action_arrays, torch_device,
    )
    after_hashes = verify_preformal_source_hashes()
    freeze_unchanged = _freeze_unchanged(before_hashes, after_hashes)
    _require(freeze_unchanged, "C3-B0 pre-formal source freeze changed")
    audit = {
        "stage": STAGE,
        "engineering_smoke_only": True,
        "scientific_experiment": False,
        "repair_failure_classification": REPAIR_FAILURE_CLASSIFICATION,
        "arm": active_arm,
        "seed": SMOKE_SEED,
        "epoch_count": SMOKE_EPOCH_COUNT,
        "output_dir": train._display(target),
        "output_files": list(OUTPUT_FILENAMES),
        "phase_a_executed": runtime["phase_a_executed"],
        "phase_b_executed": runtime["phase_b_executed"],
        "semantic_optimizer_count": runtime["semantic_optimizer_count"],
        "native_optimizer_count": runtime["native_optimizer_count"],
        "relation_loss_finite": runtime["relation_loss_finite"],
        "native_loss_finite": runtime["native_loss_finite"],
        "gradient_finite": runtime["gradient_finite"],
        "parameter_finite": runtime["parameter_finite"],
        "sample_id_alignment_pass": runtime["sample_id_alignment_pass"],
        "anchor_count": runtime["anchor_count"],
        "view_count": runtime["view_count"],
        "GT_loaded": False,
        "scientific_metrics_computed": False,
        "scientific_metrics_printed": False,
        "scientific_metrics_saved": False,
        "formal_output_path_used": False,
        "formal_gate_read_or_applied": False,
        "formal_training_helpers_reused": True,
        "alternative_relation_loss_defined": False,
        "preformal_source_hash_before": before_hashes,
        "preformal_source_hash_after": after_hashes,
        "preformal_source_hash_unchanged": freeze_unchanged,
        "determinism": determinism,
        "action_provenance": action_provenance,
        "feature_provenance": feature_provenance,
        "fixed_feature_lineage": fixed_feature_lineage,
        "checkpoint_provenance": checkpoint_provenance,
        "model_training_config": model_config["training"],
        "stateful_layer_audit": stateful_layer_audit,
        "optimizer_audit": optimizer_audit,
        "C3_B0_SMOKE_PASS": runtime["C3_B0_SMOKE_PASS"],
    }
    train.c1.write_json(target / "smoke_runtime.json", runtime)
    train.c1.write_json(target / "smoke_audit.json", audit)
    actual_files = tuple(sorted(path.name for path in target.iterdir()))
    _require(
        actual_files == tuple(sorted(OUTPUT_FILENAMES)),
        "smoke output contains a forbidden file",
    )
    return {"output_dir": target, "audit": audit, "runtime": runtime}


def run_engineering_smoke(device="cuda:0"):
    """Run exactly the four preregistered engineering arms."""
    _require(SMOKE_ARMS == c3b0.ARMS, "smoke arm set differs from formal arms")
    for arm in SMOKE_ARMS:
        _require(
            not default_smoke_output_dir(arm).exists(),
            "one or more C3-B0 smoke output directories already exist",
        )
    results = {arm: run_smoke_arm(arm, device=device) for arm in SMOKE_ARMS}
    passed = bool(all(
        result["audit"]["C3_B0_SMOKE_PASS"] for result in results.values()
    ))
    _require(passed, "one or more C3-B0 engineering smoke arms failed")
    return {
        "stage": STAGE,
        "arms": list(SMOKE_ARMS),
        "seed": SMOKE_SEED,
        "epoch_count": SMOKE_EPOCH_COUNT,
        "C3_B0_ENGINEERING_SMOKE_PASS": True,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = run_engineering_smoke(device=args.device)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
