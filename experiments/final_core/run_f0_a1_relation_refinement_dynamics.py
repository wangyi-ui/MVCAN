"""Run one instrumented F0-A1 trajectory and seal it before any GT boundary.

Importing this module performs no training and reads no ground-truth labels.
The CLI is intentionally limited to the preregistered arm, seed, device, and
output directory.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
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
from experiments.cyclic_utility import materialize_c3_b0_true_u_carrier as replay
from experiments.cyclic_utility import train_c3_b0_relation_action_pilot as c3_train
from experiments.cyclic_utility import train_vsa_a0_decoupled_action as vsa_train
from experiments.cyclic_utility import vsa_decoupled_action_protocol as vsa
from experiments.e1_pairwise_utility import train_e1_pairwise_utility as e1_train
from experiments.final_core import f0_a1_relation_refinement_dynamics_protocol as f0a1
from experiments.final_core import final_core_protocol as f0
from experiments.final_core import train_final_core as f0_runner
from irv.b3_audit import hash_backbone


def _require(condition, message="F0_A1_RUNNER_FAIL_CLOSED"):
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


def _write_json(path, record):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(record, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def verify_parent_integrity():
    """Verify F0-A1 pins plus the complete frozen F0-A0/C3-B0 lineage."""
    preregistration = f0a1.verify_preregistered_protocol()
    parent_pins = f0a1.verify_parent_freezes_manifest()
    historical_lineage = f0_runner.verify_parent_integrity()
    _require(
        preregistration["exact_match"]
        and parent_pins["all_parent_hashes_pass"]
        and historical_lineage["all_parent_hashes_pass"],
        "F0_A1_PARENT_INTEGRITY_FAIL_CLOSED",
    )
    return {
        "F0_A1_preregistration": preregistration,
        "F0_A1_parent_pins": parent_pins,
        "F0_A0_C3_B0_lineage": historical_lineage,
        "all_parent_hashes_pass": True,
    }


def _all_optimizers(semantic_optimizers, native_optimizers):
    return (
        ([] if semantic_optimizers is None else list(semantic_optimizers))
        + list(native_optimizers)
    )


def run_instrumented_schedule(
    arm, model, semantic_optimizers, native_optimizers, full_views,
    sample_ids, action_arrays, orders, device, training_seed,
    snapshot_fn=None,
):
    """Run the frozen 20-epoch schedule with observational q-only snapshots."""
    active_arm = f0a1.validate_arm(arm)
    active_seed = f0a1.validate_seed(training_seed)
    ids = np.asarray(sample_ids, dtype=np.int64)
    _require(
        ids.shape == (f0a1.N,) and np.array_equal(ids, np.arange(f0a1.N)),
        "F0_A1_TRAINING_SAMPLE_IDS_FAIL_CLOSED",
    )
    _require(
        len(orders["semantic_orders"]) == f0a1.EPOCHS
        and len(orders["native_orders"]) == f0a1.EPOCHS,
        "F0_A1_EPOCH_ORDER_BOUNDARY_FAIL_CLOSED",
    )
    if active_arm == "BASE":
        _require(
            semantic_optimizers is None,
            "F0_A1_BASE_SEMANTIC_OPTIMIZER_FAIL_CLOSED",
        )
    else:
        _require(
            semantic_optimizers is not None
            and len(semantic_optimizers) == f0a1.V,
            "F0_A1_SEMANTIC_OPTIMIZER_BOUNDARY_FAIL_CLOSED",
        )
    observe = (
        f0a1.full_data_q_local_snapshot if snapshot_fn is None else snapshot_fn
    )
    optimizers = _all_optimizers(semantic_optimizers, native_optimizers)
    snapshots = []
    snapshot_audits = []
    snapshot_epochs = []
    snapshot_phases = []

    def record(epoch, phase):
        # Each q snapshot is [N,V,K], detached CPU float32 and observational.
        q_local, audit = observe(
            model, full_views, ids, device, optimizers=optimizers,
            return_audit=True,
        )
        snapshots.append(q_local)
        snapshot_audits.append(audit)
        snapshot_epochs.append(epoch)
        snapshot_phases.append(phase)

    # Q_0 precedes Phase A, the scheduled refresh, and Phase B.
    record(0, "INITIAL")
    view_weights = [1.0] * f0a1.V
    native_p_all = None
    native_matches = None
    refresh_count = 0
    epoch_records = []
    for epoch_index in range(f0a1.EPOCHS):
        epoch = epoch_index + 1
        if active_arm == "BASE":
            semantic_record = {
                "phase": "relation_semantic",
                "executed": False,
                "reason": "BASE has no Phase A",
            }
        else:
            semantic_record = c3_train.relation_semantic_phase(
                model, semantic_optimizers, full_views, action_arrays,
                active_arm, orders["semantic_orders"][epoch_index], device,
            )
            record(epoch, "POST_A")

        if epoch_index % e1_train.TARGET_REFRESH_INTERVAL == 0:
            native_p_all, native_matches, _, view_weights = (
                e1_train.refresh_native_target(
                    model, full_views, view_weights, device,
                    training_seed=active_seed,
                )
            )
            refresh_count += 1
        native_record = c3_train.native_consolidation_phase(
            model, native_optimizers, full_views, native_p_all,
            native_matches, orders["native_orders"][epoch_index], device,
        )
        record(epoch, "POST_B")
        epoch_records.append({
            "epoch": epoch,
            "phase_A": semantic_record,
            "phase_B": native_record,
        })

    # The only post-training refresh is the frozen final-prediction refresh.
    _, final_matches, predictions, view_weights = e1_train.refresh_native_target(
        model, full_views, view_weights, device, training_seed=active_seed
    )
    del final_matches, view_weights
    refresh_count += 1
    expected_epoch, expected_phase = f0a1.snapshot_schedule(active_arm)
    actual_epoch = np.ascontiguousarray(snapshot_epochs, dtype=np.int64)
    actual_phase = np.ascontiguousarray(snapshot_phases, dtype="<U7")
    _require(
        refresh_count == 2
        and np.array_equal(actual_epoch, expected_epoch)
        and np.array_equal(actual_phase, expected_phase),
        "F0_A1_INSTRUMENTED_SCHEDULE_FAIL_CLOSED",
    )
    # q_local_snapshots: [snapshot_count,N,V,K].
    q_local_snapshots = torch.stack(snapshots, dim=0).detach().cpu().numpy()
    q_local_snapshots = np.ascontiguousarray(q_local_snapshots, dtype=np.float32)
    return {
        "predictions": np.ascontiguousarray(predictions, dtype=np.int64),
        "sample_ids": ids,
        "snapshot_epoch": actual_epoch,
        "snapshot_phase": actual_phase,
        "q_local_snapshots": q_local_snapshots,
        "snapshot_audits": snapshot_audits,
        "runtime": {
            "epochs": f0a1.EPOCHS,
            "arm": active_arm,
            "epoch_records": epoch_records,
            "phase_A_before_phase_B_every_epoch": True,
            "native_target_refresh_count_inside_epochs": 1,
            "final_native_prediction_refresh_count": 1,
            "refresh_native_target_total_count": refresh_count,
            "native_refresh_interval": e1_train.TARGET_REFRESH_INTERVAL,
            "extra_refresh_for_snapshot": False,
        },
    }


def _artifact_arrays(
    active_arm, active_seed, result, action_arrays, action_provenance,
    final_model_hash,
):
    return OrderedDict((
        ("sample_ids", np.ascontiguousarray(result["sample_ids"], dtype=np.int64)),
        ("labeled_ids", np.ascontiguousarray(
            action_arrays["labeled_ids"], dtype=np.int64
        )),
        ("unlabeled_ids", np.ascontiguousarray(
            action_arrays["unlabeled_ids"], dtype=np.int64
        )),
        ("snapshot_epoch", result["snapshot_epoch"]),
        ("snapshot_phase", result["snapshot_phase"]),
        ("q_local_snapshots", result["q_local_snapshots"]),
        ("final_predictions", np.ascontiguousarray(
            result["predictions"], dtype=np.int64
        )),
        ("final_model_hash", np.asarray(
            final_model_hash["aggregate"], dtype="<U64"
        )),
        ("frozen_U_logical_hash", np.asarray(
            action_provenance["logical_sha256"]["U_cycle"], dtype="<U64"
        )),
        ("relation_target_logical_hash", np.asarray(
            action_provenance["logical_sha256"]["PredRelation_true"],
            dtype="<U64",
        )),
        ("relation_balance_weights_logical_hash", np.asarray(
            action_provenance["logical_sha256"][
                "relation_balance_weights_true"
            ], dtype="<U64",
        )),
        ("arm", np.asarray(active_arm, dtype="<U12")),
        ("seed", np.asarray(active_seed, dtype=np.int64)),
    ))


def _persist_pre_gt(target, arrays, audit):
    """Atomically establish a new directory; never overwrite an existing run."""
    _require(not target.exists(), "refusing to overwrite F0-A1 output")
    f0a1.validate_artifact_arrays(arrays)
    target.mkdir(parents=True, exist_ok=False)
    artifact_path = target / "f0_a1_trajectory_pre_gt.npz"
    np.savez(artifact_path, **arrays)
    with open(artifact_path, "rb+") as artifact_file:
        os.fsync(artifact_file.fileno())
    reloaded = f0a1.load_artifact_arrays(artifact_path)
    _require(
        all(np.array_equal(reloaded[name], arrays[name]) for name in arrays),
        "F0_A1_ARTIFACT_RELOAD_FAIL_CLOSED",
    )
    audit["arrays"] = f0a1.array_records(arrays)
    audit["artifact_path"] = _display(artifact_path)
    audit["artifact_file_sha256"] = f0a1.file_sha256(artifact_path)
    audit_path = target / "f0_a1_trajectory_audit.json"
    _write_json(audit_path, audit)
    seal = f0a1.build_pre_gt_seal(audit, artifact_path, audit_path)
    seal_path = target / "f0_a1_trajectory_seal.json"
    _write_json(seal_path, seal)
    f0a1.validate_pre_gt_seal(artifact_path, audit_path, seal_path)
    return {
        "output_dir": target,
        "artifact_path": artifact_path,
        "audit_path": audit_path,
        "seal_path": seal_path,
        "audit": audit,
        "seal": seal,
    }


def run_trajectory(seed, arm, device, output_dir):
    """Execute one frozen arm and stop after its validated pre-GT seal."""
    active_seed = f0a1.validate_seed(seed)
    active_arm = f0a1.validate_arm(arm)
    target = _resolve(output_dir)
    _require(not target.exists(), "refusing to overwrite F0-A1 output")

    # All provenance gates run before model construction or training.
    parent_integrity = verify_parent_integrity()
    action_arrays, action_provenance = c3_train.load_frozen_c3a0_action_bundle(
        active_seed
    )
    reference = f0a1.load_frozen_c3_b0_reference(active_seed, active_arm)
    views, sample_ids, feature_provenance = e1_train.load_frozen_feature_artifact(
        c3_train.DEFAULT_FEATURE_PATH, c3_train.DEFAULT_FEATURE_AUDIT_PATH
    )
    sample_gate = f0.validate_sample_ids(sample_ids, reference["sample_ids"])
    fixed_feature_lineage = vsa_train.validate_fixed_feature_realization(
        c3_train.DEFAULT_FEATURE_PATH,
        c3_train.DEFAULT_FEATURE_AUDIT_PATH,
        feature_provenance,
    )
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        _require(torch.cuda.is_available(), "requested CUDA unavailable")
        torch.cuda.set_device(torch_device)
    determinism = vsa_train.enable_strict_determinism(active_seed)
    model_dir, model_audit = replay.seed_specific_e1_paths(active_seed)
    model, model_config, checkpoint_provenance = (
        vsa_train.load_lineage_aware_e1_lwc_model(
            model_dir, model_audit, torch_device, active_seed
        )
    )
    seed_lineage = replay.validate_seed_lineage(
        active_seed, checkpoint_provenance, action_provenance
    )
    initial_model_hash = hash_backbone(model.autoencoders)
    stateful_layer_audit = vsa.audit_model_stateful_layers(model)
    semantic_optimizers, native_optimizers, optimizer_audit = (
        c3_train.build_arm_optimizers(model, active_arm)
    )
    orders = vsa.precompute_epoch_orders(
        c3b0.FORMAL_EPOCHS, sample_ids, seed=active_seed
    )
    id_tensor = torch.from_numpy(np.asarray(sample_ids, dtype=np.int64))
    training_dataset = torch.utils.data.TensorDataset(
        *[torch.from_numpy(view) for view in views], id_tensor
    )
    _require(
        len(training_dataset.tensors) == f0a1.V + 1
        and torch.equal(training_dataset.tensors[-1], id_tensor),
        "F0_A1_TENSOR_DATASET_SAMPLE_IDS_FAIL_CLOSED",
    )
    full_views = list(training_dataset.tensors[:f0a1.V])
    result = run_instrumented_schedule(
        active_arm, model, semantic_optimizers, native_optimizers,
        full_views, sample_ids, action_arrays, orders, torch_device,
        active_seed,
    )
    final_model_hash = hash_backbone(model.autoencoders)
    parity = f0.validate_replay_parity(
        reference["final_model_hash"], final_model_hash,
        reference["predictions"], result["predictions"],
        reference["sample_ids"], result["sample_ids"],
    )
    arrays = _artifact_arrays(
        active_arm, active_seed, result, action_arrays, action_provenance,
        final_model_hash,
    )
    pre_gt_diagnostics = f0a1.compute_trajectory_diagnostics(
        arrays, action_arrays["PredRelation_true"], action_arrays["U_cycle"],
        action_arrays["relation_balance_weights_true"],
    )
    expected_count = 21 if active_arm == "BASE" else 41
    expected_epoch, expected_phase = f0a1.snapshot_schedule(active_arm)
    snapshot_audits = result["snapshot_audits"]
    flags = f0a1.default_forbidden_flags()
    audit = {
        "stage": f0a1.STAGE,
        "purpose": f0a1.PURPOSE,
        "seed": active_seed,
        "arm": active_arm,
        "preregistered_protocol_path": _display(
            f0a1.PREREGISTERED_PROTOCOL_PATH
        ),
        "preregistered_protocol_sha256": (
            f0a1.PREREGISTERED_PROTOCOL_SHA256
        ),
        "preregistered_protocol_hash_pass": True,
        "all_parent_hashes_pass": True,
        "parent_integrity": parent_integrity,
        "sample_ids_equal": sample_gate["exact_equal"],
        "labeled_ids_equal": True,
        "unlabeled_ids_equal": True,
        "snapshot_count_expected": expected_count,
        "snapshot_count_actual": int(arrays["q_local_snapshots"].shape[0]),
        "snapshot_count_equal": arrays["q_local_snapshots"].shape[0] == expected_count,
        "snapshot_shape_pass": arrays["q_local_snapshots"].shape == (
            expected_count, f0a1.N, f0a1.V, f0a1.K
        ),
        "snapshot_order_pass": (
            np.array_equal(arrays["snapshot_epoch"], expected_epoch)
            and np.array_equal(arrays["snapshot_phase"], expected_phase)
        ),
        "snapshot_finite_pass": bool(np.isfinite(
            arrays["q_local_snapshots"]
        ).all()),
        "q_local_primary_state": True,
        "q_aligned_primary_state": False,
        "extra_refresh_for_snapshot": False,
        "refresh_native_target_expected_count": 2,
        "refresh_native_target_actual_count": result["runtime"][
            "refresh_native_target_total_count"
        ],
        "trajectory_used_for_training": False,
        "trajectory_gradient_enabled": False,
        "GT_loaded_during_trajectory": False,
        "GT_used_for_checkpoint_selection": False,
        "scientific_metric_loaded_during_runner": False,
        "full_GT_present_in_pre_gt_artifact": False,
        "final_sample_ids_equal": parity["final_prediction_sample_ids_equal"],
        "final_predictions_equal": parity["final_predictions_equal"],
        "final_model_hash_supported": True,
        "final_model_hash_equal": parity["final_model_hash_equal"],
        "model_state_changed_by_snapshot": not all(
            item["model_parameters_unchanged"]
            and item["model_buffers_unchanged"]
            and item["model_aggregate_hash_equal"]
            and item["model_training_flags_unchanged"]
            for item in snapshot_audits
        ),
        "optimizer_state_changed_by_snapshot": not all(
            item["optimizer_state_unchanged"] for item in snapshot_audits
        ),
        "rng_state_changed_by_snapshot": not all(
            item["rng_state_unchanged"] for item in snapshot_audits
        ),
        "semantic_phase_executed": active_arm != "BASE",
        "U_used_for_training": active_arm == "TRUE_U",
        "forbidden_flags": dict(flags),
        **dict(flags),
        "historical_reference": {
            key: value for key, value in reference.items()
            if key not in ("predictions", "sample_ids", "final_model_hash")
        },
        "gate_0_exact_replay": parity,
        "action_provenance": action_provenance,
        "feature_provenance": feature_provenance,
        "fixed_feature_lineage": fixed_feature_lineage,
        "checkpoint_provenance": checkpoint_provenance,
        "seed_lineage": seed_lineage,
        "initial_model_hash": initial_model_hash,
        "final_model_hash": final_model_hash,
        "model_config": model_config,
        "optimizer_audit": optimizer_audit,
        "stateful_layer_audit": stateful_layer_audit,
        "determinism": determinism,
        "pre_gt_diagnostics": pre_gt_diagnostics,
        "runtime": result["runtime"],
        "snapshot_non_interference": snapshot_audits,
        "tensor_contract": {
            "q_local_snapshots": [expected_count, f0a1.N, f0a1.V, f0a1.K],
            "relation_state": [f0a1.NU, f0a1.L, f0a1.V],
            "U_cycle": [f0a1.NU, f0a1.S],
            "PredRelation_true": [f0a1.NU, f0a1.L, f0a1.S],
            "relation_balance_weights_true": [f0a1.NU, f0a1.L, f0a1.S],
        },
        "gradient_semantics": {
            "snapshot_torch_no_grad": True,
            "snapshot_detached": True,
            "anchor_posterior_stop_gradient_reused": True,
            "trajectory_diagnostics_no_grad": True,
        },
    }
    _require(
        not audit["model_state_changed_by_snapshot"]
        and not audit["optimizer_state_changed_by_snapshot"]
        and not audit["rng_state_changed_by_snapshot"],
        "F0_A1_SNAPSHOT_AGGREGATE_NON_INTERFERENCE_FAIL_CLOSED",
    )
    return _persist_pre_gt(target, arrays, audit)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=f0a1.SEEDS)
    parser.add_argument("--arm", required=True, choices=f0a1.ARMS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_trajectory(
        seed=args.seed,
        arm=args.arm,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "stage": f0a1.STAGE,
        "seed": args.seed,
        "arm": args.arm,
        "pre_gt_seal": _display(result["seal_path"]),
        "pre_gt_seal_valid": True,
        "GT_loaded": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
