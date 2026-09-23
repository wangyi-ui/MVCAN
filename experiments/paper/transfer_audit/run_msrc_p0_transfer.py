"""Run one frozen MSRC P0 arm without accepting or importing full GT."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _thread_name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
):
    os.environ[_thread_name] = "1"

import numpy as np
import torch

from release_core.backbone import MultiViewBackbone
from release_core.backbone.clustering import native_refresh_from_latents
from release_core.config import get_native_config
from release_core.data.weak_quality import ndarray_sha256
from release_core.runtime import ProvenanceConfig, RuntimeConfig, run_pre_gt
import release_core.runtime.entrypoint as runtime_entrypoint
from release_core.semantics import build_relation_semantics
from release_core.training import (
    initial_native_target_state,
    precompute_training_orders,
    refresh_native_state_if_due,
    run_native_consolidation_phase,
)
from release_core.utility import build_directional_actions, compute_directional_cycle_utility

from . import msrc_p0_protocol as protocol
from .input_artifacts import file_sha256, load_materialized_inputs


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _checkpoint_identity():
    _require(protocol.CHECKPOINT_AUDIT.is_file(), "checkpoint audit is missing")
    _require(protocol.CHECKPOINT_SOURCE_MANIFEST.is_file(), "checkpoint manifest is missing")
    _require(protocol.CHECKPOINT_SOURCE_PROVENANCE.is_file(), "checkpoint provenance is missing")
    actual = tuple(file_sha256(path) for path in protocol.CHECKPOINT_PATHS)
    _require(actual == protocol.CHECKPOINT_SHA256, "checkpoint whole-file SHA256 mismatch")
    with protocol.CHECKPOINT_AUDIT.open("r", encoding="utf-8") as stream:
        audit = json.load(stream)
    _require(
        audit.get("checkpoint_sha256") == list(protocol.CHECKPOINT_SHA256)
        and audit.get("release_runtime_initial_model_sha256")
        == protocol.INITIAL_MODEL_SHA256
        and audit.get("source_condition") == protocol.CHECKPOINT_SOURCE_CONDITION,
        "checkpoint audit identity mismatch",
    )
    return actual


def _load_frozen_initial_model(device):
    model = MultiViewBackbone(
        get_native_config(protocol.DATASET),
        view_num=protocol.V,
        view_size=protocol.VIEW_DIMS,
        n_clusters=protocol.K,
        seed=protocol.TRAINING_SEED,
    )
    states = tuple(
        runtime_entrypoint._load_torch_state(path) for path in protocol.CHECKPOINT_PATHS
    )
    model.load_state_dicts(states, strict=True)
    _require(
        runtime_entrypoint._model_sha256(model) == protocol.INITIAL_MODEL_SHA256,
        "release_core initial-model SHA256 mismatch",
    )
    model.to_device(device)
    for autoencoder in model.autoencoders:
        autoencoder.train()
    return model


def select_arm_utility(true_u_cycle, arm):
    """Apply exactly the historical TRUE_U/TRUE_UNIFORM carrier semantics."""
    active_arm = protocol.validate_arm(arm)
    values = np.asarray(true_u_cycle)
    _require(values.shape == (protocol.N, protocol.ACTION_COUNT), "U_cycle shape mismatch")
    _require(np.isfinite(values).all() and np.all(values >= 0), "U_cycle values invalid")
    if active_arm == "UNIFORM":
        return np.ones_like(values, dtype=np.float64)
    return np.ascontiguousarray(values, dtype=np.float64)


def build_action_artifacts(*, views, split, arm, device, output_dir):
    """Build R2/R3 inputs from the frozen initial checkpoint, without GT."""
    active_arm = protocol.validate_arm(arm)
    target = Path(output_dir)
    _require(not target.exists(), "refusing to overwrite action inputs")
    target.mkdir(parents=True, exist_ok=False)
    model = _load_frozen_initial_model(device)
    latent_views = []
    q_local_views = []
    with torch.no_grad():
        for autoencoder, view in zip(model.autoencoders, views):
            tensor = torch.from_numpy(np.ascontiguousarray(view)).to(device)
            latent = autoencoder.encoder(tensor)
            posterior = autoencoder.clustering(latent)
            latent_views.append(np.ascontiguousarray(latent.detach().cpu().numpy()))
            q_local_views.append(np.ascontiguousarray(posterior.detach().cpu().numpy()))
    _, matrices, _, weights, _ = native_refresh_from_latents(
        latent_views,
        q_local_views,
        tuple(1.0 for _ in range(protocol.V)),
        n_clusters=protocol.K,
        random_state=protocol.TRAINING_SEED,
    )
    q_local = np.ascontiguousarray(np.stack(q_local_views, axis=1), dtype=np.float32)
    matrices = np.ascontiguousarray(matrices, dtype=np.float32)
    q_aligned = np.ascontiguousarray(np.stack([
        q_local[:, view_id, :] @ matrices[view_id].T
        for view_id in range(protocol.V)
    ], axis=1), dtype=np.float32)
    actions = build_directional_actions(protocol.V)
    cycle = compute_directional_cycle_utility(torch.from_numpy(q_aligned), actions)
    true_u = np.ascontiguousarray(cycle["U_cycle"].cpu().numpy(), dtype=np.float64)
    y_gen = np.ascontiguousarray(cycle["y_gen"].cpu().numpy(), dtype=np.int64)
    semantics = build_relation_semantics(y_gen, split, actions)
    pred_relation = np.ascontiguousarray(semantics.pred_relation, dtype=np.bool_)
    balance = np.ascontiguousarray(semantics.balance_weights, dtype=np.float64)
    selected_u = select_arm_utility(true_u, active_arm)
    expected_relation_shape = (protocol.N_U, protocol.L, protocol.ACTION_COUNT)
    _require(
        true_u.shape == (protocol.N, protocol.ACTION_COUNT)
        and pred_relation.shape == expected_relation_shape
        and balance.shape == expected_relation_shape,
        "R2/R3 transfer tensor shape mismatch",
    )

    utility_path = target / "utility.npz"
    semantic_path = target / "semantics.npz"
    utility_audit_path = target / "utility_audit.json"
    semantic_audit_path = target / "semantic_audit.json"
    np.savez(
        utility_path,
        U_cycle=selected_u,
        unlabeled_ids=np.ascontiguousarray(split.unlabeled_ids, dtype=np.int64),
    )
    np.savez(
        semantic_path,
        PredRelation_true=pred_relation,
        relation_balance_weights_true=balance,
        labeled_ids=np.ascontiguousarray(split.labeled_ids, dtype=np.int64),
        unlabeled_ids=np.ascontiguousarray(split.unlabeled_ids, dtype=np.int64),
    )
    utility_hash = file_sha256(utility_path)
    semantic_hash = file_sha256(semantic_path)
    action_record = {
        "schema": "paper-msrc-p0-action-provenance-v1",
        "dataset": protocol.DATASET,
        "arm": active_arm,
        "initial_model_sha256": protocol.INITIAL_MODEL_SHA256,
        "q_local_logical_sha256": ndarray_sha256(q_local),
        "q_aligned_logical_sha256": ndarray_sha256(q_aligned),
        "M_v_logical_sha256": ndarray_sha256(matrices),
        "view_weights": [float(value) for value in weights],
        "true_u_cycle_logical_sha256": ndarray_sha256(true_u),
        "selected_u_cycle_logical_sha256": ndarray_sha256(selected_u),
        "y_gen_logical_sha256": ndarray_sha256(y_gen),
        "pred_relation_logical_sha256": ndarray_sha256(pred_relation),
        "balance_logical_sha256": ndarray_sha256(balance),
        "pred_relation_shape": list(pred_relation.shape),
        "balance_shape": list(balance.shape),
        "U_cycle_detached": True,
        "PredRelation_detached": True,
        "balance_detached": True,
        "full_gt_loaded": False,
        "uniform_changes_pred_relation": False,
    }
    _write_json(utility_audit_path, {
        **action_record,
        "artifact_path": str(utility_path),
        "artifact_sha256": utility_hash,
        "utility_semantics": (
            "ones_like(U_cycle)" if active_arm == "UNIFORM" else "true U_cycle"
        ),
    })
    _write_json(semantic_audit_path, {
        **action_record,
        "artifact_path": str(semantic_path),
        "artifact_sha256": semantic_hash,
        "semantic_target": "true PredRelation + true balance",
    })
    return {
        "utility": utility_path,
        "utility_audit": utility_audit_path,
        "semantic": semantic_path,
        "semantic_audit": semantic_audit_path,
        "record": action_record,
    }


def _provenance(input_paths, action_paths, output_dir):
    checkpoint_hashes = _checkpoint_identity()
    bound = (
        input_paths["feature"], input_paths["feature_audit"],
        input_paths["split"], input_paths["split_audit"],
        action_paths["utility"], action_paths["utility_audit"],
        action_paths["semantic"], action_paths["semantic_audit"],
        protocol.CHECKPOINT_AUDIT, *protocol.CHECKPOINT_PATHS,
    )
    expected = tuple((path, file_sha256(path)) for path in bound)
    _require(
        tuple(digest for path, digest in expected if path in protocol.CHECKPOINT_PATHS)
        == checkpoint_hashes,
        "checkpoint expected-hash binding mismatch",
    )
    return ProvenanceConfig(
        feature_artifact=input_paths["feature"],
        feature_audit=input_paths["feature_audit"],
        sparse_split_artifact=input_paths["split"],
        sparse_split_audit=input_paths["split_audit"],
        utility_artifact=action_paths["utility"],
        utility_audit=action_paths["utility_audit"],
        semantic_artifact=action_paths["semantic"],
        semantic_audit=action_paths["semantic_audit"],
        checkpoint_paths=protocol.CHECKPOINT_PATHS,
        checkpoint_audit=protocol.CHECKPOINT_AUDIT,
        output_root=Path(output_dir),
        strict_replay=True,
        expected_file_sha256=expected,
        expected_initial_model_sha256=protocol.INITIAL_MODEL_SHA256,
    )


def _build_base_native_optimizers(model, learning_rate):
    """Historical BASE topology: native optimizers exist; semantic ones do not."""
    optimizers = tuple(torch.optim.Adam(
        autoencoder.parameters(),
        lr=float(learning_rate), betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0, amsgrad=False, maximize=False, foreach=None,
        capturable=False, differentiable=False, fused=None,
    ) for autoencoder in model.autoencoders)
    _require(len(optimizers) == protocol.V, "BASE native optimizer count mismatch")
    return optimizers


def _run_base_pre_gt(runtime, provenance):
    """Orchestrate authoritative BASE using only frozen native core primitives."""
    _require(not provenance.output_root.exists(), "refusing to overwrite an existing run")
    runtime_entrypoint._verify_expected_files(provenance)
    views_numpy, sample_ids, contract, feature_hash = runtime_entrypoint._load_feature_only(
        runtime, provenance
    )
    split, split_hash = runtime_entrypoint._load_sparse_split(
        runtime, provenance, sample_ids, contract.n_clusters
    )
    _, _, _, utility_hash, semantic_hash = runtime_entrypoint._load_action_inputs(
        provenance, split, contract.n_samples
    )
    device, determinism = runtime_entrypoint._configure_determinism(
        runtime.training_seed, runtime.device
    )
    model, initial_hash, checkpoint_hashes, checkpoint_audit_hash = (
        runtime_entrypoint._load_initial_model(runtime, provenance, contract, device)
    )
    views = tuple(torch.from_numpy(view) for view in views_numpy)
    native_optimizers = _build_base_native_optimizers(model, runtime.learning_rate)
    orders = precompute_training_orders(sample_ids, runtime.epochs, runtime.training_seed)
    native_state = initial_native_target_state(protocol.V)
    epochs = []
    for epoch in range(runtime.epochs):
        native_state, refresh = refresh_native_state_if_due(
            model, views, epoch=epoch, refresh_interval=runtime.refresh_interval,
            previous_state=native_state, seed=runtime.training_seed, device=device,
        )
        phase_b = run_native_consolidation_phase(
            model, native_optimizers, views, sample_ids, native_state,
            orders.native_orders[epoch], runtime.batch_size,
            runtime.native_lambda1, device,
        )
        epochs.append({
            "epoch": epoch,
            "event_sequence": ["PHASE_A_SKIPPED", "REFRESH" if refresh.executed else "NO_REFRESH", "PHASE_B"],
            "phase_a": {"executed": False, "reason": "BASE has no R4 objective"},
            "target_refresh": runtime_entrypoint._jsonable(refresh),
            "phase_b": runtime_entrypoint._jsonable(phase_b),
        })
    final_hash = runtime_entrypoint._model_sha256(model)
    predictions, q_local, q_aligned, matrix, refresh_audit = (
        runtime_entrypoint._final_prediction_state(
            model, views, native_state, runtime.training_seed, device
        )
    )
    _require(runtime_entrypoint._model_sha256(model) == final_hash, "final refresh mutated model")
    input_hashes = (
        feature_hash, split_hash, utility_hash, semantic_hash,
        checkpoint_audit_hash, *checkpoint_hashes,
    )
    arrays = {
        "sample_ids": np.ascontiguousarray(sample_ids, dtype=np.int64),
        "final_predictions": predictions,
        "labeled_ids": np.ascontiguousarray(split.labeled_ids, dtype=np.int64),
        "q_local": q_local,
        "q_aligned": q_aligned,
        "M_v": matrix,
        "input_sha256": np.asarray(input_hashes, dtype="<U64"),
        "initial_model_sha256": np.asarray(initial_hash, dtype="<U64"),
        "final_model_sha256": np.asarray(final_hash, dtype="<U64"),
    }
    audit = {
        "schema": "release-core-pre-gt-audit-v1",
        "scientific_role": "paper BASE experiment/runtime orchestration",
        "paper_arm": "BASE",
        "scientific_config": runtime_entrypoint._jsonable(runtime),
        "strict_replay": provenance.strict_replay,
        "new_scientific_mechanism": False,
        "new_information_utility": False,
        "new_training_loss": False,
        "base_semantics": "no R4 objective; native Phase B only",
        "semantic_optimizer_created": False,
        "phase_a_executed": False,
        "phase_b_executed": True,
        "epochs": epochs,
        "input_identity": {
            "feature_sha256": feature_hash, "sparse_split_sha256": split_hash,
            "utility_sha256": utility_hash, "semantic_sha256": semantic_hash,
            "checkpoint_audit_sha256": checkpoint_audit_hash,
            "checkpoint_sha256": list(checkpoint_hashes),
            "sparse_split_logical_sha256": split.digest,
        },
        "initial_model_sha256": initial_hash,
        "final_model_sha256": final_hash,
        "final_refresh": refresh_audit,
        "prediction_logical_sha256": runtime_entrypoint._payload_sha256(predictions),
        "sample_id_logical_sha256": runtime_entrypoint._payload_sha256(sample_ids),
        "clean_source_sha256": runtime_entrypoint._source_hashes(),
        "determinism": determinism,
        "gt_firewall": {
            "full_gt_argument_accepted": False,
            "full_gt_loaded": False,
            "metrics_computed": False,
            "full_gt_present_in_bundle": False,
        },
    }
    return runtime_entrypoint._persist_pre_gt(provenance.output_root, arrays, audit)


def _release_source_hashes():
    root = protocol.REPOSITORY_ROOT
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted((root / "release_core").rglob("*.py"))
    }


def _git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(protocol.REPOSITORY_ROOT), text=True
    ).strip()


def run_arm(
    *, arm, input_dir, output_dir, device, training_seed, epochs,
    batch_size, learning_rate, refresh_interval,
):
    """Run one pre-GT arm and create its immutable manifest; no evaluation."""
    active_arm = protocol.validate_arm(arm)
    frozen = protocol.validate_frozen_training_values(
        training_seed=training_seed, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, refresh_interval=refresh_interval,
    )
    output = Path(output_dir)
    _require(not output.exists(), "refusing to overwrite an existing arm output")
    input_paths, views, _, split = load_materialized_inputs(input_dir)
    _checkpoint_identity()
    runtime = RuntimeConfig(
        dataset=protocol.DATASET,
        training_seed=frozen.training_seed,
        epochs=frozen.epochs,
        batch_size=frozen.batch_size,
        learning_rate=frozen.learning_rate,
        native_lambda1=protocol.NATIVE_LAMBDA1,
        refresh_interval=frozen.refresh_interval,
        label_seed=frozen.label_seed,
        labels_per_class=frozen.labels_per_class,
        device=device,
    )
    with tempfile.TemporaryDirectory(prefix="msrc_p0_action_") as temporary:
        action_paths = build_action_artifacts(
            views=views, split=split, arm=active_arm, device=device,
            output_dir=Path(temporary) / "inputs",
        )
        provenance = _provenance(input_paths, action_paths, output)
        if active_arm == "BASE":
            sealed = _run_base_pre_gt(runtime, provenance)
        else:
            sealed = run_pre_gt(runtime, provenance)
        action_record = action_paths["record"]
        manifest = {
            "schema": "paper-msrc-p0-run-manifest-v1",
            "git_commit": _git_commit(),
            "dataset": protocol.DATASET,
            "dataset_path": str(protocol.DATASET_PATH),
            "dataset_sha256": protocol.DATASET_SHA256,
            "arm": active_arm,
            "arm_semantics": protocol.ARM_SEMANTICS[active_arm],
            "training_seed": frozen.training_seed,
            "label_seed": frozen.label_seed,
            "weak_quality_seed": frozen.weak_quality_seed,
            "snr_db": frozen.snr_db,
            "epochs": frozen.epochs,
            "batch_size": frozen.batch_size,
            "learning_rate": frozen.learning_rate,
            "refresh_interval": frozen.refresh_interval,
            "input_artifact_sha256": {
                name: file_sha256(path) for name, path in input_paths.items()
            },
            "checkpoint_paths": [str(path) for path in protocol.CHECKPOINT_PATHS],
            "checkpoint_sha256": list(protocol.CHECKPOINT_SHA256),
            "initial_model_sha256": protocol.INITIAL_MODEL_SHA256,
            "checkpoint_source_condition": protocol.CHECKPOINT_SOURCE_CONDITION,
            "release_core_source_sha256": _release_source_hashes(),
            "action_provenance": action_record,
            "pre_gt_bundle_sha256": file_sha256(sealed.bundle),
            "pre_gt_audit_sha256": file_sha256(sealed.audit),
            "pre_gt_seal_sha256": file_sha256(sealed.seal),
            "GT-loaded-before-seal": False,
            "formal_result": False,
            "auto_tuning": False,
        }
        _write_json(output / "run_manifest.json", manifest)
    return sealed


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=protocol.ARMS)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--refresh-interval", type=int, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sealed = run_arm(
        arm=args.arm, input_dir=args.input_dir, output_dir=args.output_dir,
        device=args.device, training_seed=args.training_seed, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        refresh_interval=args.refresh_interval,
    )
    print(json.dumps({
        "arm": args.arm, "pre_gt_bundle": str(sealed.bundle),
        "pre_gt_audit": str(sealed.audit), "pre_gt_seal": str(sealed.seal),
        "GT-loaded-before-seal": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
