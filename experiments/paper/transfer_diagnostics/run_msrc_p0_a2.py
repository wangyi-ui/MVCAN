"""Run one P0-A2 arm from the shared same-condition initialization."""

import argparse
import json
import os
import tempfile
from pathlib import Path


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _thread_name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_name] = "1"

import numpy as np
import torch

from experiments.paper.transfer_audit.input_artifacts import (
    file_sha256,
    load_materialized_inputs,
)
from experiments.paper.transfer_audit import run_msrc_p0_transfer as p0_runner
from release_core.backbone.clustering import native_refresh_from_latents
from release_core.data.weak_quality import ndarray_sha256
from release_core.runtime import ProvenanceConfig, RuntimeConfig, run_pre_gt
from release_core.semantics import build_relation_semantics
from release_core.utility import build_directional_actions, compute_directional_cycle_utility

from . import msrc_p0_a2_protocol as protocol
from .materialize_msrc_current_condition_init import verify_initialization


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def select_arm_utility(true_u_cycle, arm):
    """Reuse the frozen P0 TRUE_U/TRUE_UNIFORM selection semantics."""
    return p0_runner.select_arm_utility(true_u_cycle, protocol.validate_arm(arm))


def build_action_artifacts(
    *, views, split, arm, device, output_dir, initialization,
):
    """Build frozen R2/R3 inputs from the shared current-condition state."""
    active_arm = protocol.validate_arm(arm)
    target = Path(output_dir)
    _require(not target.exists(), "refusing to overwrite action inputs")
    target.mkdir(parents=True, exist_ok=False)
    model = initialization["model"]
    model.to_device(device)
    for autoencoder in model.autoencoders:
        autoencoder.train()
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
    _require(
        true_u.shape == (protocol.N, 20)
        and pred_relation.shape == (196, 14, 20)
        and balance.shape == (196, 14, 20),
        "P0-A2 R2/R3 tensor shape mismatch",
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
    record = {
        "schema": "paper-msrc-p0-a2-action-provenance-v1",
        "dataset": protocol.DATASET,
        "arm": active_arm,
        "initialization_source_condition": protocol.INITIALIZATION_SOURCE_CONDITION,
        "initial_model_sha256": initialization["model_sha256"],
        "initial_checkpoint_sha256": list(initialization["checkpoint_sha256"]),
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
        "U_cycle_frozen_before_closed_loop": True,
        "U_cycle_detached": True,
        "PredRelation_detached": True,
        "balance_detached": True,
        "full_gt_loaded": False,
        "uniform_changes_pred_relation": False,
    }
    _write_json(utility_audit_path, {
        **record,
        "artifact_path": str(utility_path),
        "artifact_sha256": file_sha256(utility_path),
        "utility_semantics": (
            "ones_like(U_cycle)" if active_arm == "UNIFORM" else "true U_cycle"
        ),
    })
    _write_json(semantic_audit_path, {
        **record,
        "artifact_path": str(semantic_path),
        "artifact_sha256": file_sha256(semantic_path),
        "semantic_target": "true PredRelation + true balance",
    })
    return {
        "utility": utility_path,
        "utility_audit": utility_audit_path,
        "semantic": semantic_path,
        "semantic_audit": semantic_audit_path,
        "record": record,
    }


def _provenance(input_paths, action_paths, output_dir, initialization):
    bound = (
        input_paths["feature"], input_paths["feature_audit"],
        input_paths["split"], input_paths["split_audit"],
        action_paths["utility"], action_paths["utility_audit"],
        action_paths["semantic"], action_paths["semantic_audit"],
        initialization["audit_path"], initialization["manifest_path"],
        initialization["seal_path"], *initialization["checkpoint_paths"],
    )
    expected = tuple((path, file_sha256(path)) for path in bound)
    return ProvenanceConfig(
        feature_artifact=input_paths["feature"],
        feature_audit=input_paths["feature_audit"],
        sparse_split_artifact=input_paths["split"],
        sparse_split_audit=input_paths["split_audit"],
        utility_artifact=action_paths["utility"],
        utility_audit=action_paths["utility_audit"],
        semantic_artifact=action_paths["semantic"],
        semantic_audit=action_paths["semantic_audit"],
        checkpoint_paths=initialization["checkpoint_paths"],
        checkpoint_audit=initialization["audit_path"],
        output_root=Path(output_dir),
        strict_replay=True,
        expected_file_sha256=expected,
        expected_initial_model_sha256=initialization["model_sha256"],
    )


def run_arm(
    *, arm, input_dir, init_dir, output_dir, device, training_seed,
    epochs, batch_size, learning_rate, refresh_interval,
):
    """Run one pre-GT diagnostic arm; the sole change is initialization."""
    active_arm = protocol.validate_arm(arm)
    frozen = protocol.validate_arm_training_values(
        training_seed=training_seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        refresh_interval=refresh_interval,
    )
    output = Path(output_dir)
    _require(
        output.resolve() == protocol.default_output_dir(active_arm).resolve(),
        "P0-A2 arm output must use its frozen diagnostic path",
    )
    _require(not output.exists(), "refusing to overwrite an existing arm output")
    _require(
        Path(input_dir).resolve() == protocol.P0_A1_INPUT_DIR.resolve(),
        "P0-A2 must reuse the exact P0-A1 inputs",
    )
    input_paths, views, _, split = load_materialized_inputs(input_dir)
    initialization = verify_initialization(init_dir)
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
    with tempfile.TemporaryDirectory(prefix="msrc_p0_a2_action_") as temporary:
        action_paths = build_action_artifacts(
            views=views,
            split=split,
            arm=active_arm,
            device=device,
            output_dir=Path(temporary) / "inputs",
            initialization=initialization,
        )
        provenance = _provenance(
            input_paths, action_paths, output, initialization
        )
        if active_arm == "BASE":
            sealed = p0_runner._run_base_pre_gt(runtime, provenance)
        else:
            sealed = run_pre_gt(runtime, provenance)
        manifest = {
            "schema": "paper-msrc-p0-a2-run-manifest-v1",
            "git_commit": p0_runner._git_commit(),
            "dataset": protocol.DATASET,
            "dataset_path": str(protocol.DATASET_PATH),
            "dataset_sha256": protocol.DATASET_SHA256,
            "input_feature_sha256": protocol.FEATURE_ARTIFACT_SHA256,
            "corruption_mask_logical_sha256": protocol.CORRUPTION_MASK_SHA256,
            "sparse_split_logical_sha256": protocol.SPLIT_SHA256,
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
            "native_lambda1": protocol.NATIVE_LAMBDA1,
            "input_artifact_sha256": {
                name: file_sha256(path) for name, path in input_paths.items()
            },
            "initialization_dir": str(protocol.INITIALIZATION_DIR),
            "initialization_source_condition": protocol.INITIALIZATION_SOURCE_CONDITION,
            "checkpoint_paths": [
                str(path) for path in initialization["checkpoint_paths"]
            ],
            "checkpoint_sha256": list(initialization["checkpoint_sha256"]),
            "initial_model_sha256": initialization["model_sha256"],
            "initialization_audit_sha256": file_sha256(initialization["audit_path"]),
            "initialization_manifest_sha256": file_sha256(
                initialization["manifest_path"]
            ),
            "initialization_seal_sha256": file_sha256(initialization["seal_path"]),
            "release_core_source_sha256": p0_runner._release_source_hashes(),
            "action_provenance": action_paths["record"],
            "pre_gt_bundle_sha256": file_sha256(sealed.bundle),
            "pre_gt_audit_sha256": file_sha256(sealed.audit),
            "pre_gt_seal_sha256": file_sha256(sealed.seal),
            "sole_changed_experimental_variable": "initialization source condition",
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
    parser.add_argument("--init-dir", required=True)
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
        arm=args.arm,
        input_dir=args.input_dir,
        init_dir=args.init_dir,
        output_dir=args.output_dir,
        device=args.device,
        training_seed=args.training_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        refresh_interval=args.refresh_interval,
    )
    print(json.dumps({
        "arm": args.arm,
        "pre_gt_bundle": str(sealed.bundle),
        "pre_gt_audit": str(sealed.audit),
        "pre_gt_seal": str(sealed.seal),
        "GT-loaded-before-seal": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
