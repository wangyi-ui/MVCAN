"""Create the label-free P0-A2 current-condition native initialization."""

import argparse
import hashlib
import json
import os
import random
import shutil
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
import torch.nn.functional as F

from release_core.backbone import MultiViewBackbone
from release_core.backbone.clustering import (
    initialize_kmeans_centers,
    native_refresh_from_latents,
)
from release_core.backbone.native_objective import native_objective
from release_core.config import get_native_config
import release_core.runtime.entrypoint as runtime_entrypoint

from . import msrc_p0_a2_protocol as protocol


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _release_source_hashes():
    root = protocol.REPOSITORY_ROOT
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted((root / "release_core").rglob("*.py"))
    }


def load_current_condition_features(input_dir):
    """Load only the already sealed feature artifact; never open split/GT files."""
    root = Path(input_dir)
    _require(
        root.resolve() == protocol.P0_A1_INPUT_DIR.resolve(),
        "P0-A2 must reuse the exact P0-A1 input directory",
    )
    feature_path = root / "msrc_trainable_features.npz"
    feature_audit_path = root / "feature_audit.json"
    materialization_seal_path = root / "materialization_seal.json"
    _require(
        feature_path.is_file()
        and feature_audit_path.is_file()
        and materialization_seal_path.is_file(),
        "sealed current-condition feature input is missing",
    )
    with materialization_seal_path.open("r", encoding="utf-8") as stream:
        seal = json.load(stream)
    with feature_audit_path.open("r", encoding="utf-8") as stream:
        audit = json.load(stream)
    feature_hash = file_sha256(feature_path)
    _require(
        seal.get("seal_valid") is True
        and seal.get("dataset") == protocol.DATASET
        and seal.get("dataset_sha256") == protocol.DATASET_SHA256
        and seal.get("full_gt_persisted") is False
        and seal.get("full_gt_available_to_training_runner") is False
        and feature_hash == protocol.FEATURE_ARTIFACT_SHA256
        and feature_hash == seal.get("feature_artifact_sha256")
        and file_sha256(feature_audit_path) == seal.get("feature_audit_sha256")
        and audit.get("artifact_sha256") == feature_hash
        and audit.get("dataset_sha256") == protocol.DATASET_SHA256
        and audit.get("mask_logical_sha256") == protocol.CORRUPTION_MASK_SHA256
        and audit.get("trainable_artifact_forbidden_fields_absent") is True
        and audit.get("full_gt_persisted") is False,
        "current-condition feature provenance mismatch",
    )
    with np.load(feature_path, allow_pickle=False) as archive:
        _require(tuple(archive.files) == protocol.FEATURE_FIELDS,
                 "feature whitelist mismatch")
        _require(
            {name.lower() for name in archive.files}.isdisjoint(
                protocol.FORBIDDEN_FEATURE_FIELDS
            ),
            "feature artifact contains a GT field",
        )
        views = tuple(
            np.ascontiguousarray(archive[name])
            for name in protocol.FEATURE_FIELDS[:-1]
        )
        sample_ids = np.ascontiguousarray(archive["sample_ids"])
    _require(
        tuple(view.shape for view in views)
        == tuple((protocol.N, size) for size in protocol.VIEW_DIMS)
        and all(view.dtype == np.float32 for view in views)
        and sample_ids.dtype == np.int64
        and np.array_equal(sample_ids, np.arange(protocol.N, dtype=np.int64)),
        "current-condition feature tensor contract mismatch",
    )
    return {
        "views": views,
        "sample_ids": sample_ids,
        "feature_path": feature_path,
        "feature_audit_path": feature_audit_path,
        "materialization_seal_path": materialization_seal_path,
        "feature_sha256": feature_hash,
        "feature_audit_sha256": file_sha256(feature_audit_path),
        "materialization_seal_sha256": file_sha256(materialization_seal_path),
    }


def _configure_determinism(seed, device):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    selected = torch.device(device)
    if selected.type == "cuda":
        _require(torch.cuda.is_available(), "requested CUDA is unavailable")
        torch.cuda.set_device(selected)
    return selected


def _build_native_model(device):
    config = get_native_config(protocol.DATASET)
    expected = protocol.FrozenInitializationConfig()
    _require(
        config["training"] == {
            "seed": expected.seed,
            "batch_size": expected.batch_size,
            "init_epoch": expected.init_epochs,
            "T_1": 2,
            "T_2": expected.refresh_interval,
            "epoch": expected.native_config_epoch,
            "lr": expected.learning_rate,
            "lambda1": expected.native_lambda1,
        },
        "release native MSRC configuration mismatch",
    )
    model = MultiViewBackbone(
        config,
        view_num=protocol.V,
        view_size=protocol.VIEW_DIMS,
        n_clusters=protocol.K,
        seed=expected.seed,
    )
    model.to_device(device)
    for autoencoder in model.autoencoders:
        autoencoder.train()
    return model, config


def _build_optimizers(model, learning_rate):
    """Match the historical one-Adam-per-view topology and defaults."""
    return tuple(
        torch.optim.Adam(autoencoder.parameters(), lr=float(learning_rate))
        for autoencoder in model.autoencoders
    )


@torch.no_grad()
def _refresh_native_state(model, full_views, view_weights, seed, device):
    latent_views = []
    q_local_views = []
    for autoencoder, full_view in zip(model.autoencoders, full_views):
        latent = autoencoder.encoder(full_view)
        posterior = autoencoder.clustering(latent)
        latent_views.append(np.ascontiguousarray(latent.detach().cpu().numpy()))
        q_local_views.append(np.ascontiguousarray(posterior.detach().cpu().numpy()))
    p_numpy, matrices, _, weights, _ = native_refresh_from_latents(
        latent_views,
        q_local_views,
        view_weights,
        n_clusters=protocol.K,
        random_state=seed,
    )
    p_all = torch.from_numpy(np.asarray(p_numpy)).float().to(device).detach()
    matches = torch.from_numpy(np.asarray(matrices)).float().to(device).detach()
    _require(
        p_all.shape == (protocol.N, protocol.K)
        and matches.shape == (protocol.V, protocol.K, protocol.K)
        and bool(torch.isfinite(p_all).all())
        and bool(torch.isfinite(matches).all()),
        "native refresh returned an invalid state",
    )
    return p_all, matches, tuple(float(value) for value in weights)


def _run_historical_native_initialization(views, device, frozen):
    """Replay G0-B0 preparation with release numerical primitives."""
    selected_device = _configure_determinism(frozen.seed, device)
    model, config = _build_native_model(selected_device)
    full_views = tuple(
        torch.from_numpy(np.ascontiguousarray(view)).to(selected_device)
        for view in views
    )
    dataset = torch.utils.data.TensorDataset(*full_views)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(frozen.seed)
    optimizers = _build_optimizers(model, frozen.learning_rate)

    for _ in range(frozen.init_epochs):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=frozen.batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        for batch in loader:
            for view_id, autoencoder in enumerate(model.autoencoders):
                latent = autoencoder.encoder(batch[view_id])
                loss = F.mse_loss(autoencoder.decoder(latent), batch[view_id])
                optimizers[view_id].zero_grad(set_to_none=True)
                loss.backward()
                _require(bool(torch.isfinite(loss)), "AE initialization loss is nonfinite")
                optimizers[view_id].step()

    with torch.no_grad():
        for autoencoder, full_view in zip(model.autoencoders, full_views):
            latent = autoencoder.encoder(full_view).detach().cpu().numpy()
            _, centers = initialize_kmeans_centers(
                latent, protocol.K, frozen.kmeans_random_state
            )
            center_tensor = torch.as_tensor(
                centers,
                dtype=autoencoder._cluster_layer.dtype,
                device=selected_device,
            )
            _require(bool(torch.isfinite(center_tensor).all()),
                     "KMeans center initialization is nonfinite")
            autoencoder._cluster_layer.data = center_tensor

    view_weights = tuple(1.0 for _ in range(protocol.V))
    p_all = None
    matches = None
    refresh_epochs = []
    for epoch in range(frozen.native_loop_iterations):
        if epoch % frozen.refresh_interval == 0:
            p_all, matches, view_weights = _refresh_native_state(
                model, full_views, view_weights, frozen.seed, selected_device
            )
            refresh_epochs.append(epoch)
        native_dataset = torch.utils.data.TensorDataset(*full_views, p_all)
        loader = torch.utils.data.DataLoader(
            native_dataset,
            batch_size=frozen.batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        for batch in loader:
            p_batch = batch[-1].detach()
            for view_id, autoencoder in enumerate(model.autoencoders):
                reconstruction, _, q_local = autoencoder(batch[view_id])
                p_local = p_batch @ matches[view_id].detach()
                loss, _, _ = native_objective(
                    reconstruction,
                    batch[view_id],
                    q_local,
                    p_local.detach(),
                    frozen.native_lambda1,
                )
                optimizers[view_id].zero_grad(set_to_none=True)
                loss.backward()
                _require(bool(torch.isfinite(loss)), "native loss is nonfinite")
                optimizers[view_id].step()

    _require(
        tuple(refresh_epochs) == protocol.NATIVE_REFRESH_EPOCHS
        and all(
            bool(torch.isfinite(parameter).all())
            for autoencoder in model.autoencoders
            for parameter in autoencoder.parameters()
        ),
        "historical native initialization completion mismatch",
    )
    return model, {
        "native_config": config,
        "initialization_epochs_completed": frozen.init_epochs,
        "native_config_epoch": frozen.native_config_epoch,
        "native_loop_iterations_completed": frozen.native_loop_iterations,
        "native_epoch_index_first": 0,
        "native_epoch_index_last": frozen.native_config_epoch,
        "native_refresh_epochs": refresh_epochs,
        "native_refresh_count": len(refresh_epochs),
        "shared_dataloader_generator_across_all_loops": True,
        "historical_per_view_backward_and_step": True,
        "release_native_objective_used": True,
        "release_native_refresh_used": True,
        "release_kmeans_center_primitive_used": True,
    }


def _checkpoint_roundtrip(model, checkpoint_paths):
    for autoencoder, path in zip(model.autoencoders, checkpoint_paths):
        state = {
            name: value.detach().cpu()
            for name, value in autoencoder.state_dict().items()
        }
        torch.save(state, path)
    expected_hash = runtime_entrypoint._model_sha256(model)
    replay = MultiViewBackbone(
        get_native_config(protocol.DATASET),
        view_num=protocol.V,
        view_size=protocol.VIEW_DIMS,
        n_clusters=protocol.K,
        seed=protocol.TRAINING_SEED,
    )
    states = tuple(
        runtime_entrypoint._load_torch_state(path) for path in checkpoint_paths
    )
    results = replay.load_state_dicts(states, strict=True)
    _require(all(not result.missing_keys and not result.unexpected_keys for result in results),
             "strict checkpoint load mismatch")
    actual_hash = runtime_entrypoint._model_sha256(replay)
    _require(actual_hash == expected_hash, "checkpoint roundtrip model hash mismatch")
    return expected_hash


def _persist_initialization(model, target, feature_record, execution_audit):
    """Persist one sealed five-view checkpoint set after the full replay."""
    output = Path(target)
    _require(output.resolve() == protocol.INITIALIZATION_DIR.resolve(),
             "initialization output must use the frozen P0-A2 path")
    _require(not output.exists(), "refusing to overwrite initialization output")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".p0_a2_init_", dir=str(output.parent)))
    try:
        checkpoint_paths = tuple(
            staging / name for name in protocol.INITIALIZATION_CHECKPOINT_NAMES
        )
        model_hash = _checkpoint_roundtrip(model, checkpoint_paths)
        checkpoint_hashes = tuple(file_sha256(path) for path in checkpoint_paths)
        final_checkpoint_paths = tuple(
            output / name for name in protocol.INITIALIZATION_CHECKPOINT_NAMES
        )
        audit = {
            "schema": "paper-msrc-p0-a2-initialization-audit-v1",
            "dataset": protocol.DATASET,
            "source_condition": protocol.INITIALIZATION_SOURCE_CONDITION,
            "input_feature_path": str(feature_record["feature_path"]),
            "input_feature_sha256": feature_record["feature_sha256"],
            "dataset_sha256": protocol.DATASET_SHA256,
            "corruption_mask_logical_sha256": protocol.CORRUPTION_MASK_SHA256,
            "training_seed": protocol.TRAINING_SEED,
            "checkpoint_paths": [str(path) for path in final_checkpoint_paths],
            "checkpoint_sha256": list(checkpoint_hashes),
            "combined_model_sha256": model_hash,
            **execution_audit,
            "full_gt_loaded": False,
            "sparse_labels_used": False,
            "U_cycle_used": False,
            "relation_used": False,
            "scientific_metrics_computed": False,
            "auto_tuning": False,
        }
        audit_path = staging / protocol.INITIALIZATION_AUDIT_NAME
        _write_json(audit_path, audit)
        manifest = {
            "schema": "paper-msrc-p0-a2-initialization-manifest-v1",
            "dataset": protocol.DATASET,
            "source_condition": protocol.INITIALIZATION_SOURCE_CONDITION,
            "checkpoint_paths": [str(path) for path in final_checkpoint_paths],
            "checkpoint_sha256": list(checkpoint_hashes),
            "combined_model_sha256": model_hash,
            "input_feature_sha256": feature_record["feature_sha256"],
            "input_feature_audit_sha256": feature_record["feature_audit_sha256"],
            "input_materialization_seal_sha256": (
                feature_record["materialization_seal_sha256"]
            ),
            "initialization_audit_sha256": file_sha256(audit_path),
            "release_core_source_sha256": _release_source_hashes(),
            "full_gt_loaded": False,
            "sparse_labels_used": False,
            "U_cycle_used": False,
            "relation_used": False,
        }
        manifest_path = staging / protocol.INITIALIZATION_MANIFEST_NAME
        _write_json(manifest_path, manifest)
        seal = {
            "schema": "paper-msrc-p0-a2-initialization-seal-v1",
            "checkpoint_sha256": list(checkpoint_hashes),
            "combined_model_sha256": model_hash,
            "initialization_audit_sha256": file_sha256(audit_path),
            "initialization_manifest_sha256": file_sha256(manifest_path),
            "full_gt_loaded": False,
            "sparse_labels_used": False,
            "seal_valid": True,
        }
        _write_json(staging / protocol.INITIALIZATION_SEAL_NAME, seal)
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return audit


def verify_initialization(init_dir=protocol.INITIALIZATION_DIR):
    """Strictly verify and load the one shared current-condition state."""
    root = Path(init_dir)
    _require(root.resolve() == protocol.INITIALIZATION_DIR.resolve(),
             "initialization path must be the frozen shared path")
    checkpoints = protocol.initialization_checkpoint_paths(root)
    audit_path = root / protocol.INITIALIZATION_AUDIT_NAME
    manifest_path = root / protocol.INITIALIZATION_MANIFEST_NAME
    seal_path = root / protocol.INITIALIZATION_SEAL_NAME
    _require(all(path.is_file() for path in (*checkpoints, audit_path, manifest_path, seal_path)),
             "initialization artifact is incomplete")
    with audit_path.open("r", encoding="utf-8") as stream:
        audit = json.load(stream)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    with seal_path.open("r", encoding="utf-8") as stream:
        seal = json.load(stream)
    checkpoint_hashes = tuple(file_sha256(path) for path in checkpoints)
    _require(
        seal.get("seal_valid") is True
        and seal.get("full_gt_loaded") is False
        and seal.get("sparse_labels_used") is False
        and checkpoint_hashes == tuple(seal.get("checkpoint_sha256", ()))
        and checkpoint_hashes == tuple(audit.get("checkpoint_sha256", ()))
        and checkpoint_hashes == tuple(manifest.get("checkpoint_sha256", ()))
        and file_sha256(audit_path) == seal.get("initialization_audit_sha256")
        and file_sha256(manifest_path) == seal.get("initialization_manifest_sha256")
        and audit.get("input_feature_sha256") == protocol.FEATURE_ARTIFACT_SHA256
        and audit.get("corruption_mask_logical_sha256")
        == protocol.CORRUPTION_MASK_SHA256
        and audit.get("native_loop_iterations_completed")
        == protocol.NATIVE_LOOP_ITERATIONS
        and audit.get("native_refresh_count") == protocol.NATIVE_REFRESH_COUNT,
        "initialization seal/provenance mismatch",
    )
    _require(
        audit.get("source_condition") == protocol.INITIALIZATION_SOURCE_CONDITION
        and manifest.get("source_condition")
        == protocol.INITIALIZATION_SOURCE_CONDITION
        and manifest.get("input_feature_sha256")
        == protocol.FEATURE_ARTIFACT_SHA256
        and manifest.get("release_core_source_sha256")
        == _release_source_hashes()
        and manifest.get("full_gt_loaded") is False
        and manifest.get("sparse_labels_used") is False
        and manifest.get("U_cycle_used") is False
        and manifest.get("relation_used") is False,
        "initialization manifest/source identity mismatch",
    )
    model = MultiViewBackbone(
        get_native_config(protocol.DATASET), protocol.V, protocol.VIEW_DIMS,
        n_clusters=protocol.K, seed=protocol.TRAINING_SEED,
    )
    states = tuple(runtime_entrypoint._load_torch_state(path) for path in checkpoints)
    results = model.load_state_dicts(states, strict=True)
    _require(all(not result.missing_keys and not result.unexpected_keys for result in results),
             "strict initialization load mismatch")
    model_hash = runtime_entrypoint._model_sha256(model)
    _require(
        model_hash == seal.get("combined_model_sha256")
        and model_hash == audit.get("combined_model_sha256")
        and model_hash == manifest.get("combined_model_sha256"),
        "combined initialization model hash mismatch",
    )
    return {
        "model": model,
        "model_sha256": model_hash,
        "checkpoint_paths": checkpoints,
        "checkpoint_sha256": checkpoint_hashes,
        "audit_path": audit_path,
        "manifest_path": manifest_path,
        "seal_path": seal_path,
        "audit": audit,
        "manifest": manifest,
    }


def materialize(
    *, input_dir, output_dir, device, seed, init_epochs,
    native_config_epoch, refresh_interval, learning_rate,
    native_lambda1, batch_size,
):
    frozen = protocol.validate_initialization_values(
        seed=seed,
        init_epochs=init_epochs,
        native_config_epoch=native_config_epoch,
        refresh_interval=refresh_interval,
        learning_rate=learning_rate,
        native_lambda1=native_lambda1,
        batch_size=batch_size,
    )
    _require(not Path(output_dir).exists(),
             "refusing to overwrite initialization output")
    feature_record = load_current_condition_features(input_dir)
    model, execution_audit = _run_historical_native_initialization(
        feature_record["views"], device, frozen
    )
    return _persist_initialization(model, output_dir, feature_record, execution_audit)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--init-epochs", type=int, required=True)
    parser.add_argument("--native-epoch", type=int, required=True)
    parser.add_argument("--refresh-interval", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--native-lambda1", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = materialize(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        init_epochs=args.init_epochs,
        native_config_epoch=args.native_epoch,
        refresh_interval=args.refresh_interval,
        learning_rate=args.learning_rate,
        native_lambda1=args.native_lambda1,
        batch_size=args.batch_size,
    )
    print(json.dumps({
        "combined_model_sha256": result["combined_model_sha256"],
        "initialization_dir": str(Path(args.output_dir)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
