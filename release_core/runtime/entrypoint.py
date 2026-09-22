"""Pre-GT runtime orchestration over frozen clean release primitives."""

import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path


_TORCH_PREIMPORTED = "torch" in sys.modules
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import torch

from release_core.backbone import MultiViewBackbone
from release_core.backbone.clustering import native_refresh_from_latents
from release_core.config import get_native_config
from release_core.data.contracts import infer_dataset_contract
from release_core.data.sample_ids import validate_sample_ids
from release_core.semantics import SparseLabelSplit, validate_sparse_label_split
from release_core.training import NativeTargetState, run_alternating_training

from . import (
    ProvenanceConfig,
    RuntimeConfig,
    SealedPredictionPaths,
    _BUNDLE_KEYS,
    _SEAL_SCHEMA,
    _payload_sha256,
)


_DATASET_SPECS = {
    "Caltech-6V": (1400, 6, 7, (48, 40, 254, 1984, 512, 928)),
    "MSRC-v1": (210, 5, 7, (24, 576, 512, 256, 254)),
    "BDGP": (2500, 2, 5, (1750, 79)),
}
_FORBIDDEN_FEATURE_KEYS = {
    "y", "gt", "labels", "label", "full_gt", "ground_truth", "y_true",
}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, record):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError("audit or seal root must be an object")
    return value


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _contains_string(value, target):
    if isinstance(value, dict):
        return any(_contains_string(item, target) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_string(item, target) for item in value)
    return isinstance(value, str) and value.lower() == target.lower()


def _expected_hash(provenance, path):
    resolved = Path(path).resolve()
    matches = [
        digest for candidate, digest in provenance.expected_file_sha256
        if candidate.resolve() == resolved
    ]
    if len(matches) > 1:
        raise RuntimeError("duplicate expected hash entries")
    return matches[0] if matches else None


def _verify_file(path, provenance):
    target = Path(path)
    _require(target.is_file(), "required provenance file is missing: " + str(target))
    actual = _file_sha256(target)
    expected = _expected_hash(provenance, target)
    if provenance.strict_replay:
        _require(expected is not None, "strict replay file hash is missing")
    if expected is not None:
        _require(actual == expected, "strict replay file hash mismatch")
    return actual


def _verify_expected_files(provenance):
    for path, expected in provenance.expected_file_sha256:
        target = Path(path)
        _require(target.is_file(), "expected source or input file is missing")
        _require(_file_sha256(target) == expected,
                 "expected source or input hash mismatch")


def _verify_bound_artifact(artifact, audit, provenance):
    artifact_hash = _verify_file(artifact, provenance)
    _verify_file(audit, provenance)
    audit_record = _read_json(audit)
    _require(
        _contains_string(audit_record, artifact_hash),
        "provenance audit does not bind the artifact hash",
    )
    return artifact_hash, audit_record


def _view_keys(keys):
    key_set = set(keys)
    for key in key_set:
        if key.lower() in _FORBIDDEN_FEATURE_KEYS:
            raise RuntimeError("feature-only artifact contains a GT field")
    numbered = [key for key in key_set if key.startswith("view_")]
    if numbered:
        try:
            ordered = sorted(numbered, key=lambda key: int(key.split("_", 1)[1]))
        except ValueError as error:
            raise RuntimeError("invalid feature view key") from error
    else:
        numbered = [key for key in key_set if key.startswith("X") and key[1:].isdigit()]
        ordered = sorted(numbered, key=lambda key: int(key[1:]))
    _require(set(ordered) | {"sample_ids"} == key_set, "feature artifact whitelist mismatch")
    return ordered


def _load_feature_only(runtime, provenance):
    artifact_hash, audit = _verify_bound_artifact(
        provenance.feature_artifact, provenance.feature_audit, provenance
    )
    with np.load(provenance.feature_artifact, allow_pickle=False) as archive:
        keys = _view_keys(archive.files)
        _require("sample_ids" in archive.files, "feature artifact lacks sample IDs")
        views = tuple(np.ascontiguousarray(archive[key]) for key in keys)
        sample_ids = np.ascontiguousarray(archive["sample_ids"])
    expected_n, expected_v, cluster_count, expected_dims = _DATASET_SPECS[runtime.dataset]
    _require(len(views) == expected_v, "dataset view-count contract mismatch")
    for view in views:
        _require(view.dtype == np.float32, "feature views must be float32")
    contract = infer_dataset_contract(views, runtime.dataset, cluster_count, expected_v)
    validate_sample_ids(sample_ids, contract.n_samples, require_identity=True)
    if runtime.dataset == "Caltech-6V":
        _require(audit.get("dataset") in (None, runtime.dataset), "feature dataset mismatch")
        _require(
            audit.get("trainable_artifact_forbidden_fields_absent") is not False,
            "feature audit permits forbidden fields",
        )
    if provenance.strict_replay:
        _require(contract.n_samples == expected_n, "strict sample-count mismatch")
        _require(contract.view_dims == expected_dims, "strict feature-dimension mismatch")
        if runtime.dataset == "Caltech-6V":
            _require(
                audit.get("per_sample_corrupted_count_unique") == [3],
                "strict weak-quality corruption-count mismatch",
            )
            snr = audit.get("snr_audit", {})
            _require(float(snr.get("target_snr_db", float("nan"))) == 2.5,
                     "strict weak-quality SNR mismatch")
    return views, sample_ids, contract, artifact_hash


def _load_sparse_split(runtime, provenance, sample_ids, class_count):
    artifact_hash, _ = _verify_bound_artifact(
        provenance.sparse_split_artifact,
        provenance.sparse_split_audit,
        provenance,
    )
    required = {"sample_ids", "labeled_ids", "labeled_targets", "unlabeled_ids"}
    with np.load(provenance.sparse_split_artifact, allow_pickle=False) as archive:
        _require(set(archive.files) == required, "sparse split whitelist mismatch")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in required}
    _require(np.array_equal(arrays["sample_ids"], sample_ids),
             "sparse split sample IDs are misaligned")
    split = validate_sparse_label_split(SparseLabelSplit(
        sample_ids=arrays["sample_ids"],
        labeled_ids=arrays["labeled_ids"],
        labeled_targets=arrays["labeled_targets"],
        unlabeled_ids=arrays["unlabeled_ids"],
        class_count=class_count,
        labels_per_class=runtime.labels_per_class,
        label_seed=runtime.label_seed,
        dataset_name=runtime.dataset,
    ))
    if provenance.strict_replay and runtime.dataset == "Caltech-6V":
        _require(split.labeled_ids.size == 14 and split.unlabeled_ids.size == 1386,
                 "strict sparse-label size mismatch")
    return split, artifact_hash


def _copy_readonly(value, dtype=None):
    result = np.array(np.asarray(value, dtype=dtype), copy=True, order="C")
    result.setflags(write=False)
    return result


def _pick_array(archive, names, description):
    available = [name for name in names if name in archive.files]
    _require(len(available) == 1, description + " field mismatch")
    return np.ascontiguousarray(archive[available[0]])


def _load_action_inputs(provenance, split, sample_count):
    utility_hash, _ = _verify_bound_artifact(
        provenance.utility_artifact, provenance.utility_audit, provenance
    )
    semantic_hash, _ = _verify_bound_artifact(
        provenance.semantic_artifact, provenance.semantic_audit, provenance
    )
    with np.load(provenance.utility_artifact, allow_pickle=False) as archive:
        cycle = _pick_array(archive, ("u_cycle", "U_cycle"), "utility")
        utility_unlabeled = np.ascontiguousarray(archive["unlabeled_ids"])
    with np.load(provenance.semantic_artifact, allow_pickle=False) as archive:
        relation = _pick_array(
            archive, ("pred_relation", "PredRelation_true"), "relation"
        )
        balance = _pick_array(
            archive, ("balance", "relation_balance_weights_true"), "balance"
        )
        semantic_labeled = np.ascontiguousarray(archive["labeled_ids"])
        semantic_unlabeled = np.ascontiguousarray(archive["unlabeled_ids"])
    _require(utility_unlabeled.dtype == np.int64, "utility IDs must be int64")
    _require(np.array_equal(utility_unlabeled, split.unlabeled_ids),
             "utility query IDs are misaligned")
    _require(np.array_equal(semantic_labeled, split.labeled_ids),
             "relation anchor IDs are misaligned")
    _require(np.array_equal(semantic_unlabeled, split.unlabeled_ids),
             "relation query IDs are misaligned")
    _require(cycle.dtype == np.float64 and cycle.ndim == 2,
             "utility must be float64 [Nu,S] or [N,S]")
    _require(cycle.shape[0] in (sample_count, split.unlabeled_ids.size),
             "utility sample axis mismatch")
    action_count = cycle.shape[1]
    expected_shape = (split.unlabeled_ids.size, split.labeled_ids.size, action_count)
    _require(relation.dtype == np.bool_ and relation.shape == expected_shape,
             "relation must be bool [Nu,L,S]")
    _require(balance.dtype == np.float64 and balance.shape == expected_shape,
             "balance must be float64 [Nu,L,S]")
    _require(np.isfinite(cycle).all() and np.all(cycle >= 0.0),
             "utility must be finite and nonnegative")
    _require(np.isfinite(balance).all() and np.all(balance > 0.0),
             "balance must be finite and positive")
    return (
        _copy_readonly(cycle),
        _copy_readonly(relation, np.bool_),
        _copy_readonly(balance, np.float64),
        utility_hash,
        semantic_hash,
    )


def _configure_determinism(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = torch.device(device)
    if selected.type == "cuda":
        _require(torch.cuda.is_available(), "requested CUDA is unavailable")
        torch.cuda.set_device(selected)
    return selected, {
        "seed": int(seed),
        "torch_preimported_before_runtime_module": _TORCH_PREIMPORTED,
        "environment_applied_before_runtime_torch_import": not _TORCH_PREIMPORTED,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "thread_environment": {
            name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "pythonhashseed_added": False,
    }


def _load_torch_state(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and set(state) == {"state_dict"}:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint must contain a state dictionary")
    return state


def _model_sha256(model):
    digest = hashlib.sha256()
    for view_index, state in enumerate(model.state_dicts()):
        digest.update(str(view_index).encode("ascii") + b"\0")
        for name in sorted(state):
            value = state[name].detach().cpu().contiguous()
            array = value.numpy()
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(array.dtype.str.encode("ascii") + b"\0")
            digest.update(str(tuple(array.shape)).encode("ascii") + b"\0")
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_initial_model(runtime, provenance, contract, device):
    _require(len(provenance.checkpoint_paths) == contract.n_views,
             "checkpoint count must equal view count")
    checkpoint_hashes = []
    audit_hash = _verify_file(provenance.checkpoint_audit, provenance)
    checkpoint_audit = _read_json(provenance.checkpoint_audit)
    for path in provenance.checkpoint_paths:
        digest = _verify_file(path, provenance)
        _require(_contains_string(checkpoint_audit, digest),
                 "checkpoint audit does not bind a checkpoint hash")
        checkpoint_hashes.append(digest)
    config = get_native_config(runtime.dataset)
    model = MultiViewBackbone(
        config,
        view_num=contract.n_views,
        view_size=contract.view_dims,
        n_clusters=contract.n_clusters,
        seed=int(config["training"]["seed"]),
    )
    states = tuple(_load_torch_state(path) for path in provenance.checkpoint_paths)
    model.load_state_dicts(states, strict=True)
    initial_hash = _model_sha256(model)
    if provenance.expected_initial_model_sha256 is not None:
        _require(initial_hash == provenance.expected_initial_model_sha256,
                 "initial model hash mismatch")
    model.to_device(device)
    for autoencoder in model.autoencoders:
        autoencoder.train()
    return model, initial_hash, tuple(checkpoint_hashes), audit_hash


def _final_prediction_state(model, views, native_state, seed, device):
    _require(isinstance(native_state, NativeTargetState), "R5 final state is missing")
    _require(len(native_state.view_weights) == len(model.autoencoders),
             "R5 final view weights are invalid")
    latent_views = []
    refresh_q_views = []
    with torch.no_grad():
        for view_index, autoencoder in enumerate(model.autoencoders):
            # views[v]:[N,D_v] -> latent_views[v]:[N,D_z] -> q_v:[N,K].
            full_view = views[view_index].to(device)
            latent = autoencoder.encoder(full_view)
            q_view = autoencoder.clustering(latent)
            latent_views.append(latent.detach().cpu().numpy())
            refresh_q_views.append(q_view.detach().cpu().numpy())
        p_global, matches, predictions, weights, _ = native_refresh_from_latents(
            latent_views,
            refresh_q_views,
            native_state.view_weights,
            n_clusters=int(model.n_clusters),
            random_state=int(seed),
        )
        final_q_views = []
        for view_index, autoencoder in enumerate(model.autoencoders):
            # views[v]:[N,D_v] -> final q_local_v:[N,K], read-only post-refresh.
            full_view = views[view_index].to(device)
            latent = autoencoder.encoder(full_view)
            final_q_views.append(
                autoencoder.clustering(latent).detach().cpu().numpy()
            )
    # q_local:[N,V,K]; M_v:[V,K,K].
    q_local = np.ascontiguousarray(np.stack(final_q_views, axis=1), dtype=np.float32)
    matrix = np.ascontiguousarray(matches, dtype=np.float32)
    # q_aligned_v:[N,K] = q_local_v:[N,K] @ M_v[v].T:[K,K].
    q_aligned = np.ascontiguousarray(np.stack([
        q_local[:, view_index, :] @ matrix[view_index].T
        for view_index in range(q_local.shape[1])
    ], axis=1), dtype=np.float32)
    final_predictions = np.ascontiguousarray(predictions, dtype=np.int64)
    _require(final_predictions.shape == (q_local.shape[0],),
             "final predictions must have shape [N]")
    _require(
        np.isfinite(p_global).all() and np.isfinite(q_local).all()
        and np.isfinite(q_aligned).all() and np.isfinite(matrix).all(),
        "final prediction state is nonfinite",
    )
    return final_predictions, q_local, q_aligned, matrix, {
        "executed_after_r5": True,
        "refresh_count": 1,
        "gradient_enabled": False,
        "optimizer_present": False,
        "model_mutated": False,
        "prediction_source": "second-pass native KMeans IDs",
        "p_global_shape": list(np.asarray(p_global).shape),
        "match_shape": list(matrix.shape),
        "view_weights": [float(value) for value in weights],
    }


def _source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = (
        "release_core/backbone/autoencoder.py",
        "release_core/backbone/multiview.py",
        "release_core/backbone/clustering.py",
        "release_core/data/sample_ids.py",
        "release_core/semantics/sparse_labels.py",
        "release_core/action/relation.py",
        "release_core/training/alternating.py",
        "release_core/runtime/__init__.py",
        "release_core/runtime/entrypoint.py",
    )
    return {path: _file_sha256(root / path) for path in paths}


def _persist_pre_gt(output_root, arrays, audit):
    target = Path(output_root)
    _require(not target.exists(), "refusing to overwrite an existing run")
    target.mkdir(parents=True, exist_ok=False)
    bundle_path = target / "pre_gt_bundle.npz"
    np.savez(bundle_path, **arrays)
    with bundle_path.open("rb+") as stream:
        os.fsync(stream.fileno())
    with np.load(bundle_path, allow_pickle=False) as archive:
        _require(tuple(archive.files) == _BUNDLE_KEYS, "bundle key whitelist mismatch")
        for name in _BUNDLE_KEYS:
            _require(np.array_equal(archive[name], arrays[name]),
                     "bundle reload mismatch: " + name)
    bundle_hash = _file_sha256(bundle_path)
    audit["bundle"] = {
        "path": str(bundle_path),
        "sha256": bundle_hash,
        "keys": list(_BUNDLE_KEYS),
        "arrays": {
            name: {
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
                "logical_sha256": _payload_sha256(value),
            }
            for name, value in arrays.items()
        },
    }
    audit_path = target / "pre_gt_audit.json"
    _write_json(audit_path, audit)
    audit_hash = _file_sha256(audit_path)
    seal = {
        "schema": _SEAL_SCHEMA,
        "hash_algorithm": "SHA256",
        "bundle_path": str(bundle_path),
        "bundle_sha256": bundle_hash,
        "audit_path": str(audit_path),
        "audit_sha256": audit_hash,
        "bundle_keys": list(_BUNDLE_KEYS),
        "prediction_logical_sha256": audit["prediction_logical_sha256"],
        "sample_id_logical_sha256": audit["sample_id_logical_sha256"],
        "full_gt_loaded_before_seal": False,
        "metrics_computed_before_seal": False,
    }
    seal_path = target / "pre_gt_seal.json"
    _write_json(seal_path, seal)
    return SealedPredictionPaths(bundle_path, audit_path, seal_path)


def run_pre_gt(runtime: RuntimeConfig, provenance: ProvenanceConfig):
    """Train once, refresh once, and seal predictions without accepting GT."""
    if not isinstance(runtime, RuntimeConfig):
        raise TypeError("runtime must be RuntimeConfig")
    if not isinstance(provenance, ProvenanceConfig):
        raise TypeError("provenance must be ProvenanceConfig")
    _require(not provenance.output_root.exists(), "refusing to overwrite an existing run")
    _verify_expected_files(provenance)

    views_numpy, sample_ids, contract, feature_hash = _load_feature_only(
        runtime, provenance
    )
    split, split_hash = _load_sparse_split(
        runtime, provenance, sample_ids, contract.n_clusters
    )
    cycle, relation, balance, utility_hash, semantic_hash = _load_action_inputs(
        provenance, split, contract.n_samples
    )
    device, determinism = _configure_determinism(runtime.training_seed, runtime.device)
    model, initial_hash, checkpoint_hashes, checkpoint_audit_hash = (
        _load_initial_model(runtime, provenance, contract, device)
    )
    views = tuple(torch.from_numpy(view) for view in views_numpy)

    trained_model, native_state, training_audit = run_alternating_training(
        model,
        views,
        sample_ids,
        split.labeled_ids,
        split.unlabeled_ids,
        cycle,
        relation,
        balance,
        epochs=runtime.epochs,
        batch_size=runtime.batch_size,
        seed=runtime.training_seed,
        refresh_interval=runtime.refresh_interval,
        learning_rate=runtime.learning_rate,
        native_lambda1=runtime.native_lambda1,
        device=device,
    )
    _require(trained_model is model, "R5 returned a different model object")
    _require(training_audit.final_prediction_refresh_executed is False,
             "R5 executed the final prediction refresh")
    final_hash = _model_sha256(trained_model)
    if provenance.expected_final_model_sha256 is not None:
        _require(final_hash == provenance.expected_final_model_sha256,
                 "final model hash mismatch")
    predictions, q_local, q_aligned, matrix, refresh_audit = (
        _final_prediction_state(
            trained_model, views, native_state, runtime.training_seed, device
        )
    )
    _require(_model_sha256(trained_model) == final_hash,
             "final refresh mutated model state")
    prediction_hash = _payload_sha256(predictions)
    if provenance.expected_prediction_sha256 is not None:
        _require(prediction_hash == provenance.expected_prediction_sha256,
                 "prediction logical hash mismatch")

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
    _require(tuple(arrays) == _BUNDLE_KEYS, "internal bundle whitelist mismatch")
    audit = {
        "schema": "release-core-pre-gt-audit-v1",
        "scientific_role": "experiment/runtime orchestration",
        "new_scientific_mechanism": False,
        "new_information_utility": False,
        "new_training_loss": False,
        "scientific_config": _jsonable(runtime),
        "runtime_config": {"device": runtime.device},
        "strict_replay": provenance.strict_replay,
        "input_identity": {
            "feature_sha256": feature_hash,
            "sparse_split_sha256": split_hash,
            "utility_sha256": utility_hash,
            "semantic_sha256": semantic_hash,
            "checkpoint_audit_sha256": checkpoint_audit_hash,
            "checkpoint_sha256": list(checkpoint_hashes),
            "sparse_split_logical_sha256": split.digest,
        },
        "initial_model_sha256": initial_hash,
        "final_model_sha256": final_hash,
        "r5_invocation_count": 1,
        "r5_training_audit": _jsonable(training_audit),
        "final_refresh": refresh_audit,
        "prediction_logical_sha256": prediction_hash,
        "sample_id_logical_sha256": _payload_sha256(sample_ids),
        "clean_source_sha256": _source_hashes(),
        "historical_reference_sha256": {
            "audit": provenance.historical_audit_sha256,
            "seal": provenance.historical_seal_sha256,
            "whole_file_equality_required": False,
        },
        "determinism": determinism,
        "gt_firewall": {
            "full_gt_argument_accepted": False,
            "full_gt_loaded": False,
            "metrics_computed": False,
            "full_gt_present_in_bundle": False,
        },
    }
    return _persist_pre_gt(provenance.output_root, arrays, audit)
