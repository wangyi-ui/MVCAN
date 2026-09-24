"""Generic, native-only BASE runtime over frozen release primitives."""

import numpy as np
import torch

from release_core.runtime import ProvenanceConfig, RuntimeConfig
from release_core.runtime import entrypoint as entry
from release_core.training import initial_native_target_state, precompute_training_orders, refresh_native_state_if_due, run_native_consolidation_phase


def base_audit_contract():
    return {"phase_a_executed": False, "semantic_optimizer_created": False, "phase_b_executed": True, "phase_order": ("PHASE_A_SKIPPED", "REFRESH", "PHASE_B"), "full_gt_loaded": False, "final_prediction_source": "final native refresh second-pass KMeans prediction IDs"}


def _native_optimizers(model, learning_rate):
    return tuple(torch.optim.Adam(autoencoder.parameters(), lr=learning_rate) for autoencoder in model.autoencoders)


def run_base_pre_gt(runtime, provenance):
    """Execute BASE: no Phase A or semantic optimizer; refresh then Phase B."""
    if not isinstance(runtime, RuntimeConfig) or not isinstance(provenance, ProvenanceConfig):
        raise TypeError("BASE requires RuntimeConfig and ProvenanceConfig")
    entry._require(not provenance.output_root.exists(), "refusing to overwrite an existing run")
    entry._verify_expected_files(provenance)
    views_numpy, sample_ids, contract, feature_hash = entry._load_feature_only(runtime, provenance)
    split, split_hash = entry._load_sparse_split(runtime, provenance, sample_ids, contract.n_clusters)
    device, determinism = entry._configure_determinism(runtime.training_seed, runtime.device)
    model, initial_hash, checkpoint_hashes, checkpoint_audit_hash = entry._load_initial_model(runtime, provenance, contract, device)
    views = tuple(torch.from_numpy(view) for view in views_numpy)
    optimizers = _native_optimizers(model, runtime.learning_rate)
    orders = precompute_training_orders(sample_ids, runtime.epochs, runtime.training_seed)
    state = initial_native_target_state(contract.n_views)
    epoch_audits = []
    for epoch in range(runtime.epochs):
        state, refresh = refresh_native_state_if_due(model, views, epoch=epoch, refresh_interval=runtime.refresh_interval, previous_state=state, seed=runtime.training_seed, device=device)
        phase_b = run_native_consolidation_phase(model, optimizers, views, sample_ids, state, orders.native_orders[epoch], runtime.batch_size, runtime.native_lambda1, device)
        epoch_audits.append({"epoch": epoch, "phase_a": "SKIPPED", "target_refresh": {"executed": refresh.executed, "refresh_count": refresh.refresh_count}, "phase_b": {"backward_count": phase_b.backward_count, "optimizer_step_count": phase_b.optimizer_step_count}})
    final_hash = entry._model_sha256(model)
    predictions, q_local, q_aligned, matrix, final_refresh = entry._final_prediction_state(model, views, state, runtime.training_seed, device)
    entry._require(entry._model_sha256(model) == final_hash, "final refresh mutated model state")
    arrays = {"sample_ids": np.ascontiguousarray(sample_ids, dtype=np.int64), "final_predictions": predictions, "labeled_ids": np.ascontiguousarray(split.labeled_ids, dtype=np.int64), "q_local": q_local, "q_aligned": q_aligned, "M_v": matrix, "input_sha256": np.asarray((feature_hash, split_hash, checkpoint_audit_hash, *checkpoint_hashes), dtype="<U64"), "initial_model_sha256": np.asarray(initial_hash, dtype="<U64"), "final_model_sha256": np.asarray(final_hash, dtype="<U64")}
    audit = {"schema": "p1-a3-base-pre-gt-audit-v1", "scientific_config": entry._jsonable(runtime), "initial_model_sha256": initial_hash, "final_model_sha256": final_hash, "base": {**base_audit_contract(), "epochs_completed": runtime.epochs, "refresh_interval": runtime.refresh_interval, "refresh_count_before_final": state.refresh_count, "epoch_audits": epoch_audits}, "final_refresh": final_refresh, "prediction_logical_sha256": entry._payload_sha256(predictions), "sample_id_logical_sha256": entry._payload_sha256(sample_ids), "determinism": determinism, "gt_firewall": {"full_gt_loaded": False, "metrics_computed": False}}
    return entry._persist_pre_gt(provenance.output_root, arrays, audit)
