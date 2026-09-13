"""Run a pre-GT C4-A0 sidecar on an exact C3-B0 TRUE_U replay.

This entry point never loads full ground truth.  It reproduces the frozen
C3-B0 control flow and observes it between Phase A and Phase B only.
"""

import argparse
import hashlib
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

from experiments.cyclic_utility import (
    c4_a0_utility_conditioned_semantic_memory_protocol as c4,
)
from experiments.cyclic_utility import (
    train_c3_b0_relation_action_pilot as c3_train,
)
from experiments.cyclic_utility import (
    train_vsa_a0_decoupled_action as vsa_train,
)
from experiments.cyclic_utility import vsa_decoupled_action_protocol as vsa
from experiments.e1_pairwise_utility import train_e1_pairwise_utility as e1_train
from irv.b3_audit import hash_backbone
from irv.b4_information_utility import tensor_sha256


STAGE = c4.STAGE
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/cyclic_utility"
E1_ROOT = REPOSITORY_ROOT / "outputs/e1_pairwise_utility"
C3_MULTISEED_FREEZE = (
    REPOSITORY_ROOT / "experiment_freeze/c3_b0_multiseed_pass_20260912"
)


def _require(condition, message):
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


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value):
    """JSON conversion for audit-only NumPy values.

    This helper is serialization-only. It must not modify any scientific
    tensor, model state, utility, memory state, or training behavior.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    raise TypeError(
        f"Object of type {value.__class__.__name__} "
        "is not JSON serializable"
    )


def _write_json_durable(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def seed_specific_e1_carrier_paths(seed):
    """Never inherit C3-B0's historical seed20 model defaults."""
    active_seed = c4.validate_seed(seed)
    root = E1_ROOT / ("lwc_100ep_seed" + str(active_seed))
    return root / "models", root / "e1_audit.json"


def validate_carrier_lineage(seed, checkpoint_provenance, action_provenance):
    active_seed = c4.validate_seed(seed)
    _require(
        checkpoint_provenance.get("source_seed") == active_seed
        and checkpoint_provenance.get("source_training_seed") == active_seed
        and checkpoint_provenance.get("requested_seed_lineage_match_pass") is True
        and checkpoint_provenance.get("model_matches_own_audit_pass") is True
        and action_provenance.get("seed") == active_seed,
        "C4-A0 carrier/action seed lineage mismatch",
    )
    return {
        "requested_seed": active_seed,
        "E1_source_seed": checkpoint_provenance["source_seed"],
        "E1_training_seed": checkpoint_provenance["source_training_seed"],
        "C3_A0_action_seed": action_provenance["seed"],
        "all_seed_lineage_equal_pass": True,
    }


def _digest_update(digest, value):
    if torch.is_tensor(value):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(b"tensor\0" + str(array.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0" + str(array.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: str(item)):
            _digest_update(digest, str(key))
            _digest_update(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(("list:" + str(len(value)) + "\0").encode("ascii"))
        for item in value:
            _digest_update(digest, item)
    else:
        digest.update((type(value).__name__ + ":" + repr(value) + "\0").encode("utf-8"))


def structured_sha256(value):
    digest = hashlib.sha256()
    _digest_update(digest, value)
    return digest.hexdigest()


def _optimizer_state_hash(semantic_optimizers, native_optimizers):
    optimizers = list(semantic_optimizers or ()) + list(native_optimizers or ())
    return structured_sha256([optimizer.state_dict() for optimizer in optimizers])


def _buffer_state(model):
    return [
        (view_id, name, buffer.detach().cpu().clone())
        for view_id, autoencoder in enumerate(model.autoencoders)
        for name, buffer in autoencoder.named_buffers()
    ]


def capture_side_effect_state(
    model, semantic_optimizers, native_optimizers, native_p_all,
    native_matches, view_weights,
):
    """Capture every state category that an extra q forward must preserve."""
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "model_hash": hash_backbone(model.autoencoders),
        "optimizer_hash": _optimizer_state_hash(semantic_optimizers, native_optimizers),
        "buffers_hash": structured_sha256(_buffer_state(model)),
        "cpu_rng": torch.random.get_rng_state().clone(),
        "cuda_rng": [value.clone() for value in cuda_rng],
        "native_P_hash": tensor_sha256(native_p_all),
        "native_M_hash": tensor_sha256(native_matches),
        "view_weights_hash": structured_sha256(np.asarray(view_weights, dtype=np.float64)),
        "training_flags": tuple(
            (view_id, name, bool(module.training))
            for view_id, autoencoder in enumerate(model.autoencoders)
            for name, module in autoencoder.named_modules()
        ),
    }


def assert_side_effect_state_equal(before, after):
    keys = (
        "model_hash", "optimizer_hash", "buffers_hash", "native_P_hash",
        "native_M_hash", "view_weights_hash", "training_flags",
    )
    ordinary_equal = all(before[key] == after[key] for key in keys)
    cpu_equal = torch.equal(before["cpu_rng"], after["cpu_rng"])
    cuda_equal = len(before["cuda_rng"]) == len(after["cuda_rng"]) and all(
        torch.equal(first, second)
        for first, second in zip(before["cuda_rng"], after["cuda_rng"])
    )
    _require(ordinary_equal and cpu_equal and cuda_equal, "C4_SIDE_EFFECT_FAIL_CLOSED")
    return {
        "model_state_unchanged": True,
        "optimizer_state_unchanged": True,
        "registered_buffers_unchanged": True,
        "CPU_RNG_unchanged": True,
        "CUDA_RNG_unchanged": True,
        "native_P_global_unchanged": True,
        "native_M_v_unchanged": True,
        "native_view_weights_unchanged": True,
        "module_training_flags_unchanged": True,
    }


def full_data_q_snapshot(model, full_views, native_matches, device):
    """Pure encoder/clustering observation; no decoder, refresh, or RNG."""
    with torch.no_grad():
        local_views = []
        for view_id in range(c4.V):
            x_view = full_views[view_id].to(device)
            latent = model.autoencoders[view_id].encoder(x_view)
            local_views.append(model.autoencoders[view_id].clustering(latent))
        q_local = torch.stack(local_views, dim=1).detach()
        q_aligned = c4.align_q_readonly(q_local, native_matches).detach()
        payload = c4.build_directional_payload(q_aligned).detach()
    return q_local, q_aligned, payload


def _tensor_to_numpy(value, dtype=None):
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    return np.ascontiguousarray(array if dtype is None else array.astype(dtype, copy=False))


def _save_npz_durable(path, arrays):
    np.savez(path, **arrays)
    with open(path, "rb+") as input_file:
        os.fsync(input_file.fileno())
    with np.load(path, allow_pickle=False) as archive:
        _require(tuple(archive.files) == tuple(arrays), "C4-A0 NPZ whitelist/order mismatch")
        for name, expected in arrays.items():
            _require(np.array_equal(archive[name], expected), "C4-A0 NPZ reload mismatch: " + name)


def _save_epoch20_prediction_seal(output_dir, holdout, readout):
    arrays = OrderedDict((
        ("H", np.asarray(holdout, dtype=np.int64)),
        *(
            ("prediction_" + arm, _tensor_to_numpy(readout[arm]["predictions"], np.int64))
            for arm in c4.MEMORY_ARMS
        ),
    ))
    path = output_dir / "c4_epoch20_predictions_pre_gt.npz"
    _save_npz_durable(path, arrays)
    seal = {
        "stage": STAGE,
        "snapshot": "epoch20_post_write_pre_PhaseB",
        "GT_loaded": False,
        "file_sha256": file_sha256(path),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype),
                   "logical_sha256": tensor_sha256(value)}
            for name, value in arrays.items()
        },
    }
    seal_path = output_dir / "c4_epoch20_predictions_pre_gt_seal.json"
    _write_json_durable(seal_path, seal)
    return path, seal_path, seal


def load_frozen_true_u_reference(seed):
    active_seed = c4.validate_seed(seed)
    source = C3_MULTISEED_FREEZE / "formal_outputs" / (
        "c3_b0_relation_action_pilot_seed" + str(active_seed) + "_true_u"
    )
    audit_path = source / "audit.json"
    prediction_path = source / "final_predictions.npz"
    _require(audit_path.is_file() and prediction_path.is_file(),
             "frozen C3-B0 TRUE_U reference missing")
    with open(audit_path, "r", encoding="utf-8") as input_file:
        audit = json.load(input_file)
    with np.load(prediction_path, allow_pickle=False) as archive:
        _require(tuple(archive.files) == ("predictions", "sample_ids"),
                 "C3-B0 reference prediction schema mismatch")
        predictions = np.array(archive["predictions"], copy=True)
        sample_ids = np.array(archive["sample_ids"], copy=True)
    _require(audit.get("seed") == active_seed and audit.get("arm") == c4.CARRIER_ARM,
             "C3-B0 TRUE_U reference lineage mismatch")
    return {
        "directory": _display(source),
        "audit_path": _display(audit_path),
        "audit_file_sha256": file_sha256(audit_path),
        "prediction_path": _display(prediction_path),
        "prediction_file_sha256": file_sha256(prediction_path),
        "final_model_hash": audit["final_model_hash"],
        "predictions": predictions,
        "sample_ids": sample_ids,
    }


def verify_carrier_parity(model, predictions, sample_ids, reference):
    model_equal = hash_backbone(model.autoencoders) == reference["final_model_hash"]
    prediction_equal = np.array_equal(np.asarray(predictions), reference["predictions"])
    ids_equal = np.array_equal(np.asarray(sample_ids), reference["sample_ids"])
    _require(model_equal and prediction_equal and ids_equal,
             "C4_CARRIER_PARITY_FAIL_CLOSED")
    return {
        "final_carrier_model_hash_equal": True,
        "final_carrier_predictions_equal": True,
        "final_carrier_sample_ids_equal": True,
        "carrier_parity_pass": True,
    }


def default_output_dir(seed):
    return DEFAULT_OUTPUT_ROOT / (
        "c4_a0_utility_conditioned_semantic_memory_seed" + str(c4.validate_seed(seed))
    )


def run_pre_gt(seed, output_dir=None, device="cuda:0"):
    """Replay TRUE_U, create/seal C4 predictions, and never load full GT."""
    active_seed = c4.validate_seed(seed)
    target = default_output_dir(active_seed) if output_dir is None else _resolve(output_dir)
    _require(not target.exists(), "refusing to overwrite C4-A0 output")
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        _require(torch.cuda.is_available(), "requested CUDA unavailable")
        torch.cuda.set_device(torch_device)

    determinism = vsa_train.enable_strict_determinism(active_seed)
    parents = c3_train.verify_frozen_parent_hashes()
    action_arrays, action_provenance = c3_train.load_frozen_c3a0_action_bundle(active_seed)
    views, sample_ids, feature_provenance = e1_train.load_frozen_feature_artifact(
        c3_train.DEFAULT_FEATURE_PATH, c3_train.DEFAULT_FEATURE_AUDIT_PATH
    )
    fixed_feature_lineage = vsa_train.validate_fixed_feature_realization(
        c3_train.DEFAULT_FEATURE_PATH, c3_train.DEFAULT_FEATURE_AUDIT_PATH,
        feature_provenance,
    )
    model_dir, model_audit_path = seed_specific_e1_carrier_paths(active_seed)
    model, model_config, checkpoint_provenance = vsa_train.load_lineage_aware_e1_lwc_model(
        model_dir, model_audit_path, torch_device, active_seed
    )
    lineage = validate_carrier_lineage(
        active_seed, checkpoint_provenance, action_provenance
    )
    stateful_layers = vsa.audit_model_stateful_layers(model)
    semantic_optimizers, native_optimizers, optimizer_audit = (
        c3_train.build_arm_optimizers(model, c4.CARRIER_ARM)
    )
    orders = vsa.precompute_epoch_orders(c4.TRAIN_EPOCHS, sample_ids, seed=active_seed)
    full_views = [torch.from_numpy(view) for view in views]
    view_weights = [1.0] * c4.V
    native_p_all = None
    native_matches = None
    split = c4.fixed_memory_split(action_arrays["unlabeled_ids"])
    generator_subsets, verifier_subsets = c4.frozen_direction_definitions()
    labeled_targets = c3_train.c3a0_eval.c3.FIXED_LABELED_TARGETS
    reference = load_frozen_true_u_reference(active_seed)
    target.mkdir(parents=True, exist_ok=False)

    memory_arms = None
    initial_state = None
    eligibility = None
    writer_utilities = None
    epoch1_matches = None
    epoch_records = []
    epoch20_z = None
    c4_readout = None
    early_prediction_seal = None

    for epoch in range(c4.TRAIN_EPOCHS):
        semantic_record = c3_train.relation_semantic_phase(
            model, semantic_optimizers, full_views, action_arrays, c4.CARRIER_ARM,
            orders["semantic_orders"][epoch], torch_device,
        )
        if epoch % e1_train.TARGET_REFRESH_INTERVAL == 0:
            native_p_all, native_matches, _, view_weights = e1_train.refresh_native_target(
                model, full_views, view_weights, torch_device,
                training_seed=active_seed,
            )
        _require(native_p_all is not None and native_matches is not None,
                 "C4-A0 native target unavailable")

        before = capture_side_effect_state(
            model, semantic_optimizers, native_optimizers, native_p_all,
            native_matches, view_weights,
        )
        _, q_aligned, directional_payload = full_data_q_snapshot(
            model, full_views, native_matches, torch_device
        )
        after = capture_side_effect_state(
            model, semantic_optimizers, native_optimizers, native_p_all,
            native_matches, view_weights,
        )
        side_effect_audit = assert_side_effect_state_equal(before, after)

        if epoch == 0:
            epoch1_matches = native_matches.detach().clone()
            initial_state = c4.initialize_label_memory(
                q_aligned, action_arrays["labeled_ids"], labeled_targets
            )
            memory_arms = c4.clone_initial_memory_arms(initial_state)
            eligibility = c4.freeze_epoch1_writer_eligibility(
                q_aligned, split["W"], action_arrays["labeled_ids"], labeled_targets
            )
            writer_utilities = c4.build_c4_writer_utilities(
                action_arrays["U_cycle"], action_arrays["unlabeled_ids"],
                eligibility["E_c"],
            )

        previous_Z = {
            arm: memory_arms[arm]["Z"].detach().clone() for arm in c4.MEMORY_ARMS
        }
        memory_arms = c4.apply_memory_event(
            memory_arms, directional_payload, eligibility["E_c"], writer_utilities
        )
        writer_index = torch.as_tensor(
            np.array(eligibility["E_c"], copy=True), dtype=torch.long,
            device=directional_payload.device,
        )
        epoch_records.append({
            "epoch": epoch + 1,
            "snapshot": "post_PhaseA_pre_PhaseB",
            "q_aligned_logical_sha256": tensor_sha256(q_aligned),
            "writer_g_direction_logical_sha256": tensor_sha256(
                directional_payload.detach()[writer_index]
            ),
            "writer_ids_logical_sha256": tensor_sha256(eligibility["E_c"]),
            "memory": {
                arm: {
                    "A_sha256": tensor_sha256(memory_arms[arm]["A"]),
                    "Z_sha256": tensor_sha256(memory_arms[arm]["Z"]),
                    "M_sha256": tensor_sha256(memory_arms[arm]["M"]),
                    "DeltaZ": _tensor_to_numpy(
                        memory_arms[arm]["Z"] - previous_Z[arm]
                    ).tolist(),
                    "row_sums": _tensor_to_numpy(
                        memory_arms[arm]["M"].sum(dim=1)
                    ).tolist(),
                }
                for arm in c4.MEMORY_ARMS
            },
            "side_effect_audit": side_effect_audit,
            "relation_phase_executed": semantic_record["executed"],
        })

        if epoch == c4.TRAIN_EPOCHS - 1:
            epoch20_z = c4.holdout_representation(q_aligned, split["H"])
            c4_readout = c4.memory_readout(epoch20_z, memory_arms)
            _, _, early_prediction_seal = _save_epoch20_prediction_seal(
                target, split["H"], c4_readout
            )

        c3_train.native_consolidation_phase(
            model, native_optimizers, full_views, native_p_all, native_matches,
            orders["native_orders"][epoch], torch_device,
        )

    _, _, carrier_predictions, _ = e1_train.refresh_native_target(
        model, full_views, view_weights, torch_device, training_seed=active_seed
    )
    parity = verify_carrier_parity(model, carrier_predictions, sample_ids, reference)

    similarities = OrderedDict(
        (arm, c4.semantic_identity_safety(memory_arms[arm]["M"], initial_state["M"]))
        for arm in c4.MEMORY_ARMS
    )
    arrays = OrderedDict((
        ("sample_ids", np.asarray(sample_ids, dtype=np.int64)),
        ("labeled_ids", np.asarray(action_arrays["labeled_ids"], dtype=np.int64)),
        ("unlabeled_ids", np.asarray(action_arrays["unlabeled_ids"], dtype=np.int64)),
        ("H", np.asarray(split["H"], dtype=np.int64)),
        ("W", np.asarray(split["W"], dtype=np.int64)),
        ("E_c", np.asarray(eligibility["E_c"], dtype=np.int64)),
        ("generator_subsets", np.asarray(generator_subsets, dtype=np.int64)),
        ("verifier_subsets", np.asarray(verifier_subsets, dtype=np.int64)),
        ("epoch1_M_v", _tensor_to_numpy(epoch1_matches)),
        ("epoch1_A0", _tensor_to_numpy(initial_state["A"])),
        ("epoch1_Z0", _tensor_to_numpy(initial_state["Z"])),
        ("epoch1_M0", _tensor_to_numpy(initial_state["M"])),
        ("semantic_margins", np.asarray(eligibility["semantic_margins"])),
        ("writer_predicted_class", np.asarray(eligibility["predicted_class"], dtype=np.int64)),
        ("class_pool_counts", np.asarray(eligibility["class_pool_counts"], dtype=np.int64)),
        ("c4_shuffle_permutation", np.asarray(writer_utilities["permutation_ids"], dtype=np.int64)),
        *(("final_memory_" + arm, _tensor_to_numpy(memory_arms[arm]["M"])) for arm in c4.MEMORY_ARMS),
        ("epoch20_holdout_z", _tensor_to_numpy(epoch20_z)),
        *(("prediction_" + arm, _tensor_to_numpy(c4_readout[arm]["predictions"], np.int64)) for arm in c4.MEMORY_ARMS),
        *(("similarity_" + arm, _tensor_to_numpy(similarities[arm])) for arm in c4.MEMORY_ARMS),
    ))
    c4.validate_pre_gt_array_whitelist(arrays)
    artifact_path = target / "c4_pre_gt_bundle.npz"
    _save_npz_durable(artifact_path, arrays)
    audit = {
        "stage": STAGE,
        "seed": active_seed,
        "carrier_arm": c4.CARRIER_ARM,
        "memory_arms": list(c4.MEMORY_ARMS),
        "GT_loaded": False,
        "C4_memory_fed_back_into_training": False,
        "carrier_lineage": lineage,
        "carrier_reference": {key: value for key, value in reference.items()
                              if key not in ("predictions", "sample_ids")},
        "carrier_parity": parity,
        "parents": parents,
        "feature_provenance": feature_provenance,
        "fixed_feature_lineage": fixed_feature_lineage,
        "checkpoint_provenance": checkpoint_provenance,
        "action_provenance": action_provenance,
        "model_config": model_config,
        "optimizer_audit": optimizer_audit,
        "determinism": determinism,
        "stateful_layer_audit": stateful_layers,
        "split": split,
        "direction_hashes": {
            "generator_subsets": c4.EXPECTED_GENERATOR_SUBSETS_SHA256,
            "verifier_subsets": c4.EXPECTED_VERIFIER_SUBSETS_SHA256,
        },
        "writer_eligibility": {
            key: value for key, value in eligibility.items()
            if key not in ("semantic_scores", "semantic_margins", "predicted_class", "class_pool_counts", "E_c")
        },
        "writer_utilities": {
            key: value for key, value in writer_utilities.items()
            if key not in ("true", "shuffle", "permutation_ids")
        },
        "epoch_records": epoch_records,
        "epoch20_prediction_seal": early_prediction_seal,
        "final_readout_snapshot": "epoch20_post_write_pre_PhaseB",
        "final_native_refresh_used_for_C4": False,
        "pre_gt_array_whitelist": list(c4.PRE_GT_ARRAY_WHITELIST),
        "pre_gt_artifact_path": _display(artifact_path),
        "pre_gt_artifact_file_sha256": file_sha256(artifact_path),
        "pre_gt_arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype),
                   "logical_sha256": tensor_sha256(value)}
            for name, value in arrays.items()
        },
        "all_snapshot_side_effect_audits_pass": True,
        "semantic_identity_safety_pass": True,
    }
    # Convert read-only NumPy values in the split audit to JSON-safe metadata.
    audit["split"] = {
        key: value for key, value in split.items() if key not in ("H", "W")
    }
    audit_path = target / "c4_pre_gt_audit.json"
    _write_json_durable(audit_path, audit)
    seal = {
        "stage": STAGE,
        "seed": active_seed,
        "GT_loaded_before_seal": False,
        "carrier_parity_pass": True,
        "all_snapshot_side_effect_audits_pass": True,
        "semantic_identity_safety_pass": True,
        "artifact_path": _display(artifact_path),
        "artifact_file_sha256": file_sha256(artifact_path),
        "audit_path": _display(audit_path),
        "audit_file_sha256": file_sha256(audit_path),
        "array_whitelist": list(c4.PRE_GT_ARRAY_WHITELIST),
    }
    seal_path = target / "c4_pre_gt_seal.json"
    _write_json_durable(seal_path, seal)
    return {"artifact_path": artifact_path, "audit_path": audit_path,
            "seal_path": seal_path, "audit": audit, "seal": seal}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=c4.SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_pre_gt(args.seed, args.output_dir, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
