"""Materialize a sealed read-only carrier from an exact C3-B0 TRUE_U replay.

This is parent-artifact reconstruction, not a new scientific experiment.  It
reuses the frozen C3-B0 training functions and never loads full ground truth.
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
from experiments.cyclic_utility import train_c3_b0_relation_action_pilot as c3_train
from experiments.cyclic_utility import train_vsa_a0_decoupled_action as vsa_train
from experiments.cyclic_utility import vsa_decoupled_action_protocol as vsa
from experiments.e1_pairwise_utility import train_e1_pairwise_utility as e1_train
from irv.b3_audit import hash_backbone


DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/cyclic_utility"
E1_ROOT = REPOSITORY_ROOT / "outputs/e1_pairwise_utility"
C3_SOURCE_FREEZE = (
    REPOSITORY_ROOT / "experiment_freeze/c3_b0_relation_action_preformal_20260912"
)
C3_FINAL_FREEZE = (
    REPOSITORY_ROOT / "experiment_freeze/c3_b0_multiseed_pass_20260912"
)
FROZEN_FORMAL_OUTPUTS = C3_FINAL_FREEZE / "formal_outputs"


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


def default_output_dir(seed):
    return DEFAULT_OUTPUT_ROOT / (
        "c3_b0_true_u_carrier_seed" + str(carrier.validate_seed(seed))
    )


def seed_specific_e1_paths(seed):
    active_seed = carrier.validate_seed(seed)
    root = E1_ROOT / ("lwc_100ep_seed" + str(active_seed))
    return root / "models", root / "e1_audit.json"


def frozen_true_u_reference_paths(seed):
    active_seed = carrier.validate_seed(seed)
    root = FROZEN_FORMAL_OUTPUTS / (
        "c3_b0_relation_action_pilot_seed" + str(active_seed) + "_true_u"
    )
    return root / "audit.json", root / "final_predictions.npz"


def verify_frozen_c3_b0_lineage():
    parent_hashes = c3_train.verify_frozen_parent_hashes()
    c3_source = c3_train.verify_sha256_manifest(
        C3_SOURCE_FREEZE / "source_sha256.txt", REPOSITORY_ROOT
    )
    c3_formal = c3_train.verify_sha256_manifest(
        C3_FINAL_FREEZE / "formal_outputs_sha256.txt", C3_FINAL_FREEZE
    )
    summary = carrier.validate_parent_hash_audits(
        parent_hashes, c3_source, c3_formal
    )
    return parent_hashes, c3_source, c3_formal, summary


def load_frozen_true_u_reference(seed):
    active_seed = carrier.validate_seed(seed)
    audit_path, prediction_path = frozen_true_u_reference_paths(active_seed)
    _require(
        audit_path.is_file() and prediction_path.is_file(),
        "frozen C3-B0 TRUE_U reference missing",
    )
    with open(audit_path, "r", encoding="utf-8") as input_file:
        audit = json.load(input_file)
    _require(
        audit.get("stage") == c3b0.STAGE
        and audit.get("arm") == carrier.CARRIER_ARM
        and int(audit.get("seed", -1)) == active_seed
        and isinstance(audit.get("final_model_hash"), dict),
        "frozen C3-B0 TRUE_U audit boundary mismatch",
    )
    with np.load(prediction_path, allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == ("predictions", "sample_ids"),
            "frozen C3-B0 prediction schema mismatch",
        )
        predictions = np.ascontiguousarray(archive["predictions"], dtype=np.int64)
        sample_ids = carrier.validate_sample_ids(archive["sample_ids"])
    _require(predictions.shape == (carrier.N,), "frozen prediction shape mismatch")
    return {
        "audit_path": _display(audit_path),
        "audit_file_sha256": carrier.file_sha256(audit_path),
        "prediction_path": _display(prediction_path),
        "prediction_file_sha256": carrier.file_sha256(prediction_path),
        "final_model_hash": audit["final_model_hash"],
        "predictions": predictions,
        "sample_ids": sample_ids,
    }


def validate_seed_lineage(seed, checkpoint_provenance, action_provenance):
    active_seed = carrier.validate_seed(seed)
    _require(
        checkpoint_provenance.get("source_seed") == active_seed
        and checkpoint_provenance.get("source_training_seed") == active_seed
        and checkpoint_provenance.get("requested_seed_lineage_match_pass") is True
        and checkpoint_provenance.get("model_matches_own_audit_pass") is True
        and action_provenance.get("seed") == active_seed,
        "C3-B0 carrier seed lineage mismatch",
    )
    return {
        "requested_seed": active_seed,
        "E1_source_seed": checkpoint_provenance["source_seed"],
        "E1_training_seed": checkpoint_provenance["source_training_seed"],
        "C3_A0_action_seed": action_provenance["seed"],
        "all_seed_lineage_equal_pass": True,
    }


def _full_data_q_local(model, full_views, device):
    """Read q_local[1400,6,7] without changing the replayed model."""
    with torch.no_grad():
        views = []
        for view_id in range(carrier.V):
            x_view = full_views[view_id].to(device)
            latent = model.autoencoders[view_id].encoder(x_view)
            views.append(model.autoencoders[view_id].clustering(latent))
        return torch.stack(views, dim=1).detach()


def replay_frozen_true_u(
    model, semantic_optimizers, native_optimizers, views, sample_ids,
    action_arrays, orders, device, training_seed,
):
    """Replay the frozen 20-epoch C3-B0 TRUE_U control flow exactly."""
    ids = carrier.validate_sample_ids(sample_ids)
    id_tensor = torch.from_numpy(ids)
    dataset = torch.utils.data.TensorDataset(
        *[torch.from_numpy(view) for view in views], id_tensor
    )
    _require(
        len(dataset.tensors) == carrier.V + 1
        and torch.equal(dataset.tensors[-1], id_tensor),
        "carrier replay TensorDataset sample-ID mismatch",
    )
    full_views = list(dataset.tensors[:carrier.V])
    view_weights = [1.0] * carrier.V
    native_p_all = None
    native_matches = None
    refresh_count = 0
    epoch_records = []
    for epoch in range(carrier.FORMAL_EPOCHS):
        semantic_record = c3_train.relation_semantic_phase(
            model, semantic_optimizers, full_views, action_arrays,
            carrier.CARRIER_ARM, orders["semantic_orders"][epoch], device,
        )
        if epoch % e1_train.TARGET_REFRESH_INTERVAL == 0:
            native_p_all, native_matches, _, view_weights = (
                e1_train.refresh_native_target(
                    model, full_views, view_weights, device,
                    training_seed=training_seed,
                )
            )
            refresh_count += 1
        native_record = c3_train.native_consolidation_phase(
            model, native_optimizers, full_views, native_p_all, native_matches,
            orders["native_orders"][epoch], device,
        )
        epoch_records.append({
            "epoch": epoch + 1,
            "phase_A": semantic_record,
            "phase_B": native_record,
        })

    # This is the exact frozen C3-B0 final prediction refresh.  It occurs after
    # epoch-20 Phase B and before save_predictions_before_GT seals predictions.
    _, final_matches, predictions, view_weights = e1_train.refresh_native_target(
        model, full_views, view_weights, device, training_seed=training_seed
    )
    q_local = _full_data_q_local(model, full_views, device)
    q_aligned = carrier.align_q_readonly(q_local, final_matches)
    snapshot = carrier.validate_snapshot_metadata(
        carrier.SNAPSHOT_STAGE,
        carrier.SNAPSHOT_EPOCH,
        carrier.SNAPSHOT_POSITION,
    )
    coordinate = carrier.validate_coordinate_snapshot(
        q_local, q_aligned, final_matches
    )
    return {
        "predictions": np.ascontiguousarray(predictions, dtype=np.int64),
        "sample_ids": ids,
        "q_local": q_local,
        "q_aligned": q_aligned,
        "M_v": final_matches.detach(),
        "snapshot": snapshot,
        "coordinate": coordinate,
        "runtime": {
            "epochs": carrier.FORMAL_EPOCHS,
            "arm": carrier.CARRIER_ARM,
            "phase_A_before_Phase_B_every_epoch": True,
            "native_target_refresh_count_inside_epochs": refresh_count,
            "final_native_prediction_refresh_count": 1,
            "native_refresh_interval": e1_train.TARGET_REFRESH_INTERVAL,
            "epoch_records": epoch_records,
        },
    }


def _to_numpy(value, dtype=None):
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def _write_npz(path, arrays):
    carrier.validate_artifact_keys(arrays)
    np.savez(path, **arrays)
    with open(path, "rb+") as artifact_file:
        os.fsync(artifact_file.fileno())
    with np.load(path, allow_pickle=False) as archive:
        _require(tuple(archive.files) == carrier.ARTIFACT_KEYS, "carrier NPZ schema mismatch")
        for key, expected in arrays.items():
            _require(np.array_equal(archive[key], expected), "carrier NPZ reload mismatch")


def materialize(seed, output_dir=None, device="cuda:0"):
    active_seed = carrier.validate_seed(seed)
    carrier.validate_scientific_parent("C3-B0")
    target = default_output_dir(active_seed) if output_dir is None else _resolve(output_dir)
    _require(not target.exists(), "refusing to overwrite carrier output")
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        _require(torch.cuda.is_available(), "requested CUDA unavailable")
        torch.cuda.set_device(torch_device)

    determinism = vsa_train.enable_strict_determinism(active_seed)
    parent_hashes, c3_source, c3_formal, hash_summary = verify_frozen_c3_b0_lineage()
    reference = load_frozen_true_u_reference(active_seed)
    action_arrays, action_provenance = c3_train.load_frozen_c3a0_action_bundle(active_seed)
    views, sample_ids, feature_provenance = e1_train.load_frozen_feature_artifact(
        c3_train.DEFAULT_FEATURE_PATH, c3_train.DEFAULT_FEATURE_AUDIT_PATH
    )
    fixed_feature_lineage = vsa_train.validate_fixed_feature_realization(
        c3_train.DEFAULT_FEATURE_PATH,
        c3_train.DEFAULT_FEATURE_AUDIT_PATH,
        feature_provenance,
    )
    model_dir, model_audit_path = seed_specific_e1_paths(active_seed)
    model, model_config, checkpoint_provenance = (
        vsa_train.load_lineage_aware_e1_lwc_model(
            model_dir, model_audit_path, torch_device, active_seed
        )
    )
    seed_lineage = validate_seed_lineage(
        active_seed, checkpoint_provenance, action_provenance
    )
    stateful_layer_audit = vsa.audit_model_stateful_layers(model)
    semantic_optimizers, native_optimizers, optimizer_audit = (
        c3_train.build_arm_optimizers(model, carrier.CARRIER_ARM)
    )
    orders = vsa.precompute_epoch_orders(
        carrier.FORMAL_EPOCHS, sample_ids, seed=active_seed
    )
    replay = replay_frozen_true_u(
        model, semantic_optimizers, native_optimizers, views, sample_ids,
        action_arrays, orders, torch_device, active_seed,
    )
    replayed_model_hash = hash_backbone(model.autoencoders)
    parity = carrier.validate_replay_parity(
        reference["final_model_hash"], replayed_model_hash,
        reference["predictions"], replay["predictions"],
        reference["sample_ids"], replay["sample_ids"],
    )

    arrays = OrderedDict((
        ("sample_ids", _to_numpy(replay["sample_ids"], np.int64)),
        ("q_local", _to_numpy(replay["q_local"])),
        ("q_aligned", _to_numpy(replay["q_aligned"])),
        ("M_v", _to_numpy(replay["M_v"])),
    ))
    carrier.validate_artifact_keys(arrays)
    target.mkdir(parents=True, exist_ok=False)
    artifact_path = target / "c3_b0_true_u_carrier.npz"
    _write_npz(artifact_path, arrays)
    audit = {
        "stage": carrier.STAGE,
        "seed": active_seed,
        "artifact_purpose": carrier.PURPOSE,
        "scientific_parent": carrier.SCIENTIFIC_PARENT,
        "carrier_arm": carrier.CARRIER_ARM,
        "training_replay_performed": True,
        "C5_training_performed": False,
        "GT_loaded_for_carrier_materialization": False,
        "C4_used": False,
        **hash_summary,
        "parent_hashes": parent_hashes,
        "frozen_C3_B0_source_provenance": c3_source,
        "frozen_C3_B0_formal_output_provenance": c3_formal,
        "E1_lineage_pass": seed_lineage["all_seed_lineage_equal_pass"],
        "C0_lineage_pass": True,
        "C3_A0_lineage_pass": action_provenance["all_required_logical_hashes_pass"],
        "dataset_realization_pass": feature_provenance["sample_ids_exact_arange_pass"],
        "weak_quality_realization_pass": fixed_feature_lineage["feature_realization_unchanged_pass"],
        "sample_ids_exact_arange_pass": True,
        "seed_lineage": seed_lineage,
        "feature_provenance": feature_provenance,
        "fixed_feature_lineage": fixed_feature_lineage,
        "checkpoint_provenance": checkpoint_provenance,
        "action_provenance": action_provenance,
        "model_config": model_config,
        "optimizer_audit": optimizer_audit,
        "determinism": determinism,
        "stateful_layer_audit": stateful_layer_audit,
        "frozen_TRUE_U_reference": {
            key: value for key, value in reference.items()
            if key not in ("predictions", "sample_ids", "final_model_hash")
        },
        **parity,
        **replay["snapshot"],
        **replay["coordinate"],
        "q_aligned_logical_sha256": carrier.logical_sha256(arrays["q_aligned"]),
        "M_v_logical_sha256": carrier.logical_sha256(arrays["M_v"]),
        "sample_ids_logical_sha256": carrier.logical_sha256(arrays["sample_ids"]),
        "artifact_keys": list(carrier.ARTIFACT_KEYS),
        "artifact_path": _display(artifact_path),
        "artifact_file_sha256": carrier.file_sha256(artifact_path),
        "runtime": replay["runtime"],
    }
    audit_path = target / "c3_b0_true_u_carrier_audit.json"
    _write_json(audit_path, audit)
    seal = carrier.build_valid_seal(audit, artifact_path, audit_path)
    seal["artifact_path"] = _display(artifact_path)
    seal["audit_path"] = _display(audit_path)
    seal_path = target / "c3_b0_true_u_carrier_seal.json"
    _write_json(seal_path, seal)
    return {
        "output_dir": target,
        "artifact_path": artifact_path,
        "audit_path": audit_path,
        "seal_path": seal_path,
        "audit": audit,
        "seal": seal,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=carrier.SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    materialize(args.seed, args.output_dir, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
