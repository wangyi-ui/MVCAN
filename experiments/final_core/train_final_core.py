"""Run the clean F0-A0 final core as an exact frozen C3-B0 TRUE_U replay.

Importing this module performs no training and never reads full ground truth.
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
from experiments.cyclic_utility import c3_b0_true_u_carrier_protocol as carrier
from experiments.cyclic_utility import materialize_c3_b0_true_u_carrier as replay
from experiments.cyclic_utility import train_c3_b0_relation_action_pilot as c3_train
from experiments.cyclic_utility import train_vsa_a0_decoupled_action as vsa_train
from experiments.cyclic_utility import vsa_decoupled_action_protocol as vsa
from experiments.e1_pairwise_utility import train_e1_pairwise_utility as e1_train
from experiments.final_core import final_core_protocol as f0
from irv.b3_audit import hash_backbone


PREREG_PATH = REPOSITORY_ROOT / (
    "experiment_freeze/f0_a0_final_core_exact_replay_preregistered_20260913/"
    "PROTOCOL.txt"
)
FRAMEWORK_PATH = REPOSITORY_ROOT / (
    "experiment_freeze/final_core_framework_decision_20260913/"
    "FINAL_FRAMEWORK.txt"
)
C3_SOURCE_MANIFEST = REPOSITORY_ROOT / (
    "experiment_freeze/c3_b0_relation_action_preformal_20260912/"
    "source_sha256.txt"
)
C3_FINAL_FREEZE = REPOSITORY_ROOT / (
    "experiment_freeze/c3_b0_multiseed_pass_20260912"
)
C3_FORMAL_MANIFEST = C3_FINAL_FREEZE / "formal_outputs_sha256.txt"
CARRIER_FREEZE = REPOSITORY_ROOT / (
    "experiment_freeze/c3_b0_true_u_carrier_multiseed_pass_20260912"
)
CARRIER_SOURCE_MANIFEST = CARRIER_FREEZE / "source_sha256.txt"
CARRIER_OUTPUT_MANIFEST = CARRIER_FREEZE / "carrier_outputs_sha256.txt"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/final_core"


def _require(condition, message="F0_A0_RUNNER_FAIL_CLOSED"):
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
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(record, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def default_output_dir(seed):
    return DEFAULT_OUTPUT_ROOT / (
        "f0_a0_final_core_exact_replay_seed" + str(f0.validate_seed(seed))
    )


def parent_routes(seed):
    """Return only canonical seed-specific frozen-parent paths."""
    active_seed = f0.validate_seed(seed)
    action_npz, action_seal = c3_train.default_action_paths(active_seed)
    model_dir, model_audit = replay.seed_specific_e1_paths(active_seed)
    reference_audit, reference_predictions = replay.frozen_true_u_reference_paths(
        active_seed
    )
    carrier_root = REPOSITORY_ROOT / "outputs/cyclic_utility" / (
        "c3_b0_true_u_carrier_seed" + str(active_seed)
    )
    c0_root = REPOSITORY_ROOT / "outputs/cyclic_utility" / (
        "c0_complementary_semantic_verification_seed" + str(active_seed)
    )
    return {
        "seed": active_seed,
        "C0_artifact": c0_root / "c0_predictions_and_scores.npz",
        "C0_seal": c0_root / "c0_prediction_seal.json",
        "C3_A0_action_npz": action_npz,
        "C3_A0_action_seal": action_seal,
        "E1_model_dir": model_dir,
        "E1_model_audit": model_audit,
        "C3_B0_reference_audit": reference_audit,
        "C3_B0_reference_predictions": reference_predictions,
        "coordinate_carrier_artifact": carrier_root / "c3_b0_true_u_carrier.npz",
        "coordinate_carrier_audit": carrier_root / "c3_b0_true_u_carrier_audit.json",
        "coordinate_carrier_seal": carrier_root / "c3_b0_true_u_carrier_seal.json",
    }


def verify_parent_integrity():
    """Verify every pinned manifest before any model or training input load."""
    pinned = OrderedDict((
        ("F0_preregistered_protocol", f0.verify_pinned_file(
            PREREG_PATH, f0.PREREGISTERED_PROTOCOL_SHA256,
            "F0_A0_PREREG_HASH_FAIL_CLOSED",
        )),
        ("final_core_framework_decision", f0.verify_pinned_file(
            FRAMEWORK_PATH, f0.FRAMEWORK_DECISION_SHA256,
            "F0_A0_FRAMEWORK_HASH_FAIL_CLOSED",
        )),
        ("C3_B0_source_manifest", f0.verify_pinned_file(
            C3_SOURCE_MANIFEST, f0.C3_SOURCE_MANIFEST_SHA256,
            "F0_A0_C3_SOURCE_MANIFEST_HASH_FAIL_CLOSED",
        )),
        ("C3_B0_formal_manifest", f0.verify_pinned_file(
            C3_FORMAL_MANIFEST, f0.C3_FORMAL_MANIFEST_SHA256,
            "F0_A0_C3_FORMAL_MANIFEST_HASH_FAIL_CLOSED",
        )),
        ("C3_B0_carrier_source_manifest", f0.verify_pinned_file(
            CARRIER_SOURCE_MANIFEST, f0.CARRIER_SOURCE_MANIFEST_SHA256,
            "F0_A0_CARRIER_SOURCE_MANIFEST_HASH_FAIL_CLOSED",
        )),
        ("C3_B0_carrier_output_manifest", f0.verify_pinned_file(
            CARRIER_OUTPUT_MANIFEST, f0.CARRIER_OUTPUT_MANIFEST_SHA256,
            "F0_A0_CARRIER_OUTPUT_MANIFEST_HASH_FAIL_CLOSED",
        )),
    ))
    recursive, c3_source, c3_formal, c3_summary = (
        replay.verify_frozen_c3_b0_lineage()
    )
    carrier_source = c3_train.verify_sha256_manifest(
        CARRIER_SOURCE_MANIFEST, REPOSITORY_ROOT
    )
    carrier_outputs = c3_train.verify_sha256_manifest(
        CARRIER_OUTPUT_MANIFEST, REPOSITORY_ROOT
    )
    summary = {
        **c3_summary,
        "frozen_carrier_source_hashes_pass": carrier_source[
            "all_entries_exact_match"
        ],
        "frozen_carrier_output_hashes_pass": carrier_outputs[
            "all_entries_exact_match"
        ],
    }
    f0.validate_parent_hash_summary(summary)
    return {
        "pinned_files": pinned,
        "recursive_C0_C3A0_parent_hashes": recursive,
        "C3_B0_authoritative_source": c3_source,
        "C3_B0_frozen_formal_outputs": c3_formal,
        "carrier_source_implementation_evidence": carrier_source,
        "carrier_output_exact_replay_evidence": carrier_outputs,
        "summary": summary,
        "all_parent_hashes_pass": True,
    }


def _load_full_action_reference(action_provenance):
    npz_path = _resolve(action_provenance["npz_path"])
    seal_path = _resolve(action_provenance["seal_path"])
    seal = _read_json(seal_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        raw = {
            name: np.array(archive[name], copy=True, order="C")
            for name in c3_train.ACTION_FIELDS
        }
    _require(
        seal.get("stage") == "C3-A0"
        and seal.get("U_cycle_frozen") is True
        and seal.get("GT_loaded_before_action_seal") is False,
        "F0_A0_C3_A0_ACTION_SEAL_FAIL_CLOSED",
    )
    return raw, seal


def _validate_coordinate_carrier(seed, routes):
    artifact = routes["coordinate_carrier_artifact"]
    audit_path = routes["coordinate_carrier_audit"]
    seal_path = routes["coordinate_carrier_seal"]
    _require(
        artifact.is_file() and audit_path.is_file() and seal_path.is_file(),
        "F0_A0_COORDINATE_CARRIER_MISSING_FAIL_CLOSED",
    )
    audit = _read_json(audit_path)
    seal = _read_json(seal_path)
    _require(
        int(seal.get("seed", -1)) == f0.validate_seed(seed)
        and seal.get("carrier_valid_for_downstream_readonly_use") is True
        and seal.get("all_parent_hashes_pass") is True
        and seal.get("final_model_hash_equal") is True
        and seal.get("final_predictions_equal") is True
        and seal.get("coordinate_mapping_pass") is True
        and seal.get("GT_loaded_for_carrier_materialization") is False
        and seal.get("C4_used") is False
        and f0.file_sha256(artifact) == seal.get("artifact_file_sha256")
        and f0.file_sha256(audit_path) == seal.get("audit_file_sha256"),
        "F0_A0_COORDINATE_CARRIER_SEAL_FAIL_CLOSED",
    )
    with np.load(artifact, allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == carrier.ARTIFACT_KEYS,
            "F0_A0_COORDINATE_CARRIER_SCHEMA_FAIL_CLOSED",
        )
        q_local = torch.from_numpy(np.array(archive["q_local"], copy=True))
        q_aligned = torch.from_numpy(np.array(archive["q_aligned"], copy=True))
        M_v = torch.from_numpy(np.array(archive["M_v"], copy=True))
        sample_ids = np.array(archive["sample_ids"], copy=True)
    carrier.validate_sample_ids(sample_ids)
    coordinate = f0.validate_coordinate_mapping(q_local, q_aligned, M_v)
    return {
        "role": "frozen exact-replay coordinate evidence; not scientific parent",
        "artifact_path": _display(artifact),
        "artifact_file_sha256": f0.file_sha256(artifact),
        "audit_path": _display(audit_path),
        "audit_file_sha256": f0.file_sha256(audit_path),
        "seal_path": _display(seal_path),
        "seal_file_sha256": f0.file_sha256(seal_path),
        "coordinate_mapping_audit_pass": coordinate["coordinate_mapping_pass"],
        "mapping_formula": coordinate["mapping_formula"],
        "M_v_orientation": coordinate["M_v_orientation"],
        "GT_loaded": False,
    }


def _to_numpy(value, dtype=None):
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def _persist_pre_gt(target, arrays, audit):
    _require(not target.exists(), "refusing to overwrite F0-A0 output")
    f0.validate_pre_gt_arrays(arrays)
    target.mkdir(parents=True, exist_ok=False)
    artifact_path = target / "final_core_pre_gt_artifact.npz"
    np.savez(artifact_path, **arrays)
    with open(artifact_path, "rb+") as artifact_file:
        os.fsync(artifact_file.fileno())
    with np.load(artifact_path, allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == f0.PRE_GT_ARRAY_KEYS
            and all(np.array_equal(archive[name], arrays[name]) for name in arrays),
            "F0_A0_PRE_GT_RELOAD_FAIL_CLOSED",
        )
    audit["arrays"] = f0.array_records(arrays)
    audit["pre_gt_array_whitelist"] = list(f0.PRE_GT_ARRAY_KEYS)
    audit["artifact_path"] = _display(artifact_path)
    audit["artifact_file_sha256"] = f0.file_sha256(artifact_path)
    audit_path = target / "final_core_pre_gt_audit.json"
    _write_json(audit_path, audit)
    seal = f0.build_pre_gt_seal(audit, artifact_path, audit_path)
    seal["artifact_path"] = _display(artifact_path)
    seal["audit_path"] = _display(audit_path)
    seal_path = target / "final_core_pre_gt_seal.json"
    _write_json(seal_path, seal)
    return {
        "output_dir": target,
        "artifact_path": artifact_path,
        "audit_path": audit_path,
        "seal_path": seal_path,
        "audit": audit,
        "seal": seal,
    }


def run_exact_replay(seed, output_dir=None, device="cuda:0", forbidden_flags=None):
    """Execute exactly one frozen C3-B0 TRUE_U replay and seal before GT."""
    active_seed = f0.validate_seed(seed)
    flags = f0.validate_forbidden_flags(
        f0.default_forbidden_flags() if forbidden_flags is None else forbidden_flags
    )
    target = default_output_dir(active_seed) if output_dir is None else _resolve(output_dir)
    _require(not target.exists(), "refusing to overwrite F0-A0 output")

    # Every parent and forbidden-path gate runs before model construction/training.
    parent_integrity = verify_parent_integrity()
    routes = parent_routes(active_seed)
    coordinate_reference = _validate_coordinate_carrier(active_seed, routes)
    action_arrays, action_provenance = c3_train.load_frozen_c3a0_action_bundle(
        active_seed
    )
    raw_action, action_seal = _load_full_action_reference(action_provenance)

    c0_artifact = routes["C0_artifact"]
    c0_seal = routes["C0_seal"]
    _require(
        c0_artifact.is_file() and c0_seal.is_file()
        and f0.file_sha256(c0_artifact) == action_seal["input_C0_NPZ_SHA256"]
        and f0.file_sha256(c0_seal)
        == action_seal["input_C0_prediction_seal_SHA256"],
        "F0_A0_C0_LINEAGE_HASH_FAIL_CLOSED",
    )

    utility = f0.validate_utility(
        raw_action["U_cycle"], action_provenance["logical_sha256"]["U_cycle"]
    )
    utility_mapping = f0.validate_training_action_view(
        raw_action["U_cycle"], action_arrays["unlabeled_ids"],
        action_arrays["U_cycle"],
    )
    labeled = f0.validate_labeled_ids(
        action_arrays["labeled_ids"], raw_action["labeled_ids"]
    )
    relation = f0.validate_relation_inputs(
        action_arrays["PredRelation_true"], raw_action["PredRelation_true"],
        action_arrays["relation_balance_weights_true"],
        raw_action["relation_balance_weights_true"],
    )

    reference = replay.load_frozen_true_u_reference(active_seed)
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
    model, model_config, checkpoint_provenance = (
        vsa_train.load_lineage_aware_e1_lwc_model(
            routes["E1_model_dir"], routes["E1_model_audit"],
            torch_device, active_seed,
        )
    )
    seed_lineage = replay.validate_seed_lineage(
        active_seed, checkpoint_provenance, action_provenance
    )
    initial_model_hash = hash_backbone(model.autoencoders)
    stateful_layer_audit = vsa.audit_model_stateful_layers(model)
    semantic_optimizers, native_optimizers, optimizer_audit = (
        c3_train.build_arm_optimizers(model, c3b0.PRIMARY_ARM)
    )
    orders = vsa.precompute_epoch_orders(
        c3b0.FORMAL_EPOCHS, sample_ids, seed=active_seed
    )

    # Authoritative replay: no local copy of the loss, gradient, or schedule.
    result = replay.replay_frozen_true_u(
        model, semantic_optimizers, native_optimizers, views, sample_ids,
        action_arrays, orders, torch_device, active_seed,
    )
    final_model_hash = hash_backbone(model.autoencoders)
    parity = f0.validate_replay_parity(
        reference["final_model_hash"], final_model_hash,
        reference["predictions"], result["predictions"],
        reference["sample_ids"], result["sample_ids"],
    )
    coordinate = f0.validate_coordinate_mapping(
        result["q_local"], result["q_aligned"], result["M_v"]
    )

    parent_hash_values = np.asarray([
        f0.PREREGISTERED_PROTOCOL_SHA256,
        f0.FRAMEWORK_DECISION_SHA256,
        f0.C3_SOURCE_MANIFEST_SHA256,
        f0.C3_FORMAL_MANIFEST_SHA256,
        f0.CARRIER_SOURCE_MANIFEST_SHA256,
        f0.CARRIER_OUTPUT_MANIFEST_SHA256,
    ], dtype="<U64")
    frozen_input_values = np.asarray([
        action_provenance["logical_sha256"][name]
        for name in f0.FROZEN_INPUT_HASH_NAMES
    ], dtype="<U64")
    arrays = OrderedDict((
        ("sample_ids", _to_numpy(result["sample_ids"], np.int64)),
        ("final_predictions", _to_numpy(result["predictions"], np.int64)),
        ("labeled_ids", _to_numpy(action_arrays["labeled_ids"], np.int64)),
        ("q_local", _to_numpy(result["q_local"])),
        ("q_aligned", _to_numpy(result["q_aligned"])),
        ("M_v", _to_numpy(result["M_v"])),
        ("frozen_input_logical_hashes", frozen_input_values),
        ("parent_lineage_hashes", parent_hash_values),
        ("final_model_hash_aggregate", np.asarray(
            final_model_hash["aggregate"], dtype="<U64"
        )),
        ("reference_prediction_file_sha256", np.asarray(
            reference["prediction_file_sha256"], dtype="<U64"
        )),
    ))

    audit = {
        "stage": f0.STAGE,
        "seed": active_seed,
        "purpose": f0.PURPOSE,
        "scientific_lineage": list(f0.SCIENTIFIC_LINEAGE),
        "scientific_parent": "C3-B0 TRUE_U",
        "carrier_role": "frozen exact-replay implementation evidence only",
        "C4_is_training_parent": False,
        "C5_A0_is_training_parent": False,
        "C5_B0_is_training_parent": False,
        "forbidden_flags": dict(flags),
        "parent_integrity": parent_integrity,
        "parent_hashes_pass": True,
        "routes": {key: _display(value) if key != "seed" else value
                   for key, value in routes.items()},
        "coordinate_reference": coordinate_reference,
        "sample_ids_equal": sample_gate["exact_equal"],
        "labeled_ids_equal": labeled["exact_equal"],
        **utility,
        "utility_sample_ID_mapping": utility_mapping,
        **relation,
        "coordinate_mapping_audit_pass": coordinate[
            "coordinate_mapping_pass"
        ],
        "coordinate_mapping": coordinate,
        "final_sample_ids_equal": parity[
            "final_prediction_sample_ids_equal"
        ],
        "final_predictions_equal": parity["final_predictions_equal"],
        "final_model_hash_supported": True,
        "final_model_hash_equal": parity["final_model_hash_equal"],
        "replay_parity": parity,
        "GT_loaded_before_pre_gt_seal": False,
        "GT_loaded_during_training": False,
        "scientific_metric_loaded_during_runner": False,
        "full_GT_present_in_pre_gt_artifact": False,
        "authoritative_training_entry": (
            "materialize_c3_b0_true_u_carrier.replay_frozen_true_u -> "
            "train_c3_b0_relation_action_pilot.relation_semantic_phase/"
            "native_consolidation_phase"
        ),
        "tensor_contract": {
            "q_local": [f0.N, f0.V, f0.K],
            "q_aligned": [f0.N, f0.V, f0.K],
            "U_cycle_full": [f0.N, f0.S],
            "U_cycle_training": [f0.NU, f0.S],
            "sample_ids": [f0.N],
            "labeled_ids": [f0.L],
            "PredRelation_true": [f0.NU, f0.L, f0.S],
            "relation_balance_weights_true": [f0.NU, f0.L, f0.S],
            "M_v": [f0.V, f0.K, f0.K],
        },
        "gradient_semantics": {
            "query_posterior_receives_Phase_A_gradient": True,
            "anchor_posterior_detached": True,
            "U_cycle_detached": True,
            "PredRelation_detached": True,
            "relation_balance_weights_detached": True,
            "native_P_global_detached": True,
            "native_M_v_detached": True,
            "additional_detach_added_by_F0": False,
            "frozen_detach_removed_by_F0": False,
        },
        "training_contract": {
            "arm": c3b0.PRIMARY_ARM,
            "epochs": c3b0.FORMAL_EPOCHS,
            "optimizer": "six semantic Adam + six native Adam",
            "learning_rate": e1_train.LEARNING_RATE,
            "batch_size": c3b0.BATCH_SIZE,
            "native_objective": c3b0.NATIVE_OBJECTIVE,
            "relation_objective": c3b0.RELATION_OBJECTIVE,
            "native_refresh_interval": e1_train.TARGET_REFRESH_INTERVAL,
            "phase_A_before_Phase_B_every_epoch": True,
            "new_loss_added": False,
        },
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
        "runtime": result["runtime"],
        "snapshot": result["snapshot"],
        "frozen_TRUE_U_reference": {
            key: value for key, value in reference.items()
            if key not in ("predictions", "sample_ids", "final_model_hash")
        },
        "frozen_input_hash_names": list(f0.FROZEN_INPUT_HASH_NAMES),
        "parent_lineage_hash_names": list(f0.PARENT_LINEAGE_HASH_NAMES),
    }
    return _persist_pre_gt(target, arrays, audit)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="F0-A0 exact replay of frozen C3-B0 TRUE_U (pre-GT)"
    )
    parser.add_argument("--seed", type=int, required=True, choices=f0.SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_exact_replay(
        args.seed, output_dir=args.output_dir, device=args.device
    )
    print(json.dumps({
        "stage": f0.STAGE,
        "seed": args.seed,
        "pre_gt_seal": _display(result["seal_path"]),
        "pre_gt_seal_valid": True,
        "GT_loaded": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
