"""Train one seed/arm of the C3-B0 relation-level semantic action pilot.

The formal CLI exists for a separately authorized execution phase. Importing
this module performs no training and reads no ground-truth labels.
"""

import argparse
import hashlib
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

from experiments.cyclic_utility import (
    c3_b0_relation_action_protocol as c3b0,
)
from experiments.cyclic_utility import (
    c1_frozen_pseudo_supervision as c1,
)
from experiments.cyclic_utility import (
    train_c1_frozen_pseudo_supervision as c1_train,
)
from experiments.cyclic_utility import (
    train_vsa_a0_decoupled_action as vsa_train,
)
from experiments.cyclic_utility import vsa_decoupled_action_protocol as vsa
from experiments.cyclic_utility import (
    evaluate_c3_a0_utility_conditioned_action_granularity as c3a0_eval,
)
from experiments.e1_pairwise_utility import (
    train_e1_pairwise_utility as e1_train,
)
from irv.b3_audit import hash_backbone
from weak_quality import ndarray_sha256


DATASET = "Caltech-6V"
DEFAULT_FEATURE_PATH = vsa_train.DEFAULT_FEATURE_PATH
DEFAULT_FEATURE_AUDIT_PATH = vsa_train.DEFAULT_FEATURE_AUDIT_PATH
DEFAULT_MODEL_DIR = vsa_train.DEFAULT_E1_LWC_MODEL_DIR
DEFAULT_MODEL_AUDIT_PATH = vsa_train.DEFAULT_E1_LWC_AUDIT_PATH
DEFAULT_FULL_GT_PATH = vsa_train.DEFAULT_FULL_GT_PATH
DEFAULT_C3A0_ROOT = REPOSITORY_ROOT / "outputs/cyclic_utility"
FROZEN_C3A0_DIR = (
    REPOSITORY_ROOT / "experiment_freeze/c3_a0_multiseed_pass_20260912"
)
FROZEN_VSA_DIR = (
    REPOSITORY_ROOT / "experiment_freeze/vsa_multiseed_provenance_pass_20260829"
)
PARENT_C2_MANIFESTS = c3a0_eval.FROZEN_PARENT_MANIFESTS
ACTION_FIELDS = (
    "U_cycle", "unlabeled_ids", "labeled_ids",
    "PredRelation_true", "PredRelation_shuffle",
    "relation_balance_weights_true",
    "relation_balance_weights_shuffle",
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


def _manifest_entries(path):
    entries = []
    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            value = line.strip()
            if value:
                expected, name = value.split(None, 1)
                entries.append((expected, name.strip()))
    _require(entries, "empty frozen manifest: " + str(path))
    return entries


def verify_sha256_manifest(path, content_root):
    manifest = _resolve(path)
    root = _resolve(content_root)
    records = OrderedDict()
    for expected, name in _manifest_entries(manifest):
        target = root / name[2:] if name.startswith("./") else root / name
        _require(target.is_file(), "frozen file is missing: " + str(target))
        actual = file_sha256(target)
        _require(actual == expected, "frozen hash mismatch: " + str(target))
        records[name] = actual
    return {
        "manifest": _display(manifest),
        "manifest_file_sha256": file_sha256(manifest),
        "entry_count": len(records),
        "entries": dict(records),
        "all_entries_exact_match": True,
    }


def verify_frozen_parent_hashes():
    """Fail closed before loading model features or any action bundle."""
    c2 = c3a0_eval.verify_frozen_parent_sources()
    c3_source = verify_sha256_manifest(
        FROZEN_C3A0_DIR / "source_sha256.txt", REPOSITORY_ROOT
    )
    c3_formal = verify_sha256_manifest(
        FROZEN_C3A0_DIR / "formal_outputs_sha256.txt", REPOSITORY_ROOT
    )
    c3_key = verify_sha256_manifest(
        FROZEN_C3A0_DIR / "key_outputs_sha256.txt",
        FROZEN_C3A0_DIR / "key_outputs",
    )
    vsa_source = verify_sha256_manifest(
        FROZEN_VSA_DIR / "source_sha256.txt", REPOSITORY_ROOT
    )
    return {
        "C2": c2,
        "C3_A0_source": c3_source,
        "C3_A0_formal_outputs": c3_formal,
        "C3_A0_key_outputs": c3_key,
        "VSA_source": vsa_source,
        "C2_B0_frozen_hashes_pass": True,
        "C2_C0_frozen_hashes_pass": True,
        "C2_C1_frozen_hashes_pass": True,
        "C2_C2_A0_frozen_hashes_pass": True,
        "C3_A0_multiseed_frozen_hashes_pass": True,
        "VSA_multiseed_provenance_frozen_hashes_pass": True,
        "all_parent_hashes_pass": True,
    }


def default_action_paths(seed):
    active_seed = c3b0.validate_seed(seed)
    root = DEFAULT_C3A0_ROOT / (
        "c3_a0_utility_conditioned_action_granularity_seed"
        + str(active_seed)
    )
    return root / "c3_a0_action_pre_gt.npz", root / "c3_a0_action_seal.json"


def load_frozen_c3a0_action_bundle(seed, npz_path=None, seal_path=None):
    """Load seven whitelisted pre-GT arrays and verify seal/file/logical hashes."""
    active_seed = c3b0.validate_seed(seed)
    default_npz, default_seal = default_action_paths(active_seed)
    npz = default_npz if npz_path is None else _resolve(npz_path)
    seal_file = default_seal if seal_path is None else _resolve(seal_path)
    _require(
        npz.resolve() == default_npz.resolve()
        and seal_file.resolve() == default_seal.resolve(),
        "C3-B0 must consume the canonical frozen C3-A0 action bundle",
    )
    _require(npz.is_file() and seal_file.is_file(), "C3-A0 action bundle missing")
    with open(seal_file, "r", encoding="utf-8") as input_file:
        seal = json.load(input_file)
    _require(
        seal.get("stage") == "C3-A0"
        and int(seal.get("seed", -1)) == active_seed
        and seal.get("N") == c3b0.SAMPLE_NUM
        and seal.get("K") == c3b0.CLASS_NUM
        and seal.get("label_count") == c3b0.LABEL_COUNT
        and seal.get("unlabeled_evaluation_count") == c3b0.UNLABELED_COUNT
        and seal.get("GT_loaded_before_action_seal") is False
        and seal.get("training_performed") is False
        and seal.get("U_cycle_frozen") is True,
        "C3-A0 action seal boundary mismatch",
    )
    _require(
        file_sha256(npz) == seal.get("pre_gt_npz_file_sha256"),
        "C3-A0 action NPZ file hash mismatch",
    )
    arrays = {}
    logical_hashes = {}
    with np.load(npz, allow_pickle=False) as archive:
        _require(all(name in archive.files for name in ACTION_FIELDS),
                 "C3-A0 action NPZ field missing")
        for name in ACTION_FIELDS:
            value = np.array(archive[name], copy=True, order="C")
            expected = seal["arrays"][name]
            actual_hash = ndarray_sha256(value)
            _require(
                list(value.shape) == expected["shape"]
                and str(value.dtype) == expected["dtype"]
                and actual_hash == expected["logical_sha256"],
                "C3-A0 action logical hash mismatch: " + name,
            )
            value.setflags(write=False)
            arrays[name] = value
            logical_hashes[name] = actual_hash
    validated = c3b0.validate_action_arrays(arrays)
    return validated, {
        "seed": active_seed,
        "npz_path": _display(npz),
        "seal_path": _display(seal_file),
        "npz_file_sha256": file_sha256(npz),
        "seal_file_sha256": file_sha256(seal_file),
        "loaded_fields": list(ACTION_FIELDS),
        "logical_sha256": logical_hashes,
        "all_required_logical_hashes_pass": True,
        "sample_ID_indexing_required": True,
        "GT_loaded": False,
        "scientific_metric_loaded": False,
    }


def _tensor_ids(values):
    return torch.as_tensor(np.asarray(values, dtype=np.int64), dtype=torch.long)


def _coverage_record(visited, expected_ids):
    actual = np.concatenate(visited).astype(np.int64, copy=False)
    expected = np.asarray(expected_ids, dtype=np.int64)
    return {
        "sample_count": int(actual.size),
        "unique_sample_count": int(np.unique(actual).size),
        "expected_sample_count": int(expected.size),
        "all_expected_sample_ids_exactly_once_pass": bool(
            actual.size == expected.size
            and np.unique(actual).size == expected.size
            and np.array_equal(np.sort(actual), np.sort(expected))
        ),
        "sample_order_logical_sha256": ndarray_sha256(actual),
    }


def build_arm_optimizers(model, arm):
    """Build the frozen optimizer topology, with a native-only BASE path."""
    active_arm = c3b0.validate_arm(arm)
    if active_arm == "BASE":
        # Exact helper used by the native half of the frozen VSA builder.
        # BASE deliberately creates no semantic optimizer object.
        native_optimizers = e1_train.build_fresh_optimizers(model)
        native_config = vsa.optimizer_configuration(native_optimizers)
        _require(
            len(native_optimizers) == c3b0.VIEW_NUM,
            "BASE native optimizer count mismatch",
        )
        return None, native_optimizers, {
            "semantic_optimizer_config": None,
            "native_optimizer_config": native_config,
            "semantic_optimizer_count": 0,
            "native_optimizer_count": len(native_optimizers),
            "total_optimizer_count": len(native_optimizers),
            "BASE_native_only_optimizer_path": True,
            "frozen_builder_arm": None,
            "native_optimizer_helper": "e1_train.build_fresh_optimizers",
            "semantic_optimizer_created": False,
            "semantic_optimizer_used_only_in_semantic_phase": None,
            "native_optimizer_used_only_in_native_phase": True,
        }
    semantic_optimizers, native_optimizers, audit = (
        vsa.build_decoupled_optimizers(model, "CYCLE")
    )
    audit = dict(audit)
    audit.update({
        "semantic_optimizer_count": len(semantic_optimizers),
        "native_optimizer_count": len(native_optimizers),
        "total_optimizer_count": (
            len(semantic_optimizers) + len(native_optimizers)
        ),
        "BASE_native_only_optimizer_path": False,
        "frozen_builder_arm": "CYCLE",
        "native_optimizer_helper": "vsa.build_decoupled_optimizers",
        "semantic_optimizer_created": True,
    })
    return semantic_optimizers, native_optimizers, audit


def relation_semantic_phase(
    model, semantic_optimizers, full_views, action_arrays, arm, order, device,
):
    """Run relation-only Phase A over unlabeled query IDs.

    Per view: q_batch_v is [B,7], q_anchor_v is [14,7]. The selected sealed
    tensors are U_batch [B,20], PredRelation_batch [B,14,20], and
    balance_weight_batch [B,14,20]. No all-sample pairwise object is built.
    """
    c3b0.validate_arm(arm)
    _require(arm != "BASE", "BASE must not execute Phase A")
    validated = c3b0.validate_action_arrays(action_arrays)
    anchor_ids = _tensor_ids(validated["labeled_ids"])
    visited = []
    records = []
    for batch_number, batch_sample_ids in enumerate(vsa.ordered_batches(order), 1):
        query_ids = c3b0.unlabeled_query_ids(
            batch_sample_ids, validated["unlabeled_ids"]
        )
        _require(query_ids.size > 0, "Phase-A batch has no unlabeled query")
        visited.append(query_ids)
        selected = c3b0.action_batch_by_sample_ids(validated, arm, query_ids)
        query_tensor_ids = _tensor_ids(query_ids)
        vsa.zero_optimizer_gradients(semantic_optimizers)
        _require(vsa.all_parameter_gradients_cleared(model),
                 "Phase-A gradients were not cleared")
        q_sample_views = []
        q_anchor_views = []
        for view_id in range(c3b0.VIEW_NUM):
            query_input = full_views[view_id][query_tensor_ids].to(device)
            query_latent = model.autoencoders[view_id].encoder(query_input)
            q_sample_views.append(
                model.autoencoders[view_id].clustering(query_latent)
            )
            with torch.no_grad():
                anchor_input = full_views[view_id][anchor_ids].to(device)
                anchor_latent = model.autoencoders[view_id].encoder(anchor_input)
                q_anchor = model.autoencoders[view_id].clustering(anchor_latent)
            q_anchor_views.append(q_anchor.detach())
        relation_loss, loss_audit = c3b0.relation_semantic_loss(
            q_sample_views=q_sample_views,
            q_anchor_views=q_anchor_views,
            PredRelation=selected["PredRelation"],
            U_cycle=selected["U_cycle"],
            balance_weight=selected["balance_weight"],
            arm=arm,
        )
        relation_loss.backward()
        gradient = vsa.parameter_gradient_audit(model)
        _require(gradient["gradient_finite_pass"],
                 "non-finite Phase-A relation gradient")
        for optimizer in semantic_optimizers:
            optimizer.step()
        records.append({
            "batch": batch_number,
            "query_count": int(query_ids.size),
            "relation_loss": float(relation_loss.detach().item()),
            "query_sample_ids_logical_sha256": ndarray_sha256(query_ids),
            "gradient": gradient,
            **loss_audit,
        })
    vsa.zero_optimizer_gradients(semantic_optimizers)
    _require(vsa.all_parameter_gradients_cleared(model),
             "Phase-A gradients survived phase transition")
    coverage = _coverage_record(visited, validated["unlabeled_ids"])
    _require(coverage["all_expected_sample_ids_exactly_once_pass"],
             "Phase A did not cover every unlabeled ID exactly once")
    return {
        "phase": "relation_semantic",
        "executed": True,
        "query_role": "unlabeled_only",
        "anchor_count": c3b0.LABEL_COUNT,
        "relation_loss_mean": float(np.mean([
            record["relation_loss"] for record in records
        ])),
        "REC_computed": False,
        "CLU_computed": False,
        "anchor_posterior_detached": True,
        "sample_ID_action_indexing": True,
        "max_action_tensor_shape": [c3b0.BATCH_SIZE, 14, 20],
        "batch_records": records,
        **coverage,
    }


def native_consolidation_phase(
    model, native_optimizers, full_views, native_p_all, native_matches,
    order, device,
):
    """Run only the frozen E1 native REC + 0.01*CLU helper in Phase B."""
    _require(
        not native_p_all.requires_grad and native_p_all.grad_fn is None
        and not native_matches.requires_grad
        and native_matches.grad_fn is None,
        "native P_global/Match must be detached",
    )
    visited = []
    records = []
    for batch_number, batch_sample_ids in enumerate(vsa.ordered_batches(order), 1):
        ids = _tensor_ids(batch_sample_ids)
        visited.append(np.asarray(batch_sample_ids, dtype=np.int64))
        vsa.zero_optimizer_gradients(native_optimizers)
        _require(vsa.all_parameter_gradients_cleared(model),
                 "Phase-B gradients were not cleared")
        x_views = [full_views[v][ids].to(device) for v in range(c3b0.VIEW_NUM)]
        p_batch = native_p_all[ids].to(device).detach()
        reconstructions = []
        q_views = []
        for view_id in range(c3b0.VIEW_NUM):
            x_hat, _, q_local = model.autoencoders[view_id](x_views[view_id])
            reconstructions.append(x_hat)
            q_views.append(q_local)
        native_view_losses, diagnostics = e1_train.native_mvcan_losses(
            x_views, reconstructions, q_views, p_batch, native_matches.detach()
        )
        native_loss = torch.stack(native_view_losses).sum()
        _require(bool(torch.isfinite(native_loss).item()),
                 "non-finite native MVCAN loss")
        native_loss.backward()
        gradient = vsa.parameter_gradient_audit(model)
        _require(gradient["gradient_finite_pass"],
                 "non-finite Phase-B native gradient")
        for optimizer in native_optimizers:
            optimizer.step()
        records.append({
            "batch": batch_number,
            "batch_size": int(ids.numel()),
            "reconstruction_loss_sum_views": float(torch.stack(
                [item[0] for item in diagnostics]
            ).sum().detach().item()),
            "clustering_loss_sum_views": float(torch.stack(
                [item[1] for item in diagnostics]
            ).sum().detach().item()),
            "native_loss": float(native_loss.detach().item()),
            "gradient": gradient,
        })
    vsa.zero_optimizer_gradients(native_optimizers)
    _require(vsa.all_parameter_gradients_cleared(model),
             "Phase-B gradients survived phase end")
    coverage = _coverage_record(visited, np.arange(c3b0.SAMPLE_NUM))
    _require(coverage["all_expected_sample_ids_exactly_once_pass"],
             "Phase B did not cover every sample ID exactly once")
    return {
        "phase": "native_consolidation",
        "executed": True,
        "objective": c3b0.NATIVE_OBJECTIVE,
        "relation_loss_computed": False,
        "P_global_detached": True,
        "Match_detached": True,
        "M_v_detached": True,
        "P_local_derived_from_detached_native_inputs": True,
        "batch_records": records,
        **coverage,
    }


def train_relation_action_arm(
    arm, epochs, model, semantic_optimizers, native_optimizers, views,
    sample_ids, action_arrays, orders, device, training_seed,
):
    """Run complete Phase A then complete Phase B in every frozen epoch."""
    c3b0.validate_arm(arm)
    _require(int(epochs) == c3b0.FORMAL_EPOCHS,
             "C3-B0 formal cadence is fixed at 20 epochs")
    ids = np.asarray(sample_ids, dtype=np.int64)
    _require(np.array_equal(ids, np.arange(c3b0.SAMPLE_NUM)),
             "feature sample IDs are not canonical")
    id_tensor = torch.from_numpy(ids)
    training_dataset = torch.utils.data.TensorDataset(
        *[torch.from_numpy(view) for view in views], id_tensor
    )
    _require(
        len(training_dataset.tensors) == c3b0.VIEW_NUM + 1
        and torch.equal(training_dataset.tensors[-1], id_tensor),
        "training TensorDataset must carry canonical sample IDs",
    )
    full_views = list(training_dataset.tensors[:c3b0.VIEW_NUM])
    view_weights = [1.0] * c3b0.VIEW_NUM
    native_p_all = None
    native_matches = None
    refresh_count = 0
    phase_sequence = []
    epoch_records = []
    if arm == "BASE":
        _require(
            semantic_optimizers is None,
            "BASE must not receive semantic optimizers",
        )
    else:
        _require(
            semantic_optimizers is not None
            and len(semantic_optimizers) == c3b0.VIEW_NUM,
            "non-BASE semantic optimizer boundary mismatch",
        )
    for epoch in range(int(epochs)):
        if arm == "BASE":
            semantic_record = {
                "phase": "relation_semantic", "executed": False,
                "reason": "BASE has no Phase A",
            }
        else:
            semantic_record = relation_semantic_phase(
                model, semantic_optimizers, full_views, action_arrays, arm,
                orders["semantic_orders"][epoch], device,
            )
        phase_sequence.append({
            "epoch": epoch + 1, "phase": "relation_semantic",
            "executed": arm != "BASE",
        })
        if epoch % e1_train.TARGET_REFRESH_INTERVAL == 0:
            native_p_all, native_matches, _, view_weights = (
                e1_train.refresh_native_target(
                    model, full_views, view_weights, device,
                    training_seed=training_seed,
                )
            )
            refresh_count += 1
        native_record = native_consolidation_phase(
            model, native_optimizers, full_views, native_p_all,
            native_matches, orders["native_orders"][epoch], device,
        )
        phase_sequence.append({
            "epoch": epoch + 1, "phase": "native_consolidation",
            "executed": True,
        })
        epoch_records.append({
            "epoch": epoch + 1,
            "phase_A": semantic_record,
            "phase_B": native_record,
        })
    _, final_matches, predictions, _ = e1_train.refresh_native_target(
        model, full_views, view_weights, device, training_seed=training_seed
    )
    return predictions, {
        "epoch_records": epoch_records,
        "phase_sequence": phase_sequence,
        "phase_A_before_phase_B_every_epoch": True,
        "separate_backward_passes": True,
        "additive_combined_loss_used": False,
        "native_target_refresh_count": refresh_count,
        "native_refresh_interval": e1_train.TARGET_REFRESH_INTERVAL,
        "final_M_detached": bool(
            not final_matches.requires_grad and final_matches.grad_fn is None
        ),
        "TensorDataset_contains_canonical_sample_ids": True,
    }


def default_output_dir(seed, arm):
    return REPOSITORY_ROOT / "outputs/cyclic_utility" / (
        "c3_b0_relation_action_pilot_seed" + str(c3b0.validate_seed(seed))
        + "_" + c3b0.ARM_DIRECTORY_NAMES[c3b0.validate_arm(arm)]
    )


def run_arm(
    arm, seed, output_dir=None, device="cuda:0",
    feature_path=DEFAULT_FEATURE_PATH,
    feature_audit_path=DEFAULT_FEATURE_AUDIT_PATH,
    model_dir=DEFAULT_MODEL_DIR, model_audit_path=DEFAULT_MODEL_AUDIT_PATH,
    action_npz_path=None, action_seal_path=None,
    full_gt_path=DEFAULT_FULL_GT_PATH,
):
    """Formal entry point; labels enter only after final predictions are sealed."""
    active_arm = c3b0.validate_arm(arm)
    active_seed = c3b0.validate_seed(seed)
    target = default_output_dir(active_seed, active_arm) if output_dir is None \
        else _resolve(output_dir)
    _require(not target.exists(), "refusing to overwrite C3-B0 output")
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        _require(torch.cuda.is_available(), "requested CUDA unavailable")
        torch.cuda.set_device(torch_device)
    determinism = vsa_train.enable_strict_determinism(active_seed)
    parents = verify_frozen_parent_hashes()
    action_arrays, action_provenance = load_frozen_c3a0_action_bundle(
        active_seed, action_npz_path, action_seal_path
    )
    views, sample_ids, feature_provenance = e1_train.load_frozen_feature_artifact(
        _resolve(feature_path), _resolve(feature_audit_path)
    )
    fixed_feature_lineage = vsa_train.validate_fixed_feature_realization(
        feature_path, feature_audit_path, feature_provenance
    )
    model, model_config, checkpoint_provenance = (
        vsa_train.load_lineage_aware_e1_lwc_model(
            model_dir, model_audit_path, torch_device, active_seed
        )
    )
    semantic_optimizers, native_optimizers, optimizer_audit = (
        build_arm_optimizers(model, active_arm)
    )
    orders = vsa.precompute_epoch_orders(
        c3b0.FORMAL_EPOCHS, sample_ids, seed=active_seed
    )
    initial_model_hash = hash_backbone(model.autoencoders)
    target.mkdir(parents=True, exist_ok=False)
    predictions, runtime = train_relation_action_arm(
        active_arm, c3b0.FORMAL_EPOCHS, model, semantic_optimizers,
        native_optimizers, views, sample_ids, action_arrays, orders,
        torch_device, active_seed,
    )
    prediction_path, prediction_audit = c1_train.save_predictions_before_GT(
        predictions, sample_ids, target
    )
    labels = e1_train.load_labels_after_predictions(
        _resolve(full_gt_path), prediction_path
    )
    metrics = e1_train.evaluate_predictions(labels, predictions)
    config = {
        "stage": c3b0.STAGE, "dataset": DATASET, "seed": active_seed,
        "arm": active_arm, "epochs": c3b0.FORMAL_EPOCHS,
        "batch_size": c3b0.BATCH_SIZE, "view_count": c3b0.VIEW_NUM,
        "class_count": c3b0.CLASS_NUM, "anchor_count": c3b0.LABEL_COUNT,
        "phase_order": ["relation_semantic", "native_consolidation"],
        "relation_objective": c3b0.RELATION_OBJECTIVE,
        "native_objective": c3b0.NATIVE_OBJECTIVE,
        "model_training_config": model_config["training"],
    }
    audit = {
        "stage": c3b0.STAGE, "seed": active_seed, "arm": active_arm,
        "parent_hashes": parents,
        "action_provenance": action_provenance,
        "feature_provenance": feature_provenance,
        "fixed_feature_lineage": fixed_feature_lineage,
        "checkpoint_provenance": checkpoint_provenance,
        "optimizer_separation": optimizer_audit,
        "VSA_optimizer_adapter_arm": optimizer_audit["frozen_builder_arm"],
        "determinism": determinism,
        "initial_model_hash": initial_model_hash,
        "final_model_hash": hash_backbone(model.autoencoders),
        "BASE_has_no_Phase_A": active_arm == "BASE",
        "unlabeled_query_only": True,
        "anchor_side_stop_gradient": True,
        "sample_ID_action_indexing": True,
        "U_cycle_frozen_detached": True,
        "PredRelation_frozen_detached": True,
        "relation_balance_weights_frozen_detached": True,
        "native_P_global_M_P_local_detached": True,
        "native_objective": c3b0.NATIVE_OBJECTIVE,
        "native_MVCAN_helper": "e1_train.native_mvcan_losses",
        "VSA_phase_schedule_reused": True,
        "VSA_decoupled_optimizer_builder_reused": True,
        "additive_combined_loss_used": False,
        "new_information_utility_constructed": False,
        "persistent_memory_created": False,
        "threshold_used": False, "top_k_used": False,
        "sparse_mapping_refit": False, "relation_action_regenerated": False,
        "GT_loaded_during_training": False,
        "GT_loaded_after_final_prediction_seal": True,
        "GT_use": "final ACC/NMI/ARI evaluation only",
        "prediction_audit": prediction_audit,
        "runtime": runtime,
    }
    metric_record = {
        "stage": c3b0.STAGE, "seed": active_seed, "arm": active_arm,
        "metrics": metrics, "evaluation_epoch": c3b0.FORMAL_EPOCHS,
        "GT_used_for_training": False,
    }
    training_summary = {
        "stage": c3b0.STAGE, "seed": active_seed, "arm": active_arm,
        "final_epoch": c3b0.FORMAL_EPOCHS, "best_epoch": None,
        "epoch_selection": "none; final prediction only",
        "test_GT_used_for_epoch_selection": False,
        "phase_A_executed": active_arm != "BASE",
        "phase_B_executed": True,
        "phase_A_before_phase_B_every_epoch": True,
        "additive_combined_loss_used": False,
    }
    c1.write_json(target / "config.json", config)
    c1.write_json(target / "audit.json", audit)
    c1.write_json(target / "metrics.json", metric_record)
    c1.write_json(target / "training_summary.json", training_summary)
    return {
        "output_dir": target, "config": config, "audit": audit,
        "metrics": metric_record, "training_summary": training_summary,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=c3b0.ARMS)
    parser.add_argument("--seed", required=True, type=int, choices=c3b0.SEEDS)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--feature-audit-path", default=str(DEFAULT_FEATURE_AUDIT_PATH))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--model-audit-path", default=str(DEFAULT_MODEL_AUDIT_PATH))
    parser.add_argument("--action-npz-path", default=None)
    parser.add_argument("--action-seal-path", default=None)
    parser.add_argument("--full-gt-path", default=str(DEFAULT_FULL_GT_PATH))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_arm(
        arm=args.arm, seed=args.seed, output_dir=args.output_dir,
        device=args.device, feature_path=args.feature_path,
        feature_audit_path=args.feature_audit_path,
        model_dir=args.model_dir, model_audit_path=args.model_audit_path,
        action_npz_path=args.action_npz_path,
        action_seal_path=args.action_seal_path,
        full_gt_path=args.full_gt_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
