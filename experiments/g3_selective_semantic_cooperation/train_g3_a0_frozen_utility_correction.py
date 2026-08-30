"""Run G3-A0 controlled continuation from the frozen D1 checkpoint.

All three arms use the same label-free training code. Ground-truth labels are
loaded only after every arm has saved a fixed final prediction.
"""

import argparse
import csv
import gc
import itertools
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.cluster import KMeans


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import ClusteringTest
from configure import get_default_config
from experiments.d2_caltech6v import evaluate_d2_a0_utility_transfer as d2
from experiments.g2_utility_semantic_consensus import g2_consensus_protocol as g2
from experiments.g3_selective_semantic_cooperation import (
    g3_target_correction_protocol as protocol,
)
from irv.b3_audit import hash_backbone
from irv.b4_information_utility import tensor_sha256
from weak_quality import apply_weak_quality_protocol


ARM_DIRECTORIES = {
    "BASE": "base",
    "U_CORRECTION": "u_correction",
    "SHUFFLED_U_CORRECTION": "shuffled_u_correction",
}
CHECKPOINT_NAMES = tuple(
    protocol.DATASET_NAME + str(view_id + 1) + "V.pth"
    for view_id in range(protocol.VIEW_NUM)
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _append_jsonl(path, value):
    with open(path, "a", encoding="utf-8") as output_file:
        output_file.write(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )


def _write_training_losses(path, records):
    fieldnames = (
        "epoch",
        "view_id_0based",
        "batch_count",
        "sample_count",
        "mean_total_loss",
        "mean_reconstruction_loss",
        "mean_cooperation_mse",
    )
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _set_global_seed(seed):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _validated_native_config(epochs):
    epochs = int(epochs)
    _require(epochs > 0, "--epochs must be positive")
    config = get_default_config(protocol.DATASET_NAME)
    training = config["training"]
    expected = {
        "batch_size": protocol.BATCH_SIZE,
        "T_1": protocol.T1,
        "T_2": protocol.T2,
        "lr": protocol.LEARNING_RATE,
        "lambda1": protocol.COOPERATION_LAMBDA,
    }
    for key, value in expected.items():
        _require(training[key] == value, "native Caltech config changed at " + key)
    training["epoch"] = epochs
    training["init_epoch"] = 0
    training["seed"] = protocol.MODEL_SEED
    config["dataset"] = protocol.DATASET_NAME
    return config


def _load_label_free_training_inputs(data_path, backbone_dir):
    """Load/replay only X1..X6 and validate the frozen noisy D1 condition."""
    clean_views = g2.load_caltech_views_only(data_path)
    noisy_views, corruption_audit = apply_weak_quality_protocol(
        clean_views,
        mode="heterogeneous_gaussian",
        k=3,
        snr_db=2.5,
        corruption_seed=protocol.MODEL_SEED,
    )
    source_mask_path, source_audit_path = d2._validate_frozen_condition(
        Path(backbone_dir), corruption_audit
    )
    provenance = {
        "data_path": protocol.display_path(data_path),
        "condition": "snr2p5_k3_seed20",
        "corruption_mask_sha256": corruption_audit["mask_sha256"],
        "source_mask_path": protocol.display_path(source_mask_path),
        "source_audit_path": protocol.display_path(source_audit_path),
        "view_shapes": [list(view.shape) for view in noisy_views],
        "view_dtypes": [str(view.dtype) for view in noisy_views],
    }
    # No corruption mask is retained or passed into any target constructor.
    corruption_audit.pop("mask", None)
    corruption_audit.pop("corruption_pairs", None)
    return noisy_views, provenance


def _load_arm_model(config, noisy_views, backbone_dir, device):
    models, _, load_audit = d2._load_native_representations(
        config=config,
        evaluation_views=noisy_views,
        backbone_dir=Path(backbone_dir),
        sample_num=protocol.SAMPLE_NUM,
        view_num=protocol.VIEW_NUM,
        cluster_num=protocol.CLASS_NUM,
        model_seed=protocol.MODEL_SEED,
    )
    models.to_device(device)
    for autoencoder in models.autoencoders:
        autoencoder.train()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(True)
            parameter.grad = None
    return models, load_audit


def _make_native_kmeans():
    return KMeans(
        n_clusters=protocol.CLASS_NUM,
        n_init=100,
        random_state=protocol.MODEL_SEED,
    )


def _native_fusion_updates(models, view_tensors, kmeans, starting_weights):
    """Replay native T1 fusion updates. This API deliberately has no U input."""
    weights = np.ascontiguousarray(starting_weights, dtype=np.float64).copy()
    _require(
        weights.shape == (protocol.VIEW_NUM,)
        and np.isfinite(weights).all()
        and np.all(weights > 0.0),
        "native fusion weights must be finite positive [6] values",
    )
    update_audit = []
    final_q = None
    final_fused = None
    final_prediction = None
    final_weights_used = None
    for update_id in range(protocol.T1):
        scaled_weighted_views = []
        q_views = []
        final_weights_used = weights.copy()
        with torch.no_grad():
            for view_id in range(protocol.VIEW_NUM):
                _, z_view, q_view = models.autoencoders[view_id](
                    view_tensors[view_id]
                )
                scaled_z = preprocessing.MinMaxScaler().fit_transform(
                    z_view.detach().cpu().numpy()
                )
                scaled_weighted_views.append(
                    scaled_z * final_weights_used[view_id]
                )
                q_views.append(q_view.detach())
        final_q = torch.stack(q_views, dim=1).detach()
        final_fused = np.hstack(scaled_weighted_views)
        final_prediction = np.asarray(
            kmeans.fit_predict(final_fused), dtype=np.int64
        )
        local_prediction = final_q.detach().cpu().numpy().argmax(axis=2)
        next_weights = np.asarray(
            [
                np.exp(
                    np.round(
                        ClusteringTest.nmi(
                            final_prediction, local_prediction[:, view_id]
                        ),
                        5,
                    )
                )
                for view_id in range(protocol.VIEW_NUM)
            ],
            dtype=np.float64,
        )
        update_audit.append({
            "update_id": int(update_id),
            "fusion_weights_used": final_weights_used.tolist(),
            "latent_fusion_sha256": tensor_sha256(final_fused),
            "global_prediction_sha256": tensor_sha256(final_prediction),
            "q_sha256": tensor_sha256(final_q),
            "native_nmi_weights_after": next_weights.tolist(),
        })
        weights = next_weights

    _require(
        tuple(final_q.shape)
        == (protocol.SAMPLE_NUM, protocol.VIEW_NUM, protocol.CLASS_NUM)
        and final_fused.shape
        == (protocol.SAMPLE_NUM, protocol.VIEW_NUM * protocol.LATENT_DIM)
        and final_prediction.shape == (protocol.SAMPLE_NUM,),
        "native fusion output boundary mismatch",
    )
    return {
        "q_local": final_q,
        "latent_fusion": np.ascontiguousarray(final_fused),
        "global_prediction": np.ascontiguousarray(final_prediction),
        "fusion_weights_used_for_P_all": np.ascontiguousarray(
            final_weights_used, dtype=np.float64
        ),
        "next_fusion_weights": np.ascontiguousarray(weights, dtype=np.float64),
        "fusion_update_audit": update_audit,
    }


def refresh_native_global_target(models, view_tensors, kmeans, starting_weights, device):
    """Create native P_all and Match matrices without Utility or labels."""
    reference = _native_fusion_updates(
        models, view_tensors, kmeans, starting_weights
    )
    q_local = reference["q_local"]
    global_prediction = reference["global_prediction"]
    local_prediction = q_local.detach().cpu().numpy().argmax(axis=2)
    matrices = []
    for view_id in range(protocol.VIEW_NUM):
        _, _, _, matrix = models.Match(
            local_prediction[:, view_id], global_prediction
        )
        matrices.append(np.ascontiguousarray(matrix, dtype=np.int64))
    alignment_matrices = np.stack(matrices, axis=0)
    protocol.validate_alignment_matrices(alignment_matrices)

    centers = np.ascontiguousarray(kmeans.cluster_centers_)
    new_p = models.new_P(reference["latent_fusion"], centers)
    p_all_array = models.target_distribution(new_p)
    p_all = protocol.numpy_to_torch_boundary(
        p_all_array,
        dtype=torch.float32,
        device=device,
    )
    protocol._validate_probability_target(p_all, "P_all")
    _require(
        not p_all.requires_grad and p_all.grad_fn is None,
        "native P_all must be stop-gradient",
    )
    reference.update({
        "alignment_matrices": alignment_matrices,
        "global_centers": centers,
        "new_P": np.ascontiguousarray(new_p),
        "P_all": p_all,
    })
    return reference


def _construct_arm_target(arm, native_refresh, utility_policy):
    """Apply U only after native fusion, KMeans, NMI, and Match are complete."""
    alignment = protocol.align_detached_q(
        native_refresh["q_local"], native_refresh["alignment_matrices"]
    )
    high = protocol.build_high_utility_target(
        alignment["aligned_q"],
        utility_policy["admission"],
        native_refresh["fusion_weights_used_for_P_all"],
    )
    p_all = native_refresh["P_all"]
    p_train = protocol.select_training_target(
        arm,
        p_all,
        p_high_u=high["P_highU"],
        rho=utility_policy["rho"],
    )
    base_identity = {
        "same_object": bool(p_train is p_all),
        "array_equal": bool(torch.equal(p_train, p_all)),
        "P_base_sha256": tensor_sha256(p_train),
        "P_all_sha256": tensor_sha256(p_all),
    }
    base_identity["sha256_equal"] = bool(
        base_identity["P_base_sha256"] == base_identity["P_all_sha256"]
    )
    if arm == "BASE":
        _require(
            base_identity["same_object"]
            and base_identity["array_equal"]
            and base_identity["sha256_equal"],
            "BASE must be a byte-exact direct P_all identity",
        )
    target_diagnostics = protocol.target_diagnostics(
        p_all,
        high["P_highU"],
        p_train,
        utility_policy["rho"],
    )
    hashes = protocol.target_hashes(
        p_all,
        native_refresh["alignment_matrices"],
        alignment["q_detached"],
        alignment["aligned_q"],
        high["P_highU"],
        utility_policy["rho"],
        p_train,
    )
    return {
        "P_train": p_train,
        "P_highU": high["P_highU"],
        "aligned_q": alignment["aligned_q"],
        "q_detached": alignment["q_detached"],
        "target_diagnostics": target_diagnostics,
        "target_hashes": hashes,
        "base_identity": base_identity,
    }


def _save_refresh_artifact(path, epoch, native_refresh, constructed, utility_policy):
    np.savez_compressed(
        path,
        epoch=np.asarray(epoch, dtype=np.int64),
        sample_ids=np.arange(protocol.SAMPLE_NUM, dtype=np.int64),
        P_all=native_refresh["P_all"].detach().cpu().numpy(),
        M=native_refresh["alignment_matrices"],
        q_detached=constructed["q_detached"].detach().cpu().numpy(),
        aligned_q=constructed["aligned_q"].detach().cpu().numpy(),
        P_highU=constructed["P_highU"].detach().cpu().numpy(),
        rho=np.asarray(utility_policy["rho"], dtype=np.float64),
        P_util=constructed["P_train"].detach().cpu().numpy(),
        admission=np.asarray(utility_policy["admission"], dtype=bool),
        fusion_weights_used_for_P_all=native_refresh[
            "fusion_weights_used_for_P_all"
        ],
        next_fusion_weights=native_refresh["next_fusion_weights"],
    )


def _protocol_record(arm, epochs, device, paths, frozen_u, data_provenance):
    return {
        "stage": protocol.STAGE,
        "arm": arm,
        "dataset": protocol.DATASET_NAME,
        "condition": "snr2p5_k3_seed20",
        "model_seed": protocol.MODEL_SEED,
        "control_seed": protocol.CONTROL_SEED,
        "N": protocol.SAMPLE_NUM,
        "V": protocol.VIEW_NUM,
        "K": protocol.CLASS_NUM,
        "latent_dim": protocol.LATENT_DIM,
        "epochs": int(epochs),
        "optimization_epoch_indices": [0, int(epochs) - 1],
        "epoch_loop": "range(epochs)",
        "refresh_rule": "epoch % T2 == 0",
        "T1": protocol.T1,
        "T2": protocol.T2,
        "batch_size": protocol.BATCH_SIZE,
        "learning_rate": protocol.LEARNING_RATE,
        "optimizer": "torch.optim.Adam(default betas/eps/weight_decay)",
        "cooperation_lambda": protocol.COOPERATION_LAMBDA,
        "target_formula": "(1-rho_i)*P_all_i + rho_i*P_highU_i",
        "rho_formula": "u_(3)-u_(4)",
        "top_k": protocol.TOP_K,
        "base_target": "direct P_all identity",
        "local_target": "P_train @ M_v",
        "loss": "MSE(x_hat_v,x_v) + 0.01*MSE(q_v,P_train@M_v)",
        "P_all_source": "native all-view weighted latent fusion -> KMeans -> new_P -> target_distribution",
        "P_highU_source": "detached aligned q weighted by current-P_all fusion weights over frozen-U Top3",
        "labels_during_training": 0,
        "cluster_count_source": "fixed preregistered K=7",
        "checkpoint_selection": "none; save final state only",
        "device": str(device),
        "paths": paths,
        "U_sha256": frozen_u["U_sha256"],
        "utility_file_sha256": frozen_u["utility_file_sha256"],
        "data_provenance": data_provenance,
    }


def _train_one_arm(
    arm,
    epochs,
    config,
    noisy_views,
    utility_policy,
    backbone_dir,
    arm_dir,
    device,
    frozen_u,
    data_provenance,
):
    _set_global_seed(protocol.MODEL_SEED)
    arm_dir.mkdir(parents=True, exist_ok=False)
    final_models_dir = arm_dir / "final_models"
    final_models_dir.mkdir()
    refresh_audit_path = arm_dir / "refresh_audit.jsonl"

    models, load_audit = _load_arm_model(
        config, noisy_views, backbone_dir, device
    )
    initial_hash = hash_backbone(models.autoencoders)
    initial_record = {
        "checkpoint_paths": load_audit["checkpoint_paths"],
        "checkpoint_file_sha256": load_audit["checkpoint_file_sha256"],
        "loaded_state_hashes": initial_hash,
    }
    _write_json(arm_dir / "initial_state_hashes.json", initial_record)

    paths = {
        "backbone_dir": protocol.display_path(backbone_dir),
        "d2_utility_path": protocol.display_path(frozen_u["utility_path"]),
        "output_dir": protocol.display_path(arm_dir),
    }
    _write_json(
        arm_dir / "protocol.json",
        _protocol_record(
            arm, epochs, device, paths, frozen_u, data_provenance
        ),
    )

    view_tensors = [
        protocol.numpy_to_torch_boundary(view, dtype=torch.float32, device=device)
        for view in noisy_views
    ]
    sample_ids_full = np.arange(protocol.SAMPLE_NUM, dtype=np.int64)
    generator = torch.Generator()
    generator.manual_seed(protocol.MODEL_SEED)
    optimizers = [
        torch.optim.Adam(
            itertools.chain(models.autoencoders[view_id].parameters()),
            lr=protocol.LEARNING_RATE,
        )
        for view_id in range(protocol.VIEW_NUM)
    ]
    kmeans = _make_native_kmeans()
    fusion_weights = np.ones(protocol.VIEW_NUM, dtype=np.float64)
    p_train = None
    alignment_matrices = None
    target_stop_gradient_pass = True
    q_gradient_path_pass = True
    sample_target_alignment_pass = True
    epoch_sample_coverage_pass = True
    base_identity_pass = True
    loss_records = []
    refresh_count = 0

    for epoch in range(int(epochs)):
        if epoch % protocol.T2 == 0:
            native_refresh = refresh_native_global_target(
                models, view_tensors, kmeans, fusion_weights, device
            )
            constructed = _construct_arm_target(
                arm, native_refresh, utility_policy
            )
            p_train = constructed["P_train"]
            alignment_matrices = native_refresh["alignment_matrices"]
            fusion_weights = native_refresh["next_fusion_weights"]
            target_stop_gradient_pass = bool(
                target_stop_gradient_pass
                and not p_train.requires_grad
                and p_train.grad_fn is None
                and not constructed["P_highU"].requires_grad
                and constructed["P_highU"].grad_fn is None
                and not constructed["aligned_q"].requires_grad
                and constructed["aligned_q"].grad_fn is None
            )
            if arm == "BASE":
                base_identity_pass = bool(
                    base_identity_pass
                    and constructed["base_identity"]["same_object"]
                    and constructed["base_identity"]["array_equal"]
                    and constructed["base_identity"]["sha256_equal"]
                )
            refresh_record = {
                "epoch": int(epoch),
                "arm": arm,
                **constructed["target_hashes"],
                **constructed["target_diagnostics"],
                "fusion_weights_used_for_P_all": native_refresh[
                    "fusion_weights_used_for_P_all"
                ].tolist(),
                "next_fusion_weights": native_refresh[
                    "next_fusion_weights"
                ].tolist(),
                "fusion_update_audit": native_refresh["fusion_update_audit"],
                "P_all_shape": list(native_refresh["P_all"].shape),
                "M_shape": list(alignment_matrices.shape),
                "q_shape": list(constructed["q_detached"].shape),
                "aligned_q_shape": list(constructed["aligned_q"].shape),
                "P_highU_shape": list(constructed["P_highU"].shape),
                "rho_shape": list(utility_policy["rho"].shape),
                "P_util_shape": list(p_train.shape),
                "target_requires_grad": bool(p_train.requires_grad),
                "target_grad_fn_is_none": bool(p_train.grad_fn is None),
                "base_identity": constructed["base_identity"],
            }
            _append_jsonl(refresh_audit_path, refresh_record)
            _save_refresh_artifact(
                arm_dir / ("refresh_epoch_%04d.npz" % epoch),
                epoch,
                native_refresh,
                constructed,
                utility_policy,
            )
            refresh_count += 1

        _require(
            p_train is not None and alignment_matrices is not None,
            "training target was not refreshed at epoch zero",
        )
        dataset = protocol.build_training_dataset(
            view_tensors, sample_ids_full, p_train
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=protocol.BATCH_SIZE,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        sums = np.zeros((protocol.VIEW_NUM, 3), dtype=np.float64)
        sample_counts = np.zeros(protocol.VIEW_NUM, dtype=np.int64)
        batch_counts = np.zeros(protocol.VIEW_NUM, dtype=np.int64)
        seen_ids = np.zeros(protocol.SAMPLE_NUM, dtype=bool)
        for x_data in loader:
            sample_ids_batch = x_data[-2]
            p_batch = x_data[-1]
            sample_target_alignment_pass = bool(
                sample_target_alignment_pass
                and protocol.batch_sample_target_alignment_pass(
                    sample_ids_batch, p_batch, p_train
                )
            )
            batch_ids_numpy = sample_ids_batch.detach().cpu().numpy()
            seen_ids[batch_ids_numpy] = True
            batch_n = int(sample_ids_batch.shape[0])
            x_h = []
            q_views = []
            for view_id in range(protocol.VIEW_NUM):
                reconstructed, _, q_view = models.autoencoders[view_id](
                    x_data[view_id]
                )
                x_h.append(reconstructed)
                q_views.append(q_view)
            q_gradient_path_pass = bool(
                q_gradient_path_pass
                and all(q_view.requires_grad for q_view in q_views)
            )
            local_targets = protocol.build_local_targets(
                p_batch, alignment_matrices
            )
            for view_id in range(protocol.VIEW_NUM):
                reconstruction_loss = F.mse_loss(x_h[view_id], x_data[view_id])
                cooperation_mse = F.mse_loss(
                    q_views[view_id], local_targets[view_id]
                )
                loss = (
                    reconstruction_loss
                    + protocol.COOPERATION_LAMBDA * cooperation_mse
                )
                optimizers[view_id].zero_grad()
                loss.backward()
                optimizers[view_id].step()
                sums[view_id, 0] += float(loss.item()) * batch_n
                sums[view_id, 1] += float(reconstruction_loss.item()) * batch_n
                sums[view_id, 2] += float(cooperation_mse.item()) * batch_n
                sample_counts[view_id] += batch_n
                batch_counts[view_id] += 1
        epoch_sample_coverage_pass = bool(
            epoch_sample_coverage_pass and np.all(seen_ids)
        )
        for view_id in range(protocol.VIEW_NUM):
            denominator = float(sample_counts[view_id])
            loss_records.append({
                "epoch": int(epoch),
                "view_id_0based": int(view_id),
                "batch_count": int(batch_counts[view_id]),
                "sample_count": int(sample_counts[view_id]),
                "mean_total_loss": float(sums[view_id, 0] / denominator),
                "mean_reconstruction_loss": float(
                    sums[view_id, 1] / denominator
                ),
                "mean_cooperation_mse": float(sums[view_id, 2] / denominator),
            })

    _write_training_losses(arm_dir / "training_losses.csv", loss_records)

    # Fix an all-view native prediction from the final trained state. No labels
    # are loaded here, and Utility is not an argument to this path.
    final_reference = _native_fusion_updates(
        models, view_tensors, _make_native_kmeans(), fusion_weights
    )
    final_predictions = final_reference["global_prediction"]
    prediction_path = arm_dir / "final_predictions.npy"
    np.save(prediction_path, final_predictions, allow_pickle=False)
    prediction_hash = tensor_sha256(final_predictions)

    final_hash = hash_backbone(models.autoencoders)
    final_checkpoint_hashes = []
    for view_id, checkpoint_name in enumerate(CHECKPOINT_NAMES):
        checkpoint_path = final_models_dir / checkpoint_name
        torch.save(models.autoencoders[view_id].state_dict(), checkpoint_path)
        final_checkpoint_hashes.append(protocol.b7_protocol.file_sha256(checkpoint_path))
    _write_json(
        arm_dir / "final_state_hashes.json",
        {
            "model_state_hashes": final_hash,
            "checkpoint_paths": [
                protocol.display_path(final_models_dir / name)
                for name in CHECKPOINT_NAMES
            ],
            "checkpoint_file_sha256": final_checkpoint_hashes,
            "final_prediction_sha256": prediction_hash,
        },
    )

    expected_refresh_count = ((int(epochs) - 1) // protocol.T2) + 1
    engineering_checks = {
        "fixed_K7_pass": True,
        "zero_training_labels_pass": True,
        "frozen_U_hash_pass": bool(
            frozen_u["U_sha256"] == protocol.EXPECTED_U_SHA256
            and frozen_u["utility_file_sha256"]
            == protocol.EXPECTED_UTILITY_FILE_SHA256
        ),
        "sample_ids_are_arange_N_pass": True,
        "sample_target_alignment_pass": sample_target_alignment_pass,
        "epoch_sample_coverage_pass": epoch_sample_coverage_pass,
        "target_stop_gradient_pass": target_stop_gradient_pass,
        "q_gradient_path_pass": q_gradient_path_pass,
        "base_exact_identity_pass": bool(
            base_identity_pass if arm == "BASE" else True
        ),
        "refresh_count_pass": bool(refresh_count == expected_refresh_count),
        "no_checkpoint_selection_pass": True,
        "no_semantic_head_pass": bool(models.semantic_heads is None),
        "final_prediction_fixed_before_labels_pass": bool(prediction_path.is_file()),
    }
    audit = {
        "stage": protocol.STAGE,
        "arm": arm,
        "ENGINEERING_PASS": bool(all(engineering_checks.values())),
        "engineering_checks": engineering_checks,
        "initial_state_hashes": initial_hash,
        "final_state_hashes": final_hash,
        "refresh_count": int(refresh_count),
        "expected_refresh_count": int(expected_refresh_count),
        "optimizer_count": len(optimizers),
        "optimizer_types": [type(optimizer).__name__ for optimizer in optimizers],
        "labels_loaded_during_training": False,
        "final_prediction_sha256": prediction_hash,
        "final_prediction_shape": list(final_predictions.shape),
        "final_prediction_fusion_weights_used": final_reference[
            "fusion_weights_used_for_P_all"
        ].tolist(),
        "utility_policy_audit": {
            "changed_row_count": utility_policy["changed_row_count"],
            "changed_top3_row_count": utility_policy[
                "changed_top3_row_count"
            ],
            "rho_multiset_preserved": utility_policy[
                "rho_multiset_preserved"
            ],
            "top3_column_counts_preserved": utility_policy[
                "top3_column_counts_preserved"
            ],
            "row_permutation_sha256": tensor_sha256(
                utility_policy["row_permutation"]
            ),
            "admission_sha256": tensor_sha256(utility_policy["admission"]),
            "rho_sha256": tensor_sha256(utility_policy["rho"]),
        },
    }
    _write_json(arm_dir / "audit.json", audit)
    del optimizers
    del models
    del view_tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return {
        "arm": arm,
        "arm_dir": arm_dir,
        "initial_state_hashes": initial_hash,
        "prediction_path": prediction_path,
        "prediction_sha256": prediction_hash,
        "engineering_pass": audit["ENGINEERING_PASS"],
    }


def _evaluate_fixed_predictions_after_seal(arm_results, data_path, output_root):
    """Load labels only after all three prediction artifacts already exist."""
    _require(
        len(arm_results) == len(protocol.ARMS)
        and all(result["prediction_path"].is_file() for result in arm_results),
        "all three predictions must be sealed before label loading",
    )
    initial_hashes = [result["initial_state_hashes"] for result in arm_results]
    identical_initial_states = bool(
        all(value == initial_hashes[0] for value in initial_hashes[1:])
    )
    _require(identical_initial_states, "arm initial model states differ")
    labels = g2.load_caltech_labels_only(data_path)
    metrics = {}
    for result in arm_results:
        prediction = np.load(result["prediction_path"], allow_pickle=False)
        _require(
            tensor_sha256(prediction) == result["prediction_sha256"],
            "fixed prediction changed before evaluation",
        )
        arm_metrics = g2.metrics_from_fixed_prediction(labels, prediction)
        metrics[result["arm"]] = arm_metrics
        _write_json(result["arm_dir"] / "final_metrics.json", arm_metrics)
        audit_path = result["arm_dir"] / "audit.json"
        audit = _read_json(audit_path)
        audit["labels_loaded_after_all_predictions_sealed"] = True
        audit["final_metrics"] = arm_metrics
        _write_json(audit_path, audit)

    def delta(left, right):
        return {
            "delta_ACC": float(metrics[left]["ACC"] - metrics[right]["ACC"]),
            "delta_NMI": float(metrics[left]["NMI"] - metrics[right]["NMI"]),
            "delta_ARI": float(metrics[left]["ARI"] - metrics[right]["ARI"]),
        }

    comparison = {
        "stage": protocol.STAGE,
        "metrics": metrics,
        "primary_gate_U_vs_SHUFFLED_U": delta(
            "U_CORRECTION", "SHUFFLED_U_CORRECTION"
        ),
        "second_gate_U_vs_BASE": delta("U_CORRECTION", "BASE"),
        "initial_state_hashes_identical": identical_initial_states,
        "all_predictions_sealed_before_labels": True,
        "all_arm_engineering_pass": bool(
            all(result["engineering_pass"] for result in arm_results)
        ),
        "scientific_pass_threshold": None,
    }
    _write_json(output_root / "comparison_summary.json", comparison)
    return comparison


def run_experiment(
    epochs,
    output_dir,
    backbone_dir=protocol.DEFAULT_BACKBONE_DIR,
    d2_dir=protocol.DEFAULT_D2_DIR,
    data_path=protocol.DEFAULT_DATA_PATH,
    device=None,
):
    """Train all arms first, seal predictions, then evaluate once with labels."""
    epochs = int(epochs)
    output_root = protocol.resolve_path(output_dir)
    backbone_root = protocol.resolve_path(backbone_dir)
    d2_root = protocol.resolve_path(d2_dir)
    data_source = protocol.resolve_path(data_path)
    _require(not output_root.exists(), "refusing to overwrite a G3 output directory")
    output_root.mkdir(parents=True)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA was requested but is unavailable")

    config = _validated_native_config(epochs)
    frozen_u = protocol.load_frozen_u_only(d2_root)
    noisy_views, data_provenance = _load_label_free_training_inputs(
        data_source, backbone_root
    )
    policies = {
        arm: protocol.build_utility_policy(
            frozen_u["U"], arm, control_seed=protocol.CONTROL_SEED
        )
        for arm in protocol.ARMS
    }
    arm_results = []
    for arm in protocol.ARMS:
        arm_results.append(
            _train_one_arm(
                arm=arm,
                epochs=epochs,
                config=config,
                noisy_views=noisy_views,
                utility_policy=policies[arm],
                backbone_dir=backbone_root,
                arm_dir=output_root / ARM_DIRECTORIES[arm],
                device=device,
                frozen_u=frozen_u,
                data_provenance=data_provenance,
            )
        )
    # This is the sole label-loading boundary and occurs only after every arm's
    # final_predictions.npy has been written and content-hashed.
    return _evaluate_fixed_predictions_after_seal(
        arm_results, data_source, output_root
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="G3-A0 frozen-utility semantic target correction"
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backbone-dir", type=Path, default=protocol.DEFAULT_BACKBONE_DIR
    )
    parser.add_argument("--d2-dir", type=Path, default=protocol.DEFAULT_D2_DIR)
    parser.add_argument("--data-path", type=Path, default=protocol.DEFAULT_DATA_PATH)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    comparison = run_experiment(
        epochs=args.epochs,
        output_dir=args.output_dir,
        backbone_dir=args.backbone_dir,
        d2_dir=args.d2_dir,
        data_path=args.data_path,
        device=args.device,
    )
    print(json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
