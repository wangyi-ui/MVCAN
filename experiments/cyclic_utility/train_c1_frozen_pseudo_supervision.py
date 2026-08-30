"""Train one frozen C1 pseudo-supervision arm from the E1-LWC seal.

This entry point deliberately has no dynamic pseudo-label refresh.  C0 arrays
and M0 are loaded/derived once before optimization, while MVCAN's native
P_all/Match refresh and E1 LWC objective remain unchanged.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import (
    c1_frozen_pseudo_supervision as c1,
)
from experiments.cyclic_utility import (
    evaluate_c0_complementary_semantic_verification as c0_evaluate,
)
from experiments.e1_pairwise_utility import (
    train_e1_pairwise_utility as e1_train,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    align_semantic_probabilities,
    pairwise_semantic_cooperation_loss,
    robust_inter_affinity,
)
from irv.b3_audit import hash_backbone
from irv.b4_information_utility import tensor_sha256


STAGE = "C1"
DATASET = "Caltech-6V"
DEFAULT_C0_DIR = (
    REPOSITORY_ROOT
    / "outputs/cyclic_utility/c0_complementary_semantic_verification_seed20"
)
DEFAULT_C0_ARTIFACT_PATH = DEFAULT_C0_DIR / "c0_predictions_and_scores.npz"
DEFAULT_C0_SEAL_PATH = DEFAULT_C0_DIR / "c0_prediction_seal.json"
DEFAULT_FEATURE_PATH = e1_train.DEFAULT_FEATURE_PATH
DEFAULT_FEATURE_AUDIT_PATH = e1_train.DEFAULT_FEATURE_AUDIT_PATH
DEFAULT_E1_LWC_MODEL_DIR = c0_evaluate.DEFAULT_E1_LWC_MODEL_DIR
DEFAULT_E1_LWC_AUDIT_PATH = c0_evaluate.DEFAULT_E1_LWC_AUDIT_PATH
DEFAULT_FULL_GT_PATH = c0_evaluate.DEFAULT_FULL_GT_PATH
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "outputs/cyclic_utility/c1_frozen_pseudo_supervision_seed20"
)

FROZEN_SOURCE_SHA256 = {
    "experiments/cyclic_utility/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "experiments/cyclic_utility/c0_complementary_semantic_verification.py": (
        "d52fbf0816557ba57a11fc490ead1a26598b35e68bac7808b7077c448bc7a4a9"
    ),
    "experiments/cyclic_utility/evaluate_c0_complementary_semantic_verification.py": (
        "adb420bbc59833ff7eca58f3b5d54162a45b4106227d8f6540fa69fc22ac691b"
    ),
    "tests/test_c0_complementary_semantic_verification.py": (
        "9e9f10ebd43e4e288759bb8cbe9b55778a7991e222128d83506be3734bd07796"
    ),
    "experiments/e1_pairwise_utility/pairwise_semantic_cooperation.py": (
        "ee893ca40a0f398be70bcd7150e8e62298d07b04ba73bddbb339f3134d657bfc"
    ),
    "experiments/e1_pairwise_utility/train_e1_pairwise_utility.py": (
        "8cecd5c3a8488ba285f686d9836ecfd2263b05e6857136f123b911e80dc83d9d"
    ),
    "model.py": (
        "3f866536857f0a5df5d451dec93232c77f0805ff216db60a2dd894b0c1a357c9"
    ),
    "run.py": (
        "3a766742ca331471e7ece7f111921b7ffc44c6e883728e43ac7583741427fdc4"
    ),
    "experiments/e4_semantic_memory_bank/semantic_memory_bank.py": (
        "8c4f99e1cf31ec03eb69183445076177362e6b4adc6e6dc56350c76f41a0afe6"
    ),
    "experiments/e4_semantic_memory_bank/evaluate_e4_a0_memory_feasibility.py": (
        "5f9e26d4ef3d446f22743dc4f7900369e9a0d82bde6f13a170eaa89278e99e4c"
    ),
    "experiments/e4_semantic_memory_bank/e4a1_memory_specific_utility.py": (
        "ec0b6af7867ac19795abe7cafcc15c5e793421c1fc2d2f4c73920a8be76ac7de"
    ),
    "experiments/e4_semantic_memory_bank/evaluate_e4_a1_memory_specific_utility.py": (
        "123e04e616df758f38d383d9c5776ef7398f2078b50556919d7faa712e40f621"
    ),
    "experiments/e4_semantic_memory_bank/e4cf0_counterfactual_utility.py": (
        "2d4dbb365089363baa42eca9793e48956f4f40db8f5a404585e9f97129853438"
    ),
    "experiments/e4_semantic_memory_bank/evaluate_e4cf0_counterfactual_utility.py": (
        "a353e02c5937e7ba9cac4e6ebf84d03c34af7139871413715dce7d95e215fb25"
    ),
    "tests/test_e4_a0_semantic_memory_bank.py": (
        "4ed15f1bb3f20e4788a07f820eb685bc75de1da5239161ed593db96d54a2e28c"
    ),
    "tests/test_e4_a1_memory_specific_utility.py": (
        "8d5906b13b061cd163a2ff2db585026dfefbdf0a324d37073c0cf0a2fafd8d37"
    ),
    "tests/test_e4cf0_counterfactual_utility.py": (
        "a04cf6de39845acf0d481dedf4552c4f433fae84899b24e6def99fb3b9669d68"
    ),
}


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


def verify_frozen_source_provenance(repository_root=REPOSITORY_ROOT):
    root = Path(repository_root)
    actual = {}
    for relative_path, expected_hash in FROZEN_SOURCE_SHA256.items():
        path = root / relative_path
        _require(path.is_file(), "frozen source is missing: " + relative_path)
        actual_hash = c1.file_sha256(path)
        _require(
            actual_hash == expected_hash,
            "frozen source provenance mismatch: " + relative_path,
        )
        actual[relative_path] = actual_hash
    return {
        "expected_sha256": dict(FROZEN_SOURCE_SHA256),
        "actual_sha256": actual,
        "all_frozen_sources_unchanged_pass": True,
        "C0_source_unchanged_pass": True,
        "E1_source_unchanged_pass": True,
        "E4_source_unchanged_pass": True,
        "model_py_unchanged_pass": True,
        "run_py_unchanged_pass": True,
    }


def load_trainable_e1_lwc_model(
    model_dir, model_audit_path, device
):
    """Reload the same six E1-LWC model-only checkpoints for every arm."""
    audit_path = _resolve(model_audit_path)
    _require(audit_path.is_file(), "E1 LWC audit is missing")
    frozen_run = c1.read_json(audit_path)
    _require(
        frozen_run.get("arm") == "LWC"
        and frozen_run.get("epochs") == 100
        and frozen_run.get("seed") == c1.SEED
        and frozen_run.get("N") == c1.SAMPLE_NUM
        and frozen_run.get("V") == c1.VIEW_NUM
        and frozen_run.get("K") == c1.CLASS_NUM,
        "E1 audit is not the frozen 100-epoch seed20 LWC run",
    )
    model, config, checkpoint_audit = e1_train.build_model_from_frozen_d1(
        _resolve(model_dir), device
    )
    initialization = c1.verify_e1_initialization(checkpoint_audit)
    expected_output_hashes = frozen_run["final_model_outputs"]["file_sha256"]
    _require(
        checkpoint_audit["checkpoint_file_sha256"] == expected_output_hashes
        and frozen_run["final_model_hash"]["aggregate"]
        == initialization["initial_model_aggregate_sha256"],
        "E1 LWC checkpoint audit binding mismatch",
    )
    return model, config, {
        **checkpoint_audit,
        **initialization,
        "source_arm": "LWC",
        "source_epochs": 100,
        "source_seed": c1.SEED,
        "source_audit_path": _display(audit_path),
        "independently_reloaded_for_this_arm": True,
    }


def build_lwc_loss(q_local, native_matches):
    """Exact E1 LWC branch; the LWC arm does not consume D2 reliability."""
    q_aligned, h_sem = align_semantic_probabilities(
        q_local, native_matches
    )
    graph_inter = robust_inter_affinity(h_sem.detach())
    # The released API validates this argument for every arm.  In the LWC arm
    # positive_pair_weights uses only G_inter and never reads this tensor.
    unused_lwc_argument = torch.zeros(
        q_local.shape[0],
        c1.VIEW_NUM,
        dtype=q_local.dtype,
        device=q_local.device,
    ).detach()
    loss, diagnostics = pairwise_semantic_cooperation_loss(
        h_sem,
        graph_inter,
        unused_lwc_argument,
        "LWC",
    )
    _require(
        q_aligned.requires_grad
        and not graph_inter.requires_grad
        and graph_inter.grad_fn is None
        and all(record["r_mean"] is None for record in diagnostics),
        "native E1 LWC stop-gradient/no-R boundary mismatch",
    )
    return loss, diagnostics


def coverage_record(visited_ids):
    ids = np.concatenate(visited_ids).astype(np.int64, copy=False)
    expected = np.arange(c1.SAMPLE_NUM, dtype=np.int64)
    exact = bool(
        ids.size == c1.SAMPLE_NUM
        and np.array_equal(np.sort(ids), expected)
        and np.unique(ids).size == c1.SAMPLE_NUM
    )
    _require(exact, "epoch did not cover every sample ID exactly once")
    return {
        "sample_count": int(ids.size),
        "unique_sample_count": int(np.unique(ids).size),
        "all_1400_sample_ids_exactly_once_pass": True,
        "sample_order_sha256": tensor_sha256(ids),
    }


def train_c1_arm(
    arm,
    epochs,
    model,
    optimizers,
    views,
    sample_ids,
    pseudo_target,
    device,
    batch_size=c1.BATCH_SIZE,
):
    """Continue native E1-LWC and add only the frozen auxiliary loss."""
    if arm not in c1.ALL_ARMS:
        raise ValueError("unknown C1 arm")
    if int(epochs) <= 0:
        raise ValueError("C1 epochs must be positive")
    if int(batch_size) != c1.BATCH_SIZE:
        raise ValueError("C1 batch size is frozen at 256")
    _require(
        (arm == "BASE" and pseudo_target is None)
        or (arm != "BASE" and pseudo_target is not None),
        "C1 arm/pseudo-target boundary mismatch",
    )

    generator = torch.Generator()
    generator.manual_seed(c1.SEED)
    full_views = [torch.from_numpy(view) for view in views]
    id_tensor = torch.from_numpy(
        np.asarray(sample_ids, dtype=np.int64)
    )
    view_weights = [1.0] * c1.VIEW_NUM
    native_p_all = None
    native_matches = None
    native_target_refresh_count = 0
    pseudo_lookup_count = 0
    epoch_records = []
    lwc_no_R_pass = True
    pseudo_detached_pass = True

    for epoch in range(int(epochs)):
        if epoch % e1_train.TARGET_REFRESH_INTERVAL == 0:
            (
                native_p_all,
                native_matches,
                _,
                view_weights,
            ) = e1_train.refresh_native_target(
                model, full_views, view_weights, device
            )
            native_target_refresh_count += 1
        _require(
            native_p_all is not None and native_matches is not None,
            "native MVCAN target is missing",
        )
        dataset = torch.utils.data.TensorDataset(
            *full_views, native_p_all, id_tensor
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=True,
            drop_last=False,
            generator=generator,
            num_workers=0,
        )
        visited = []
        native_sum = 0.0
        reconstruction_sum = 0.0
        clustering_sum = 0.0
        lwc_sum = 0.0
        pseudo_sum = 0.0
        total_sum = 0.0
        batch_count = 0

        for packed in loader:
            x_views = [
                packed[view_id].to(device)
                for view_id in range(c1.VIEW_NUM)
            ]
            p_batch = packed[c1.VIEW_NUM].to(device)
            sample_ids_batch = packed[c1.VIEW_NUM + 1]
            visited.append(sample_ids_batch.numpy())
            for optimizer in optimizers:
                optimizer.zero_grad()

            reconstructions = []
            q_views = []
            for view_id in range(c1.VIEW_NUM):
                x_hat, _, q_local_view = model.autoencoders[view_id](
                    x_views[view_id]
                )
                reconstructions.append(x_hat)
                q_views.append(q_local_view)
            q_local = torch.stack(q_views, dim=1)
            native_view_losses, native_diagnostics = (
                e1_train.native_mvcan_losses(
                    x_views,
                    reconstructions,
                    q_views,
                    p_batch,
                    native_matches,
                )
            )
            rec_total = torch.stack(
                [record[0] for record in native_diagnostics]
            ).sum()
            clu_total = torch.stack(
                [record[1] for record in native_diagnostics]
            ).sum()
            rec_clu_total = torch.stack(native_view_losses).sum()
            lwc_loss, lwc_diagnostics = build_lwc_loss(
                q_local, native_matches
            )
            lwc_no_R_pass = bool(
                lwc_no_R_pass
                and all(
                    record["r_mean"] is None
                    for record in lwc_diagnostics
                )
            )
            native_e1_lwc_loss = (
                rec_clu_total + e1_train.LAMBDA1 * lwc_loss
            )

            pseudo_loss = None
            if arm != "BASE":
                target_batch, strength_batch = (
                    c1.pseudo_batch_by_sample_ids(
                        pseudo_target, sample_ids_batch
                    )
                )
                pseudo_lookup_count += 1
                pseudo_detached_pass = bool(
                    pseudo_detached_pass
                    and not target_batch.requires_grad
                    and target_batch.grad_fn is None
                    and not strength_batch.requires_grad
                    and strength_batch.grad_fn is None
                )
                pseudo_loss = c1.soft_pseudo_cross_entropy(
                    q_local, target_batch, strength_batch
                )
            total_loss = c1.combine_native_and_pseudo_loss(
                arm, native_e1_lwc_loss, pseudo_loss
            )
            _require(
                bool(torch.isfinite(native_e1_lwc_loss).item())
                and bool(torch.isfinite(total_loss).item())
                and (
                    pseudo_loss is None
                    or bool(torch.isfinite(pseudo_loss).item())
                ),
                "non-finite C1 loss",
            )
            total_loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            reconstruction_sum += float(rec_total.detach().item())
            clustering_sum += float(clu_total.detach().item())
            lwc_sum += float(lwc_loss.detach().item())
            native_sum += float(native_e1_lwc_loss.detach().item())
            pseudo_sum += (
                0.0
                if pseudo_loss is None
                else float(pseudo_loss.detach().item())
            )
            total_sum += float(total_loss.detach().item())
            batch_count += 1

        coverage = coverage_record(visited)
        epoch_records.append({
            "epoch": epoch + 1,
            "batch_count": batch_count,
            "reconstruction_loss_sum_views_mean": (
                reconstruction_sum / batch_count
            ),
            "clustering_loss_sum_views_mean": (
                clustering_sum / batch_count
            ),
            "lwc_loss_mean": lwc_sum / batch_count,
            "native_e1_lwc_loss_mean": native_sum / batch_count,
            "pseudo_loss_mean": pseudo_sum / batch_count,
            "total_loss_mean": total_sum / batch_count,
            **coverage,
        })

    _, final_matches, predictions, _ = e1_train.refresh_native_target(
        model, full_views, view_weights, device
    )
    return predictions, {
        "epoch_records": epoch_records,
        "sample_order_sha256_per_epoch": [
            record["sample_order_sha256"] for record in epoch_records
        ],
        "native_target_refresh_count": native_target_refresh_count,
        "native_P_all_definition": (
            "MVCAN target_distribution(new_P(latent_fusion, centers))"
        ),
        "native_target_rewritten": False,
        "P_all_rewritten": False,
        "P_corr_used": False,
        "P_util_used": False,
        "native_Match_refreshed": True,
        "auxiliary_M0_refreshed": False,
        "final_native_Match_shape": list(final_matches.shape),
        "LWC_preserved": True,
        "LWC_R_argument_unused_pass": lwc_no_R_pass,
        "pseudo_sample_id_lookup_count": pseudo_lookup_count,
        "pseudo_sample_id_lookup_used": arm != "BASE",
        "batch_idx_used_for_semantic_indexing": False,
        "pseudo_batch_tensors_detached_pass": pseudo_detached_pass,
        "all_epochs_exact_sample_coverage_pass": True,
        "loss_finite_pass": True,
    }


def save_predictions_before_GT(predictions, sample_ids, output_dir):
    payload = c1.prediction_seal_payload(predictions, sample_ids)
    path = output_dir / "final_predictions.npz"
    np.savez(path, **payload)
    with np.load(path, allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == ("predictions", "sample_ids")
            and np.array_equal(archive["predictions"], payload["predictions"])
            and np.array_equal(archive["sample_ids"], payload["sample_ids"]),
            "C1 prediction seal reload mismatch",
        )
    return path, {
        "path": _display(path),
        "file_sha256": c1.file_sha256(path),
        "prediction_logical_sha256": tensor_sha256(
            payload["predictions"]
        ),
        "sample_ids_logical_sha256": tensor_sha256(payload["sample_ids"]),
        "shape": list(payload["predictions"].shape),
        "saved_before_GT_pass": True,
        "hashed_before_GT_pass": True,
        "reloaded_before_GT_pass": True,
    }


def run_arm(
    arm,
    epochs=c1.DEFAULT_EPOCHS,
    seed=c1.SEED,
    output_dir=None,
    device="cuda:0",
    c0_artifact_path=DEFAULT_C0_ARTIFACT_PATH,
    c0_seal_path=DEFAULT_C0_SEAL_PATH,
    feature_path=DEFAULT_FEATURE_PATH,
    feature_audit_path=DEFAULT_FEATURE_AUDIT_PATH,
    model_dir=DEFAULT_E1_LWC_MODEL_DIR,
    model_audit_path=DEFAULT_E1_LWC_AUDIT_PATH,
    full_gt_path=DEFAULT_FULL_GT_PATH,
):
    if arm not in c1.ALL_ARMS:
        raise ValueError("unknown C1 arm")
    _require(int(seed) == c1.SEED, "C1 seed is frozen at 20")
    _require(int(epochs) > 0, "C1 epochs must be positive")
    output_root = (
        DEFAULT_OUTPUT_ROOT / arm
        if output_dir is None
        else _resolve(output_dir)
    )
    _require(not output_root.exists(), "refusing to overwrite C1 output")
    target_device = torch.device(device)
    if target_device.type == "cuda":
        _require(torch.cuda.is_available(), "requested CUDA device unavailable")
        torch.cuda.set_device(target_device)

    c1.set_deterministic_seed(seed)
    source_provenance = verify_frozen_source_provenance()
    c0_arrays, c0_provenance = c1.load_frozen_c0_artifact(
        _resolve(c0_artifact_path), _resolve(c0_seal_path)
    )
    views, sample_ids, feature_provenance = (
        e1_train.load_frozen_feature_artifact(
            _resolve(feature_path), _resolve(feature_audit_path)
        )
    )
    _require(
        np.array_equal(
            sample_ids,
            np.arange(c1.SAMPLE_NUM, dtype=np.int64),
        )
        and c0_arrays["native_global_cluster"].shape
        == (c1.SAMPLE_NUM,),
        "sealed global coordinate/sample ID row alignment mismatch",
    )
    c0_provenance[
        "native_global_cluster_sample_ids_exact_arange_pass"
    ] = True
    model, config, checkpoint_provenance = (
        load_trainable_e1_lwc_model(
            model_dir, model_audit_path, target_device
        )
    )
    initial_hash = hash_backbone(model.autoencoders)
    _require(
        initial_hash["aggregate"]
        == c1.EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256,
        "C1 initial model aggregate SHA256 mismatch",
    )
    M0, M0_audit = c1.derive_frozen_M0(
        model,
        views,
        c0_arrays["native_global_cluster"],
        target_device,
    )
    _require(
        hash_backbone(model.autoencoders) == initial_hash,
        "M0 derivation changed the initial model",
    )
    pseudo_target, pseudo_audit = c1.build_frozen_pseudo_target(
        arm, c0_arrays, M0
    )
    pseudo_audit.update({
        "C0_artifact_provenance": c0_provenance,
        "M0_audit": M0_audit,
        "native_global_cluster_used_for_weighting": False,
        "native_global_cluster_used_for_pseudo_label": False,
        "native_global_cluster_used_for_M0_coordinate_only": True,
        "same_y_gen_all_pseudo_arms_contract_sha256": (
            c1.EXPECTED_C0_ARRAY_SHA256["y_gen"]
        ),
        "dynamic_refresh_used": False,
        "threshold_used": False,
        "top_k_used": False,
        "temperature_used": False,
        "entropy_multiplication_used": False,
        "EMA_utility_used": False,
    })
    optimizers = e1_train.build_fresh_optimizers(model)
    optimizer_audit = c1.optimizer_configuration_audit(optimizers)
    _require(
        int(config["training"]["batch_size"]) == c1.BATCH_SIZE,
        "native batch size mismatch",
    )

    output_root.mkdir(parents=True)
    predictions, runtime = train_c1_arm(
        arm=arm,
        epochs=epochs,
        model=model,
        optimizers=optimizers,
        views=views,
        sample_ids=sample_ids,
        pseudo_target=pseudo_target,
        device=target_device,
        batch_size=c1.BATCH_SIZE,
    )
    prediction_path, prediction_audit = save_predictions_before_GT(
        predictions, sample_ids, output_root
    )
    # This is the only full-GT boundary and is necessarily post-seal.
    labels = e1_train.load_labels_after_predictions(
        _resolve(full_gt_path), prediction_path
    )
    metrics = e1_train.evaluate_predictions(labels, predictions)
    final_hash = hash_backbone(model.autoencoders)

    loss_history = {
        "stage": STAGE,
        "arm": arm,
        "epochs": int(epochs),
        "epoch_records": runtime["epoch_records"],
    }
    metrics_record = {
        "stage": STAGE,
        "arm": arm,
        "epochs": int(epochs),
        "seed": int(seed),
        "metrics": metrics,
    }
    train_audit = {
        "stage": STAGE,
        "arm": arm,
        "formal_scientific_arm": arm in c1.FORMAL_ARMS,
        "engineering_only_arm": arm == c1.ENGINEERING_ARM,
        "epochs": int(epochs),
        "seed": int(seed),
        "N": c1.SAMPLE_NUM,
        "V": c1.VIEW_NUM,
        "K": c1.CLASS_NUM,
        "batch_size": c1.BATCH_SIZE,
        "initial_model_aggregate_sha256": initial_hash["aggregate"],
        "initial_model_hash": initial_hash,
        "final_model_aggregate_sha256": final_hash["aggregate"],
        "final_model_hash": final_hash,
        "checkpoint_provenance": checkpoint_provenance,
        "source_provenance": source_provenance,
        "feature_provenance": feature_provenance,
        "C0_artifact_provenance": c0_provenance,
        "M0_audit": M0_audit,
        "optimizer_configuration": optimizer_audit,
        "sample_order_sha256_per_epoch": runtime[
            "sample_order_sha256_per_epoch"
        ],
        "sample_ids_explicit": True,
        "sample_ids_exact_arange": True,
        "pseudo_y_gen_logical_sha256": (
            None
            if arm == "BASE"
            else c1.EXPECTED_C0_ARRAY_SHA256["y_gen"]
        ),
        "lambda_pseudo": c1.effective_lambda_pseudo(arm),
        "lambda_pseudo_sweep_used": False,
        "native_objective": "REC + 0.01*CLU + 0.01*LWC",
        "auxiliary_objective": (
            "none"
            if arm == "BASE"
            else "lambda_pseudo * mean_{i,v}(a_i * CE_soft_i_v)"
        ),
        "prediction_audit": prediction_audit,
        "prediction_logical_sha256": prediction_audit[
            "prediction_logical_sha256"
        ],
        "loss_history": runtime["epoch_records"],
        "metrics": metrics,
        "runtime": runtime,
        "leakage": {
            "full_GT_loaded_during_training": False,
            "full_GT_loaded_after_prediction_seal": True,
            "sparse_labels_loaded": False,
            "R_loaded": False,
            "corruption_mask_loaded": False,
            "oracle_used": False,
        },
        "historical_non_repetition": {
            "dynamic_pseudo_refresh_used": False,
            "dynamic_auxiliary_Match_used": False,
            "Memory_used": False,
            "P_corr_used": False,
            "P_util_used": False,
            "native_P_all_rewritten": False,
            "top_k_used": False,
            "threshold_used": False,
            "temperature_used": False,
            "entropy_multiplication_used": False,
            "EMA_utility_used": False,
        },
        "CYCLE_ZERO_full_cycle_path": (
            arm == c1.ENGINEERING_ARM
            and runtime["pseudo_sample_id_lookup_used"]
            and c1.effective_lambda_pseudo(arm) == 0.0
        ),
        "C1_ENGINEERING_PASS": True,
    }
    c1.write_json(output_root / "metrics.json", metrics_record)
    c1.write_json(output_root / "train_audit.json", train_audit)
    c1.write_json(
        output_root / "pseudo_target_audit.json", pseudo_audit
    )
    c1.write_json(output_root / "loss_history.json", loss_history)
    print("C1_ENGINEERING_PASS=True")
    print("ARM=" + arm)
    print("ACC={:.10f}".format(metrics["ACC"]))
    print("NMI={:.10f}".format(metrics["NMI"]))
    print("ARI={:.10f}".format(metrics["ARI"]))
    print("Saved: " + _display(output_root))
    return {
        "output_dir": output_root,
        "metrics": metrics_record,
        "train_audit": train_audit,
        "pseudo_target_audit": pseudo_audit,
        "loss_history": loss_history,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=c1.ALL_ARMS)
    parser.add_argument("--epochs", type=int, default=c1.DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=c1.SEED)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--c0-artifact-path", default=str(DEFAULT_C0_ARTIFACT_PATH)
    )
    parser.add_argument(
        "--c0-seal-path", default=str(DEFAULT_C0_SEAL_PATH)
    )
    parser.add_argument(
        "--feature-path", default=str(DEFAULT_FEATURE_PATH)
    )
    parser.add_argument(
        "--feature-audit-path", default=str(DEFAULT_FEATURE_AUDIT_PATH)
    )
    parser.add_argument(
        "--model-dir", default=str(DEFAULT_E1_LWC_MODEL_DIR)
    )
    parser.add_argument(
        "--model-audit-path", default=str(DEFAULT_E1_LWC_AUDIT_PATH)
    )
    parser.add_argument(
        "--full-gt-path", default=str(DEFAULT_FULL_GT_PATH)
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_arm(
        arm=args.arm,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
        c0_artifact_path=args.c0_artifact_path,
        c0_seal_path=args.c0_seal_path,
        feature_path=args.feature_path,
        feature_audit_path=args.feature_audit_path,
        model_dir=args.model_dir,
        model_audit_path=args.model_audit_path,
        full_gt_path=args.full_gt_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
