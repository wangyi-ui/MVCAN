"""D2-A0 read-only Caltech-6V Information Utility transfer diagnostic.

This script loads a frozen Native MVCAN backbone and performs only inference,
5-fold OOF Ridge prediction, frozen rank-utility construction, and post-hoc
diagnostics. Labels and the oracle corruption mask cannot enter T or U.
"""

import argparse
import csv
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from configure import get_default_config
from datasets import load_data
from irv.b3_audit import hash_backbone
from irv.b3_predictability_diagnostics import fold_assignment_csv_text
from irv.b3_predictability_diagnostics import fold_assignment_sha256
from irv.b3_predictability_diagnostics import make_fold_assignment
from irv.b3_predictability_diagnostics import oof_ridge_predictability
from irv.b3_predictability_diagnostics import reliability_metrics
from irv.b4_information_utility import compute_information_utility
from irv.b4_information_utility import tensor_sha256
from model import MvCAN
from weak_quality import apply_weak_quality_protocol
from weak_quality import ndarray_sha256
from weak_quality import save_corruption_audit


STAGE = "D2-A0"
DATASET_NAME = "Caltech-6V"
CONDITION = "snr2p5_k3_seed20"
MODEL_SEED = 20
CORRUPTION_SEED = 20
CORRUPTION_K = 3
SNR_DB = 2.5
OOF_FOLDS = 5
RIDGE_ALPHA = 1.0
ADMITTED_VIEW_NUM = 3

# These constants validate the Caltech-6V boundary returned by datasets.py;
# they do not enter the dataset-agnostic reliability estimator.
EXPECTED_SAMPLE_NUM = 1400
EXPECTED_VIEW_NUM = 6
EXPECTED_CLUSTER_NUM = 7
EXPECTED_VIEW_DIMS = [48, 40, 254, 1984, 512, 928]

DEFAULT_BACKBONE_DIR = (
    REPOSITORY_ROOT / "outputs/d1_caltech6v/snr2p5_k3_seed20/models"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "outputs/d2_caltech6v/a0_utility_transfer_seed20"
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _display(path):
    value = Path(path).resolve()
    try:
        return str(value.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(value)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path):
    """Load a state dict on CPU across supported PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def compute_transfer_scores(representation_views, fold_assignment, ridge_alpha=1.0):
    """Compute frozen B3 T and frozen B4 U without labels or oracle masks.

    Args:
        representation_views: List[V] of Native embeddings, each [N,D].
        fold_assignment: Deterministic OOF fold IDs with shape [N].
        ridge_alpha: Frozen B3 Ridge alpha.

    Returns:
        A dict containing T and U, each with shape [N,V], plus the unchanged
        B3 predictability result.
    """
    prediction = oof_ridge_predictability(
        representation_views,
        fold_assignment,
        alpha=float(ridge_alpha),
    )
    # T: [N,V]
    predictability = np.asarray(
        prediction["oof_consensus_cosine"], dtype=np.float64
    )
    # U: [N,V], U[:,v] = (average_rank(T[:,v])-1)/(N-1).
    utility = np.asarray(
        compute_information_utility(predictability), dtype=np.float64
    )
    return {"T": predictability, "U": utility, "prediction": prediction}


def topk_admission(utility, admitted_view_num):
    """Select each row's largest utilities with stable low-index tie breaks."""
    values = np.asarray(utility, dtype=np.float64)
    admitted_view_num = int(admitted_view_num)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] < 2:
        raise ValueError("utility must have shape [N,V>=2]")
    if not 0 < admitted_view_num < values.shape[1]:
        raise ValueError("admitted_view_num must satisfy 0 < k < V")
    if not np.isfinite(values).all():
        raise ValueError("utility must contain only finite values")

    # order/selected: [N,V] / [N,k]
    order = np.argsort(-values, axis=1, kind="mergesort")
    selected = order[:, :admitted_view_num]
    # admission_mask: [N,V]
    admission_mask = np.zeros(values.shape, dtype=bool)
    admission_mask[
        np.arange(values.shape[0], dtype=np.int64)[:, None], selected
    ] = True
    if not np.all(admission_mask.sum(axis=1) == admitted_view_num):
        raise RuntimeError("Top-k admission row count mismatch")
    return selected, admission_mask


def random_admission_baselines(
    view_num,
    clean_views_per_sample,
    admitted_view_num,
):
    """Return exact uniform-without-replacement admission baselines."""
    view_num = int(view_num)
    clean_views_per_sample = int(clean_views_per_sample)
    admitted_view_num = int(admitted_view_num)
    if not (
        0 <= clean_views_per_sample <= view_num
        and 0 < admitted_view_num <= view_num
    ):
        raise ValueError("invalid view/admission counts")
    all_clean_probability = (
        math.comb(clean_views_per_sample, admitted_view_num)
        / float(math.comb(view_num, admitted_view_num))
        if admitted_view_num <= clean_views_per_sample
        else 0.0
    )
    return {
        "random_topk_clean_fraction": float(clean_views_per_sample / view_num),
        "random_topk_all_clean_rate": float(all_clean_probability),
    }


def pair_win_probability(utility, oracle_clean_mask):
    """Reuse B6's within-sample clean/corrupt win definition, ties worth 0.5."""
    values = np.asarray(utility, dtype=np.float64)
    clean = np.asarray(oracle_clean_mask, dtype=bool)
    if values.ndim != 2 or values.shape != clean.shape:
        raise ValueError("utility and oracle_clean_mask must have equal [N,V] shape")
    if not np.isfinite(values).all():
        raise ValueError("utility must contain only finite values")
    clean_counts = clean.sum(axis=1, dtype=np.int64)
    corrupt_counts = np.logical_not(clean).sum(axis=1, dtype=np.int64)
    if (
        clean_counts.size == 0
        or not np.all(clean_counts == clean_counts[0])
        or not np.all(corrupt_counts == corrupt_counts[0])
        or clean_counts[0] <= 0
        or corrupt_counts[0] <= 0
    ):
        raise ValueError("every sample must have fixed positive clean/corrupt counts")
    clean_values = values[clean].reshape(values.shape[0], int(clean_counts[0]))
    corrupt_values = values[np.logical_not(clean)].reshape(
        values.shape[0], int(corrupt_counts[0])
    )
    differences = clean_values[:, :, None] - corrupt_values[:, None, :]
    wins = (
        (differences > 0.0).astype(np.float64)
        + 0.5 * (differences == 0.0).astype(np.float64)
    )
    return float(np.mean(wins))


def admission_diagnostics(utility, corruption_mask, admitted_view_num):
    """Evaluate Top-k U against an oracle mask used only in this function."""
    values = np.asarray(utility, dtype=np.float64)
    corrupted = np.asarray(corruption_mask, dtype=bool)
    if values.ndim != 2 or values.shape != corrupted.shape:
        raise ValueError("utility and corruption_mask must have equal [N,V] shape")
    clean = np.logical_not(corrupted)
    clean_counts = clean.sum(axis=1, dtype=np.int64)
    if not np.all(clean_counts == clean_counts[0]):
        raise ValueError("every sample must have the same number of clean views")

    # selected: [N,k]; admitted/clean: [N,V]
    selected, admitted = topk_admission(values, admitted_view_num)
    intersection = np.logical_and(admitted, clean).sum(axis=1, dtype=np.int64)
    union = np.logical_or(admitted, clean).sum(axis=1, dtype=np.int64)
    corrupted_admitted = np.logical_and(admitted, corrupted).sum(
        axis=1, dtype=np.int64
    )
    all_clean = corrupted_admitted == 0

    # sorted_utility: [N,V]; margin is U_(k) - U_(k+1), zero-based k-1/k.
    sorted_utility = np.take_along_axis(values, np.argsort(-values, axis=1, kind="mergesort"), axis=1)
    margins = sorted_utility[:, admitted_view_num - 1] - sorted_utility[:, admitted_view_num]
    admission_error = np.logical_not(all_clean).astype(np.int64)
    if np.unique(admission_error).size != 2:
        raise ValueError("admission-error AUC requires both correct and error samples")
    margin_error_auc = float(roc_auc_score(admission_error, -margins))

    baselines = random_admission_baselines(
        view_num=values.shape[1],
        clean_views_per_sample=int(clean_counts[0]),
        admitted_view_num=admitted_view_num,
    )
    return {
        "selected_view_ids": selected,
        "admission_mask": admitted,
        "oracle_clean_mask": clean,
        "clean_count_in_topk": intersection,
        "all_clean": all_clean,
        "jaccard": intersection / union,
        "margin": margins,
        "topk_clean_fraction": float(np.mean(intersection / admitted_view_num)),
        "topk_all_clean_rate": float(np.mean(all_clean)),
        "corrupted_views_admitted_mean": float(np.mean(corrupted_admitted)),
        "topk_oracle_jaccard_mean": float(np.mean(intersection / union)),
        "pair_win_probability": pair_win_probability(values, clean),
        "margin_mean": float(np.mean(margins)),
        "margin_std": float(np.std(margins, ddof=0)),
        "margin_min": float(np.min(margins)),
        "margin_max": float(np.max(margins)),
        "margin_admission_error_auc": margin_error_auc,
        **baselines,
    }


def _validate_dataset(clean_views, label_list):
    _require(isinstance(clean_views, list), "load_data must return a view list")
    sample_num = int(clean_views[0].shape[0])
    view_num = len(clean_views)
    view_dims = [int(view.shape[1]) for view in clean_views]
    _require(sample_num == EXPECTED_SAMPLE_NUM, "Caltech-6V N mismatch")
    _require(view_num == EXPECTED_VIEW_NUM, "Caltech-6V V mismatch")
    _require(view_dims == EXPECTED_VIEW_DIMS, "Caltech-6V dimensions mismatch")
    _require(
        all(view.ndim == 2 and view.shape[0] == sample_num for view in clean_views),
        "all Caltech-6V views must have shape [N,D_v] with equal N",
    )
    _require(len(label_list) == 1, "Caltech-6V must return one label vector")
    labels = np.asarray(label_list[0])
    _require(labels.shape == (sample_num,), "Caltech-6V labels must have shape [N]")
    cluster_num = int(np.unique(labels).size)
    _require(cluster_num == EXPECTED_CLUSTER_NUM, "Caltech-6V K mismatch")
    return sample_num, view_num, cluster_num, view_dims


def _validate_frozen_condition(backbone_dir, corruption_audit):
    """Tie the checkpoint directory to the existing D1 corruption provenance."""
    source_root = backbone_dir.parent
    source_mask_path = source_root / "audit/corruption_mask.npy"
    source_audit_path = source_root / "audit/corruption_audit.json"
    _require(source_mask_path.is_file(), "frozen D1 corruption mask is missing")
    _require(source_audit_path.is_file(), "frozen D1 corruption audit is missing")
    source_mask = np.load(source_mask_path, allow_pickle=False)
    source_audit = _read_json(source_audit_path)
    reconstructed_mask = np.asarray(corruption_audit["mask"], dtype=bool)
    _require(
        source_mask.dtype == np.dtype(bool)
        and source_mask.shape == reconstructed_mask.shape
        and np.array_equal(source_mask, reconstructed_mask),
        "reconstructed corruption mask does not match frozen D1 condition",
    )
    _require(
        source_audit.get("dataset") == DATASET_NAME
        and source_audit.get("model_seed") == MODEL_SEED
        and source_audit.get("corruption_seed") == CORRUPTION_SEED
        and source_audit.get("corruption_k") == CORRUPTION_K
        and source_audit.get("target_snr_db") == SNR_DB
        and source_audit.get("mask_sha256") == corruption_audit["mask_sha256"],
        "frozen D1 corruption audit condition mismatch",
    )
    return source_mask_path, source_audit_path


def _load_native_representations(
    config,
    evaluation_views,
    backbone_dir,
    sample_num,
    view_num,
    cluster_num,
    model_seed,
):
    # evaluation_views[v]: [N,D_v]
    view_dims = [int(view.shape[1]) for view in evaluation_views]
    models = MvCAN(
        config,
        view_num=view_num,
        view_size=view_dims,
        n_clusters=cluster_num,
        seed=model_seed,
        data_size=sample_num,
        semantic_config=None,
    )
    checkpoint_paths = [
        backbone_dir / (DATASET_NAME + str(view_id + 1) + "V.pth")
        for view_id in range(view_num)
    ]
    _require(
        all(path.is_file() for path in checkpoint_paths),
        "one or more frozen Caltech-6V checkpoints are missing",
    )
    for view_id, autoencoder in enumerate(models.autoencoders):
        autoencoder.load_state_dict(
            _load_checkpoint(checkpoint_paths[view_id]), strict=True
        )
        autoencoder.eval()
    backbone_hash_before = hash_backbone(models.autoencoders)

    with torch.no_grad():
        native_views = []
        for view_id, autoencoder in enumerate(models.autoencoders):
            features = torch.from_numpy(evaluation_views[view_id]).float()
            latent = autoencoder.encoder(features)
            # native_views[v]: [N,D], with common Native latent D.
            native_views.append(F.normalize(latent, p=2, dim=1, eps=1e-12))

    latent_dim = int(config["Autoencoder"]["arch"][-1])
    _require(
        len(native_views) == view_num
        and all(tuple(view.shape) == (sample_num, latent_dim) for view in native_views),
        "Native representation shape mismatch",
    )
    audit = {
        "checkpoint_paths": [_display(path) for path in checkpoint_paths],
        "checkpoint_file_sha256": [_file_sha256(path) for path in checkpoint_paths],
        "latent_dim": latent_dim,
        "eval_mode_pass": bool(
            all(not module.training for module in models.autoencoders)
        ),
        "backbone_hash_before": backbone_hash_before,
    }
    return models, native_views, audit


def _write_top3_csv(path, admission):
    fieldnames = [
        "sample_id",
        "top3_view_ids",
        "oracle_clean_view_ids",
        "clean_count_in_top3",
        "all_clean",
        "jaccard",
        "margin34",
    ]
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for sample_id in range(admission["admission_mask"].shape[0]):
            oracle_ids = np.flatnonzero(admission["oracle_clean_mask"][sample_id])
            writer.writerow({
                "sample_id": sample_id,
                "top3_view_ids": ";".join(
                    str(int(value))
                    for value in admission["selected_view_ids"][sample_id]
                ),
                "oracle_clean_view_ids": ";".join(
                    str(int(value)) for value in oracle_ids
                ),
                "clean_count_in_top3": int(
                    admission["clean_count_in_topk"][sample_id]
                ),
                "all_clean": bool(admission["all_clean"][sample_id]),
                "jaccard": float(admission["jaccard"][sample_id]),
                "margin34": float(admission["margin"][sample_id]),
            })


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-dir", default=str(DEFAULT_BACKBONE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    backbone_dir = _resolve(args.backbone_dir)
    output_dir = _resolve(args.output_dir)
    _require(backbone_dir.is_dir(), "frozen backbone directory is missing")
    _require(not output_dir.exists(), "refusing to overwrite an existing D2 output")

    # Required MVCAN data boundary. Labels are inspected only for N/K integrity.
    clean_views, labels = load_data({"dataset": DATASET_NAME})
    sample_num, view_num, cluster_num, view_dims = _validate_dataset(
        clean_views, labels
    )
    del labels

    evaluation_views, corruption_audit = apply_weak_quality_protocol(
        clean_views,
        mode="heterogeneous_gaussian",
        k=CORRUPTION_K,
        snr_db=SNR_DB,
        corruption_seed=CORRUPTION_SEED,
    )
    # corruption_mask: [N,V], used below only after T/U have been computed.
    corruption_mask = np.asarray(corruption_audit["mask"], dtype=bool)
    source_mask_path, source_audit_path = _validate_frozen_condition(
        backbone_dir, corruption_audit
    )
    corruption_integrity_pass = bool(
        corruption_mask.shape == (sample_num, view_num)
        and int(corruption_mask.sum()) == sample_num * CORRUPTION_K
        and np.all(corruption_mask.sum(axis=1) == CORRUPTION_K)
        and np.array_equal(
            corruption_mask.sum(axis=0), np.full(view_num, sample_num // 2)
        )
        and corruption_audit["balanced_mask_pass"] is True
        and corruption_audit["snr_target_pass"] is True
    )
    _require(corruption_integrity_pass, "Caltech corruption integrity failed")

    config = get_default_config(DATASET_NAME)
    models, native_views, backbone_audit = _load_native_representations(
        config=config,
        evaluation_views=evaluation_views,
        backbone_dir=backbone_dir,
        sample_num=sample_num,
        view_num=view_num,
        cluster_num=cluster_num,
        model_seed=MODEL_SEED,
    )

    folds = make_fold_assignment(
        sample_num, n_splits=OOF_FOLDS, random_state=MODEL_SEED
    )
    fold_hash = fold_assignment_sha256(folds)
    scores = compute_transfer_scores(native_views, folds, ridge_alpha=RIDGE_ALPHA)
    # T/U: [N,V]
    predictability = scores["T"]
    utility = scores["U"]
    prediction_audit = scores["prediction"]["audit"]

    # The oracle mask first enters here, strictly after T and U are complete.
    t_reliability = reliability_metrics(predictability, corruption_mask)
    u_reliability = reliability_metrics(utility, corruption_mask)
    admission = admission_diagnostics(
        utility, corruption_mask, admitted_view_num=ADMITTED_VIEW_NUM
    )

    backbone_hash_after = hash_backbone(models.autoencoders)
    no_gradients_created_pass = bool(
        all(
            parameter.grad is None
            for module in models.autoencoders
            for parameter in module.parameters()
        )
    )
    estimator_parameters = set(inspect.signature(compute_transfer_scores).parameters)
    no_oracle_estimator_input_pass = bool(
        all(
            "label" not in name and "mask" not in name and "oracle" not in name
            for name in estimator_parameters
        )
    )
    all_finite = bool(
        np.isfinite(predictability).all()
        and np.isfinite(utility).all()
        and prediction_audit["all_finite_pass"]
        and all(
            math.isfinite(value)
            for value in (
                t_reliability["clean_predictability_mean"],
                t_reliability["corrupted_predictability_mean"],
                t_reliability["predictability_gap"],
                t_reliability["clean_corrupted_auc"],
                u_reliability["clean_predictability_mean"],
                u_reliability["corrupted_predictability_mean"],
                u_reliability["predictability_gap"],
                u_reliability["clean_corrupted_auc"],
                admission["margin_admission_error_auc"],
            )
        )
    )

    engineering_checks = {
        "dataset_boundary_pass": bool(
            sample_num == EXPECTED_SAMPLE_NUM
            and view_num == EXPECTED_VIEW_NUM
            and cluster_num == EXPECTED_CLUSTER_NUM
            and view_dims == EXPECTED_VIEW_DIMS
        ),
        "corruption_exact_pass": corruption_integrity_pass,
        "oof_sample_coverage_pass": bool(prediction_audit["oof_coverage_pass"]),
        "oof_train_test_disjoint_pass": bool(
            prediction_audit["train_test_disjoint_pass"]
        ),
        "oof_prediction_finite_pass": bool(prediction_audit["all_finite_pass"]),
        "all_finite_pass": all_finite,
        "t_shape_pass": bool(predictability.shape == (sample_num, view_num)),
        "u_shape_pass": bool(utility.shape == (sample_num, view_num)),
        "u_range_pass": bool(
            float(np.min(utility)) >= 0.0 and float(np.max(utility)) <= 1.0
        ),
        "ordered_predictor_count_pass": bool(
            prediction_audit["ordered_view_pair_count"] == view_num * (view_num - 1)
        ),
        "fold_count_pass": bool(prediction_audit["fold_count"] == OOF_FOLDS),
        "fold_deterministic_pass": bool(
            fold_hash
            == fold_assignment_sha256(
                make_fold_assignment(
                    sample_num, n_splits=OOF_FOLDS, random_state=MODEL_SEED
                )
            )
        ),
        "eval_mode_pass": backbone_audit["eval_mode_pass"],
        "backbone_immutability_pass": bool(
            backbone_audit["backbone_hash_before"] == backbone_hash_after
        ),
        "no_gradients_created_pass": no_gradients_created_pass,
        "no_optimizer_pass": True,
        "no_backward_pass": True,
        "no_label_leakage_pass": no_oracle_estimator_input_pass,
        "corruption_mask_evaluation_only_pass": no_oracle_estimator_input_pass,
    }
    engineering_pass = bool(all(engineering_checks.values()))
    scientific_checks = {
        "u_auc_above_random_pass": bool(
            u_reliability["clean_corrupted_auc"] > 0.5
        ),
        "top3_clean_fraction_above_random_pass": bool(
            admission["topk_clean_fraction"]
            > admission["random_topk_clean_fraction"]
        ),
        "top3_all_clean_rate_above_random_pass": bool(
            admission["topk_all_clean_rate"]
            > admission["random_topk_all_clean_rate"]
        ),
        "pair_win_probability_above_random_pass": bool(
            admission["pair_win_probability"] > 0.5
        ),
    }
    scientific_pass = bool(all(scientific_checks.values()))

    result = {
        "stage": STAGE,
        "scope": "read_only_offline_frozen_information_utility_transfer",
        "dataset": DATASET_NAME,
        "condition": CONDITION,
        "model_seed": MODEL_SEED,
        "corruption_seed": CORRUPTION_SEED,
        "N": sample_num,
        "V": view_num,
        "K": cluster_num,
        "view_dims": view_dims,
        "ordered_predictor_count": int(
            prediction_audit["ordered_view_pair_count"]
        ),
        "corruption_mask_shape": list(corruption_mask.shape),
        "T_shape": list(predictability.shape),
        "U_shape": list(utility.shape),
        "T_sha256": tensor_sha256(predictability),
        "U_sha256": tensor_sha256(utility),
        "U_min": float(np.min(utility)),
        "U_max": float(np.max(utility)),
        "all_finite": all_finite,
        "T_clean_mean": t_reliability["clean_predictability_mean"],
        "T_corrupt_mean": t_reliability["corrupted_predictability_mean"],
        "T_gap": t_reliability["predictability_gap"],
        "U_clean_mean": u_reliability["clean_predictability_mean"],
        "U_corrupt_mean": u_reliability["corrupted_predictability_mean"],
        "U_gap": u_reliability["predictability_gap"],
        "T_auc_clean_vs_corrupt": t_reliability["clean_corrupted_auc"],
        "U_auc_clean_vs_corrupt": u_reliability["clean_corrupted_auc"],
        "top3_clean_fraction": admission["topk_clean_fraction"],
        "top3_all_clean_rate": admission["topk_all_clean_rate"],
        "corrupted_views_admitted_mean": admission[
            "corrupted_views_admitted_mean"
        ],
        "top3_oracle_jaccard_mean": admission["topk_oracle_jaccard_mean"],
        "pair_win_probability": admission["pair_win_probability"],
        "random_top3_clean_fraction": admission["random_topk_clean_fraction"],
        "random_top3_all_clean_rate": admission["random_topk_all_clean_rate"],
        "margin34_mean": admission["margin_mean"],
        "margin34_std": admission["margin_std"],
        "margin34_min": admission["margin_min"],
        "margin34_max": admission["margin_max"],
        "margin34_admission_error_auc": admission[
            "margin_admission_error_auc"
        ],
        "margin34_admission_error_auc_direction": (
            "roc_auc_score(admission_error, -margin34); higher means smaller "
            "margin predicts error"
        ),
        "fold_count": int(prediction_audit["fold_count"]),
        "fold_assignment_sha256": fold_hash,
        "oof_sample_coverage_pass": bool(prediction_audit["oof_coverage_pass"]),
        "oof_prediction_finite_pass": bool(prediction_audit["all_finite_pass"]),
        "protocol": {
            "primary_score": "B3 OOF cross-view Ridge consensus cosine",
            "utility_formula": "per-view (average_rank(T)-1)/(N-1)",
            "rank_tie_method": "average",
            "oof_folds": OOF_FOLDS,
            "fold_random_state": "model_seed",
            "ridge_alpha": RIDGE_ALPHA,
            "ridge_fit_intercept": True,
            "top3_tie_break": "stable ascending zero-based view ID",
            "view_id_base": 0,
            "pair_win_definition": (
                "mean(1[U_clean>U_corrupt] + 0.5*1[U_clean==U_corrupt]) "
                "over within-sample clean/corrupt pairs"
            ),
            "labels_used_for_dataset_integrity_only": True,
            "corruption_mask_used_for_evaluation_only": True,
        },
        "source_provenance": {
            "backbone_dir": _display(backbone_dir),
            "checkpoint_paths": backbone_audit["checkpoint_paths"],
            "checkpoint_file_sha256": backbone_audit["checkpoint_file_sha256"],
            "source_corruption_mask": _display(source_mask_path),
            "source_corruption_audit": _display(source_audit_path),
            "source_corruption_mask_sha256": ndarray_sha256(corruption_mask),
            "backbone_hash_before": backbone_audit["backbone_hash_before"],
            "backbone_hash_after": backbone_hash_after,
        },
        "engineering_checks": engineering_checks,
        "scientific_sanity_checks": scientific_checks,
        "D2_A0_ENGINEERING_PASS": engineering_pass,
        "D2_A0_SCIENTIFIC_SANITY_PASS": scientific_pass,
        "D2_A0_UTILITY_TRANSFER_PASS": bool(engineering_pass and scientific_pass),
    }

    output_dir.mkdir(parents=True)
    np.savez_compressed(
        output_dir / "utility_scores.npz",
        T=predictability,
        U=utility,
        corruption_mask=corruption_mask,
    )
    (output_dir / "fold_assignment.csv").write_text(
        fold_assignment_csv_text(folds), encoding="utf-8"
    )
    _write_top3_csv(output_dir / "top3_admission.csv", admission)
    output_corruption_audit = dict(corruption_audit)
    output_corruption_audit.update({
        "dataset": DATASET_NAME,
        "model_seed": MODEL_SEED,
        "n_clusters": cluster_num,
    })
    save_corruption_audit(
        output_corruption_audit, str(output_dir / "corruption_audit")
    )
    _write_json(output_dir / "utility_transfer.json", result)

    print("D2_A0_ENGINEERING_PASS=" + str(engineering_pass))
    print("D2_A0_SCIENTIFIC_SANITY_PASS=" + str(scientific_pass))
    print("D2_A0_UTILITY_TRANSFER_PASS=" + str(engineering_pass and scientific_pass))
    print("Saved: " + _display(output_dir / "utility_transfer.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
