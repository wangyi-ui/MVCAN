"""B6-WQ0 frozen Utility-guided MVCAN view-admission evaluation."""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ClusteringTest import acc as mvcan_acc
from ClusteringTest import ari as mvcan_ari
from ClusteringTest import nmi as mvcan_nmi
from configure import get_default_config
from datasets import load_data
from irv.b3_audit import hash_backbone
from irv.b4_information_utility import tensor_sha256
from irv.b4_information_utility import tensor_view_list_sha256
from irv.b5_shared_semantic_rate import normalize_native_z
from model import MvCAN
from model import min_max_scaler as original_min_max_scaler
from weak_quality import apply_weak_quality_protocol

STAGE = "B6-WQ0"
DATASET_NAME = "MSRC-v1"
CONDITION = "snr2p5_k2"
SUPPORTED_SEEDS = (20, 30, 50)
SAMPLE_NUM = 210
VIEW_NUM = 5
CLUSTER_NUM = 7
LATENT_DIM = 10
FUSION_DIM = 50
EPSILON = 1e-6
SHUFFLE_SEED = 20260816
NULL_REPEATS = 200
BASE_METRIC_TOLERANCE = 1e-12
DEFAULT_OUTPUT_DIR = "outputs/b6_weak_quality/wq0_utility_semantic_admission"
CANONICAL_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "experiments/b6_weak_quality/b6_wq0_frozen_baseline_manifest.json"
)
HISTORICAL_REFERENCE_KIND = "B2-training-loop-transient"

SEED_PROVENANCE = {
    20: {
        "backbone_dir": "outputs/b2_weak_quality/snr2p5_k2_seed20/models",
        "backbone_hash": "f4f026f47938c619bfdfb868cfcdb5eac354652a51d71199edbc997c0a5de0ab",
        "z_hash": "6dd6f0d4c9fea4a5b44d61a0984daa7137541dcb0762489d668967fb610bc894",
        "utility_dir": "outputs/b5_semantic_rate/a04_relation_utility_alignment_step100_seed20",
        "utility_hash": "0c4183cf58d464269d194e2980345f157e6dc7e81a57522ebeb43a998813d25d",
        "b2_log": "logs/b2_snr2p5_k2_seed20.log",
        "base_metrics": {"acc": 0.6904761904761905, "nmi": 0.6468195301776315, "ari": 0.5374688673991558},
    },
    30: {
        "backbone_dir": "outputs/b2_weak_quality/snr2p5_k2_seed30/models",
        "backbone_hash": "d553d1cd5627c5aa2c09de4914000a37efd1dca839c72712d5ec55db2e32c6e8",
        "z_hash": "31bbf7426b0982ef301da938669a1ac482284fdf98e830663ca87b1ceaebbf27",
        "utility_dir": "outputs/b5_semantic_rate/a05_utility_alignment_step100_seed30",
        "utility_hash": "ecc1af32cf18b39dfbc9dc70d1ccb151adf1e6f0b49c4da8d068e38a88cd8cc1",
        "b2_log": "logs/b2_snr2p5_k2_seed30.log",
        "base_metrics": {"acc": 0.7476190476190476, "nmi": 0.6455551255982171, "ari": 0.558920656674637},
    },
    50: {
        "backbone_dir": "outputs/b2_weak_quality/snr2p5_k2_seed50/models",
        "backbone_hash": "a1c6560ab11844350ed6c22849f3a97010ed5be60ada7526529f1be0fc3a12d9",
        "z_hash": "da83e291d70a787f9da4b237fbe6beb555eac8955e6fca2d867dbd41eee1b606",
        "utility_dir": "outputs/b5_semantic_rate/a05_utility_alignment_step100_seed50",
        "utility_hash": "8fce1e5cec1a981c825962c147beaf46b2713a6164b5dda1520e5e4361af18c3",
        "b2_log": "logs/b2_snr2p5_k2_seed50.log",
        "base_metrics": {"acc": 0.8238095238095238, "nmi": 0.7117129578755498, "ari": 0.656789623317153},
    },
}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def architecture_audit():
    return {
        "latent_generation": {
            "function": "Autoencoder.encoder",
            "model_location": "model.py:108",
            "z_views": "List[V=5], each [N=210, latent_dim=10]",
        },
        "sample_specific_native_attention_exists": False,
        "native_view_fusion": {
            "exists": True,
            "weight_variable": "www",
            "weight_shape": [VIEW_NUM],
            "sample_specific": False,
            "weight_update": "exp(round(NMI(fused_assignment, view_assignment), 5))",
            "adapter_attention_shape": [SAMPLE_NUM, VIEW_NUM],
            "adapter_operation": "normalize native scales and broadcast over samples",
            "model_location": "model.py:519-549",
        },
        "clustering_readout": {
            "construction": "hstack(minmax(z_v) * native_view_scale_v)",
            "shape": [SAMPLE_NUM, FUSION_DIM],
            "model_location": "model.py:547-564",
            "evaluator": "KMeans then ClusteringTest ACC/NMI/ARI",
            "kmeans_branch": "KMeans for N=210; MiniBatchKMeans only when N>10000",
            "kmeans_parameters": "n_clusters=7, n_init=100, random_state=model_seed",
        },
        "frozen_weight_substitution_feasible": True,
        "base_geometry_note": "scale normalization changes BASE only by one positive global scalar",
    }


def checkpoint_paths(backbone_dir):
    root = _resolve(backbone_dir)
    return [root / (DATASET_NAME + str(v + 1) + "V.pth") for v in range(VIEW_NUM)]


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_frozen_backbone(seed, evaluation_views):
    seed = int(seed)
    _require(seed in SUPPORTED_SEEDS, "unsupported seed")
    provenance = SEED_PROVENANCE[seed]
    paths = checkpoint_paths(provenance["backbone_dir"])
    _require(len(paths) == VIEW_NUM and all(p.is_file() for p in paths), "missing B2 checkpoints")
    config = get_default_config(DATASET_NAME)
    config["dataset"] = DATASET_NAME
    models = MvCAN(
        config,
        view_num=VIEW_NUM,
        view_size=[int(view.shape[1]) for view in evaluation_views],
        n_clusters=CLUSTER_NUM,
        seed=seed,
        data_size=SAMPLE_NUM,
        semantic_config=None,
    )
    for view_id, autoencoder in enumerate(models.autoencoders):
        autoencoder.load_state_dict(_load_checkpoint(paths[view_id]), strict=True)
        autoencoder.eval()
        autoencoder.requires_grad_(False)
    backbone = hash_backbone(models.autoencoders)
    _require(backbone["aggregate"] == provenance["backbone_hash"], "backbone hash mismatch")
    return models, backbone, paths


def reconstruct_noisy_views(seed):
    config = get_default_config(DATASET_NAME)
    config["dataset"] = DATASET_NAME
    clean_views, label_list = load_data(config)
    labels = np.asarray(label_list[0], dtype=np.int64)
    evaluation_views, runtime = apply_weak_quality_protocol(
        clean_views,
        mode="heterogeneous_gaussian",
        k=2,
        snr_db=2.5,
        corruption_seed=int(seed),
    )
    stored_path = _resolve(
        "outputs/b2_weak_quality/snr2p5_k2_seed" + str(seed) + "/audit/corruption_audit.json"
    )
    stored = _read_json(stored_path)
    _require(
        runtime["mask_sha256"] == stored["mask_sha256"]
        and runtime["corrupted_view_sha256"] == stored["corrupted_view_sha256"],
        "B2 condition mismatch",
    )
    runtime.pop("mask", None)
    runtime.pop("corruption_pairs", None)
    _require(labels.shape == (SAMPLE_NUM,), "label shape mismatch")
    return evaluation_views, labels, runtime, stored_path


def _kmeans(seed, sample_num=SAMPLE_NUM):
    """Mirror model.py's exact KMeans/MiniBatchKMeans branch."""
    if int(sample_num) > 10000:
        return MiniBatchKMeans(
            n_clusters=CLUSTER_NUM,
            n_init=100,
            batch_size=10000,
            random_state=int(seed),
        )
    return KMeans(
        n_clusters=CLUSTER_NUM,
        n_init=100,
        random_state=int(seed),
    )


def extract_fusion_inputs(models, evaluation_views, seed):
    raw_z_views = []
    normalized_z_views = []
    scaled_z_views = []
    checkpoint_native_assignments = []
    original_style_assignments = []
    per_view_audit = []
    with torch.no_grad():
        for view_id, autoencoder in enumerate(models.autoencoders):
            features = torch.from_numpy(evaluation_views[view_id]).float()
            raw_z = autoencoder.encoder(features).detach()
            raw_z_numpy = raw_z.cpu().numpy()
            normalized_z = normalize_native_z(raw_z).detach()
            checkpoint_assignment = (
                autoencoder.clustering(raw_z).detach().cpu().numpy().argmax(1)
            )

            # model.py uses one module-global scaler but calls fit_transform for
            # each view. A fresh scaler should be equivalent; record both paths.
            original_style_scaled = original_min_max_scaler.fit_transform(raw_z_numpy)
            current_adapter_scaled = MinMaxScaler().fit_transform(raw_z_numpy)
            scaled_max_abs_error = float(
                np.max(np.abs(original_style_scaled - current_adapter_scaled))
            )

            # Reproduce the pre-fusion center initialization without mutating
            # the frozen checkpoint. q.argmax is checked on a temporary clone.
            per_view_kmeans = _kmeans(seed, sample_num=raw_z_numpy.shape[0])
            initialized_assignment = per_view_kmeans.fit_predict(raw_z_numpy)
            temporary_autoencoder = copy.deepcopy(autoencoder)
            temporary_autoencoder._cluster_layer.data = torch.tensor(
                per_view_kmeans.cluster_centers_
            ).to(temporary_autoencoder._cluster_layer.device)
            initialized_q_assignment = (
                temporary_autoencoder.clustering(raw_z)
                .detach()
                .cpu()
                .numpy()
                .argmax(1)
            )
            initialized_q_matches_kmeans = bool(
                np.array_equal(initialized_q_assignment, initialized_assignment)
            )
            _require(
                initialized_q_matches_kmeans,
                "temporary initialized q assignment differs from KMeans",
            )

            raw_z_views.append(raw_z)
            normalized_z_views.append(normalized_z)
            scaled_z_views.append(original_style_scaled)
            checkpoint_native_assignments.append(checkpoint_assignment)
            original_style_assignments.append(initialized_assignment)
            per_view_audit.append({
                "view_id": int(view_id),
                "raw_z_sha256": tensor_sha256(raw_z),
                "normalized_z_sha256": tensor_sha256(normalized_z),
                "original_style_scaled_z_sha256": tensor_sha256(original_style_scaled),
                "current_adapter_scaled_z_sha256": tensor_sha256(current_adapter_scaled),
                "scaled_z_max_abs_error": scaled_max_abs_error,
                "scaled_z_exact_match": bool(
                    np.array_equal(original_style_scaled, current_adapter_scaled)
                ),
                "checkpoint_native_assignment_sha256": tensor_sha256(
                    checkpoint_assignment
                ),
                "kmeans_initialized_assignment_sha256": tensor_sha256(
                    initialized_assignment
                ),
                "kmeans_initialized_q_assignment_sha256": tensor_sha256(
                    initialized_q_assignment
                ),
                "initialized_q_matches_kmeans": initialized_q_matches_kmeans,
                "assignment_exact_match": bool(
                    np.array_equal(checkpoint_assignment, initialized_assignment)
                ),
                "assignment_disagreement_fraction": float(
                    np.mean(checkpoint_assignment != initialized_assignment)
                ),
            })
    _require(
        len(raw_z_views) == VIEW_NUM
        and all(tuple(z.shape) == (SAMPLE_NUM, LATENT_DIM) for z in raw_z_views),
        "Native-z shape mismatch",
    )
    _require(all(np.isfinite(z).all() for z in scaled_z_views), "scaled z non-finite")
    prefusion_assignment_match = bool(
        all(row["assignment_exact_match"] for row in per_view_audit)
    )
    return {
        "raw_z_views": raw_z_views,
        "normalized_z_views": normalized_z_views,
        "scaled_z_views": scaled_z_views,
        "checkpoint_native_view_assignments": checkpoint_native_assignments,
        "original_style_view_assignments": original_style_assignments,
        # The fusion replay must consume original MVCAN's initialized y_view.
        "view_assignments": original_style_assignments,
        "per_view_audit": per_view_audit,
        "B6_WQ0_PREFUSION_ASSIGNMENT_MATCH": prefusion_assignment_match,
    }


def original_mvcan_fusion_adapter(
    scaled_z_views,
    view_assignments,
    seed,
    fusion_updates,
):
    """Expose the Native global-scale hstack readout for config T_1 updates."""
    _require(len(scaled_z_views) == VIEW_NUM, "scaled view count mismatch")
    _require(len(view_assignments) == VIEW_NUM, "assignment count mismatch")
    _require(int(fusion_updates) > 0, "fusion update count must be positive")
    fusion_scales = np.ones(VIEW_NUM, dtype=np.float64)
    estimator = _kmeans(seed, sample_num=SAMPLE_NUM)
    update_audit = []
    for update_id in range(int(fusion_updates)):
        scales_used = fusion_scales.copy()
        fused = np.hstack([
            scaled_z_views[v] * scales_used[v] for v in range(VIEW_NUM)
        ])
        predictions = estimator.fit_predict(fused)
        fusion_scales = np.asarray([
            np.exp(np.round(mvcan_nmi(predictions, assignment), 5))
            for assignment in view_assignments
        ], dtype=np.float64)
        update_audit.append({
            "update_id": int(update_id),
            "www_before": scales_used.tolist(),
            "fused_representation_sha256": tensor_sha256(fused),
            "fused_cluster_assignment_sha256": tensor_sha256(predictions),
            "www_after": fusion_scales.tolist(),
        })
    attention_row = scales_used / np.sum(scales_used)
    attention = np.repeat(attention_row[None, :], SAMPLE_NUM, axis=0)
    _require(fused.shape == (SAMPLE_NUM, FUSION_DIM), "fusion shape mismatch")
    return {
        "fused_representation": fused,
        "predictions": predictions,
        "native_fusion_scales": scales_used,
        "next_native_fusion_scales": fusion_scales,
        "fusion_update_audit": update_audit,
        "final_www_used_for_historical_fusion": scales_used.tolist(),
        "historical_style_final_fused_sha256": tensor_sha256(fused),
        "historical_style_prediction_sha256": tensor_sha256(predictions),
        "original_attention": attention,
    }


def validate_attention(weights):
    values = np.asarray(weights, dtype=np.float64)
    _require(values.shape == (SAMPLE_NUM, VIEW_NUM), "weight shape mismatch")
    _require(np.isfinite(values).all(), "weights non-finite")
    _require(float(values.min()) >= 0.0, "weights negative")
    _require(
        np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-12),
        "weight row sum mismatch",
    )
    return True


def admission_weights(original_attention, source_utility, epsilon=EPSILON):
    """Apply only normalized original attention times frozen source-U."""
    attention = np.asarray(original_attention, dtype=np.float64)
    utility = np.asarray(source_utility, dtype=np.float64)
    _require(epsilon == EPSILON, "epsilon must be exactly 1e-6")
    _require(attention.shape == utility.shape == (SAMPLE_NUM, VIEW_NUM), "admission shape mismatch")
    numerator = attention * (utility + epsilon)
    result = numerator / np.sum(numerator, axis=1, keepdims=True)
    validate_attention(result)
    return result


def fused_representation_from_admission(scaled_z_views, weights):
    validate_attention(weights)
    fused = np.hstack([
        scaled_z_views[v] * weights[:, v:v + 1] for v in range(VIEW_NUM)
    ])
    _require(fused.shape == (SAMPLE_NUM, FUSION_DIM), "fusion shape mismatch")
    _require(np.isfinite(fused).all(), "fusion non-finite")
    return fused


def generate_within_view_permutations(
    sample_num=SAMPLE_NUM,
    view_num=VIEW_NUM,
    repeats=NULL_REPEATS,
    seed=SHUFFLE_SEED,
):
    rng = np.random.RandomState(int(seed))
    bank = np.empty((int(repeats), int(view_num), int(sample_num)), dtype=np.int64)
    for repeat_id in range(int(repeats)):
        for view_id in range(int(view_num)):
            bank[repeat_id, view_id] = rng.permutation(int(sample_num))
    return bank


def shuffle_utility_within_views(source_utility, view_permutations):
    utility = np.asarray(source_utility, dtype=np.float64)
    permutations = np.asarray(view_permutations, dtype=np.int64)
    _require(utility.shape == (SAMPLE_NUM, VIEW_NUM), "Utility shape mismatch")
    _require(permutations.shape == (VIEW_NUM, SAMPLE_NUM), "permutation shape mismatch")
    shuffled = np.empty_like(utility)
    for view_id in range(VIEW_NUM):
        _require(
            np.array_equal(np.sort(permutations[view_id]), np.arange(SAMPLE_NUM)),
            "invalid within-view permutation",
        )
        shuffled[:, view_id] = utility[permutations[view_id], view_id]
        _require(
            np.array_equal(np.sort(shuffled[:, view_id]), np.sort(utility[:, view_id])),
            "Utility marginal changed",
        )
    return shuffled


def evaluate_fused_representation(fused, seed, labels):
    predictions = _kmeans(seed).fit_predict(fused)
    return predictions, metrics_from_predictions(labels, predictions)


def metrics_from_predictions(labels, predictions):
    return {
        "acc": float(mvcan_acc(labels, predictions)),
        "nmi": float(mvcan_nmi(labels, predictions)),
        "ari": float(mvcan_ari(labels, predictions)),
    }


def base_reproduction_audit(actual, expected, tolerance=BASE_METRIC_TOLERANCE):
    errors = {
        metric: float(abs(actual[metric] - expected[metric]))
        for metric in ("acc", "nmi", "ari")
    }
    return {
        "expected_metrics": dict(expected),
        "absolute_errors": errors,
        "tolerance": float(tolerance),
        "B6_WQ0_BASE_REPRO_PASS": bool(all(v <= tolerance for v in errors.values())),
    }


def load_canonical_manifest(path=CANONICAL_MANIFEST_PATH):
    manifest_path = _resolve(path)
    if not manifest_path.is_file():
        return {
            "schema_version": "b6-wq0-frozen-baseline-v1",
            "canonical_entries": {},
            "registration_status": "unregistered",
        }, manifest_path
    manifest = _read_json(manifest_path)
    _require(
        isinstance(manifest.get("canonical_entries"), dict),
        "canonical manifest entries must be a mapping",
    )
    return manifest, manifest_path


def canonical_base_reproduction_audit(
    seed,
    actual_metrics,
    actual_backbone_hash,
    actual_normalized_z_hash,
    actual_fused_hash,
    actual_prediction_hash,
    manifest_path=CANONICAL_MANIFEST_PATH,
):
    """Compare only against a separately registered canonical manifest entry."""
    manifest, resolved_path = load_canonical_manifest(manifest_path)
    entry = manifest["canonical_entries"].get(str(int(seed)))
    if entry is None:
        return {
            "canonical_baseline_registered": False,
            "canonical_manifest_path": _display(resolved_path),
            "canonical_manifest_registration_status": manifest.get(
                "registration_status", "unregistered"
            ),
            "expected_canonical_entry": None,
            "checks": None,
            "B6_WQ0_CANONICAL_BASE_REPRO_PASS": False,
            "scientific_action_block_reason": (
                "canonical frozen baseline is not formally registered"
            ),
        }
    checks = {
        "backbone_hash_exact": bool(
            actual_backbone_hash == entry["expected_backbone_hash"]
        ),
        "normalized_z_hash_exact": bool(
            actual_normalized_z_hash == entry["expected_normalized_z_hash"]
        ),
        "canonical_fused_hash_exact": bool(
            actual_fused_hash == entry["expected_canonical_fused_hash"]
        ),
        "prediction_hash_exact": bool(
            actual_prediction_hash == entry["expected_prediction_hash"]
        ),
        "canonical_metrics_exact": bool(
            actual_metrics == entry["expected_canonical_metrics"]
        ),
    }
    canonical_pass = bool(all(checks.values()))
    return {
        "canonical_baseline_registered": True,
        "canonical_manifest_path": _display(resolved_path),
        "canonical_manifest_registration_status": manifest.get(
            "registration_status", "registered"
        ),
        "expected_canonical_entry": entry,
        "checks": checks,
        "B6_WQ0_CANONICAL_BASE_REPRO_PASS": canonical_pass,
        "scientific_action_block_reason": (
            None if canonical_pass else "registered canonical baseline mismatch"
        ),
    }


def scientific_action_gate(
    canonical_base_repro_pass,
    identity_admission_pass,
    backbone_frozen_pass,
):
    """Gate scientific action without consulting historical B2 diagnostics."""
    return bool(
        canonical_base_repro_pass
        and identity_admission_pass
        and backbone_frozen_pass
    )


def load_source_utility(seed):
    provenance = SEED_PROVENANCE[int(seed)]
    root = _resolve(provenance["utility_dir"])
    utility_path = root / "source_utility.npy"
    record_path = root / "b5_a04_relation_utility_alignment.json"
    _require(utility_path.is_file() and record_path.is_file(), "missing A0.5 Utility")
    utility = np.load(utility_path, allow_pickle=False)
    record = _read_json(record_path)
    utility_hash = tensor_sha256(utility)
    _require(utility.shape == (SAMPLE_NUM, VIEW_NUM), "Utility shape mismatch")
    _require(np.isfinite(utility).all(), "Utility non-finite")
    _require(float(utility.min()) >= 0.0 and float(utility.max()) <= 1.0, "Utility outside [0,1]")
    _require(
        utility_hash == provenance["utility_hash"]
        and utility_hash == record["source_utility_sha256"],
        "A0.5 Utility hash mismatch",
    )
    return utility, utility_hash, utility_path, record_path


def attention_summary(weights):
    validate_attention(weights)
    values = np.asarray(weights, dtype=np.float64)
    entropy_rows = -np.sum(
        np.where(values > 0.0, values * np.log(values), 0.0), axis=1
    )
    mean_entropy = float(np.mean(entropy_rows))
    return {
        "mean_entropy": mean_entropy,
        "effective_view_number": float(math.exp(mean_entropy)),
    }


def delta_metrics(left, right):
    return {name: float(left[name] - right[name]) for name in ("acc", "nmi", "ari")}


def summarize_null(values, correct_value):
    array = np.asarray(values, dtype=np.float64)
    _require(array.shape == (NULL_REPEATS,), "null metric shape mismatch")
    _require(np.isfinite(array).all(), "null metric non-finite")
    count_ge = int(np.sum(array >= float(correct_value)))
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "fraction_shuffled_lt_correct": float(np.mean(array < float(correct_value))),
        "count_shuffled_ge_correct": count_ge,
        "one_sided_empirical_p": float((1 + count_ge) / (NULL_REPEATS + 1)),
    }


def save_null_arrays(output_dir, shuffled_acc, shuffled_nmi, shuffled_ari):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    arrays = {
        "shuffled_acc": np.asarray(shuffled_acc, dtype=np.float64),
        "shuffled_nmi": np.asarray(shuffled_nmi, dtype=np.float64),
        "shuffled_ari": np.asarray(shuffled_ari, dtype=np.float64),
    }
    for name, values in arrays.items():
        _require(values.shape == (NULL_REPEATS,), name + " shape mismatch")
        np.save(root / (name + ".npy"), values)
    return arrays


def multiseed_gate(seed_results):
    _require(len(seed_results) == 3, "multi-seed gate requires three seeds")
    signals = [bool(row["B6_WQ0_SEED_UTILITY_ADMISSION_SIGNAL"]) for row in seed_results]
    acc_delta = np.asarray([row["delta_vs_base"]["acc"] for row in seed_results])
    nmi_delta = np.asarray([row["delta_vs_base"]["nmi"] for row in seed_results])
    ari_delta = np.asarray([row["delta_vs_base"]["ari"] for row in seed_results])
    result = {
        "seed_signal_count": int(sum(signals)),
        "mean_delta_acc_vs_base": float(np.mean(acc_delta)),
        "mean_delta_nmi_vs_base": float(np.mean(nmi_delta)),
        "mean_delta_ari_vs_base": float(np.mean(ari_delta)),
        "no_catastrophic_acc_degradation": bool(np.all(acc_delta >= -0.02)),
    }
    result["B6_WQ0_MULTISEED_ADMISSION_PASS"] = bool(
        result["seed_signal_count"] >= 2
        and result["mean_delta_acc_vs_base"] > 0.0
        and result["mean_delta_nmi_vs_base"] > 0.0
        and result["mean_delta_ari_vs_base"] > 0.0
        and result["no_catastrophic_acc_degradation"]
    )
    return result


def identity_admission_audit(base, scaled_z_views, seed, labels):
    """Verify U=1 preserves BASE up to one positive global scalar."""
    ones = np.ones((SAMPLE_NUM, VIEW_NUM), dtype=np.float64)
    identity_weights = admission_weights(base["original_attention"], ones)
    identity_fused = fused_representation_from_admission(
        scaled_z_views, identity_weights
    )
    identity_predictions, identity_metrics = evaluate_fused_representation(
        identity_fused, seed, labels
    )
    base_metrics = metrics_from_predictions(labels, base["predictions"])
    positive_global_scalar = float(np.sum(base["native_fusion_scales"]))
    base_array = np.asarray(base["fused_representation"])
    geometry_dtype = np.dtype(np.float32)
    _require(
        all(np.asarray(view).dtype == geometry_dtype for view in scaled_z_views),
        "canonical scaled-z dtype must be float32",
    )
    geometry_machine_epsilon = float(np.finfo(geometry_dtype).eps)
    geometry_scale = float(max(1.0, np.max(np.abs(base_array))))
    geometry_atol = float(16.0 * geometry_machine_epsilon * geometry_scale)
    geometry_max_abs_error = float(np.max(np.abs(
        identity_fused * positive_global_scalar - base_array
    )))
    geometry_within_tolerance = bool(
        geometry_max_abs_error <= geometry_atol
    )

    native_relative_scale = (
        base["native_fusion_scales"]
        / np.sum(base["native_fusion_scales"])
    )
    relative_scale_max_abs_error = float(np.max(np.abs(
        identity_weights
        - np.repeat(native_relative_scale[None, :], SAMPLE_NUM, axis=0)
    )))
    relative_scale_atol = float(16.0 * np.finfo(np.float64).eps)
    relative_view_scale_preserved = bool(
        relative_scale_max_abs_error <= relative_scale_atol
    )
    assignment_exact_match = bool(
        np.array_equal(identity_predictions, base["predictions"])
    )
    metrics_exact_match = bool(identity_metrics == base_metrics)
    identity_pass = bool(
        positive_global_scalar > 0.0
        and relative_view_scale_preserved
        and geometry_within_tolerance
        and assignment_exact_match
        and metrics_exact_match
    )
    return {
        "utility": "ones([210,5])",
        "positive_global_scalar": positive_global_scalar,
        "relative_view_scale_max_abs_error": relative_scale_max_abs_error,
        "relative_view_scale_atol": relative_scale_atol,
        "relative_view_scale_preserved": relative_view_scale_preserved,
        "geometry_dtype": str(geometry_dtype),
        "geometry_machine_epsilon": geometry_machine_epsilon,
        "geometry_scale": geometry_scale,
        "geometry_atol": geometry_atol,
        "geometry_max_abs_error_after_rescaling": geometry_max_abs_error,
        "geometry_within_tolerance": geometry_within_tolerance,
        "geometry_global_scalar_pass": geometry_within_tolerance,
        "identity_fused_sha256": tensor_sha256(identity_fused),
        "identity_prediction_sha256": tensor_sha256(identity_predictions),
        "assignment_exact_match": assignment_exact_match,
        "identity_metrics": identity_metrics,
        "metrics_exact_match": metrics_exact_match,
        "B6_WQ0_IDENTITY_ADMISSION_PASS": identity_pass,
    }


def prepare_seed(seed):
    seed = int(seed)
    config = get_default_config(DATASET_NAME)
    config["dataset"] = DATASET_NAME
    fusion_updates = int(config["training"]["T_1"])
    views, labels, condition_audit, condition_path = reconstruct_noisy_views(seed)
    models, backbone, paths = load_frozen_backbone(seed, views)
    fusion_inputs = extract_fusion_inputs(models, views, seed)
    z_hash = tensor_view_list_sha256(fusion_inputs["normalized_z_views"])
    _require(z_hash == SEED_PROVENANCE[seed]["z_hash"], "A0.5 z hash mismatch")
    utility, utility_hash, utility_path, utility_record = load_source_utility(seed)
    return {
        "seed": seed,
        "models": models,
        "labels": labels,
        "condition_audit": condition_audit,
        "condition_audit_path": condition_path,
        "checkpoint_paths": paths,
        "backbone_hash_before": backbone,
        "fusion_inputs": fusion_inputs,
        "fusion_updates": fusion_updates,
        "fusion_updates_source": "get_default_config(dataset)['training']['T_1']",
        "z_hash": z_hash,
        "utility": utility,
        "utility_hash": utility_hash,
        "utility_path": utility_path,
        "utility_record_path": utility_record,
    }


def run_base(prepared):
    inputs = prepared["fusion_inputs"]
    base = original_mvcan_fusion_adapter(
        inputs["scaled_z_views"],
        inputs["view_assignments"],
        prepared["seed"],
        prepared["fusion_updates"],
    )
    metrics = metrics_from_predictions(prepared["labels"], base["predictions"])
    audit = base_reproduction_audit(
        metrics, SEED_PROVENANCE[prepared["seed"]]["base_metrics"]
    )
    identity_audit = identity_admission_audit(
        base,
        inputs["scaled_z_views"],
        prepared["seed"],
        prepared["labels"],
    )
    if not inputs["B6_WQ0_PREFUSION_ASSIGNMENT_MATCH"]:
        first_divergence = (
            "checkpoint_native_assignment_vs_"
            "original_style_kmeans_initialized_assignment"
        )
    elif not all(row["scaled_z_exact_match"] for row in inputs["per_view_audit"]):
        first_divergence = "original_global_scaler_vs_fresh_per_view_scaler"
    elif not audit["B6_WQ0_BASE_REPRO_PASS"]:
        first_divergence = (
            "historical_epoch1000_in_loop_state_vs_post_update_frozen_checkpoint"
        )
    else:
        first_divergence = None
    exact_replay_feasible = bool(audit["B6_WQ0_BASE_REPRO_PASS"])
    base.update({
        "identity_admission_audit": identity_audit,
        "first_divergence_point": first_divergence,
        "B6_WQ0_PREFUSION_ASSIGNMENT_MATCH": (
            inputs["B6_WQ0_PREFUSION_ASSIGNMENT_MATCH"]
        ),
        "B6_WQ0_IDENTITY_ADMISSION_PASS": (
            identity_audit["B6_WQ0_IDENTITY_ADMISSION_PASS"]
        ),
        "B6_WQ0_FROZEN_B2_EXACT_REPLAY_FEASIBLE": exact_replay_feasible,
    })
    return base, metrics, audit


def _base_result(
    prepared,
    base,
    metrics,
    audit,
    canonical_manifest_path=CANONICAL_MANIFEST_PATH,
):
    seed = prepared["seed"]
    backbone_after = hash_backbone(prepared["models"].autoencoders)
    frozen_pass = bool(backbone_after == prepared["backbone_hash_before"])
    exact_replay_feasible = base["B6_WQ0_FROZEN_B2_EXACT_REPLAY_FEASIBLE"]
    replay_provenance = {
        "historical_metric_timing": (
            "model.py evaluates epoch=1000 fusion before the epoch=1000 "
            "optimizer updates"
        ),
        "checkpoint_timing": (
            "run.py saves autoencoder state_dict files after Models.train "
            "returns, hence after the epoch=1000 optimizer updates"
        ),
        "historical_www_state": (
            "www is initialized once before the epoch loop and the epoch=1000 "
            "fusion starts from www carried from epoch=900"
        ),
        "www_or_final_assignment_checkpointed": False,
        "available_b2_artifacts": "corruption audit, corruption mask/pairs, five model checkpoints, text log",
        "infeasibility_reason": (
            None
            if exact_replay_feasible
            else (
                "The frozen checkpoints contain post-epoch1000 model parameters "
                "but not the pre-update parameters, carried www state, or final "
                "fusion assignment used by the historical in-loop metrics."
            )
        ),
    }
    inputs = prepared["fusion_inputs"]
    canonical_audit = canonical_base_reproduction_audit(
        seed=seed,
        actual_metrics=metrics,
        actual_backbone_hash=prepared["backbone_hash_before"]["aggregate"],
        actual_normalized_z_hash=prepared["z_hash"],
        actual_fused_hash=base["historical_style_final_fused_sha256"],
        actual_prediction_hash=base["historical_style_prediction_sha256"],
        manifest_path=canonical_manifest_path,
    )
    historical_pass = bool(audit["B6_WQ0_BASE_REPRO_PASS"])
    canonical_pass = bool(
        canonical_audit["B6_WQ0_CANONICAL_BASE_REPRO_PASS"]
    )
    identity_pass = bool(base["B6_WQ0_IDENTITY_ADMISSION_PASS"])
    scientific_action_gate_pass = scientific_action_gate(
        canonical_pass,
        identity_pass,
        frozen_pass,
    )
    if not canonical_pass:
        scientific_action_block_reason = canonical_audit[
            "scientific_action_block_reason"
        ]
    elif not identity_pass:
        scientific_action_block_reason = "identity admission audit failed"
    elif not frozen_pass:
        scientific_action_block_reason = "frozen backbone audit failed"
    else:
        scientific_action_block_reason = None
    return {
        "stage": STAGE,
        "dataset": DATASET_NAME,
        "condition": CONDITION,
        "model_seed": seed,
        "architecture_audit": architecture_audit(),
        "backbone_hash": prepared["backbone_hash_before"],
        "backbone_hash_before": prepared["backbone_hash_before"],
        "backbone_hash_after": backbone_after,
        "z_hash": prepared["z_hash"],
        "source_utility_sha256": prepared["utility_hash"],
        "source_utility_shape": list(prepared["utility"].shape),
        "source_utility_path": _display(prepared["utility_path"]),
        "fusion_updates": prepared["fusion_updates"],
        "fusion_updates_source": prepared["fusion_updates_source"],
        "kmeans_replay": {
            "estimator": type(_kmeans(seed, SAMPLE_NUM)).__name__,
            "n_clusters": CLUSTER_NUM,
            "n_init": 100,
            "random_state": seed,
            "mini_batch_threshold": 10000,
        },
        "per_view_base_audit": inputs["per_view_audit"],
        "fusion_update_audit": base["fusion_update_audit"],
        "final_www_used_for_historical_fusion": (
            base["final_www_used_for_historical_fusion"]
        ),
        "historical_style_final_fused_sha256": (
            base["historical_style_final_fused_sha256"]
        ),
        "historical_style_prediction_sha256": (
            base["historical_style_prediction_sha256"]
        ),
        "first_divergence_point": base["first_divergence_point"],
        "historical_replay_provenance": replay_provenance,
        "base_metrics": metrics,
        "canonical_frozen_metrics": metrics,
        "canonical_baseline_definition": (
            "fixed B2 checkpoints + fixed weak-quality corruption + fixed "
            "model seed + original-MVCAN-style pre-fusion KMeans replay + "
            "per-view MinMax scaling + config T1 update order + frozen "
            "KMeans readout, with no optimizer/backward/parameter updates"
        ),
        "historical_reference_kind": HISTORICAL_REFERENCE_KIND,
        "historical_b2_metrics": audit["expected_metrics"],
        "historical_b2_reproduction": audit,
        "base_reproduction": audit,
        "canonical_base_reproduction": canonical_audit,
        "identity_admission_audit": base["identity_admission_audit"],
        "correct_u_metrics": None,
        "shuffled_primary_metrics": None,
        "shuffled_null_summary": None,
        "delta_vs_base": None,
        "delta_vs_shuffled": None,
        "original_attention_entropy": attention_summary(base["original_attention"]),
        "correct_u_attention_entropy": None,
        "shuffled_attention_entropy": None,
        "labels_used_for_training": False,
        "labels_used_for_admission": False,
        "labels_used_for_evaluation": True,
        "corruption_mask_used_for_admission": False,
        "optimizer_created": False,
        "backward_performed": False,
        "parameter_updates": False,
        "scientific_comparison_performed": False,
        "B6_WQ0_PREFUSION_ASSIGNMENT_MATCH": (
            base["B6_WQ0_PREFUSION_ASSIGNMENT_MATCH"]
        ),
        "B6_WQ0_IDENTITY_ADMISSION_PASS": (
            base["B6_WQ0_IDENTITY_ADMISSION_PASS"]
        ),
        "B6_WQ0_HISTORICAL_B2_REPRO_PASS": historical_pass,
        "B6_WQ0_CANONICAL_BASE_REPRO_PASS": canonical_pass,
        "B6_WQ0_SCIENTIFIC_ACTION_GATE_PASS": scientific_action_gate_pass,
        "B6_WQ0_SCIENTIFIC_ACTION_BLOCKED": not scientific_action_gate_pass,
        "scientific_action_block_reason": scientific_action_block_reason,
        # Backward-compatible diagnostic alias; never gates B6 action.
        "B6_WQ0_BASE_REPRO_PASS": historical_pass,
        "B6_WQ0_FROZEN_B2_EXACT_REPLAY_FEASIBLE": exact_replay_feasible,
        "B6_WQ0_BACKBONE_FROZEN_PASS": frozen_pass,
        "B6_WQ0_SEED_UTILITY_ADMISSION_SIGNAL": None,
    }


def evaluate_seed(
    seed,
    output_root=DEFAULT_OUTPUT_DIR,
    base_only=False,
    canonical_manifest_path=CANONICAL_MANIFEST_PATH,
):
    prepared = prepare_seed(seed)
    base, base_metrics, base_audit = run_base(prepared)
    result = _base_result(
        prepared,
        base,
        base_metrics,
        base_audit,
        canonical_manifest_path=canonical_manifest_path,
    )
    seed_dir = _resolve(output_root) / ("seed" + str(seed))
    result_path = seed_dir / ("b6_wq0_seed" + str(seed) + ".json")
    if base_only or not result["B6_WQ0_SCIENTIFIC_ACTION_GATE_PASS"]:
        _write_json(result_path, result)
        return result

    utility = prepared["utility"]
    original = base["original_attention"]
    correct = admission_weights(original, utility)
    correct_fused = fused_representation_from_admission(
        prepared["fusion_inputs"]["scaled_z_views"], correct
    )
    _, correct_metrics = evaluate_fused_representation(correct_fused, seed, prepared["labels"])
    bank = generate_within_view_permutations()
    null_values = {name: np.empty(NULL_REPEATS) for name in ("acc", "nmi", "ari")}
    primary_utility = primary_attention = primary_metrics = None
    for repeat_id in range(NULL_REPEATS):
        shuffled_utility = shuffle_utility_within_views(utility, bank[repeat_id])
        shuffled_attention = admission_weights(original, shuffled_utility)
        shuffled_fused = fused_representation_from_admission(
            prepared["fusion_inputs"]["scaled_z_views"], shuffled_attention
        )
        _, metrics = evaluate_fused_representation(shuffled_fused, seed, prepared["labels"])
        for name in null_values:
            null_values[name][repeat_id] = metrics[name]
        if repeat_id == 0:
            primary_utility = shuffled_utility
            primary_attention = shuffled_attention
            primary_metrics = metrics
    save_null_arrays(
        seed_dir, null_values["acc"], null_values["nmi"], null_values["ari"]
    )
    np.save(seed_dir / "original_attention.npy", original)
    np.save(seed_dir / "source_utility.npy", utility)
    np.save(seed_dir / "correct_u_admission_weight.npy", correct)
    np.save(seed_dir / "shuffled_primary_utility.npy", primary_utility)
    np.save(seed_dir / "shuffled_primary_admission_weight.npy", primary_attention)
    null_summary = {
        name: summarize_null(null_values[name], correct_metrics[name])
        for name in null_values
    }
    delta_base = delta_metrics(correct_metrics, base_metrics)
    delta_shuffled = delta_metrics(correct_metrics, primary_metrics)
    signal = bool(
        delta_base["acc"] > 0.0
        and correct_metrics["acc"] > null_summary["acc"]["p95"]
        and null_summary["acc"]["one_sided_empirical_p"] < 0.05
    )
    backbone_after = hash_backbone(prepared["models"].autoencoders)
    frozen_pass = bool(backbone_after == prepared["backbone_hash_before"])
    _require(frozen_pass, "backbone changed")
    result.update({
        "backbone_hash_after": backbone_after,
        "correct_u_metrics": correct_metrics,
        "shuffled_primary_metrics": primary_metrics,
        "shuffled_null_summary": null_summary,
        "shuffled_null_repeats": NULL_REPEATS,
        "shuffled_null_seed": SHUFFLE_SEED,
        "delta_vs_base": delta_base,
        "delta_vs_shuffled": delta_shuffled,
        "correct_u_attention_entropy": attention_summary(correct),
        "shuffled_attention_entropy": attention_summary(primary_attention),
        "scientific_comparison_performed": True,
        "B6_WQ0_BACKBONE_FROZEN_PASS": frozen_pass,
        "B6_WQ0_SEED_UTILITY_ADMISSION_SIGNAL": signal,
    })
    _write_json(result_path, result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SUPPORTED_SEEDS))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument(
        "--canonical-manifest",
        default=str(CANONICAL_MANIFEST_PATH),
    )
    return parser.parse_args(argv)


def _validated_seed_mode(seeds, base_only=False):
    selected = tuple(int(seed) for seed in seeds)
    _require(
        len(selected) == len(set(selected)),
        "duplicate seed",
    )
    _require(
        all(seed in SUPPORTED_SEEDS for seed in selected),
        "unsupported seed",
    )
    if not base_only:
        _require(
            len(selected) == 1 or selected == SUPPORTED_SEEDS,
            "full evaluation requires one supported seed or seeds 20 30 50",
        )
    return selected


def _print_base_diagnostics(result):
    seed = int(result["model_seed"])
    print(
        "seed" + str(seed) + " historical B2 metrics="
        + str(result["base_reproduction"]["expected_metrics"])
    )
    print("seed" + str(seed) + " replayed BASE metrics=" + str(result["base_metrics"]))
    for row in result["per_view_base_audit"]:
        print(
            "seed" + str(seed)
            + " pre-fusion view" + str(row["view_id"])
            + " assignment_disagreement_fraction="
            + str(row["assignment_disagreement_fraction"])
        )
    for update in result["fusion_update_audit"]:
        print(
            "seed" + str(seed)
            + " www_before_update_" + str(update["update_id"]) + "="
            + str(update["www_before"])
        )
        print(
            "seed" + str(seed)
            + " www_after_update_" + str(update["update_id"]) + "="
            + str(update["www_after"])
        )
    print(
        "seed" + str(seed) + " final_www_used_for_historical_fusion="
        + str(result["final_www_used_for_historical_fusion"])
    )
    print(
        "seed" + str(seed) + " first_divergence_point="
        + str(result["first_divergence_point"])
    )
    for flag in (
        "B6_WQ0_PREFUSION_ASSIGNMENT_MATCH",
        "B6_WQ0_IDENTITY_ADMISSION_PASS",
        "B6_WQ0_HISTORICAL_B2_REPRO_PASS",
        "B6_WQ0_CANONICAL_BASE_REPRO_PASS",
        "B6_WQ0_SCIENTIFIC_ACTION_GATE_PASS",
        "B6_WQ0_FROZEN_B2_EXACT_REPLAY_FEASIBLE",
    ):
        print(flag + "=" + str(result[flag]).lower())


def main(argv=None):
    args = parse_args(argv)
    seeds = _validated_seed_mode(
        args.seeds,
        base_only=args.base_only,
    )
    results = []
    for seed in seeds:
        result = evaluate_seed(
            seed,
            args.output_dir,
            base_only=args.base_only,
            canonical_manifest_path=args.canonical_manifest,
        )
        _print_base_diagnostics(result)
        results.append(result)
        if (
            not args.base_only
            and not result["B6_WQ0_SCIENTIFIC_ACTION_GATE_PASS"]
        ):
            print("B6_WQ0_STOP_SCIENTIFIC_ACTION_GATE_FAILURE=true")
            print(
                "scientific_action_block_reason="
                + str(result["scientific_action_block_reason"])
            )
            return result
    if args.base_only:
        print("B6_WQ0_BASE_ONLY_COMPLETE=true")
        return results
    if len(seeds) == 1:
        result = results[0]
        print("B6_WQ0_SINGLE_SEED_SCIENTIFIC_COMPLETE=true")
        print("model_seed=" + str(result["model_seed"]))
        return result

    gate = multiseed_gate(results)
    summary = {
        "stage": STAGE,
        "dataset": DATASET_NAME,
        "condition": CONDITION,
        "seed_results": results,
        **gate,
    }
    _write_json(_resolve(args.output_dir) / "b6_wq0_multiseed_summary.json", summary)
    print(
        "B6_WQ0_MULTISEED_ADMISSION_PASS="
        + str(gate["B6_WQ0_MULTISEED_ADMISSION_PASS"]).lower()
    )
    return summary


if __name__ == "__main__":
    main()
