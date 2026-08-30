"""B6-WQ0-R1 read-only cross-path canonical replay audit."""

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import sklearn
import torch
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import experiments.b6_weak_quality.audit_b6_wq0_frozen_baseline_provenance as provenance
import experiments.b6_weak_quality.evaluate_b6_wq0_utility_semantic_admission as b6
from irv.b3_audit import hash_backbone
from irv.b4_information_utility import tensor_sha256

STAGE = "B6-WQ0-R1"
DEFAULT_SEED = 20
SUPPORTED_SEEDS = (20, 30, 50)
DEFAULT_OUTPUT_DIR = "outputs/b6_weak_quality/wq0_cross_path_replay/seed20"
OLD_R1_SNAPSHOT_ARTIFACT = (
    Path(DEFAULT_OUTPUT_DIR) / "b6_wq0_cross_path_seed20.json"
)
AMBIENT_SEED_PATH_A_OFFSET = 11000
AMBIENT_SEED_PATH_B_OFFSET = 91000


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _validate_seed(seed):
    seed = int(seed)
    _require(
        int(seed) in SUPPORTED_SEEDS,
        "unsupported R1 model seed",
    )
    return seed


def _default_output_dir(seed):
    return (
        Path(DEFAULT_OUTPUT_DIR).parent / ("seed" + str(int(seed)))
    )


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _display(path):
    value = Path(path).resolve()
    try:
        return str(value.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(value)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def partition_equivalent(prediction_a, prediction_b):
    """Return true only for exact sample partitions up to a label bijection."""
    left = np.asarray(prediction_a)
    right = np.asarray(prediction_b)
    if left.shape != right.shape or left.ndim != 1:
        return False
    mapping = {}
    used_targets = set()
    for label in np.unique(left):
        targets = np.unique(right[left == label])
        if targets.size != 1:
            return False
        target = targets[0].item()
        if target in used_targets:
            return False
        mapping[label.item()] = target
        used_targets.add(target)
    return bool(len(mapping) == np.unique(right).size)


def dtype_aware_atol(left, right):
    arrays = [np.asarray(left), np.asarray(right)]
    float_dtypes = [
        array.dtype for array in arrays
        if np.issubdtype(array.dtype, np.floating)
    ]
    if not float_dtypes:
        return 0.0
    dtype = np.result_type(*float_dtypes)
    if dtype.itemsize > np.dtype(np.float32).itemsize:
        dtype = np.dtype(np.float64)
    scale = max(
        1.0,
        *(float(np.max(np.abs(array))) for array in arrays if array.size),
    )
    return float(16.0 * np.finfo(dtype).eps * scale)


def compare_arrays(left, right):
    left_array = np.ascontiguousarray(_as_numpy(left))
    right_array = np.ascontiguousarray(_as_numpy(right))
    same_shape = bool(left_array.shape == right_array.shape)
    result = {
        "hash_a": tensor_sha256(left_array),
        "hash_b": tensor_sha256(right_array),
        "hash_exact": bool(tensor_sha256(left_array) == tensor_sha256(right_array)),
        "dtype_a": str(left_array.dtype),
        "dtype_b": str(right_array.dtype),
        "shape_a": list(left_array.shape),
        "shape_b": list(right_array.shape),
        "array_exact": bool(same_shape and np.array_equal(left_array, right_array)),
        "dtype_aware_atol": dtype_aware_atol(left_array, right_array),
    }
    if not same_shape:
        result.update({
            "max_abs_error": None,
            "mean_abs_error": None,
            "relative_l2_error": None,
            "within_dtype_tolerance": False,
        })
        return result
    delta = (
        left_array.astype(np.float64, copy=False)
        - right_array.astype(np.float64, copy=False)
    )
    absolute = np.abs(delta)
    denominator = max(
        float(np.linalg.norm(left_array.astype(np.float64, copy=False))),
        np.finfo(np.float64).tiny,
    )
    max_abs_error = float(np.max(absolute)) if absolute.size else 0.0
    result.update({
        "max_abs_error": max_abs_error,
        "mean_abs_error": float(np.mean(absolute)) if absolute.size else 0.0,
        "relative_l2_error": float(np.linalg.norm(delta) / denominator),
        "within_dtype_tolerance": bool(
            max_abs_error <= result["dtype_aware_atol"]
        ),
    })
    return result


def compare_partitions(left, right):
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    same_shape = bool(left_array.shape == right_array.shape)
    if not same_shape:
        return {
            "raw_hash_a": tensor_sha256(left_array),
            "raw_hash_b": tensor_sha256(right_array),
            "raw_hash_exact": False,
            "array_exact": False,
            "ari": None,
            "nmi": None,
            "partition_equivalent": False,
        }
    return {
        "raw_hash_a": tensor_sha256(left_array),
        "raw_hash_b": tensor_sha256(right_array),
        "raw_hash_exact": bool(
            tensor_sha256(left_array) == tensor_sha256(right_array)
        ),
        "array_exact": bool(np.array_equal(left_array, right_array)),
        "ari": float(adjusted_rand_score(left_array, right_array)),
        "nmi": float(normalized_mutual_info_score(left_array, right_array)),
        "partition_equivalent": partition_equivalent(left_array, right_array),
    }


def first_divergence_from_layers(layers):
    for layer in layers:
        if not bool(layer["equal"]):
            return layer["stage"]
    return "none"


class _CapturedEstimator:
    def __init__(self, estimator, factory_id, metadata, events):
        self._estimator = estimator
        self._factory_id = int(factory_id)
        self._metadata = metadata
        self._events = events
        self._fit_id = 0

    def fit_predict(self, values):
        input_array = np.ascontiguousarray(np.asarray(values)).copy()
        predictions = self._estimator.fit_predict(values)
        prediction_array = np.ascontiguousarray(
            np.asarray(predictions)
        ).copy()
        self._events.append({
            "factory_id": self._factory_id,
            "fit_id": int(self._fit_id),
            "input": input_array,
            "prediction": prediction_array,
            "estimator": dict(self._metadata),
        })
        self._fit_id += 1
        return predictions

    def __getattr__(self, name):
        return getattr(self._estimator, name)


def _kmeans_capture_factory(events, original_kmeans):
    factory_count = {"value": 0}

    def factory(seed, sample_num=b6.SAMPLE_NUM):
        estimator = original_kmeans(seed, sample_num=sample_num)
        factory_id = factory_count["value"]
        factory_count["value"] += 1
        params = estimator.get_params(deep=False)
        metadata = {
            "implementation_class": type(estimator).__name__,
            "random_state": params.get("random_state"),
            "n_init": params.get("n_init"),
            "algorithm": params.get("algorithm"),
            "batch_size": params.get("batch_size"),
            "sample_num": int(sample_num),
            "mini_batch_condition": bool(int(sample_num) > 10000),
        }
        return _CapturedEstimator(
            estimator, factory_id, metadata, events
        )

    return factory


def _set_ambient_seeds(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return {
        "python_random_seed": seed,
        "numpy_global_seed": seed,
        "torch_seed": seed,
        "cuda_available": cuda_available,
        "cuda_seed": seed if cuda_available else None,
    }


def _select_kmeans_events(events, fusion_updates):
    raw_events = [
        event for event in events
        if tuple(event["input"].shape) == (b6.SAMPLE_NUM, b6.LATENT_DIM)
    ]
    fusion_like = [
        event for event in events
        if tuple(event["input"].shape) == (b6.SAMPLE_NUM, b6.FUSION_DIM)
    ]
    _require(len(raw_events) == b6.VIEW_NUM, "missing per-view KMeans events")
    _require(
        len(fusion_like) >= int(fusion_updates),
        "missing fusion KMeans events",
    )
    return raw_events, fusion_like[:int(fusion_updates)]


def _temporary_q_assignments(prepared):
    assignments = []
    for assignment, row in zip(
        prepared["fusion_inputs"]["original_style_view_assignments"],
        prepared["fusion_inputs"]["per_view_audit"],
    ):
        temporary_q = np.asarray(assignment, dtype=np.int64)
        _require(
            tensor_sha256(temporary_q)
            == row["kmeans_initialized_q_assignment_sha256"],
            "temporary-q reconstruction hash mismatch",
        )
        assignments.append(temporary_q)
    return assignments


def _runtime_metadata(ambient):
    return {
        **ambient,
        "python_rng_policy": "ambient seed probe; replay code does not use Python random",
        "numpy_rng_policy": (
            "ambient seed probe; corruption uses local RandomState(explicit seed) "
            "and KMeans uses explicit random_state"
        ),
        "torch_rng_policy": (
            "ambient seed probe; model initialization consumes Torch RNG, then "
            "strict checkpoint loading replaces all model parameters"
        ),
        "kmeans_random_state": "explicit model_seed",
        "kmeans_n_init": 100,
        "kmeans_implementation_class": "KMeans for N=210",
        "mini_batch_kmeans_condition": "N>10000 (false for N=210)",
        "thread_settings": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "torch_num_threads": int(torch.get_num_threads()),
        },
        "library_versions": {
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }


def _build_state(
    path_name,
    prepared,
    base,
    metrics,
    historical_audit,
    events,
    ambient,
    provenance_record=None,
):
    raw_events, fusion_events = _select_kmeans_events(
        events, prepared["fusion_updates"]
    )
    backbone_after = hash_backbone(prepared["models"].autoencoders)
    return {
        "path_name": path_name,
        "prepared": prepared,
        "base": base,
        "metrics": metrics,
        "historical_audit": historical_audit,
        "provenance_record": provenance_record,
        "raw_kmeans_events": raw_events,
        "fusion_kmeans_events": fusion_events,
        "temporary_q_assignments": _temporary_q_assignments(prepared),
        "backbone_hash_before": prepared["backbone_hash_before"],
        "backbone_hash_after": backbone_after,
        "backbone_unchanged": bool(
            prepared["backbone_hash_before"] == backbone_after
        ),
        "runtime_metadata": _runtime_metadata(ambient),
    }


def run_provenance_path(seed, ambient_seed):
    """Call the real provenance replay_once path and capture its live tensors."""
    original_kmeans = b6._kmeans
    original_prepare = b6.prepare_seed
    original_run_base = b6.run_base
    events = []
    captured = {}

    def prepare_capture(model_seed):
        prepared = original_prepare(model_seed)
        captured["prepared"] = prepared
        return prepared

    def run_base_capture(prepared):
        result = original_run_base(prepared)
        captured["base_result"] = result
        return result

    ambient = _set_ambient_seeds(ambient_seed)
    factory = _kmeans_capture_factory(events, original_kmeans)
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(b6, "_kmeans", factory)
        )
        stack.enter_context(
            patch.object(b6, "prepare_seed", prepare_capture)
        )
        stack.enter_context(
            patch.object(b6, "run_base", run_base_capture)
        )
        record = provenance.replay_once(seed)
    _require(
        "prepared" in captured and "base_result" in captured,
        "provenance helper capture failed",
    )
    base, metrics, historical_audit = captured["base_result"]
    return _build_state(
        "path_a_provenance_helper",
        captured["prepared"],
        base,
        metrics,
        historical_audit,
        events,
        ambient,
        provenance_record=record,
    )


def run_evaluator_path(seed, ambient_seed):
    """Call the real evaluator prepare_seed/run_base path."""
    original_kmeans = b6._kmeans
    events = []
    ambient = _set_ambient_seeds(ambient_seed)
    factory = _kmeans_capture_factory(events, original_kmeans)
    with patch.object(b6, "_kmeans", factory):
        prepared = b6.prepare_seed(seed)
        base, metrics, historical_audit = b6.run_base(prepared)
    return _build_state(
        "path_b_evaluator_helper",
        prepared,
        base,
        metrics,
        historical_audit,
        events,
        ambient,
    )


def _assignment_layers(path_a, path_b):
    inputs_a = path_a["prepared"]["fusion_inputs"]
    inputs_b = path_b["prepared"]["fusion_inputs"]
    kinds = (
        (
            "checkpoint_native",
            inputs_a["checkpoint_native_view_assignments"],
            inputs_b["checkpoint_native_view_assignments"],
        ),
        (
            "kmeans_initialized",
            inputs_a["original_style_view_assignments"],
            inputs_b["original_style_view_assignments"],
        ),
        (
            "temporary_q_initialized",
            path_a["temporary_q_assignments"],
            path_b["temporary_q_assignments"],
        ),
    )
    output = {}
    layers = []
    for kind, assignments_a, assignments_b in kinds:
        rows = []
        for view_id, (assignment_a, assignment_b) in enumerate(
            zip(assignments_a, assignments_b)
        ):
            comparison = compare_partitions(assignment_a, assignment_b)
            comparison["view_id"] = int(view_id)
            rows.append(comparison)
            layers.append({
                "stage": (
                    "prefusion_" + kind + "_assignment_view"
                    + str(view_id)
                ),
                "equal": comparison["array_exact"],
            })
        output[kind] = rows
    return output, layers


def compare_live_paths(path_a, path_b):
    prepared_a = path_a["prepared"]
    prepared_b = path_b["prepared"]
    inputs_a = prepared_a["fusion_inputs"]
    inputs_b = prepared_b["fusion_inputs"]
    layers = []
    input_audit = {
        "backbone_hash_a": path_a["backbone_hash_before"]["aggregate"],
        "backbone_hash_b": path_b["backbone_hash_before"]["aggregate"],
        "backbone_hash_exact": bool(
            path_a["backbone_hash_before"]["aggregate"]
            == path_b["backbone_hash_before"]["aggregate"]
        ),
        "corruption_mask_hash_a": prepared_a["condition_audit"]["mask_sha256"],
        "corruption_mask_hash_b": prepared_b["condition_audit"]["mask_sha256"],
        "corruption_mask_hash_exact": bool(
            prepared_a["condition_audit"]["mask_sha256"]
            == prepared_b["condition_audit"]["mask_sha256"]
        ),
        "corrupted_view_hashes_a": prepared_a["condition_audit"][
            "corrupted_view_sha256"
        ],
        "corrupted_view_hashes_b": prepared_b["condition_audit"][
            "corrupted_view_sha256"
        ],
        "corrupted_view_hashes_exact": bool(
            prepared_a["condition_audit"]["corrupted_view_sha256"]
            == prepared_b["condition_audit"]["corrupted_view_sha256"]
        ),
    }
    layers.extend([
        {"stage": "backbone", "equal": input_audit["backbone_hash_exact"]},
        {
            "stage": "corruption_mask",
            "equal": input_audit["corruption_mask_hash_exact"],
        },
        {
            "stage": "corrupted_views",
            "equal": input_audit["corrupted_view_hashes_exact"],
        },
    ])

    representations = {}
    representation_specs = (
        ("raw_z", inputs_a["raw_z_views"], inputs_b["raw_z_views"]),
        (
            "normalized_z",
            inputs_a["normalized_z_views"],
            inputs_b["normalized_z_views"],
        ),
        (
            "scaled_z",
            inputs_a["scaled_z_views"],
            inputs_b["scaled_z_views"],
        ),
    )
    for name, views_a, views_b in representation_specs:
        rows = []
        for view_id, (view_a, view_b) in enumerate(zip(views_a, views_b)):
            comparison = compare_arrays(view_a, view_b)
            comparison["view_id"] = int(view_id)
            rows.append(comparison)
            layers.append({
                "stage": name + "_view" + str(view_id),
                "equal": comparison["array_exact"],
            })
        representations[name] = rows

    assignments, assignment_layers = _assignment_layers(path_a, path_b)
    layers.extend(assignment_layers)
    fusion_updates = []
    for update_id, (event_a, event_b, audit_a, audit_b) in enumerate(zip(
        path_a["fusion_kmeans_events"],
        path_b["fusion_kmeans_events"],
        path_a["base"]["fusion_update_audit"],
        path_b["base"]["fusion_update_audit"],
    )):
        www_before = compare_arrays(
            np.asarray(audit_a["www_before"]),
            np.asarray(audit_b["www_before"]),
        )
        fused = compare_arrays(event_a["input"], event_b["input"])
        prediction = compare_partitions(
            event_a["prediction"], event_b["prediction"]
        )
        www_after = compare_arrays(
            np.asarray(audit_a["www_after"]),
            np.asarray(audit_b["www_after"]),
        )
        fusion_updates.append({
            "update_id": int(update_id),
            "www_before": www_before,
            "fused_representation": fused,
            "prediction": prediction,
            "www_after": www_after,
            "kmeans_path_a": event_a["estimator"],
            "kmeans_path_b": event_b["estimator"],
        })
        layers.extend([
            {
                "stage": "www_before_update" + str(update_id),
                "equal": www_before["array_exact"],
            },
            {
                "stage": "fused_update" + str(update_id),
                "equal": fused["array_exact"],
            },
            {
                "stage": "prediction_update" + str(update_id),
                "equal": prediction["array_exact"],
            },
            {
                "stage": "www_after_update" + str(update_id),
                "equal": www_after["array_exact"],
            },
        ])

    final_www = compare_arrays(
        np.asarray(path_a["base"]["final_www_used_for_historical_fusion"]),
        np.asarray(path_b["base"]["final_www_used_for_historical_fusion"]),
    )
    final_fused = compare_arrays(
        path_a["base"]["fused_representation"],
        path_b["base"]["fused_representation"],
    )
    final_prediction = compare_partitions(
        path_a["base"]["predictions"],
        path_b["base"]["predictions"],
    )
    metrics_exact = bool(path_a["metrics"] == path_b["metrics"])
    layers.extend([
        {"stage": "final_www_used", "equal": final_www["array_exact"]},
        {"stage": "final_fused", "equal": final_fused["array_exact"]},
        {
            "stage": "final_prediction",
            "equal": final_prediction["array_exact"],
        },
        {"stage": "final_metrics", "equal": metrics_exact},
    ])
    return {
        "input_provenance": input_audit,
        "representations": representations,
        "prefusion_assignments": assignments,
        "fusion_updates": fusion_updates,
        "final": {
            "final_www_used": final_www,
            "final_fused": final_fused,
            "final_prediction": final_prediction,
            "metrics_path_a": path_a["metrics"],
            "metrics_path_b": path_b["metrics"],
            "metrics_exact": metrics_exact,
        },
        "layers": layers,
        "first_divergence_stage": first_divergence_from_layers(layers),
    }


def compare_registered_provenance(live_record, registered_record):
    layers = [
        {
            "stage": "backbone",
            "equal": (
                live_record["backbone_hash"]
                == registered_record["backbone_hash"]
            ),
        },
        {
            "stage": "corruption_mask",
            "equal": (
                live_record["corruption_mask_sha256"]
                == registered_record["corruption_mask_sha256"]
            ),
        },
        {
            "stage": "corrupted_views",
            "equal": (
                live_record["corrupted_view_sha256"]
                == registered_record["corrupted_view_sha256"]
            ),
        },
    ]
    list_specs = (
        ("raw_z", "raw_z_sha256_per_view"),
        ("normalized_z", "normalized_z_sha256_per_view"),
        ("scaled_z", "scaled_z_sha256_per_view"),
        (
            "checkpoint_native_assignment",
            "checkpoint_native_assignment_sha256_per_view",
        ),
        (
            "prefusion_assignment",
            "kmeans_initialized_assignment_sha256_per_view",
        ),
    )
    for stage_prefix, key in list_specs:
        for view_id, (live_hash, registered_hash) in enumerate(zip(
            live_record[key], registered_record[key]
        )):
            layers.append({
                "stage": stage_prefix + "_view" + str(view_id),
                "equal": bool(live_hash == registered_hash),
            })
    for update_id, (live_update, registered_update) in enumerate(zip(
        live_record["fusion_updates"],
        registered_record["fusion_updates"],
    )):
        layers.extend([
            {
                "stage": "www_before_update" + str(update_id),
                "equal": (
                    live_update["www_before"]
                    == registered_update["www_before"]
                ),
            },
            {
                "stage": "fused_update" + str(update_id),
                "equal": (
                    live_update["fused_representation_sha256"]
                    == registered_update["fused_representation_sha256"]
                ),
            },
            {
                "stage": "prediction_update" + str(update_id),
                "equal": (
                    live_update["prediction_sha256"]
                    == registered_update["prediction_sha256"]
                ),
            },
            {
                "stage": "www_after_update" + str(update_id),
                "equal": (
                    live_update["www_after"]
                    == registered_update["www_after"]
                ),
            },
        ])
    layers.extend([
        {
            "stage": "final_fused",
            "equal": (
                live_record["canonical_fused_representation_sha256"]
                == registered_record[
                    "canonical_fused_representation_sha256"
                ]
            ),
        },
        {
            "stage": "final_prediction",
            "equal": (
                live_record["canonical_prediction_sha256"]
                == registered_record["canonical_prediction_sha256"]
            ),
        },
        {
            "stage": "final_metrics",
            "equal": (
                live_record["canonical_metrics"]
                == registered_record["canonical_metrics"]
            ),
        },
    ])
    return {
        "layers": layers,
        "first_divergence_stage": first_divergence_from_layers(layers),
        "all_exact": bool(all(layer["equal"] for layer in layers)),
    }


def _reference_metadata(state):
    return {
        "model_seed": int(state["prepared"]["seed"]),
        "backbone_hash": state["backbone_hash_before"]["aggregate"],
        "normalized_z_hash": state["prepared"]["z_hash"],
        "canonical_fused_hash": tensor_sha256(
            state["base"]["fused_representation"]
        ),
        "canonical_prediction_hash": tensor_sha256(
            state["base"]["predictions"]
        ),
        "metrics": state["metrics"],
    }


def write_reference(path, state):
    reference_path = _resolve(path)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    fusion_events = state["fusion_kmeans_events"]
    update_audit = state["base"]["fusion_update_audit"]
    payload = {
        "final_fused": np.asarray(state["base"]["fused_representation"]),
        "final_prediction": np.asarray(state["base"]["predictions"]),
        "www_before": np.asarray(
            [row["www_before"] for row in update_audit],
            dtype=np.float64,
        ),
        "www_after": np.asarray(
            [row["www_after"] for row in update_audit],
            dtype=np.float64,
        ),
        "critical_metadata_json": np.asarray(
            json.dumps(_reference_metadata(state), sort_keys=True)
        ),
    }
    for update_id, event in enumerate(fusion_events):
        payload["fused_update_" + str(update_id)] = event["input"]
        payload["prediction_update_" + str(update_id)] = event["prediction"]
    np.savez(reference_path, **payload)
    return reference_path


def compare_reference(path, state):
    reference_path = _resolve(path)
    _require(reference_path.is_file(), "reference NPZ missing")
    with np.load(reference_path, allow_pickle=False) as reference:
        metadata = json.loads(
            str(reference["critical_metadata_json"].item())
        )
        requested_seed = int(state["prepared"]["seed"])
        _require(
            metadata.get("model_seed") == requested_seed,
            "reference model_seed does not match requested seed",
        )
        fused = compare_arrays(
            reference["final_fused"],
            state["base"]["fused_representation"],
        )
        prediction = compare_partitions(
            reference["final_prediction"],
            state["base"]["predictions"],
        )
        www_before = compare_arrays(
            reference["www_before"],
            np.asarray([
                row["www_before"]
                for row in state["base"]["fusion_update_audit"]
            ]),
        )
        www_after = compare_arrays(
            reference["www_after"],
            np.asarray([
                row["www_after"]
                for row in state["base"]["fusion_update_audit"]
            ]),
        )
        per_update = []
        for update_id, event in enumerate(state["fusion_kmeans_events"]):
            per_update.append({
                "update_id": int(update_id),
                "fused": compare_arrays(
                    reference["fused_update_" + str(update_id)],
                    event["input"],
                ),
                "prediction": compare_partitions(
                    reference["prediction_update_" + str(update_id)],
                    event["prediction"],
                ),
            })
    metrics_exact = bool(metadata["metrics"] == state["metrics"])
    return {
        "reference_path": _display(reference_path),
        "fused": fused,
        "prediction": prediction,
        "www_before": www_before,
        "www_after": www_after,
        "per_update": per_update,
        "critical_hashes_exact": bool(
            metadata["backbone_hash"]
            == state["backbone_hash_before"]["aggregate"]
            and metadata["normalized_z_hash"]
            == state["prepared"]["z_hash"]
            and metadata["canonical_fused_hash"]
            == tensor_sha256(state["base"]["fused_representation"])
            and metadata["canonical_prediction_hash"]
            == tensor_sha256(state["base"]["predictions"])
        ),
        "metrics_reference": metadata["metrics"],
        "metrics_current": state["metrics"],
        "metrics_exact": metrics_exact,
    }


def _cross_path_fingerprint(live_comparison):
    final = live_comparison["final"]
    return {
        "path_a_fused_hash": final["final_fused"]["hash_a"],
        "path_b_fused_hash": final["final_fused"]["hash_b"],
        "path_a_prediction_hash": final["final_prediction"]["raw_hash_a"],
        "path_b_prediction_hash": final["final_prediction"]["raw_hash_b"],
        "path_a_metrics": final["metrics_path_a"],
        "path_b_metrics": final["metrics_path_b"],
    }


def _optional_first_divergence(layers):
    for layer in layers:
        if not bool(layer["equal"]):
            return layer["stage"]
    return None


def _old_r1_snapshot_audit(live_comparison):
    reference_path = _resolve(OLD_R1_SNAPSHOT_ARTIFACT)
    _require(
        reference_path.is_file(),
        "old R1 diagnostic snapshot missing",
    )
    with open(reference_path, "r", encoding="utf-8") as input_file:
        reference = json.load(input_file)
    _require(
        reference.get("model_seed") == DEFAULT_SEED,
        "old R1 diagnostic snapshot model_seed mismatch",
    )
    expected = _cross_path_fingerprint(
        reference["live_cross_path_comparison"]
    )
    actual = _cross_path_fingerprint(live_comparison)
    checks = {
        key: bool(actual[key] == expected[key])
        for key in expected
    }
    layers = [
        {
            "stage": "final_fused",
            "equal": (
                checks["path_a_fused_hash"]
                and checks["path_b_fused_hash"]
            ),
        },
        {
            "stage": "final_prediction",
            "equal": (
                checks["path_a_prediction_hash"]
                and checks["path_b_prediction_hash"]
            ),
        },
        {
            "stage": "final_metrics",
            "equal": (
                checks["path_a_metrics"]
                and checks["path_b_metrics"]
            ),
        },
    ]
    return {
        "reference_artifact": _display(reference_path),
        "old_r1_snapshot_reference_kind": "historical-diagnostic-only",
        "old_r1_snapshot_is_scientific_gate": False,
        "old_r1_expected_fused_hash": expected["path_a_fused_hash"],
        "current_live_fused_hash": actual["path_a_fused_hash"],
        "old_r1_expected_prediction_hash": expected[
            "path_a_prediction_hash"
        ],
        "current_live_prediction_hash": actual[
            "path_a_prediction_hash"
        ],
        "old_r1_expected_metrics": expected["path_a_metrics"],
        "current_live_metrics": actual["path_a_metrics"],
        "fingerprint_expected": expected,
        "fingerprint_actual": actual,
        "checks": checks,
        "B6_WQ0_OLD_R1_SNAPSHOT_MATCH": bool(
            all(checks.values())
        ),
        "old_r1_vs_live_first_divergence_stage": (
            _optional_first_divergence(layers)
        ),
    }


def _registered_canonical_audit(seed, state):
    return b6.canonical_base_reproduction_audit(
        seed=seed,
        actual_metrics=state["metrics"],
        actual_backbone_hash=state[
            "backbone_hash_before"
        ]["aggregate"],
        actual_normalized_z_hash=state["prepared"]["z_hash"],
        actual_fused_hash=tensor_sha256(
            state["base"]["fused_representation"]
        ),
        actual_prediction_hash=tensor_sha256(
            state["base"]["predictions"]
        ),
    )


def _registered_canonical_first_divergence(path_a_audit, path_b_audit):
    check_stages = (
        ("backbone_hash_exact", "backbone"),
        ("normalized_z_hash_exact", "normalized_z"),
        ("canonical_fused_hash_exact", "canonical_fused"),
        ("prediction_hash_exact", "prediction"),
        ("canonical_metrics_exact", "canonical_metrics"),
    )
    layers = []
    for path_name, audit in (
        ("path_a", path_a_audit),
        ("path_b", path_b_audit),
    ):
        checks = audit["checks"]
        if checks is None:
            layers.append({
                "stage": path_name + "_canonical_unregistered",
                "equal": False,
            })
            continue
        layers.extend({
            "stage": path_name + "_" + stage,
            "equal": checks[key],
        } for key, stage in check_stages)
    return _optional_first_divergence(layers)


def _load_registered_record(seed):
    seed = _validate_seed(seed)
    manifest, _ = b6.load_canonical_manifest()
    entry = manifest["canonical_entries"].get(str(seed))
    _require(entry is not None, "registered canonical entry missing")
    _require(
        entry.get("model_seed") == seed,
        "registered canonical entry model_seed mismatch",
    )
    source_path = _resolve(entry["source_provenance"])
    _require(source_path.is_file(), "registered source provenance missing")
    with open(source_path, "r", encoding="utf-8") as input_file:
        record = json.load(input_file)
    _require(
        record.get("model_seed") == seed,
        "registered source provenance model_seed mismatch",
    )
    return record, source_path


def run_cross_path_audit(
    seed=DEFAULT_SEED,
    output_dir=None,
    write_reference_path=None,
    compare_reference_path=None,
):
    seed = _validate_seed(seed)
    manifest_hash_before = _file_sha256(b6.CANONICAL_MANIFEST_PATH)
    path_a = run_provenance_path(
        seed, seed + AMBIENT_SEED_PATH_A_OFFSET
    )
    path_b = run_evaluator_path(
        seed, seed + AMBIENT_SEED_PATH_B_OFFSET
    )
    live_comparison = compare_live_paths(path_a, path_b)
    old_snapshot = (
        _old_r1_snapshot_audit(live_comparison)
        if seed == DEFAULT_SEED else None
    )
    old_snapshot_output = {
        "old_r1_snapshot_reference_kind": None,
        "old_r1_snapshot_is_scientific_gate": False,
        "old_r1_expected_fused_hash": None,
        "current_live_fused_hash": None,
        "old_r1_expected_prediction_hash": None,
        "current_live_prediction_hash": None,
        "old_r1_expected_metrics": None,
        "current_live_metrics": None,
        "B6_WQ0_OLD_R1_SNAPSHOT_MATCH": None,
        "old_r1_vs_live_first_divergence_stage": None,
    }
    if old_snapshot is not None:
        old_snapshot_output.update(old_snapshot)

    path_a_canonical = _registered_canonical_audit(seed, path_a)
    path_b_canonical = _registered_canonical_audit(seed, path_b)
    path_a_canonical_match = bool(
        path_a_canonical["B6_WQ0_CANONICAL_BASE_REPRO_PASS"]
    )
    path_b_canonical_match = bool(
        path_b_canonical["B6_WQ0_CANONICAL_BASE_REPRO_PASS"]
    )
    live_paths_canonical_pass = bool(
        path_a_canonical_match and path_b_canonical_match
    )
    registered_first_divergence = (
        _registered_canonical_first_divergence(
            path_a_canonical, path_b_canonical
        )
    )
    cross_path_first_divergence = (
        None
        if live_comparison["first_divergence_stage"] == "none"
        else live_comparison["first_divergence_stage"]
    )

    registered_record, registered_path = _load_registered_record(seed)
    registered_source_comparison = compare_registered_provenance(
        path_a["provenance_record"], registered_record
    )

    final = live_comparison["final"]
    fused = final["final_fused"]
    prediction = final["final_prediction"]
    metrics_exact = final["metrics_exact"]
    fused_bytewise_only = bool(
        not fused["hash_exact"]
        and fused["within_dtype_tolerance"]
        and prediction["ari"] == 1.0
        and metrics_exact
    )
    prediction_label_permutation_only = bool(
        not prediction["raw_hash_exact"]
        and prediction["ari"] == 1.0
        and prediction["partition_equivalent"]
    )
    ambient_dependency = bool(
        live_comparison["first_divergence_stage"] != "none"
    )

    written_reference = None
    if write_reference_path is not None:
        written_reference = write_reference(write_reference_path, path_b)
    cross_process = None
    if compare_reference_path is not None:
        cross_process = compare_reference(compare_reference_path, path_b)

    manifest_hash_after = _file_sha256(b6.CANONICAL_MANIFEST_PATH)
    result = {
        "stage": STAGE,
        "dataset": b6.DATASET_NAME,
        "condition": b6.CONDITION,
        "model_seed": seed,
        "path_a": {
            "name": path_a["path_name"],
            "runtime_metadata": path_a["runtime_metadata"],
            "backbone_unchanged": path_a["backbone_unchanged"],
            "metrics": path_a["metrics"],
        },
        "path_b": {
            "name": path_b["path_name"],
            "runtime_metadata": path_b["runtime_metadata"],
            "backbone_unchanged": path_b["backbone_unchanged"],
            "metrics": path_b["metrics"],
        },
        "independent_model_objects": bool(
            path_a["prepared"]["models"] is not path_b["prepared"]["models"]
        ),
        "live_cross_path_comparison": live_comparison,
        "old_r1_snapshot_diagnostic": old_snapshot,
        **old_snapshot_output,
        "path_a_registered_canonical_audit": path_a_canonical,
        "path_b_registered_canonical_audit": path_b_canonical,
        "B6_WQ0_PATH_A_REGISTERED_CANONICAL_MATCH": (
            path_a_canonical_match
        ),
        "B6_WQ0_PATH_B_REGISTERED_CANONICAL_MATCH": (
            path_b_canonical_match
        ),
        "B6_WQ0_LIVE_PATHS_REGISTERED_CANONICAL_PASS": (
            live_paths_canonical_pass
        ),
        "registered_source_provenance": _display(registered_path),
        "registered_source_provenance_vs_live_path_a_diagnostic": (
            registered_source_comparison
        ),
        "old_r1_vs_live_first_divergence_stage": (
            old_snapshot_output[
                "old_r1_vs_live_first_divergence_stage"
            ]
        ),
        "registered_vs_live_first_divergence_stage": (
            registered_first_divergence
        ),
        "cross_path_first_divergence_stage": (
            cross_path_first_divergence
        ),
        "written_reference": (
            _display(written_reference)
            if written_reference is not None else None
        ),
        "cross_process_comparison": cross_process,
        "cross_path_fused_hash_exact": fused["hash_exact"],
        "cross_path_fused_max_abs_error": fused["max_abs_error"],
        "cross_path_prediction_hash_exact": prediction["raw_hash_exact"],
        "cross_path_prediction_array_exact": prediction["array_exact"],
        "cross_path_prediction_partition_equivalent": prediction[
            "partition_equivalent"
        ],
        "cross_path_prediction_ARI": prediction["ari"],
        "cross_path_metrics_exact": metrics_exact,
        "cross_process_fused_hash_exact": (
            cross_process["fused"]["hash_exact"]
            if cross_process is not None else None
        ),
        "cross_process_fused_max_abs_error": (
            cross_process["fused"]["max_abs_error"]
            if cross_process is not None else None
        ),
        "cross_process_prediction_partition_equivalent": (
            cross_process["prediction"]["partition_equivalent"]
            if cross_process is not None else None
        ),
        "B6_WQ0_AMBIENT_RNG_DEPENDENCY": ambient_dependency,
        "B6_WQ0_FUSED_HASH_BYTEWISE_ONLY_MISMATCH": fused_bytewise_only,
        "B6_WQ0_PREDICTION_HASH_LABEL_PERMUTATION_ONLY": (
            prediction_label_permutation_only
        ),
        "optimizer_created": False,
        "backward_performed": False,
        "parameter_updates": False,
        "backbone_unchanged": bool(
            path_a["backbone_unchanged"]
            and path_b["backbone_unchanged"]
        ),
        "formal_manifest_sha256_before": manifest_hash_before,
        "formal_manifest_sha256_after": manifest_hash_after,
        "formal_manifest_modified": bool(
            manifest_hash_before != manifest_hash_after
        ),
    }
    output_root = _resolve(
        _default_output_dir(seed) if output_dir is None else output_dir
    )
    output_path = output_root / (
        "b6_wq0_cross_path_seed" + str(seed) + ".json"
    )
    _write_json(output_path, result)
    return result


def _print_result(result):
    for key in (
        "old_r1_snapshot_reference_kind",
        "old_r1_snapshot_is_scientific_gate",
        "B6_WQ0_OLD_R1_SNAPSHOT_MATCH",
        "old_r1_vs_live_first_divergence_stage",
        "registered_vs_live_first_divergence_stage",
        "cross_path_first_divergence_stage",
        "B6_WQ0_PATH_A_REGISTERED_CANONICAL_MATCH",
        "B6_WQ0_PATH_B_REGISTERED_CANONICAL_MATCH",
        "B6_WQ0_LIVE_PATHS_REGISTERED_CANONICAL_PASS",
        "cross_path_fused_hash_exact",
        "cross_path_fused_max_abs_error",
        "cross_path_prediction_hash_exact",
        "cross_path_prediction_array_exact",
        "cross_path_prediction_partition_equivalent",
        "cross_path_prediction_ARI",
        "cross_path_metrics_exact",
        "cross_process_fused_hash_exact",
        "cross_process_fused_max_abs_error",
        "cross_process_prediction_partition_equivalent",
        "B6_WQ0_AMBIENT_RNG_DEPENDENCY",
        "B6_WQ0_FUSED_HASH_BYTEWISE_ONLY_MISMATCH",
        "B6_WQ0_PREDICTION_HASH_LABEL_PERMUTATION_ONLY",
    ):
        value = result[key]
        if isinstance(value, bool):
            value = str(value).lower()
        print(key + "=" + str(value))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir")
    parser.add_argument("--write-reference")
    parser.add_argument("--compare-reference")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_cross_path_audit(
        seed=args.seed,
        output_dir=args.output_dir,
        write_reference_path=args.write_reference,
        compare_reference_path=args.compare_reference,
    )
    _print_result(result)
    return result


if __name__ == "__main__":
    main()
