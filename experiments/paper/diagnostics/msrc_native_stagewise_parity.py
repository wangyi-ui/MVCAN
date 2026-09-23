"""P0-A4 diagnostic-only stagewise parity tracer for MSRC initialization.

The module deliberately delegates all scientific work to the frozen historical
adapter and the current P0-A2 initializer.  Monkey patches are scoped and
restored; this file is not imported by release code.
"""

import os


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _thread_name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_name] = "1"

import argparse
import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)
from sklearn.cluster import KMeans

from experiments.paper.transfer_diagnostics import (
    materialize_msrc_current_condition_init as current_initializer,
)
from .msrc_native_initializer_parity_replay import (
    HISTORICAL_INPUT_DIR,
    load_frozen_legacy_adapter,
)
from .state_hashing import (
    array_logical_sha256,
    array_statistics,
    gradient_snapshot,
    model_state_snapshot,
    optimizer_state_snapshot,
    stage_snapshot,
)


STAGES = (
    "S0_CONSTRUCTION",
    "S1_POST_AE_EPOCH_0",
    "S2_POST_AE_EPOCH_1",
    "S3_POST_AE_EPOCH_10",
    "S4_POST_AE_COMPLETION",
    "S5_POST_KMEANS",
    "S6_PRE_NATIVE_0_UPDATE",
    "S7_POST_NATIVE_0",
    "S8_POST_NATIVE_1",
    "S9_POST_NATIVE_99",
    "S10_POST_REFRESH_100_PRE_UPDATE",
    "S11_POST_NATIVE_100",
    "S12_POST_NATIVE_500",
    "S13_POST_NATIVE_1000",
    "S14_FINAL_PRE_R2",
)
AE_CAPTURE_EPOCHS = {0: STAGES[1], 1: STAGES[2], 10: STAGES[3], 199: STAGES[4]}
NATIVE_CAPTURE_EPOCHS = {
    0: STAGES[7], 1: STAGES[8], 99: STAGES[9], 100: STAGES[11],
    500: STAGES[12], 1000: STAGES[13],
}
SELECTED_NATIVE_UPDATES = frozenset(NATIVE_CAPTURE_EPOCHS)
ALLOWED_DECISIONS = frozenset((
    "MODEL_CONSTRUCTION_DIVERGENCE",
    "DATALOADER_RNG_TRAJECTORY_DIVERGENCE",
    "AE_OPTIMIZATION_DIVERGENCE",
    "OPTIMIZER_STATE_CONTINUITY_DIVERGENCE",
    "KMEANS_INITIALIZATION_DIVERGENCE",
    "NATIVE_REFRESH_DIVERGENCE",
    "NATIVE_UPDATE_ORCHESTRATION_DIVERGENCE",
    "NUMERICAL_NONDETERMINISM",
    "EXACT_INITIALIZATION_PARITY",
))


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _json_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def preview_first_batch_ids(generator, sample_count, batch_size):
    """Preview RandomSampler IDs without advancing the live generator."""
    clone = torch.Generator(device="cpu")
    clone.set_state(generator.get_state().clone())
    torch.empty((), dtype=torch.int64).random_(generator=clone)
    order = torch.randperm(int(sample_count), generator=clone).tolist()
    return [int(value) for value in order[:min(int(batch_size), 32)]]


def canonical_native_total(reconstruction, clustering, lambda1):
    """Derive a path-independent diagnostic total from persisted components."""
    return float(reconstruction) + float(lambda1) * float(clustering)


def static_audit_table():
    """Frozen A-H static audit; dynamic-looking differences are not decisions."""
    return [
        {
            "id": "A", "boundary": "RNG setup",
            "historical": "NumPy and torch CPU/CUDA seeds; deterministic "
                          "algorithms/cuDNN; TF32 off",
            "current": "same, plus Python random seed, single-thread controls "
                       "and explicit CUDA device",
            "status": "OBSERVABLE_WRAPPER_DIFFERENCE_REQUIRES_STAGE_HASHES",
        },
        {
            "id": "B", "boundary": "model construction",
            "historical": "MvCAN; view-order construction with per-view seed reset",
            "current": "MultiViewBackbone; same view order, layers and seed reset",
            "status": "NO_OBVIOUS_SCIENTIFIC_DIFFERENCE",
        },
        {
            "id": "C", "boundary": "optimizer topology and continuity",
            "historical": "one Adam per view, created before AE and reused in native",
            "current": "one Adam per view, created before AE and reused in native",
            "status": "NO_OBVIOUS_DIFFERENCE",
        },
        {
            "id": "D", "boundary": "DataLoader RNG trajectory",
            "historical": "one shared seeded CPU generator; new shuffled loader each epoch",
            "current": "one shared seeded CPU generator; new shuffled loader each epoch",
            "status": "NO_OBVIOUS_DIFFERENCE",
        },
        {
            "id": "E", "boundary": "AE pretraining",
            "historical": "200 epochs; per-view MSE, zero_grad/backward/step",
            "current": "200 epochs; per-view MSE, zero_grad/backward/step",
            "status": "NO_OBVIOUS_DIFFERENCE",
        },
        {
            "id": "F", "boundary": "KMeans center initialization",
            "historical": "one KMeans object reused for five fit_predict calls",
            "current": "fresh helper-owned KMeans object for each view",
            "status": "CALL_STRUCTURE_DIFFERENCE_REQUIRES_S4_S5_HASHES",
        },
        {
            "id": "G", "boundary": "native loop",
            "historical": "epochs 0..1000; refresh before update; view 0..4 updates",
            "current": "epochs 0..1000; refresh before update; view 0..4 updates",
            "status": "NO_OBVIOUS_ORDER_DIFFERENCE",
        },
        {
            "id": "H", "boundary": "native refresh",
            "historical": "inline two-pass latent/q recomputation and mutable weights",
            "current": "precompute latent/q, then two-pass helper; tuple weights",
            "status": "WRAPPER_TOPOLOGY_DIFFERENCE_REQUIRES_REFRESH_HASHES",
        },
    ]


def _hash_kmeans_event(estimator, values, labels):
    centers = np.ascontiguousarray(estimator.cluster_centers_)
    return {
        "input_hash": array_logical_sha256(values),
        "labels_hash": array_logical_sha256(labels),
        "centers_hash": array_logical_sha256(centers),
        "input_statistics": array_statistics(values),
        "centers_statistics": array_statistics(centers),
        "n_clusters": int(estimator.n_clusters),
        "n_init": int(estimator.n_init),
        "random_state": int(estimator.random_state),
    }


class TraceController:
    def __init__(self, path_name, native_lambda1):
        self.path_name = path_name
        self.native_lambda1 = float(native_lambda1)
        self.model = None
        self.optimizers = None
        self.full_views = None
        self.generator = None
        self.phase = "ae"
        self.ae_epoch = -1
        self.native_epoch = -1
        self.refresh_count = 0
        self.refresh_details = {}
        self.snapshots = {}
        self.batch_orders = {"ae": {}, "native": {}}
        self.kmeans_events = []
        self.losses = {}
        self.update_details = {}

    def bind_optimizers(self, model, optimizers):
        self.model = model
        self.optimizers = tuple(optimizers)
        for view_id, optimizer in enumerate(self.optimizers):
            original = optimizer.step

            def step_wrapper(*args, _original=original, _view=view_id, **kwargs):
                return self._optimizer_step(_view, _original, args, kwargs)

            optimizer.step = step_wrapper

    def observe_loader(self, dataset, batch_size, generator):
        _require(generator is not None, "diagnostic requires explicit DataLoader generator")
        _require(hasattr(dataset, "tensors"), "diagnostic expects TensorDataset")
        if self.full_views is None:
            self.full_views = tuple(dataset.tensors[:len(self.model.autoencoders)])
            self.generator = generator
            self.capture(STAGES[0])
        _require(generator is self.generator, "DataLoader generator identity changed")
        if self.phase == "ae":
            self.ae_epoch += 1
            epoch = self.ae_epoch
        else:
            self.native_epoch += 1
            epoch = self.native_epoch
        if epoch in ({0, 1, 10, 199} if self.phase == "ae" else SELECTED_NATIVE_UPDATES):
            self.batch_orders[self.phase][str(epoch)] = preview_first_batch_ids(
                generator, len(dataset), batch_size
            )

    def _optimizer_step(self, view_id, original, args, kwargs):
        selected = self.phase == "native" and self.native_epoch in SELECTED_NATIVE_UPDATES
        if selected:
            model_before = model_state_snapshot(self.model)["per_view"][view_id]
            optimizer_before = optimizer_state_snapshot(self.optimizers, self.model)
            gradient = gradient_snapshot(self.model)[view_id]
        result = original(*args, **kwargs)
        if selected:
            epoch_key = str(self.native_epoch)
            detail = self.update_details.setdefault(epoch_key, [])
            detail.append({
                "view_id": view_id,
                "losses": self.losses.get(epoch_key, {}).get(str(view_id)),
                "model_before": model_before,
                "gradient": gradient,
                "optimizer_before": optimizer_before,
                "model_after": model_state_snapshot(self.model)["per_view"][view_id],
                "optimizer_after": optimizer_state_snapshot(self.optimizers, self.model),
            })
        if view_id == len(self.optimizers) - 1:
            if self.phase == "ae" and self.ae_epoch in AE_CAPTURE_EPOCHS:
                self.capture(
                    AE_CAPTURE_EPOCHS[self.ae_epoch],
                    first_batch_ids=self.batch_orders["ae"].get(str(self.ae_epoch)),
                )
            elif self.phase == "native" and self.native_epoch in NATIVE_CAPTURE_EPOCHS:
                self.capture(
                    NATIVE_CAPTURE_EPOCHS[self.native_epoch],
                    first_batch_ids=self.batch_orders["native"].get(str(self.native_epoch)),
                    native_update=self.update_details.get(str(self.native_epoch)),
                    native_refresh=self.refresh_details.get(str(self.native_epoch)),
                )
        return result

    def capture_loss(self, view_id, rec, clu, raw_total):
        if self.phase != "native" or self.native_epoch not in SELECTED_NATIVE_UPDATES:
            return
        reconstruction = float(rec.detach().cpu().item())
        clustering = float(clu.detach().cpu().item())
        self.losses.setdefault(str(self.native_epoch), {})[str(view_id)] = {
            "reconstruction": reconstruction,
            "clustering": clustering,
            "canonical_total": canonical_native_total(
                reconstruction, clustering, self.native_lambda1
            ),
            "raw_total_tensor_value": float(raw_total.detach().cpu().item()),
        }

    def before_refresh(self):
        if self.refresh_count == 0:
            self.capture(STAGES[5], kmeans_initialization=self.kmeans_events[:5])

    def after_refresh(self, result):
        epoch = self.refresh_count * 100
        recent = self.kmeans_events[-2:]
        p_all = result[0]
        matches = result[1]
        weights = result[-1]
        refresh = {
            "epoch": epoch,
            "kmeans_passes": recent,
            "global_latent_concat_hash": recent[-1]["input_hash"],
            "global_kmeans_ids_hash": recent[-1]["labels_hash"],
            "global_kmeans_centers_hash": recent[-1]["centers_hash"],
            "p_all_hash": array_logical_sha256(p_all.detach().cpu().numpy()),
            "matches_hash": array_logical_sha256(matches.detach().cpu().numpy()),
            "p_local_per_view_hash": [
                array_logical_sha256(
                    (p_all @ matches[view_id].detach()).detach().cpu().numpy()
                )
                for view_id in range(matches.shape[0])
            ],
            "view_weights": [float(value) for value in weights],
        }
        self.refresh_details[str(epoch)] = refresh
        self.phase = "native"
        if epoch == 0:
            self.capture(STAGES[6], native_refresh=refresh)
        elif epoch == 100:
            self.capture(STAGES[10], native_refresh=refresh)
        self.refresh_count += 1

    def capture(self, stage, **extra):
        _require(stage not in self.snapshots, "duplicate snapshot: " + stage)
        _require(self.model is not None and self.optimizers is not None,
                 "model/optimizer not bound")
        _require(self.full_views is not None and self.generator is not None,
                 "features/generator not bound")
        self.snapshots[stage] = stage_snapshot(
            stage, self.model, self.optimizers, self.generator,
            self.full_views, **extra
        )

    def finish(self):
        self.capture(STAGES[14])
        _require(tuple(self.snapshots) == STAGES, "incomplete stage sequence")
        return {
            "path": self.path_name,
            "snapshots": [self.snapshots[stage] for stage in STAGES],
            "selected_first_batch_ids": self.batch_orders,
            "kmeans_fit_predict_call_count": len(self.kmeans_events),
            "optimizer_identity_count": len({id(value) for value in self.optimizers}),
            "optimizer_instances_reused_across_ae_and_native": True,
        }


@contextmanager
def _instrument_path(module, controller, historical):
    original_build = module._build_optimizers
    original_loader = torch.utils.data.DataLoader
    original_fit_predict = KMeans.fit_predict
    refresh_name = "native_refresh" if historical else "_refresh_native_state"
    original_refresh = getattr(module, refresh_name)
    original_mse = module.F.mse_loss if historical else None
    original_objective = None if historical else module.native_objective

    def build_wrapper(model, learning_rate):
        optimizers = original_build(model, learning_rate)
        controller.bind_optimizers(model, optimizers)
        return optimizers

    def loader_wrapper(dataset, *args, **kwargs):
        batch_size = kwargs.get("batch_size", args[0] if args else 1)
        controller.observe_loader(dataset, batch_size, kwargs.get("generator"))
        return original_loader(dataset, *args, **kwargs)

    def fit_predict_wrapper(estimator, values, *args, **kwargs):
        labels = original_fit_predict(estimator, values, *args, **kwargs)
        controller.kmeans_events.append(_hash_kmeans_event(
            estimator, np.ascontiguousarray(values), np.ascontiguousarray(labels)
        ))
        return labels

    def refresh_wrapper(*args, **kwargs):
        controller.before_refresh()
        result = original_refresh(*args, **kwargs)
        controller.after_refresh(result)
        return result

    mse_counter = {"epoch": None, "count": 0}
    historical_reconstruction_tensors = {}

    def mse_wrapper(input_value, target, *args, **kwargs):
        result = original_mse(input_value, target, *args, **kwargs)
        if controller.phase == "native" and controller.native_epoch in SELECTED_NATIVE_UPDATES:
            if mse_counter["epoch"] != controller.native_epoch:
                mse_counter.update(epoch=controller.native_epoch, count=0)
            index = mse_counter["count"]
            view_id, loss_id = divmod(index, 2)
            record = controller.losses.setdefault(str(controller.native_epoch), {}).setdefault(
                str(view_id), {}
            )
            record["reconstruction" if loss_id == 0 else "clustering"] = float(
                result.detach().cpu().item()
            )
            tensor_key = (controller.native_epoch, view_id)
            if loss_id == 0:
                historical_reconstruction_tensors[tensor_key] = result.detach()
            if loss_id == 1:
                record["canonical_total"] = canonical_native_total(
                    record["reconstruction"], record["clustering"],
                    controller.native_lambda1,
                )
                record["raw_total_tensor_value"] = float(
                    (
                        historical_reconstruction_tensors.pop(tensor_key)
                        + controller.native_lambda1 * result.detach()
                    ).cpu().item()
                )
            mse_counter["count"] += 1
        return result

    def objective_wrapper(reconstruction, target, q_local, p_local, lambda1):
        result = original_objective(reconstruction, target, q_local, p_local, lambda1)
        view_id = len(controller.losses.get(str(controller.native_epoch), {}))
        controller.capture_loss(view_id, result[1], result[2], result[0])
        return result

    module._build_optimizers = build_wrapper
    torch.utils.data.DataLoader = loader_wrapper
    KMeans.fit_predict = fit_predict_wrapper
    setattr(module, refresh_name, refresh_wrapper)
    if historical:
        module.F.mse_loss = mse_wrapper
    else:
        module.native_objective = objective_wrapper
    try:
        yield
    finally:
        module._build_optimizers = original_build
        torch.utils.data.DataLoader = original_loader
        KMeans.fit_predict = original_fit_predict
        setattr(module, refresh_name, original_refresh)
        if historical:
            module.F.mse_loss = original_mse
        else:
            module.native_objective = original_objective


def _remove_diagnostic_statistics(value):
    if isinstance(value, dict):
        return {
            key: _remove_diagnostic_statistics(item)
            for key, item in value.items()
            if not key.startswith("_") and key not in ("python_random_hash", "raw_total_tensor_value") and not key.endswith("_statistics")
        }
    if isinstance(value, list):
        return [_remove_diagnostic_statistics(item) for item in value]
    return value


def _flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key in sorted(value):
            result.update(_flatten(value[key], prefix + ("." if prefix else "") + key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(_flatten(item, "%s[%d]" % (prefix, index)))
    else:
        result[prefix] = value
    return result


def _stage_map(trace):
    return {item["stage"]: item for item in trace["snapshots"]}


def _scientific_component(path):
    if "first_batch_ids" in path:
        return "dataloader_first_batch"
    if "dataloader_generator_state_hash" in path:
        return "dataloader_generator"
    if path.startswith("optimizer."):
        return "optimizer_state"
    if path.startswith("model.") and "cluster_centers" in path:
        return "cluster_centers"
    if path.startswith("model."):
        return "model_weights"
    if path.startswith("representation.") and "latent" in path:
        return "latent"
    if path.startswith("representation.") and "q_local" in path:
        return "q_local"
    if "kmeans_initialization" in path and "labels_hash" in path:
        return "kmeans_labels"
    if "kmeans_initialization" in path and "centers_hash" in path:
        return "cluster_centers"
    if path.startswith("native_refresh."):
        return "native_refresh"
    if path.startswith("native_update."):
        return "native_update"
    if path.startswith("rng."):
        return "rng_state"
    return path.split(".", 1)[0].split("[", 1)[0]


def classify_first_divergence(historical_trace, current_trace):
    """Return the first exact mismatch and one of the frozen decisions."""
    historical = _stage_map(historical_trace)
    current = _stage_map(current_trace)
    last_exact = None
    first_stage = None
    components = []
    paths = []
    for stage in STAGES:
        left = _flatten(_remove_diagnostic_statistics(historical[stage]))
        right = _flatten(_remove_diagnostic_statistics(current[stage]))
        paths = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        if paths:
            first_stage = stage
            components = sorted({_scientific_component(key) for key in paths})
            break
        last_exact = stage
    if first_stage is None:
        decision = "EXACT_INITIALIZATION_PARITY"
    elif first_stage == STAGES[0] and any(key.startswith("model.") for key in paths):
        decision = "MODEL_CONSTRUCTION_DIVERGENCE"
    elif any("first_batch_ids" in key or "dataloader_generator_state_hash" in key for key in paths):
        decision = "DATALOADER_RNG_TRAJECTORY_DIVERGENCE"
    elif first_stage in STAGES[1:5]:
        decision = "AE_OPTIMIZATION_DIVERGENCE"
    elif first_stage == STAGES[5]:
        decision = "KMEANS_INITIALIZATION_DIVERGENCE"
    elif first_stage in (STAGES[6], STAGES[10]):
        decision = "NATIVE_REFRESH_DIVERGENCE"
    elif any(key.startswith("native_refresh.") for key in paths):
        decision = "NATIVE_REFRESH_DIVERGENCE"
    elif first_stage in STAGES[7:14]:
        decision = "NATIVE_UPDATE_ORCHESTRATION_DIVERGENCE"
    else:
        decision = "NUMERICAL_NONDETERMINISM"

    # Continuity is a narrow override requiring the full preregistered pattern.
    historical_stages = _stage_map(historical_trace)
    current_stages = _stage_map(current_trace)
    if first_stage in STAGES[1:5] and paths and all(key.startswith("optimizer.") for key in paths):
        ae_models_exact = (
            historical_stages[STAGES[4]]["model"]
            == current_stages[STAGES[4]]["model"]
        )
        ae_optimizers_differ = (
            historical_stages[STAGES[4]]["optimizer"]
            != current_stages[STAGES[4]]["optimizer"]
        )
        native_zero_models_differ = (
            historical_stages[STAGES[7]]["model"]
            != current_stages[STAGES[7]]["model"]
        )
        if ae_models_exact and ae_optimizers_differ and native_zero_models_differ:
            decision = "OPTIMIZER_STATE_CONTINUITY_DIVERGENCE"
    _require(decision in ALLOWED_DECISIONS, "unregistered P0-A4 decision")
    return {
        "decision": decision,
        "last_exact_stage": last_exact,
        "first_divergent_stage": first_stage,
        "first_divergent_components": components,
        "first_divergent_paths": paths,
    }


def _extract_representation_statistics(trace, stage_names):
    by_stage = _stage_map(trace)
    return {
        stage: copy.deepcopy(by_stage[stage].get("_statistics", {}))
        for stage in stage_names if stage is not None
    }


def _collect_statistics(value, prefix=""):
    records = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = prefix + ("." if prefix else "") + key
            if key.startswith("_") or key.endswith("_statistics"):
                records[path] = copy.deepcopy(item)
            else:
                records.update(_collect_statistics(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            records.update(_collect_statistics(item, "%s[%d]" % (prefix, index)))
    return records


def _extract_adjacent_statistics(trace, stage_names):
    by_stage = _stage_map(trace)
    return {
        stage: _collect_statistics(by_stage[stage])
        for stage in stage_names if stage is not None
    }


def _strip_all_statistics(trace):
    result = copy.deepcopy(trace)
    for snapshot in result["snapshots"]:
        snapshot.pop("_statistics", None)
        for event_key in ("kmeans_initialization",):
            for event in snapshot.get(event_key, []):
                event.pop("input_statistics", None)
                event.pop("centers_statistics", None)
        refresh = snapshot.get("native_refresh")
        if refresh:
            for event in refresh.get("kmeans_passes", []):
                event.pop("input_statistics", None)
                event.pop("centers_statistics", None)
    return result


def validate_frozen_training_seeds(legacy):
    """Fail closed unless current and historical protocols both freeze seed20."""
    current_seed = int(current_initializer.protocol.TRAINING_SEED)
    historical_seed = int(
        legacy.MSRC_RUNTIME_SPEC.expected_config["training"]["seed"]
    )
    _require(current_seed == 20, "current P0-A2 training seed is not frozen at 20")
    _require(historical_seed == 20, "historical training seed is not frozen at 20")
    _require(current_seed == historical_seed,
             "historical/current frozen training seeds differ")
    return current_seed, historical_seed


def run_stagewise(output_dir, device="cuda:0"):
    """Run the explicitly authorized full diagnostic replay."""
    output = Path(output_dir)
    _require(str(device) == "cuda:0", "P0-A4 device is frozen at cuda:0")
    _require(not output.exists(), "refusing to overwrite P0-A4 output")
    release_verification = current_initializer.verify_initialization()

    legacy = load_frozen_legacy_adapter()
    current_seed, historical_seed = validate_frozen_training_seeds(legacy)
    historical_inputs = legacy.validate_materialized_inputs(
        HISTORICAL_INPUT_DIR, legacy.MSRC_RUNTIME_SPEC
    )
    historical_controller = TraceController(
        "frozen_historical",
        legacy.MSRC_RUNTIME_SPEC.expected_config["training"]["lambda1"],
    )
    with _instrument_path(legacy, historical_controller, historical=True):
        legacy.prepare_native_backbone(
            historical_inputs["views"], historical_inputs["contract"],
            torch.device(device), current_seed,
            legacy.MSRC_RUNTIME_SPEC,
        )
    historical_trace = historical_controller.finish()

    feature_record = current_initializer.load_current_condition_features(
        current_initializer.protocol.P0_A1_INPUT_DIR
    )
    frozen = current_initializer.protocol.FrozenInitializationConfig()
    current_controller = TraceController(
        "current_p0_a2", frozen.native_lambda1
    )
    with _instrument_path(current_initializer, current_controller, historical=False):
        current_initializer._run_historical_native_initialization(
            feature_record["views"], device, frozen
        )
    current_trace = current_controller.finish()

    first = classify_first_divergence(historical_trace, current_trace)
    adjacent = tuple(dict.fromkeys((
        first["last_exact_stage"], first["first_divergent_stage"]
    )))
    report = {
        "schema": "paper-msrc-p0-a4-stagewise-parity-v1",
        "diagnostic_only": True,
        "dataset": "MSRC",
        "training_seed": current_seed,
        "historical_training_seed": historical_seed,
        "device": device,
        "release_code_modified": False,
        "release_core_source_sha256": release_verification["manifest"]["release_core_source_sha256"],
        "static_audit": static_audit_table(),
        "first_divergence": first,
        "adjacent_stage_statistics": {
            "historical": _extract_adjacent_statistics(historical_trace, adjacent),
            "current": _extract_adjacent_statistics(current_trace, adjacent),
        },
        "historical": _strip_all_statistics(historical_trace),
        "current": _strip_all_statistics(current_trace),
    }
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "msrc_native_stagewise_parity.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    args = parser.parse_args(argv)
    report = run_stagewise(args.output_dir, args.device)
    print(json.dumps(report["first_divergence"], sort_keys=True))


if __name__ == "__main__":
    main()
