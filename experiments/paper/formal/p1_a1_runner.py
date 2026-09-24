"""Non-overwriting, pre-GT P1-A1 formal runner orchestration.

This module intentionally has no full-GT argument, evaluator import, metric
import, scientific constant, or hyperparameter CLI override.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from release_core.runtime import RuntimeConfig, run_pre_gt

from . import p1_a0_formal_protocol as protocol
from . import p1_a1_action_materialization as actions
from . import p1_a1_base_runtime as base_runtime
from . import p1_a1_native_preparation as preparation
from . import p1_a2_execution_contract as execution


ROOT = Path("outputs/paper/formal")


@dataclass(frozen=True)
class FormalRun:
    dataset: str
    training_seed: int
    arm: str
    device: str


def _dataset(dataset):
    return next((item for item in protocol.FORMAL_DATASETS if item.name == dataset), None)


def formal_arms():
    return tuple(protocol.ARM_PROTOCOL["main"]) + tuple(protocol.ARM_PROTOCOL["ablation"])


def validate_request(run):
    if protocol.validate_formal_protocol() is not True:
        raise RuntimeError("FORMAL_PROTOCOL_UNRESOLVED")
    execution.validate_execution_contract()
    if _dataset(run.dataset) is None:
        raise RuntimeError("FORMAL_DATASET_NOT_AUTHORIZED")
    if run.training_seed not in protocol.FORMAL_TRAINING_SEEDS:
        raise RuntimeError("FORMAL_TRAINING_SEED_NOT_AUTHORIZED")
    if run.arm not in formal_arms():
        raise RuntimeError("FORMAL_ARM_NOT_AUTHORIZED")
    return run


def slug(dataset):
    return dataset.lower().replace("-", "").replace("_", "")


def execution_capability(run):
    return execution.arm_capability(run.dataset, run.arm)

def paths_for(run):
    base = ROOT / "main" / slug(run.dataset) / ("seed" + str(run.training_seed))
    input_root = ROOT / "inputs" / slug(run.dataset)
    return {
        "features": input_root / "features.npz", "feature_audit": input_root / "feature_audit.json",
        "split": input_root / "sparse_split.npz", "split_audit": input_root / "sparse_split_audit.json",
        "initialization": ROOT / "preparation" / slug(run.dataset) / ("seed" + str(run.training_seed)),
        "action": ROOT / "actions" / slug(run.dataset) / ("seed" + str(run.training_seed)) / "true_action_state",
        "output": base / run.arm,
    }


def runtime_config(run):
    item = _dataset(run.dataset)
    native, alternating, labels = (protocol.NATIVE_PREPARATION_PROTOCOL,
                                   protocol.ALTERNATING_PROTOCOL,
                                   protocol.SPARSE_LABEL_PROTOCOL)
    return RuntimeConfig(dataset=run.dataset, training_seed=run.training_seed,
                         epochs=alternating.epochs, batch_size=alternating.batch_size,
                         learning_rate=alternating.learning_rate,
                         native_lambda1=item.native_lambda1,
                         refresh_interval=alternating.refresh_interval,
                         label_seed=labels["label_seed"], labels_per_class=labels["labels_per_class"],
                         device=run.device)


def plan(run):
    validate_request(run)
    item, paths = _dataset(run.dataset), paths_for(run)
    return {"dataset": run.dataset, "training_seed": run.training_seed, "arm": run.arm,
            "device": run.device, "paths": {key: str(value) for key, value in paths.items()},
            "native_config_seed": item.native_config_seed, "kmeans_seed": run.training_seed,
            "order_seed": run.training_seed, "weak_quality_seed": protocol.WEAK_QUALITY_PROTOCOL["realization_seed"],
            "label_seed": protocol.SPARSE_LABEL_PROTOCOL["label_seed"],
            "native_schedule": protocol.NATIVE_PREPARATION_PROTOCOL,
            "alternating_schedule": protocol.ALTERNATING_PROTOCOL.__dict__,
            "gt_firewall": protocol.EVALUATION_PROTOCOL, "execution_capability": execution_capability(run)[0], "blocked_reason": execution_capability(run)[1], "native_generator_contract": execution.NATIVE_DATALOADER_GENERATOR_CONTRACT, "action_carrier_contract": execution.ACTION_CARRIER_CONTRACT, "status": "PLAN VALID"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_formal_inputs(paths):
    required = (paths["features"], paths["feature_audit"], paths["split"], paths["split_audit"])
    if not all(path.is_file() for path in required):
        raise RuntimeError("FORMAL_INPUT_NOT_MATERIALIZED")


def _failed(run, output, error):
    path = Path(str(output) + ".failed.json")
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "paper-p1-a1-failure-v1", "stage": "pre_gt",
        "exception_type": type(error).__name__, "message": str(error), "dataset": run.dataset,
        "training_seed": run.training_seed, "arm": run.arm, "full_gt_loaded": False}, indent=2) + "\n", encoding="utf-8")


def run_formal(run):
    """Run exactly one authorized pre-GT arm; existing paths are never overwritten."""
    validate_request(run)
    paths = paths_for(run)
    capability, reason = execution_capability(run)
    if capability != "AUTHORIZED":
        raise RuntimeError(reason)
    if paths["output"].exists():
        raise RuntimeError("FORMAL_OUTPUT_ALREADY_EXISTS")
    print("[P1-A1] START", flush=True)
    print("[P1-A1] protocol validated", flush=True)
    try:
        _require_formal_inputs(paths)
        print("[P1-A1] formal inputs validated", flush=True)
        initialization = preparation.verify_initialization(paths["initialization"], dataset=run.dataset, training_seed=run.training_seed)
        print("[P1-A1] initialization verified/built", flush=True)
        runtime = runtime_config(run)
        if run.arm == "BASE":
            print("[P1-A1] arm utility materialized (BASE skips Phase A)", flush=True)
            print("[P1-A1] pre-GT training started", flush=True)
            sealed = base_runtime.run_base_pre_gt(runtime, paths, initialization)
        else:
            true_action = actions.verify_true_action(paths["action"], dataset=run.dataset, training_seed=run.training_seed, initial_model_sha256=initialization["initial_model_sha256"])
            print("[P1-A1] true R2/R3 state verified/built", flush=True)
            selected = actions.select_arm_utility(true_action, run.arm, protocol.ARM_PROTOCOL["ablation"], dataset=run.dataset)
            print("[P1-A1] arm utility materialized", flush=True)
            print("[P1-A1] pre-GT training started", flush=True)
            sealed = run_pre_gt(runtime, selected["provenance"])
        print("[P1-A1] pre-GT training completed", flush=True)
        print("[P1-A1] final refresh completed", flush=True)
        print("[P1-A1] seal completed", flush=True)
        print("[P1-A1] DONE", flush=True)
        return sealed
    except BaseException as error:
        _failed(run, paths["output"], error)
        raise

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=tuple(item.name for item in protocol.FORMAL_DATASETS))
    parser.add_argument("--training-seed", required=True, type=int, choices=protocol.FORMAL_TRAINING_SEEDS)
    parser.add_argument("--arm", required=True, choices=formal_arms())
    parser.add_argument("--device", required=True)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run = FormalRun(args.dataset, args.training_seed, args.arm, args.device)
    if args.plan_only:
        print(json.dumps(plan(run), indent=2, sort_keys=True), flush=True)
        return 0
    run_formal(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
