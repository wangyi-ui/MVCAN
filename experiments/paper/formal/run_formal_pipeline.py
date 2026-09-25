"""Resumable orchestration over already-frozen formal stages."""

import argparse
import hashlib
import json
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from release_core.runtime import SealedPredictionPaths
from . import p1_a1_action_materialization as actions
from . import p1_a1_native_preparation as preparation
from . import p1_a1_runner as runner
from . import p1_a3_materialize as materialize
from . import p1_a4_postseal_evaluation as postseal


ROOT = Path("outputs/paper/formal")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _slug(dataset):
    return dataset.lower().replace("-", "").replace("_", "")


def _elapsed(seconds):
    seconds = int(seconds)
    return "%02d:%02d:%02d" % (seconds // 3600, (seconds // 60) % 60, seconds % 60)


def _paths(dataset, seed, arm=None):
    slug = _slug(dataset)
    inputs = ROOT / "inputs" / slug
    result = {"inputs": inputs, "init": ROOT / "preparation" / slug / ("seed" + str(seed)), "action": ROOT / "actions" / slug / ("seed" + str(seed)) / "true_action_state"}
    if arm is not None:
        result["arm"] = ROOT / "main" / slug / ("seed" + str(seed)) / arm
    return result


def _verify_inputs(root, dataset):
    root = Path(root)
    names = ("features.npz", "feature_audit.json", "sparse_split.npz", "sparse_split_audit.json", "weak_quality_audit.json", "materialization_seal.json")
    if not root.is_dir() or not all((root / name).is_file() for name in names):
        raise RuntimeError("FORMAL_INPUT_NOT_MATERIALIZED")
    seal = json.loads((root / "materialization_seal.json").read_text(encoding="utf-8"))
    if not (seal.get("seal_valid") is True and seal.get("dataset") == dataset and _sha(root / "features.npz") == seal.get("feature_sha256") and _sha(root / "sparse_split.npz") == seal.get("split_sha256") and _sha(root / "feature_audit.json") == seal.get("feature_audit_sha256") and _sha(root / "sparse_split_audit.json") == seal.get("split_audit_sha256")):
        raise RuntimeError("FORMAL_INPUT_PROVENANCE_MISMATCH")
    return root


def _sealed_arm(dataset, seed, arm):
    root = _paths(dataset, seed, arm)["arm"]
    return SealedPredictionPaths(root / "pre_gt_bundle.npz", root / "pre_gt_audit.json", root / "pre_gt_seal.json")


def _verify_arm(dataset, seed, arm):
    return postseal._validate(arm, _sealed_arm(dataset, seed, arm))


def _run(label, func):
    started, done = time.monotonic(), threading.Event()
    def heartbeat():
        while not done.wait(60):
            print("%s RUNNING | elapsed=%s" % (label, _elapsed(time.monotonic() - started)), flush=True)
    thread = threading.Thread(target=heartbeat, daemon=True)
    print("%s START" % label, flush=True); thread.start()
    try:
        value = func()
    except BaseException:
        print("%s FAIL | elapsed=%s" % (label, _elapsed(time.monotonic() - started)), flush=True)
        raise
    finally:
        done.set(); thread.join(timeout=0.1)
    print("%s PASS | elapsed=%s" % (label, _elapsed(time.monotonic() - started)), flush=True)
    return value


def _existing_or_run(label, target, verify, build, resume):
    target = Path(target)
    if target.exists():
        if not resume:
            raise RuntimeError("FORMAL_EXISTING_ARTIFACT_REQUIRES_RESUME")
        _run(label + "/VERIFY", lambda: verify(target))
        print("%s SKIP" % label, flush=True)
        return
    _run(label, build)
    _run(label + "/VERIFY", lambda: verify(target))


def _summary(dataset, seeds, arms):
    rows = []
    for seed in seeds:
        values = {}
        for arm in arms:
            path = _paths(dataset, seed, arm)["arm"] / "postseal_metrics.json"
            values[arm] = json.loads(path.read_text(encoding="utf-8"))["metrics"]
        delta = {name: values["OURS_TRUE_U"][name] - values["BASE"][name] for name in ("acc", "nmi", "ari")}
        rows.append({"training_seed": seed, "BASE": values["BASE"], "OURS_TRUE_U": values["OURS_TRUE_U"], "delta": delta})
    aggregate = {group: {name: {"mean": float(np.mean([row[group][name] for row in rows])), "std": float(np.std([row[group][name] for row in rows]))} for name in ("acc", "nmi", "ari")} for group in ("BASE", "OURS_TRUE_U", "delta")}
    return {"dataset": dataset, "seeds": list(seeds), "rows": rows, "aggregate": aggregate}


def _write_or_verify_summary(dataset, seeds, arms, resume):
    root = ROOT / "main" / _slug(dataset)
    targets = tuple(root / name for name in ("formal_summary.json", "formal_summary.txt", "formal_source_sha256.json"))
    source_hashes = {str(path): _sha(path) for path in (Path(__file__), Path(materialize.__file__), Path(runner.__file__), Path(postseal.__file__))}
    if any(path.exists() for path in targets):
        if not resume:
            raise RuntimeError("FORMAL_SUMMARY_OUTPUT_ALREADY_EXISTS")
        if not all(path.is_file() for path in targets):
            raise RuntimeError("FORMAL_SUMMARY_RESUME_ARTIFACT_INCOMPLETE")
        saved = json.loads(targets[0].read_text(encoding="utf-8"))
        if saved != _summary(dataset, seeds, arms) or targets[1].read_text(encoding="utf-8") != json.dumps(saved, sort_keys=True, indent=2) + "\n" or json.loads(targets[2].read_text(encoding="utf-8")) != source_hashes:
            raise RuntimeError("FORMAL_SUMMARY_RESUME_MISMATCH")
        print("[SUMMARY] SKIP", flush=True)
        return saved
    result = _summary(dataset, seeds, arms)
    for suffix, content in (("formal_summary.json", json.dumps(result, sort_keys=True, indent=2) + "\n"), ("formal_summary.txt", json.dumps(result, sort_keys=True, indent=2) + "\n"), ("formal_source_sha256.json", json.dumps(source_hashes, sort_keys=True, indent=2) + "\n")):
        (root / suffix).write_text(content, encoding="utf-8")
    return result


def run_pipeline(*, dataset, training_seeds, arms, device, full_gt_path, resume=False):
    if tuple(training_seeds) != tuple(sorted(training_seeds)) or len(set(training_seeds)) != len(training_seeds):
        raise RuntimeError("FORMAL_PIPELINE_SEED_ORDER_INVALID")
    if tuple(arms) != ("BASE", "OURS_TRUE_U"):
        raise RuntimeError("FORMAL_PIPELINE_ARM_SET_INVALID")
    first = training_seeds[0]
    paths = _paths(dataset, first)
    _existing_or_run("[INPUTS]", paths["inputs"], lambda target: _verify_inputs(target, dataset), lambda: materialize.main(["--dataset", dataset, "--training-seed", str(first), "--device", device, "--stage", "inputs"]), resume)
    for seed in training_seeds:
        paths = _paths(dataset, seed)
        _existing_or_run("[seed%d][INITIALIZATION]" % seed, paths["init"], lambda target, seed=seed: preparation.verify_initialization(target, dataset=dataset, training_seed=seed), lambda seed=seed: materialize.main(["--dataset", dataset, "--training-seed", str(seed), "--device", device, "--stage", "initialization"]), resume)
        init = preparation.verify_initialization(paths["init"], dataset=dataset, training_seed=seed)
        _existing_or_run("[seed%d][ACTIONS]" % seed, paths["action"], lambda target, seed=seed, init=init: actions.verify_true_action(target, dataset=dataset, training_seed=seed, initial_model_sha256=init["initial_model_sha256"]), lambda seed=seed: materialize.main(["--dataset", dataset, "--training-seed", str(seed), "--device", device, "--stage", "actions"]), resume)
        for arm in arms:
            arm_root = _paths(dataset, seed, arm)["arm"]
            _existing_or_run("[seed%d][%s]" % (seed, arm), arm_root, lambda target, seed=seed, arm=arm: _verify_arm(dataset, seed, arm), lambda seed=seed, arm=arm: runner.run_formal(runner.FormalRun(dataset, seed, arm, device)), resume)
        for arm in arms:
            arm_root = _paths(dataset, seed, arm)["arm"]
            metric = arm_root / "postseal_metrics.json"
            _existing_or_run("[seed%d][%s][EVALUATION]" % (seed, arm), metric, lambda target: json.loads(target.read_text(encoding="utf-8")), lambda seed=seed, arm=arm: postseal.evaluate_formal_postseal(dataset=dataset, training_seed=seed, arm=arm, full_gt_path=full_gt_path), resume)
    result = _write_or_verify_summary(dataset, training_seeds, arms, resume)
    print("[SUMMARY] FINAL METRICS " + json.dumps(result["aggregate"], sort_keys=True), flush=True)
    return result



class _Tee:
    def __init__(self, console, stream):
        self.console, self.stream = console, stream
    def write(self, value):
        self.console.write(value); self.stream.write(value)
    def flush(self):
        self.console.flush(); self.stream.flush()

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("Caltech-6V", "MSRC-v1", "BDGP"))
    parser.add_argument("--training-seeds", required=True, nargs="+", type=int)
    parser.add_argument("--arms", required=True, nargs="+", choices=("BASE", "OURS_TRUE_U"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--full-gt-path", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-file")
    args = parser.parse_args(argv)
    if args.log_file is None:
        return run_pipeline(dataset=args.dataset, training_seeds=tuple(args.training_seeds), arms=tuple(args.arms), device=args.device, full_gt_path=args.full_gt_path, resume=args.resume)
    log_path = Path(args.log_file); log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        previous_stdout, previous_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(previous_stdout, stream), _Tee(previous_stderr, stream)
        try:
            return run_pipeline(dataset=args.dataset, training_seeds=tuple(args.training_seeds), arms=tuple(args.arms), device=args.device, full_gt_path=args.full_gt_path, resume=args.resume)
        except BaseException as error:
            print("[PIPELINE] FAIL | %s" % error, flush=True)
            raise
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            sys.stdout, sys.stderr = previous_stdout, previous_stderr


if __name__ == "__main__":
    main()
