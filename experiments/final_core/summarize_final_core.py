"""Summarize three post-seal F0-A0 exact-replay metric records."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.final_core import evaluate_final_core as evaluator
from experiments.final_core import final_core_protocol as f0


DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "outputs/final_core/f0_a0_multiseed_summary.json"
)


def _require(condition, message="F0_A0_SUMMARY_FAIL_CLOSED"):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def summarize(metric_paths, output_path=DEFAULT_OUTPUT):
    _require(len(metric_paths) == len(f0.SEEDS), "F0_A0_SEED_SET_FAIL_CLOSED")
    records = {}
    for path in metric_paths:
        metric_path = _resolve(path)
        _require(metric_path.is_file(), "F0_A0_METRIC_FILE_MISSING")
        with open(metric_path, "r", encoding="utf-8") as input_file:
            record = json.load(input_file)
        seed = f0.validate_seed(record.get("seed"))
        _require(
            seed not in records
            and record.get("stage") == f0.STAGE
            and record.get("pre_gt_seal_valid") is True
            and record.get("GT_loaded_only_after_pre_gt_seal") is True
            and record.get("final_predictions_equal") is True
            and record.get("ACC_exact_equal") is True
            and record.get("NMI_exact_equal") is True
            and record.get("ARI_exact_equal") is True,
            "F0_A0_METRIC_RECORD_FAIL_CLOSED",
        )
        f0.validate_metric_parity(record.get("metrics"), f0.frozen_metrics(seed))
        records[seed] = record
    _require(set(records) == set(f0.SEEDS), "F0_A0_SEED_SET_FAIL_CLOSED")
    means = {
        metric: float(np.mean([
            records[seed]["metrics"][metric] for seed in f0.SEEDS
        ]))
        for metric in ("ACC", "NMI", "ARI")
    }
    summary = {
        "stage": f0.STAGE,
        "seeds": list(f0.SEEDS),
        "scientific_lineage": list(f0.SCIENTIFIC_LINEAGE),
        "all_seed_final_predictions_exact_equal": True,
        "all_seed_metrics_exact_equal": True,
        "all_seed_model_hash_gate_supported": True,
        "all_seed_model_hash_exact_equal": True,
        "mean_metrics": means,
        "per_seed": {
            str(seed): records[seed]["metrics"] for seed in f0.SEEDS
        },
        "decision": "F0_A0_FINAL_CORE_EXACT_REPLAY_PASS",
    }
    target = _resolve(output_path)
    _require(not target.exists(), "refusing to overwrite F0-A0 summary")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize sealed F0-A0 seed20/30/50 exact replay metrics"
    )
    parser.add_argument("--metrics-path", action="append", default=None)
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = args.metrics_path
    if paths is None:
        paths = [evaluator.default_paths(seed)["metrics"] for seed in f0.SEEDS]
    result = summarize(paths, args.output_path)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
