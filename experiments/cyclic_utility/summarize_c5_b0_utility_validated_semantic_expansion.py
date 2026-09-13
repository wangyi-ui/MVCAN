"""Apply the pre-registered C5-B0 two-gate multi-seed ACC decision."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import c5_b0_utility_validated_semantic_expansion_protocol as c5b0
from experiments.cyclic_utility import evaluate_c5_b0_utility_validated_semantic_expansion as evaluator


def _require(condition, message="C5_B0_SUMMARY_FAIL_CLOSED"):
    if not condition:
        raise RuntimeError(message)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, record):
    with open(path, "x", encoding="utf-8") as output_file:
        json.dump(record, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def default_metric_paths():
    return {
        seed: {
            arm: evaluator.default_paths(seed, arm)["metrics"]
            for arm in c5b0.ARMS
        }
        for seed in c5b0.SEEDS
    }


def load_metric_records(paths_by_seed):
    _require(set(paths_by_seed) == set(c5b0.SEEDS), "C5_B0_SEED_SET_FAIL_CLOSED")
    records = {}
    metrics = {}
    for seed in c5b0.SEEDS:
        _require(
            set(paths_by_seed[seed]) == set(c5b0.ARMS),
            "C5_B0_ARM_SET_FAIL_CLOSED",
        )
        records[seed] = {}
        metrics[seed] = {}
        for arm in c5b0.ARMS:
            record = _read_json(paths_by_seed[seed][arm])
            values = record.get("metrics", {})
            _require(
                record.get("stage") == c5b0.STAGE
                and int(record.get("seed", -1)) == seed
                and record.get("arm") == arm
                and record.get("primary_metric") == c5b0.PRIMARY_METRIC
                and record.get("GT_loaded_only_after_pre_gt_seal") is True
                and set(values) == {"ACC", "NMI", "ARI", "Balanced_ACC"}
                and all(np.isfinite(values[name]) for name in values),
                "C5_B0_METRIC_RECORD_FAIL_CLOSED",
            )
            records[seed][arm] = record
            metrics[seed][arm] = values
    return records, metrics


def build_summary(metrics_by_seed):
    decision = c5b0.summarize_multiseed_metrics(metrics_by_seed)
    mean_metrics = {
        arm: {
            metric: float(np.mean([
                metrics_by_seed[seed][arm][metric] for seed in c5b0.SEEDS
            ]))
            for metric in ("ACC", "NMI", "ARI", "Balanced_ACC")
        }
        for arm in c5b0.ARMS
    }
    return {
        **decision,
        "seeds": list(c5b0.SEEDS),
        "arms": list(c5b0.ARMS),
        "mean_metrics": mean_metrics,
        "minimum_positive_margin_used": False,
        "post_hoc_threshold_used": False,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = default_metric_paths()
    for seed in c5b0.SEEDS:
        for arm in c5b0.ARMS:
            parser.add_argument(
                "--seed" + str(seed) + "-" + arm.lower().replace("_", "-") + "-metrics",
                dest="seed" + str(seed) + "_" + arm,
                default=str(defaults[seed][arm]),
            )
    parser.add_argument(
        "--output-path",
        default=str(
            REPOSITORY_ROOT / "outputs/cyclic_utility/"
            "c5_b0_utility_validated_semantic_expansion_multiseed_summary.json"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = {
        seed: {
            arm: Path(getattr(args, "seed" + str(seed) + "_" + arm))
            for arm in c5b0.ARMS
        }
        for seed in c5b0.SEEDS
    }
    _, metrics = load_metric_records(paths)
    summary = build_summary(metrics)
    output = Path(args.output_path)
    _require(not output.exists(), "refusing to overwrite C5-B0 summary")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
