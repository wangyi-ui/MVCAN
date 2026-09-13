"""Descriptive multi-seed summary for C4-A0; no scientific gate is defined."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import (
    c4_a0_utility_conditioned_semantic_memory_protocol as c4,
)
from experiments.cyclic_utility import (
    run_c4_a0_utility_conditioned_semantic_memory as c4_run,
)


METRICS = ("Balanced_ACC", "ACC", "NMI", "ARI")


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def summarize(input_root=None, output_path=None):
    root = c4_run.DEFAULT_OUTPUT_ROOT if input_root is None else Path(input_root)
    records = {}
    provenance = {}
    for seed in c4.SEEDS:
        path = root / ("c4_a0_utility_conditioned_semantic_memory_seed" + str(seed)) / "c4_metrics.json"
        _require(path.is_file(), "missing C4-A0 metric record: " + str(path))
        with open(path, "r", encoding="utf-8") as input_file:
            record = json.load(input_file)
        _require(
            record.get("stage") == c4.STAGE
            and record.get("seed") == seed
            and record.get("primary_metric") == "Balanced_ACC"
            and record.get("final_PASS_FAIL_decision_implemented") is False
            and set(record.get("metrics", {})) == set(c4.MEMORY_ARMS),
            "C4-A0 metric record boundary mismatch",
        )
        for arm in c4.MEMORY_ARMS:
            _require(set(record["metrics"][arm]) == set(METRICS),
                     "C4-A0 metric schema mismatch")
        records[str(seed)] = record["metrics"]
        provenance[str(seed)] = {
            "path": str(path), "file_sha256": c4_run.file_sha256(path)
        }
    means = {
        arm: {
            metric: float(np.mean([
                records[str(seed)][arm][metric] for seed in c4.SEEDS
            ]))
            for metric in METRICS
        }
        for arm in c4.MEMORY_ARMS
    }
    result = {
        "stage": c4.STAGE,
        "seeds": list(c4.SEEDS),
        "arms": list(c4.MEMORY_ARMS),
        "metrics_by_seed": records,
        "mean_metrics": means,
        "input_provenance": provenance,
        "descriptive_summary_only": True,
        "final_PASS_FAIL_decision_implemented": False,
    }
    destination = (
        root / "c4_a0_utility_conditioned_semantic_memory_multiseed_summary.json"
        if output_path is None else Path(output_path)
    )
    _require(not destination.exists(), "refusing to overwrite C4-A0 summary")
    destination.parent.mkdir(parents=True, exist_ok=True)
    c4_run._write_json_durable(destination, result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summarize(args.input_root, args.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
