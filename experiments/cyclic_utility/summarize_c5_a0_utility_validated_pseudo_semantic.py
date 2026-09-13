"""Summarize sealed C5-A0 seed20/30/50 evaluation records without mutation."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import c5_a0_utility_validated_pseudo_semantic_protocol as c5


def _require(condition, message="C5_A0_SUMMARY_FAIL_CLOSED"):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def default_metric_path(seed):
    active_seed = c5.validate_seed(seed)
    return REPOSITORY_ROOT / "outputs/cyclic_utility" / (
        "c5_a0_utility_validated_pseudo_semantic_seed" + str(active_seed)
    ) / "c5_metrics.json"


def default_output_path():
    return (
        REPOSITORY_ROOT / "outputs/cyclic_utility"
        / "c5_a0_utility_validated_pseudo_semantic_multiseed"
        / "c5_multiseed_summary.json"
    )


def _read_metric(path, expected_seed):
    source = _resolve(path)
    _require(source.is_file(), "C5-A0 seed metric is missing")
    with open(source, "r", encoding="utf-8") as input_file:
        record = json.load(input_file)
    _require(
        record.get("stage") == c5.STAGE
        and int(record.get("seed", -1)) == expected_seed
        and record.get("GT_loaded_only_after_pre_gt_seal") is True
        and record.get("training_performed") is False,
        "C5-A0 seed metric provenance mismatch",
    )
    auc = record.get("Action_Utility_Predictiveness", {})
    p20 = record.get("Precision@20", {})
    required = (
        auc.get("AUC_TRUE"), auc.get("AUC_PERMUTED"), auc.get("Delta_AUC"),
        p20.get("TRUE_U"), p20.get("UNIFORM"), p20.get("PERMUTED_U"),
        p20.get("Delta_TRUE_U_UNIFORM"), p20.get("Delta_TRUE_U_PERMUTED_U"),
    )
    _require(
        all(value is not None and np.isfinite(float(value)) for value in required),
        "C5-A0 seed metric is incomplete or non-finite",
    )
    return record


def build_summary(records):
    _require(set(records) == set(c5.SEEDS), "C5-A0 requires seeds 20/30/50")
    per_seed = {}
    auc_true = []
    auc_permuted = []
    delta_auc = []
    delta_uniform = []
    delta_permuted = []
    for seed in c5.SEEDS:
        record = records[seed]
        auc = record["Action_Utility_Predictiveness"]
        p20 = record["Precision@20"]
        row = {
            "AUC_TRUE": float(auc["AUC_TRUE"]),
            "AUC_PERMUTED": float(auc["AUC_PERMUTED"]),
            "Delta_AUC": float(auc["Delta_AUC"]),
            "AUC_TRUE_gt_AUC_PERMUTED": bool(
                auc["AUC_TRUE"] > auc["AUC_PERMUTED"]
            ),
            "P20_TRUE_U": float(p20["TRUE_U"]),
            "P20_UNIFORM": float(p20["UNIFORM"]),
            "P20_PERMUTED_U": float(p20["PERMUTED_U"]),
            "TRUE_U_gt_UNIFORM": bool(p20["TRUE_U"] > p20["UNIFORM"]),
            "TRUE_U_gt_PERMUTED_U": bool(p20["TRUE_U"] > p20["PERMUTED_U"]),
            "Delta_P20_TRUE_U_UNIFORM": float(p20["Delta_TRUE_U_UNIFORM"]),
            "Delta_P20_TRUE_U_PERMUTED_U": float(
                p20["Delta_TRUE_U_PERMUTED_U"]
            ),
        }
        per_seed[str(seed)] = row
        auc_true.append(row["AUC_TRUE"])
        auc_permuted.append(row["AUC_PERMUTED"])
        delta_auc.append(row["Delta_AUC"])
        delta_uniform.append(row["Delta_P20_TRUE_U_UNIFORM"])
        delta_permuted.append(row["Delta_P20_TRUE_U_PERMUTED_U"])
    return {
        "stage": c5.STAGE,
        "seeds": list(c5.SEEDS),
        "arms": list(c5.ARMS),
        "per_seed": per_seed,
        "mean_AUC_TRUE": float(np.mean(auc_true)),
        "mean_AUC_PERMUTED": float(np.mean(auc_permuted)),
        "mean_Delta_AUC": float(np.mean(delta_auc)),
        "AUC_TRUE_gt_AUC_PERMUTED_positive_seed_count": int(sum(
            row["AUC_TRUE_gt_AUC_PERMUTED"] for row in per_seed.values()
        )),
        "mean_Delta_P20_TRUE_U_UNIFORM": float(np.mean(delta_uniform)),
        "mean_Delta_P20_TRUE_U_PERMUTED_U": float(np.mean(delta_permuted)),
        "P20_TRUE_U_gt_UNIFORM_positive_seed_count": int(sum(
            row["TRUE_U_gt_UNIFORM"] for row in per_seed.values()
        )),
        "P20_TRUE_U_gt_PERMUTED_U_positive_seed_count": int(sum(
            row["TRUE_U_gt_PERMUTED_U"] for row in per_seed.values()
        )),
        "formal_decision_made": False,
        "formal_outputs_modified": False,
    }


def summarize(metric_paths, output_path):
    records = {
        seed: _read_metric(metric_paths[seed], seed) for seed in c5.SEEDS
    }
    summary = build_summary(records)
    target = _resolve(output_path)
    _require(not target.exists(), "refusing to overwrite C5-A0 summary")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "x", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed20-metrics", default=str(default_metric_path(20)))
    parser.add_argument("--seed30-metrics", default=str(default_metric_path(30)))
    parser.add_argument("--seed50-metrics", default=str(default_metric_path(50)))
    parser.add_argument("--output-path", default=str(default_output_path()))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summarize(
        {
            20: args.seed20_metrics,
            30: args.seed30_metrics,
            50: args.seed50_metrics,
        },
        args.output_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
