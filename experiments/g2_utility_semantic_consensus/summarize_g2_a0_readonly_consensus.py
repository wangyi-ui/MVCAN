"""Summarize G2-A0 metrics and preregistered read-only comparisons."""

import argparse
import csv
import json
import sys
from pathlib import Path


LABELED_METRICS = ("ACC", "NMI", "ARI")
UNLABELED_METRICS = (
    "mean_entropy",
    "mean_top1_top2_margin",
    "mean_max_confidence",
)
COMPARISONS = (
    ("primary", "U_TOP3", "ALL"),
    ("primary", "U_TOP3", "SHUFFLED_U"),
    ("auxiliary", "ORACLE", "ALL"),
    ("auxiliary", "ORACLE", "SHUFFLED_U"),
)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _delta_name(metric):
    names = {
        "mean_entropy": "delta_entropy",
        "mean_top1_top2_margin": "delta_top1_top2_margin",
        "mean_max_confidence": "delta_max_confidence",
    }
    return names.get(metric, "delta_" + metric)


def summarize_outputs(output_dir):
    root = Path(output_dir)
    source = _read_json(root / "results.json")
    arm_results = source["results"]
    pairwise = {}
    rows = []
    for comparison_tier, numerator, denominator in COMPARISONS:
        name = numerator + "-" + denominator
        values = {}
        for metric in LABELED_METRICS + UNLABELED_METRICS:
            delta_name = _delta_name(metric)
            delta = float(arm_results[numerator][metric]) - float(
                arm_results[denominator][metric]
            )
            values[delta_name] = delta
            rows.append({
                "comparison_tier": comparison_tier,
                "comparison": name,
                "metric": delta_name,
                "delta": delta,
            })
        pairwise[name] = {
            "comparison_tier": comparison_tier,
            **values,
        }

    summary = {
        "stage": "G2-A0",
        "mode": source["mode"],
        "evaluation_count": source["evaluation_count"],
        "full_reference_count": source["full_reference_count"],
        "primary_comparisons": {
            numerator + "-" + denominator: pairwise[
                numerator + "-" + denominator
            ]
            for tier, numerator, denominator in COMPARISONS
            if tier == "primary"
        },
        "auxiliary_comparisons": {
            numerator + "-" + denominator: pairwise[
                numerator + "-" + denominator
            ]
            for tier, numerator, denominator in COMPARISONS
            if tier == "auxiliary"
        },
        "native_mvcan_global_descriptive_baseline": arm_results[
            "NATIVE_MVCAN_GLOBAL"
        ],
        "all_arm_results": arm_results,
        "ENGINEERING_PASS": bool(source["ENGINEERING_PASS"]),
        "SCIENTIFIC_VERDICT": source["SCIENTIFIC_VERDICT"],
        "scientific_pass_threshold_defined": False,
        "paper_claim_made": False,
    }
    _write_json(root / "summary.json", summary)
    with open(
        root / "pairwise_deltas.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=("comparison_tier", "comparison", "metric", "delta"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = summarize_outputs(args.output_dir)
    print("ENGINEERING_PASS=" + str(summary["ENGINEERING_PASS"]))
    print("SCIENTIFIC_VERDICT=" + summary["SCIENTIFIC_VERDICT"])
    print(json.dumps(summary["primary_comparisons"], sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
