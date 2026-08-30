"""Summarize saved B7-A1 metrics and preregistered pairwise deltas."""

import argparse
import csv
import json
import sys
from pathlib import Path


DELTA_METRICS = (
    "nearest_prototype_ACC",
    "margin_mean",
    "positive_margin_rate",
    "semantic_gap",
)
COMPARISONS = (
    ("primary_coverage_matched", "ORACLE", "ORACLE_CLASS_SHUFFLED"),
    ("primary_coverage_matched", "U_TOP3", "U_CLASS_SHUFFLED"),
    ("secondary_global_shuffle", "ORACLE", "SHUFFLED_U"),
    ("secondary_global_shuffle", "U_TOP3", "SHUFFLED_U"),
)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def summarize_outputs(output_dir):
    root = Path(output_dir)
    source = _read_json(root / "results.json")
    rows = []
    deltas = {}
    for representation in source["representations"]:
        policies = source["results"][representation]
        deltas[representation] = {}
        for comparison_tier, numerator, denominator in COMPARISONS:
            comparison_name = numerator + "-" + denominator
            comparison = {}
            for metric in DELTA_METRICS:
                value = float(policies[numerator][metric]) - float(
                    policies[denominator][metric]
                )
                delta_name = "delta_" + (
                    "ACC" if metric == "nearest_prototype_ACC" else metric
                )
                comparison[delta_name] = value
                rows.append({
                    "comparison_tier": comparison_tier,
                    "representation": representation,
                    "comparison": comparison_name,
                    "metric": delta_name,
                    "delta": value,
                })
            deltas[representation][comparison_name] = comparison
    summary = {
        "stage": "B7-A1",
        "mode": source["mode"],
        "query_count": source["query_count"],
        "primary_representations": [
            value for value in ("z", "q") if value in source["representations"]
        ],
        "auxiliary_representations": [
            value for value in ("h",) if value in source["representations"]
        ],
        "pairwise_deltas": deltas,
        "primary_coverage_matched_comparisons": {
            representation: {
                numerator + "-" + denominator: deltas[representation][
                    numerator + "-" + denominator
                ]
                for _, numerator, denominator in COMPARISONS[:2]
            }
            for representation in source["representations"]
        },
        "secondary_global_shuffle_comparisons": {
            representation: {
                numerator + "-" + denominator: deltas[representation][
                    numerator + "-" + denominator
                ]
                for _, numerator, denominator in COMPARISONS[2:]
            }
            for representation in source["representations"]
        },
        "descriptive_controls": {
            representation: {
                policy: source["results"][representation][policy]
                for policy in ("ALL", "SHUFFLED_LABEL")
            }
            for representation in source["representations"]
        },
        "ENGINEERING_PASS": bool(source["ENGINEERING_PASS"]),
        "scientific_pass_threshold_defined": False,
        "scientific_interpretation_allowed": bool(
            source["scientific_interpretation_allowed"]
        ),
    }
    _write_json(root / "summary.json", summary)
    with open(root / "pairwise_deltas.csv", "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "representation",
                "comparison_tier",
                "comparison",
                "metric",
                "delta",
            ),
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
    for representation, comparisons in summary["pairwise_deltas"].items():
        print(representation + ": " + json.dumps(comparisons, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
