"""Summarize the two completed P0-A3 post-seal semantic diagnostics."""

import argparse
import json
from pathlib import Path


def _read(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema") != "paper-p0-a3-relation-utility-quality-v1":
        raise RuntimeError("diagnostic schema mismatch: " + str(path))
    if value.get("permutation_control", {}).get("permutations") != 1000:
        raise RuntimeError("diagnostic permutation count mismatch: " + str(path))
    if value.get("pre_gt_validation", {}).get(
        "completed_before_full_gt_load"
    ) is not True:
        raise RuntimeError("GT firewall evidence missing: " + str(path))
    return value


def _summary(value):
    validity = value["utility_action_validity"]
    return {
        "relation_class_accuracy": value["relation_class_accuracy"],
        "Q_uniform": value["Q_uniform"],
        "Q_U": value["Q_U"],
        "utility_lift": value["utility_semantic_lift"],
        "utility_roc_auc": validity["roc_auc"],
        "utility_average_precision": validity["average_precision"],
        "utility_spearman": validity["spearman_U_vs_anchor_balanced_R"],
        "high_vs_low_U_semantic_gap": value["stratification"][
            "high_vs_low_U_semantic_gap"
        ],
    }


def summarize(msrc_path, caltech_path, output_path):
    msrc, caltech = _read(msrc_path), _read(caltech_path)
    if msrc.get("dataset") != "MSRC-v1" or caltech.get("dataset") != "Caltech-6V":
        raise RuntimeError("cross-dataset input identity mismatch")
    record = {
        "schema": "paper-p0-a3-cross-dataset-relation-utility-summary-v1",
        "MSRC-v1": _summary(msrc),
        "Caltech-6V": _summary(caltech),
        "interpretation_policy": (
            "descriptive only; no selection, pruning, thresholding, tuning, "
            "or scientific-method modification"
        ),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msrc", required=True)
    parser.add_argument("--caltech", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(summarize(args.msrc, args.caltech, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
