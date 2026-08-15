"""Summarize B3-A2.2 Native-z reliability replication and controls."""

import argparse
import json
import os
import sys

from irv.b3_reliability_replication import aggregate_multiseed
from irv.b3_reliability_replication import aggregate_severity
from irv.b3_reliability_replication import all_finite_nested


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    _require(isinstance(value, dict), path + " must contain a JSON object")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-json", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    evaluation = _load_json(args.evaluation_json)
    _require(evaluation.get("stage") == "B3-A2.2", "stage must be B3-A2.2")
    _require(
        evaluation.get("primary_score") == "oof_consensus_cosine",
        "primary reliability score mismatch",
    )
    _require(
        evaluation["integrity"]["B3_A22_ENGINEERING_PASS"] is True,
        "evaluation engineering gate failed",
    )
    records = evaluation["condition_results"]
    _require(len(records) == 5, "exactly five unique conditions are required")

    multiseed_records = [
        record for record in records if float(record["snr_db"]) == 2.5
    ]
    severity_records = [
        record for record in records if int(record["model_seed"]) == 20
    ]
    multiseed = aggregate_multiseed(multiseed_records)
    severity = aggregate_severity(severity_records)
    controls = evaluation["controls"]
    per_view = evaluation["per_view"]

    engineering_pass = evaluation["integrity"]["B3_A22_ENGINEERING_PASS"]
    multiseed_pass = multiseed["multiseed_replication_pass"]
    severity_pass = severity["severity_robustness_pass"]
    specificity_pass = controls["corruption_label_permutation"][
        "specificity_pass"
    ]
    crossview_pass = controls["crossview_correspondence_shuffle"][
        "crossview_dependence_pass"
    ]
    view_pass = per_view["view_robustness_pass"]
    strong_candidate = bool(
        engineering_pass
        and multiseed_pass
        and specificity_pass
        and crossview_pass
        and view_pass
    )
    gates = {
        "B3_A22_ENGINEERING_PASS": bool(engineering_pass),
        "B3_A22_MULTISEED_REPLICATION_PASS": bool(multiseed_pass),
        "B3_A22_SEVERITY_ROBUSTNESS_PASS": bool(severity_pass),
        "B3_A22_CORRUPTION_SPECIFICITY_PASS": bool(specificity_pass),
        "B3_A22_CROSSVIEW_DEPENDENCE_PASS": bool(crossview_pass),
        "B3_A22_VIEW_ROBUSTNESS_PASS": bool(view_pass),
        "B3_A22_RELIABILITY_EVIDENCE_STRONG_CANDIDATE": strong_candidate,
    }

    summary = {
        "stage": "B3-A2.2",
        "scope": "offline_native_z_reliability_replication_and_specificity",
        "primary_score": "oof_consensus_cosine",
        "multiseed": multiseed,
        "severity": severity,
        "per_view": per_view,
        "controls": {
            "corruption_label_permutation": controls[
                "corruption_label_permutation"
            ],
            "crossview_correspondence_shuffle": controls[
                "crossview_correspondence_shuffle"
            ],
            "target_only": controls["target_only"],
        },
        "condition_integrity": evaluation["condition_integrity"],
        "engineering_integrity": evaluation["integrity"],
        "gates": gates,
        "B4_READY_DECLARED": False,
        "U_I_V_DEFINED": False,
        "scientific_scope_note": (
            "Reliability evidence is evaluated only; B4 readiness remains a user decision."
        ),
    }
    _require(all_finite_nested(summary), "summary contains non-finite values")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        "b3_a22_reliability_summary.json",
    )
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")

    print("Multiseed AUC aggregate: " + str(multiseed["auc_across_seeds"]))
    print("Multiseed gap aggregate: " + str(multiseed["gap_across_seeds"]))
    print("Severity individual pass: " + str(severity["individual_absolute_pass"]))
    for gate_name, gate_value in gates.items():
        print(gate_name + "=" + str(gate_value).lower())
    print("B4_READY_DECLARED=false")
    print("B3-A2.2 summary: " + output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
