"""Summarize the preregistered 1001-step B4-A1a seed20 pilot."""

import argparse
import csv
import json
import os
import sys


METRICS = (
    "ACC",
    "NMI",
    "ARI",
    "effective_rank",
    "correspondence_gap",
    "retrieval_top1",
    "final_semantic_loss",
)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _metric_values(arm_audit):
    diagnostics = arm_audit["final_semantic_diagnostics"]
    clustering = diagnostics["fused_clustering"]
    return {
        "ACC": clustering["acc"],
        "NMI": clustering["nmi"],
        "ARI": clustering["ari"],
        "effective_rank": diagnostics["effective_rank"],
        "correspondence_gap": diagnostics["correspondence_gap"],
        "retrieval_top1": diagnostics["retrieval_top1"],
        "final_semantic_loss": arm_audit["final_semantic_loss"],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    conditions = {}
    csv_rows = []
    for condition in ("clean", "snr2p5_k2"):
        condition_dir = os.path.join(args.input_dir, condition)
        metadata = _load_json(os.path.join(condition_dir, "metadata.json"))
        if metadata.get("stage") != "B4-A1a":
            raise RuntimeError("stage mismatch for " + condition)
        arms = {
            arm: _load_json(os.path.join(condition_dir, arm, "arm_audit.json"))
            for arm in ("uniform", "t_utility")
        }
        uniform = _metric_values(arms["uniform"])
        utility = _metric_values(arms["t_utility"])
        delta = {metric: utility[metric] - uniform[metric] for metric in METRICS}
        for metric in METRICS:
            csv_rows.append({
                "condition": condition,
                "metric": metric,
                "uniform": uniform[metric],
                "t_utility": utility[metric],
                "delta_utility_minus_uniform": delta[metric],
            })
        conditions[condition] = {
            "metadata": metadata,
            "uniform": uniform,
            "t_utility": utility,
            "delta_utility_minus_uniform": delta,
            "arm_audits": arms,
        }

    noisy_delta = conditions["snr2p5_k2"]["delta_utility_minus_uniform"]
    improved_count = sum(noisy_delta[metric] > 0.0 for metric in ("ACC", "NMI", "ARI"))
    engineering_pass = all(
        conditions[condition]["metadata"]["gates"]["B4_A1A_ENGINEERING_PASS"]
        for condition in conditions
    )
    full_steps_pass = all(
        conditions[condition]["metadata"]["semantic_steps"] == 1001
        for condition in conditions
    )
    noisy_no_collapse = bool(
        conditions["snr2p5_k2"]["t_utility"]["effective_rank"] > 1.5
        and conditions["snr2p5_k2"]["arm_audits"]["t_utility"][
            "final_semantic_diagnostics"
        ]["collapse"]["variance_mean"] > 1e-6
    )
    action_candidate = bool(
        engineering_pass
        and full_steps_pass
        and improved_count >= 2
        and noisy_no_collapse
    )
    summary = {
        "stage": "B4-A1a",
        "scope": "frozen_final_z_mechanism_probe",
        "model_seed": 20,
        "conditions": conditions,
        "B4_A1A_ENGINEERING_PASS": engineering_pass,
        "B4_A1A_ACTION_SIGNAL_CANDIDATE": action_candidate,
        "action_signal_diagnostics": {
            "snr2p5_acc_nmi_ari_improved_count": improved_count,
            "snr2p5_no_collapse": noisy_no_collapse,
            "both_conditions_1001_steps": full_steps_pass,
            "clean_report_only": True,
        },
        "B4_FINAL_PASS_DECLARED": False,
        "B4_A1B_STARTED": False,
        "mechanism_probe_only": True,
        "online_deployable_method": False,
    }
    summary_path = os.path.join(args.input_dir, "b4_a1a_seed20_summary.json")
    with open(summary_path, "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
    csv_path = os.path.join(args.input_dir, "b4_a1a_seed20_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "condition",
                "metric",
                "uniform",
                "t_utility",
                "delta_utility_minus_uniform",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print("B4_A1A_ENGINEERING_PASS=" + str(engineering_pass).lower())
    print("B4_A1A_ACTION_SIGNAL_CANDIDATE=" + str(action_candidate).lower())
    print("B4-A1a summary: " + summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
