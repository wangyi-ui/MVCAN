"""Summarize Clean and SNR2.5 B3-A2.1 failure attribution."""

import argparse
import csv
import json
import os
import sys


def _load_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(path + " must contain a JSON object")
    return value


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _absolute_pass(reliability):
    return bool(
        reliability["auc_ci_low"] > 0.5
        and reliability["gap_ci_low"] > 0.0
    )


def _combine_null_csv(clean_json_path, noisy_json_path, output_path):
    input_paths = [
        os.path.join(os.path.dirname(path), "random_head_null.csv")
        for path in (clean_json_path, noisy_json_path)
    ]
    rows = []
    fieldnames = None
    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            _require(reader.fieldnames == fieldnames, "random null CSV schema mismatch")
            condition_rows = list(reader)
            _require(len(condition_rows) == 100, "each condition must have 100 null rows")
            rows.extend(condition_rows)
    _require(len(rows) == 200, "combined random null CSV must have 200 rows")
    with open(output_path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-json", required=True)
    parser.add_argument("--noisy-json", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    clean = _load_json(args.clean_json)
    noisy = _load_json(args.noisy_json)
    for record, condition in ((clean, "clean"), (noisy, "snr2p5_k2")):
        _require(record.get("stage") == "B3-A2.1", "stage must be B3-A2.1")
        _require(record.get("condition") == condition, "condition mismatch")
        _require(record.get("dataset") == "MSRC-v1", "dataset mismatch")
        _require(record.get("model_seed") == 20, "model seed must be 20")
        _require(
            record["integrity"]["B3_A21_ENGINEERING_PASS"] is True,
            "condition engineering gate failed",
        )

    noisy_bootstrap = noisy["bootstrap"]
    reliability_z = noisy_bootstrap["arms"]["z_native"]
    reliability_a0 = noisy_bootstrap["arms"]["a0_random"]
    reliability_a1 = noisy_bootstrap["arms"]["a1_uniform"]
    a1_minus_a0 = noisy_bootstrap["paired"]["a1_uniform_minus_a0_random"]
    a1_minus_z = noisy_bootstrap["paired"]["a1_uniform_minus_z_native"]
    gates = noisy["gates"]

    z_absolute_pass = _absolute_pass(reliability_z)
    a0_absolute_pass = _absolute_pass(reliability_a0)
    a1_absolute_pass = _absolute_pass(reliability_a1)
    a1_alignment_exists = bool(
        noisy["a1_uniform"]["correspondence_gap"] > 0.0
        and noisy["a1_uniform"]["retrieval_top1"] > (1.0 / noisy["sample_num"])
    )
    a1_auc_ci_covers_half = bool(
        reliability_a1["auc_ci_low"] <= 0.5 <= reliability_a1["auc_ci_high"]
    )
    all_predictability_fail = bool(
        not z_absolute_pass and not a0_absolute_pass and not a1_absolute_pass
    )

    case_a = bool(
        a1_absolute_pass
        and gates["B3_A21_PREDICTABILITY_GAIN_OVER_A0_PASS"]
    )
    case_b = bool(
        z_absolute_pass
        and not gates["B3_A21_PREDICTABILITY_GAIN_OVER_A0_PASS"]
        and not gates["B3_A21_PREDICTABILITY_GAIN_OVER_Z_PASS"]
    )
    case_c = bool(a1_alignment_exists and a1_auc_ci_covers_half)
    case_d = all_predictability_fail
    if case_a:
        selected_case = "CASE_A"
    elif case_b:
        selected_case = "CASE_B"
    elif case_d:
        selected_case = "CASE_D"
    elif case_c:
        selected_case = "CASE_C"
    else:
        selected_case = "NO_PREDEFINED_CASE"

    decision_diagnostics = {
        "selected_case": selected_case,
        "case_precedence": ["CASE_A", "CASE_B", "CASE_D", "CASE_C"],
        "CASE_A": {
            "criteria_met": case_a,
            "explanation": (
                "Trained shared semantic contains new reliability evidence. "
                "This is the case most supportive of later U_i^v design."
            ),
        },
        "CASE_B": {
            "criteria_met": case_b,
            "explanation": (
                "Reliability evidence exists in the backbone, but the current "
                "semantic objective does not add value. Do not enter B4 before "
                "deciding whether reliability should govern z-to-s admission."
            ),
        },
        "CASE_C": {
            "criteria_met": case_c,
            "explanation": (
                "Semantic space aligns instances but does not expose reliable "
                "sample-view quality; the current B3-A1 objective is insufficient."
            ),
        },
        "CASE_D": {
            "criteria_met": case_d,
            "explanation": (
                "Z/A0/A1 all fail the absolute predictability gate; the current "
                "predictability construction cannot diagnose B2 corruption. "
                "Do not enter B4."
            ),
        },
        "supporting_flags": {
            "z_absolute_pass": z_absolute_pass,
            "a0_absolute_pass": a0_absolute_pass,
            "a1_absolute_pass": a1_absolute_pass,
            "a1_alignment_exists": a1_alignment_exists,
            "a1_auc_ci_covers_half": a1_auc_ci_covers_half,
            "all_predictability_fail": all_predictability_fail,
        },
    }

    seed1020_acc_percentile = clean["a0_seed1020_vs_random_null"][
        "fused_acc_percentile"
    ]
    seed1020_extreme_high_tail = bool(seed1020_acc_percentile >= 95.0)
    random_head_interpretation = {
        "clean_seed1020_a0_fused_acc": clean["a0_seed1020"]["fused_acc"],
        "clean_seed1020_a0_fused_acc_percentile": seed1020_acc_percentile,
        "clean_seed1020_a0_fused_acc_z_score": clean[
            "a0_seed1020_vs_random_null"
        ]["fused_acc_z_score"],
        "clean_a1_fused_acc": clean["a1_uniform"]["fused_acc"],
        "clean_a1_fused_acc_percentile": clean["a1_vs_random_null"][
            "fused_acc_percentile"
        ],
        "clean_a1_fused_acc_z_score": clean["a1_vs_random_null"][
            "fused_acc_z_score"
        ],
        "seed1020_extreme_high_tail_at_or_above_p95": (
            seed1020_extreme_high_tail
        ),
        "interpretation": (
            "single-seed A0 was a lucky random-head realization"
            if seed1020_extreme_high_tail
            else (
                "seed1020 A0 is not in the predefined extreme high tail; "
                "A1 Clean clustering degradation remains concerning"
            )
        ),
        "threshold_definition": "extreme high tail means empirical percentile >= 95",
    }

    summary = {
        "stage": "B3-A2.1",
        "scope": "seed20_offline_failure_attribution_only",
        "model_seed": 20,
        "clean": {
            "native_z": clean["native_z"],
            "random_head_null": clean["random_head_null"],
            "a0_seed1020": clean["a0_seed1020"],
            "a1_uniform": clean["a1_uniform"],
            "a0_seed1020_vs_random_null": clean[
                "a0_seed1020_vs_random_null"
            ],
            "a1_vs_random_null": clean["a1_vs_random_null"],
            "predictability": clean["predictability"],
        },
        "snr2p5_k2": {
            "native_z": noisy["native_z"],
            "random_head_null": noisy["random_head_null"],
            "a0_seed1020": noisy["a0_seed1020"],
            "a1_uniform": noisy["a1_uniform"],
            "a0_seed1020_vs_random_null": noisy[
                "a0_seed1020_vs_random_null"
            ],
            "a1_vs_random_null": noisy["a1_vs_random_null"],
            "predictability": noisy["predictability"],
            "bootstrap": noisy_bootstrap,
            "agreement_bootstrap": noisy["agreement_bootstrap"],
        },
        "fold_assignment_sha256": {
            "clean": clean["integrity"]["fold_assignment_sha256"],
            "snr2p5_k2": noisy["integrity"]["fold_assignment_sha256"],
            "same_across_conditions": bool(
                clean["integrity"]["fold_assignment_sha256"]
                == noisy["integrity"]["fold_assignment_sha256"]
            ),
        },
        "random_head_scientific_interpretation": random_head_interpretation,
        "gates": gates,
        "decision_diagnostics": decision_diagnostics,
        "B3_FINAL_PASS_DECLARED": False,
        "B4_READY_DECLARED": False,
        "scientific_scope_note": (
            "No U_i^v is defined. Final B3 status and B4 readiness require user judgment."
        ),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        "b3_a21_seed20_summary.json",
    )
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
    _combine_null_csv(
        args.clean_json,
        args.noisy_json,
        os.path.join(args.output_dir, "random_head_null.csv"),
    )

    print("Clean random-head ACC summary: " + str(clean["random_head_null"]["fused_acc"]))
    print("SNR2.5 random-head ACC summary: " + str(noisy["random_head_null"]["fused_acc"]))
    print("Random-head interpretation: " + random_head_interpretation["interpretation"])
    print("Decision diagnostic: " + selected_case)
    for gate_name, gate_value in gates.items():
        print(gate_name + "=" + str(gate_value).lower())
    print("B3-A2.1 seed20 summary: " + output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
