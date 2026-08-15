"""Summarize a completed B5-A0 seed20 two-condition run."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


ARMS = ("semantic_only", "isotropic_rate", "conditional_rate")
CONDITIONS = ("clean", "snr2p5_k2")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output_json = Path(args.output_json) if args.output_json else (
        input_dir / "b5_a0_seed20_summary.json"
    )
    output_csv = Path(args.output_csv) if args.output_csv else (
        input_dir / "b5_a0_seed20_summary.csv"
    )
    records = []
    condition_engineering = []
    rate_channel_checks = []
    isotropic_null_confirmations = []
    for condition in CONDITIONS:
        metadata = _load_json(input_dir / condition / "metadata.json")
        condition_engineering.append(
            bool(metadata["gates"]["B5_A0_ENGINEERING_PASS"])
        )
        isotropic_null_confirmations.append(bool(
            metadata["B5_A0_ISOTROPIC_DIRECTIONAL_NULL_CONFIRMED"]
        ))
        for arm in ARMS:
            diagnostics = _load_json(
                input_dir / condition / arm / "final_semantic_diagnostics.json"
            )
            rate = _load_json(input_dir / condition / arm / "rate_audit.json")
            trace_path = input_dir / condition / arm / "training_trace.csv"
            with open(trace_path, "r", newline="", encoding="utf-8") as input_file:
                trace = list(csv.DictReader(input_file))
            first_rate = float(trace[0]["rate_loss_per_dim"])
            final_rate = float(trace[-1]["rate_loss_per_dim"])
            logvar_saturated = bool(
                rate["posterior_logvar_at_min_fraction"] > 0.90
                or rate["posterior_logvar_at_max_fraction"] > 0.90
            )
            if arm == "conditional_rate":
                rate_channel_checks.append(bool(
                    math.isfinite(rate["rate_mean"])
                    and rate["rate_min"] >= -1e-6
                    and rate["rate_mean"] > 1e-8
                    and math.isfinite(first_rate)
                    and math.isfinite(final_rate)
                    and abs(final_rate - first_rate) > 1e-8
                    and not logvar_saturated
                    and diagnostics["effective_rank"] > 2.0
                ))
            clustering = diagnostics["fused_clustering"]
            records.append({
                "condition": condition,
                "arm": arm,
                "fused_ACC": clustering["acc"],
                "fused_NMI": clustering["nmi"],
                "fused_ARI": clustering["ari"],
                "effective_rank": diagnostics["effective_rank"],
                "correspondence_gap": diagnostics["correspondence_gap"],
                "retrieval_top1": diagnostics["retrieval_top1"],
                "rate_mean_per_dim": rate["rate_mean"],
                "rate_p50": rate["rate_p50"],
                "rate_p90": rate["rate_p90"],
                "first_rate_loss_per_dim": first_rate,
                "final_rate_loss_per_dim": final_rate,
                "logvar_saturation_warning": logvar_saturated,
                "conditional_context_delta": rate.get(
                    "conditional_context_shuffled_minus_correct", ""
                ),
            })

    engineering_pass = all(condition_engineering)
    rate_channel_candidate = bool(
        engineering_pass and all(rate_channel_checks)
    )
    summary = {
        "stage": "B5-A0",
        "model_seed": 20,
        "conditions": list(CONDITIONS),
        "arms": list(ARMS),
        "records": records,
        "B5_A0_ENGINEERING_PASS": engineering_pass,
        "isotropic_fixed_radius_directional_null": True,
        "isotropic_role": "fixed_radius_directional_null_control",
        "B5_A0_ISOTROPIC_DIRECTIONAL_NULL_CONFIRMED": all(
            isotropic_null_confirmations
        ),
        "B5_A0_RATE_CHANNEL_CANDIDATE": rate_channel_candidate,
        "B5_FINAL_PASS_DECLARED": False,
        "B5_A1_STARTED": False,
        "note": (
            "Rate-channel candidacy is a mechanism-probe qualification, not a "
            "final-model or clustering-improvement claim."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_json, summary)
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=list(records[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    print("B5_A0_ENGINEERING_PASS=" + str(engineering_pass).lower())
    print("B5_A0_RATE_CHANNEL_CANDIDATE=" + str(rate_channel_candidate).lower())
    print("B5_FINAL_PASS_DECLARED=false")
    print("B5_A1_STARTED=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
