"""Summarize Clean and SNR2.5 B3-A2 diagnostics for seed 20."""

import argparse
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


def _fused_metrics(record):
    clustering = record["fused_clustering"]
    return {
        "acc": float(clustering["acc"]),
        "nmi": float(clustering["nmi"]),
        "ari": float(clustering["ari"]),
    }


def _metric_delta(record):
    delta = record["delta_a1_minus_a0"]
    return {
        "acc": float(delta["fused_acc"]),
        "nmi": float(delta["fused_nmi"]),
        "ari": float(delta["fused_ari"]),
    }


def _engineering_pass(record):
    integrity = record["integrity"]
    required = (
        "backbone_hash_pass",
        "a0_initialization_hash_match_pass",
        "a1_semantic_checkpoint_hash_pass",
        "corruption_protocol_pass",
        "all_outputs_finite_pass",
        "expected_shapes_pass",
        "eval_mode_pass",
        "no_optimizer_pass",
        "no_backward_pass",
        "no_parameter_update_pass",
        "shared_final_backbone_pass",
        "b3_a2_engineering_pass",
    )
    return all(integrity.get(key) is True for key in required)


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
    for record, expected_condition in (
        (clean, "clean"),
        (noisy, "snr2p5_k2"),
    ):
        _require(record.get("stage") == "B3-A2", "stage must be B3-A2")
        _require(
            record.get("condition") == expected_condition,
            "condition mismatch for " + expected_condition,
        )
        _require(record.get("dataset") == "MSRC-v1", "dataset mismatch")
        _require(record.get("model_seed") == 20, "model seed must be 20")

    clean_rank_a0 = float(clean["a0"]["collapse"]["fused"]["effective_rank"])
    clean_rank_a1 = float(clean["a1"]["collapse"]["fused"]["effective_rank"])
    noisy_rank_a0 = float(noisy["a0"]["collapse"]["fused"]["effective_rank"])
    noisy_rank_a1 = float(noisy["a1"]["collapse"]["fused"]["effective_rank"])
    clean_variance_a1 = float(
        clean["a1"]["collapse"]["fused"]["variance_mean"]
    )
    noisy_variance_a1 = float(
        noisy["a1"]["collapse"]["fused"]["variance_mean"]
    )

    clean_gap_a0 = float(
        clean["a0"]["correspondence"]["correspondence_gap"]
    )
    clean_gap_a1 = float(
        clean["a1"]["correspondence"]["correspondence_gap"]
    )
    noisy_gap_a0 = float(
        noisy["a0"]["correspondence"]["correspondence_gap"]
    )
    noisy_gap_a1 = float(
        noisy["a1"]["correspondence"]["correspondence_gap"]
    )
    clean_retrieval_a0 = float(
        clean["a0"]["correspondence"]["retrieval_top1"]
    )
    clean_retrieval_a1 = float(
        clean["a1"]["correspondence"]["retrieval_top1"]
    )
    noisy_retrieval_a0 = float(
        noisy["a0"]["correspondence"]["retrieval_top1"]
    )
    noisy_retrieval_a1 = float(
        noisy["a1"]["correspondence"]["retrieval_top1"]
    )

    noise_a0 = noisy["noise_diagnostic"]["a0"]
    noise_a1 = noisy["noise_diagnostic"]["a1"]
    engineering_pass = _engineering_pass(clean) and _engineering_pass(noisy)
    noncollapse_pass = bool(
        clean_rank_a1 > 1.5
        and noisy_rank_a1 > 1.5
        and clean_variance_a1 > 1e-6
        and noisy_variance_a1 > 1e-6
        and clean["a1"]["collapse"]["all_finite_pass"] is True
        and noisy["a1"]["collapse"]["all_finite_pass"] is True
    )
    alignment_pass = bool(
        clean_gap_a1 > clean_gap_a0
        and noisy_gap_a1 > noisy_gap_a0
        and clean_retrieval_a1 > clean_retrieval_a0
    )
    noise_direction_pass = bool(
        noise_a1["clean_agreement_mean"]
        > noise_a1["corrupted_agreement_mean"]
        and noise_a1["clean_corrupted_auc"] > 0.5
    )
    scientific_signal_candidate_pass = bool(
        engineering_pass
        and noncollapse_pass
        and alignment_pass
        and noise_direction_pass
    )

    summary = {
        "stage": "B3-A2",
        "scope": "seed20_offline_diagnostics_only",
        "model_seed": 20,
        "clean_fused_metrics_a0": _fused_metrics(clean["a0"]),
        "clean_fused_metrics_a1": _fused_metrics(clean["a1"]),
        "clean_fused_metric_delta": _metric_delta(clean),
        "noisy_fused_metrics_a0": _fused_metrics(noisy["a0"]),
        "noisy_fused_metrics_a1": _fused_metrics(noisy["a1"]),
        "noisy_fused_metric_delta": _metric_delta(noisy),
        "clean_correspondence_gap_a0": clean_gap_a0,
        "clean_correspondence_gap_a1": clean_gap_a1,
        "clean_correspondence_gap_delta": clean_gap_a1 - clean_gap_a0,
        "noisy_correspondence_gap_a0": noisy_gap_a0,
        "noisy_correspondence_gap_a1": noisy_gap_a1,
        "noisy_correspondence_gap_delta": noisy_gap_a1 - noisy_gap_a0,
        "clean_retrieval_top1_a0": clean_retrieval_a0,
        "clean_retrieval_top1_a1": clean_retrieval_a1,
        "noisy_retrieval_top1_a0": noisy_retrieval_a0,
        "noisy_retrieval_top1_a1": noisy_retrieval_a1,
        "clean_effective_rank_a0": clean_rank_a0,
        "clean_effective_rank_a1": clean_rank_a1,
        "noisy_effective_rank_a0": noisy_rank_a0,
        "noisy_effective_rank_a1": noisy_rank_a1,
        "noisy_clean_corrupted_gap_a0": float(
            noise_a0["agreement_gap_clean_minus_corrupted"]
        ),
        "noisy_clean_corrupted_gap_a1": float(
            noise_a1["agreement_gap_clean_minus_corrupted"]
        ),
        "noisy_clean_corrupted_auc_a0": float(
            noise_a0["clean_corrupted_auc"]
        ),
        "noisy_clean_corrupted_auc_a1": float(
            noise_a1["clean_corrupted_auc"]
        ),
        "noisy_spearman_a0": float(
            noise_a0["spearman_clean_indicator_vs_agreement"]
        ),
        "noisy_spearman_a1": float(
            noise_a1["spearman_clean_indicator_vs_agreement"]
        ),
        "B3_A2_ENGINEERING_PASS": engineering_pass,
        "B3_A2_NONCOLLAPSE_PASS": noncollapse_pass,
        "B3_A2_ALIGNMENT_SIGNAL_PASS": alignment_pass,
        "B3_A2_NOISE_DIRECTION_PASS": noise_direction_pass,
        "B3_A2_SCIENTIFIC_SIGNAL_CANDIDATE_PASS": (
            scientific_signal_candidate_pass
        ),
        "scientific_scope_note": (
            "Seed20 is a candidate signal only; this is not B3 Strong PASS."
        ),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "b3_a2_seed20_summary.json")
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")

    print("Clean fused A0/A1: " + str(summary["clean_fused_metrics_a0"]) + " / " + str(summary["clean_fused_metrics_a1"]))
    print("Noisy fused A0/A1: " + str(summary["noisy_fused_metrics_a0"]) + " / " + str(summary["noisy_fused_metrics_a1"]))
    print("Clean correspondence gap A0/A1: " + str(clean_gap_a0) + " / " + str(clean_gap_a1))
    print("Noisy correspondence gap A0/A1: " + str(noisy_gap_a0) + " / " + str(noisy_gap_a1))
    print("Clean retrieval@1 A0/A1: " + str(clean_retrieval_a0) + " / " + str(clean_retrieval_a1))
    print("Noisy retrieval@1 A0/A1: " + str(noisy_retrieval_a0) + " / " + str(noisy_retrieval_a1))
    print("A1 effective rank Clean/Noisy: " + str(clean_rank_a1) + " / " + str(noisy_rank_a1))
    print("A1 noisy clean/corrupted agreement: " + str(noise_a1["clean_agreement_mean"]) + " / " + str(noise_a1["corrupted_agreement_mean"]))
    print("A1 noisy clean-corrupted gap: " + str(noise_a1["agreement_gap_clean_minus_corrupted"]))
    print("A1 noisy ROC-AUC: " + str(noise_a1["clean_corrupted_auc"]))
    print("A1 noisy Spearman: " + str(noise_a1["spearman_clean_indicator_vs_agreement"]))
    for gate in (
        "B3_A2_ENGINEERING_PASS",
        "B3_A2_NONCOLLAPSE_PASS",
        "B3_A2_ALIGNMENT_SIGNAL_PASS",
        "B3_A2_NOISE_DIRECTION_PASS",
        "B3_A2_SCIENTIFIC_SIGNAL_CANDIDATE_PASS",
    ):
        print(gate + "=" + str(summary[gate]).lower())
    print("B3-A2 seed20 summary: " + output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
