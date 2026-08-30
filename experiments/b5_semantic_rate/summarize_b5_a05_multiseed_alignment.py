"""Pure JSON summary for B5-A0.5 multi-seed alignment replication."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

EXPECTED_SEEDS = (20, 30, 50)
DEFAULT_INPUTS = {
    20: (
        REPOSITORY_ROOT
        / "outputs/b5_semantic_rate/"
        "a04_relation_utility_alignment_step100_seed20/"
        "b5_a04_relation_utility_alignment.json"
    ),
    30: (
        REPOSITORY_ROOT
        / "outputs/b5_semantic_rate/"
        "a05_utility_alignment_step100_seed30/"
        "b5_a04_relation_utility_alignment.json"
    ),
    50: (
        REPOSITORY_ROOT
        / "outputs/b5_semantic_rate/"
        "a05_utility_alignment_step100_seed50/"
        "b5_a04_relation_utility_alignment.json"
    ),
}
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "outputs/b5_semantic_rate/a05_multiseed_alignment_summary.json"
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _display_path(path):
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def load_alignment_json(path, expected_seed):
    """Load one completed A0.4 JSON and validate its seed/protocol metadata."""
    path = _resolve(path)
    _require(path.is_file(), "missing alignment JSON: " + str(path))
    with open(path, "r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    _require(isinstance(value, dict), "alignment JSON must be an object")
    _require(value.get("stage") == "B5-A0.4", "alignment stage mismatch")
    _require(value.get("condition") == "snr2p5_k2", "condition mismatch")
    _require(
        int(value.get("model_seed", -1)) == int(expected_seed),
        "alignment seed mismatch",
    )
    _require(
        value.get("B5_A04_ALIGNMENT_AUDIT_COMPLETE") is True,
        "alignment audit is incomplete",
    )
    _require(value.get("training_performed") is False, "summary input trained")
    specificity = value.get("global_specificity")
    _require(isinstance(specificity, dict), "global specificity is missing")
    return value, path


def alignment_summary_row(value):
    """Extract only preregistered A0.5 metrics from one A0.4 result."""
    source = value["fe_adjusted_metrics"]["source"]
    pair = value["fe_adjusted_metrics"]["pair"]
    intervals = value["bootstrap_cis"]
    differences = value["paired_differences"]
    specificity = value["global_specificity"]
    return {
        "seed": int(value["model_seed"]),
        "global_specificity_p": float(specificity["permutation_p_value"]),
        "global_specificity_delta": float(specificity["delta_mean"]),
        "global_specificity_pass": bool(specificity["specificity_pass"]),
        "AUC_FE_source": float(source["auc"]),
        "AUC_FE_source_CI": [
            float(intervals["auc_fe_source"]["lower"]),
            float(intervals["auc_fe_source"]["upper"]),
        ],
        "Spearman_FE_source": float(source["spearman"]),
        "Spearman_FE_source_CI": [
            float(intervals["spearman_fe_source"]["lower"]),
            float(intervals["spearman_fe_source"]["upper"]),
        ],
        "SOURCE_U_SIGNAL_PASS": bool(value["B5_A04_SOURCE_U_SIGNAL_PASS"]),
        "AUC_FE_pair": float(pair["auc"]),
        "Spearman_FE_pair": float(pair["spearman"]),
        "PAIRWISE_T_SIGNAL_PASS": bool(value["B5_A04_PAIRWISE_T_SIGNAL_PASS"]),
        "Delta_AUC_pair_minus_source": float(differences["delta_auc"]),
        "Delta_Spearman_pair_minus_source": float(
            differences["delta_spearman"]
        ),
    }


def compute_replication_gates(rows):
    """Apply the preregistered 2/3 and all-three A0.5 decision rules."""
    if len(rows) != 3 or sorted(row["seed"] for row in rows) != [20, 30, 50]:
        raise ValueError("replication gates require seeds 20, 30, and 50")
    source_pass_count = sum(row["SOURCE_U_SIGNAL_PASS"] for row in rows)
    source_all_auc_positive = all(row["AUC_FE_source"] > 0.5 for row in rows)
    source_all_rho_positive = all(
        row["Spearman_FE_source"] > 0.0 for row in rows
    )
    pair_not_preferred_count = sum(
        row["AUC_FE_pair"] < row["AUC_FE_source"]
        and row["Spearman_FE_pair"] < row["Spearman_FE_source"]
        for row in rows
    )
    specificity_pass_count = sum(
        row["global_specificity_pass"] for row in rows
    )
    specificity_all_delta_positive = all(
        row["global_specificity_delta"] > 0.0 for row in rows
    )
    source_replication = bool(
        source_pass_count >= 2
        and source_all_auc_positive
        and source_all_rho_positive
    )
    pair_not_preferred = bool(pair_not_preferred_count >= 2)
    rate_replication = bool(
        specificity_pass_count >= 2 and specificity_all_delta_positive
    )
    entry_eligible = bool(
        source_replication and pair_not_preferred and rate_replication
    )
    return {
        "source_u_signal_pass_count": int(source_pass_count),
        "source_u_all_auc_point_estimates_gt_0p5": source_all_auc_positive,
        "source_u_all_spearman_point_estimates_gt_0": source_all_rho_positive,
        "pair_t_not_preferred_count": int(pair_not_preferred_count),
        "global_specificity_pass_count": int(specificity_pass_count),
        "global_specificity_all_delta_mean_gt_0": (
            specificity_all_delta_positive
        ),
        "B5_A05_SOURCE_U_REPLICATION_PASS": source_replication,
        "B5_A05_PAIRWISE_T_NOT_PREFERRED": pair_not_preferred,
        "B5_A05_CONDITIONAL_RATE_REPLICATION_PASS": rate_replication,
        "B5_A1_ENTRY_ELIGIBLE": entry_eligible,
    }


def aggregate_statistics(rows):
    """Return descriptive mean/population-std across exactly three seeds."""
    names = (
        "AUC_FE_source",
        "Spearman_FE_source",
        "AUC_FE_pair",
        "Spearman_FE_pair",
    )
    result = {}
    for name in names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        result[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        }
    result["seed_count"] = len(rows)
    result["inference_performed"] = False
    return result


def summarize_alignment_jsons(seed_paths):
    """Read three JSONs and return the complete A0.5 summary object."""
    _require(set(seed_paths) == set(EXPECTED_SEEDS), "seed input set mismatch")
    rows = []
    inputs = {}
    for seed in EXPECTED_SEEDS:
        value, path = load_alignment_json(seed_paths[seed], seed)
        rows.append(alignment_summary_row(value))
        inputs[str(seed)] = _display_path(path)
    gates = compute_replication_gates(rows)
    return {
        "stage": "B5-A0.5",
        "condition": "snr2p5_k2",
        "seeds": list(EXPECTED_SEEDS),
        "input_jsons": inputs,
        "summary_reads_json_only": True,
        "training_performed": False,
        "recomputation_performed": False,
        "significance_test_across_three_seeds_performed": False,
        "rows": rows,
        "aggregate_statistics": aggregate_statistics(rows),
        **gates,
        "B5_A1_STARTED": False,
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _print_summary(result):
    header = (
        "seed  spec_p  spec_pass  AUC_source [CI]  rho_source [CI]  "
        "U_pass  AUC_pair  rho_pair  T_pass  dAUC  dRho"
    )
    print(header)
    for row in result["rows"]:
        auc_ci = row["AUC_FE_source_CI"]
        rho_ci = row["Spearman_FE_source_CI"]
        print(
            f"{row['seed']:>4d}  {row['global_specificity_p']:.6f}  "
            f"{str(row['global_specificity_pass']):>9s}  "
            f"{row['AUC_FE_source']:.6f} "
            f"[{auc_ci[0]:.6f},{auc_ci[1]:.6f}]  "
            f"{row['Spearman_FE_source']:.6f} "
            f"[{rho_ci[0]:.6f},{rho_ci[1]:.6f}]  "
            f"{str(row['SOURCE_U_SIGNAL_PASS']):>6s}  "
            f"{row['AUC_FE_pair']:.6f}  "
            f"{row['Spearman_FE_pair']:.6f}  "
            f"{str(row['PAIRWISE_T_SIGNAL_PASS']):>6s}  "
            f"{row['Delta_AUC_pair_minus_source']:.6f}  "
            f"{row['Delta_Spearman_pair_minus_source']:.6f}"
        )
    for name in (
        "B5_A05_SOURCE_U_REPLICATION_PASS",
        "B5_A05_PAIRWISE_T_NOT_PREFERRED",
        "B5_A05_CONDITIONAL_RATE_REPLICATION_PASS",
        "B5_A1_ENTRY_ELIGIBLE",
    ):
        print(name + "=" + str(result[name]))
    print("B5_A1_STARTED=false")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed20-json", default=str(DEFAULT_INPUTS[20]))
    parser.add_argument("--seed30-json", default=str(DEFAULT_INPUTS[30]))
    parser.add_argument("--seed50-json", default=str(DEFAULT_INPUTS[50]))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = summarize_alignment_jsons({
        20: args.seed20_json,
        30: args.seed30_json,
        50: args.seed50_json,
    })
    _write_json(_resolve(args.output), result)
    _print_summary(result)
    return result


if __name__ == "__main__":
    main()
