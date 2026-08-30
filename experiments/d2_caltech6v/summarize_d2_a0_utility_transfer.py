"""Print a compact summary from saved D2-A0 results without recomputation."""

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "outputs/d2_caltech6v/a0_utility_transfer_seed20/utility_transfer.json"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = REPOSITORY_ROOT / input_path
    with open(input_path, "r", encoding="utf-8") as input_file:
        result = json.load(input_file)

    print("Dataset:", result["dataset"])
    print("Seed:", result["model_seed"])
    print("N/V/K:", result["N"], result["V"], result["K"])
    print("T AUC:", result["T_auc_clean_vs_corrupt"])
    print("U AUC:", result["U_auc_clean_vs_corrupt"])
    print("T clean/corrupt gap:", result["T_gap"])
    print("U clean/corrupt gap:", result["U_gap"])
    print("Top3 clean fraction:", result["top3_clean_fraction"])
    print("Top3 all-clean rate:", result["top3_all_clean_rate"])
    print(
        "Random baselines:",
        result["random_top3_clean_fraction"],
        result["random_top3_all_clean_rate"],
    )
    print("Corrupted admitted mean:", result["corrupted_views_admitted_mean"])
    print("Jaccard:", result["top3_oracle_jaccard_mean"])
    print("Pair-win probability:", result["pair_win_probability"])
    print("Margin34:", result["margin34_mean"])
    print(
        "Margin34 admission-error AUC:",
        result["margin34_admission_error_auc"],
    )
    print(
        "D2_A0_UTILITY_TRANSFER_PASS =",
        result["D2_A0_UTILITY_TRANSFER_PASS"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
