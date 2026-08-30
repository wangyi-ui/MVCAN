"""Create the preregistered one-row-per-arm B7-A0 results table."""

import argparse
import csv
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.b7_sparse_supervision.b7_sparse_label_protocol import ARMS


ARM_DIRECTORIES = {
    "UNSUP": "unsup",
    "LABEL_ONLY": "label_only",
    "U_LABEL": "u_label",
    "SHUFFLED_U_LABEL": "shuffled_u_label",
    "ORACLE_LABEL": "oracle_label",
}
RESULT_FIELDS = (
    "arm",
    "ACC_all",
    "NMI_all",
    "ARI_all",
    "ACC_unlabeled",
    "NMI_unlabeled",
    "ARI_unlabeled",
    "ACC_labeled",
    "NMI_labeled",
    "ARI_labeled",
    "supervised_channel_count",
    "final_loss",
    "final_unsup_loss",
    "final_sup_loss",
)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def summarize_outputs(output_dir):
    root = Path(output_dir)
    rows = []
    for arm in ARMS:
        arm_root = root / ARM_DIRECTORIES[arm]
        metadata = _read_json(arm_root / "metadata.json")
        metrics = _read_json(arm_root / "metrics.json")
        row = {
            "arm": arm,
            **{name: metrics[name] for name in RESULT_FIELDS if name in metrics},
            "supervised_channel_count": metadata["supervised_channel_count"],
            "final_loss": metadata["final_loss"],
            "final_unsup_loss": metadata["final_unsup_loss"],
            "final_sup_loss": metadata["final_sup_loss"],
        }
        missing = set(RESULT_FIELDS) - set(row)
        if missing:
            raise RuntimeError("missing result fields: " + ", ".join(sorted(missing)))
        rows.append(row)
    path = root / "b7a0_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=RESULT_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = summarize_outputs(args.output_dir)
    print("Wrote " + str(len(rows)) + " B7-A0 result rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
