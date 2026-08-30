"""Audit persisted B7-A0 artifacts without retraining or model selection."""

import argparse
import csv
import inspect
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.b7_sparse_supervision.b7_sparse_label_protocol import ARMS
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
    LABELS_PER_CLASS,
)
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
    SAMPLE_NUM,
)
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import TOP_K
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import VIEW_NUM
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
    build_normal_supervised_admission,
)
from experiments.b7_sparse_supervision.summarize_b7_a0_sparse_supervision import (
    ARM_DIRECTORIES,
)
from weak_quality import ndarray_sha256


EXPECTED_CHANNEL_COUNTS = {
    "UNSUP": 0,
    "LABEL_ONLY": 84,
    "U_LABEL": 42,
    "SHUFFLED_U_LABEL": 42,
    "ORACLE_LABEL": 42,
}


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def audit_outputs(output_dir):
    root = Path(output_dir)
    metadata = _read_json(root / "metadata.json")
    split = _read_json(root / "label_split.json")
    labeled_ids = np.load(root / "labeled_sample_ids.npy", allow_pickle=False)
    unlabeled_ids = np.load(root / "unlabeled_sample_ids.npy", allow_pickle=False)
    split_exact_pass = bool(
        labeled_ids.dtype == np.dtype(np.int64)
        and unlabeled_ids.dtype == np.dtype(np.int64)
        and labeled_ids.shape == (14,)
        and unlabeled_ids.shape == (SAMPLE_NUM - 14,)
        and ndarray_sha256(labeled_ids) == split["labeled_ids_sha256"]
        and np.intersect1d(labeled_ids, unlabeled_ids).size == 0
        and np.array_equal(
            np.sort(np.concatenate((labeled_ids, unlabeled_ids))),
            np.arange(SAMPLE_NUM),
        )
        and all(
            len(sample_ids) == LABELS_PER_CLASS
            for sample_ids in split["per_class_labeled_ids"].values()
        )
    )

    arm_metadata = {}
    admissions = {}
    all_artifacts_present_pass = True
    all_metrics_finite_pass = True
    label_split_shared_pass = True
    initial_state_shared_pass = True
    channel_counts_pass = True
    admission_row_counts_pass = True
    for arm in ARMS:
        arm_root = root / ARM_DIRECTORIES[arm]
        required = (
            "metadata.json",
            "metrics.json",
            "training_history.json",
            "final_semantic.npy",
            "prediction.npy",
            "supervised_admission_mask.npy",
            "unsupervised_admission_mask.npy",
            "training_labeled_sample_ids.npy",
            "final_semantic_head.pt",
        )
        all_artifacts_present_pass = bool(
            all_artifacts_present_pass
            and all((arm_root / name).is_file() for name in required)
        )
        arm_metadata[arm] = _read_json(arm_root / "metadata.json")
        metrics = _read_json(arm_root / "metrics.json")
        admissions[arm] = np.load(
            arm_root / "supervised_admission_mask.npy", allow_pickle=False
        )
        training_ids = np.load(
            arm_root / "training_labeled_sample_ids.npy", allow_pickle=False
        )
        label_split_shared_pass = bool(
            label_split_shared_pass
            and np.array_equal(training_ids, labeled_ids)
            and arm_metadata[arm]["label_split_sha256"]
            == split["label_split_sha256"]
        )
        initial_state_shared_pass = bool(
            initial_state_shared_pass
            and arm_metadata[arm]["initial_head_sha256"]
            == metadata["initial_head_sha256"]
        )
        channel_count = int(admissions[arm].sum(dtype=np.int64))
        channel_counts_pass = bool(
            channel_counts_pass
            and channel_count == EXPECTED_CHANNEL_COUNTS[arm]
            and arm_metadata[arm]["supervised_channel_count"] == channel_count
        )
        row_counts = admissions[arm][labeled_ids].sum(axis=1)
        expected_per_row = {
            "UNSUP": 0,
            "LABEL_ONLY": VIEW_NUM,
            "U_LABEL": TOP_K,
            "SHUFFLED_U_LABEL": TOP_K,
            "ORACLE_LABEL": TOP_K,
        }[arm]
        admission_row_counts_pass = bool(
            admission_row_counts_pass
            and admissions[arm].shape == (SAMPLE_NUM, VIEW_NUM)
            and np.all(row_counts == expected_per_row)
            and not admissions[arm][unlabeled_ids].any()
        )
        all_metrics_finite_pass = bool(
            all_metrics_finite_pass
            and all(np.isfinite(float(value)) for value in metrics.values())
            and all(
                np.isfinite(float(arm_metadata[arm][name]))
                for name in (
                    "final_loss",
                    "final_unsup_loss",
                    "final_sup_loss",
                )
            )
        )

    common_unsup_masks = [
        np.load(
            root / ARM_DIRECTORIES[arm] / "unsupervised_admission_mask.npy",
            allow_pickle=False,
        )
        for arm in ARMS
    ]
    common_unsup_admission_pass = bool(
        all(
            np.array_equal(common_unsup_masks[0], value)
            for value in common_unsup_masks[1:]
        )
        and common_unsup_masks[0].shape == (SAMPLE_NUM, VIEW_NUM)
        and np.all(common_unsup_masks[0].sum(axis=1) == TOP_K)
    )
    results_path = root / "b7a0_results.csv"
    results_rows = []
    if results_path.is_file():
        with open(results_path, "r", encoding="utf-8", newline="") as input_file:
            results_rows = list(csv.DictReader(input_file))
    results_complete_pass = bool(
        len(results_rows) == len(ARMS)
        and [row["arm"] for row in results_rows] == list(ARMS)
    )
    normal_parameters = inspect.signature(
        build_normal_supervised_admission
    ).parameters
    normal_api_oracle_isolation_pass = bool(
        all(
            token not in name.lower()
            for name in normal_parameters
            for token in ("oracle", "corrupt", "clean", "mask")
        )
    )
    propagated_passes = {
        name: bool(
            metadata.get(name) is True
            and all(arm_metadata[arm].get(name) is True for arm in ARMS)
        )
        for name in (
            "backbone_requires_grad_pass",
            "backbone_parameter_immutability_pass",
            "u_frozen_input_pass",
            "same_initialization_pass",
            "no_unlabeled_label_training_pass",
            "oracle_isolation_pass",
            "shuffle_deterministic_pass",
            "all_finite_pass",
        )
    }
    engineering_checks = {
        "all_artifacts_present_pass": all_artifacts_present_pass,
        "split_exact_pass": split_exact_pass,
        "label_split_shared_pass": label_split_shared_pass,
        "initial_state_shared_pass": initial_state_shared_pass,
        "channel_counts_pass": channel_counts_pass,
        "admission_row_counts_pass": admission_row_counts_pass,
        "common_unsup_admission_pass": common_unsup_admission_pass,
        "normal_api_oracle_isolation_pass": normal_api_oracle_isolation_pass,
        "all_metrics_finite_pass": all_metrics_finite_pass,
        "results_complete_pass": results_complete_pass,
        **propagated_passes,
    }
    result = {
        "stage": "B7-A0",
        "dataset": metadata["dataset"],
        "mode": metadata["mode"],
        "label_split_sha256": split["label_split_sha256"],
        "initial_head_sha256": metadata["initial_head_sha256"],
        "u_sha256": metadata["u_sha256"],
        "supervised_channel_counts": EXPECTED_CHANNEL_COUNTS,
        "engineering_checks": engineering_checks,
        "B7_A0_ENGINEERING_PASS": bool(all(engineering_checks.values())),
    }
    _write_json(root / "b7a0_audit.json", result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = audit_outputs(args.output_dir)
    print("B7_A0_ENGINEERING_PASS=" + str(result["B7_A0_ENGINEERING_PASS"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
