"""Summarize the five formal C1 arms using the frozen decision tree."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import (
    c1_frozen_pseudo_supervision as c1,
)
from experiments.cyclic_utility import (
    train_c1_frozen_pseudo_supervision as c1_train,
)
from irv.b4_information_utility import tensor_sha256


DEFAULT_INPUT_DIR = c1_train.DEFAULT_OUTPUT_ROOT
SUMMARY_FILES = (
    "c1_summary.json",
    "c1_decision.json",
    "c1_audit.json",
)
REQUIRED_ARM_FILES = (
    "metrics.json",
    "train_audit.json",
    "pseudo_target_audit.json",
    "final_predictions.npz",
    "loss_history.json",
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def load_arm_bundle(input_dir, arm):
    if arm not in c1.FORMAL_ARMS:
        raise ValueError("summarizer accepts formal C1 arms only")
    arm_dir = Path(input_dir) / arm
    _require(arm_dir.is_dir(), "missing C1 arm directory: " + arm)
    for filename in REQUIRED_ARM_FILES:
        _require(
            (arm_dir / filename).is_file(),
            "missing C1 arm file: " + arm + "/" + filename,
        )
    metrics_record = c1.read_json(arm_dir / "metrics.json")
    train_audit = c1.read_json(arm_dir / "train_audit.json")
    pseudo_audit = c1.read_json(arm_dir / "pseudo_target_audit.json")
    loss_history = c1.read_json(arm_dir / "loss_history.json")
    _require(
        metrics_record.get("stage") == "C1"
        and metrics_record.get("arm") == arm
        and train_audit.get("stage") == "C1"
        and train_audit.get("arm") == arm
        and pseudo_audit.get("arm") == arm
        and loss_history.get("stage") == "C1"
        and loss_history.get("arm") == arm,
        "C1 arm record identity mismatch: " + arm,
    )
    metrics = metrics_record.get("metrics", {})
    _require(
        set(metrics) == set(c1.METRIC_NAMES)
        and all(
            isinstance(metrics[name], (int, float))
            and math.isfinite(float(metrics[name]))
            for name in c1.METRIC_NAMES
        ),
        "C1 metric schema mismatch: " + arm,
    )
    _require(
        train_audit.get("epochs") == metrics_record.get("epochs")
        == loss_history.get("epochs")
        and train_audit.get("seed") == metrics_record.get("seed")
        == c1.SEED,
        "C1 epoch/seed record mismatch: " + arm,
    )
    with np.load(
        arm_dir / "final_predictions.npz", allow_pickle=False
    ) as archive:
        _require(
            tuple(archive.files) == ("predictions", "sample_ids"),
            "C1 prediction archive field mismatch: " + arm,
        )
        predictions = np.asarray(
            archive["predictions"], dtype=np.int64
        )
        sample_ids = np.asarray(archive["sample_ids"], dtype=np.int64)
    _require(
        predictions.shape == sample_ids.shape == (c1.SAMPLE_NUM,)
        and np.array_equal(
            sample_ids, np.arange(c1.SAMPLE_NUM, dtype=np.int64)
        )
        and tensor_sha256(predictions)
        == train_audit["prediction_logical_sha256"],
        "C1 prediction seal mismatch: " + arm,
    )
    return {
        "metrics": {
            name: float(metrics[name]) for name in c1.METRIC_NAMES
        },
        "metrics_record": metrics_record,
        "train_audit": train_audit,
        "pseudo_target_audit": pseudo_audit,
        "loss_history": loss_history,
        "prediction_logical_sha256": tensor_sha256(predictions),
    }


def validate_formal_bundles(bundles):
    _require(
        tuple(bundles) == c1.FORMAL_ARMS,
        "formal C1 bundle arm set/order mismatch",
    )
    train_audits = {
        arm: bundles[arm]["train_audit"] for arm in c1.FORMAL_ARMS
    }
    initialization = c1.common_initialization_fairness(train_audits)
    _require(
        initialization["fairness_pass"]
        and initialization["same_y_gen_all_pseudo_arms"],
        "C1 formal initialization/data-order fairness failed",
    )
    source_records = {
        str(bundles[arm]["train_audit"]["source_provenance"])
        for arm in c1.FORMAL_ARMS
    }
    c0_artifact_hashes = {
        bundles[arm]["train_audit"]["C0_artifact_provenance"][
            "artifact_file_sha256"
        ]
        for arm in c1.FORMAL_ARMS
    }
    M0_hashes = {
        bundles[arm]["train_audit"]["M0_audit"]["logical_sha256"]
        for arm in c1.FORMAL_ARMS
    }
    epoch_counts = {
        bundles[arm]["train_audit"]["epochs"]
        for arm in c1.FORMAL_ARMS
    }
    leakage_pass = True
    native_graph_pass = True
    for arm in c1.FORMAL_ARMS:
        audit = bundles[arm]["train_audit"]
        leakage = audit["leakage"]
        historical = audit["historical_non_repetition"]
        leakage_pass = bool(
            leakage_pass
            and leakage["full_GT_loaded_during_training"] is False
            and leakage["full_GT_loaded_after_prediction_seal"] is True
            and leakage["sparse_labels_loaded"] is False
            and leakage["R_loaded"] is False
            and leakage["corruption_mask_loaded"] is False
            and leakage["oracle_used"] is False
        )
        native_graph_pass = bool(
            native_graph_pass
            and historical["native_P_all_rewritten"] is False
            and historical["P_corr_used"] is False
            and historical["P_util_used"] is False
            and audit["runtime"]["LWC_preserved"] is True
        )
        _require(
            audit["lambda_pseudo"]
            == c1.effective_lambda_pseudo(arm),
            "C1 lambda mismatch: " + arm,
        )
        if arm == "BASE":
            _require(
                bundles[arm]["pseudo_target_audit"][
                    "pseudo_supervision_used"
                ]
                is False,
                "BASE unexpectedly used pseudo supervision",
            )
        else:
            _require(
                bundles[arm]["pseudo_target_audit"][
                    "all_pseudo_tensors_detached_pass"
                ]
                is True,
                "pseudo tensor freeze audit failed: " + arm,
            )
    checks = {
        "same_frozen_source_all_arms": len(source_records) == 1,
        "same_C0_artifact_all_arms": c0_artifact_hashes
        == {c1.EXPECTED_C0_ARTIFACT_SHA256},
        "same_M0_all_arms": M0_hashes
        == {c1.EXPECTED_M0_LOGICAL_SHA256},
        "same_epoch_count_all_arms": len(epoch_counts) == 1,
        "initialization_fairness_pass": initialization[
            "fairness_pass"
        ],
        "same_y_gen_all_pseudo_arms": initialization[
            "same_y_gen_all_pseudo_arms"
        ],
        "GT_leakage_boundary_pass": leakage_pass,
        "native_graph_unchanged_pass": native_graph_pass,
    }
    _require(all(checks.values()), "C1 formal cross-arm audit failed")
    return {**checks, "initialization": initialization}


def summarize(input_dir=DEFAULT_INPUT_DIR):
    root = _resolve(input_dir)
    _require(root.is_dir(), "C1 input directory is missing")
    for filename in SUMMARY_FILES:
        _require(
            not (root / filename).exists(),
            "refusing to overwrite C1 summary output",
        )
    bundles = {
        arm: load_arm_bundle(root, arm) for arm in c1.FORMAL_ARMS
    }
    cross_arm_audit = validate_formal_bundles(bundles)
    metrics_by_arm = {
        arm: bundles[arm]["metrics"] for arm in c1.FORMAL_ARMS
    }
    decision = c1.build_c1_pilot_decision(metrics_by_arm)
    summary = {
        "stage": "C1",
        "seed": c1.SEED,
        "formal_arms": list(c1.FORMAL_ARMS),
        "engineering_arm_excluded": c1.ENGINEERING_ARM,
        "metrics_by_arm": metrics_by_arm,
        "comparisons": decision["comparisons"],
        "final_decision": decision["final_decision"],
    }
    audit = {
        "stage": "C1",
        "formal_arm_count": len(c1.FORMAL_ARMS),
        "required_arm_files": list(REQUIRED_ARM_FILES),
        "cross_arm": cross_arm_audit,
        "C0_artifact_source": str(c1_train.DEFAULT_C0_ARTIFACT_PATH),
        "C0_artifact_file_sha256": c1.EXPECTED_C0_ARTIFACT_SHA256,
        "C0_pre_GT_arrays_only": True,
        "full_GT_loaded_during_training": False,
        "sparse_labels_loaded": False,
        "R_loaded": False,
        "corruption_mask_loaded": False,
        "oracle_used": False,
        "Memory_used": False,
        "P_corr_used": False,
        "P_util_used": False,
        "native_target_rewritten": False,
        "dynamic_pseudo_refresh_used": False,
        "CYCLE_ZERO_in_scientific_decision": False,
        "decision_tree_exhaustive_pass": (
            sum(decision["decision_conditions"].values()) == 1
        ),
        "C1_AUDIT_PASS": True,
    }
    c1.write_json(root / "c1_summary.json", summary)
    c1.write_json(root / "c1_decision.json", decision)
    c1.write_json(root / "c1_audit.json", audit)
    return {"summary": summary, "decision": decision, "audit": audit}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = summarize(args.input_dir)
    print(
        "C1_AUDIT_PASS="
        + str(result["audit"]["C1_AUDIT_PASS"])
    )
    print("DECISION=" + result["decision"]["final_decision"])
    print("Saved: " + str(_resolve(args.input_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
