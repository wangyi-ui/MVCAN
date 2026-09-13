"""Evaluate F0-A0 only after validating the immutable pre-GT seal."""

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import materialize_c3_b0_true_u_carrier as replay
from experiments.cyclic_utility import train_c3_b0_relation_action_pilot as c3_train
from experiments.e1_pairwise_utility import train_e1_pairwise_utility as e1_train
from experiments.final_core import final_core_protocol as f0
from experiments.final_core import train_final_core as f0_train


def _require(condition, message="F0_A0_EVALUATOR_FAIL_CLOSED"):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, record):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(record, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def default_paths(seed):
    root = f0_train.default_output_dir(seed)
    return {
        "artifact": root / "final_core_pre_gt_artifact.npz",
        "audit": root / "final_core_pre_gt_audit.json",
        "seal": root / "final_core_pre_gt_seal.json",
        "metrics": root / "final_core_metrics.json",
    }


def load_frozen_metric_reference(seed):
    active_seed = f0.validate_seed(seed)
    audit_path, _ = replay.frozen_true_u_reference_paths(active_seed)
    metric_path = audit_path.parent / "metrics.json"
    _require(metric_path.is_file(), "F0_A0_FROZEN_METRIC_REFERENCE_MISSING")
    record = _read_json(metric_path)
    expected = f0.frozen_metrics(active_seed)
    _require(
        record.get("stage") == "C3-B0"
        and record.get("arm") == "TRUE_U"
        and int(record.get("seed", -1)) == active_seed
        and int(record.get("evaluation_epoch", -1)) == f0.EPOCHS
        and record.get("GT_used_for_training") is False,
        "F0_A0_FROZEN_METRIC_REFERENCE_BOUNDARY_FAIL_CLOSED",
    )
    f0.validate_metric_parity(record.get("metrics"), expected)
    return expected, {
        "path": str(metric_path.relative_to(REPOSITORY_ROOT)),
        "file_sha256": f0.file_sha256(metric_path),
        "frozen_reference_matches_audited_constants": True,
    }


def evaluate(
    seed, artifact_path, audit_path, seal_path, full_gt_path, output_path,
):
    active_seed = f0.validate_seed(seed)
    output = _resolve(output_path)
    _require(not output.exists(), "refusing to overwrite F0-A0 metrics")

    # This complete seal/hash/schema check necessarily precedes the sole GT load.
    arrays, pre_gt = f0.validate_pre_gt_seal(
        active_seed, _resolve(artifact_path), _resolve(audit_path),
        _resolve(seal_path),
    )
    parent_integrity = f0_train.verify_parent_integrity()
    expected_metrics, metric_reference = load_frozen_metric_reference(active_seed)
    reference = replay.load_frozen_true_u_reference(active_seed)
    prediction_parity = f0.validate_exact_array(
        "final_predictions", arrays["final_predictions"],
        reference["predictions"], (f0.N,),
    )
    f0.validate_sample_ids(arrays["sample_ids"], reference["sample_ids"])

    # Full GT is loaded exactly here, after the valid pre-GT seal and parents.
    labels = e1_train.load_labels_after_predictions(
        _resolve(full_gt_path), _resolve(artifact_path)
    )
    actual_metrics = e1_train.evaluate_predictions(
        labels, arrays["final_predictions"]
    )
    parity = f0.validate_metric_parity(actual_metrics, expected_metrics)
    record = {
        "stage": f0.STAGE,
        "seed": active_seed,
        "scientific_parent": "C3-B0 TRUE_U",
        "pre_gt_seal_valid": True,
        "GT_loaded_only_after_pre_gt_seal": True,
        "GT_used_for_training": False,
        "parent_hashes_pass": parent_integrity["all_parent_hashes_pass"],
        "final_predictions_equal": prediction_parity["exact_equal"],
        "frozen_metric_reference": metric_reference,
        "metrics": actual_metrics,
        "frozen_C3_B0_TRUE_U_metrics": expected_metrics,
        **parity,
        "pre_gt_provenance": pre_gt,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, record)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Post-seal F0-A0 exact metric evaluator"
    )
    parser.add_argument("--seed", type=int, required=True, choices=f0.SEEDS)
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--audit-path", default=None)
    parser.add_argument("--seal-path", default=None)
    parser.add_argument("--full-gt-path", default=str(c3_train.DEFAULT_FULL_GT_PATH))
    parser.add_argument("--output-path", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    defaults = default_paths(args.seed)
    result = evaluate(
        args.seed,
        defaults["artifact"] if args.artifact_path is None else args.artifact_path,
        defaults["audit"] if args.audit_path is None else args.audit_path,
        defaults["seal"] if args.seal_path is None else args.seal_path,
        args.full_gt_path,
        defaults["metrics"] if args.output_path is None else args.output_path,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
