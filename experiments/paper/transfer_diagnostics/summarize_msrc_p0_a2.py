"""Summarize P0-A2 using only the preregistered diagnostic comparisons."""

import argparse
import json
from pathlib import Path

from . import msrc_p0_a2_protocol as protocol


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _diagnostic_case(acc):
    base = float(acc["BASE"])
    uniform = float(acc["UNIFORM"])
    true_u = float(acc["TRUE_U"])
    if true_u > uniform > base:
        return "A"
    if uniform > base and true_u <= uniform:
        return "B"
    return "C"


def summarize(*, root_dir, output_path):
    root = Path(root_dir)
    target = Path(output_path)
    _require(root.resolve() == protocol.OUTPUT_ROOT.resolve(),
             "summary root must be the frozen P0-A2 output root")
    _require(target.resolve() == (protocol.OUTPUT_ROOT / "seed20_three_arm_summary.json").resolve(),
             "summary path must be the frozen P0-A2 summary path")
    _require(not target.exists(), "refusing to overwrite summary output")
    records = {}
    manifests = {}
    frozen_fields = (
        "dataset", "dataset_sha256", "input_feature_sha256",
        "corruption_mask_logical_sha256", "sparse_split_logical_sha256",
        "training_seed", "label_seed", "weak_quality_seed", "snr_db",
        "epochs", "batch_size", "learning_rate", "refresh_interval",
        "native_lambda1", "initialization_dir", "checkpoint_sha256",
        "initial_model_sha256", "initialization_audit_sha256",
        "initialization_manifest_sha256", "initialization_seal_sha256",
    )
    reference = None
    for arm in protocol.ARMS:
        arm_root = root / protocol.ARM_DIRECTORY_NAMES[arm]
        with (arm_root / "run_manifest.json").open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        with (arm_root / "metrics.json").open("r", encoding="utf-8") as stream:
            metrics = json.load(stream)
        _require(manifest.get("arm") == arm, "arm identity mismatch")
        identity = tuple(manifest.get(name) for name in frozen_fields)
        if reference is None:
            reference = identity
        _require(identity == reference, "three arms do not share frozen inputs/init")
        _require(
            metrics.get("seal_verified_before_full_gt_load") is True
            and metrics.get("full_gt_used_for_training") is False,
            "metrics GT-firewall mismatch",
        )
        manifests[arm] = manifest
        records[arm] = metrics["metrics"]

    true_action = manifests["TRUE_U"]["action_provenance"]
    uniform_action = manifests["UNIFORM"]["action_provenance"]
    shared_action_fields = (
        "initial_model_sha256", "initial_checkpoint_sha256",
        "q_local_logical_sha256", "q_aligned_logical_sha256",
        "M_v_logical_sha256", "true_u_cycle_logical_sha256",
        "y_gen_logical_sha256", "pred_relation_logical_sha256",
        "balance_logical_sha256",
    )
    _require(
        all(true_action[name] == uniform_action[name] for name in shared_action_fields)
        and true_action["selected_u_cycle_logical_sha256"]
        != uniform_action["selected_u_cycle_logical_sha256"],
        "TRUE_U/UNIFORM single-variable action contract mismatch",
    )
    deltas = {}
    for metric in ("acc", "nmi", "ari"):
        deltas[metric] = {
            "relation_UNIFORM_minus_BASE": float(
                records["UNIFORM"][metric] - records["BASE"][metric]
            ),
            "utility_TRUE_U_minus_UNIFORM": float(
                records["TRUE_U"][metric] - records["UNIFORM"][metric]
            ),
            "full_TRUE_U_minus_BASE": float(
                records["TRUE_U"][metric] - records["BASE"][metric]
            ),
        }
    diagnostic_case = _diagnostic_case({
        arm: records[arm]["acc"] for arm in protocol.ARMS
    })
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump({
            "schema": "paper-msrc-p0-a2-three-arm-summary-v1",
            "dataset": protocol.DATASET,
            "arms": records,
            "deltas": deltas,
            "diagnostic_case_by_ACC": diagnostic_case,
            "absolute_threshold_used": False,
            "selection_or_tuning_performed": False,
            "method_modified": False,
            "formal_result": False,
        }, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(
        summarize(root_dir=args.root_dir, output_path=args.output_path),
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
