"""Summarize the three fixed MSRC P0 arms without selection or tuning."""

import argparse
import json
from pathlib import Path

from . import msrc_p0_protocol as protocol


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def summarize(*, root_dir, output_path):
    root = Path(root_dir)
    target = Path(output_path)
    _require(not target.exists(), "refusing to overwrite summary output")
    records = {}
    frozen_fields = (
        "dataset", "dataset_sha256", "training_seed", "label_seed",
        "weak_quality_seed", "snr_db", "epochs", "batch_size",
        "learning_rate", "refresh_interval", "initial_model_sha256",
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
        _require(identity == reference, "cross-arm frozen configuration mismatch")
        _require(
            metrics.get("seal_verified_before_full_gt_load") is True
            and metrics.get("full_gt_used_for_training") is False,
            "metrics GT-firewall mismatch",
        )
        records[arm] = metrics["metrics"]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump({
            "schema": "paper-msrc-p0-three-arm-summary-v1",
            "dataset": protocol.DATASET,
            "arms": records,
            "selection_or_tuning_performed": False,
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
    print(json.dumps(summarize(root_dir=args.root_dir, output_path=args.output_path),
                     sort_keys=True))


if __name__ == "__main__":
    main()
