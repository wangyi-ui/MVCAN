"""Post-seal-only MSRC P0 evaluation wrapper."""

import argparse
import json
from pathlib import Path

from release_core.runtime import SealedPredictionPaths, evaluate_postseal

from . import msrc_p0_protocol as protocol
from .input_artifacts import file_sha256


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def evaluate(*, arm, run_dir, dataset_path):
    active_arm = protocol.validate_arm(arm)
    root = Path(run_dir)
    _require(root.resolve() == protocol.default_output_dir(active_arm).resolve(),
             "run directory does not match the frozen arm output path")
    path = Path(dataset_path)
    _require(path.is_file(), "explicit MSRC-v1 dataset path does not exist")
    _require(path.resolve() == protocol.DATASET_PATH.resolve(),
             "dataset path must be authoritative; no fallback is allowed")
    _require(file_sha256(path) == protocol.DATASET_SHA256,
             "MSRC-v1 dataset SHA256 mismatch")
    manifest_path = root / "run_manifest.json"
    _require(manifest_path.is_file(), "run manifest is missing")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    _require(
        manifest.get("arm") == active_arm
        and manifest.get("GT-loaded-before-seal") is False,
        "run manifest arm or GT-firewall mismatch",
    )
    sealed = SealedPredictionPaths(
        root / "pre_gt_bundle.npz",
        root / "pre_gt_audit.json",
        root / "pre_gt_seal.json",
    )
    metrics_path = root / "metrics.json"
    _require(not metrics_path.exists(), "refusing to overwrite metrics output")
    return evaluate_postseal(sealed, path, output_path=metrics_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=protocol.ARMS)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = evaluate(arm=args.arm, run_dir=args.run_dir, dataset_path=args.dataset_path)
    print(json.dumps({"arm": args.arm, "acc": result.acc, "nmi": result.nmi,
                      "ari": result.ari}, sort_keys=True))


if __name__ == "__main__":
    main()
