"""CLI for D5-R1; plan-only by default and training needs --execute."""
import argparse
import json
from . import d5_r1_refresh_cadence as diagnostic

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-seed", required=True, type=int, choices=diagnostic.SEEDS)
    parser.add_argument("--device", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--full-gt-path")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args(argv)
    if args.evaluate and not args.execute:
        raise RuntimeError("D5_R1_EVALUATION_REQUIRES_EXECUTE")
    if args.evaluate and not args.full_gt_path:
        raise RuntimeError("D5_R1_FULL_GT_PATH_REQUIRED_AFTER_SEAL")
    if not args.execute:
        v1, runtime = diagnostic.diagnostic_runtime(args.training_seed, args.device)
        print(json.dumps({"status": "PLAN ONLY", "dataset": diagnostic.DATASET, "training_seed": args.training_seed, "v1_refresh_interval": v1.refresh_interval, "diagnostic_refresh_interval": runtime.refresh_interval, "refresh_epochs": list(diagnostic.refresh_epochs(runtime)), "full_gt_loaded": False}, sort_keys=True), flush=True)
        return 0
    sealed, record = diagnostic.run(args.training_seed, args.device)
    print(json.dumps({"status": "PRE_GT SEALED", "bundle": str(sealed.bundle), "audit": record}, sort_keys=True), flush=True)
    if args.evaluate:
        metrics = diagnostic.evaluate_after_seal(args.training_seed, args.full_gt_path)
        print(json.dumps({"status": "POSTSEAL METRICS", "metrics": metrics["metrics"]}, sort_keys=True), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
