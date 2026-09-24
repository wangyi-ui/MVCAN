"""Explicit P1-A3 stage CLI; this module never selects hyperparameters."""

import argparse

from . import p1_a0_formal_protocol as protocol


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=tuple(item.name for item in protocol.FORMAL_DATASETS))
    parser.add_argument("--training-seed", required=True, type=int, choices=protocol.FORMAL_TRAINING_SEEDS)
    parser.add_argument("--device", required=True)
    parser.add_argument("--stage", required=True, choices=("inputs", "initialization", "actions", "all"))
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.plan_only:
        raise RuntimeError("P1_A3_REAL_MATERIALIZATION_REQUIRES_EXPLICIT_NEXT_STAGE_AUTHORIZATION")
    print("[P1-A3] PLAN VALID", flush=True)
    return 0


if __name__ == "__main__":
    main()
