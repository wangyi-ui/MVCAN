"""Explicit P1-A3 stage CLI; this module never selects hyperparameters."""

import argparse
from pathlib import Path

from . import p1_a0_formal_protocol as protocol
from . import p1_a1_native_preparation as initialization
from . import p1_a1_action_materialization as actions
from .p1_a3_input_materialization import materialize_caltech_inputs


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
    slug = args.dataset.lower().replace("-", "").replace("_", "")
    root = Path("outputs/paper/formal")
    inputs = root / "inputs" / slug
    init = root / "preparation" / slug / ("seed" + str(args.training_seed))
    action = root / "actions" / slug / ("seed" + str(args.training_seed)) / "true_action_state"
    if args.plan_only:
        print("[P1-A3] PLAN VALID", flush=True)
        return 0
    if args.dataset != "Caltech-6V":
        raise RuntimeError("FORMAL_DATASET_SOURCE_UNRESOLVED")
    if args.stage in ("inputs", "all"):
        print("[P1-A3] INPUT START", flush=True)
        materialize_caltech_inputs(inputs)
        print("[P1-A3] input seal complete", flush=True)
    if args.stage in ("initialization", "all"):
        print("[P1-A3] NATIVE PREPARATION START", flush=True)
        initialization.build_initialization(dataset=args.dataset, training_seed=args.training_seed, feature_path=inputs / "features.npz", output_dir=init, device=args.device)
    if args.stage in ("actions", "all"):
        print("[P1-A3] ACTION START", flush=True)
        actions.build_true_action(dataset=args.dataset, training_seed=args.training_seed, carrier_state_path=init / "carrier_state.npz", split_path=inputs / "sparse_split.npz", output_dir=action)
    print("[P1-A3] DONE", flush=True)
    return 0


if __name__ == "__main__":
    main()
