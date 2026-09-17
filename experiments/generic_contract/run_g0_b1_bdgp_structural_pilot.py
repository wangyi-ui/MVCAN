"""CLI for the sealed-input, pre-GT G0-B1 BDGP structural pilot."""

import argparse
import json
import os


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _thread_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_name] = "1"

from .generic_final_core_adapter import BDGP_RUNTIME_SPEC, run_structural_pilot


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_structural_pilot(
        args.input_dir,
        args.output_dir,
        args.training_seed,
        args.device,
        runtime_spec=BDGP_RUNTIME_SPEC,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
