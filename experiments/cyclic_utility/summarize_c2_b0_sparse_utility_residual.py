"""Summarize exactly the three preregistered C2-B0 seed bundles."""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import c2_b0_sparse_utility_residual_protocol as c2
from experiments.cyclic_utility import evaluate_c2_b0_sparse_utility_residual as evaluate


STAGE = c2.STAGE
REQUIRED_SEED_FILES = evaluate.PER_SEED_OUTPUT_FILES
MULTISEED_OUTPUT_FILES = (
    "c2_b0_multiseed_summary.json",
    "c2_b0_multiseed_decision.json",
    "c2_b0_multiseed_audit.json",
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _display(path):
    value = _resolve(path).resolve()
    try:
        return str(value.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(value)


def default_seed_dir(seed):
    active_seed = c2.validate_seed(seed)
    return (
        REPOSITORY_ROOT
        / "outputs/cyclic_utility"
        / ("c2_b0_sparse_utility_residual_seed" + str(active_seed))
    )


def load_seed_bundle(seed, seed_dir):
    active_seed = c2.validate_seed(seed)
    root = _resolve(seed_dir)
    _require(root.is_dir(), "C2-B0 seed directory is missing")
    paths = {name: root / name for name in REQUIRED_SEED_FILES}
    _require(
        all(path.is_file() for path in paths.values()),
        "C2-B0 seed output schema is incomplete",
    )
    seal = evaluate.read_json(paths[evaluate.RESIDUAL_SEAL_NAME])
    metrics = evaluate.read_json(paths["c2_b0_metrics_by_direction.json"])
    summary = evaluate.read_json(paths["c2_b0_seed_summary.json"])
    audit = evaluate.read_json(paths["c2_b0_audit.json"])
    strata = evaluate.read_json(paths["c2_b0_confidence_strata_audit.json"])
    _require(
        seal.get("stage") == metrics.get("stage") == summary.get("stage")
        == audit.get("stage") == strata.get("stage") == STAGE
        and seal.get("seed") == metrics.get("seed") == summary.get("seed")
        == audit.get("seed") == strata.get("seed") == active_seed,
        "C2-B0 seed identity mismatch",
    )
    _require(
        evaluate.file_sha256(paths[evaluate.PRE_GT_NPZ_NAME])
        == seal.get("pre_gt_npz_file_sha256")
        and audit.get("residual_seal") == seal
        and audit.get("residual_seal_file_sha256")
        == evaluate.file_sha256(paths[evaluate.RESIDUAL_SEAL_NAME])
        and seal.get("NPZ_saved_fsynced_reloaded_hash_verified") is True
        and audit.get("durable_residual_seal_verified_before_GT") is True,
        "C2-B0 seed durable-seal boundary mismatch",
    )
    false_boundaries = (
        "GT_loaded_before_residual_seal",
        "c0_audit_loaded_before_residual_seal",
        "Bridge_correctness_loaded_before_residual_seal",
        "corruption_mask_loaded",
        "training_performed",
        "backward_called",
        "optimizer_created",
        "optimizer_step_called",
        "model_forward_called",
        "C_conf_used_for_residual_propagation",
        "C2_specific_GT_mapping_fit",
        "direction_specific_mapping_fit",
        "bin_specific_mapping_fit",
    )
    _require(
        all(audit.get(name) is False for name in false_boundaries)
        and audit.get("GT_loaded_after_residual_seal") is True
        and audit.get("c0_audit_loaded_after_residual_seal") is True
        and audit.get("Bridge_correctness_built_after_residual_seal") is True
        and audit.get("evaluation_only_unlabeled") is True
        and audit.get("evaluation_sample_count") == c2.UNLABELED_EVAL_COUNT,
        "C2-B0 seed leakage/evaluation boundary mismatch",
    )
    recomputed = c2.build_seed_summary(
        active_seed, metrics["residual"], metrics["calibration"]
    )
    _require(summary == recomputed, "C2-B0 seed summary differs from frozen gate helper")
    return {
        "seed": active_seed,
        "directory": _display(root),
        "summary": summary,
        "file_sha256": {
            name: evaluate.file_sha256(path) for name, path in paths.items()
        },
    }


def summarize(seed20_dir, seed30_dir, seed50_dir, output_dir):
    directories = OrderedDict(((20, seed20_dir), (30, seed30_dir), (50, seed50_dir)))
    _require(tuple(directories) == c2.SEEDS, "C2-B0 multi-seed set/order mismatch")
    bundles = OrderedDict(
        (seed, load_seed_bundle(seed, directories[seed])) for seed in c2.SEEDS
    )
    summaries = OrderedDict((seed, bundles[seed]["summary"]) for seed in c2.SEEDS)
    aggregate = c2.build_multiseed_decision(summaries)
    output_root = _resolve(output_dir)
    _require(not output_root.exists(), "refusing to overwrite C2-B0 multi-seed output")
    audit = {
        "stage": STAGE,
        "seeds": list(c2.SEEDS),
        "dataset": evaluate.DATASET,
        "label_count": c2.LABEL_COUNT,
        "unlabeled_evaluation_count_per_seed": c2.UNLABELED_EVAL_COUNT,
        "all_three_seed_audits_pass": True,
        "seed_directories": {
            str(seed): bundles[seed]["directory"] for seed in c2.SEEDS
        },
        "seed_file_sha256": {
            str(seed): bundles[seed]["file_sha256"] for seed in c2.SEEDS
        },
        "GT_loaded_by_summarizer": False,
        "c0_audit_loaded_by_summarizer": False,
        "Bridge_correctness_loaded_by_summarizer": False,
        "corruption_mask_loaded": False,
        "training_performed": False,
        "backward_called": False,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "model_forward_called": False,
        "post_hoc_gate_change": False,
    }
    records = OrderedDict((
        ("c2_b0_multiseed_summary.json", aggregate["summary"]),
        ("c2_b0_multiseed_decision.json", aggregate["decision"]),
        ("c2_b0_multiseed_audit.json", audit),
    ))
    _require(tuple(records) == MULTISEED_OUTPUT_FILES, "C2-B0 multi-seed schema mismatch")
    output_root.mkdir(parents=True, exist_ok=False)
    for name, record in records.items():
        evaluate.write_json_durable(output_root / name, record)
    evaluate._fsync_directory(output_root)
    _require(
        all(evaluate.read_json(output_root / name) == record for name, record in records.items()),
        "C2-B0 multi-seed durable reload mismatch",
    )
    print("FINAL_DECISION=" + aggregate["decision"]["final_decision"])
    print("Saved: " + _display(output_root))
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed20-dir", default=str(default_seed_dir(20)))
    parser.add_argument("--seed30-dir", default=str(default_seed_dir(30)))
    parser.add_argument("--seed50-dir", default=str(default_seed_dir(50)))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summarize(args.seed20_dir, args.seed30_dir, args.seed50_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
