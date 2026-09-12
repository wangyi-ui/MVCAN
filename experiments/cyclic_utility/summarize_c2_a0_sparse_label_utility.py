"""Apply the frozen C2-A0 multi-seed gate to three completed seed bundles."""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import (
    c2_a0_sparse_label_utility_protocol as c2,
)
from experiments.cyclic_utility import (
    evaluate_c2_a0_sparse_label_utility as evaluate,
)


STAGE = c2.STAGE
REQUIRED_SEED_FILES = evaluate.PER_SEED_OUTPUT_FILES
MULTISEED_OUTPUT_FILES = (
    "c2_multiseed_summary.json",
    "c2_multiseed_decision.json",
    "c2_multiseed_audit.json",
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
        / ("c2_a0_sparse_label_utility_seed" + str(active_seed))
    )


def load_seed_bundle(seed, seed_dir):
    """Validate one completed C2 seed without loading GT or C0 artifacts."""
    active_seed = c2.validate_seed(seed)
    root = _resolve(seed_dir)
    _require(root.is_dir(), "C2-A0 seed directory is missing")
    paths = {name: root / name for name in REQUIRED_SEED_FILES}
    _require(all(path.is_file() for path in paths.values()), "C2-A0 seed output schema is incomplete")
    seal = evaluate.read_json(paths[evaluate.CALIBRATION_SEAL_NAME])
    strata = evaluate.read_json(paths["confidence_strata_audit.json"])
    directional = evaluate.read_json(paths["c2_metrics_by_direction.json"])
    summary = evaluate.read_json(paths["c2_seed_summary.json"])
    audit = evaluate.read_json(paths["c2_audit.json"])
    _require(
        seal.get("stage") == strata.get("stage") == audit.get("stage") == STAGE
        and seal.get("seed") == strata.get("seed") == audit.get("seed") == active_seed
        and summary.get("stage") == STAGE
        and summary.get("seed") == active_seed,
        "C2-A0 seed identity mismatch",
    )
    _require(
        evaluate.file_sha256(paths[evaluate.PRE_GT_NPZ_NAME]) == seal.get("pre_gt_npz_file_sha256")
        and audit.get("calibration_seal") == seal
        and audit.get("calibration_seal_file_sha256")
        == evaluate.file_sha256(paths[evaluate.CALIBRATION_SEAL_NAME])
        and seal.get("NPZ_saved_fsynced_reloaded_hash_verified") is True
        and audit.get("durable_calibration_seal_verified_before_GT") is True,
        "C2-A0 seed durable-seal boundary mismatch",
    )
    false_boundaries = (
        "GT_loaded_before_calibration_seal",
        "c0_audit_loaded_before_calibration_seal",
        "Bridge_correctness_loaded_before_calibration_seal",
        "corruption_mask_loaded",
        "p_gen_loaded",
        "recovered_p_gen_loaded",
        "model_forward_called",
        "training_performed",
        "backward_called",
        "optimizer_created",
        "optimizer_step_called",
        "C_conf_used_for_calibration",
        "C2_specific_GT_mapping_fit",
        "direction_specific_mapping_fit",
        "bin_specific_mapping_fit",
    )
    _require(
        all(audit.get(name) is False for name in false_boundaries)
        and seal.get("GT_loaded_before_calibration_seal") is False
        and seal.get("c0_audit_loaded_before_calibration_seal") is False
        and seal.get("Bridge_correctness_loaded_before_calibration_seal") is False
        and audit.get("GT_loaded_after_calibration_seal") is True
        and audit.get("c0_audit_loaded_after_calibration_seal") is True
        and audit.get("Bridge_correctness_built_after_calibration_seal") is True,
        "C2-A0 seed leakage/order boundary mismatch",
    )
    _require(
        seal.get("FORMAL_C2_USES_PGEN") is False
        and seal.get("FORMAL_C2_USES_YGEN_VOTE_STATE") is True
        and seal.get("GT_PRESEAL_ISOLATION") is True
        and seal.get("TRAINING_PATH_PRESENT") is False
        and seal.get("label_count") == c2.LABEL_COUNT
        and seal.get("unlabeled_evaluation_count") == c2.UNLABELED_EVAL_COUNT
        and audit.get("evaluation_only_unlabeled") is True
        and audit.get("labeled_evaluation_intersection_count") == 0,
        "C2-A0 seed formal-policy boundary mismatch",
    )
    _require(
        strata.get("confidence_bin_count") == c2.CONFIDENCE_BIN_COUNT
        and strata.get("evaluation_sample_count") == c2.UNLABELED_EVAL_COUNT
        and strata.get("GT_used_for_stratification") is False
        and strata.get("stable_sort_key") == ["C_conf", "sample_id"]
        and len(strata.get("directions", ())) == c2.DIRECTION_COUNT
        and directional.get("evaluation_sample_count") == c2.UNLABELED_EVAL_COUNT
        and directional.get("labeled_evaluation_intersection_count") == 0,
        "C2-A0 seed matched-confidence schema mismatch",
    )
    recomputed = c2.build_seed_summary(active_seed, directional)
    _require(summary == recomputed, "C2-A0 seed summary does not match frozen gate helper")
    return {
        "seed": active_seed,
        "directory": _display(root),
        "summary": summary,
        "file_sha256": {name: evaluate.file_sha256(path) for name, path in paths.items()},
    }


def summarize(seed20_dir, seed30_dir, seed50_dir, output_dir):
    directories = OrderedDict(((20, seed20_dir), (30, seed30_dir), (50, seed50_dir)))
    _require(tuple(directories) == c2.SEEDS, "C2-A0 multi-seed set/order mismatch")
    bundles = OrderedDict(
        (seed, load_seed_bundle(seed, directory)) for seed, directory in directories.items()
    )
    summaries = OrderedDict((seed, bundles[seed]["summary"]) for seed in c2.SEEDS)
    aggregate = c2.build_multiseed_decision(summaries)
    output_root = _resolve(output_dir)
    _require(not output_root.exists(), "refusing to overwrite C2-A0 multi-seed output")
    audit = {
        "stage": STAGE,
        "seeds": list(c2.SEEDS),
        "dataset": evaluate.DATASET,
        "N": c2.SAMPLE_NUM,
        "K": c2.CLASS_NUM,
        "direction_count": c2.DIRECTION_COUNT,
        "label_count": c2.LABEL_COUNT,
        "unlabeled_evaluation_count_per_seed": c2.UNLABELED_EVAL_COUNT,
        "C2_A0_MULTISEED_AUDIT_PASS": True,
        "all_three_seed_audits_pass": True,
        "seed_directories": {str(seed): bundles[seed]["directory"] for seed in c2.SEEDS},
        "seed_file_sha256": {str(seed): bundles[seed]["file_sha256"] for seed in c2.SEEDS},
        "GT_loaded_by_summarizer": False,
        "c0_audit_loaded_by_summarizer": False,
        "corruption_mask_loaded": False,
        "p_gen_loaded": False,
        "recovered_p_gen_loaded": False,
        "model_forward_called": False,
        "training_performed": False,
        "backward_called": False,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "post_hoc_gate_change": False,
        "aggregate_logic_source": (
            "c2_a0_sparse_label_utility_protocol.build_multiseed_decision"
        ),
    }
    records = OrderedDict((
        ("c2_multiseed_summary.json", aggregate["summary"]),
        ("c2_multiseed_decision.json", aggregate["decision"]),
        ("c2_multiseed_audit.json", audit),
    ))
    _require(tuple(records) == MULTISEED_OUTPUT_FILES, "C2-A0 multi-seed output schema mismatch")
    output_root.mkdir(parents=True, exist_ok=False)
    for name, record in records.items():
        evaluate.write_json_durable(output_root / name, record)
    evaluate._fsync_directory(output_root)
    _require(
        all(evaluate.read_json(output_root / name) == record for name, record in records.items()),
        "C2-A0 multi-seed durable reload mismatch",
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
