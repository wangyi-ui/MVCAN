"""Summarize the three frozen Bridge-P0 seed diagnostics."""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import (
    bridge_p0_residual_utility_protocol as bridge,
)
from experiments.cyclic_utility import (
    evaluate_bridge_p0_residual_utility as evaluate,
)


STAGE = "Bridge-P0"
REQUIRED_SEED_FILES = evaluate.PER_SEED_OUTPUT_FILES
MULTISEED_OUTPUT_FILES = (
    "bridge_multiseed_summary.json",
    "bridge_multiseed_decision.json",
    "bridge_multiseed_audit.json",
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


def load_seed_bundle(seed, seed_dir):
    """Validate one completed seed without loading GT or model artifacts."""
    active_seed = bridge.validate_seed(seed)
    root = _resolve(seed_dir)
    _require(root.is_dir(), "Bridge-P0 seed directory is missing")
    paths = {name: root / name for name in REQUIRED_SEED_FILES}
    _require(
        all(path.is_file() for path in paths.values()),
        "Bridge-P0 seed output schema is incomplete",
    )
    records = {
        name: evaluate.read_json(path) for name, path in paths.items()
    }
    seal = records["bridge_input_seal.json"]
    audit = records["bridge_audit.json"]
    strata = records["confidence_strata_audit.json"]
    conditional = records["conditional_auc_by_direction.json"]
    action = records["action_lift_by_direction.json"]
    summary = records["bridge_seed_summary.json"]

    _require(
        seal.get("stage") == audit.get("stage") == STAGE
        and seal.get("seed") == audit.get("seed") == active_seed
        and summary.get("seed") == active_seed
        and strata.get("seed") == conditional.get("seed")
        == action.get("seed") == active_seed,
        "Bridge-P0 seed identity mismatch",
    )
    _require(
        seal.get("dataset") == audit.get("dataset") == evaluate.DATASET
        and seal.get("N") == audit.get("N") == bridge.SAMPLE_NUM
        and seal.get("V") == audit.get("V") == bridge.VIEW_NUM
        and seal.get("K") == audit.get("K") == bridge.CLASS_NUM
        and seal.get("direction_count")
        == audit.get("direction_count")
        == bridge.DIRECTION_COUNT
        and audit.get("confidence_bin_count")
        == bridge.CONFIDENCE_BIN_COUNT,
        "Bridge-P0 seed dataset dimensions mismatch",
    )
    _require(
        audit.get("BRIDGE_AUDIT_PASS") is True
        and audit.get("bridge_input_seal") == seal
        and audit.get("bridge_input_seal_reloaded_after_GT_pass") is True
        and seal.get("inputs_verified_before_GT") is True
        and seal.get("input_seal_saved_before_GT") is True
        and seal.get("input_seal_fsynced_before_GT") is True
        and seal.get("GT_loaded_before_bridge_input_seal") is False
        and audit.get("GT_loaded_before_bridge_input_seal") is False
        and audit.get("GT_loaded_after_bridge_input_seal") is True
        and audit.get("full_GT", {}).get(
            "loaded_after_bridge_input_seal"
        ) is True,
        "Bridge-P0 seed seal/GT ordering mismatch",
    )
    false_boundaries = (
        "training_entered",
        "optimizer_created",
        "backward_called",
        "model_loaded_for_training",
        "R_loaded",
        "corruption_mask_loaded",
        "sparse_labels_loaded",
        "sparse_label_ids_loaded",
        "sparse_label_targets_loaded",
        "B7_artifact_loaded",
        "new_utility_constructed",
        "VSA_modified",
        "C0_modified",
    )
    _require(
        all(audit.get(key) is False for key in false_boundaries)
        and audit.get("no_per_direction_GT_mapping") is True
        and audit.get("no_per_bin_GT_mapping") is True
        and audit.get("frozen_C0_global_mapping", {}).get(
            "one_frozen_global_mapping_only"
        ) is True
        and audit.get("frozen_C0_global_mapping", {}).get(
            "Hungarian_fit_in_Bridge"
        ) is False,
        "Bridge-P0 seed leakage/mapping boundary mismatch",
    )
    _require(
        tuple(seal.get("required_arrays", ())) == evaluate.REQUIRED_ARRAYS
        and seal.get("C0_E1_same_seed_lineage_pass") is True
        and seal.get("E1_training_seed") == active_seed
        and seal.get("fixed_weak_quality_condition")
        == bridge.WEAK_QUALITY_CONDITION,
        "Bridge-P0 seed input lineage mismatch",
    )
    expected_source_hashes = {
        "protocol": evaluate.file_sha256(Path(bridge.__file__)),
        "evaluator": evaluate.file_sha256(Path(evaluate.__file__)),
    }
    _require(
        seal.get("bridge_source_sha256") == expected_source_hashes
        and audit.get("bridge_source_sha256") == expected_source_hashes
        and audit.get("source_integrity", {}).get(
            "all_existing_scientific_sources_unchanged_pass"
        ) is True,
        "Bridge-P0 protocol/evaluator source provenance mismatch",
    )
    _require(
        strata.get("confidence_bin_count")
        == bridge.CONFIDENCE_BIN_COUNT
        and strata.get("GT_used_for_stratification") is False
        and len(strata.get("directions", ())) == bridge.DIRECTION_COUNT
        and conditional.get("shared_correctness_derived_valid_bin_mask")
        is True
        and len(conditional.get("directions", ()))
        == bridge.DIRECTION_COUNT
        and action.get("matched_within_confidence_quintile") is True
        and len(action.get("directions", ())) == bridge.DIRECTION_COUNT,
        "Bridge-P0 seed diagnostic schema mismatch",
    )
    recomputed_summary = bridge.build_seed_summary(
        active_seed,
        conditional,
        action,
    )
    _require(
        summary == recomputed_summary,
        "Bridge-P0 seed summary does not match frozen decision helper",
    )
    return {
        "seed": active_seed,
        "directory": _display(root),
        "records": records,
        "summary": summary,
        "bridge_source_sha256": expected_source_hashes,
        "fixed_weak_quality": {
            "condition": seal["fixed_weak_quality_condition"],
            "feature_path": seal["fixed_feature_path"],
            "feature_file_sha256": seal["fixed_feature_file_sha256"],
        },
        "file_sha256": {
            name: evaluate.file_sha256(path)
            for name, path in paths.items()
        },
    }


def summarize(
    seed20_dir,
    seed30_dir,
    seed50_dir,
    output_dir,
):
    """Apply only the frozen 2-of-3 Bridge-P0 aggregate decision."""
    directories = OrderedDict((
        (20, seed20_dir),
        (30, seed30_dir),
        (50, seed50_dir),
    ))
    _require(tuple(directories) == bridge.SEEDS, "Bridge-P0 seed set mismatch")
    bundles = OrderedDict(
        (seed, load_seed_bundle(seed, directory))
        for seed, directory in directories.items()
    )
    source_records = {
        tuple(sorted(bundle["bridge_source_sha256"].items()))
        for bundle in bundles.values()
    }
    weak_quality_records = {
        tuple(sorted(bundle["fixed_weak_quality"].items()))
        for bundle in bundles.values()
    }
    _require(
        len(source_records) == 1,
        "Bridge-P0 source differs across seeds",
    )
    _require(
        len(weak_quality_records) == 1,
        "Bridge-P0 weak-quality realization differs across seeds",
    )
    seed_summaries = OrderedDict(
        (seed, bundles[seed]["summary"]) for seed in bridge.SEEDS
    )
    aggregate = bridge.build_multiseed_decision(seed_summaries)
    output_root = _resolve(output_dir)
    _require(
        not output_root.exists(),
        "refusing to overwrite Bridge-P0 multi-seed output",
    )
    audit = {
        "stage": STAGE,
        "seeds": list(bridge.SEEDS),
        "dataset": evaluate.DATASET,
        "N": bridge.SAMPLE_NUM,
        "V": bridge.VIEW_NUM,
        "K": bridge.CLASS_NUM,
        "direction_count": bridge.DIRECTION_COUNT,
        "confidence_bin_count": bridge.CONFIDENCE_BIN_COUNT,
        "BRIDGE_MULTISEED_AUDIT_PASS": True,
        "all_three_seed_audits_pass": True,
        "same_bridge_protocol_source_all_seeds": True,
        "bridge_source_sha256": bundles[20]["bridge_source_sha256"],
        "same_fixed_weak_quality_realization_all_seeds": True,
        "fixed_weak_quality": bundles[20]["fixed_weak_quality"],
        "same_dataset_dimensions_all_seeds": True,
        "seed_directories": {
            str(seed): bundles[seed]["directory"] for seed in bridge.SEEDS
        },
        "seed_file_sha256": {
            str(seed): bundles[seed]["file_sha256"]
            for seed in bridge.SEEDS
        },
        "training_entered": False,
        "optimizer_created": False,
        "backward_called": False,
        "model_loaded_for_training": False,
        "R_loaded": False,
        "corruption_mask_loaded": False,
        "sparse_labels_loaded": False,
        "sparse_label_ids_loaded": False,
        "sparse_label_targets_loaded": False,
        "B7_artifact_loaded": False,
        "GT_loaded_by_summarizer": False,
        "all_seed_GT_loaded_only_after_bridge_input_seal": True,
        "new_utility_constructed": False,
        "post_hoc_threshold_change": False,
        "aggregate_logic_source": (
            "bridge_p0_residual_utility_protocol.build_multiseed_decision"
        ),
        "required_output_files": list(MULTISEED_OUTPUT_FILES),
    }
    records = OrderedDict((
        ("bridge_multiseed_summary.json", aggregate["summary"]),
        ("bridge_multiseed_decision.json", aggregate["decision"]),
        ("bridge_multiseed_audit.json", audit),
    ))
    _require(
        tuple(records) == MULTISEED_OUTPUT_FILES,
        "Bridge-P0 multi-seed output schema mismatch",
    )
    output_root.mkdir(parents=True, exist_ok=False)
    for name, record in records.items():
        evaluate.write_json(output_root / name, record)
    print("FINAL_DECISION=" + aggregate["decision"]["final_decision"])
    print("Saved: " + _display(output_root))
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed20-dir", required=True)
    parser.add_argument("--seed30-dir", required=True)
    parser.add_argument("--seed50-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summarize(
        seed20_dir=args.seed20_dir,
        seed30_dir=args.seed30_dir,
        seed50_dir=args.seed50_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
