"""Evaluate C3-A0 behind a durable pre-GT semantic-action seal.

Formal execution is exposed only for a later separately authorized phase.
This implementation phase is exercised through pure and temporary-path tests.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import c3_a0_utility_conditioned_action_granularity_protocol as c3
from experiments.cyclic_utility import evaluate_c2_b0_sparse_utility_residual as b0_evaluator
from experiments.cyclic_utility import evaluate_c2_c0_directional_consensus_utility as c0_evaluator
from experiments.cyclic_utility import evaluate_c2_c2_a0_action_support_diagnostic as c2_c2_evaluator
from weak_quality import ndarray_sha256


STAGE = c3.STAGE
DATASET = "Caltech-6V"
FROZEN_PARENT_MANIFESTS = OrderedDict((
    (
        "C2_B0",
        "experiment_freeze/c2_b0_multiseed_fail_closed_20260908/"
        "source_sha256.txt",
    ),
    (
        "C2_C0",
        "experiment_freeze/c2_c0_multiseed_fail_closed_20260908/"
        "source_sha256.txt",
    ),
    (
        "C2_C1",
        "experiment_freeze/c2_c1_multiseed_fail_closed_20260908/"
        "source_sha256.txt",
    ),
    (
        "C2_C2_A0",
        "experiment_freeze/c2_c2_a0_multiseed_fail_closed_20260912/"
        "source_sha256.txt",
    ),
))
PRE_GT_NPZ_NAME = "c3_a0_action_pre_gt.npz"
ACTION_SEAL_NAME = "c3_a0_action_seal.json"
PRE_GT_ARRAY_NAMES = (
    "sample_ids",
    "unlabeled_ids",
    "labeled_ids",
    "labeled_targets",
    "shuffled_labeled_targets",
    "U_cycle",
    "y_gen",
    "R",
    "class_pred_true",
    "class_pred_shuffle",
    "PredRelation_true",
    "PredRelation_shuffle",
    "relation_balance_weights_true",
    "relation_balance_weights_shuffle",
    "utility_quintile_assignment",
    "P_true",
    "P_shuffle",
    "P_candidate_true",
    "P_candidate_shuffle",
)
PER_SEED_OUTPUT_FILES = (
    PRE_GT_NPZ_NAME,
    ACTION_SEAL_NAME,
    "c3_a0_class_metrics.json",
    "c3_a0_relation_metrics.json",
    "c3_a0_memory_write_metrics.json",
    "c3_a0_seed_summary.json",
    "c3_a0_audit.json",
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


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _json_native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            key: _json_native(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    return value


def write_json_durable(path, value):
    with open(path, "x", encoding="utf-8") as output_file:
        json.dump(
            _json_native(value),
            output_file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def _verify_json_durable_readback(path, original_record):
    _require(
        read_json(path) == _json_native(original_record),
        "C3-A0 durable JSON reload mismatch",
    )


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_frozen_parent_sources():
    manifests = OrderedDict()
    for stage, name in FROZEN_PARENT_MANIFESTS.items():
        audit = b0_evaluator.verify_sha256_manifest(name)
        _require(
            audit.get("all_entries_exact_match") is True,
            "C3-A0 frozen parent SHA256 verification failed: " + stage,
        )
        manifests[stage] = audit
    return {
        "manifests": dict(manifests),
        "C2_B0_frozen_hashes_pass": True,
        "C2_C0_frozen_hashes_pass": True,
        "C2_C1_frozen_hashes_pass": True,
        "C2_C2_A0_frozen_hashes_pass": True,
        "all_frozen_parent_hashes_pass": True,
    }


@dataclass(frozen=True)
class C3A0Paths:
    c0_artifact_path: Path
    c0_seal_path: Path
    label_split_dir: Path
    full_gt_path: Path
    output_dir: Path


def default_paths(seed, output_dir=None):
    active_seed = c3.validate_seed(seed)
    source = c0_evaluator.default_paths(active_seed)
    target = (
        REPOSITORY_ROOT
        / "outputs/cyclic_utility"
        / (
            "c3_a0_utility_conditioned_action_granularity_seed"
            + str(active_seed)
        )
        if output_dir is None else _resolve(output_dir)
    )
    return C3A0Paths(
        c0_artifact_path=source.c0_artifact_path,
        c0_seal_path=source.c0_seal_path,
        label_split_dir=source.label_split_dir,
        full_gt_path=source.full_gt_path,
        output_dir=target,
    )


def load_c0_inputs_before_gt(seed, c0_artifact_path, c0_seal_path):
    arrays, provenance = c0_evaluator.load_c0_inputs_before_gt(
        seed, c0_artifact_path, c0_seal_path
    )
    _require(
        provenance["GT_loaded"] is False
        and provenance["c0_audit_loaded"] is False
        and provenance["Bridge_correctness_loaded"] is False
        and provenance["corruption_mask_loaded"] is False,
        "C3-A0 pre-GT loader crossed an oracle boundary",
    )
    return arrays, provenance


def load_fixed_sparse_labels(split_dir):
    split = c0_evaluator.load_fixed_sparse_labels(split_dir)
    validated = c3.validate_fixed_sparse_split(
        split["labeled_sample_ids"],
        split["unlabeled_ids"],
        split["labeled_targets"],
    )
    shuffled = c3.validate_negative_control(
        validated["labeled_targets"],
        split["shuffled_labeled_targets"],
    )
    return {
        **split,
        **validated,
        "shuffled_labeled_targets": shuffled,
    }


def build_pre_gt_bundle(
    seed,
    c0_arrays,
    c0_provenance,
    sparse_split,
    frozen_boundary,
):
    """Construct all three action families without accepting full GT."""
    active_seed = c3.validate_seed(seed)
    _require(
        c0_provenance.get("GT_loaded") is False
        and c0_provenance.get("c0_audit_loaded") is False
        and c0_provenance.get("Bridge_correctness_loaded") is False
        and c0_provenance.get("corruption_mask_loaded") is False,
        "C3-A0 pre-GT bundle received leaked provenance",
    )
    _require(
        frozen_boundary.get("all_frozen_parent_hashes_pass") is True,
        "C3-A0 cannot proceed without frozen parent verification",
    )
    state = c3.build_vote_semantic_state(c0_arrays["y_gen"])
    outputs = c3.build_pre_gt_actions(
        sample_ids=c0_arrays["sample_ids"],
        unlabeled_ids=sparse_split["unlabeled_ids"],
        labeled_ids=sparse_split["labeled_sample_ids"],
        labeled_targets=sparse_split["labeled_targets"],
        shuffled_labeled_targets=sparse_split[
            "shuffled_labeled_targets"
        ],
        U_cycle=c0_arrays["U_cycle"],
        y_gen=c0_arrays["y_gen"],
        R=state,
    )
    arrays = OrderedDict(
        (name, outputs[name]) for name in PRE_GT_ARRAY_NAMES
    )
    _require(
        tuple(arrays) == PRE_GT_ARRAY_NAMES,
        "C3-A0 pre-GT array whitelist mismatch",
    )
    return {
        "seed": active_seed,
        "arrays": arrays,
        "outputs": outputs,
        "C0_provenance": c0_provenance,
        "sparse_split": sparse_split,
        "frozen_boundary": frozen_boundary,
    }


def _array_records(arrays):
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
            "logical_sha256": ndarray_sha256(value),
        }
        for name, value in arrays.items()
    }


def persist_action_bundle(bundle, output_dir):
    """Write, fsync, reload, and logically hash every pre-GT action array."""
    output_root = _resolve(output_dir)
    _require(
        not output_root.exists(),
        "refusing to overwrite C3-A0 seed output",
    )
    arrays = bundle["arrays"]
    _require(
        tuple(arrays) == PRE_GT_ARRAY_NAMES,
        "C3-A0 persistence schema mismatch",
    )
    records = _array_records(arrays)
    output_root.mkdir(parents=True, exist_ok=False)
    npz_path = output_root / PRE_GT_NPZ_NAME
    with open(npz_path, "xb") as output_file:
        np.savez(output_file, **arrays)
        output_file.flush()
        os.fsync(output_file.fileno())
    _fsync_directory(output_root)
    with np.load(npz_path, allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == PRE_GT_ARRAY_NAMES,
            "C3-A0 sealed NPZ field mismatch",
        )
        for name in PRE_GT_ARRAY_NAMES:
            reloaded = np.array(archive[name], copy=True, order="C")
            _require(
                np.array_equal(reloaded, np.asarray(arrays[name]))
                and ndarray_sha256(reloaded)
                == records[name]["logical_sha256"],
                "C3-A0 sealed array reload/hash mismatch: " + name,
            )
    provenance = bundle["C0_provenance"]
    split = bundle["sparse_split"]
    seal = {
        "stage": STAGE,
        "seed": bundle["seed"],
        "dataset": DATASET,
        "N": c3.SAMPLE_NUM,
        "K": c3.CLASS_NUM,
        "direction_count": c3.DIRECTION_COUNT,
        "label_count": c3.LABEL_COUNT,
        "unlabeled_evaluation_count": c3.UNLABELED_EVAL_COUNT,
        "pre_gt_npz_path": _display(npz_path),
        "pre_gt_npz_file_sha256": file_sha256(npz_path),
        "arrays": records,
        "input_C0_NPZ_SHA256": provenance["C0_artifact_file_sha256"],
        "input_C0_prediction_seal_SHA256": provenance[
            "C0_prediction_seal_file_sha256"
        ],
        "label_split_sha256": split["label_split_sha256"],
        "frozen_parent_verification": bundle["frozen_boundary"],
        "true_target_hash": records["labeled_targets"]["logical_sha256"],
        "shuffle_target_hash": records[
            "shuffled_labeled_targets"
        ]["logical_sha256"],
        "NPZ_saved_fsynced_reloaded_hash_verified": True,
        "GT_loaded_before_action_seal": False,
        "C0_audit_loaded_before_action_seal": False,
        "Bridge_loaded_before_action_seal": False,
        "GT_used_for_class_action": False,
        "GT_used_for_relation_action": False,
        "GT_used_for_relation_weights": False,
        "GT_used_for_utility_bins": False,
        "GT_used_for_memory_candidate": False,
        "training_performed": False,
        "model_forward_called": False,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "backward_called": False,
        "new_information_utility_constructed": False,
        "U_cycle_frozen": True,
        "shuffle_mapping_independently_rebuilt": True,
        "shuffle_relation_independently_rebuilt": True,
        "shuffle_relation_weights_independently_rebuilt": True,
        "shuffle_prototype_independently_rebuilt": True,
        "shuffle_candidate_independently_rebuilt": True,
        "persistent_memory_created": False,
    }
    _require(
        seal["true_target_hash"] != seal["shuffle_target_hash"],
        "C3-A0 true/shuffle target hashes unexpectedly match",
    )
    seal_path = output_root / ACTION_SEAL_NAME
    write_json_durable(seal_path, seal)
    _fsync_directory(output_root)
    _verify_json_durable_readback(seal_path, seal)
    return {
        "output_dir": output_root,
        "npz_path": npz_path,
        "seal_path": seal_path,
        "seal": seal,
        "seal_file_sha256": file_sha256(seal_path),
        "durable_action_seal_verified": True,
    }


def load_full_ground_truth_after_action_seal(path):
    return c2_c2_evaluator.load_full_ground_truth_after_support_seal(path)


def evaluate_after_action_seal(bundle, persistence, paths):
    """Load full GT only after durable action verification."""
    _require(
        persistence.get("durable_action_seal_verified") is True
        and read_json(persistence["seal_path"]) == persistence["seal"],
        "C3-A0 durable action seal is not verified",
    )
    full_GT, full_GT_audit = load_full_ground_truth_after_action_seal(
        paths.full_gt_path
    )
    result = c3.build_postseal_evaluation(bundle["arrays"], full_GT)
    summary = c3.build_seed_summary(
        bundle["seed"],
        result["class_metrics"],
        result["relation_metrics"],
        result["memory_metrics"],
    )
    return {
        **result,
        "summary": summary,
        "full_GT_audit": {
            **full_GT_audit,
            "loaded_after_action_seal": True,
        },
    }


def save_postseal_results(bundle, persistence, evaluation):
    output_root = persistence["output_dir"]
    class_record = {
        "stage": STAGE,
        "seed": bundle["seed"],
        "metrics": evaluation["class_metrics"],
    }
    relation_record = {
        "stage": STAGE,
        "seed": bundle["seed"],
        "metrics": evaluation["relation_metrics"],
    }
    memory_record = {
        "stage": STAGE,
        "seed": bundle["seed"],
        "metrics": evaluation["memory_metrics"],
    }
    outcomes = evaluation["outcomes"]
    audit = {
        "stage": STAGE,
        "seed": bundle["seed"],
        "C3_A0_AUDIT_PASS": True,
        "action_seal": persistence["seal"],
        "action_seal_file_sha256": persistence["seal_file_sha256"],
        "durable_action_seal_verified_before_GT": True,
        "GT_loaded_before_action_seal": False,
        "GT_loaded_after_action_seal": True,
        "C0_audit_loaded_before_action_seal": False,
        "C0_audit_loaded_after_action_seal": False,
        "Bridge_loaded_before_action_seal": False,
        "Bridge_loaded_after_action_seal": False,
        "GT_used_for_class_action": False,
        "GT_used_for_relation_action": False,
        "GT_used_for_relation_weights": False,
        "GT_used_for_utility_bins": False,
        "GT_used_for_memory_candidate": False,
        "training_performed": False,
        "model_forward_called": False,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "backward_called": False,
        "new_information_utility_constructed": False,
        "U_cycle_frozen": True,
        "persistent_memory_created": False,
        "evaluation_only_unlabeled": True,
        "evaluation_sample_count": c3.UNLABELED_EVAL_COUNT,
        "full_GT": evaluation["full_GT_audit"],
        "postseal_outcome_logical_sha256": {
            name: ndarray_sha256(value)
            for name, value in outcomes.items()
        },
        "MemoryWriteBenefit_used_in_gate": False,
        "branch_decision_precommitted_before_formal_results": True,
    }
    records = OrderedDict((
        ("c3_a0_class_metrics.json", class_record),
        ("c3_a0_relation_metrics.json", relation_record),
        ("c3_a0_memory_write_metrics.json", memory_record),
        ("c3_a0_seed_summary.json", evaluation["summary"]),
        ("c3_a0_audit.json", audit),
    ))
    for name, record in records.items():
        write_json_durable(output_root / name, record)
    _fsync_directory(output_root)
    for name, record in records.items():
        _verify_json_durable_readback(output_root / name, record)
    return records


def run_evaluation(seed, output_dir=None, full_gt_path=None):
    """Run one formal seed; callers must obtain separate authorization."""
    frozen_boundary = verify_frozen_parent_sources()
    active_seed = c3.validate_seed(seed)
    paths = default_paths(active_seed, output_dir=output_dir)
    if full_gt_path is not None:
        paths = C3A0Paths(
            paths.c0_artifact_path,
            paths.c0_seal_path,
            paths.label_split_dir,
            _resolve(full_gt_path),
            paths.output_dir,
        )
    arrays, provenance = load_c0_inputs_before_gt(
        active_seed, paths.c0_artifact_path, paths.c0_seal_path
    )
    split = load_fixed_sparse_labels(paths.label_split_dir)
    bundle = build_pre_gt_bundle(
        active_seed, arrays, provenance, split, frozen_boundary
    )
    persistence = persist_action_bundle(bundle, paths.output_dir)
    evaluation = evaluate_after_action_seal(bundle, persistence, paths)
    records = save_postseal_results(
        bundle, persistence, evaluation
    )
    print("Saved: " + _display(paths.output_dir))
    return {
        "bundle": bundle,
        "persistence": persistence,
        "evaluation": evaluation,
        "records": records,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=c3.SEEDS)
    parser.add_argument("--output-dir")
    parser.add_argument("--full-gt-path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_evaluation(
        args.seed,
        output_dir=args.output_dir,
        full_gt_path=args.full_gt_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
