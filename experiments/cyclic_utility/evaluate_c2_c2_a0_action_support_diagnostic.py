"""Evaluate C2-C2-A0 behind a durable pre-GT action-support seal.

Formal execution is exposed only for a later separately authorized phase.
This implementation phase exercises pure tests and temporary seals only.
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

from experiments.cyclic_utility import c2_c2_a0_action_support_diagnostic_protocol as c2
from experiments.cyclic_utility import evaluate_c2_b0_sparse_utility_residual as b0_evaluator
from experiments.cyclic_utility import evaluate_c2_c0_directional_consensus_utility as c0_evaluator
from weak_quality import ndarray_sha256


STAGE = c2.STAGE
DATASET = "Caltech-6V"
FROZEN_SOURCE_MANIFESTS = (
    "experiment_freeze/c2_b0_multiseed_fail_closed_20260908/"
    "source_sha256.txt",
    "experiment_freeze/c2_c0_multiseed_fail_closed_20260908/"
    "source_sha256.txt",
    "experiment_freeze/c2_c1_multiseed_fail_closed_20260908/"
    "source_sha256.txt",
)
PRE_GT_NPZ_NAME = "c2_c2_a0_support_pre_gt.npz"
SUPPORT_SEAL_NAME = "c2_c2_a0_support_seal.json"
PRE_GT_ARRAY_NAMES = (
    "sample_ids",
    "unlabeled_ids",
    "labeled_ids",
    "labeled_targets",
    "shuffled_labeled_targets",
    "U_cycle",
    "C_conf",
    "y_gen",
    "R",
    "z_L",
    "e_L",
    "d_L",
    "weights",
    "D",
    "support_mass",
    "support_square_mass",
    "ESS",
    "active_anchor_count",
    "z_L_shuffle",
    "e_L_shuffle",
    "d_L_shuffle",
    "weights_shuffle",
    "D_shuffle",
    "support_mass_shuffle",
    "support_square_mass_shuffle",
    "ESS_shuffle",
    "active_anchor_count_shuffle",
)
PER_SEED_OUTPUT_FILES = (
    PRE_GT_NPZ_NAME,
    SUPPORT_SEAL_NAME,
    "c2_c2_a0_metrics_by_direction.json",
    "c2_c2_a0_support_quantile_audit.json",
    "c2_c2_a0_seed_summary.json",
    "c2_c2_a0_audit.json",
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
    """Convert NumPy containers/scalars without changing scientific values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            key: _json_native(item)
            for key, item in value.items()
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
    expected = _json_native(original_record)
    actual = read_json(path)
    _require(
        actual == expected,
        "C2-C2-A0 durable post-seal JSON reload mismatch",
    )


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_frozen_sources():
    manifests = OrderedDict()
    for name in FROZEN_SOURCE_MANIFESTS:
        audit = b0_evaluator.verify_sha256_manifest(name)
        _require(
            audit.get("all_entries_exact_match") is True,
            "C2-C2-A0 frozen-source SHA256 verification failed",
        )
        manifests[name] = audit
    return {
        "manifests": dict(manifests),
        "C2_B0_frozen_hashes_pass": True,
        "C2_C0_frozen_hashes_pass": True,
        "C2_C1_frozen_hashes_pass": True,
        "all_frozen_source_hashes_pass": True,
    }


@dataclass(frozen=True)
class C2C2A0Paths:
    c0_artifact_path: Path
    c0_seal_path: Path
    c0_audit_path: Path
    label_split_dir: Path
    full_gt_path: Path
    output_dir: Path


def default_paths(seed, output_dir=None):
    active_seed = c2.validate_seed(seed)
    source = c0_evaluator.default_paths(active_seed)
    target = (
        REPOSITORY_ROOT
        / "outputs/cyclic_utility"
        / ("c2_c2_a0_action_support_diagnostic_seed" + str(active_seed))
        if output_dir is None
        else _resolve(output_dir)
    )
    return C2C2A0Paths(
        c0_artifact_path=source.c0_artifact_path,
        c0_seal_path=source.c0_seal_path,
        c0_audit_path=source.c0_audit_path,
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
        "C2-C2-A0 pre-GT loader crossed an oracle boundary",
    )
    return arrays, provenance


def load_fixed_sparse_labels(split_dir):
    split = c0_evaluator.load_fixed_sparse_labels(split_dir)
    validated = c2.validate_fixed_sparse_split(
        split["labeled_sample_ids"],
        split["unlabeled_ids"],
        split["labeled_targets"],
    )
    shuffled = c2.validate_negative_control(
        validated["labeled_targets"], split["shuffled_labeled_targets"]
    )
    return {**split, **validated, "shuffled_labeled_targets": shuffled}


def build_pre_gt_bundle(
    seed, c0_arrays, c0_provenance, sparse_split, frozen_boundary
):
    """Construct support diagnostics without accepting correctness inputs."""
    active_seed = c2.validate_seed(seed)
    _require(
        c0_provenance.get("GT_loaded") is False
        and c0_provenance.get("c0_audit_loaded") is False
        and c0_provenance.get("Bridge_correctness_loaded") is False
        and c0_provenance.get("corruption_mask_loaded") is False,
        "C2-C2-A0 pre-GT bundle received leaked provenance",
    )
    _require(
        frozen_boundary.get("all_frozen_source_hashes_pass") is True,
        "C2-C2-A0 cannot proceed without frozen-source verification",
    )
    outputs = c2.build_action_support_outputs(
        y_gen=c0_arrays["y_gen"],
        U_cycle=c0_arrays["U_cycle"],
        labeled_ids=sparse_split["labeled_sample_ids"],
        labeled_targets=sparse_split["labeled_targets"],
        shuffled_targets=sparse_split["shuffled_labeled_targets"],
    )
    arrays = OrderedDict((
        ("sample_ids", c0_arrays["sample_ids"]),
        ("unlabeled_ids", sparse_split["unlabeled_ids"]),
        ("labeled_ids", sparse_split["labeled_sample_ids"]),
        ("labeled_targets", sparse_split["labeled_targets"]),
        ("shuffled_labeled_targets", sparse_split["shuffled_labeled_targets"]),
        ("U_cycle", c0_arrays["U_cycle"]),
        ("C_conf", c0_arrays["C_conf"]),
        ("y_gen", c0_arrays["y_gen"]),
        ("R", outputs["R"]),
        ("z_L", outputs["z_L"]),
        ("e_L", outputs["e_L"]),
        ("d_L", outputs["d_L"]),
        ("weights", outputs["weights"]),
        ("D", outputs["D"]),
        ("support_mass", outputs["support_mass"]),
        ("support_square_mass", outputs["support_square_mass"]),
        ("ESS", outputs["ESS"]),
        ("active_anchor_count", outputs["active_anchor_count"]),
        ("z_L_shuffle", outputs["z_L_shuffle"]),
        ("e_L_shuffle", outputs["e_L_shuffle"]),
        ("d_L_shuffle", outputs["d_L_shuffle"]),
        ("weights_shuffle", outputs["weights_shuffle"]),
        ("D_shuffle", outputs["D_shuffle"]),
        ("support_mass_shuffle", outputs["support_mass_shuffle"]),
        ("support_square_mass_shuffle", outputs["support_square_mass_shuffle"]),
        ("ESS_shuffle", outputs["ESS_shuffle"]),
        ("active_anchor_count_shuffle", outputs["active_anchor_count_shuffle"]),
    ))
    _require(tuple(arrays) == PRE_GT_ARRAY_NAMES, "C2-C2-A0 pre-GT key whitelist mismatch")
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


def persist_support_bundle(bundle, output_dir):
    output_root = _resolve(output_dir)
    _require(not output_root.exists(), "refusing to overwrite C2-C2-A0 seed output")
    arrays = bundle["arrays"]
    _require(tuple(arrays) == PRE_GT_ARRAY_NAMES, "C2-C2-A0 persistence schema mismatch")
    array_records = _array_records(arrays)
    output_root.mkdir(parents=True, exist_ok=False)
    npz_path = output_root / PRE_GT_NPZ_NAME
    with open(npz_path, "xb") as output_file:
        np.savez(output_file, **arrays)
        output_file.flush()
        os.fsync(output_file.fileno())
    _fsync_directory(output_root)
    with np.load(npz_path, allow_pickle=False) as archive:
        _require(tuple(archive.files) == PRE_GT_ARRAY_NAMES, "C2-C2-A0 sealed NPZ field mismatch")
        for name in PRE_GT_ARRAY_NAMES:
            reloaded = np.array(archive[name], copy=True, order="C")
            _require(
                np.array_equal(reloaded, np.asarray(arrays[name]))
                and ndarray_sha256(reloaded) == array_records[name]["logical_sha256"],
                "C2-C2-A0 sealed array reload/hash mismatch: " + name,
            )
    provenance = bundle["C0_provenance"]
    split = bundle["sparse_split"]
    seal = {
        "stage": STAGE,
        "seed": bundle["seed"],
        "dataset": DATASET,
        "N": c2.SAMPLE_NUM,
        "K": c2.CLASS_NUM,
        "direction_count": c2.DIRECTION_COUNT,
        "label_count": c2.LABEL_COUNT,
        "unlabeled_evaluation_count": c2.UNLABELED_EVAL_COUNT,
        "pre_gt_npz_path": _display(npz_path),
        "pre_gt_npz_file_sha256": file_sha256(npz_path),
        "arrays": array_records,
        "input_C0_NPZ_SHA256": provenance["C0_artifact_file_sha256"],
        "input_C0_prediction_seal_SHA256": provenance[
            "C0_prediction_seal_file_sha256"
        ],
        "label_split_sha256": split["label_split_sha256"],
        "frozen_source_verification": bundle["frozen_boundary"],
        "true_target_hash": array_records["labeled_targets"]["logical_sha256"],
        "shuffle_target_hash": array_records["shuffled_labeled_targets"][
            "logical_sha256"
        ],
        "true_support_hashes": {
            name: array_records[name]["logical_sha256"]
            for name in (
                "weights", "D", "support_mass", "support_square_mass",
                "ESS", "active_anchor_count",
            )
        },
        "shuffle_support_hashes": {
            name: array_records[name]["logical_sha256"]
            for name in (
                "weights_shuffle", "D_shuffle", "support_mass_shuffle",
                "support_square_mass_shuffle", "ESS_shuffle",
                "active_anchor_count_shuffle",
            )
        },
        "NPZ_saved_fsynced_reloaded_hash_verified": True,
        "GT_loaded_before_support_seal": False,
        "c0_audit_loaded_before_support_seal": False,
        "Bridge_correctness_loaded_before_support_seal": False,
        "corruption_mask_loaded": False,
        "training_performed": False,
        "model_forward_called": False,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "backward_called": False,
        "GT_used_for_support": False,
        "weights_shuffle_independently_rebuilt": True,
        "true_weights_reused_for_shuffle": False,
        "ESS_shuffle_independently_computed": True,
        "new_utility_method_constructed": False,
        "new_admission_gate_constructed": False,
    }
    _require(
        seal["true_target_hash"] != seal["shuffle_target_hash"],
        "C2-C2-A0 true/shuffle target hashes unexpectedly match",
    )
    seal_path = output_root / SUPPORT_SEAL_NAME
    write_json_durable(seal_path, seal)
    _fsync_directory(output_root)
    _require(read_json(seal_path) == seal, "C2-C2-A0 support seal durable reload mismatch")
    return {
        "output_dir": output_root,
        "npz_path": npz_path,
        "seal_path": seal_path,
        "seal": seal,
        "seal_file_sha256": file_sha256(seal_path),
        "durable_support_seal_verified": True,
    }


def load_full_ground_truth_after_support_seal(path):
    return c0_evaluator.load_full_ground_truth_after_directional_seal(path)


def load_c0_audit_after_support_seal(c0_audit_path, c0_prediction_seal):
    return c0_evaluator.load_c0_audit_after_directional_seal(
        c0_audit_path, c0_prediction_seal
    )


def load_native_global_cluster_after_support_seal(
    c0_artifact_path, c0_prediction_seal
):
    return c0_evaluator.load_native_global_cluster_after_directional_seal(
        c0_artifact_path, c0_prediction_seal
    )


def build_bridge_correctness_after_support_seal(arrays, c0_audit, full_GT):
    return c0_evaluator.build_bridge_correctness_after_directional_seal(
        arrays, c0_audit, full_GT
    )


def evaluate_after_support_seal(bundle, persistence, paths):
    _require(
        persistence.get("durable_support_seal_verified") is True
        and read_json(persistence["seal_path"]) == persistence["seal"],
        "C2-C2-A0 durable support seal is not verified",
    )
    full_GT, full_GT_audit = load_full_ground_truth_after_support_seal(
        paths.full_gt_path
    )
    c0_audit, c0_audit_provenance = load_c0_audit_after_support_seal(
        paths.c0_audit_path, bundle["C0_provenance"]["C0_prediction_seal"]
    )
    native = load_native_global_cluster_after_support_seal(
        paths.c0_artifact_path, bundle["C0_provenance"]["C0_prediction_seal"]
    )
    correct, mapping_audit = build_bridge_correctness_after_support_seal(
        {"y_gen": bundle["arrays"]["y_gen"], "native_global_cluster": native},
        c0_audit,
        full_GT,
    )
    arrays = bundle["arrays"]
    unlabeled = arrays["unlabeled_ids"]
    _require(
        unlabeled.shape == (c2.UNLABELED_EVAL_COUNT,)
        and np.intersect1d(unlabeled, arrays["labeled_ids"]).size == 0,
        "C2-C2-A0 post-seal evaluation split mismatch",
    )
    z_gt = np.ascontiguousarray(correct[unlabeled], dtype=np.float64)
    e_gt = np.ascontiguousarray(
        z_gt - arrays["U_cycle"][unlabeled], dtype=np.float64
    )
    frozen_action = c2.frozen_c0_action_and_benefit(
        z_gt, arrays["U_cycle"][unlabeled], arrays["D"][unlabeled]
    )
    metrics = c2.analyze_support_directions(
        arrays["ESS"][unlabeled],
        arrays["ESS_shuffle"][unlabeled],
        arrays["D"][unlabeled],
        arrays["D_shuffle"][unlabeled],
        e_gt,
        frozen_action["ActionBenefit"],
        unlabeled,
    )
    summary = c2.build_seed_summary(bundle["seed"], metrics)
    return {
        "support_metrics": metrics,
        "summary": summary,
        "full_GT_audit": {**full_GT_audit, "loaded_after_support_seal": True},
        "c0_audit_provenance": {
            **c0_audit_provenance, "loaded_after_support_seal": True
        },
        "mapping_audit": mapping_audit,
        "correctness_logical_sha256": ndarray_sha256(correct),
        "e_GT_logical_sha256": ndarray_sha256(e_gt),
        "U_C0_logical_sha256": ndarray_sha256(frozen_action["U_C0"]),
        "ActionBenefit_logical_sha256": ndarray_sha256(
            frozen_action["ActionBenefit"]
        ),
    }


def save_postseal_results(bundle, persistence, evaluation):
    output_root = persistence["output_dir"]
    metrics = evaluation["support_metrics"]
    metric_record = {
        "stage": STAGE,
        "seed": bundle["seed"],
        "evaluation_sample_count": c2.UNLABELED_EVAL_COUNT,
        "support_metrics": metrics,
    }
    quantile_audit = {
        "stage": STAGE,
        "seed": bundle["seed"],
        "quantile_count": c2.SUPPORT_QUANTILE_COUNT,
        "evaluation_sample_count": c2.UNLABELED_EVAL_COUNT,
        "ESS_only_for_assignment": True,
        "GT_used_for_assignment": False,
        "directions": [
            {
                "direction_id": record["direction_id"],
                "joint_valid": record["joint_valid"],
                "support_quantiles": record["support_quantiles"],
            }
            for record in metrics["directions"]
        ],
    }
    audit = {
        "stage": STAGE,
        "seed": bundle["seed"],
        "C2_C2_A0_AUDIT_PASS": True,
        "support_seal": persistence["seal"],
        "support_seal_file_sha256": persistence["seal_file_sha256"],
        "durable_support_seal_verified_before_GT": True,
        "GT_loaded_before_support_seal": False,
        "GT_loaded_after_support_seal": True,
        "c0_audit_loaded_before_support_seal": False,
        "c0_audit_loaded_after_support_seal": True,
        "Bridge_correctness_loaded_before_support_seal": False,
        "Bridge_correctness_built_after_support_seal": True,
        "corruption_mask_loaded": False,
        "training_performed": False,
        "model_forward_called": False,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "backward_called": False,
        "GT_used_for_support": False,
        "GT_used_for_quantile_assignment": False,
        "new_utility_method_constructed": False,
        "new_admission_gate_constructed": False,
        "ActionBenefit_used_in_gate": False,
        "support_mass_used_in_gate": False,
        "active_anchor_count_used_in_gate": False,
        "C1_admission_rate_used_in_gate": False,
        "C2_specific_GT_mapping_fit": False,
        "direction_specific_mapping_fit": False,
        "bin_specific_mapping_fit": False,
        "evaluation_only_unlabeled": True,
        "evaluation_sample_count": c2.UNLABELED_EVAL_COUNT,
        "labeled_evaluation_intersection_count": 0,
        "full_GT": evaluation["full_GT_audit"],
        "c0_audit": evaluation["c0_audit_provenance"],
        "frozen_C0_global_mapping": evaluation["mapping_audit"],
        "correctness_logical_sha256": evaluation["correctness_logical_sha256"],
        "e_GT_logical_sha256": evaluation["e_GT_logical_sha256"],
        "U_C0_logical_sha256": evaluation["U_C0_logical_sha256"],
        "ActionBenefit_logical_sha256": evaluation[
            "ActionBenefit_logical_sha256"
        ],
    }
    records = OrderedDict((
        ("c2_c2_a0_metrics_by_direction.json", metric_record),
        ("c2_c2_a0_support_quantile_audit.json", quantile_audit),
        ("c2_c2_a0_seed_summary.json", evaluation["summary"]),
        ("c2_c2_a0_audit.json", audit),
    ))
    for name, record in records.items():
        write_json_durable(output_root / name, record)
    _fsync_directory(output_root)
    for name, record in records.items():
        _verify_json_durable_readback(output_root / name, record)
    return records


def run_evaluation(seed, output_dir=None, full_gt_path=None):
    """Run one formal seed; callers must obtain separate authorization."""
    frozen_boundary = verify_frozen_sources()
    active_seed = c2.validate_seed(seed)
    paths = default_paths(active_seed, output_dir=output_dir)
    if full_gt_path is not None:
        paths = C2C2A0Paths(
            paths.c0_artifact_path,
            paths.c0_seal_path,
            paths.c0_audit_path,
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
    persistence = persist_support_bundle(bundle, paths.output_dir)
    evaluation = evaluate_after_support_seal(bundle, persistence, paths)
    records = save_postseal_results(bundle, persistence, evaluation)
    print("C2_C2_A0_SEED_PASS=" + str(evaluation["summary"]["C2_C2_A0_SEED_PASS"]))
    print("Saved: " + _display(paths.output_dir))
    return {
        "bundle": bundle,
        "persistence": persistence,
        "evaluation": evaluation,
        "records": records,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=c2.SEEDS)
    parser.add_argument("--output-dir")
    parser.add_argument("--full-gt-path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_evaluation(args.seed, output_dir=args.output_dir, full_gt_path=args.full_gt_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
