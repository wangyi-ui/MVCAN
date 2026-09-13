"""Build and seal C5-A0 pseudo-semantic outputs without loading full GT."""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import c3_b0_true_u_carrier_protocol as carrier_protocol
from experiments.cyclic_utility import c5_a0_utility_validated_pseudo_semantic_protocol as c5
from weak_quality import ndarray_sha256


OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/cyclic_utility"
CARRIER_KEYS = ("sample_ids", "q_local", "q_aligned", "M_v")
ACTION_REQUIRED_KEYS = (
    "sample_ids", "labeled_ids", "unlabeled_ids", "labeled_targets", "U_cycle"
)


def _require(condition, message="C5_A0_PRE_GT_FAIL_CLOSED"):
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


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, record):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(record, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def default_output_dir(seed):
    return OUTPUT_ROOT / (
        "c5_a0_utility_validated_pseudo_semantic_seed" + str(c5.validate_seed(seed))
    )


def seed_specific_carrier_paths(seed):
    active_seed = c5.validate_seed(seed)
    root = OUTPUT_ROOT / ("c3_b0_true_u_carrier_seed" + str(active_seed))
    return (
        root / "c3_b0_true_u_carrier.npz",
        root / "c3_b0_true_u_carrier_audit.json",
        root / "c3_b0_true_u_carrier_seal.json",
    )


def seed_specific_action_paths(seed):
    active_seed = c5.validate_seed(seed)
    root = OUTPUT_ROOT / (
        "c3_a0_utility_conditioned_action_granularity_seed" + str(active_seed)
    )
    return root / "c3_a0_action_pre_gt.npz", root / "c3_a0_action_seal.json"


def load_sealed_carrier(seed):
    active_seed = c5.validate_seed(seed)
    artifact_path, audit_path, seal_path = seed_specific_carrier_paths(active_seed)
    _require(
        artifact_path.is_file() and audit_path.is_file() and seal_path.is_file(),
        "C5_A0_CARRIER_MISSING_FAIL_CLOSED",
    )
    seal = _read_json(seal_path)
    audit = _read_json(audit_path)
    _require(
        seal.get("stage") == carrier_protocol.STAGE
        and int(seal.get("seed", -1)) == active_seed
        and seal.get("carrier_valid_for_downstream_readonly_use") is True
        and seal.get("all_parent_hashes_pass") is True
        and seal.get("final_model_hash_equal") is True
        and seal.get("final_predictions_equal") is True
        and seal.get("sample_ids_equal") is True
        and seal.get("coordinate_mapping_pass") is True
        and seal.get("GT_loaded_for_carrier_materialization") is False
        and seal.get("C4_used") is False
        and seal.get("artifact_keys") == list(CARRIER_KEYS)
        and carrier_protocol.file_sha256(artifact_path)
        == seal.get("artifact_file_sha256")
        and carrier_protocol.file_sha256(audit_path) == seal.get("audit_file_sha256"),
        "C5_A0_CARRIER_SEAL_FAIL_CLOSED",
    )
    _require(
        audit.get("stage") == carrier_protocol.STAGE
        and int(audit.get("seed", -1)) == active_seed
        and audit.get("C4_used") is False
        and audit.get("GT_loaded_for_carrier_materialization") is False
        and audit.get("all_parent_hashes_pass") is True
        and audit.get("final_model_hash_equal") is True
        and audit.get("final_predictions_equal") is True
        and audit.get("final_prediction_sample_ids_equal") is True
        and audit.get("coordinate_mapping_pass") is True,
        "C5_A0_CARRIER_AUDIT_FAIL_CLOSED",
    )
    carrier_protocol.validate_snapshot_metadata(
        audit.get("snapshot_stage"),
        audit.get("snapshot_epoch"),
        audit.get("snapshot_position"),
    )
    with np.load(artifact_path, allow_pickle=False) as archive:
        _require(tuple(archive.files) == CARRIER_KEYS, "C5_A0_CARRIER_SCHEMA_FAIL_CLOSED")
        arrays = {
            key: np.array(archive[key], copy=True, order="C") for key in CARRIER_KEYS
        }
    sample_ids = carrier_protocol.validate_sample_ids(arrays["sample_ids"])
    q_local = torch.from_numpy(arrays["q_local"]).detach()
    q_aligned = torch.from_numpy(arrays["q_aligned"]).detach()
    matches = torch.from_numpy(arrays["M_v"]).detach()
    coordinate = carrier_protocol.validate_coordinate_snapshot(
        q_local, q_aligned, matches
    )
    _require(
        c5.logical_sha256(arrays["q_aligned"])
        == audit.get("q_aligned_logical_sha256")
        and c5.logical_sha256(arrays["M_v"]) == audit.get("M_v_logical_sha256")
        and c5.logical_sha256(sample_ids) == audit.get("sample_ids_logical_sha256"),
        "C5_A0_CARRIER_LOGICAL_HASH_FAIL_CLOSED",
    )
    return arrays, {
        "seed": active_seed,
        "artifact_path": _display(artifact_path),
        "artifact_file_sha256": carrier_protocol.file_sha256(artifact_path),
        "audit_path": _display(audit_path),
        "audit_file_sha256": carrier_protocol.file_sha256(audit_path),
        "seal_path": _display(seal_path),
        "seal_file_sha256": carrier_protocol.file_sha256(seal_path),
        "snapshot_stage": audit["snapshot_stage"],
        "snapshot_epoch": audit["snapshot_epoch"],
        "snapshot_position": audit["snapshot_position"],
        "coordinate_mapping_pass": coordinate["coordinate_mapping_pass"],
        "carrier_valid_for_downstream_readonly_use": True,
        "seed_specific_carrier_provenance_pass": True,
        "C3_A0_action_npz_file_sha256": audit["action_provenance"]["npz_file_sha256"],
        "C3_A0_action_seal_file_sha256": audit["action_provenance"]["seal_file_sha256"],
        "GT_loaded": False,
        "C4_used": False,
    }


def load_sealed_c3a0_action(seed):
    active_seed = c5.validate_seed(seed)
    artifact_path, seal_path = seed_specific_action_paths(active_seed)
    _require(
        artifact_path.is_file() and seal_path.is_file(),
        "C5_A0_C3_A0_ACTION_MISSING_FAIL_CLOSED",
    )
    seal = _read_json(seal_path)
    _require(
        seal.get("stage") == "C3-A0"
        and int(seal.get("seed", -1)) == active_seed
        and seal.get("N") == c5.N
        and seal.get("K") == c5.K
        and seal.get("direction_count") == c5.S
        and seal.get("label_count") == c5.L
        and seal.get("unlabeled_evaluation_count") == c5.NU
        and seal.get("GT_loaded_before_action_seal") is False
        and seal.get("training_performed") is False
        and seal.get("U_cycle_frozen") is True
        and seal.get("frozen_parent_verification", {}).get(
            "all_frozen_parent_hashes_pass"
        ) is True
        and carrier_protocol.file_sha256(artifact_path)
        == seal.get("pre_gt_npz_file_sha256"),
        "C5_A0_C3_A0_ACTION_SEAL_FAIL_CLOSED",
    )
    with np.load(artifact_path, allow_pickle=False) as archive:
        _require(
            all(key in archive.files for key in ACTION_REQUIRED_KEYS),
            "C5_A0_C3_A0_ACTION_SCHEMA_FAIL_CLOSED",
        )
        arrays = {
            key: np.array(archive[key], copy=True, order="C")
            for key in ACTION_REQUIRED_KEYS
        }
    for key, value in arrays.items():
        record = seal.get("arrays", {}).get(key, {})
        _require(
            list(value.shape) == record.get("shape")
            and str(value.dtype) == record.get("dtype")
            and ndarray_sha256(value) == record.get("logical_sha256"),
            "C5_A0_C3_A0_ACTION_LOGICAL_HASH_FAIL_CLOSED",
        )
    c5.validate_sparse_inputs(
        arrays["labeled_ids"], arrays["labeled_targets"], arrays["unlabeled_ids"]
    )
    c5.extract_unlabeled_utility(
        arrays["U_cycle"], arrays["sample_ids"], arrays["unlabeled_ids"]
    )
    return arrays, {
        "seed": active_seed,
        "artifact_path": _display(artifact_path),
        "artifact_file_sha256": carrier_protocol.file_sha256(artifact_path),
        "seal_path": _display(seal_path),
        "seal_file_sha256": carrier_protocol.file_sha256(seal_path),
        "C0_provenance_file_sha256": seal["input_C0_NPZ_SHA256"],
        "all_parent_hashes_pass": True,
        "GT_loaded": False,
        "all_14_original_sparse_anchors_loaded": True,
        "anchor_consistency_filter_used": False,
    }


def _to_numpy(value, dtype=None):
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def _array_records(arrays):
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "logical_sha256": c5.logical_sha256(value),
        }
        for key, value in arrays.items()
    }


def _persist_pre_gt(arrays, output_dir):
    c5.validate_pre_gt_array_keys(arrays)
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = output_dir / "c5_pre_gt_bundle.npz"
    with open(artifact_path, "xb") as output_file:
        np.savez(output_file, **arrays)
        output_file.flush()
        os.fsync(output_file.fileno())
    with np.load(artifact_path, allow_pickle=False) as archive:
        _require(tuple(archive.files) == c5.PRE_GT_ARRAY_KEYS, "C5_A0_NPZ_SCHEMA_FAIL_CLOSED")
        for key, expected in arrays.items():
            _require(np.array_equal(archive[key], expected), "C5_A0_NPZ_RELOAD_FAIL_CLOSED")
    return artifact_path


def run_pre_gt(seed, output_dir=None):
    active_seed = c5.validate_seed(seed)
    target = default_output_dir(active_seed) if output_dir is None else _resolve(output_dir)
    _require(not target.exists(), "refusing to overwrite C5-A0 output")
    carrier_arrays, carrier_provenance = load_sealed_carrier(active_seed)
    action_arrays, action_provenance = load_sealed_c3a0_action(active_seed)
    _require(
        np.array_equal(carrier_arrays["sample_ids"], action_arrays["sample_ids"])
        and carrier_provenance["C3_A0_action_npz_file_sha256"]
        == action_provenance["artifact_file_sha256"]
        and carrier_provenance["C3_A0_action_seal_file_sha256"]
        == action_provenance["seal_file_sha256"],
        "C5_A0_CARRIER_ACTION_SAMPLE_ORDER_FAIL_CLOSED",
    )
    generator, verifier = c5.frozen_direction_definitions()
    q_aligned = torch.from_numpy(carrier_arrays["q_aligned"]).detach()
    candidates = c5.build_directional_semantic_candidates(
        q_aligned,
        action_arrays["labeled_ids"],
        action_arrays["labeled_targets"],
        action_arrays["unlabeled_ids"],
        generator,
    )
    U_true = c5.extract_unlabeled_utility(
        action_arrays["U_cycle"],
        action_arrays["sample_ids"],
        action_arrays["unlabeled_ids"],
    )
    eligibility = c5.build_zero_mass_abstention(
        U_true, action_arrays["unlabeled_ids"]
    )
    eligible_mask_tensor = torch.from_numpy(
        np.array(eligibility["eligible_mask"], copy=True)
    )
    eligible_directional = candidates["directional_posterior"][
        eligible_mask_tensor
    ].detach()
    permutation = c5.build_permuted_utility(
        eligibility["U_true_eligible"], eligibility["eligible_ids"]
    )
    arms = c5.build_three_arm_outputs(
        eligible_directional, eligibility["U_true_eligible"],
        permutation["U_permuted"]
    )
    arrays = OrderedDict((
        ("sample_ids", _to_numpy(action_arrays["sample_ids"], np.int64)),
        ("labeled_ids", _to_numpy(action_arrays["labeled_ids"], np.int64)),
        ("unlabeled_ids", _to_numpy(action_arrays["unlabeled_ids"], np.int64)),
        ("eligible_mask", _to_numpy(eligibility["eligible_mask"], np.bool_)),
        ("eligible_ids", _to_numpy(eligibility["eligible_ids"], np.int64)),
        ("abstained_ids", _to_numpy(eligibility["abstained_ids"], np.int64)),
        ("utility_mass", _to_numpy(eligibility["utility_mass"], np.float64)),
        ("generator_subsets", _to_numpy(generator, np.int64)),
        ("verifier_subsets", _to_numpy(verifier, np.int64)),
        ("permutation_source_ids", _to_numpy(permutation["permutation_source_ids"], np.int64)),
        ("U_true", _to_numpy(U_true, np.float64)),
        ("U_true_eligible", _to_numpy(eligibility["U_true_eligible"], np.float64)),
        ("U_permuted", _to_numpy(permutation["U_permuted"], np.float64)),
        ("directional_evidence", _to_numpy(candidates["directional_evidence"])),
        ("directional_posterior", _to_numpy(candidates["directional_posterior"])),
        ("directional_pred", _to_numpy(candidates["directional_pred"], np.int64)),
        ("directional_margin", _to_numpy(candidates["directional_margin"])),
        ("eligible_directional_posterior", _to_numpy(eligible_directional)),
        *(('posterior_' + arm, _to_numpy(arms[arm]["posterior"])) for arm in c5.ARMS),
        *(('prediction_' + arm, _to_numpy(arms[arm]["prediction"], np.int64)) for arm in c5.ARMS),
        *(('confidence_' + arm, _to_numpy(arms[arm]["confidence"])) for arm in c5.ARMS),
    ))
    c5.validate_pre_gt_array_keys(arrays)
    records = _array_records(arrays)
    artifact_path = _persist_pre_gt(arrays, target)
    audit = {
        "stage": c5.STAGE,
        "seed": active_seed,
        "GT_loaded": False,
        "training_performed": False,
        "backward_called": False,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "model_updated": False,
        "C5_output_fed_back_into_training": False,
        "C4_used_as_parent": False,
        "scientific_parent": "C3-B0 TRUE_U",
        "historical_C3_B0_SHUFFLE_U_role": "SEMANTIC_SHUFFLE historical control; not a C5 arm",
        "N": c5.N, "V": c5.V, "K": c5.K, "S": c5.S,
        "L": c5.L, "Nu": c5.NU,
        "eligible_count": eligibility["eligible_count"],
        "abstention_count": eligibility["abstention_count"],
        "eligibility_rate": eligibility["eligibility_rate"],
        "abstention_rate": eligibility["abstention_rate"],
        "same_eligible_set_for_all_arms": True,
        "uniform_fallback_for_abstained_rows": False,
        "sample_ids_exact_arange_pass": True,
        "labeled_unlabeled_disjointness_pass": True,
        "all_14_original_sparse_anchors_used": True,
        "anchor_consistency_filter_used": False,
        "direction_hashes": {
            "generator_subsets": c5.logical_sha256(generator),
            "verifier_subsets": c5.logical_sha256(verifier),
        },
        "generator_views_only_for_candidate": True,
        "verifier_views_used_for_candidate": False,
        "U_cycle_original_shape": list(action_arrays["U_cycle"].shape),
        "U_true_shape": list(U_true.shape),
        "U_true_logical_sha256": c5.logical_sha256(U_true),
        "U_true_eligible_logical_sha256": c5.logical_sha256(eligibility["U_true_eligible"]),
        "no_scalar_or_view_utility_constructed": True,
        "permutation_audit": {
            key: value for key, value in permutation.items()
            if key not in ("U_permuted", "permutation_source_ids")
        },
        "posterior_checks": {
            arm: {
                "shape": list(arrays["posterior_" + arm].shape),
                "finite": True,
                "non_negative": True,
                "rows_sum_to_one": True,
            }
            for arm in c5.ARMS
        },
        "candidate_checks": {
            "directional_evidence_shape": list(arrays["directional_evidence"].shape),
            "directional_posterior_shape": list(arrays["directional_posterior"].shape),
            "finite": True,
            "non_negative": True,
            "posterior_rows_sum_to_one": True,
        },
        "q_aligned_provenance": carrier_provenance,
        "C0_provenance": {
            "artifact_file_sha256": action_provenance["C0_provenance_file_sha256"],
            "direction_hashes_pass": True,
        },
        "C3_A0_provenance": action_provenance,
        "C3_B0_provenance": carrier_provenance,
        "all_parent_hashes_pass": True,
        "pre_gt_array_whitelist": list(c5.PRE_GT_ARRAY_KEYS),
        "arrays": records,
        "artifact_path": _display(artifact_path),
        "artifact_file_sha256": carrier_protocol.file_sha256(artifact_path),
    }
    audit_path = target / "c5_pre_gt_audit.json"
    _write_json(audit_path, audit)
    seal = {
        "stage": c5.STAGE,
        "seed": active_seed,
        "GT_loaded_before_pre_gt_seal": False,
        "training_performed": False,
        "C4_used_as_parent": False,
        "all_parent_hashes_pass": True,
        "carrier_seal_verified": True,
        "Nu": c5.NU,
        "eligible_count": eligibility["eligible_count"],
        "abstention_count": eligibility["abstention_count"],
        "eligibility_rate": eligibility["eligibility_rate"],
        "abstention_rate": eligibility["abstention_rate"],
        "same_eligible_set_for_all_arms": True,
        "artifact_path": _display(artifact_path),
        "artifact_file_sha256": carrier_protocol.file_sha256(artifact_path),
        "audit_path": _display(audit_path),
        "audit_file_sha256": carrier_protocol.file_sha256(audit_path),
        "array_whitelist": list(c5.PRE_GT_ARRAY_KEYS),
        "arrays": records,
        "pre_gt_seal_valid": True,
    }
    seal_path = target / "c5_pre_gt_seal.json"
    _write_json(seal_path, seal)
    return {
        "artifact_path": artifact_path,
        "audit_path": audit_path,
        "seal_path": seal_path,
        "audit": audit,
        "seal": seal,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=c5.SEEDS)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_pre_gt(args.seed, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
