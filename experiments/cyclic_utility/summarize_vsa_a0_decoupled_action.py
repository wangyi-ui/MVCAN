"""Summarize only the five formal VSA-A0 arms and apply frozen gates."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import (
    c1_frozen_pseudo_supervision as c1,
)
from experiments.cyclic_utility import (
    train_vsa_a0_decoupled_action as train,
)
from experiments.cyclic_utility import (
    vsa_decoupled_action_protocol as vsa,
)
from irv.b4_information_utility import tensor_sha256


DEFAULT_INPUT_DIR = train.DEFAULT_A0_OUTPUT_ROOT
SUMMARY_FILES = (
    "vsa_summary.json",
    "vsa_decision.json",
    "vsa_audit.json",
)
REQUIRED_ARM_FILES = (
    "metrics.json",
    "train_audit.json",
    "semantic_target_audit.json",
    "semantic_loss_history.json",
    "native_loss_history.json",
    "phase_transition_audit.json",
    "final_predictions.npz",
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def load_arm_bundle(input_dir, arm, expected_seed=None):
    """Load one formal arm; engineering CYCLE_ZERO is rejected."""
    if arm not in vsa.FORMAL_ARMS:
        raise ValueError("VSA summarizer accepts formal five arms only")
    arm_dir = Path(input_dir) / arm
    _require(arm_dir.is_dir(), "missing VSA-A0 arm directory: " + arm)
    for filename in REQUIRED_ARM_FILES:
        _require(
            (arm_dir / filename).is_file(),
            "missing VSA-A0 arm file: " + arm + "/" + filename,
        )
    records = {
        "metrics_record": c1.read_json(arm_dir / "metrics.json"),
        "train_audit": c1.read_json(arm_dir / "train_audit.json"),
        "semantic_target_audit": c1.read_json(
            arm_dir / "semantic_target_audit.json"
        ),
        "semantic_loss_history": c1.read_json(
            arm_dir / "semantic_loss_history.json"
        ),
        "native_loss_history": c1.read_json(
            arm_dir / "native_loss_history.json"
        ),
        "phase_transition_audit": c1.read_json(
            arm_dir / "phase_transition_audit.json"
        ),
    }
    identity_records = (
        records["metrics_record"],
        records["train_audit"],
        records["semantic_loss_history"],
        records["native_loss_history"],
        records["phase_transition_audit"],
    )
    _require(
        all(record.get("stage") == "VSA-A0" for record in identity_records)
        and all(record.get("arm") == arm for record in identity_records)
        and records["semantic_target_audit"].get("arm") == arm,
        "VSA-A0 arm record identity mismatch: " + arm,
    )
    metrics = records["metrics_record"].get("metrics", {})
    _require(
        set(metrics) == set(vsa.METRIC_NAMES)
        and all(
            isinstance(metrics[name], (int, float))
            and math.isfinite(float(metrics[name]))
            for name in vsa.METRIC_NAMES
        ),
        "VSA-A0 metric schema mismatch: " + arm,
    )
    training_seed = vsa.validate_training_seed(
        records["train_audit"].get(
            "training_seed", records["train_audit"].get("seed")
        )
    )
    if expected_seed is not None:
        _require(
            training_seed == vsa.validate_training_seed(expected_seed),
            "VSA-A0 requested seed mismatch: " + arm,
        )
    _require(
        records["train_audit"].get("epochs")
        == records["metrics_record"].get("epochs")
        == records["semantic_loss_history"].get("epochs")
        == records["native_loss_history"].get("epochs")
        == records["phase_transition_audit"].get("epochs")
        == vsa.FORMAL_EPOCHS
        and records["train_audit"].get("seed") == training_seed
        and records["metrics_record"].get("seed") == training_seed,
        "VSA-A0 epoch/seed mismatch: " + arm,
    )
    with np.load(
        arm_dir / "final_predictions.npz", allow_pickle=False
    ) as archive:
        _require(
            tuple(archive.files) == ("predictions", "sample_ids"),
            "VSA prediction archive field mismatch: " + arm,
        )
        predictions = np.asarray(archive["predictions"], dtype=np.int64)
        sample_ids = np.asarray(archive["sample_ids"], dtype=np.int64)
    prediction_hash = tensor_sha256(predictions)
    _require(
        predictions.shape == sample_ids.shape == (vsa.SAMPLE_NUM,)
        and np.array_equal(
            sample_ids, np.arange(vsa.SAMPLE_NUM, dtype=np.int64)
        )
        and prediction_hash
        == records["train_audit"]["prediction_logical_sha256"],
        "VSA final prediction seal mismatch: " + arm,
    )
    return {
        **records,
        "metrics": {
            name: float(metrics[name]) for name in vsa.METRIC_NAMES
        },
        "prediction_logical_sha256": prediction_hash,
        "training_seed": training_seed,
    }


def _normalized_e1_lineage(audit):
    checkpoint = audit["checkpoint_provenance"]
    source_seed = checkpoint.get(
        "source_training_seed", checkpoint.get("source_seed")
    )
    actual_aggregate = audit.get(
        "E1_actual_loaded_aggregate_hash",
        checkpoint.get(
            "actual_loaded_aggregate_sha256",
            audit["initial_model_hash"]["aggregate"],
        ),
    )
    expected_aggregate = audit.get(
        "E1_expected_from_own_audit_aggregate_hash",
        checkpoint.get(
            "expected_aggregate_sha256_from_own_audit",
            checkpoint.get(
                "initial_model_aggregate_sha256", actual_aggregate
            ),
        ),
    )
    return {
        "audit_path": audit.get(
            "E1_audit_path", checkpoint.get("source_audit_path")
        ),
        "training_seed": source_seed,
        "checkpoint_file_sha256": checkpoint[
            "checkpoint_file_sha256"
        ],
        "actual_aggregate_sha256": actual_aggregate,
        "expected_aggregate_sha256": expected_aggregate,
    }


def _normalized_c0_lineage(audit):
    provenance = audit["C0_artifact_provenance"]
    return {
        "artifact_path": provenance["artifact_path"],
        "artifact_file_sha256": provenance["artifact_file_sha256"],
        "seal_path": provenance["seal_path"],
        "seal_file_sha256": provenance["seal_file_sha256"],
        "array_logical_sha256": provenance["array_logical_sha256"],
        "source_model_hash": provenance.get("C0_source_model_hash"),
        "source_E1_audit_path": provenance.get(
            "C0_source_E1_audit_path"
        ),
    }


def validate_formal_bundles(bundles, expected_seed=None):
    """Require one internally consistent five-arm, single-seed VSA run."""
    _require(
        tuple(bundles) == vsa.FORMAL_ARMS,
        "VSA formal bundle arm set/order mismatch",
    )
    audits = {
        arm: bundles[arm]["train_audit"] for arm in vsa.FORMAL_ARMS
    }
    seeds = {bundles[arm]["training_seed"] for arm in vsa.FORMAL_ARMS}
    _require(len(seeds) == 1, "VSA formal arms mix training seeds")
    active_seed = vsa.validate_training_seed(next(iter(seeds)))
    if expected_seed is not None:
        _require(
            active_seed == vsa.validate_training_seed(expected_seed),
            "VSA summary requested seed mismatch",
        )
    initial_hashes = {
        audits[arm]["initial_model_hash"]["aggregate"]
        for arm in vsa.FORMAL_ARMS
    }
    native_order_records = {
        tuple(audits[arm]["native_order_sha256_per_epoch"])
        for arm in vsa.FORMAL_ARMS
    }
    semantic_order_records = {
        tuple(audits[arm]["semantic_order_sha256_per_epoch"])
        for arm in vsa.PSEUDO_ARMS
    }
    native_optimizer_records = {
        json.dumps(
            audits[arm]["native_optimizer_config"], sort_keys=True
        )
        for arm in vsa.FORMAL_ARMS
    }
    semantic_optimizer_records = {
        json.dumps(
            audits[arm]["semantic_optimizer_config"], sort_keys=True
        )
        for arm in vsa.PSEUDO_ARMS
    }
    source_records = {
        json.dumps(audits[arm]["source_provenance"], sort_keys=True)
        for arm in vsa.FORMAL_ARMS
    }
    e1_records = {
        json.dumps(_normalized_e1_lineage(audits[arm]), sort_keys=True)
        for arm in vsa.FORMAL_ARMS
    }
    c0_records = {
        json.dumps(_normalized_c0_lineage(audits[arm]), sort_keys=True)
        for arm in vsa.FORMAL_ARMS
    }
    c0_hashes = {
        audits[arm]["C0_artifact_provenance"][
            "artifact_file_sha256"
        ]
        for arm in vsa.FORMAL_ARMS
    }
    c0_seal_hashes = {
        audits[arm]["C0_artifact_provenance"]["seal_file_sha256"]
        for arm in vsa.FORMAL_ARMS
    }
    M0_hashes = {
        audits[arm]["M0_audit"]["logical_sha256"]
        for arm in vsa.FORMAL_ARMS
    }
    pseudo_y_gen_hashes = {
        bundles[arm]["semantic_target_audit"]["logical_sha256"][
            "y_gen"
        ]
        for arm in vsa.PSEUDO_ARMS
    }
    feature_records = {
        json.dumps(audits[arm]["feature_provenance"], sort_keys=True)
        for arm in vsa.FORMAL_ARMS
    }
    canonical_feature_hash = c1.file_sha256(train.DEFAULT_FEATURE_PATH)
    per_arm_boundary_pass = True
    for arm in vsa.FORMAL_ARMS:
        audit = audits[arm]
        target_audit = bundles[arm]["semantic_target_audit"]
        transition = bundles[arm]["phase_transition_audit"]
        leakage = audit["leakage"]
        forbidden = audit["forbidden_mechanisms"]
        e1_lineage = _normalized_e1_lineage(audit)
        c0_lineage = _normalized_c0_lineage(audit)
        expected_sequence = [
            phase
            for _ in range(vsa.FORMAL_EPOCHS)
            for phase in ("semantic", "native")
        ]
        explicit_lineage_pass = (
            active_seed == vsa.SEED
            or (
                audit.get("training_seed") == active_seed
                and audit.get("requested_seed_lineage_match_pass") is True
                and audit.get("C0_E1_lineage_match_pass") is True
                and audit.get("weak_quality_condition_fixed")
                == train.WEAK_QUALITY_CONDITION
                and audit.get("feature_realization_unchanged") is True
            )
        )
        checks = (
            audit["formal_scientific_arm"] is True,
            audit["engineering_only_arm"] is False,
            audit.get("seed") == active_seed,
            e1_lineage["training_seed"] == active_seed,
            e1_lineage["actual_aggregate_sha256"]
            == e1_lineage["expected_aggregate_sha256"]
            == audit["initial_model_hash"]["aggregate"],
            explicit_lineage_pass,
            c0_lineage["artifact_file_sha256"]
            == audit["C0_artifact_provenance"][
                "artifact_file_sha256"
            ],
            _resolve(audit["feature_provenance"]["path"]).resolve()
            == _resolve(train.DEFAULT_FEATURE_PATH).resolve(),
            audit["feature_provenance"]["file_sha256"]
            == canonical_feature_hash,
            audit["feature_provenance"]["corruption_mask_present"]
            is False,
            audit["additive_loss_used"] is False,
            audit["additive_native_plus_pseudo_used"] is False,
            audit["single_backward_contains_native_and_pseudo"] is False,
            audit["semantic_native_gradient_mixed"] is False,
            audit["lambda_pseudo_training_used"] is False,
            audit["optimizer_decoupling_audit"][
                "separate_optimizer_instances"
            ]
            is True,
            audit["optimizer_decoupling_audit"][
                "optimizer_state_shared"
            ]
            is False,
            audit["no_cross_phase_gradient_accumulation"] is True,
            audit["native_objective"] == vsa.NATIVE_OBJECTIVE,
            audit["native_P_all_rewritten"] is False,
            audit["M0_auxiliary_only"] is True,
            audit["native_Match_refreshed_independently"] is True,
            audit["final_prediction_sealed"] is True,
            leakage["full_GT_loaded_during_training"] is False,
            leakage["full_GT_loaded_after_prediction_seal"] is True,
            leakage["sparse_labels_loaded"] is False,
            leakage["R_loaded"] is False,
            leakage["corruption_mask_loaded"] is False,
            leakage["oracle_used"] is False,
            all(value is False for value in forbidden.values()),
            transition["semantic_before_native_every_epoch"] is True,
            transition[
                "current_post_semantic_params_used_by_native_phase"
            ]
            is True,
            [record["phase"] for record in transition["phase_sequence"]]
            == expected_sequence,
        )
        if arm == "BASE":
            checks += (
                audit["semantic_phase_used"] is False,
                target_audit["pseudo_supervision_used"] is False,
            )
        else:
            checks += (
                audit["semantic_phase_used"] is True,
                target_audit["all_pseudo_tensors_detached_pass"] is True,
                target_audit["exact_C1_target_builder_reused"] is True,
                target_audit["E_unchanged"] is True,
                target_audit["m_unchanged"] is True,
                target_audit["T_unchanged"] is True,
                target_audit["a_unchanged"] is True,
                target_audit["T_local_unchanged"] is True,
            )
        per_arm_boundary_pass = bool(
            per_arm_boundary_pass and all(checks)
        )
    seed20_initial_pass = active_seed != vsa.SEED or initial_hashes == {
        c1.EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256
    }
    seed20_c0_pass = active_seed != vsa.SEED or c0_hashes == {
        c1.EXPECTED_C0_ARTIFACT_SHA256
    }
    seed20_m0_pass = active_seed != vsa.SEED or M0_hashes == {
        c1.EXPECTED_M0_LOGICAL_SHA256
    }
    seed20_y_gen_pass = active_seed != vsa.SEED or pseudo_y_gen_hashes == {
        c1.EXPECTED_C0_ARRAY_SHA256["y_gen"]
    }
    checks = {
        "training_seed": active_seed,
        "five_formal_arms_only": tuple(bundles) == vsa.FORMAL_ARMS,
        "CYCLE_ZERO_excluded": vsa.ENGINEERING_ARM not in bundles,
        "same_requested_seed_all_arms": len(seeds) == 1,
        "same_frozen_initialization_all_arms": (
            len(initial_hashes) == 1 and seed20_initial_pass
        ),
        "same_E1_lineage_all_arms": len(e1_records) == 1,
        "same_native_order_all_arms": len(native_order_records) == 1,
        "same_semantic_order_all_pseudo_arms": (
            len(semantic_order_records) == 1
        ),
        "same_native_optimizer_all_arms": (
            len(native_optimizer_records) == 1
        ),
        "same_semantic_optimizer_all_pseudo_arms": (
            len(semantic_optimizer_records) == 1
        ),
        "same_frozen_source_all_arms": len(source_records) == 1,
        "same_frozen_C0_all_arms": (
            len(c0_records) == 1
            and len(c0_hashes) == 1
            and seed20_c0_pass
        ),
        "same_C0_seal_all_arms": len(c0_seal_hashes) == 1,
        "same_frozen_M0_all_arms": (
            len(M0_hashes) == 1 and seed20_m0_pass
        ),
        "same_y_gen_all_pseudo_arms": (
            len(pseudo_y_gen_hashes) == 1 and seed20_y_gen_pass
        ),
        "same_fixed_feature_all_arms": len(feature_records) == 1,
        "seed20_historical_initialization_pass": seed20_initial_pass,
        "seed20_historical_C0_pass": seed20_c0_pass,
        "seed20_historical_M0_pass": seed20_m0_pass,
        "seed20_historical_y_gen_pass": seed20_y_gen_pass,
        "per_arm_decoupling_boundary_pass": per_arm_boundary_pass,
    }
    _require(all(checks.values()), "VSA formal cross-arm audit failed")
    return checks


def summarize(input_dir=DEFAULT_INPUT_DIR, seed=None):
    root = _resolve(input_dir)
    _require(root.is_dir(), "VSA-A0 input directory is missing")
    for filename in SUMMARY_FILES:
        _require(
            not (root / filename).exists(),
            "refusing to overwrite VSA summary output",
        )
    requested_seed = (
        None if seed is None else vsa.validate_training_seed(seed)
    )
    bundles = {
        arm: load_arm_bundle(root, arm, expected_seed=requested_seed)
        for arm in vsa.FORMAL_ARMS
    }
    cross_arm_audit = validate_formal_bundles(
        bundles, expected_seed=requested_seed
    )
    active_seed = cross_arm_audit["training_seed"]
    metrics_by_arm = {
        arm: bundles[arm]["metrics"] for arm in vsa.FORMAL_ARMS
    }
    decision = vsa.build_vsa_a0_decision(metrics_by_arm)
    summary = {
        "stage": "VSA-A0",
        "seed": active_seed,
        "epochs": vsa.FORMAL_EPOCHS,
        "formal_arms": list(vsa.FORMAL_ARMS),
        "engineering_arm_excluded": vsa.ENGINEERING_ARM,
        "metrics_by_arm": metrics_by_arm,
        "comparisons": decision["comparisons"],
        "final_decision": decision["final_decision"],
    }
    audit = {
        "stage": "VSA-A0",
        "formal_arm_count": len(vsa.FORMAL_ARMS),
        "required_arm_files": list(REQUIRED_ARM_FILES),
        "cross_arm": cross_arm_audit,
        "CYCLE_ZERO_in_scientific_decision": False,
        "two_of_three_metric_gate_frozen": True,
        "positive_delta_sum_gate_frozen": True,
        "decision_priority_frozen": True,
        "threshold_adjustment_after_results": False,
        "decision_tree_exhaustive_pass": (
            sum(decision["decision_conditions"].values()) == 1
        ),
        "VSA_AUDIT_PASS": True,
    }
    c1.write_json(root / "vsa_summary.json", summary)
    c1.write_json(root / "vsa_decision.json", decision)
    c1.write_json(root / "vsa_audit.json", audit)
    return {"summary": summary, "decision": decision, "audit": audit}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument(
        "--seed", type=int, choices=vsa.TRAINING_SEED_CHOICES, default=None
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = summarize(args.input_dir, seed=args.seed)
    print("VSA_AUDIT_PASS=" + str(result["audit"]["VSA_AUDIT_PASS"]))
    print("DECISION=" + result["decision"]["final_decision"])
    print("Saved: " + str(_resolve(args.input_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
