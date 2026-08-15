"""Manifest-driven offline B3-A2.2 reliability replication evaluation."""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from configure import get_default_config
from datasets import load_data
from irv.b3_audit import hash_backbone
from irv.b3_predictability_diagnostics import bootstrap_reliability
from irv.b3_predictability_diagnostics import fold_assignment_sha256
from irv.b3_predictability_diagnostics import make_fold_assignment
from irv.b3_predictability_diagnostics import oof_ridge_predictability
from irv.b3_predictability_diagnostics import (
    oof_ridge_predictability_with_shuffled_correspondence,
)
from irv.b3_reliability_replication import all_finite_nested
from irv.b3_reliability_replication import corruption_label_permutation_test
from irv.b3_reliability_replication import oof_target_only_centroid_score
from irv.b3_reliability_replication import paired_score_control
from irv.b3_reliability_replication import per_view_reliability
from irv.b3_semantic_diagnostics import reconstruct_changed_row_mask
from model import MvCAN
from weak_quality import apply_weak_quality_protocol


DATASET_NAME = "MSRC-v1"
EXPECTED_SAMPLE_NUM = 210
EXPECTED_VIEW_NUM = 5
EXPECTED_CLUSTER_NUM = 7
EXPECTED_LATENT_DIM = 10
EXPECTED_CONDITIONS = {
    (20, 20, 2.5),
    (30, 30, 2.5),
    (50, 50, 2.5),
    (20, 20, 5.0),
    (20, 20, 0.0),
}
CONDITION_FIELDS = [
    "condition",
    "snr_db",
    "model_seed",
    "corruption_seed",
    "mask_sha256",
    "clean_mean",
    "corrupted_mean",
    "gap",
    "gap_ci_low",
    "gap_ci_high",
    "auc",
    "auc_ci_low",
    "auc_ci_high",
    "spearman",
    "spearman_ci_low",
    "spearman_ci_high",
    "fold_sha256",
]
PER_VIEW_FIELDS = [
    "condition",
    "model_seed",
    "target_view",
    "clean_count",
    "corrupted_count",
    "clean_mean",
    "corrupted_mean",
    "gap",
    "gap_ci_low",
    "gap_ci_high",
    "auc",
    "auc_ci_low",
    "auc_ci_high",
]


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _all_parameters_have_no_grad(modules):
    return all(
        parameter.grad is None
        for module in modules
        for parameter in module.parameters()
    )


def _validate_source_log(entry):
    log_path = _resolve(entry["source_log"])
    _require(log_path.is_file(), "missing source log: " + str(log_path))
    text = log_path.read_text(encoding="utf-8")
    required_lines = (
        "Reproducibility seed: " + str(entry["model_seed"]),
        "Weak-quality corruption: heterogeneous_gaussian",
        "Model seed: " + str(entry["model_seed"]),
        "Corruption seed: " + str(entry["corruption_seed"]),
        "Target SNR dB: " + str(float(entry["snr_db"])),
        "Per-view corruption counts: [84, 84, 84, 84, 84]",
        "Saving...",
    )
    missing = [line for line in required_lines if line not in text]
    _require(not missing, "source log evidence missing: " + str(missing))
    return {
        "source_log": entry["source_log"],
        "required_evidence_lines": list(required_lines),
        "source_log_validation_pass": True,
        "source_log_sha256": _file_sha256(log_path),
    }


def _validate_manifest(path):
    manifest = _load_json(path)
    _require(isinstance(manifest, list), "condition manifest must be a JSON list")
    required_keys = {
        "condition",
        "dataset",
        "model_seed",
        "corruption_seed",
        "corruption_k",
        "snr_db",
        "backbone_dir",
        "corruption_audit",
        "corruption_mask",
        "source_log",
        "expected_mask_sha256",
    }
    validated = []
    found_conditions = set()
    for entry in manifest:
        _require(isinstance(entry, dict), "manifest entries must be objects")
        _require(required_keys <= set(entry), "manifest entry is missing fields")
        condition_key = (
            int(entry["model_seed"]),
            int(entry["corruption_seed"]),
            float(entry["snr_db"]),
        )
        _require(condition_key not in found_conditions, "duplicate condition")
        found_conditions.add(condition_key)
        _require(entry["dataset"] == DATASET_NAME, "manifest dataset mismatch")
        _require(int(entry["corruption_k"]) == 2, "manifest k must equal 2")

        audit_path = _resolve(entry["corruption_audit"])
        mask_path = _resolve(entry["corruption_mask"])
        backbone_dir = _resolve(entry["backbone_dir"])
        missing_paths = [
            str(value)
            for value in (audit_path, mask_path, backbone_dir)
            if not value.exists()
        ]
        _require(
            not missing_paths,
            "MISSING_CANONICAL_CONDITION "
            + entry["condition"]
            + ": "
            + str(missing_paths),
        )
        checkpoints = [
            backbone_dir / (DATASET_NAME + str(view_id + 1) + "V.pth")
            for view_id in range(EXPECTED_VIEW_NUM)
        ]
        missing_checkpoints = [str(value) for value in checkpoints if not value.is_file()]
        _require(
            not missing_checkpoints,
            "MISSING_CANONICAL_CONDITION "
            + entry["condition"]
            + ": "
            + str(missing_checkpoints),
        )

        audit = _load_json(audit_path)
        audit_match_pass = bool(
            audit.get("dataset") == DATASET_NAME
            and audit.get("model_seed") == entry["model_seed"]
            and audit.get("corruption_seed") == entry["corruption_seed"]
            and audit.get("corruption_k") == entry["corruption_k"]
            and audit.get("target_snr_db") == entry["snr_db"]
            and audit.get("mask_sha256") == entry["expected_mask_sha256"]
            and audit.get("mode") == "heterogeneous_gaussian"
            and audit.get("n_samples") == EXPECTED_SAMPLE_NUM
            and audit.get("n_views") == EXPECTED_VIEW_NUM
            and audit.get("corrupted_pair_count") == 420
            and audit.get("per_view_corrupted_counts") == [84] * 5
            and audit.get("all_finite_pass") is True
            and audit.get("balanced_mask_pass") is True
            and audit.get("snr_target_pass") is True
        )
        _require(audit_match_pass, "canonical audit mismatch for " + entry["condition"])
        source_log_audit = _validate_source_log(entry)
        validated.append({
            **entry,
            "checkpoint_paths": [str(value) for value in checkpoints],
            "checkpoint_file_sha256": [
                _file_sha256(value) for value in checkpoints
            ],
            "corruption_audit_sha256": _file_sha256(audit_path),
            "stored_mask_file_sha256": _file_sha256(mask_path),
            "audit_match_pass": True,
            **source_log_audit,
        })
    missing = EXPECTED_CONDITIONS - found_conditions
    extra = found_conditions - EXPECTED_CONDITIONS
    _require(not missing, "MISSING_CANONICAL_CONDITION: " + str(sorted(missing)))
    _require(not extra, "unexpected canonical condition: " + str(sorted(extra)))
    return sorted(
        validated,
        key=lambda entry: (float(entry["snr_db"]), int(entry["model_seed"])),
    )


def _condition_record(entry, reliability, fold_hash):
    return {
        "condition": entry["condition"],
        "snr_db": float(entry["snr_db"]),
        "model_seed": int(entry["model_seed"]),
        "corruption_seed": int(entry["corruption_seed"]),
        "mask_sha256": entry["expected_mask_sha256"],
        "clean_mean": reliability["clean_predictability_mean"],
        "corrupted_mean": reliability["corrupted_predictability_mean"],
        "gap": reliability["predictability_gap"],
        "gap_ci_low": reliability["gap_ci_low"],
        "gap_ci_high": reliability["gap_ci_high"],
        "auc": reliability["clean_corrupted_auc"],
        "auc_ci_low": reliability["auc_ci_low"],
        "auc_ci_high": reliability["auc_ci_high"],
        "spearman": reliability[
            "spearman_clean_indicator_vs_predictability"
        ],
        "spearman_ci_low": reliability["spearman_ci_low"],
        "spearman_ci_high": reliability["spearman_ci_high"],
        "fold_sha256": fold_hash,
    }


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-manifest", required=True)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    parser.add_argument("--permutation-repeats", type=int, default=2000)
    parser.add_argument("--permutation-seed", type=int, default=20260815)
    parser.add_argument("--shuffle-seed", type=int, default=20260815)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _require(args.oof_folds == 5, "OOF fold count must be 5")
    _require(args.ridge_alpha == 1.0, "Ridge alpha must be 1.0")
    _require(args.bootstrap_repeats == 2000, "bootstrap repeats must be 2000")
    _require(args.bootstrap_seed == 20260815, "bootstrap seed mismatch")
    _require(args.permutation_repeats == 2000, "permutation repeats must be 2000")
    _require(args.permutation_seed == 20260815, "permutation seed mismatch")
    _require(args.shuffle_seed == 20260815, "shuffle seed mismatch")

    manifest = _validate_manifest(args.condition_manifest)
    config = get_default_config(DATASET_NAME)
    config["dataset"] = DATASET_NAME
    clean_views, _ = load_data(config)
    _require(len(clean_views) == EXPECTED_VIEW_NUM, "MSRC-v1 must have 5 views")
    _require(
        all(view.shape[0] == EXPECTED_SAMPLE_NUM for view in clean_views),
        "MSRC-v1 views must contain 210 samples",
    )

    condition_records = []
    condition_audits = []
    raw_conditions = {}
    global_integrity = {
        "manifest_complete_pass": True,
        "canonical_source_audits_pass": True,
        "canonical_source_logs_pass": True,
        "checkpoint_files_complete_pass": True,
        "corruption_masks_exact_pass": True,
        "backbone_immutability_pass": True,
        "eval_mode_pass": True,
        "no_gradients_created_pass": True,
        "oof_train_test_disjoint_pass": True,
        "oof_coverage_pass": True,
        "all_20_ordered_predictors_pass": True,
        "fold_deterministic_pass": True,
        "all_outputs_finite_pass": True,
        "no_optimizer_pass": True,
        "no_backward_pass": True,
        "no_parameter_update_pass": True,
    }

    for entry in manifest:
        evaluation_views, reconstructed_audit = apply_weak_quality_protocol(
            clean_views,
            mode="heterogeneous_gaussian",
            k=entry["corruption_k"],
            snr_db=entry["snr_db"],
            corruption_seed=entry["corruption_seed"],
        )
        reconstructed_mask = reconstruct_changed_row_mask(
            clean_views,
            evaluation_views,
        )
        stored_mask = np.load(_resolve(entry["corruption_mask"]), allow_pickle=False)
        mask_integrity_pass = bool(
            reconstructed_mask.shape == (EXPECTED_SAMPLE_NUM, EXPECTED_VIEW_NUM)
            and np.array_equal(reconstructed_mask, stored_mask)
            and np.array_equal(reconstructed_mask, reconstructed_audit["mask"])
            and reconstructed_audit["mask_sha256"]
            == entry["expected_mask_sha256"]
            and np.all(reconstructed_mask.sum(axis=1) == 2)
            and np.array_equal(reconstructed_mask.sum(axis=0), np.full(5, 84))
        )
        _require(mask_integrity_pass, "mask integrity failed for " + entry["condition"])

        view_sizes = [int(view.shape[1]) for view in evaluation_views]
        models = MvCAN(
            config,
            view_num=EXPECTED_VIEW_NUM,
            view_size=view_sizes,
            n_clusters=EXPECTED_CLUSTER_NUM,
            seed=entry["model_seed"],
            data_size=EXPECTED_SAMPLE_NUM,
            semantic_config=None,
        )
        for view_id, autoencoder in enumerate(models.autoencoders):
            autoencoder.load_state_dict(
                torch.load(entry["checkpoint_paths"][view_id], map_location="cpu"),
                strict=True,
            )
            autoencoder.eval()
        backbone_hash_before = hash_backbone(models.autoencoders)
        eval_mode_pass = all(not module.training for module in models.autoencoders)
        with torch.no_grad():
            native_views = []
            for view_id, autoencoder in enumerate(models.autoencoders):
                features = torch.from_numpy(evaluation_views[view_id]).float()
                latent = autoencoder.encoder(features)
                native_views.append(
                    F.normalize(latent, p=2, dim=1, eps=1e-12)
                )
        representation_shape_pass = bool(
            len(native_views) == EXPECTED_VIEW_NUM
            and all(
                tuple(view.shape) == (EXPECTED_SAMPLE_NUM, EXPECTED_LATENT_DIM)
                for view in native_views
            )
        )
        _require(representation_shape_pass, "Native z shape mismatch")

        folds = make_fold_assignment(
            EXPECTED_SAMPLE_NUM,
            n_splits=args.oof_folds,
            random_state=entry["model_seed"],
        )
        fold_hash = fold_assignment_sha256(folds)
        fold_deterministic_pass = bool(
            fold_hash
            == fold_assignment_sha256(
                make_fold_assignment(
                    EXPECTED_SAMPLE_NUM,
                    n_splits=args.oof_folds,
                    random_state=entry["model_seed"],
                )
            )
        )
        real = oof_ridge_predictability(
            native_views,
            folds,
            alpha=args.ridge_alpha,
        )
        bootstrap = bootstrap_reliability(
            {"z_native": real["oof_consensus_cosine"]},
            reconstructed_mask,
            repeats=args.bootstrap_repeats,
            seed=args.bootstrap_seed,
        )
        reliability = bootstrap["arms"]["z_native"]
        record = _condition_record(entry, reliability, fold_hash)
        condition_records.append(record)

        backbone_hash_after = hash_backbone(models.autoencoders)
        backbone_immutability_pass = backbone_hash_before == backbone_hash_after
        no_gradients_created_pass = _all_parameters_have_no_grad(
            models.autoencoders
        )
        condition_integrity = {
            "condition": entry["condition"],
            "canonical_audit_match_pass": entry["audit_match_pass"],
            "canonical_log_match_pass": entry["source_log_validation_pass"],
            "checkpoint_count_pass": len(entry["checkpoint_paths"]) == 5,
            "corruption_mask_exact_pass": mask_integrity_pass,
            "representation_shape_pass": representation_shape_pass,
            "eval_mode_pass": eval_mode_pass,
            "backbone_immutability_pass": backbone_immutability_pass,
            "no_gradients_created_pass": no_gradients_created_pass,
            "oof_train_test_disjoint_pass": real["audit"][
                "train_test_disjoint_pass"
            ],
            "oof_coverage_pass": real["audit"]["oof_coverage_pass"],
            "all_20_ordered_predictors_pass": (
                real["audit"]["ordered_view_pair_count"] == 20
            ),
            "fold_deterministic_pass": fold_deterministic_pass,
            "all_finite_pass": bool(
                all_finite_nested(record)
                and real["audit"]["all_finite_pass"]
            ),
            "loaded_backbone_hash": backbone_hash_before,
        }
        condition_audits.append({
            **condition_integrity,
            "source": {
                key: value
                for key, value in entry.items()
                if key not in {"checkpoint_paths"}
            },
            "checkpoint_paths": entry["checkpoint_paths"],
            "checkpoint_file_sha256": entry["checkpoint_file_sha256"],
        })
        for global_key, condition_key in (
            ("corruption_masks_exact_pass", "corruption_mask_exact_pass"),
            ("backbone_immutability_pass", "backbone_immutability_pass"),
            ("eval_mode_pass", "eval_mode_pass"),
            ("no_gradients_created_pass", "no_gradients_created_pass"),
            ("oof_train_test_disjoint_pass", "oof_train_test_disjoint_pass"),
            ("oof_coverage_pass", "oof_coverage_pass"),
            ("all_20_ordered_predictors_pass", "all_20_ordered_predictors_pass"),
            ("fold_deterministic_pass", "fold_deterministic_pass"),
            ("all_outputs_finite_pass", "all_finite_pass"),
        ):
            global_integrity[global_key] = bool(
                global_integrity[global_key]
                and condition_integrity[condition_key]
            )
        raw_conditions[(entry["model_seed"], float(entry["snr_db"]))] = {
            "entry": entry,
            "mask": reconstructed_mask,
            "native_views": native_views,
            "folds": folds,
            "real": real,
        }
        print(
            entry["condition"]
            + " AUC=" + str(record["auc"])
            + " gap=" + str(record["gap"])
        )

    reference = raw_conditions[(20, 2.5)]
    real_scores = reference["real"]["oof_consensus_cosine"]
    reference_mask = reference["mask"]
    label_permutation = corruption_label_permutation_test(
        real_scores,
        reference_mask,
        repeats=args.permutation_repeats,
        seed=args.permutation_seed,
    )
    shuffled = oof_ridge_predictability_with_shuffled_correspondence(
        reference["native_views"],
        reference["folds"],
        alpha=args.ridge_alpha,
        shuffle_seed=args.shuffle_seed,
    )
    correspondence_control = paired_score_control(
        real_scores,
        shuffled["oof_consensus_cosine"],
        reference_mask,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
        real_name="real",
        control_name="shuffled",
    )
    correspondence_delta = correspondence_control["real_minus_control"]
    correspondence_control.update({
        "condition": "snr2p5_k2_seed20",
        "shuffle_protocol": "within_training_fold_deterministic_derangement",
        "shuffle_seed": args.shuffle_seed,
        "shuffled_audit": shuffled["audit"],
        "predictability_real_mean": float(np.mean(real_scores)),
        "predictability_shuffled_mean": float(
            np.mean(shuffled["oof_consensus_cosine"])
        ),
        "crossview_dependence_pass": bool(
            correspondence_delta["delta_auc_ci_low"] > 0.0
            or correspondence_delta["delta_gap_ci_low"] > 0.0
        ),
    })

    target_only_scores = np.stack(
        [
            oof_target_only_centroid_score(
                reference["native_views"][view_id],
                reference["folds"],
            )
            for view_id in range(EXPECTED_VIEW_NUM)
        ],
        axis=1,
    )
    target_only_control = paired_score_control(
        real_scores,
        target_only_scores,
        reference_mask,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
        real_name="crossview",
        control_name="target_only",
    )
    target_only_control.update({
        "condition": "snr2p5_k2_seed20",
        "target_only_definition": "cosine_to_OOF_target_view_training_centroid",
        "threshold_tuned": False,
    })

    per_view = per_view_reliability(
        real_scores,
        reference_mask,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    per_view_rows = [
        {
            "condition": "snr2p5_k2_seed20",
            "model_seed": 20,
            **record,
        }
        for record in per_view["records"]
    ]
    controls = {
        "corruption_label_permutation": label_permutation,
        "crossview_correspondence_shuffle": correspondence_control,
        "target_only": target_only_control,
    }
    global_integrity["correspondence_shuffle_train_test_disjoint_pass"] = bool(
        shuffled["audit"]["train_test_disjoint_pass"]
    )
    global_integrity["correspondence_shuffle_zero_fixed_points_pass"] = bool(
        shuffled["audit"]["target_training_fixed_point_count"] == 0
    )
    global_integrity["correspondence_shuffle_within_train_fold_pass"] = bool(
        shuffled["audit"]["target_training_permutation_pass"]
    )
    global_integrity["all_outputs_finite_pass"] = bool(
        global_integrity["all_outputs_finite_pass"]
        and all_finite_nested(controls)
        and all_finite_nested(per_view)
    )
    engineering_pass = all(
        value is True
        for key, value in global_integrity.items()
        if key.endswith("_pass")
    )
    global_integrity["B3_A22_ENGINEERING_PASS"] = bool(engineering_pass)
    _require(engineering_pass, "B3-A2.2 engineering integrity failed")

    output = {
        "stage": "B3-A2.2",
        "scope": "offline_native_z_reliability_replication",
        "primary_score": "oof_consensus_cosine",
        "protocol": {
            "oof_folds": args.oof_folds,
            "oof_shuffle": True,
            "fold_random_state": "model_seed",
            "ridge_alpha": args.ridge_alpha,
            "ridge_fit_intercept": True,
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_unit": "sample_id_with_all_views",
            "permutation_repeats": args.permutation_repeats,
            "permutation_seed": args.permutation_seed,
            "corruption_mask_used_for_evaluation_only": True,
        },
        "condition_manifest": args.condition_manifest,
        "condition_results": condition_records,
        "condition_integrity": condition_audits,
        "per_view": per_view,
        "controls": controls,
        "integrity": global_integrity,
    }
    _require(all_finite_nested(output), "evaluation output contains non-finite values")

    os.makedirs(args.output_dir, exist_ok=True)
    evaluation_path = os.path.join(args.output_dir, "b3_a22_evaluation.json")
    controls_path = os.path.join(args.output_dir, "b3_a22_controls.json")
    _write_json(evaluation_path, output)
    _write_json(controls_path, controls)
    _write_csv(
        os.path.join(args.output_dir, "b3_a22_condition_results.csv"),
        CONDITION_FIELDS,
        condition_records,
    )
    _write_csv(
        os.path.join(args.output_dir, "per_view_reliability.csv"),
        PER_VIEW_FIELDS,
        per_view_rows,
    )
    print("B3_A22_ENGINEERING_PASS=true")
    print("B3-A2.2 evaluation: " + evaluation_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
