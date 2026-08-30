"""Audit deterministic B6-WQ0 canonical frozen-checkpoint baselines."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import experiments.b6_weak_quality.evaluate_b6_wq0_utility_semantic_admission as b6
from irv.b3_audit import hash_backbone

STAGE = "B6-WQ0-frozen-baseline-provenance"
DEFAULT_OUTPUT_DIR = (
    "outputs/b6_weak_quality/wq0_frozen_baseline_provenance"
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _display(path):
    value = Path(path).resolve()
    try:
        return str(value.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(value)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _all_finite(values):
    for value in values:
        if torch.is_tensor(value):
            finite = bool(torch.isfinite(value).all().item())
        else:
            finite = bool(np.isfinite(np.asarray(value)).all())
        if not finite:
            return False
    return True


def _fusion_update_records(base):
    return [
        {
            "update_id": int(row["update_id"]),
            "www_before": list(row["www_before"]),
            "fused_representation_sha256": row[
                "fused_representation_sha256"
            ],
            "prediction_sha256": row[
                "fused_cluster_assignment_sha256"
            ],
            "www_after": list(row["www_after"]),
        }
        for row in base["fusion_update_audit"]
    ]


def replay_once(seed):
    """Construct and evaluate one completely independent frozen replay."""
    seed = int(seed)
    prepared = b6.prepare_seed(seed)
    base, metrics, historical_audit = b6.run_base(prepared)
    inputs = prepared["fusion_inputs"]
    backbone_after_detail = hash_backbone(
        prepared["models"].autoencoders
    )
    backbone_before_detail = prepared["backbone_hash_before"]
    backbone_frozen_pass = bool(
        backbone_after_detail == backbone_before_detail
    )
    condition = prepared["condition_audit"]
    per_view = inputs["per_view_audit"]
    checkpoint_paths = [Path(path) for path in prepared["checkpoint_paths"]]
    historical_metrics = dict(
        b6.SEED_PROVENANCE[seed]["base_metrics"]
    )
    metric_delta = b6.delta_metrics(metrics, historical_metrics)
    all_tensors_finite = _all_finite(
        inputs["raw_z_views"]
        + inputs["normalized_z_views"]
        + inputs["scaled_z_views"]
        + [
            base["fused_representation"],
            base["predictions"],
            base["original_attention"],
        ]
    )
    canonical_metric_finite = bool(
        all(math.isfinite(value) for value in metrics.values())
    )
    required_hashes = (
        [row["raw_z_sha256"] for row in per_view]
        + [row["normalized_z_sha256"] for row in per_view]
        + [row["original_style_scaled_z_sha256"] for row in per_view]
        + [
            row["checkpoint_native_assignment_sha256"]
            for row in per_view
        ]
        + [
            row["kmeans_initialized_assignment_sha256"]
            for row in per_view
        ]
        + [
            base["historical_style_final_fused_sha256"],
            base["historical_style_prediction_sha256"],
            prepared["z_hash"],
            backbone_before_detail["aggregate"],
        ]
    )
    all_artifacts_and_hashes_exist = bool(
        all(path.is_file() for path in checkpoint_paths)
        and Path(prepared["condition_audit_path"]).is_file()
        and _resolve(b6.SEED_PROVENANCE[seed]["b2_log"]).is_file()
        and all(isinstance(value, str) and len(value) == 64 for value in required_hashes)
    )
    record = {
        "stage": STAGE,
        "baseline_name": "B6-WQ0 Canonical Frozen Baseline",
        "dataset": b6.DATASET_NAME,
        "condition": b6.CONDITION,
        "model_seed": seed,
        "backbone_dir": b6.SEED_PROVENANCE[seed]["backbone_dir"],
        "checkpoint_paths": [_display(path) for path in checkpoint_paths],
        "backbone_hash": backbone_before_detail["aggregate"],
        "backbone_hash_before": backbone_before_detail["aggregate"],
        "backbone_hash_after": backbone_after_detail["aggregate"],
        "backbone_hash_detail_before": backbone_before_detail,
        "backbone_hash_detail_after": backbone_after_detail,
        "backbone_frozen_pass": backbone_frozen_pass,
        "corruption_seed": int(condition["corruption_seed"]),
        "corruption_mode": condition["mode"],
        "snr_db": float(condition["target_snr_db"]),
        "corruption_mask_sha256": condition["mask_sha256"],
        "corrupted_view_sha256": list(
            condition["corrupted_view_sha256"]
        ),
        "raw_z_sha256_per_view": [
            row["raw_z_sha256"] for row in per_view
        ],
        "normalized_z_sha256": prepared["z_hash"],
        "normalized_z_sha256_per_view": [
            row["normalized_z_sha256"] for row in per_view
        ],
        "scaled_z_sha256_per_view": [
            row["original_style_scaled_z_sha256"] for row in per_view
        ],
        "checkpoint_native_assignment_sha256_per_view": [
            row["checkpoint_native_assignment_sha256"]
            for row in per_view
        ],
        "kmeans_initialized_assignment_sha256_per_view": [
            row["kmeans_initialized_assignment_sha256"]
            for row in per_view
        ],
        "prefusion_assignment_disagreement_fraction_per_view": [
            row["assignment_disagreement_fraction"] for row in per_view
        ],
        "T1": int(prepared["fusion_updates"]),
        "fusion_updates": _fusion_update_records(base),
        "final_www_used_for_fusion": list(
            base["final_www_used_for_historical_fusion"]
        ),
        "canonical_fused_representation_sha256": base[
            "historical_style_final_fused_sha256"
        ],
        "canonical_prediction_sha256": base[
            "historical_style_prediction_sha256"
        ],
        "canonical_metrics": metrics,
        "identity_admission_audit": base[
            "identity_admission_audit"
        ],
        "historical_reference_kind": b6.HISTORICAL_REFERENCE_KIND,
        "historical_b2_metrics": historical_metrics,
        "historical_vs_canonical_metric_delta": metric_delta,
        "historical_exact_replay_pass": historical_audit[
            "B6_WQ0_BASE_REPRO_PASS"
        ],
        "labels_used_for_training": False,
        "labels_used_for_admission": False,
        "labels_used_for_evaluation": True,
        "corruption_mask_used_for_admission": False,
        "optimizer_created": False,
        "backward_performed": False,
        "parameter_updates": False,
        "all_tensors_finite": all_tensors_finite,
        "canonical_metric_finite": canonical_metric_finite,
        "all_artifacts_and_hashes_exist": all_artifacts_and_hashes_exist,
        "B6_WQ0_HISTORICAL_B2_REPRO_PASS": historical_audit[
            "B6_WQ0_BASE_REPRO_PASS"
        ],
        "B6_WQ0_IDENTITY_ADMISSION_PASS": base[
            "B6_WQ0_IDENTITY_ADMISSION_PASS"
        ],
        "B6_WQ0_BACKBONE_FROZEN_PASS": backbone_frozen_pass,
    }
    return record


def _repeat_fingerprint(record):
    return {
        "backbone_hash": record["backbone_hash"],
        "raw_z_sha256_per_view": record["raw_z_sha256_per_view"],
        "normalized_z_sha256": record["normalized_z_sha256"],
        "scaled_z_sha256_per_view": record[
            "scaled_z_sha256_per_view"
        ],
        "kmeans_initialized_assignment_sha256_per_view": record[
            "kmeans_initialized_assignment_sha256_per_view"
        ],
        "fusion_updates": record["fusion_updates"],
        "final_www_used_for_fusion": record[
            "final_www_used_for_fusion"
        ],
        "canonical_fused_representation_sha256": record[
            "canonical_fused_representation_sha256"
        ],
        "canonical_prediction_sha256": record[
            "canonical_prediction_sha256"
        ],
        "canonical_metrics": record["canonical_metrics"],
    }


def determinism_audit(first, second):
    checks = {
        "backbone_hash_exact": (
            first["backbone_hash"] == second["backbone_hash"]
        ),
        "raw_z_hashes_exact": (
            first["raw_z_sha256_per_view"]
            == second["raw_z_sha256_per_view"]
        ),
        "normalized_z_hash_exact": (
            first["normalized_z_sha256"]
            == second["normalized_z_sha256"]
        ),
        "scaled_z_hashes_exact": (
            first["scaled_z_sha256_per_view"]
            == second["scaled_z_sha256_per_view"]
        ),
        "checkpoint_native_assignment_hashes_exact": (
            first["checkpoint_native_assignment_sha256_per_view"]
            == second["checkpoint_native_assignment_sha256_per_view"]
        ),
        "kmeans_initialized_assignment_hashes_exact": (
            first["kmeans_initialized_assignment_sha256_per_view"]
            == second["kmeans_initialized_assignment_sha256_per_view"]
        ),
        "www_sequence_exact": (
            [
                (row["www_before"], row["www_after"])
                for row in first["fusion_updates"]
            ]
            == [
                (row["www_before"], row["www_after"])
                for row in second["fusion_updates"]
            ]
        ),
        "fusion_update_hashes_exact": (
            first["fusion_updates"] == second["fusion_updates"]
        ),
        "final_fused_hash_exact": (
            first["canonical_fused_representation_sha256"]
            == second["canonical_fused_representation_sha256"]
        ),
        "final_prediction_hash_exact": (
            first["canonical_prediction_sha256"]
            == second["canonical_prediction_sha256"]
        ),
        "canonical_metrics_exact": (
            first["canonical_metrics"] == second["canonical_metrics"]
        ),
    }
    return {
        "checks": checks,
        "B6_WQ0_FROZEN_BASELINE_DETERMINISTIC_PASS": bool(
            all(checks.values())
        ),
    }


def audit_seed(seed):
    """Run two independent replays and assemble one seed provenance."""
    first = replay_once(seed)
    second = replay_once(seed)
    deterministic = determinism_audit(first, second)
    seed_pass = bool(
        first["backbone_frozen_pass"]
        and second["backbone_frozen_pass"]
        and deterministic[
            "B6_WQ0_FROZEN_BASELINE_DETERMINISTIC_PASS"
        ]
        and first["B6_WQ0_IDENTITY_ADMISSION_PASS"]
        and second["B6_WQ0_IDENTITY_ADMISSION_PASS"]
        and first["all_tensors_finite"]
        and second["all_tensors_finite"]
        and first["canonical_metric_finite"]
        and second["canonical_metric_finite"]
        and first["all_artifacts_and_hashes_exist"]
        and second["all_artifacts_and_hashes_exist"]
        and not first["labels_used_for_training"]
        and not first["labels_used_for_admission"]
        and not first["corruption_mask_used_for_admission"]
    )
    first.update({
        "determinism_audit": deterministic,
        "repeat_replay_fingerprint": _repeat_fingerprint(second),
        "B6_WQ0_FROZEN_BASELINE_DETERMINISTIC_PASS": deterministic[
            "B6_WQ0_FROZEN_BASELINE_DETERMINISTIC_PASS"
        ],
        "B6_WQ0_FROZEN_BASELINE_SEED_PASS": seed_pass,
    })
    return first


def _candidate_entry(record, provenance_path):
    return {
        "model_seed": int(record["model_seed"]),
        "backbone_dir": record["backbone_dir"],
        "expected_backbone_hash": record["backbone_hash"],
        "expected_normalized_z_hash": record[
            "normalized_z_sha256"
        ],
        "expected_canonical_fused_hash": record[
            "canonical_fused_representation_sha256"
        ],
        "expected_prediction_hash": record[
            "canonical_prediction_sha256"
        ],
        "expected_canonical_metrics": record["canonical_metrics"],
        "source_provenance": _display(provenance_path),
    }


def audit_provenance(
    seeds,
    output_dir=DEFAULT_OUTPUT_DIR,
    formal_manifest_path=b6.CANONICAL_MANIFEST_PATH,
):
    seeds = tuple(int(seed) for seed in seeds)
    _require(
        len(seeds) == len(set(seeds))
        and all(seed in b6.SUPPORTED_SEEDS for seed in seeds),
        "seeds must be unique supported model seeds",
    )
    output_root = _resolve(output_dir)
    manifest_path = _resolve(formal_manifest_path)
    _require(manifest_path.is_file(), "formal manifest template missing")
    manifest_hash_before = _file_sha256(manifest_path)
    records = []
    provenance_paths = []
    candidates = {}
    for seed in seeds:
        record = audit_seed(seed)
        provenance_path = output_root / (
            "seed" + str(seed) + "_provenance.json"
        )
        _write_json(provenance_path, record)
        records.append(record)
        provenance_paths.append(provenance_path)
        candidates[str(seed)] = _candidate_entry(
            record, provenance_path
        )
    manifest_hash_after = _file_sha256(manifest_path)
    formal_manifest_modified = bool(
        manifest_hash_before != manifest_hash_after
    )
    exact_seed_set = bool(set(seeds) == set(b6.SUPPORTED_SEEDS))
    provenance_complete = bool(
        exact_seed_set
        and len(records) == len(b6.SUPPORTED_SEEDS)
        and all(
            record["B6_WQ0_FROZEN_BASELINE_SEED_PASS"]
            and record[
                "B6_WQ0_FROZEN_BASELINE_DETERMINISTIC_PASS"
            ]
            and record["all_artifacts_and_hashes_exist"]
            for record in records
        )
        and not formal_manifest_modified
    )
    summary = {
        "stage": STAGE,
        "baseline_name": "B6-WQ0 Canonical Frozen Baseline",
        "dataset": b6.DATASET_NAME,
        "condition": b6.CONDITION,
        "seeds": list(seeds),
        "seed_provenance_paths": [
            _display(path) for path in provenance_paths
        ],
        "canonical_frozen_metrics": {
            str(record["model_seed"]): record["canonical_metrics"]
            for record in records
        },
        "canonical_manifest_candidate_entries": candidates,
        "formal_manifest_path": _display(manifest_path),
        "formal_manifest_sha256_before": manifest_hash_before,
        "formal_manifest_sha256_after": manifest_hash_after,
        "formal_manifest_modified": formal_manifest_modified,
        "B6_WQ0_FROZEN_BASELINE_PROVENANCE_COMPLETE": (
            provenance_complete
        ),
    }
    summary_path = (
        output_root / "b6_wq0_frozen_baseline_summary.json"
    )
    _write_json(summary_path, summary)
    return summary, records


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(b6.SUPPORTED_SEEDS),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary, records = audit_provenance(
        args.seeds,
        output_dir=args.output_dir,
    )
    for record in records:
        seed = record["model_seed"]
        print(
            "seed" + str(seed) + " canonical frozen metrics="
            + str(record["canonical_metrics"])
        )
        print(
            "seed" + str(seed) + " historical B2 metrics="
            + str(record["historical_b2_metrics"])
        )
        print(
            "seed" + str(seed) + " historical delta="
            + str(record["historical_vs_canonical_metric_delta"])
        )
        for flag in (
            "B6_WQ0_IDENTITY_ADMISSION_PASS",
            "B6_WQ0_FROZEN_BASELINE_DETERMINISTIC_PASS",
            "B6_WQ0_FROZEN_BASELINE_SEED_PASS",
        ):
            print(
                "seed" + str(seed) + " " + flag + "="
                + str(record[flag]).lower()
            )
    print(
        "B6_WQ0_FROZEN_BASELINE_PROVENANCE_COMPLETE="
        + str(
            summary[
                "B6_WQ0_FROZEN_BASELINE_PROVENANCE_COMPLETE"
            ]
        ).lower()
    )
    print(
        "formal_manifest_modified="
        + str(summary["formal_manifest_modified"]).lower()
    )
    return summary


if __name__ == "__main__":
    main()
