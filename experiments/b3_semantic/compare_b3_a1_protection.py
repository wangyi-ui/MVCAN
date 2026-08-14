"""Strict B3-A1 backbone-protection and semantic-update comparator."""

import argparse
import json
import math
import sys


CORRUPTION_FIELDS = (
    "corruption_protocol_version",
    "corruption_mode",
    "corruption_seed",
    "corruption_k",
    "target_snr_db",
    "corruption_mask_sha256",
)


def _load_json(path):
    with open(path, "r") as audit_file:
        audit = json.load(audit_file)
    if not isinstance(audit, dict):
        raise ValueError(path + " must contain a JSON object")
    return audit


def _finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _hash_exists(hash_record, expected_views=None):
    if not isinstance(hash_record, dict):
        return False
    aggregate = hash_record.get("aggregate")
    if not isinstance(aggregate, str) or not aggregate:
        return False
    if expected_views is None:
        return True
    per_view = hash_record.get("per_view")
    return bool(
        isinstance(per_view, list)
        and len(per_view) == expected_views
        and all(isinstance(value, str) and value for value in per_view)
    )


def compare_audits(off_audit, uniform_audit):
    """Return exact backbone checks and strict semantic-learning checks."""
    off_backbone = off_audit.get("backbone_hash") or {}
    uniform_backbone = uniform_audit.get("backbone_hash") or {}
    runtime = uniform_audit.get("runtime") or {}
    view_num = uniform_audit.get("view_num")
    expected_pair_count = (
        view_num * (view_num - 1) // 2
        if isinstance(view_num, int) and view_num >= 2
        else None
    )

    protection_checks = {
        "stage_match": (
            off_audit.get("stage") == "B3-A1"
            and uniform_audit.get("stage") == "B3-A1"
        ),
        "dataset_match": off_audit.get("dataset") == uniform_audit.get("dataset"),
        "seed_match": off_audit.get("model_seed") == uniform_audit.get("model_seed"),
        "corruption_match": all(
            off_audit.get(field) == uniform_audit.get(field)
            for field in CORRUPTION_FIELDS
        ),
        "native_metrics_exact": (
            off_audit.get("native_metrics")
            == uniform_audit.get("native_metrics")
        ),
        "backbone_aggregate_hash_exact": (
            _hash_exists(off_backbone)
            and off_backbone.get("aggregate")
            == uniform_backbone.get("aggregate")
        ),
        "backbone_per_view_hash_exact": (
            isinstance(off_backbone.get("per_view"), list)
            and len(off_backbone["per_view"]) == off_audit.get("view_num")
            and off_backbone.get("per_view")
            == uniform_backbone.get("per_view")
        ),
    }

    loss_values = [
        runtime.get("semantic_loss_first"),
        runtime.get("semantic_loss_last"),
        runtime.get("semantic_loss_mean"),
        runtime.get("semantic_loss_min"),
        runtime.get("semantic_loss_max"),
    ]
    initial_hash = uniform_audit.get("semantic_hash_initial")
    final_hash = uniform_audit.get("semantic_hash_final")
    initial_aggregate = (
        initial_hash.get("aggregate") if isinstance(initial_hash, dict) else None
    )
    final_aggregate = (
        final_hash.get("aggregate") if isinstance(final_hash, dict) else None
    )
    grad_l2_last = runtime.get("semantic_grad_l2_last")
    grad_l2_mean = runtime.get("semantic_grad_l2_mean")

    semantic_checks = {
        "uniform_semantic_enabled": (
            uniform_audit.get("semantic_mode") == "uniform"
            and uniform_audit.get("semantic_enabled") is True
        ),
        "uniform_hyperparameters": (
            uniform_audit.get("semantic_dim")
            == uniform_audit.get("latent_dim")
            and uniform_audit.get("semantic_lr") == 1e-4
            and uniform_audit.get("semantic_temperature") == 0.2
        ),
        "semantic_optimizer_ran": (
            runtime.get("semantic_optimizer_steps", 0) > 0
            and runtime.get("semantic_forward_calls", 0) > 0
        ),
        "semantic_pair_count": (
            expected_pair_count is not None
            and runtime.get("semantic_pair_count") == expected_pair_count
        ),
        "semantic_loss_finite": (
            runtime.get("semantic_loss_finite_pass") is True
            and all(_finite_number(value) for value in loss_values)
        ),
        "semantic_gradient_finite_nonzero": (
            runtime.get("semantic_grad_finite_pass") is True
            and runtime.get("semantic_grad_nonzero_pass") is True
            and runtime.get("semantic_all_heads_grad_pass") is True
            and _finite_number(grad_l2_last)
            and grad_l2_last > 0.0
            and _finite_number(grad_l2_mean)
            and grad_l2_mean > 0.0
        ),
        "semantic_head_updated": (
            _hash_exists(initial_hash, expected_views=view_num)
            and _hash_exists(final_hash, expected_views=view_num)
            and initial_aggregate != final_aggregate
            and runtime.get("semantic_head_hash_initial") == initial_aggregate
            and runtime.get("semantic_head_hash_final") == final_aggregate
            and runtime.get("semantic_head_updated_pass") is True
        ),
        "semantic_detach_runtime": (
            runtime.get("semantic_input_detached_pass") is True
            and runtime.get("semantic_shape_pass") is True
            and runtime.get("semantic_finite_pass") is True
            and runtime.get("semantic_norm_finite_pass") is True
        ),
    }
    return protection_checks, semantic_checks


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--off_audit", required=True)
    parser.add_argument("--uniform_audit", required=True)
    args = parser.parse_args(argv)

    try:
        off_audit = _load_json(args.off_audit)
        uniform_audit = _load_json(args.uniform_audit)
        protection_checks, semantic_checks = compare_audits(
            off_audit,
            uniform_audit,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print("B3-A1 PROTECTION AUDIT")
        print("audit_load: FAIL (" + str(error) + ")")
        print("B3_A1_BACKBONE_PROTECTION_PASS=false")
        print("B3_A1_SEMANTIC_UPDATE_PASS=false")
        return 1

    print("B3-A1 PROTECTION AUDIT")
    print("")
    for name, passed in protection_checks.items():
        print(name + ": " + ("PASS" if passed else "FAIL"))
    for name, passed in semantic_checks.items():
        print(name + ": " + ("PASS" if passed else "FAIL"))

    protection_pass = all(protection_checks.values())
    semantic_pass = all(semantic_checks.values())
    print("")
    print(
        "B3_A1_BACKBONE_PROTECTION_PASS="
        + ("true" if protection_pass else "false")
    )
    print(
        "B3_A1_SEMANTIC_UPDATE_PASS="
        + ("true" if semantic_pass else "false")
    )
    return 0 if protection_pass and semantic_pass else 1


if __name__ == "__main__":
    sys.exit(main())
