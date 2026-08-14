"""Strictly compare B3-A0 semantic-OFF and DETACHED audit artifacts."""

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


def _semantic_hash_exists(audit):
    semantic_hash = audit.get("semantic_hash")
    if not isinstance(semantic_hash, dict):
        return False
    per_view = semantic_hash.get("per_view")
    return bool(
        isinstance(semantic_hash.get("aggregate"), str)
        and semantic_hash["aggregate"]
        and isinstance(per_view, list)
        and len(per_view) == audit.get("view_num")
        and all(isinstance(value, str) and value for value in per_view)
    )


def compare_audits(off_audit, detached_audit):
    """Return named exact-protection checks for two loaded audit objects."""
    off_backbone = off_audit.get("backbone_hash") or {}
    detached_backbone = detached_audit.get("backbone_hash") or {}
    runtime = detached_audit.get("runtime") or {}

    norm_values = (
        runtime.get("semantic_norm_mean"),
        runtime.get("semantic_norm_min"),
        runtime.get("semantic_norm_max"),
    )
    norm_scalars_finite = all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in norm_values
    )

    checks = {
        "dataset_match": off_audit.get("dataset") == detached_audit.get("dataset"),
        "seed_match": off_audit.get("model_seed") == detached_audit.get("model_seed"),
        "corruption_match": all(
            off_audit.get(field) == detached_audit.get(field)
            for field in CORRUPTION_FIELDS
        ),
        "native_metrics_exact": (
            off_audit.get("native_metrics")
            == detached_audit.get("native_metrics")
        ),
        "backbone_aggregate_hash_exact": (
            isinstance(off_backbone.get("aggregate"), str)
            and off_backbone.get("aggregate")
            == detached_backbone.get("aggregate")
        ),
        "backbone_per_view_hash_exact": (
            isinstance(off_backbone.get("per_view"), list)
            and off_backbone.get("per_view")
            == detached_backbone.get("per_view")
        ),
        "semantic_off_contract": (
            off_audit.get("semantic_mode") == "off"
            and off_audit.get("semantic_enabled") is False
            and off_audit.get("semantic_hash") is None
            and (off_audit.get("runtime") or {}).get("semantic_forward_calls") == 0
        ),
        "semantic_dimensions": (
            detached_audit.get("semantic_dim")
            == detached_audit.get("latent_dim")
            and detached_audit.get("view_num")
            == len((detached_audit.get("semantic_hash") or {}).get("per_view", []))
        ),
        "semantic_detach_runtime": (
            detached_audit.get("semantic_mode") == "detached"
            and detached_audit.get("semantic_enabled") is True
            and runtime.get("semantic_forward_calls", 0) > 0
            and runtime.get("semantic_input_detached_pass") is True
        ),
        "semantic_shape": runtime.get("semantic_shape_pass") is True,
        "semantic_finite": (
            runtime.get("semantic_finite_pass") is True
            and runtime.get("semantic_norm_finite_pass") is True
            and norm_scalars_finite
        ),
        "semantic_hash_exists": _semantic_hash_exists(detached_audit),
    }
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--off_audit", required=True)
    parser.add_argument("--detached_audit", required=True)
    args = parser.parse_args(argv)

    try:
        off_audit = _load_json(args.off_audit)
        detached_audit = _load_json(args.detached_audit)
        checks = compare_audits(off_audit, detached_audit)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print("B3-A0 IDENTITY AUDIT")
        print("audit_load: FAIL (" + str(error) + ")")
        print("B3_A0_EXACT_PROTECTION_PASS=false")
        return 1

    print("B3-A0 IDENTITY AUDIT")
    print("")
    for name, passed in checks.items():
        print(name + ": " + ("PASS" if passed else "FAIL"))

    protection_pass = all(checks.values())
    print("")
    print(
        "B3_A0_EXACT_PROTECTION_PASS="
        + ("true" if protection_pass else "false")
    )
    return 0 if protection_pass else 1


if __name__ == "__main__":
    sys.exit(main())
