"""Audit canonical clean B1 backbone provenance for B5-A0.6 preflight.

This module is deliberately read-only with respect to model state.  It loads
the frozen Native MVCAN checkpoints, extracts normalized Native-z twice, and
writes provenance JSON files plus manifest candidates.  It performs no
training and does not consume labels, corruption masks, Utility, posterior,
prior, or semantic-context data.
"""

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

from configure import get_default_config
from datasets import load_data
from irv.b3_audit import hash_backbone
from irv.b4_information_utility import tensor_view_list_sha256
from irv.b5_shared_semantic_rate import normalize_native_z
from irv.b5_shared_semantic_rate import z_l2_normalization_audit
from model import MvCAN


STAGE = "B5-A0.6-preflight"
PROVENANCE_KIND = "B5-A0.6-clean-provenance"
DATASET_NAME = "MSRC-v1"
CONDITION = "clean"
CORRUPTION_MODE = "none"
SUPPORTED_SEEDS = (20, 30, 50)
EXPECTED_SAMPLE_NUM = 210
EXPECTED_VIEW_NUM = 5
EXPECTED_CLUSTER_NUM = 7
EXPECTED_LATENT_DIM = 10
EXPECTED_HISTORICAL_BACKBONE_HASH = (
    "c4ce4ad8a08bbdf5eeb5b83ec51c0666a5649f7a184b1d8423ccc3688b975426"
)
EXPECTED_HISTORICAL_Z_HASH = (
    "10a5a99dde71de2429b17f7d5bb225be7c0e491df61819e61f9df0a8403f6886"
)
B1_BACKBONE_TEMPLATE = "outputs/b1_msrcv1_clean/seed{seed}/models"
B3_SEED20_BACKBONE_DIR = (
    "outputs/b3_semantic/a1_uniform_clean_seed20/models"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/b5_semantic_rate/a06_clean_b1_provenance"
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def file_sha256(path):
    """Hash the raw bytes of one checkpoint file."""
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def checkpoint_paths_for(backbone_dir):
    """Return the canonical ordered five-view MVCAN checkpoint paths."""
    backbone_dir = _resolve(backbone_dir)
    return [
        backbone_dir / (DATASET_NAME + str(view_id + 1) + "V.pth")
        for view_id in range(EXPECTED_VIEW_NUM)
    ]


def _load_checkpoint(path):
    """Load tensor state only, retaining compatibility with older PyTorch."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_frozen_backbone(backbone_dir, model_seed, clean_views):
    """Load one five-view Native MVCAN backbone without parameter updates."""
    model_seed = int(model_seed)
    _require(model_seed in SUPPORTED_SEEDS, "unsupported model seed")
    _require(len(clean_views) == EXPECTED_VIEW_NUM, "expected five clean views")
    _require(
        all(int(view.shape[0]) == EXPECTED_SAMPLE_NUM for view in clean_views),
        "clean view sample count mismatch",
    )
    paths = checkpoint_paths_for(backbone_dir)
    _require(len(paths) == EXPECTED_VIEW_NUM, "expected five checkpoints")
    missing = [str(path) for path in paths if not path.is_file()]
    _require(not missing, "missing checkpoint files: " + str(missing))

    config = get_default_config(DATASET_NAME)
    config["dataset"] = DATASET_NAME
    models = MvCAN(
        config,
        view_num=EXPECTED_VIEW_NUM,
        view_size=[int(view.shape[1]) for view in clean_views],
        n_clusters=EXPECTED_CLUSTER_NUM,
        seed=model_seed,
        data_size=EXPECTED_SAMPLE_NUM,
        semantic_config=None,
    )
    for view_id, autoencoder in enumerate(models.autoencoders):
        autoencoder.load_state_dict(_load_checkpoint(paths[view_id]), strict=True)
        autoencoder.eval()
        autoencoder.requires_grad_(False)
    return models, paths


def extract_normalized_native_z(autoencoders, clean_views):
    """Use the current B5 L2-normalized Native-z extraction path."""
    _require(len(autoencoders) == EXPECTED_VIEW_NUM, "expected five encoders")
    _require(len(clean_views) == EXPECTED_VIEW_NUM, "expected five clean views")
    with torch.no_grad():
        z_views = []
        for view_id, autoencoder in enumerate(autoencoders):
            features = torch.from_numpy(clean_views[view_id]).float()
            raw_z = autoencoder.encoder(features)
            z_views.append(normalize_native_z(raw_z).detach())
    _require(len(z_views) == EXPECTED_VIEW_NUM, "expected five z views")
    _require(
        all(
            tuple(value.shape) == (EXPECTED_SAMPLE_NUM, EXPECTED_LATENT_DIM)
            for value in z_views
        ),
        "normalized Native-z shape mismatch",
    )
    _require(
        all(not value.requires_grad for value in z_views),
        "normalized Native-z must be detached",
    )
    _require(
        all(bool(torch.isfinite(value).all().item()) for value in z_views),
        "normalized Native-z must be finite",
    )
    return z_views


def max_z_tensor_difference(first, second):
    """Return the maximum elementwise difference across two ordered z lists."""
    _require(len(first) == len(second), "z view count mismatch")
    differences = [
        float(torch.max(torch.abs(a.detach() - b.detach())).item())
        for a, b in zip(first, second)
    ]
    return max(differences, default=0.0)


def audit_one_backbone(backbone_dir, model_seed, clean_views):
    """Audit one frozen backbone and repeat extraction exactly once."""
    models, paths = load_frozen_backbone(
        backbone_dir, model_seed, clean_views
    )
    checkpoint_hashes = [file_sha256(path) for path in paths]
    backbone_hash_first = hash_backbone(models.autoencoders)
    z_first = extract_normalized_native_z(models.autoencoders, clean_views)
    z_hash_first = tensor_view_list_sha256(z_first)
    norm_audit = z_l2_normalization_audit(z_first)

    backbone_hash_repeat = hash_backbone(models.autoencoders)
    z_repeat = extract_normalized_native_z(models.autoencoders, clean_views)
    z_hash_repeat = tensor_view_list_sha256(z_repeat)
    backbone_hash_after = hash_backbone(models.autoencoders)
    max_difference = max_z_tensor_difference(z_first, z_repeat)
    deterministic_pass = bool(
        backbone_hash_first == backbone_hash_repeat == backbone_hash_after
        and z_hash_first == z_hash_repeat
        and max_difference == 0.0
    )
    all_finite = bool(
        all(bool(torch.isfinite(value).all().item()) for value in z_first)
        and math.isfinite(norm_audit["z_l2_norm_max_abs_error_from_one"])
    )
    _require(norm_audit["z_normalization_pass"], "Native-z normalization failed")
    _require(all_finite, "non-finite provenance value")
    _require(deterministic_pass, "non-deterministic provenance extraction")

    return {
        "backbone_dir": _display_path(backbone_dir),
        "checkpoint_paths": [_display_path(path) for path in paths],
        "checkpoint_sha256_per_view": checkpoint_hashes,
        "checkpoint_count": len(paths),
        "all_checkpoint_files_exist": all(path.is_file() for path in paths),
        "backbone_hash": backbone_hash_first,
        "z_shape": [list(value.shape) for value in z_first],
        "z_l2_norm_max_abs_error_from_one": norm_audit[
            "z_l2_norm_max_abs_error_from_one"
        ],
        "z_hash_b4_compatible": z_hash_first,
        "all_values_finite": all_finite,
        "backbone_hash_repeat": backbone_hash_repeat,
        "backbone_hash_after": backbone_hash_after,
        "z_hash_b4_compatible_repeat": z_hash_repeat,
        "max_z_tensor_difference": max_difference,
        "backbone_hash_exact_repeat": bool(
            backbone_hash_first == backbone_hash_repeat
        ),
        "z_hash_exact_repeat": bool(z_hash_first == z_hash_repeat),
        "parameters_unchanged_exact": bool(
            backbone_hash_first == backbone_hash_after
        ),
        "deterministic_pass": deterministic_pass,
    }


def _base_provenance(primary, model_seed):
    result = {
        "stage": STAGE,
        "provenance_kind": PROVENANCE_KIND,
        "dataset": DATASET_NAME,
        "condition": CONDITION,
        "model_seed": int(model_seed),
        "corruption_mode": CORRUPTION_MODE,
        "labels_used": False,
        "corruption_mask_used": False,
        "training_performed": False,
        "backward_performed": False,
        "optimizer_created": False,
        "parameter_updates": False,
    }
    result.update(primary)
    return result


def _seed20_reference_checks(b1, b3):
    checkpoint_exact = bool(
        b1["checkpoint_sha256_per_view"] == b3["checkpoint_sha256_per_view"]
    )
    backbone_exact = bool(b1["backbone_hash"] == b3["backbone_hash"])
    z_exact = bool(
        b1["z_hash_b4_compatible"] == b3["z_hash_b4_compatible"]
    )
    b1_backbone_historical = bool(
        b1["backbone_hash"]["aggregate"]
        == EXPECTED_HISTORICAL_BACKBONE_HASH
    )
    b3_backbone_historical = bool(
        b3["backbone_hash"]["aggregate"]
        == EXPECTED_HISTORICAL_BACKBONE_HASH
    )
    b1_z_historical = bool(
        b1["z_hash_b4_compatible"] == EXPECTED_HISTORICAL_Z_HASH
    )
    b3_z_historical = bool(
        b3["z_hash_b4_compatible"] == EXPECTED_HISTORICAL_Z_HASH
    )
    return {
        "b1_backbone_dir": b1["backbone_dir"],
        "b3_backbone_dir": b3["backbone_dir"],
        "b1_checkpoint_sha256_per_view": b1[
            "checkpoint_sha256_per_view"
        ],
        "b3_checkpoint_sha256_per_view": b3[
            "checkpoint_sha256_per_view"
        ],
        "b1_backbone_hash": b1["backbone_hash"]["aggregate"],
        "b3_backbone_hash": b3["backbone_hash"]["aggregate"],
        "expected_historical_backbone_hash": (
            EXPECTED_HISTORICAL_BACKBONE_HASH
        ),
        "b1_z_hash": b1["z_hash_b4_compatible"],
        "b3_z_hash": b3["z_hash_b4_compatible"],
        "expected_historical_z_hash": EXPECTED_HISTORICAL_Z_HASH,
        "checkpoint_file_exact_match_all_views": checkpoint_exact,
        "backbone_hash_exact_match": backbone_exact,
        "z_hash_exact_match": z_exact,
        "b1_historical_backbone_hash_exact_match": b1_backbone_historical,
        "b3_historical_backbone_hash_exact_match": b3_backbone_historical,
        "b1_historical_z_hash_exact_match": b1_z_historical,
        "b3_historical_z_hash_exact_match": b3_z_historical,
        "B5_A06_B1_B3_SEED20_EQUIVALENCE_PASS": bool(
            backbone_exact and z_exact
        ),
    }


def manifest_candidate_entry(provenance, source_audit):
    """Build, but do not install, one clean B5 manifest candidate."""
    return {
        "condition": CONDITION,
        "dataset": DATASET_NAME,
        "model_seed": provenance["model_seed"],
        "corruption_seed": None,
        "corruption_mode": CORRUPTION_MODE,
        "corruption_k": None,
        "snr_db": None,
        "backbone_dir": provenance["backbone_dir"],
        "checkpoint_paths": provenance["checkpoint_paths"],
        "source_audit": _display_path(source_audit),
        "expected_backbone_hash": provenance["backbone_hash"]["aggregate"],
        "expected_z_hash_b4_compatible": provenance[
            "z_hash_b4_compatible"
        ],
    }


def run_provenance_audit(seeds=SUPPORTED_SEEDS, output_dir=DEFAULT_OUTPUT_DIR):
    """Run the clean B1 provenance audit and write the four requested JSONs."""
    seeds = tuple(int(seed) for seed in seeds)
    _require(seeds == SUPPORTED_SEEDS, "seeds must be exactly 20 30 50")
    output_dir = _resolve(output_dir)

    config = get_default_config(DATASET_NAME)
    config["dataset"] = DATASET_NAME
    # load_data must load the dataset labels internally; only item zero (the
    # original clean feature views) is retained or passed to audit functions.
    clean_views = load_data(config)[0]
    _require(len(clean_views) == EXPECTED_VIEW_NUM, "expected five clean views")
    _require(
        all(
            isinstance(view, np.ndarray)
            and view.dtype == np.float32
            and view.shape[0] == EXPECTED_SAMPLE_NUM
            and bool(np.isfinite(view).all())
            for view in clean_views
        ),
        "invalid clean MSRC-v1 views",
    )

    provenance_by_seed = {}
    provenance_paths = {}
    for seed in seeds:
        backbone_dir = B1_BACKBONE_TEMPLATE.format(seed=seed)
        primary = audit_one_backbone(backbone_dir, seed, clean_views)
        provenance = _base_provenance(primary, seed)
        provenance_path = output_dir / (
            "clean_seed" + str(seed) + "_provenance.json"
        )
        provenance_by_seed[seed] = provenance
        provenance_paths[seed] = provenance_path

    b3_seed20 = audit_one_backbone(
        B3_SEED20_BACKBONE_DIR, 20, clean_views
    )
    reference_checks = _seed20_reference_checks(
        provenance_by_seed[20], b3_seed20
    )
    provenance_by_seed[20].update(reference_checks)
    provenance_by_seed[20]["seed20_reference_checks"] = reference_checks
    provenance_by_seed[20]["b3_reference_audit"] = b3_seed20

    for seed in seeds:
        _write_json(provenance_paths[seed], provenance_by_seed[seed])

    deterministic_pass = bool(
        all(provenance_by_seed[seed]["deterministic_pass"] for seed in seeds)
    )
    equivalence_pass = reference_checks[
        "B5_A06_B1_B3_SEED20_EQUIVALENCE_PASS"
    ]
    b3_historical_reference_pass = bool(
        reference_checks["b3_historical_backbone_hash_exact_match"]
        and reference_checks["b3_historical_z_hash_exact_match"]
    )
    provenance_complete = bool(
        deterministic_pass
        and b3_historical_reference_pass
        and all(
            provenance_by_seed[seed]["checkpoint_count"]
            == EXPECTED_VIEW_NUM
            and provenance_by_seed[seed]["all_checkpoint_files_exist"]
            and provenance_by_seed[seed]["all_values_finite"]
            and provenance_by_seed[seed]["parameters_unchanged_exact"]
            for seed in seeds
        )
    )
    candidates = [
        manifest_candidate_entry(
            provenance_by_seed[seed], provenance_paths[seed]
        )
        for seed in seeds
    ]
    summary = {
        "stage": STAGE,
        "provenance_kind": PROVENANCE_KIND,
        "dataset": DATASET_NAME,
        "condition": CONDITION,
        "seeds": list(seeds),
        "corruption_mode": CORRUPTION_MODE,
        "labels_used": False,
        "corruption_mask_used": False,
        "training_performed": False,
        "backward_performed": False,
        "optimizer_created": False,
        "parameter_updates": False,
        "formal_manifest_modified": False,
        "provenance_files": {
            str(seed): _display_path(provenance_paths[seed]) for seed in seeds
        },
        "seed20_reference_checks": reference_checks,
        "per_seed": {
            str(seed): {
                "backbone_hash": provenance_by_seed[seed]["backbone_hash"],
                "z_hash_b4_compatible": provenance_by_seed[seed][
                    "z_hash_b4_compatible"
                ],
                "z_l2_norm_max_abs_error_from_one": provenance_by_seed[seed][
                    "z_l2_norm_max_abs_error_from_one"
                ],
                "deterministic_pass": provenance_by_seed[seed][
                    "deterministic_pass"
                ],
            }
            for seed in seeds
        },
        "manifest_candidate_entries": candidates,
        "seed20_b3_historical_reference_pass": (
            b3_historical_reference_pass
        ),
        "B5_A06_B1_B3_SEED20_EQUIVALENCE_PASS": equivalence_pass,
        "B5_A06_PROVENANCE_DETERMINISTIC_PASS": deterministic_pass,
        "B5_A06_CLEAN_PROVENANCE_COMPLETE": provenance_complete,
    }
    summary_path = output_dir / "clean_multiseed_provenance_summary.json"
    _write_json(summary_path, summary)
    return summary, provenance_by_seed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit canonical clean B1 provenance for B5-A0.6"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(SUPPORTED_SEEDS)
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _print_results(summary, provenance_by_seed):
    reference = summary["seed20_reference_checks"]
    print("seed20:")
    print("B1 backbone hash=" + reference["b1_backbone_hash"])
    print("B3 backbone hash=" + reference["b3_backbone_hash"])
    print(
        "historical backbone hash="
        + reference["expected_historical_backbone_hash"]
    )
    print("B1 z hash=" + reference["b1_z_hash"])
    print("B3 z hash=" + reference["b3_z_hash"])
    print("historical z hash=" + reference["expected_historical_z_hash"])
    print(
        "checkpoint file exact match="
        + str(reference["checkpoint_file_exact_match_all_views"])
    )
    print("backbone exact match=" + str(reference["backbone_hash_exact_match"]))
    print("z exact match=" + str(reference["z_hash_exact_match"]))
    for seed in (30, 50):
        provenance = provenance_by_seed[seed]
        print("seed" + str(seed) + ":")
        print("backbone hash=" + provenance["backbone_hash"]["aggregate"])
        print("z hash=" + provenance["z_hash_b4_compatible"])
        print(
            "norm error="
            + str(provenance["z_l2_norm_max_abs_error_from_one"])
        )
    for flag in (
        "B5_A06_B1_B3_SEED20_EQUIVALENCE_PASS",
        "B5_A06_PROVENANCE_DETERMINISTIC_PASS",
        "B5_A06_CLEAN_PROVENANCE_COMPLETE",
    ):
        print(flag + "=" + str(summary[flag]))


def main(argv=None):
    args = parse_args(argv)
    summary, provenance_by_seed = run_provenance_audit(
        args.seeds, args.output_dir
    )
    _print_results(summary, provenance_by_seed)
    _require(
        summary["B5_A06_CLEAN_PROVENANCE_COMPLETE"],
        "clean B1 provenance audit did not complete",
    )
    return summary


if __name__ == "__main__":
    main()
