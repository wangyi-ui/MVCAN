"""One-shot BDGP feature/noise/sparse-label materialization boundary."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from weak_quality import ndarray_sha256

from .dataset_contract import infer_dataset_contract
from .generic_final_core_adapter import BDGP_RUNTIME_SPEC, SPLIT_FIELDS
from .generic_weak_quality import (
    apply_half_gaussian_corruption,
    generate_half_corruption_mask,
)
from .sparse_label_contract import materialize_hash_ranked_sparse_split


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "BDGP"
DATASET_PATH = REPOSITORY_ROOT / "data/BDGP2V_N.mat"
DATASET_SHA256 = "998b6d22fc5142857f957496023d2c51254dfe09ac0402a520e53d69576b0a0f"
FEATURE_FIELDS = BDGP_RUNTIME_SPEC.feature_fields


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    target = Path(path)
    _require(not target.exists(), "refusing to overwrite " + str(target))
    with open(target, "x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _array_record(value):
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "logical_sha256": ndarray_sha256(array),
        "finite": bool(np.isfinite(array).all()),
    }


def load_authoritative_bdgp():
    """Load exactly the two native views and labels; X3 is never requested."""
    mat = sio.loadmat(DATASET_PATH, variable_names=["X1", "X2", "Y"])
    _require(all(name in mat for name in ("X1", "X2", "Y")), "BDGP fields missing")
    clean_views = [
        np.ascontiguousarray(np.asarray(mat["X1"]).astype("float32")),
        np.ascontiguousarray(np.asarray(mat["X2"]).astype("float32")),
    ]
    labels = np.ascontiguousarray(np.squeeze(np.asarray(mat["Y"])), dtype=np.int64)
    _require(
        clean_views[0].shape == (2500, 1750)
        and clean_views[1].shape == (2500, 79)
        and labels.shape == (2500,)
        and all(np.isfinite(view).all() for view in clean_views)
        and np.array_equal(np.unique(labels), np.arange(5, dtype=np.int64))
        and np.array_equal(
            np.bincount(labels, minlength=5), np.full(5, 500, dtype=np.int64)
        ),
        "authoritative BDGP data contract mismatch",
    )
    return clean_views, labels


def materialize(
    *,
    dataset,
    label_seed,
    labels_per_class,
    corruption_seed,
    snr_db,
    output_dir,
):
    spec = BDGP_RUNTIME_SPEC
    _require(dataset == DATASET_NAME, "dataset must be canonical BDGP")
    _require(int(label_seed) == 20, "BDGP label seed must be 20")
    _require(int(labels_per_class) == 2, "BDGP labels_per_class must be 2")
    _require(int(corruption_seed) == 20, "BDGP corruption seed must be 20")
    _require(float(snr_db) == 2.5, "BDGP SNR must be 2.5 dB")
    target = Path(output_dir)
    _require(not target.exists(), "materialization output already exists")
    _require(
        DATASET_PATH.is_file() and file_sha256(DATASET_PATH) == DATASET_SHA256,
        "BDGP dataset SHA256 mismatch",
    )

    clean_views, labels = load_authoritative_bdgp()
    contract = infer_dataset_contract(
        clean_views,
        dataset_name=DATASET_NAME,
        K=spec.K,
        labels_per_class=labels_per_class,
    )
    _require(
        (
            contract.N,
            contract.V,
            contract.K,
            contract.L,
            contract.N_u,
            contract.S,
            contract.view_dims,
        )
        == (2500, 2, 5, 10, 2490, 2, (1750, 79)),
        "BDGP dataset contract mismatch",
    )

    corrupted_views, weak_audit = apply_half_gaussian_corruption(
        clean_views, snr_db, corruption_seed
    )
    regenerated_mask, regenerated_metadata = generate_half_corruption_mask(
        contract.N, contract.V, corruption_seed
    )
    mask = np.ascontiguousarray(weak_audit["mask"], dtype=np.bool_)
    row_counts = mask.sum(axis=1, dtype=np.int64)
    view_counts = mask.sum(axis=0, dtype=np.int64)
    _require(
        np.array_equal(mask, regenerated_mask)
        and weak_audit["mask_sha256"] == regenerated_metadata["mask_sha256"]
        and mask.shape == (2500, 2)
        and np.array_equal(row_counts, np.ones(2500, dtype=np.int64))
        and int(mask.sum()) == 2500
        and mask.size == 5000
        and int(view_counts.max()) - int(view_counts.min()) <= 1
        and weak_audit["shape_preserved_pass"] is True
        and weak_audit["dtype_preserved_pass"] is True
        and weak_audit["all_finite_pass"] is True
        and weak_audit["unchanged_clean_pairs_max_abs_error"] == 0.0
        and weak_audit["snr_target_pass"] is True,
        "BDGP weak-quality materialization mismatch",
    )

    split = materialize_hash_ranked_sparse_split(
        labels,
        dataset_name=DATASET_NAME,
        label_seed=label_seed,
        labels_per_class=labels_per_class,
    )
    _require(
        split.labeled_ids.shape == split.labeled_targets.shape == (10,)
        and split.unlabeled_ids.shape == (2490,)
        and np.array_equal(
            np.bincount(split.labeled_targets, minlength=5),
            np.full(5, 2, dtype=np.int64),
        ),
        "BDGP sparse split mismatch",
    )

    target.mkdir(parents=True)
    audit_dir = target / "audit"
    audit_dir.mkdir()
    feature_path = target / spec.feature_artifact_name
    split_path = target / spec.split_artifact_name
    split_seal_path = target / spec.split_seal_name
    mask_path = audit_dir / "corruption_mask.npy"
    weak_audit_path = audit_dir / "corruption_audit.json"
    audit_path = target / "materialization_audit.json"
    seal_path = target / "materialization_seal.json"

    feature_payload = {
        "X1": np.ascontiguousarray(corrupted_views[0]),
        "X2": np.ascontiguousarray(corrupted_views[1]),
        "sample_ids": np.arange(contract.N, dtype=np.int64),
    }
    _require(tuple(feature_payload) == FEATURE_FIELDS, "feature field whitelist mismatch")
    np.savez(feature_path, **feature_payload)

    split_payload = {
        "labeled_ids": np.array(split.labeled_ids, copy=True),
        "labeled_targets": np.array(split.labeled_targets, copy=True),
        "unlabeled_ids": np.array(split.unlabeled_ids, copy=True),
    }
    _require(tuple(split_payload) == SPLIT_FIELDS, "split field whitelist mismatch")
    np.savez(split_path, **split_payload)
    np.save(mask_path, mask, allow_pickle=False)

    split_seal = {
        "dataset": DATASET_NAME,
        "label_seed": int(label_seed),
        "labels_per_class": int(labels_per_class),
        "split_sha256": split.split_sha256,
        "artifact_path": str(split_path),
        "artifact_file_sha256": file_sha256(split_path),
        "fields": list(SPLIT_FIELDS),
        "arrays": {name: _array_record(value) for name, value in split_payload.items()},
        "full_GT_persisted": False,
        "split_seal_valid": True,
    }
    _write_json(split_seal_path, split_seal)

    serializable_weak_audit = {
        key: value for key, value in weak_audit.items() if key != "mask"
    }
    _write_json(weak_audit_path, serializable_weak_audit)

    materialization_audit = {
        "stage": "G0-B1-MATERIALIZATION",
        "dataset": DATASET_NAME,
        "dataset_path": str(DATASET_PATH.relative_to(REPOSITORY_ROOT)),
        "dataset_file_sha256": DATASET_SHA256,
        "dataset_contract": {
            "N": contract.N,
            "V": contract.V,
            "K": contract.K,
            "L": contract.L,
            "N_u": contract.N_u,
            "S": contract.S,
            "view_dims": list(contract.view_dims),
        },
        "authoritative_view_policy": ["X1", "X2"],
        "X3_ignored": True,
        "feature_fields": list(FEATURE_FIELDS),
        "feature_arrays": {
            name: _array_record(value) for name, value in feature_payload.items()
        },
        "trainable_feature_artifact_path": str(feature_path),
        "trainable_feature_artifact_file_sha256": file_sha256(feature_path),
        "sparse_split_path": str(split_path),
        "sparse_split_file_sha256": file_sha256(split_path),
        "sparse_split_seal_path": str(split_seal_path),
        "sparse_split_seal_file_sha256": file_sha256(split_seal_path),
        "split_sha256": split.split_sha256,
        "labeled_ids": split.labeled_ids.tolist(),
        "labeled_targets": split.labeled_targets.tolist(),
        "unlabeled_count": int(split.unlabeled_ids.size),
        "mask_path": str(mask_path),
        "mask_file_sha256": file_sha256(mask_path),
        "mask_logical_sha256": ndarray_sha256(mask),
        "mask_deterministic_regeneration_equal": True,
        "mask_row_count_histogram": {"1": 2500},
        "per_view_corrupted_counts": view_counts.tolist(),
        "corrupted_pair_count": int(mask.sum()),
        "total_sample_view_pairs": int(mask.size),
        "corrupted_pair_ratio": float(mask.sum() / mask.size),
        "weak_quality_audit_path": str(weak_audit_path),
        "weak_quality_audit_file_sha256": file_sha256(weak_audit_path),
        "target_snr_db": float(snr_db),
        "per_view_aggregate_achieved_snr_db": weak_audit[
            "per_view_aggregate_achieved_snr_db"
        ],
        "global_aggregate_achieved_snr_db": weak_audit[
            "global_aggregate_achieved_snr_db"
        ],
        "full_GT_loaded_only_for_sparse_split_materialization": True,
        "full_GT_persisted": False,
        "GT_used_for_feature_corruption": False,
        "GT_used_for_training": False,
    }
    _write_json(audit_path, materialization_audit)
    seal = {
        "stage": "G0-B1-MATERIALIZATION",
        "dataset": DATASET_NAME,
        "dataset_file_sha256": DATASET_SHA256,
        "materialization_seal_valid": True,
        "trainable_feature_artifact_file_sha256": file_sha256(feature_path),
        "sparse_split_file_sha256": file_sha256(split_path),
        "sparse_split_seal_file_sha256": file_sha256(split_seal_path),
        "corruption_mask_file_sha256": file_sha256(mask_path),
        "corruption_audit_file_sha256": file_sha256(weak_audit_path),
        "materialization_audit_file_sha256": file_sha256(audit_path),
        "feature_fields": list(FEATURE_FIELDS),
        "split_fields": list(SPLIT_FIELDS),
        "full_GT_persisted": False,
        "full_GT_available_to_training_runner": False,
    }
    _write_json(seal_path, seal)
    return seal


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--label-seed", type=int, required=True)
    parser.add_argument("--labels-per-class", type=int, required=True)
    parser.add_argument("--corruption-seed", type=int, required=True)
    parser.add_argument("--snr-db", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seal = materialize(
        dataset=args.dataset,
        label_seed=args.label_seed,
        labels_per_class=args.labels_per_class,
        corruption_seed=args.corruption_seed,
        snr_db=args.snr_db,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "stage": seal["stage"],
        "dataset": seal["dataset"],
        "materialization_seal_valid": seal["materialization_seal_valid"],
        "full_GT_persisted": seal["full_GT_persisted"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
