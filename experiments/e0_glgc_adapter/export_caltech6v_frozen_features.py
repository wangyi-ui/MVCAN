"""Export frozen Caltech-6V feature artifacts for the released GLGC carrier."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import datasets
import weak_quality


STAGE = "E0-A"
DATASET = "Caltech-6V"
SAMPLE_NUM = 1400
VIEW_NUM = 6
CLASS_NUM = 7
VIEW_DIMS = (48, 40, 254, 1984, 512, 928)
SNR_DB = 2.5
CORRUPTION_K = 3
SEED = 20
NOISE_SEED = 1000023
EXPECTED_MASK_SHA256 = (
    "e6250535db6e33b9263102cf335fcf9e8552a24cae4b70ec5a94887f4e58bf0b"
)
NPZ_KEYS = tuple("X" + str(index) for index in range(1, VIEW_NUM + 1)) + (
    "sample_ids",
)
FORBIDDEN_TRAINABLE_KEYS = {
    "Y",
    "labels",
    "corruption_mask",
    "clean_mask",
    "noise_mask",
    "U",
    "T",
    "quality",
    "reliability",
}
DEFAULT_D1_DIR = (
    REPOSITORY_ROOT / "outputs/d1_caltech6v/snr2p5_k3_seed20"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs/e0_glgc_adapter"


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


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _repository_working_directory():
    previous = Path.cwd()
    os.chdir(REPOSITORY_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def _git_head(repository):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()


def load_preprocessed_caltech():
    """Reuse datasets.py, including its X3-only MinMax preprocessing."""
    with _repository_working_directory():
        views, label_list = datasets.load_data({"dataset": DATASET})
    clean_views = [
        np.ascontiguousarray(view, dtype=np.float32) for view in views
    ]
    _require(len(clean_views) == VIEW_NUM, "Caltech view-count mismatch")
    for view_index, (view, dimension) in enumerate(zip(clean_views, VIEW_DIMS)):
        _require(
            view.shape == (SAMPLE_NUM, dimension)
            and view.dtype == np.dtype(np.float32)
            and np.isfinite(view).all(),
            "Caltech view boundary mismatch at view " + str(view_index + 1),
        )
    _require(len(label_list) == 1, "Caltech label-list boundary mismatch")
    labels = np.asarray(label_list[0], dtype=np.int64)
    _require(
        labels.shape == (SAMPLE_NUM,)
        and np.array_equal(np.unique(labels), np.arange(CLASS_NUM)),
        "Caltech class-count boundary mismatch",
    )
    return clean_views


def load_d1_provenance(d1_dir=DEFAULT_D1_DIR):
    source_root = _resolve(d1_dir)
    audit_path = source_root / "audit/corruption_audit.json"
    mask_path = source_root / "audit/corruption_mask.npy"
    _require(audit_path.is_file(), "missing D1 corruption audit")
    _require(mask_path.is_file(), "missing D1 corruption mask provenance")
    with open(audit_path, "r", encoding="utf-8") as input_file:
        audit = json.load(input_file)
    mask = np.load(mask_path, allow_pickle=False)
    _require(
        mask.shape == (SAMPLE_NUM, VIEW_NUM) and mask.dtype == np.dtype(bool),
        "D1 corruption mask boundary mismatch",
    )
    return {
        "root": source_root,
        "audit_path": audit_path,
        "mask_path": mask_path,
        "audit_file_sha256": file_sha256(audit_path),
        "mask_file_sha256": file_sha256(mask_path),
        "audit": audit,
        "mask": mask,
    }


def build_frozen_feature_sets(d1_dir=DEFAULT_D1_DIR):
    """Replay the frozen protocol and hard-match every D1 content hash."""
    clean_views = load_preprocessed_caltech()
    noisy_views, runtime_audit = weak_quality.apply_weak_quality_protocol(
        X_list=clean_views,
        mode="heterogeneous_gaussian",
        k=CORRUPTION_K,
        snr_db=SNR_DB,
        corruption_seed=SEED,
    )
    noisy_views = [
        np.ascontiguousarray(view, dtype=np.float32) for view in noisy_views
    ]
    d1 = load_d1_provenance(d1_dir)
    d1_audit = d1["audit"]
    clean_hashes = [weak_quality.ndarray_sha256(view) for view in clean_views]
    noisy_hashes = [weak_quality.ndarray_sha256(view) for view in noisy_views]
    runtime_mask = np.asarray(runtime_audit["mask"], dtype=bool)
    runtime_mask_hash = weak_quality.ndarray_sha256(runtime_mask)
    stored_mask_hash = weak_quality.ndarray_sha256(d1["mask"])

    _require(d1_audit.get("dataset") == DATASET, "D1 dataset mismatch")
    _require(d1_audit.get("n_samples") == SAMPLE_NUM, "D1 N mismatch")
    _require(d1_audit.get("n_views") == VIEW_NUM, "D1 V mismatch")
    _require(d1_audit.get("n_clusters") == CLASS_NUM, "D1 K mismatch")
    _require(d1_audit.get("corruption_k") == CORRUPTION_K, "D1 k mismatch")
    _require(d1_audit.get("corruption_seed") == SEED, "D1 seed mismatch")
    _require(d1_audit.get("mask_seed") == SEED, "D1 mask seed mismatch")
    _require(d1_audit.get("noise_seed") == NOISE_SEED, "D1 noise seed mismatch")
    _require(d1_audit.get("target_snr_db") == SNR_DB, "D1 SNR mismatch")
    _require(d1_audit.get("clean_view_sha256") == clean_hashes,
             "clean feature content hash mismatch against D1")
    _require(d1_audit.get("corrupted_view_sha256") == noisy_hashes,
             "noisy feature content hash mismatch against D1")
    _require(
        d1_audit.get("mask_sha256")
        == runtime_mask_hash
        == stored_mask_hash
        == EXPECTED_MASK_SHA256,
        "logical corruption-mask SHA256 mismatch",
    )
    _require(np.array_equal(runtime_mask, d1["mask"]),
             "runtime corruption mask differs from D1")
    _require(
        np.all(runtime_mask.sum(axis=1) == CORRUPTION_K)
        and np.array_equal(
            runtime_mask.sum(axis=0), np.full(VIEW_NUM, SAMPLE_NUM // 2)
        ),
        "runtime corruption mask is not exactly-k3/column-balanced",
    )
    _require(runtime_audit["snr_target_pass"] is True,
             "runtime aggregate SNR audit failed")

    sample_ids = np.arange(SAMPLE_NUM, dtype=np.int64)
    sample_ids.setflags(write=False)
    for view in clean_views + noisy_views:
        view.setflags(write=False)
    return {
        "clean_views": clean_views,
        "noisy_views": noisy_views,
        "sample_ids": sample_ids,
        "clean_hashes": clean_hashes,
        "noisy_hashes": noisy_hashes,
        "sample_ids_hash": weak_quality.ndarray_sha256(sample_ids),
        "runtime_audit": runtime_audit,
        "d1": d1,
    }


def _save_trainable_npz(path, views, sample_ids):
    payload = {
        "X" + str(view_index + 1): view
        for view_index, view in enumerate(views)
    }
    payload["sample_ids"] = sample_ids
    _require(set(payload) == set(NPZ_KEYS), "trainable payload key mismatch")
    _require(not (set(payload) & FORBIDDEN_TRAINABLE_KEYS),
             "oracle/label field reached trainable payload")
    np.savez_compressed(path, **payload)
    with np.load(path, allow_pickle=False) as archive:
        _require(tuple(archive.files) == NPZ_KEYS, "saved NPZ key/order mismatch")
    return file_sha256(path)


def export_frozen_features(
    output_dir=DEFAULT_OUTPUT_DIR,
    d1_dir=DEFAULT_D1_DIR,
):
    output_root = _resolve(output_dir)
    clean_path = output_root / "caltech6v_clean_seed20.npz"
    noisy_path = output_root / "caltech6v_snr2p5_k3_seed20.npz"
    protocol_path = output_root / "export_protocol.json"
    audit_path = output_root / "export_audit.json"
    _require(
        not any(path.exists() for path in (clean_path, noisy_path, protocol_path, audit_path)),
        "refusing to overwrite an E0-A export",
    )

    fixed = build_frozen_feature_sets(d1_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    clean_file_hash = _save_trainable_npz(
        clean_path, fixed["clean_views"], fixed["sample_ids"]
    )
    noisy_file_hash = _save_trainable_npz(
        noisy_path, fixed["noisy_views"], fixed["sample_ids"]
    )

    protocol = {
        "stage": STAGE,
        "scope": "frozen_feature_export_only",
        "source_repository": str(REPOSITORY_ROOT),
        "source_repository_commit": _git_head(REPOSITORY_ROOT),
        "source_files": {
            "datasets.py": file_sha256(REPOSITORY_ROOT / "datasets.py"),
            "weak_quality.py": file_sha256(REPOSITORY_ROOT / "weak_quality.py"),
        },
        "dataset": DATASET,
        "N": SAMPLE_NUM,
        "V": VIEW_NUM,
        "K": CLASS_NUM,
        "dims": list(VIEW_DIMS),
        "preprocessing": "reuse datasets.py exactly; X3 MinMaxScaler only",
        "corruption": {
            "mode": "heterogeneous_gaussian",
            "snr_db": SNR_DB,
            "k": CORRUPTION_K,
            "seed": SEED,
            "mask_seed": SEED,
            "noise_seed": NOISE_SEED,
        },
        "trainable_npz_keys": list(NPZ_KEYS),
        "oracle_mask_in_trainable_artifact": False,
        "labels_in_trainable_artifact": False,
        "D2_or_utility_in_trainable_artifact": False,
    }
    runtime = fixed["runtime_audit"]
    d1 = fixed["d1"]
    audit = {
        "stage": STAGE,
        "dataset": DATASET,
        "sample_ids": {
            "shape": [SAMPLE_NUM],
            "dtype": "int64",
            "sha256": fixed["sample_ids_hash"],
            "exact_arange_1400": True,
        },
        "clean_artifact": {
            "path": _display(clean_path),
            "file_sha256": clean_file_hash,
            "view_content_sha256": fixed["clean_hashes"],
        },
        "noisy_artifact": {
            "path": _display(noisy_path),
            "file_sha256": noisy_file_hash,
            "view_content_sha256": fixed["noisy_hashes"],
        },
        "view_schema": [
            {"key": "X" + str(index + 1), "shape": [SAMPLE_NUM, dimension],
             "dtype": "float32"}
            for index, dimension in enumerate(VIEW_DIMS)
        ],
        "d1_provenance": {
            "root": _display(d1["root"]),
            "audit_path": _display(d1["audit_path"]),
            "audit_file_sha256": d1["audit_file_sha256"],
            "mask_path": _display(d1["mask_path"]),
            "mask_file_sha256": d1["mask_file_sha256"],
            "logical_mask_sha256": EXPECTED_MASK_SHA256,
            "clean_six_view_hash_match": True,
            "noisy_six_view_hash_match": True,
            "mask_exact_match": True,
            "D1_PROVENANCE_MATCH": True,
        },
        "snr_audit": {
            "target_snr_db": SNR_DB,
            "global_aggregate_achieved_snr_db": runtime[
                "global_aggregate_achieved_snr_db"
            ],
            "per_view_aggregate_achieved_snr_db": runtime[
                "per_view_aggregate_achieved_snr_db"
            ],
            "snr_target_pass": runtime["snr_target_pass"],
        },
        "per_view_corrupted_counts": runtime["per_view_corrupted_counts"],
        "per_sample_corrupted_count_unique": runtime[
            "per_sample_corrupted_count_unique"
        ],
        "trainable_artifact_forbidden_fields_absent": True,
        "corruption_location_metadata_isolated_from_trainable_artifacts": True,
        "EXPORT_PASS": True,
    }
    _write_json(protocol_path, protocol)
    _write_json(audit_path, audit)
    return {"protocol": protocol, "audit": audit}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export frozen MVCAN Caltech-6V features for GLGC"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--d1-dir", type=Path, default=DEFAULT_D1_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = export_frozen_features(args.output_dir, args.d1_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
