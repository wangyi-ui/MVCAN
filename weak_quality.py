"""Deterministic weak-quality corruption utilities for Native MVCAN.

The corruption functions intentionally have no label argument. Corruption
selection depends only on dataset dimensions and a local mask RNG; Gaussian
noise depends only on features, the mask, the target SNR, and a separate local
noise RNG.
"""

import csv
import hashlib
import json
import math
import os

import numpy as np


PROTOCOL_VERSION = "b2-fallback-v1"
NOISE_SEED_OFFSET = 1000003
SNR_TOLERANCE_DB = 1e-4
UINT32_MODULUS = 2 ** 32


def ndarray_sha256(array):
    """Hash an ndarray's dtype, shape, and C-contiguous bytes."""
    contiguous = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(size) for size in contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _validate_seed(seed, name):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(name + " must be an integer")
    seed = int(seed)
    if not 0 <= seed < UINT32_MODULUS:
        raise ValueError(name + " must be in [0, 2**32)")
    return seed


def _validate_dimensions(n_samples, n_views, k):
    values = {"n_samples": n_samples, "n_views": n_views, "k": k}
    for name, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(name + " must be an integer")

    n_samples = int(n_samples)
    n_views = int(n_views)
    k = int(k)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_views <= 0:
        raise ValueError("n_views must be positive")
    if not 0 <= k <= n_views:
        raise ValueError("k must satisfy 0 <= k <= n_views")
    return n_samples, n_views, k


def generate_balanced_corruption_mask(n_samples, n_views, k, seed):
    """Generate a deterministic row-exact, column-balanced corruption mask."""
    n_samples, n_views, k = _validate_dimensions(n_samples, n_views, k)
    seed = _validate_seed(seed, "seed")

    rng = np.random.RandomState(seed)
    sample_permutation = rng.permutation(n_samples)
    view_permutation = rng.permutation(n_views)

    mask = np.zeros((n_samples, n_views), dtype=bool)
    for shuffled_rank, sample_index in enumerate(sample_permutation):
        base = (shuffled_rank * k) % n_views
        for offset in range(k):
            view_position = (base + offset) % n_views
            mask[sample_index, view_permutation[view_position]] = True

    row_counts = mask.sum(axis=1, dtype=np.int64)
    view_counts = mask.sum(axis=0, dtype=np.int64)
    balanced = bool(
        np.all(row_counts == k)
        and int(view_counts.max()) - int(view_counts.min()) <= 1
    )
    metadata = {
        "mask_seed": seed,
        "per_sample_corrupted_counts": row_counts.tolist(),
        "per_view_corrupted_counts": view_counts.tolist(),
        "balanced_mask_pass": balanced,
        "mask_sha256": ndarray_sha256(mask),
    }
    return mask, metadata


def _validate_views(X_list):
    if not isinstance(X_list, (list, tuple)) or not X_list:
        raise ValueError("X_list must be a non-empty list or tuple of views")

    clean_views = []
    n_samples = None
    for view_index, view in enumerate(X_list):
        array = np.asarray(view)
        if array.ndim != 2:
            raise ValueError("view " + str(view_index) + " must have shape [N, D]")
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError("view " + str(view_index) + " must have floating dtype")
        if n_samples is None:
            n_samples = array.shape[0]
            if n_samples <= 0:
                raise ValueError("views must contain at least one sample")
        elif array.shape[0] != n_samples:
            raise ValueError("all views must have the same number of samples")
        if array.shape[1] <= 0:
            raise ValueError("views must contain at least one feature")
        if not np.isfinite(array).all():
            raise ValueError("view " + str(view_index) + " contains NaN or Inf")
        clean_views.append(array)
    return clean_views, n_samples


def _mask_statistics(mask, k):
    row_counts = mask.sum(axis=1, dtype=np.int64)
    view_counts = mask.sum(axis=0, dtype=np.int64)
    unique_row_counts = np.unique(row_counts)
    return {
        "total_sample_view_pairs": int(mask.size),
        "corrupted_pair_count": int(mask.sum()),
        "expected_corrupted_pair_count": int(mask.shape[0] * k),
        "per_sample_corrupted_count_min": int(row_counts.min()),
        "per_sample_corrupted_count_max": int(row_counts.max()),
        "per_sample_corrupted_count_unique": [
            int(value) for value in unique_row_counts
        ],
        "per_view_corrupted_counts": [int(value) for value in view_counts],
        "per_view_count_min": int(view_counts.min()),
        "per_view_count_max": int(view_counts.max()),
        "balanced_mask_pass": bool(
            np.all(row_counts == k)
            and int(view_counts.max()) - int(view_counts.min()) <= 1
        ),
    }


def _base_audit(clean_views, corrupted_views, mask, mode, k):
    shape_preserved = all(
        clean.shape == corrupted.shape
        for clean, corrupted in zip(clean_views, corrupted_views)
    )
    dtype_preserved = all(
        clean.dtype == corrupted.dtype
        for clean, corrupted in zip(clean_views, corrupted_views)
    )
    all_finite = all(np.isfinite(view).all() for view in corrupted_views)

    unchanged_error = 0.0
    changed_pairs = 0
    for view_index, (clean, corrupted) in enumerate(
        zip(clean_views, corrupted_views)
    ):
        clean_rows = ~mask[:, view_index]
        if np.any(clean_rows):
            delta = np.abs(
                corrupted[clean_rows].astype(np.float64)
                - clean[clean_rows].astype(np.float64)
            )
            if delta.size:
                unchanged_error = max(unchanged_error, float(delta.max()))

        selected_rows = mask[:, view_index]
        if np.any(selected_rows):
            row_changed = np.any(
                corrupted[selected_rows] != clean[selected_rows], axis=1
            )
            changed_pairs += int(row_changed.sum())

    corrupted_pair_count = int(mask.sum())
    changed_fraction = (
        float(changed_pairs / corrupted_pair_count)
        if corrupted_pair_count
        else 0.0
    )

    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "n_samples": int(mask.shape[0]),
        "n_views": int(mask.shape[1]),
        "corruption_k": int(k),
        **_mask_statistics(mask, k),
        "mask_sha256": ndarray_sha256(mask),
        "clean_view_sha256": [ndarray_sha256(view) for view in clean_views],
        "corrupted_view_sha256": [
            ndarray_sha256(view) for view in corrupted_views
        ],
        "shape_preserved_pass": bool(shape_preserved),
        "dtype_preserved_pass": bool(dtype_preserved),
        "all_finite_pass": bool(all_finite),
        "unchanged_clean_pairs_max_abs_error": float(unchanged_error),
        "corrupted_pairs_changed_fraction": changed_fraction,
        "mask": mask,
        "corruption_pairs": [],
    }
    return audit


def apply_heterogeneous_gaussian_corruption(
    X_list,
    k,
    snr_db,
    corruption_seed,
):
    """Apply balanced sample-view Gaussian noise at per-view aggregate SNR."""
    clean_views, n_samples = _validate_views(X_list)
    n_views = len(clean_views)
    _, _, k = _validate_dimensions(n_samples, n_views, k)
    if k <= 0:
        raise ValueError("heterogeneous_gaussian requires k >= 1")
    corruption_seed = _validate_seed(corruption_seed, "corruption_seed")
    if isinstance(snr_db, (bool, np.bool_)) or not np.isscalar(snr_db):
        raise TypeError("snr_db must be a finite scalar")
    snr_db = float(snr_db)
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")

    mask_seed = corruption_seed
    noise_seed = (corruption_seed + NOISE_SEED_OFFSET) % UINT32_MODULUS
    mask, mask_metadata = generate_balanced_corruption_mask(
        n_samples=n_samples,
        n_views=n_views,
        k=k,
        seed=mask_seed,
    )
    noise_rng = np.random.RandomState(noise_seed)
    target_linear = 10.0 ** (snr_db / 10.0)

    corrupted_views = [np.array(view, copy=True, order="C") for view in clean_views]
    pair_records = []
    per_view_snr = []
    global_signal_energy = 0.0
    global_noise_energy = 0.0

    for view_index, (clean, corrupted) in enumerate(
        zip(clean_views, corrupted_views)
    ):
        selected_rows = np.flatnonzero(mask[:, view_index])
        selected_clean = clean[selected_rows].astype(np.float64)
        signal_power = float(np.mean(np.square(selected_clean, dtype=np.float64)))
        if not math.isfinite(signal_power) or signal_power <= 0.0:
            raise ValueError(
                "selected signal power must be positive for view " + str(view_index)
            )

        raw_noise = noise_rng.normal(size=selected_clean.shape)
        raw_noise_power = float(np.mean(np.square(raw_noise, dtype=np.float64)))
        if not math.isfinite(raw_noise_power) or raw_noise_power <= 0.0:
            raise ValueError("raw Gaussian noise power must be positive")
        noise_scale = math.sqrt(signal_power / (target_linear * raw_noise_power))
        corrupted[selected_rows] = selected_clean + raw_noise * noise_scale

        actual_noise = corrupted[selected_rows].astype(np.float64) - selected_clean
        signal_energy = float(np.sum(np.square(selected_clean, dtype=np.float64)))
        noise_energy = float(np.sum(np.square(actual_noise, dtype=np.float64)))
        if not math.isfinite(noise_energy) or noise_energy <= 0.0:
            raise ValueError(
                "realized Gaussian noise power must be positive for view "
                + str(view_index)
            )
        achieved_snr = 10.0 * math.log10(signal_energy / noise_energy)
        per_view_snr.append(float(achieved_snr))
        global_signal_energy += signal_energy
        global_noise_energy += noise_energy

        for local_index, sample_index in enumerate(selected_rows):
            clean_row = selected_clean[local_index]
            noise_row = actual_noise[local_index]
            pair_signal_power = float(np.mean(np.square(clean_row, dtype=np.float64)))
            pair_noise_power = float(np.mean(np.square(noise_row, dtype=np.float64)))
            if pair_noise_power > 0.0:
                if pair_signal_power > 0.0:
                    pair_snr = 10.0 * math.log10(
                        pair_signal_power / pair_noise_power
                    )
                else:
                    pair_snr = float("-inf")
            else:
                pair_snr = None
            pair_records.append({
                "sample_index_0based": int(sample_index),
                "view_index_0based": int(view_index),
                "view_index_1based": int(view_index + 1),
                "target_snr_db": float(snr_db),
                "achieved_pair_snr_db": pair_snr,
                "signal_power": pair_signal_power,
                "noise_power": pair_noise_power,
            })

    global_snr = 10.0 * math.log10(
        global_signal_energy / global_noise_energy
    )
    audit = _base_audit(
        clean_views=clean_views,
        corrupted_views=corrupted_views,
        mask=mask,
        mode="heterogeneous_gaussian",
        k=k,
    )
    audit.update({
        "corruption_seed": corruption_seed,
        "mask_seed": mask_seed,
        "noise_seed": noise_seed,
        "target_snr_db": snr_db,
        "per_view_aggregate_achieved_snr_db": per_view_snr,
        "global_aggregate_achieved_snr_db": float(global_snr),
        "snr_target_pass": bool(
            all(abs(value - snr_db) < SNR_TOLERANCE_DB for value in per_view_snr)
            and abs(global_snr - snr_db) < SNR_TOLERANCE_DB
        ),
        "no_op_exact_pass": False,
        "corruption_pairs": pair_records,
    })
    if audit["mask_sha256"] != mask_metadata["mask_sha256"]:
        raise RuntimeError("internal mask hash mismatch")
    return corrupted_views, audit


def apply_weak_quality_protocol(
    X_list,
    mode="none",
    k=2,
    snr_db=5.0,
    corruption_seed=None,
):
    """Apply the selected weak-quality protocol without using global RNG state."""
    if mode == "heterogeneous_gaussian":
        if corruption_seed is None:
            raise ValueError(
                "corruption_seed is required for heterogeneous_gaussian"
            )
        return apply_heterogeneous_gaussian_corruption(
            X_list=X_list,
            k=k,
            snr_db=snr_db,
            corruption_seed=corruption_seed,
        )
    if mode != "none":
        raise ValueError("unsupported weak-quality corruption mode: " + str(mode))

    clean_views, n_samples = _validate_views(X_list)
    corrupted_views = list(clean_views)
    mask = np.zeros((n_samples, len(clean_views)), dtype=bool)
    audit = _base_audit(
        clean_views=clean_views,
        corrupted_views=corrupted_views,
        mask=mask,
        mode="none",
        k=0,
    )
    audit.update({
        "corruption_seed": None,
        "mask_seed": None,
        "noise_seed": None,
        "target_snr_db": None,
        "per_view_aggregate_achieved_snr_db": [None] * len(clean_views),
        "global_aggregate_achieved_snr_db": None,
        "snr_target_pass": True,
        "no_op_exact_pass": bool(
            all(
                np.array_equal(clean, corrupted)
                for clean, corrupted in zip(clean_views, corrupted_views)
            )
        ),
    })
    return corrupted_views, audit


def save_corruption_audit(audit, audit_dir):
    """Write JSON metadata, the mask, and one CSV row per corrupted pair."""
    os.makedirs(audit_dir, exist_ok=True)
    mask = np.asarray(audit["mask"], dtype=bool)
    pair_records = audit.get("corruption_pairs", [])

    np.save(os.path.join(audit_dir, "corruption_mask.npy"), mask)

    json_audit = {
        key: value
        for key, value in audit.items()
        if key not in {"mask", "corruption_pairs"}
    }
    with open(
        os.path.join(audit_dir, "corruption_audit.json"),
        "w",
        encoding="utf-8",
    ) as audit_file:
        json.dump(json_audit, audit_file, indent=2, sort_keys=True, allow_nan=False)
        audit_file.write("\n")

    fieldnames = [
        "sample_index_0based",
        "view_index_0based",
        "view_index_1based",
        "target_snr_db",
        "achieved_pair_snr_db",
        "signal_power",
        "noise_power",
    ]
    with open(
        os.path.join(audit_dir, "corruption_pairs.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as pairs_file:
        writer = csv.DictWriter(pairs_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_records)
