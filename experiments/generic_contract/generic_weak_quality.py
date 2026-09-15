"""Label-free generic half-corruption primitives."""

import math

import numpy as np

import weak_quality as legacy


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(name + " must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(name + " must be positive")
    return value


def generate_half_corruption_mask(n_samples, n_views, seed):
    """Return a deterministic global-floor(50%) sample-view mask."""
    n_samples = _positive_integer(n_samples, "n_samples")
    n_views = _positive_integer(n_views, "n_views")
    if n_views < 2:
        raise ValueError("n_views must be at least two")
    seed = legacy._validate_seed(seed, "seed")

    if n_views % 2 == 0:
        return legacy.generate_balanced_corruption_mask(
            n_samples, n_views, n_views // 2, seed
        )

    budget = (n_samples * n_views) // 2
    low = n_views // 2
    high = low + 1
    high_row_count = budget - n_samples * low
    rng = np.random.RandomState(seed)
    sample_permutation = rng.permutation(n_samples)
    view_permutation = rng.permutation(n_views)

    mask = np.zeros((n_samples, n_views), dtype=bool)
    cumulative = 0
    for shuffled_rank, sample_index in enumerate(sample_permutation):
        row_count = high if shuffled_rank < high_row_count else low
        base = cumulative % n_views
        for offset in range(row_count):
            mask[sample_index, view_permutation[(base + offset) % n_views]] = True
        cumulative += row_count

    row_counts = mask.sum(axis=1, dtype=np.int64)
    view_counts = mask.sum(axis=0, dtype=np.int64)
    balanced = bool(
        set(row_counts.tolist()).issubset({low, high})
        and int(mask.sum()) == budget
        and int(view_counts.max()) - int(view_counts.min()) <= 1
    )
    if not balanced:
        raise RuntimeError("internal odd-view half-corruption balance failure")
    metadata = {
        "mask_seed": seed,
        "half_corruption_budget": budget,
        "corruption_k_low": low,
        "corruption_k_high": high,
        "high_cardinality_row_count": high_row_count,
        "per_sample_corrupted_counts": row_counts.tolist(),
        "per_view_corrupted_counts": view_counts.tolist(),
        "balanced_mask_pass": True,
        "mask_sha256": legacy.ndarray_sha256(mask),
    }
    return mask, metadata


def _apply_odd_gaussian(clean_views, n_samples, snr_db, corruption_seed):
    n_views = len(clean_views)
    mask, mask_metadata = generate_half_corruption_mask(
        n_samples, n_views, corruption_seed
    )
    noise_seed = (corruption_seed + legacy.NOISE_SEED_OFFSET) % legacy.UINT32_MODULUS
    noise_rng = np.random.RandomState(noise_seed)
    target_linear = 10.0 ** (snr_db / 10.0)
    corrupted_views = [np.array(view, copy=True, order="C") for view in clean_views]
    pair_records = []
    per_view_snr = []
    global_signal_energy = 0.0
    global_noise_energy = 0.0

    for view_index, (clean, corrupted) in enumerate(zip(clean_views, corrupted_views)):
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
                pair_snr = (
                    10.0 * math.log10(pair_signal_power / pair_noise_power)
                    if pair_signal_power > 0.0
                    else float("-inf")
                )
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

    global_snr = 10.0 * math.log10(global_signal_energy / global_noise_energy)
    low = n_views // 2
    high = low + 1
    row_counts = mask.sum(axis=1, dtype=np.int64)
    view_counts = mask.sum(axis=0, dtype=np.int64)
    unchanged_error = 0.0
    changed_pairs = 0
    for view_index, (clean, corrupted) in enumerate(zip(clean_views, corrupted_views)):
        clean_rows = ~mask[:, view_index]
        if np.any(clean_rows):
            delta = np.abs(
                corrupted[clean_rows].astype(np.float64)
                - clean[clean_rows].astype(np.float64)
            )
            if delta.size:
                unchanged_error = max(unchanged_error, float(delta.max()))
        selected_rows = mask[:, view_index]
        changed_pairs += int(
            np.any(corrupted[selected_rows] != clean[selected_rows], axis=1).sum()
        )
    corrupted_count = int(mask.sum())
    audit = {
        "protocol_version": legacy.PROTOCOL_VERSION,
        "mode": "heterogeneous_gaussian_half",
        "n_samples": n_samples,
        "n_views": n_views,
        "corruption_k": None,
        "corruption_k_low": low,
        "corruption_k_high": high,
        "total_sample_view_pairs": int(mask.size),
        "corrupted_pair_count": corrupted_count,
        "expected_corrupted_pair_count": (n_samples * n_views) // 2,
        "per_sample_corrupted_count_min": int(row_counts.min()),
        "per_sample_corrupted_count_max": int(row_counts.max()),
        "per_sample_corrupted_count_unique": [
            int(value) for value in np.unique(row_counts)
        ],
        "per_view_corrupted_counts": [int(value) for value in view_counts],
        "per_view_count_min": int(view_counts.min()),
        "per_view_count_max": int(view_counts.max()),
        "balanced_mask_pass": mask_metadata["balanced_mask_pass"],
        "mask_sha256": legacy.ndarray_sha256(mask),
        "clean_view_sha256": [legacy.ndarray_sha256(view) for view in clean_views],
        "corrupted_view_sha256": [
            legacy.ndarray_sha256(view) for view in corrupted_views
        ],
        "shape_preserved_pass": all(
            clean.shape == corrupted.shape
            for clean, corrupted in zip(clean_views, corrupted_views)
        ),
        "dtype_preserved_pass": all(
            clean.dtype == corrupted.dtype
            for clean, corrupted in zip(clean_views, corrupted_views)
        ),
        "all_finite_pass": all(np.isfinite(view).all() for view in corrupted_views),
        "unchanged_clean_pairs_max_abs_error": float(unchanged_error),
        "corrupted_pairs_changed_fraction": float(changed_pairs / corrupted_count),
        "mask": mask,
        "corruption_seed": corruption_seed,
        "mask_seed": corruption_seed,
        "noise_seed": noise_seed,
        "target_snr_db": snr_db,
        "per_view_aggregate_achieved_snr_db": per_view_snr,
        "global_aggregate_achieved_snr_db": float(global_snr),
        "snr_target_pass": bool(
            all(
                abs(value - snr_db) < legacy.SNR_TOLERANCE_DB
                for value in per_view_snr
            )
            and abs(global_snr - snr_db) < legacy.SNR_TOLERANCE_DB
        ),
        "no_op_exact_pass": False,
        "corruption_pairs": pair_records,
    }
    return corrupted_views, audit


def apply_half_gaussian_corruption(X_list, snr_db, corruption_seed):
    """Apply frozen Gaussian noise physics to exactly floor(N*V/2) pairs."""
    clean_views, n_samples = legacy._validate_views(X_list)
    n_views = len(clean_views)
    if n_views < 2:
        raise ValueError("X_list must contain at least two views")
    corruption_seed = legacy._validate_seed(corruption_seed, "corruption_seed")
    if isinstance(snr_db, (bool, np.bool_)) or not np.isscalar(snr_db):
        raise TypeError("snr_db must be a finite scalar")
    snr_db = float(snr_db)
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if n_views % 2 == 0:
        return legacy.apply_heterogeneous_gaussian_corruption(
            X_list=X_list,
            k=n_views // 2,
            snr_db=snr_db,
            corruption_seed=corruption_seed,
        )
    return _apply_odd_gaussian(clean_views, n_samples, snr_db, corruption_seed)
