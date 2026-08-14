import csv
import json
import inspect

import numpy as np

from weak_quality import (
    apply_heterogeneous_gaussian_corruption,
    apply_weak_quality_protocol,
    generate_balanced_corruption_mask,
    save_corruption_audit,
)


def _toy_views():
    rng = np.random.RandomState(101)
    dimensions = (24, 57, 31, 19, 26)
    return [
        (rng.normal(size=(210, dimension)) + 0.25).astype(np.float32)
        for dimension in dimensions
    ]


def _assert_balanced(mask):
    assert mask.shape == (210, 5)
    assert mask.dtype == np.bool_
    assert np.array_equal(mask.sum(axis=1), np.full(210, 2))
    assert np.array_equal(mask.sum(axis=0), np.full(5, 84))
    assert int(mask.sum()) == 420


def test_balanced_mask_exact_counts():
    mask, metadata = generate_balanced_corruption_mask(210, 5, 2, 20)

    _assert_balanced(mask)
    assert metadata["balanced_mask_pass"] is True
    assert metadata["per_view_corrupted_counts"] == [84, 84, 84, 84, 84]


def test_same_seed_is_exactly_deterministic():
    views = _toy_views()
    first_views, first_audit = apply_heterogeneous_gaussian_corruption(
        views, k=2, snr_db=5.0, corruption_seed=20
    )
    second_views, second_audit = apply_heterogeneous_gaussian_corruption(
        views, k=2, snr_db=5.0, corruption_seed=20
    )

    assert np.array_equal(first_audit["mask"], second_audit["mask"])
    assert all(
        np.array_equal(first, second)
        for first, second in zip(first_views, second_views)
    )
    assert first_audit["mask_sha256"] == second_audit["mask_sha256"]


def test_different_seeds_change_mask_but_preserve_balance():
    seed20_mask, _ = generate_balanced_corruption_mask(210, 5, 2, 20)
    seed30_mask, _ = generate_balanced_corruption_mask(210, 5, 2, 30)

    assert not np.array_equal(seed20_mask, seed30_mask)
    _assert_balanced(seed20_mask)
    _assert_balanced(seed30_mask)


def test_nondivisible_total_is_balanced_within_one():
    mask, metadata = generate_balanced_corruption_mask(7, 5, 2, 20)
    view_counts = mask.sum(axis=0)

    assert np.array_equal(mask.sum(axis=1), np.full(7, 2))
    assert int(mask.sum()) == 14
    assert int(view_counts.max()) - int(view_counts.min()) <= 1
    assert metadata["balanced_mask_pass"] is True


def test_none_is_exact_identity_with_matching_shape_and_dtype():
    views = _toy_views()
    outputs, audit = apply_weak_quality_protocol(
        views,
        mode="none",
        k=2,
        snr_db=5.0,
        corruption_seed=20,
    )

    for clean, output in zip(views, outputs):
        assert output.shape == clean.shape
        assert output.dtype == clean.dtype
        assert np.array_equal(output, clean)
    assert audit["no_op_exact_pass"] is True
    assert audit["corruption_seed"] is None
    assert audit["corrupted_pair_count"] == 0


def test_clean_pairs_are_bitwise_unchanged():
    views = _toy_views()
    outputs, audit = apply_heterogeneous_gaussian_corruption(
        views, k=2, snr_db=5.0, corruption_seed=20
    )
    mask = audit["mask"]

    for view_index, (clean, output) in enumerate(zip(views, outputs)):
        clean_rows = ~mask[:, view_index]
        assert np.array_equal(output[clean_rows], clean[clean_rows])
    assert audit["unchanged_clean_pairs_max_abs_error"] == 0.0


def test_every_selected_pair_changes():
    views = _toy_views()
    outputs, audit = apply_heterogeneous_gaussian_corruption(
        views, k=2, snr_db=5.0, corruption_seed=20
    )
    mask = audit["mask"]

    for view_index, (clean, output) in enumerate(zip(views, outputs)):
        selected_rows = mask[:, view_index]
        assert np.all(np.any(output[selected_rows] != clean[selected_rows], axis=1))
    assert audit["corrupted_pairs_changed_fraction"] == 1.0


def test_per_view_and_global_aggregate_snr_match_target():
    views = _toy_views()
    _, audit = apply_heterogeneous_gaussian_corruption(
        views, k=2, snr_db=5.0, corruption_seed=20
    )

    assert np.allclose(
        audit["per_view_aggregate_achieved_snr_db"],
        np.full(5, 5.0),
        atol=1e-4,
        rtol=0.0,
    )
    assert abs(audit["global_aggregate_achieved_snr_db"] - 5.0) < 1e-4
    assert audit["snr_target_pass"] is True


def test_corrupted_features_are_finite_and_preserve_dtype():
    views = _toy_views()
    outputs, audit = apply_heterogeneous_gaussian_corruption(
        views, k=2, snr_db=5.0, corruption_seed=20
    )

    assert all(np.isfinite(output).all() for output in outputs)
    assert all(output.dtype == clean.dtype for clean, output in zip(views, outputs))
    assert audit["all_finite_pass"] is True
    assert audit["dtype_preserved_pass"] is True


def test_corruption_does_not_consume_global_numpy_rng():
    views = _toy_views()
    np.random.seed(123)
    expected = np.random.random(12)

    np.random.seed(123)
    apply_heterogeneous_gaussian_corruption(
        views, k=2, snr_db=5.0, corruption_seed=20
    )
    actual_after_corruption = np.random.random(12)

    assert np.array_equal(actual_after_corruption, expected)


def test_none_does_not_consume_global_numpy_rng():
    views = _toy_views()
    np.random.seed(456)
    expected = np.random.random(12)

    np.random.seed(456)
    apply_weak_quality_protocol(views, mode="none")
    actual_after_no_op = np.random.random(12)

    assert np.array_equal(actual_after_no_op, expected)


def test_audit_writer_creates_exact_mask_and_420_csv_rows(tmp_path):
    views = _toy_views()
    _, audit = apply_heterogeneous_gaussian_corruption(
        views, k=2, snr_db=5.0, corruption_seed=20
    )
    save_corruption_audit(audit, str(tmp_path))

    with open(tmp_path / "corruption_audit.json", encoding="utf-8") as audit_file:
        saved_audit = json.load(audit_file)
    saved_mask = np.load(tmp_path / "corruption_mask.npy")
    with open(
        tmp_path / "corruption_pairs.csv", newline="", encoding="utf-8"
    ) as pairs_file:
        saved_pairs = list(csv.DictReader(pairs_file))

    assert saved_audit["corrupted_pair_count"] == 420
    assert np.array_equal(saved_mask, audit["mask"])
    assert len(saved_pairs) == 420


def test_corruption_core_api_has_no_label_argument():
    forbidden_names = {"y", "y_list", "labels", "label"}
    for function in (
        generate_balanced_corruption_mask,
        apply_heterogeneous_gaussian_corruption,
        apply_weak_quality_protocol,
    ):
        assert forbidden_names.isdisjoint(inspect.signature(function).parameters)
