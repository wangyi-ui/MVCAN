"""Targeted isolation tests for B7-A1 frozen sparse semantic anchors."""

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from experiments.b7_sparse_supervision import b7_sparse_anchor_protocol as protocol


def _sparse_inputs(representation_dim=4):
    generator = torch.Generator().manual_seed(9)
    # labeled_repr: [L,V,D]
    labeled_repr = torch.randn(
        protocol.LABELED_NUM,
        protocol.VIEW_NUM,
        representation_dim,
        generator=generator,
    )
    labeled_targets = np.repeat(
        np.arange(protocol.CLASS_NUM, dtype=np.int64), 2
    )
    anchor_mask = np.ones(
        (protocol.LABELED_NUM, protocol.VIEW_NUM), dtype=bool
    )
    return labeled_repr, labeled_targets, anchor_mask


def _utility():
    return np.random.RandomState(4).uniform(
        size=(protocol.LABELED_NUM, protocol.VIEW_NUM)
    )


def test_prototype_tensor_and_count_shapes_are_k_v_d_and_k_v():
    labeled_repr, targets, anchor_mask = _sparse_inputs()
    result = protocol.build_within_view_prototypes(
        labeled_repr, targets, anchor_mask, protocol.CLASS_NUM
    )

    assert result["prototypes"].shape == (7, 6, 4)
    assert result["prototype_count"].shape == (7, 6)
    assert result["prototype_valid_mask"].shape == (7, 6)
    assert result["class_coverage"].shape == (7,)
    assert result["view_coverage"].shape == (6,)
    assert torch.equal(result["prototype_count"], torch.full((7, 6), 2))


def test_missing_prototype_is_invalid_not_zero_but_valid():
    labeled_repr, targets, anchor_mask = _sparse_inputs()
    anchor_mask[targets == 0, 2] = False
    result = protocol.build_within_view_prototypes(
        labeled_repr, targets, anchor_mask, protocol.CLASS_NUM
    )

    assert result["prototype_count"][0, 2].item() == 0
    assert result["prototype_valid_mask"][0, 2].item() is False
    assert torch.equal(result["prototypes"][0, 2], torch.zeros(4))
    assert torch.equal(
        result["prototype_valid_mask"], result["prototype_count"] > 0
    )


def test_query_and_effective_mask_shapes_are_exact():
    labeled_repr, targets, anchor_mask = _sparse_inputs()
    prototype = protocol.build_within_view_prototypes(
        labeled_repr, targets, anchor_mask, protocol.CLASS_NUM
    )
    query_repr = torch.randn(5, 6, 4, generator=torch.Generator().manual_seed(3))
    query_mask = np.zeros((5, 6), dtype=bool)
    query_mask[:, :3] = True
    score = protocol.score_unlabeled_queries(
        query_repr,
        prototype["prototypes"],
        prototype["prototype_valid_mask"],
        query_mask,
    )

    assert query_mask.shape == (5, 6)
    assert score["per_view_scores"].shape == (5, 7, 6)
    assert score["effective_mask"].shape == (5, 7, 6)
    assert score["aggregated_scores"].shape == (5, 7)
    assert score["query_class_score_available"].shape == (5, 7)


def test_missing_prototype_is_excluded_from_aggregation_denominator():
    labeled_repr, targets, anchor_mask = _sparse_inputs()
    anchor_mask[targets == 0, 1] = False
    prototype = protocol.build_within_view_prototypes(
        labeled_repr, targets, anchor_mask, protocol.CLASS_NUM
    )
    query_repr = torch.randn(3, 6, 4, generator=torch.Generator().manual_seed(5))
    query_mask = np.zeros((3, 6), dtype=bool)
    query_mask[:, :3] = True
    score = protocol.score_unlabeled_queries(
        query_repr,
        prototype["prototypes"],
        prototype["prototype_valid_mask"],
        query_mask,
    )

    assert torch.equal(score["effective_view_count"][:, 0], torch.full((3,), 2))
    expected = score["per_view_scores"][:, 0, [0, 2]].mean(dim=1)
    assert torch.allclose(score["aggregated_scores"][:, 0], expected, atol=1e-7)


def test_unavailable_class_uses_nan_and_cannot_win_argmax():
    labeled_repr, targets, anchor_mask = _sparse_inputs()
    anchor_mask[targets == 0, :] = False
    prototype = protocol.build_within_view_prototypes(
        labeled_repr, targets, anchor_mask, protocol.CLASS_NUM
    )
    query_repr = torch.randn(4, 6, 4, generator=torch.Generator().manual_seed(6))
    query_mask = np.zeros((4, 6), dtype=bool)
    query_mask[:, :3] = True
    score = protocol.score_unlabeled_queries(
        query_repr,
        prototype["prototypes"],
        prototype["prototype_valid_mask"],
        query_mask,
    )

    assert not score["query_class_score_available"][:, 0].any()
    assert torch.isnan(score["aggregated_scores"][:, 0]).all()
    assert not torch.any(score["predictions"] == 0)


def test_normal_policy_builder_api_cannot_accept_corruption_mask():
    parameters = inspect.signature(protocol.build_normal_anchor_policies).parameters

    assert set(parameters) == {"labeled_utility", "control_seed"}
    assert all(
        token not in name.lower()
        for name in parameters
        for token in ("oracle", "corrupt", "clean", "mask")
    )


def test_prototype_builder_api_cannot_accept_full_labels():
    parameters = inspect.signature(
        protocol.build_within_view_prototypes
    ).parameters

    assert tuple(parameters) == (
        "labeled_repr",
        "labeled_targets",
        "anchor_mask",
        "num_classes",
    )
    assert all(
        forbidden not in parameters
        for forbidden in (
            "full_targets",
            "evaluation_targets",
            "unlabeled_targets",
        )
    )


def test_u_top3_has_exactly_three_views_per_labeled_row():
    policies = protocol.build_normal_anchor_policies(_utility())
    u_top3 = policies["anchor_masks"]["U_TOP3"]

    assert u_top3.shape == (14, 6)
    assert np.all(u_top3.sum(axis=1) == 3)


def test_shuffled_u_preserves_row_and_per_view_counts():
    policies = protocol.build_normal_anchor_policies(_utility())
    original = policies["anchor_masks"]["U_TOP3"]
    shuffled = policies["anchor_masks"]["SHUFFLED_U"]
    permutation = policies["shuffled_u_row_permutation"]

    assert np.array_equal(shuffled, original[permutation])
    assert np.all(shuffled.sum(axis=1) == 3)
    assert np.array_equal(shuffled.sum(axis=0), original.sum(axis=0))
    assert sorted(permutation.tolist()) == list(range(14))


def test_oracle_uses_exactly_three_clean_views_per_labeled_row():
    corruption_mask = np.zeros((1400, 6), dtype=bool)
    for sample_id in range(1400):
        corruption_mask[sample_id, (sample_id + np.arange(3)) % 6] = True
    labeled_ids = np.arange(14, dtype=np.int64)

    oracle = protocol.build_oracle_anchor_mask(corruption_mask, labeled_ids)

    assert oracle.shape == (14, 6)
    assert np.all(oracle.sum(axis=1) == 3)
    assert np.array_equal(oracle, ~corruption_mask[labeled_ids])


def test_shuffled_label_preserves_exact_sparse_class_histogram():
    targets = np.repeat(np.arange(7, dtype=np.int64), 2)

    shuffled, permutation = protocol.build_shuffled_labeled_targets(targets)

    assert np.array_equal(shuffled, targets[permutation])
    assert np.array_equal(np.bincount(shuffled, minlength=7), np.full(7, 2))
    assert sorted(permutation.tolist()) == list(range(14))


def test_formal_labeled_unlabeled_split_is_disjoint_and_complete():
    split = protocol.load_fixed_label_split()
    labeled = split["labeled_sample_ids"]
    unlabeled = split["unlabeled_sample_ids"]

    assert labeled.shape == (14,)
    assert unlabeled.shape == (1386,)
    assert np.intersect1d(labeled, unlabeled).size == 0
    assert np.array_equal(np.union1d(labeled, unlabeled), np.arange(1400))
    assert split["label_split_sha256"] == protocol.EXPECTED_LABEL_SPLIT_SHA256


class _RawTrackingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.received_by_clustering = None

    def encoder(self, features):
        scale = torch.arange(1, 11, dtype=features.dtype)[None, :]
        return features * scale

    def clustering(self, latent):
        self.received_by_clustering = latent.detach().clone()
        return torch.softmax(latent[:, :7], dim=1)


def test_q_extraction_passes_raw_z_not_normalized_z_to_clustering():
    modules = [_RawTrackingAutoencoder() for _ in range(6)]
    views = [
        np.full((3, 10), view_id + 1.0, dtype=np.float32)
        for view_id in range(6)
    ]

    extracted = protocol.extract_native_z_q(modules, views)

    assert extracted["raw_z_stack"].shape == (3, 6, 10)
    assert extracted["z_stack"].shape == (3, 6, 10)
    assert extracted["q_stack"].shape == (3, 6, 7)
    for view_id, module in enumerate(modules):
        raw = extracted["raw_z_stack"][:, view_id, :]
        normalized = extracted["z_stack"][:, view_id, :]
        assert torch.equal(module.received_by_clustering, raw)
        assert not torch.equal(module.received_by_clustering, normalized)


def test_changing_unlabeled_full_labels_cannot_change_prototypes_or_scores():
    labeled_repr, sparse_targets, anchor_mask = _sparse_inputs()
    query_repr = torch.randn(8, 6, 4, generator=torch.Generator().manual_seed(31))
    query_mask = np.zeros((8, 6), dtype=bool)
    query_mask[:, :3] = True
    first_full_labels = np.arange(8, dtype=np.int64) % 7
    second_full_labels = first_full_labels[::-1].copy()

    first_prototype = protocol.build_within_view_prototypes(
        labeled_repr, sparse_targets, anchor_mask, protocol.CLASS_NUM
    )
    first_score = protocol.score_unlabeled_queries(
        query_repr,
        first_prototype["prototypes"],
        first_prototype["prototype_valid_mask"],
        query_mask,
    )
    second_prototype = protocol.build_within_view_prototypes(
        labeled_repr, sparse_targets, anchor_mask, protocol.CLASS_NUM
    )
    second_score = protocol.score_unlabeled_queries(
        query_repr,
        second_prototype["prototypes"],
        second_prototype["prototype_valid_mask"],
        query_mask,
    )

    assert not np.array_equal(first_full_labels, second_full_labels)
    assert torch.equal(first_prototype["prototypes"], second_prototype["prototypes"])
    assert torch.equal(
        first_score["query_class_score_available"],
        second_score["query_class_score_available"],
    )
    assert torch.allclose(
        first_score["aggregated_scores"],
        second_score["aggregated_scores"],
        equal_nan=True,
    )


def test_frozen_u_only_loader_does_not_return_corruption_mask():
    frozen = protocol.load_frozen_u_only()

    assert set(frozen) == {
        "T",
        "U",
        "T_sha256",
        "U_sha256",
        "utility_file_sha256",
        "utility_path",
        "transfer_path",
    }
    assert frozen["T"].shape == frozen["U"].shape == (1400, 6)
    assert frozen["U"].flags.writeable is False
    assert frozen["U_sha256"] == protocol.EXPECTED_U_SHA256


def test_evaluator_source_has_no_update_or_refit_calls():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "experiments/b7_sparse_supervision/evaluate_b7_a1_sparse_anchor_feasibility.py"
    )
    source = source_path.read_text(encoding="utf-8")

    for forbidden in (
        "torch.optim",
        ".backward(",
        ".train(",
        "compute_transfer_scores(",
        "oof_ridge_predictability(",
        "compute_information_utility(",
    ):
        assert forbidden not in source

def _coverage_matched_u_case():
    labeled_repr, targets, _ = _sparse_inputs()
    utility = np.zeros((protocol.LABELED_NUM, protocol.VIEW_NUM), dtype=np.float64)
    for row_id in range(protocol.LABELED_NUM):
        selected = (row_id + np.arange(protocol.TOP_K)) % protocol.VIEW_NUM
        utility[row_id, selected] = np.array([3.0, 2.0, 1.0])
    u_top3 = protocol.frozen_top3_mask(utility)
    targets_before = targets.copy()
    matched, audit = protocol.build_classwise_swapped_anchor_mask(
        u_top3, targets, protocol.CLASS_NUM
    )
    source_prototype = protocol.build_within_view_prototypes(
        labeled_repr, targets, u_top3, protocol.CLASS_NUM
    )
    matched_prototype = protocol.build_within_view_prototypes(
        labeled_repr, targets, matched, protocol.CLASS_NUM
    )
    return {
        "targets": targets,
        "targets_before": targets_before,
        "source_mask": u_top3,
        "matched_mask": matched,
        "audit": audit,
        "source_prototype": source_prototype,
        "matched_prototype": matched_prototype,
    }


def test_u_class_shuffled_prototype_count_is_elementwise_identical():
    case = _coverage_matched_u_case()

    assert torch.equal(
        case["source_prototype"]["prototype_count"],
        case["matched_prototype"]["prototype_count"],
    )


def test_u_class_shuffled_prototype_valid_mask_is_identical():
    case = _coverage_matched_u_case()

    assert torch.equal(
        case["source_prototype"]["prototype_valid_mask"],
        case["matched_prototype"]["prototype_valid_mask"],
    )


def test_u_class_shuffled_class_view_coverage_is_identical():
    case = _coverage_matched_u_case()

    assert (
        case["source_prototype"]["class_view_coverage_rate"]
        == case["matched_prototype"]["class_view_coverage_rate"]
    )
    assert torch.equal(
        case["source_prototype"]["class_coverage"],
        case["matched_prototype"]["class_coverage"],
    )
    assert torch.equal(
        case["source_prototype"]["view_coverage"],
        case["matched_prototype"]["view_coverage"],
    )


def test_u_class_shuffled_keeps_exactly_three_views_per_row():
    case = _coverage_matched_u_case()

    assert np.all(case["source_mask"].sum(axis=1) == 3)
    assert np.all(case["matched_mask"].sum(axis=1) == 3)


def test_u_class_shuffled_does_not_change_sparse_targets():
    case = _coverage_matched_u_case()

    assert np.array_equal(case["targets"], case["targets_before"])
    assert case["audit"]["targets_unchanged"] is True


def test_u_class_shuffled_changes_rows_when_same_class_masks_differ():
    case = _coverage_matched_u_case()

    for class_id in range(protocol.CLASS_NUM):
        rows = np.flatnonzero(case["targets"] == class_id)
        assert not np.array_equal(
            case["source_mask"][rows[0]], case["source_mask"][rows[1]]
        )
        assert np.array_equal(
            case["matched_mask"][rows[0]], case["source_mask"][rows[1]]
        )
        assert np.array_equal(
            case["matched_mask"][rows[1]], case["source_mask"][rows[0]]
        )
    assert case["audit"]["changed_row_count"] == protocol.LABELED_NUM
    assert case["audit"]["identical_mask_pair_count"] == 0
    assert len(case["audit"]["classwise_swap_audit"]) == protocol.CLASS_NUM


def test_oracle_class_shuffled_is_coverage_matched_and_informative():
    labeled_repr, targets, _ = _sparse_inputs()
    corruption_mask = np.zeros((protocol.SAMPLE_NUM, protocol.VIEW_NUM), dtype=bool)
    for sample_id in range(protocol.SAMPLE_NUM):
        corruption_mask[sample_id, (sample_id + np.arange(3)) % 6] = True
    labeled_ids = np.arange(protocol.LABELED_NUM, dtype=np.int64)
    oracle = protocol.build_oracle_anchor_mask(corruption_mask, labeled_ids)
    matched, audit = protocol.build_classwise_swapped_anchor_mask(
        oracle, targets, protocol.CLASS_NUM
    )
    source_prototype = protocol.build_within_view_prototypes(
        labeled_repr, targets, oracle, protocol.CLASS_NUM
    )
    matched_prototype = protocol.build_within_view_prototypes(
        labeled_repr, targets, matched, protocol.CLASS_NUM
    )

    assert np.all(oracle.sum(axis=1) == 3)
    assert np.all(matched.sum(axis=1) == 3)
    assert np.array_equal(targets, np.repeat(np.arange(7), 2))
    assert torch.equal(
        source_prototype["prototype_count"],
        matched_prototype["prototype_count"],
    )
    assert torch.equal(
        source_prototype["prototype_valid_mask"],
        matched_prototype["prototype_valid_mask"],
    )
    assert (
        source_prototype["class_view_coverage_rate"]
        == matched_prototype["class_view_coverage_rate"]
    )
    assert audit["changed_row_count"] > 0
    assert audit["targets_unchanged"] is True


def test_classwise_swap_hard_fails_when_every_pair_is_identical():
    targets = np.repeat(np.arange(7, dtype=np.int64), 2)
    anchor_mask = np.zeros((14, 6), dtype=bool)
    anchor_mask[:, :3] = True

    with pytest.raises(RuntimeError, match="uninformative"):
        protocol.build_classwise_swapped_anchor_mask(
            anchor_mask, targets, protocol.CLASS_NUM
        )


def test_read_only_numpy_inputs_are_copied_only_at_torch_boundary():
    import warnings

    labeled_repr, targets, anchor_mask = _sparse_inputs()
    labeled_numpy = labeled_repr.numpy().copy()
    labeled_numpy.setflags(write=False)
    targets.setflags(write=False)
    anchor_mask.setflags(write=False)
    query_repr = np.ones((3, 6, 4), dtype=np.float32)
    query_mask = np.zeros((3, 6), dtype=bool)
    query_mask[:, :3] = True
    query_repr.setflags(write=False)
    query_mask.setflags(write=False)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        prototype = protocol.build_within_view_prototypes(
            labeled_numpy, targets, anchor_mask, protocol.CLASS_NUM
        )
        protocol.score_unlabeled_queries(
            query_repr,
            prototype["prototypes"],
            prototype["prototype_valid_mask"],
            query_mask,
        )

    assert all("not writable" not in str(item.message) for item in captured)

