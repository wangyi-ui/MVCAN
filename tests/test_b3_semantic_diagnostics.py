"""Synthetic unit tests for the pure B3-A2 semantic diagnostics."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from irv.b3_audit import hash_semantic_heads
from irv.b3_semantic_diagnostics import agreement_diagnostic
from irv.b3_semantic_diagnostics import collapse_diagnostic
from irv.b3_semantic_diagnostics import correspondence_diagnostic
from irv.b3_semantic_diagnostics import effective_rank
from irv.b3_semantic_diagnostics import noise_separation_diagnostic
from irv.b3_semantic_diagnostics import reconstruct_changed_row_mask
from irv.semantic_head import DetachedSemanticHeadBank


def test_effective_rank_rank_one_and_multidimensional():
    coefficients = torch.arange(1, 9, dtype=torch.float64).unsqueeze(1)
    direction = torch.arange(1, 6, dtype=torch.float64).unsqueeze(0)
    rank_one = coefficients @ direction
    multidimensional = torch.eye(8, dtype=torch.float64)

    rank_one_value = effective_rank(rank_one)
    multidimensional_value = effective_rank(multidimensional)

    assert rank_one_value == pytest.approx(1.0, abs=1e-8)
    assert multidimensional_value > rank_one_value


def test_identical_views_have_unit_same_sample_agreement():
    base = F.normalize(torch.arange(1, 41).reshape(8, 5).float(), dim=1)
    agreement, summary = agreement_diagnostic([base.clone() for _ in range(5)])

    assert tuple(agreement.shape) == (8, 5)
    assert torch.allclose(agreement, torch.ones_like(agreement), atol=1e-6)
    assert summary["agreement_mean"] == pytest.approx(1.0, abs=1e-6)


def test_aligned_correspondence_has_positive_gap():
    aligned = torch.eye(8)
    result = correspondence_diagnostic([aligned, aligned.clone()])

    assert result["correspondence_gap"] > 0.0
    assert result["retrieval_top1"] == pytest.approx(1.0)


def test_permuted_correspondence_reduces_gap_and_retrieval():
    aligned = torch.eye(8)
    baseline = correspondence_diagnostic([aligned, aligned.clone()])
    permuted = correspondence_diagnostic([aligned, torch.roll(aligned, 1, dims=0)])

    assert permuted["correspondence_gap"] < baseline["correspondence_gap"]
    assert permuted["retrieval_top1"] < baseline["retrieval_top1"]


def test_collapse_diagnostic_distinguishes_constant_and_diverse_embeddings():
    constant = torch.ones(16, 6)
    diverse = torch.eye(8)

    constant_result = collapse_diagnostic(constant)
    diverse_result = collapse_diagnostic(diverse)

    assert constant_result["effective_rank"] == pytest.approx(1.0)
    assert constant_result["variance_mean"] == pytest.approx(0.0)
    assert diverse_result["effective_rank"] > constant_result["effective_rank"]
    assert diverse_result["variance_mean"] > constant_result["variance_mean"]


def test_reconstruct_changed_row_mask_is_exact():
    clean = [np.zeros((4, 3)), np.ones((4, 2))]
    corrupted = [view.copy() for view in clean]
    corrupted[0][1, 2] = 3.0
    corrupted[0][3, 0] = -1.0
    corrupted[1][0, 1] = 5.0
    expected = np.array(
        [
            [False, True],
            [True, False],
            [False, False],
            [True, False],
        ]
    )

    actual = reconstruct_changed_row_mask(clean, corrupted)

    assert actual.dtype == np.bool_
    assert np.array_equal(actual, expected)


def test_clean_corrupted_agreement_auc_and_spearman_have_positive_direction():
    corruption_mask = np.array(
        [[False, True], [False, True], [False, True], [False, True]]
    )
    agreement = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]]
    )

    result = noise_separation_diagnostic(agreement, corruption_mask)

    assert result["clean_corrupted_auc"] > 0.5
    assert result["spearman_clean_indicator_vs_agreement"] > 0.0
    assert result["agreement_gap_clean_minus_corrupted"] > 0.0


def test_semantic_hash_same_seed_matches_and_different_seed_differs():
    kwargs = {
        "view_num": 5,
        "latent_dim": 10,
        "semantic_dim": 10,
    }
    first = DetachedSemanticHeadBank(semantic_seed=1020, **kwargs)
    repeated = DetachedSemanticHeadBank(semantic_seed=1020, **kwargs)
    different = DetachedSemanticHeadBank(semantic_seed=1021, **kwargs)

    assert hash_semantic_heads(first) == hash_semantic_heads(repeated)
    assert hash_semantic_heads(first) != hash_semantic_heads(different)
