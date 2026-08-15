"""Pure offline shared-semantic diagnostics for B3-A2."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import roc_auc_score

from ClusteringTest import acc as clustering_accuracy


PERCENTILES = (10, 25, 50, 75, 90)


def _validate_semantic_views(semantic_views):
    if not isinstance(semantic_views, (list, tuple)) or len(semantic_views) < 2:
        raise ValueError("semantic_views must contain at least two views")
    first = semantic_views[0]
    if not torch.is_tensor(first) or first.ndim != 2:
        raise ValueError("each semantic view must have shape [N, semantic_dim]")
    expected_shape = tuple(first.shape)
    if expected_shape[0] <= 0 or expected_shape[1] <= 0:
        raise ValueError("N and semantic_dim must be positive")
    for semantic_view in semantic_views:
        if not torch.is_tensor(semantic_view) or semantic_view.ndim != 2:
            raise ValueError("each semantic view must have shape [N, semantic_dim]")
        if tuple(semantic_view.shape) != expected_shape:
            raise ValueError("all semantic views must have the same shape")
        if semantic_view.device != first.device:
            raise ValueError("all semantic views must be on the same device")
        if not bool(torch.isfinite(semantic_view).all().item()):
            raise ValueError("semantic views must contain only finite values")
    return expected_shape


def fuse_semantic_views(semantic_views):
    """Mean per-view semantics and L2-normalize the fused representation."""
    _validate_semantic_views(semantic_views)
    # stacked: [N, view_num, semantic_dim]
    stacked = torch.stack(list(semantic_views), dim=1)
    # semantic_mean: [N, semantic_dim]
    semantic_mean = stacked.mean(dim=1)
    # semantic_fused: [N, semantic_dim]
    semantic_fused = F.normalize(semantic_mean, p=2, dim=1, eps=1e-12)
    return semantic_fused


def clustering_diagnostic(embedding, labels, n_clusters, random_state=20):
    """Evaluate a fixed embedding with deterministic KMeans and labels."""
    if not torch.is_tensor(embedding) or embedding.ndim != 2:
        raise ValueError("embedding must have shape [N, semantic_dim]")
    features = embedding.detach().cpu().numpy()
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (features.shape[0],):
        raise ValueError("labels must have shape [N]")
    if not np.isfinite(features).all():
        raise ValueError("embedding must contain only finite values")
    predictions = KMeans(
        n_clusters=int(n_clusters),
        n_init=100,
        random_state=int(random_state),
    ).fit_predict(features)
    return {
        "acc": float(clustering_accuracy(labels, predictions)),
        "nmi": float(normalized_mutual_info_score(labels, predictions)),
        "ari": float(adjusted_rand_score(labels, predictions)),
    }


def agreement_diagnostic(semantic_views):
    """Return the [N,V] sample-view agreement matrix and scalar summary."""
    _validate_semantic_views(semantic_views)
    # stacked: [N, view_num, semantic_dim]
    stacked = torch.stack(list(semantic_views), dim=1)
    # pairwise: [N, view_num, view_num]
    pairwise = torch.matmul(stacked, stacked.transpose(1, 2))
    view_num = int(stacked.shape[1])
    diagonal = torch.diagonal(pairwise, dim1=1, dim2=2)
    # agreement: [N, view_num]
    agreement = (pairwise.sum(dim=2) - diagonal) / (view_num - 1)
    flat = agreement.detach().cpu().numpy().astype(np.float64).reshape(-1)
    percentiles = np.percentile(flat, PERCENTILES)
    summary = {
        "agreement_mean": float(np.mean(flat)),
        "agreement_std": float(np.std(flat)),
        "agreement_min": float(np.min(flat)),
        "agreement_max": float(np.max(flat)),
        "agreement_p10": float(percentiles[0]),
        "agreement_p25": float(percentiles[1]),
        "agreement_p50": float(percentiles[2]),
        "agreement_p75": float(percentiles[3]),
        "agreement_p90": float(percentiles[4]),
    }
    return agreement, summary


def correspondence_diagnostic(semantic_views):
    """Measure matched-vs-mismatched cosine separation and retrieval@1."""
    sample_num, _ = _validate_semantic_views(semantic_views)
    view_num = len(semantic_views)
    identity = torch.eye(
        sample_num,
        dtype=torch.bool,
        device=semantic_views[0].device,
    )
    same_sample_values = []
    mismatched_values = []
    retrieval_correct = 0
    retrieval_total = 0

    for view_v in range(view_num):
        for view_w in range(view_v + 1, view_num):
            # similarities_vw: [N, N]
            similarities_vw = torch.matmul(
                semantic_views[view_v],
                semantic_views[view_w].T,
            )
            same_sample_values.append(torch.diagonal(similarities_vw))
            mismatched_values.append(similarities_vw[~identity])

            # retrieved_vw/retrieved_wv: [N]
            retrieved_vw = similarities_vw.argmax(dim=1)
            retrieved_wv = similarities_vw.argmax(dim=0)
            target = torch.arange(sample_num, device=similarities_vw.device)
            retrieval_correct += int((retrieved_vw == target).sum().item())
            retrieval_correct += int((retrieved_wv == target).sum().item())
            retrieval_total += 2 * sample_num

    same_sample = torch.cat(same_sample_values)
    mismatched = torch.cat(mismatched_values)
    same_mean = float(same_sample.mean().item())
    mismatched_mean = float(mismatched.mean().item())
    return {
        "same_sample_cosine_mean": same_mean,
        "mismatched_cosine_mean": mismatched_mean,
        "correspondence_gap": same_mean - mismatched_mean,
        "retrieval_top1": float(retrieval_correct / retrieval_total),
        "unordered_view_pair_count": view_num * (view_num - 1) // 2,
        "retrieval_direction_count": view_num * (view_num - 1),
    }


def effective_rank(embedding, eps=1e-12):
    """Compute entropy effective rank from squared centered singular values."""
    if not torch.is_tensor(embedding) or embedding.ndim != 2:
        raise ValueError("embedding must have shape [N, semantic_dim]")
    if not bool(torch.isfinite(embedding).all().item()):
        raise ValueError("embedding must contain only finite values")
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    squared = singular_values.pow(2)
    total = squared.sum()
    if float(total.item()) <= eps:
        return 1.0
    probabilities = squared / total
    entropy = -(probabilities * torch.log(probabilities + eps)).sum()
    return float(torch.exp(entropy).item())


def collapse_diagnostic(embedding):
    """Audit variance, norms, pairwise cosines, and effective rank."""
    if not torch.is_tensor(embedding) or embedding.ndim != 2:
        raise ValueError("embedding must have shape [N, semantic_dim]")
    if embedding.shape[0] < 2:
        raise ValueError("collapse diagnostic requires at least two samples")
    if not bool(torch.isfinite(embedding).all().item()):
        raise ValueError("embedding must contain only finite values")

    variance = torch.var(embedding, dim=0, unbiased=False)
    feature_norms = torch.linalg.vector_norm(embedding, ord=2, dim=1)
    normalized = F.normalize(embedding, p=2, dim=1, eps=1e-12)
    pairwise_cosine = torch.matmul(normalized, normalized.T)
    off_diagonal_mask = ~torch.eye(
        embedding.shape[0],
        dtype=torch.bool,
        device=embedding.device,
    )
    off_diagonal = pairwise_cosine[off_diagonal_mask]
    result = {
        "variance_per_dim": [float(value) for value in variance.tolist()],
        "variance_mean": float(variance.mean().item()),
        "variance_min": float(variance.min().item()),
        "variance_max": float(variance.max().item()),
        "feature_norm_mean": float(feature_norms.mean().item()),
        "feature_norm_std": float(feature_norms.std(unbiased=False).item()),
        "off_diagonal_pairwise_cosine_mean": float(
            off_diagonal.mean().item()
        ),
        "off_diagonal_pairwise_cosine_std": float(
            off_diagonal.std(unbiased=False).item()
        ),
        "effective_rank": effective_rank(embedding),
    }
    result["all_finite_pass"] = all_finite_nested(result)
    return result


def reconstruct_changed_row_mask(clean_views, corrupted_views):
    """Recover a read-only [N,V] mask from exact row changes."""
    if len(clean_views) != len(corrupted_views) or not clean_views:
        raise ValueError("clean and corrupted view lists must align")
    columns = []
    sample_num = None
    for clean_view, corrupted_view in zip(clean_views, corrupted_views):
        clean_array = np.asarray(clean_view)
        corrupted_array = np.asarray(corrupted_view)
        if clean_array.shape != corrupted_array.shape or clean_array.ndim != 2:
            raise ValueError("clean/corrupted views must have equal [N,D] shape")
        if sample_num is None:
            sample_num = clean_array.shape[0]
        elif clean_array.shape[0] != sample_num:
            raise ValueError("all views must share the same sample count")
        columns.append(np.any(np.abs(corrupted_array - clean_array) > 0, axis=1))
    return np.stack(columns, axis=1).astype(bool, copy=False)


def noise_separation_diagnostic(agreement, corruption_mask):
    """Evaluate agreement as a score for clean-vs-corrupted sample-views."""
    agreement_array = (
        agreement.detach().cpu().numpy()
        if torch.is_tensor(agreement)
        else np.asarray(agreement)
    )
    corruption_mask = np.asarray(corruption_mask, dtype=bool)
    if agreement_array.shape != corruption_mask.shape:
        raise ValueError("agreement and corruption mask must have shape [N,V]")
    if not np.isfinite(agreement_array).all():
        raise ValueError("agreement must contain only finite values")

    scores = agreement_array.astype(np.float64).reshape(-1)
    corrupted = corruption_mask.reshape(-1)
    clean = ~corrupted
    if not np.any(clean) or not np.any(corrupted):
        raise ValueError("both clean and corrupted sample-view pairs are required")
    clean_indicator = clean.astype(np.int64)
    clean_scores = scores[clean]
    corrupted_scores = scores[corrupted]
    spearman_result = spearmanr(clean_indicator, scores)
    spearman_value = float(spearman_result.correlation)
    result = {
        "clean_pair_count": int(clean.sum()),
        "corrupted_pair_count": int(corrupted.sum()),
        "clean_agreement_mean": float(np.mean(clean_scores)),
        "clean_agreement_median": float(np.median(clean_scores)),
        "corrupted_agreement_mean": float(np.mean(corrupted_scores)),
        "corrupted_agreement_median": float(np.median(corrupted_scores)),
        "agreement_gap_clean_minus_corrupted": float(
            np.mean(clean_scores) - np.mean(corrupted_scores)
        ),
        "clean_corrupted_auc": float(
            roc_auc_score(clean_indicator, scores)
        ),
        "spearman_clean_indicator_vs_agreement": spearman_value,
    }
    result["all_finite_pass"] = all_finite_nested(result)
    return result


def all_finite_nested(value):
    """Return whether every numeric leaf in a nested result is finite."""
    if isinstance(value, dict):
        return all(all_finite_nested(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite_nested(item) for item in value)
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True
