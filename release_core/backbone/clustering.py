"""Frozen clustering and latent-fusion primitives."""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler


def student_t_posterior(latent, centers, alpha=1.0):
    q = 1.0 / (1.0 + torch.sum(
        torch.pow(latent.unsqueeze(1) - centers, 2), 2
    ) / alpha)
    q = q.pow((alpha + 1.0) / 2.0)
    return (q.t() / torch.sum(q, 1)).t()


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def new_p(inputs, centers):
    alpha = 1
    q = 1.0 / (1.0 + np.sum(
        np.square(np.expand_dims(inputs, axis=1) - centers), axis=2
    ) / alpha)
    q **= (alpha + 1.0) / 2.0
    return np.transpose(np.transpose(q) / np.sum(q, axis=1))


def match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    dimension = max(y_pred.max(), y_true.max()) + 1
    contingency = np.zeros((dimension, dimension), dtype=np.int64)
    for index in range(y_pred.size):
        contingency[y_pred[index], y_true[index]] += 1
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    new_y = np.zeros(y_true.shape[0])
    matrix = np.zeros((dimension, dimension), dtype=np.int64)
    matrix[rows, columns] = 1
    for index in range(y_pred.size):
        for row in rows:
            if y_true[index] == columns[row]:
                new_y[index] = rows[row]
    return new_y, rows, columns, matrix


def initialize_kmeans_centers(latent, n_clusters, random_state):
    estimator = KMeans(
        n_clusters=int(n_clusters), n_init=100, random_state=int(random_state)
    )
    assignments = estimator.fit_predict(np.asarray(latent))
    return assignments.astype(np.int64, copy=False), estimator.cluster_centers_


def native_refresh_from_latents(
    latent_views, q_local_views, view_weights, n_clusters, random_state
):
    """Run the frozen two-pass KMeans/weight/alignment refresh."""
    if len(latent_views) != len(q_local_views):
        raise ValueError("latent and posterior view counts differ")
    weights = np.asarray(view_weights, dtype=np.float64).copy()
    estimator = KMeans(
        n_clusters=int(n_clusters), n_init=100, random_state=int(random_state)
    )
    local_assignments = None
    latent_fusion = None
    for _ in range(2):
        fused_views = []
        local_assignments = []
        for view_index in range(len(latent_views)):
            scaled = MinMaxScaler().fit_transform(np.asarray(latent_views[view_index]))
            fused_views.append(scaled * weights[view_index])
            local_assignments.append(np.asarray(q_local_views[view_index]).argmax(1))
        latent_fusion = np.hstack(fused_views)
        predictions = estimator.fit_predict(latent_fusion)
        for view_index in range(len(latent_views)):
            value = round(
                normalized_mutual_info_score(
                    predictions, local_assignments[view_index]
                ), 5
            )
            weights[view_index] = float(np.exp(value))
    matrices = []
    for assignments in local_assignments:
        matrices.append(match(assignments, predictions)[3])
    p_all = target_distribution(new_p(latent_fusion, estimator.cluster_centers_))
    return (
        p_all,
        np.stack(matrices),
        np.asarray(predictions, dtype=np.int64),
        weights,
        estimator.cluster_centers_,
    )


new_P = new_p
Match = match
