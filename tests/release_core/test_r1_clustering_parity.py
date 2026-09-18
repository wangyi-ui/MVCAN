import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler

from model import Autoencoder as LegacyAutoencoder
from model import MvCAN
from release_core.backbone.clustering import (
    match,
    native_refresh_from_latents,
    new_p,
    student_t_posterior,
    target_distribution,
)


def test_student_t_posterior_exact():
    torch.manual_seed(77)
    legacy = LegacyAutoencoder([6, 10], "relu", False, 3, True, [1])
    latent = torch.arange(40, dtype=torch.float32).reshape(4, 10) / 19.0
    old = legacy.clustering(latent)
    new = student_t_posterior(latent, legacy._cluster_layer, alpha=1.0)
    assert torch.equal(old, new)
    assert torch.equal(old.sum(1), new.sum(1))
    assert torch.equal(old.argmax(1), new.argmax(1))
    assert torch.isfinite(new).all()


def test_target_match_and_new_p_exact():
    q = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    np.testing.assert_array_equal(target_distribution(q), MvCAN.target_distribution(None, q))
    latent = np.arange(24, dtype=np.float64).reshape(4, 6) / 11.0
    centers = np.array([latent[0], latent[2], latent[3]])
    np.testing.assert_array_equal(new_p(latent, centers), MvCAN.new_P(None, latent, centers))
    y_true = np.array([2, 2, 0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([1, 1, 2, 2, 0, 0], dtype=np.int64)
    for old, new in zip(MvCAN.Match(None, y_true, y_pred), match(y_true, y_pred)):
        np.testing.assert_array_equal(old, new)


def _reference_refresh(latents, posteriors, weights, clusters, seed):
    weights = np.asarray(weights, dtype=np.float64).copy()
    estimator = KMeans(n_clusters=clusters, n_init=100, random_state=seed)
    for _ in range(2):
        fused = []
        assignments = []
        for index in range(len(latents)):
            fused.append(MinMaxScaler().fit_transform(latents[index]) * weights[index])
            assignments.append(posteriors[index].argmax(1))
        latent_fusion = np.hstack(fused)
        predictions = estimator.fit_predict(latent_fusion)
        for index in range(len(latents)):
            nmi = round(normalized_mutual_info_score(predictions, assignments[index]), 5)
            weights[index] = float(np.exp(nmi))
    matrices = np.stack([MvCAN.Match(None, value, predictions)[3] for value in assignments])
    p_all = MvCAN.target_distribution(
        None, MvCAN.new_P(None, latent_fusion, estimator.cluster_centers_)
    )
    return p_all, matrices, predictions.astype(np.int64), weights, estimator.cluster_centers_


def test_synthetic_native_refresh_exact():
    rng = np.random.RandomState(91)
    latents = [rng.normal(size=(18, 4)), rng.normal(size=(18, 5))]
    labels = np.tile(np.arange(3), 6)
    posteriors = []
    for shift in (0, 1):
        q = np.full((18, 3), 0.05, dtype=np.float64)
        q[np.arange(18), (labels + shift) % 3] = 0.9
        posteriors.append(q)
    old = _reference_refresh(latents, posteriors, [1.0, 1.0], 3, 29)
    new = native_refresh_from_latents(latents, posteriors, [1.0, 1.0], 3, 29)
    for old_value, new_value in zip(old, new):
        np.testing.assert_array_equal(old_value, new_value)
