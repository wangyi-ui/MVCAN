import numpy as np

import ClusteringTest
from release_core.evaluation.clustering import acc, ari, nmi


def test_evaluation_primitives_exact_on_synthetic_labels():
    y_true = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    y_pred = np.array([2, 2, 0, 0, 1, 1, 0], dtype=np.int64)
    assert acc(y_true, y_pred) == ClusteringTest.acc(y_true, y_pred)
    assert nmi(y_true, y_pred) == ClusteringTest.nmi(y_true, y_pred)
    assert ari(y_true, y_pred) == ClusteringTest.ari(y_true, y_pred)
