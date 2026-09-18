"""Frozen clustering evaluation primitives."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    dimension = max(y_pred.max(), y_true.max()) + 1
    contingency = np.zeros((dimension, dimension), dtype=np.int64)
    for index in range(y_pred.size):
        contingency[y_pred[index], y_true[index]] += 1
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    return contingency[rows, columns].sum() * 1.0 / y_pred.size
