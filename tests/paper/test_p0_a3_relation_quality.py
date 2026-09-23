import numpy as np

from experiments.paper.diagnostics.relation_utility_quality import (
    compute_diagnostic,
    deterministic_positive_quartiles,
)


def test_deterministic_positive_quartiles_keep_zero_separate():
    utility = np.asarray([[0.0, 0.4, 0.1], [0.2, 0.3, 0.5]])
    groups = deterministic_positive_quartiles(utility)
    assert groups[0, 0] == 0
    assert sorted(groups[utility > 0].tolist()) == [1, 1, 2, 3, 4]


def test_synthetic_relation_quality_shapes_and_lift():
    sample_ids = np.arange(6, dtype=np.int64)
    labeled_ids = np.asarray([0, 1], dtype=np.int64)
    unlabeled_ids = np.asarray([2, 3, 4, 5], dtype=np.int64)
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    class_pred = np.asarray([[0, 1], [0, 1], [0, 0], [0, 1]], dtype=np.int64)
    truth = labels[unlabeled_ids, None, None] == labels[labeled_ids][None, :, None]
    pred = np.repeat(truth, 2, axis=2)
    pred[1, :, 0] = ~pred[1, :, 0]
    pred[2, :, 1] = ~pred[2, :, 1]
    balance = np.ones(pred.shape, dtype=np.float64)
    utility = np.asarray([
        [0.9, 0.8], [0.1, 0.7], [0.8, 0.1], [0.7, 0.9]
    ])
    result = compute_diagnostic(
        dataset="synthetic", utility=utility, class_pred=class_pred,
        pred_relation=pred, balance=balance, sample_ids=sample_ids,
        labeled_ids=labeled_ids, unlabeled_ids=unlabeled_ids, labels=labels,
        generators=((0,), (1,)), verifiers=((1,), (0,)),
        closure=np.ones_like(utility, dtype=np.bool_), permutation_count=10,
        permutation_seed=20,
    )
    assert result["tensor_shapes"]["E"] == [4, 2, 2]
    assert len(result["per_action"]) == 2
    assert result["permutation_control"]["permutations"] == 10

