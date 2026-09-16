import hashlib
import inspect

import numpy as np

from configure import get_default_config
from datasets import load_data
from experiments.generic_contract.dataset_contract import infer_dataset_contract
from experiments.generic_contract.generic_final_core_adapter import (
    FEATURE_FIELDS,
    PRE_GT_FIELDS,
)
from experiments.generic_contract.generic_weak_quality import (
    generate_half_corruption_mask,
)
from experiments.generic_contract.run_g0_b0_msrc_structural_pilot import main
from experiments.generic_contract.sparse_label_contract import (
    materialize_hash_ranked_sparse_split,
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_msrc():
    config = get_default_config("MSRC-v1")
    config["dataset"] = "MSRC-v1"
    return load_data(config)


def test_msrc_dataset_sha_and_runtime_contract():
    assert _sha256("data/MSRC_v1.mat") == (
        "38d89aa41ae984f0b026f4baa8b5dcb7c569c471e2fb42b73a0f6743398e9103"
    )
    views, labels = _load_msrc()
    contract = infer_dataset_contract(
        views, dataset_name="MSRC-v1", K=7, labels_per_class=2
    )
    assert (contract.N, contract.V, contract.K) == (210, 5, 7)
    assert (contract.L, contract.N_u, contract.S) == (14, 196, 20)
    assert contract.view_dims == tuple(view.shape[1] for view in views)
    assert labels[0].shape == (210,)


def test_msrc_odd_mask_exact_and_deterministic():
    first, first_audit = generate_half_corruption_mask(210, 5, 20)
    second, second_audit = generate_half_corruption_mask(210, 5, 20)
    rows = first.sum(axis=1)
    assert np.array_equal(first, second)
    assert first_audit["mask_sha256"] == second_audit["mask_sha256"]
    assert int(first.sum()) == 525
    assert np.count_nonzero(rows == 2) == 105
    assert np.count_nonzero(rows == 3) == 105
    counts = first.sum(axis=0)
    assert int(counts.max()) - int(counts.min()) <= 1


def test_msrc_sparse_split_two_per_class():
    _, labels = _load_msrc()
    split = materialize_hash_ranked_sparse_split(
        labels[0],
        dataset_name="MSRC-v1",
        label_seed=20,
        labels_per_class=2,
    )
    assert split.labeled_ids.shape == split.labeled_targets.shape == (14,)
    assert split.unlabeled_ids.shape == (196,)
    assert np.array_equal(
        np.bincount(split.labeled_targets, minlength=7),
        np.full(7, 2),
    )
    assert np.array_equal(
        np.sort(np.concatenate((split.labeled_ids, split.unlabeled_ids))),
        np.arange(210),
    )


def test_trainable_and_pre_gt_whitelists_exclude_full_gt():
    assert FEATURE_FIELDS == ("X1", "X2", "X3", "X4", "X5", "sample_ids")
    lowered = tuple(name.lower() for name in PRE_GT_FIELDS)
    assert not any(name in lowered for name in ("gt", "labels", "metrics", "acc", "nmi", "ari"))
    assert {
        "sample_ids", "labeled_ids", "unlabeled_ids", "final_predictions",
        "q_local", "q_aligned", "M_v", "U_cycle",
        "PredRelation_true", "relation_balance_weights_true",
    }.issubset(PRE_GT_FIELDS)


def test_training_runner_has_no_dataset_or_mat_loader():
    source = inspect.getsource(main)
    module_source = inspect.getsource(
        __import__(
            "experiments.generic_contract.run_g0_b0_msrc_structural_pilot",
            fromlist=["unused"],
        )
    )
    combined = source + module_source
    assert "datasets.load_data" not in combined
    assert "load_data(" not in combined
    assert "scipy.io.loadmat" not in combined
    assert "loadmat(" not in combined


def test_phase_and_final_shape_contracts_are_runtime_derived():
    views = [
        np.zeros((210, dimension), dtype=np.float32)
        for dimension in (3, 4, 5, 6, 7)
    ]
    contract = infer_dataset_contract(
        views, dataset_name="MSRC-v1", K=7, labels_per_class=2
    )
    assert contract.tensor_shapes["q_local"] == (210, 5, 7)
    assert contract.tensor_shapes["q_aligned"] == (210, 5, 7)
    assert contract.tensor_shapes["U_cycle"] == (210, 20)
    assert contract.tensor_shapes["PredRelation"] == (196, 14, 20)
    assert contract.tensor_shapes["final_predictions"] == (210,)
