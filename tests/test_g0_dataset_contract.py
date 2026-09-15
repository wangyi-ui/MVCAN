import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from experiments.generic_contract.dataset_contract import (
    DatasetContract,
    infer_dataset_contract,
)


def test_explicit_contract_derives_all_scientific_shapes():
    contract = DatasetContract("Synthetic-5V", 210, 5, 7, (4, 8, 3, 9, 2), 2)
    assert (contract.L, contract.N_u, contract.S) == (14, 196, 20)
    assert contract.tensor_shapes == {
        "q_local": (210, 5, 7),
        "q_aligned": (210, 5, 7),
        "U_cycle": (210, 20),
        "labeled_ids": (14,),
        "unlabeled_ids": (196,),
        "PredRelation": (196, 14, 20),
        "relation_balance_weights": (196, 14, 20),
        "final_predictions": (210,),
    }


def test_contract_is_frozen_and_view_dimensions_are_immutable():
    contract = DatasetContract("Synthetic-2V", 20, 2, 3, [2, 5], 1)
    assert contract.view_dims == (2, 5)
    with pytest.raises(FrozenInstanceError):
        contract.N = 21


def test_inference_reads_only_view_structure_and_supports_different_dimensions():
    views = [np.zeros((30, width)) for width in (2, 7, 4)]
    contract = infer_dataset_contract(
        views, dataset_name="Synthetic-3V", K=5, labels_per_class=2
    )
    assert (contract.N, contract.V, contract.K) == (30, 3, 5)
    assert contract.view_dims == (2, 7, 4)


def test_inference_api_has_no_label_or_target_parameter():
    forbidden = {"y", "Y", "label", "labels", "target", "targets", "gt", "GT"}
    assert forbidden.isdisjoint(inspect.signature(infer_dataset_contract).parameters)


def test_inference_rejects_inconsistent_samples_and_nonmatrix_views():
    with pytest.raises(ValueError, match="same N"):
        infer_dataset_contract(
            [np.zeros((8, 2)), np.zeros((9, 3))], dataset_name="Bad", K=2
        )
    with pytest.raises(ValueError, match="shape"):
        infer_dataset_contract([np.zeros(8), np.zeros((8, 2))], dataset_name="Bad", K=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"N": 0},
        {"V": 1, "view_dims": (3,)},
        {"K": 1},
        {"view_dims": (3, 0)},
        {"view_dims": (3,)},
        {"labels_per_class": 0},
        {"N": 6, "K": 3, "labels_per_class": 2},
    ],
)
def test_invalid_dimension_contracts_fail_closed(kwargs):
    values = dict(
        dataset_name="Invalid", N=20, V=2, K=3, view_dims=(3, 4), labels_per_class=1
    )
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        DatasetContract(**values)


@pytest.mark.parametrize("field", ("N", "V", "K", "labels_per_class"))
def test_bool_integer_inputs_are_rejected(field):
    values = dict(
        dataset_name="Invalid", N=20, V=2, K=3, view_dims=(3, 4), labels_per_class=1
    )
    values[field] = True
    with pytest.raises(TypeError):
        DatasetContract(**values)


def test_bool_view_dimension_is_rejected():
    with pytest.raises(TypeError):
        DatasetContract("Invalid", 20, 2, 3, (3, False), 1)
