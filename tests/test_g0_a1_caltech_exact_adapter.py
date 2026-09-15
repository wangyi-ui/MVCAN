import inspect

import numpy as np

import datasets
from experiments.cyclic_utility import c2_a0_sparse_label_utility_protocol as c2
from experiments.e1_pairwise_utility import train_e1_pairwise_utility as e1
from experiments.final_core import train_final_core
from experiments.generic_contract import caltech_exact_adapter as adapter


def test_parent_integrity_passes():
    audit = adapter.validate_caltech_parent_integrity()
    assert audit["pass_status"] is True
    assert audit["all_parent_hashes_pass"] is True
    assert audit["g0_a0_parent_commit"] == adapter.G0_A0_PARENT_COMMIT


def test_dataset_contract_and_view_dimensions_are_exact():
    audit = adapter.validate_caltech_dataset_contract()
    contract = audit["contract"]
    assert (contract.N, contract.V, contract.K) == (1400, 6, 7)
    assert (contract.L, contract.N_u, contract.S) == (14, 1386, 20)
    assert contract.view_dims == adapter.CALTECH_VIEW_DIMS == tuple(e1.VIEW_DIMS)
    assert audit["dataset_content_parsed"] is False


def test_tensor_shape_contract_is_exact():
    shapes = adapter.validate_caltech_dataset_contract()["tensor_shapes"]
    assert shapes["q_local"] == shapes["q_aligned"] == (1400, 6, 7)
    assert shapes["U_cycle"] == (1400, 20)
    assert shapes["PredRelation"] == (1386, 14, 20)
    assert shapes["relation_balance_weights"] == (1386, 14, 20)
    assert shapes["final_predictions"] == (1400,)


def test_action_arrays_order_and_hashes_are_exact():
    audit = adapter.validate_caltech_action_contract()
    assert audit["direction_count"] == 20
    assert audit["generator_exact_equal"] is True
    assert audit["verifier_exact_equal"] is True
    assert audit["action_order_exact"] is True
    assert audit["generator_logical_sha256"] == adapter.GENERATOR_TENSOR_SHA256
    assert audit["verifier_logical_sha256"] == adapter.VERIFIER_TENSOR_SHA256


def test_generic_mask_matches_frozen_mask_and_counts():
    audit = adapter.validate_caltech_weak_quality_contract()
    assert audit["mask_exact_equal"] is True
    assert audit["mask_shape"] == (1400, 6)
    assert audit["mask_dtype"] == "bool"
    assert audit["corrupted_pair_count"] == 4200
    assert audit["per_row_count"] == 3
    assert audit["per_view_counts"] == (700,) * 6
    assert audit["mask_logical_sha256"] == adapter.CORRUPTION_MASK_NDARRAY_SHA256
    assert audit["gaussian_corruption_applied"] is False


def test_frozen_sparse_ids_targets_counts_and_sample_hash():
    audit = adapter.validate_caltech_sparse_label_contract()
    assert audit["labeled_ids"] == c2.FIXED_LABELED_IDS
    assert audit["labeled_targets"] == c2.FIXED_LABELED_TARGETS
    assert audit["class_counts"] == (2,) * 7
    assert audit["unlabeled_count"] == 1386
    assert audit["sample_ids_logical_sha256"] == adapter.SAMPLE_IDS_NDARRAY_SHA256
    assert audit["generic_ranking_used"] is False


def test_u_cycle_two_hash_namespaces_are_exact():
    record = adapter.load_and_validate_caltech_semantic_references()["arrays"]["U_cycle"]
    assert record["shape"] == (1400, 20)
    assert record["dtype"] == "float64"
    assert record["parent_logical_sha256"] == adapter.U_CYCLE_NDARRAY_SHA256
    assert record["f0_logical_sha256"] == adapter.U_CYCLE_TENSOR_SHA256


def test_pred_relation_two_hash_namespaces_are_exact():
    record = adapter.load_and_validate_caltech_semantic_references()["arrays"]["PredRelation_true"]
    assert record["shape"] == (1386, 14, 20)
    assert record["dtype"] == "bool"
    assert record["parent_logical_sha256"] == adapter.PRED_RELATION_NDARRAY_SHA256
    assert record["f0_logical_sha256"] == adapter.PRED_RELATION_TENSOR_SHA256


def test_balance_weights_two_hash_namespaces_are_exact():
    record = adapter.load_and_validate_caltech_semantic_references()["arrays"]["relation_balance_weights_true"]
    assert record["shape"] == (1386, 14, 20)
    assert record["dtype"] == "float64"
    assert record["parent_logical_sha256"] == adapter.BALANCE_WEIGHT_NDARRAY_SHA256
    assert record["f0_logical_sha256"] == adapter.BALANCE_WEIGHT_TENSOR_SHA256


def test_prediction_file_logical_hash_and_model_metadata_are_exact():
    audit = adapter.validate_caltech_frozen_prediction_reference()
    assert audit["prediction_file_sha256"] == adapter.REFERENCE_PREDICTION_FILE_SHA256
    assert audit["prediction_logical_sha256"] == adapter.PREDICTION_TENSOR_SHA256
    assert audit["final_predictions_exact_equal"] is True
    assert audit["final_sample_ids_exact_equal"] is True
    assert audit["final_model_hash_supported"] is True
    assert audit["final_model_hash_equal"] is True
    assert audit["final_model_aggregate_sha256"] == adapter.FINAL_MODEL_AGGREGATE_SHA256
    assert audit["final_model_per_view_sha256"] == adapter.FINAL_MODEL_PER_VIEW_SHA256


def test_f0_pre_gt_seal_validates_read_only():
    audit = adapter.validate_caltech_f0_pre_gt_reference()
    assert audit["pre_gt_seal_valid"] is True
    assert audit["artifact_file_sha256"] == adapter.F0_ARTIFACT_SHA256
    assert audit["audit_file_sha256"] == adapter.F0_AUDIT_SHA256
    assert audit["seal_file_sha256"] == adapter.F0_SEAL_SHA256


def test_gate0_through_gate5_pass_and_gate6_is_not_run():
    gates = adapter.build_caltech_gate_readiness()
    assert gates["Gate0_parent_integrity"] == "PASS"
    assert gates["Gate1_dataset_contract"] == "PASS"
    assert gates["Gate2_action_space"] == "PASS"
    assert gates["Gate3_weak_quality"] == "PASS"
    assert gates["Gate4_sparse_labels"] == "PASS"
    assert gates["Gate5_U_and_relation_semantics"] == "PASS"
    assert gates["Gate6_exact_replay"] == "NOT_RUN"
    assert gates["Overall_G0_A1"] == "NOT_COMPLETE"


def test_complete_audit_never_calls_exact_replay(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("G0-A1-A0 must not run exact replay")

    monkeypatch.setattr(train_final_core, "run_exact_replay", forbidden)
    audit = adapter.audit_caltech_exact_adapter()
    assert audit["exact_replay_run"] is False
    assert audit["training_run"] is False
    assert audit["future_replay_delegate"] == (
        "experiments.final_core.train_final_core.run_exact_replay"
    )


def test_complete_audit_never_calls_gt_loaders(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("G0-A1-A0 must not load full GT")

    monkeypatch.setattr(datasets, "load_data", forbidden)
    monkeypatch.setattr(e1, "load_labels_after_predictions", forbidden)
    audit = adapter.audit_caltech_exact_adapter()
    assert audit["full_ground_truth_loaded"] is False


def test_public_adapter_signatures_accept_no_full_label_input():
    forbidden = {"labels", "gt", "GT", "y", "Y"}
    public = (
        adapter.build_caltech_dataset_contract,
        adapter.validate_caltech_dataset_contract,
        adapter.validate_caltech_action_contract,
        adapter.validate_caltech_weak_quality_contract,
        adapter.validate_caltech_sparse_label_contract,
        adapter.load_and_validate_caltech_semantic_references,
        adapter.validate_caltech_frozen_prediction_reference,
        adapter.validate_caltech_f0_pre_gt_reference,
        adapter.validate_caltech_parent_integrity,
        adapter.build_caltech_gate_readiness,
        adapter.audit_caltech_exact_adapter,
    )
    assert all(forbidden.isdisjoint(inspect.signature(function).parameters) for function in public)


def test_adapter_source_exposes_no_training_execution_surface():
    source = inspect.getsource(adapter)
    forbidden = (
        "torch.optim",
        ".backward(",
        "KMeans",
        "fit_predict(",
        "run_exact_replay(",
        "materialize_hash_ranked_sparse_split",
    )
    assert all(token not in source for token in forbidden)
    assert "FUTURE_REPLAY_DELEGATE" in source


def test_audit_records_are_read_only():
    audit = adapter.validate_caltech_action_contract()
    with np.testing.assert_raises(TypeError):
        audit["direction_count"] = 19
