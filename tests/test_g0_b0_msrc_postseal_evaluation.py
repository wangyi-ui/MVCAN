import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments.generic_contract import evaluate_g0_b0_msrc_postseal as evaluator


ARTIFACT = Path(
    "outputs/generic_contract/g0_b0_msrc_structural_pilot_seed20_20260916/"
    "msrc_structural_pre_gt_artifact.npz"
)
AUDIT = ARTIFACT.with_name("msrc_structural_pre_gt_audit.json")
SEAL = ARTIFACT.with_name("msrc_structural_pre_gt_seal.json")
DATASET = Path("data/MSRC_v1.mat")


def _synthetic_gt():
    return np.repeat(np.arange(1, 8, dtype=np.int64), 30).reshape(-1, 1)


def _evaluate_with_fake_gt(monkeypatch, output_dir, events=None):
    def guarded_loadmat(path, variable_names):
        if events is not None:
            assert "seal" in events
            events.append("gt")
        assert Path(path) == DATASET
        assert variable_names == ["gt"]
        return {"gt": _synthetic_gt()}

    monkeypatch.setattr(evaluator.scipy.io, "loadmat", guarded_loadmat)
    return evaluator.evaluate_postseal(ARTIFACT, AUDIT, SEAL, DATASET, output_dir)


def test_authoritative_file_sha_constants_and_real_files():
    assert evaluator.DATASET_FILE_SHA256 == (
        "38d89aa41ae984f0b026f4baa8b5dcb7c569c471e2fb42b73a0f6743398e9103"
    )
    assert evaluator.PRE_GT_ARTIFACT_SHA256 == (
        "b04f107f6279839d534e55500fbb0f9aba94d7686944ab20f255337a511af234"
    )
    assert evaluator.PRE_GT_AUDIT_SHA256 == (
        "7674a6d46f5092e20a9be8f166cb6d28c29f10422176f79aad259b20ce291b1e"
    )
    assert evaluator.PRE_GT_SEAL_SHA256 == (
        "c2fd8b35ffe0a065dbc8df69b7488fd75d88d39c2d3a100e665cc014729fa50e"
    )
    assert evaluator.file_sha256(DATASET) == evaluator.DATASET_FILE_SHA256
    assert evaluator.file_sha256(ARTIFACT) == evaluator.PRE_GT_ARTIFACT_SHA256
    assert evaluator.file_sha256(AUDIT) == evaluator.PRE_GT_AUDIT_SHA256
    assert evaluator.file_sha256(SEAL) == evaluator.PRE_GT_SEAL_SHA256


def test_real_frozen_seal_predictions_and_sample_ids_validate():
    arrays, seal = evaluator.validate_structural_pre_gt_seal(ARTIFACT, AUDIT, SEAL)
    predictions, ndarray_hash, tensor_hash = evaluator._verify_frozen_predictions(
        arrays
    )
    assert seal["pre_gt_seal_valid"] is True
    assert predictions.shape == (210,)
    assert ndarray_hash == evaluator.PREDICTION_NDARRAY_SHA256
    assert tensor_hash == evaluator.PREDICTION_TENSOR_SHA256
    assert np.array_equal(arrays["sample_ids"], np.arange(210, dtype=np.int64))


def test_gt_normalization_contract_is_exact():
    labels = evaluator._normalize_gt({"gt": _synthetic_gt()})
    assert labels.shape == (210,)
    assert labels.dtype == np.int64
    assert np.array_equal(np.unique(labels), np.arange(7, dtype=np.int64))


def test_metric_functions_are_the_preregistered_definitions(monkeypatch):
    calls = []

    def fake_acc(labels, predictions):
        calls.append("acc")
        return 0.25

    def fake_nmi(labels, predictions):
        calls.append("nmi")
        return 0.5

    def fake_ari(labels, predictions):
        calls.append("ari")
        return -0.125

    monkeypatch.setattr(evaluator.ClusteringTest, "acc", fake_acc)
    monkeypatch.setattr(evaluator, "normalized_mutual_info_score", fake_nmi)
    monkeypatch.setattr(evaluator, "adjusted_rand_score", fake_ari)
    values = evaluator._compute_metrics(np.arange(4), np.arange(4))
    assert values == {"ACC": 0.25, "NMI": 0.5, "ARI": -0.125}
    assert calls == ["acc", "nmi", "ari"]


def test_seal_validation_precedes_gt_access_and_outputs_are_non_gating(
    monkeypatch, tmp_path
):
    events = []
    real_validator = evaluator.validate_structural_pre_gt_seal

    def recording_validator(*args):
        result = real_validator(*args)
        events.append("seal")
        return result

    parent_hashes_before = tuple(
        evaluator.file_sha256(path) for path in (ARTIFACT, AUDIT, SEAL)
    )
    monkeypatch.setattr(evaluator, "validate_structural_pre_gt_seal", recording_validator)
    output = tmp_path / "postseal"
    result = _evaluate_with_fake_gt(monkeypatch, output, events)
    parent_hashes_after = tuple(
        evaluator.file_sha256(path) for path in (ARTIFACT, AUDIT, SEAL)
    )

    assert events == ["seal", "gt"]
    assert parent_hashes_after == parent_hashes_before
    assert sorted(path.name for path in output.iterdir()) == [
        "msrc_postseal_evaluation_audit.json",
        "msrc_postseal_evaluation_seal.json",
        "msrc_postseal_metrics.json",
    ]
    assert result["metrics"]["metrics_are_G0_gate"] is False
    assert result["metrics"]["Gate6_status"] == "PASS"
    assert result["audit"]["GT_LOADED_ONLY_AFTER_PRE_GT_SEAL_VERIFIED"] is True
    assert result["audit"]["Gate6_changed_by_metrics"] is False
    assert result["seal"]["metrics_non_gating"] is True
    assert result["seal"]["Gate6_remains_PASS"] is True


def test_invalid_seal_prevents_gt_loading(monkeypatch, tmp_path):
    calls = []

    def invalid_validator(*args):
        raise RuntimeError("deliberately invalid seal")

    def forbidden_loadmat(*args, **kwargs):
        calls.append("gt")
        raise AssertionError("GT must remain inaccessible")

    monkeypatch.setattr(evaluator, "validate_structural_pre_gt_seal", invalid_validator)
    monkeypatch.setattr(evaluator.scipy.io, "loadmat", forbidden_loadmat)
    with pytest.raises(RuntimeError, match="deliberately invalid seal"):
        evaluator.evaluate_postseal(
            ARTIFACT, AUDIT, SEAL, DATASET, tmp_path / "must-not-exist"
        )
    assert calls == []
    assert not (tmp_path / "must-not-exist").exists()


def test_invalid_prediction_hash_prevents_gt_loading(monkeypatch, tmp_path):
    arrays, seal_record = evaluator.validate_structural_pre_gt_seal(
        ARTIFACT, AUDIT, SEAL
    )
    arrays["final_predictions"][0] += 1
    calls = []

    monkeypatch.setattr(
        evaluator,
        "validate_structural_pre_gt_seal",
        lambda *args: (arrays, seal_record),
    )

    def forbidden_loadmat(*args, **kwargs):
        calls.append("gt")
        raise AssertionError("GT must remain inaccessible")

    monkeypatch.setattr(evaluator.scipy.io, "loadmat", forbidden_loadmat)
    with pytest.raises(RuntimeError, match="prediction ndarray hash mismatch"):
        evaluator.evaluate_postseal(
            ARTIFACT, AUDIT, SEAL, DATASET, tmp_path / "must-not-exist"
        )
    assert calls == []
    assert not (tmp_path / "must-not-exist").exists()


def test_evaluator_source_contains_no_training_or_regeneration_paths():
    source = inspect.getsource(evaluator)
    forbidden = (
        "torch.optim",
        ".backward(",
        "KMeans",
        "MvCAN(",
        "run_structural_pilot(",
        "prepare_native_backbone(",
        "run_final_core(",
        "build_cycle_utility(",
        "build_relation_semantics(",
        "datasets.load_data",
    )
    assert all(token not in source for token in forbidden)


def test_audit_proves_no_execution_or_recomputation(monkeypatch, tmp_path):
    result = _evaluate_with_fake_gt(monkeypatch, tmp_path / "postseal")
    audit = result["audit"]
    assert audit["MODEL_EXECUTED"] is False
    assert audit["TRAINING_RUN"] is False
    assert audit["OPTIMIZER_RUN"] is False
    assert audit["BACKWARD_RUN"] is False
    assert audit["KMEANS_RUN"] is False
    assert audit["U_RECOMPUTED"] is False
    assert audit["RELATION_RECOMPUTED"] is False
    assert audit["SPARSE_SPLIT_RECOMPUTED"] is False
    assert audit["CORRUPTION_RECOMPUTED"] is False

