import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

import release_core.runtime.entrypoint as entrypoint
import release_core.runtime.evaluation as evaluation
from release_core.runtime import SealedPredictionPaths, _payload_sha256


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_sealed_fixture(tmp_path, sample_count=6):
    sample_ids = np.arange(sample_count, dtype=np.int64)
    predictions = np.arange(sample_count, dtype=np.int64) % 5
    q_local = np.full((sample_count, 2, 5), 0.2, dtype=np.float32)
    matrix = np.repeat(np.eye(5, dtype=np.float32)[None, :, :], 2, axis=0)
    arrays = {
        "sample_ids": sample_ids,
        "final_predictions": predictions,
        "labeled_ids": np.arange(5, dtype=np.int64),
        "q_local": q_local,
        "q_aligned": q_local.copy(),
        "M_v": matrix,
        "input_sha256": np.asarray(["a" * 64], dtype="<U64"),
        "initial_model_sha256": np.asarray("b" * 64, dtype="<U64"),
        "final_model_sha256": np.asarray("c" * 64, dtype="<U64"),
    }
    bundle = tmp_path / "pre_gt_bundle.npz"
    np.savez(bundle, **arrays)
    audit = tmp_path / "pre_gt_audit.json"
    audit.write_text(json.dumps({
        "schema": "release-core-pre-gt-audit-v1",
        "scientific_config": {"dataset": "BDGP"},
        "gt_firewall": {
            "full_gt_loaded": False,
            "metrics_computed": False,
            "full_gt_present_in_bundle": False,
        },
    }, sort_keys=True), encoding="utf-8")
    seal = tmp_path / "pre_gt_seal.json"
    seal.write_text(json.dumps({
        "schema": evaluation._SEAL_SCHEMA,
        "hash_algorithm": "SHA256",
        "bundle_sha256": _sha256(bundle),
        "audit_sha256": _sha256(audit),
        "bundle_keys": list(evaluation._BUNDLE_KEYS),
        "prediction_logical_sha256": _payload_sha256(predictions),
        "sample_id_logical_sha256": _payload_sha256(sample_ids),
        "full_gt_loaded_before_seal": False,
        "metrics_computed_before_seal": False,
    }, sort_keys=True), encoding="utf-8")
    return SealedPredictionPaths(bundle, audit, seal), arrays


def test_entrypoint_has_no_gt_or_metric_import_path():
    source = inspect.getsource(entrypoint)
    assert "scipy" not in source
    assert "load_dataset" not in source
    assert "evaluate_postseal" not in source
    assert "full_gt_path" not in inspect.signature(entrypoint.run_pre_gt).parameters
    assert not any(name in source for name in (
        "adjusted_rand_score", "normalized_mutual_info_score", "ClusteringTest",
    ))


def test_evaluator_has_no_model_optimizer_or_r5_dependency():
    source = inspect.getsource(evaluation)
    assert "torch" not in source
    assert "release_core.training" not in source
    assert "release_core.backbone" not in source
    assert "optimizer" not in source.lower()
    assert "run_alternating_training" not in source


@pytest.mark.parametrize("corruption", ["bundle", "audit", "schema"])
def test_invalid_seal_fails_before_first_gt_access(tmp_path, monkeypatch, corruption):
    sealed, _ = make_sealed_fixture(tmp_path)
    if corruption == "bundle":
        sealed.bundle.write_bytes(sealed.bundle.read_bytes() + b"corrupt")
    elif corruption == "audit":
        sealed.audit.write_text("{}", encoding="utf-8")
    else:
        record = json.loads(sealed.seal.read_text(encoding="utf-8"))
        record["schema"] = "invalid"
        sealed.seal.write_text(json.dumps(record), encoding="utf-8")
    reached_gt = []

    def forbidden_gt(*args, **kwargs):
        reached_gt.append(True)
        raise AssertionError("GT loader must not be reached")

    monkeypatch.setattr(evaluation, "_load_full_gt", forbidden_gt)
    with pytest.raises(RuntimeError):
        evaluation.evaluate_postseal(sealed, tmp_path / "full_gt.mat")
    assert reached_gt == []


def test_full_labels_absent_from_pre_gt_bundle(tmp_path):
    sealed, _ = make_sealed_fixture(tmp_path)
    with np.load(sealed.bundle, allow_pickle=False) as archive:
        lowered = {key.lower() for key in archive.files}
        assert lowered.isdisjoint({"y", "gt", "labels", "full_gt", "ground_truth"})


def test_metrics_are_created_only_after_valid_seal_and_gt_load(tmp_path, monkeypatch):
    sealed, arrays = make_sealed_fixture(tmp_path)
    labels = arrays["final_predictions"]
    gt_path = tmp_path / "full_gt.mat"
    sio.savemat(gt_path, {
        "X1": np.zeros((labels.size, 2), dtype=np.float32),
        "X2": np.zeros((labels.size, 3), dtype=np.float32),
        "Y": labels[:, None],
    })
    events = []
    original_validate = evaluation._validate_sealed_prediction
    original_load = evaluation._load_full_gt

    def validate_spy(value):
        result = original_validate(value)
        events.append("seal_verified")
        return result

    def gt_spy(dataset, path):
        assert events == ["seal_verified"]
        events.append("gt_loaded")
        return original_load(dataset, path)

    monkeypatch.setattr(evaluation, "_validate_sealed_prediction", validate_spy)
    monkeypatch.setattr(evaluation, "_load_full_gt", gt_spy)
    metrics = evaluation.evaluate_postseal(sealed, gt_path)
    assert events == ["seal_verified", "gt_loaded"]
    assert metrics.acc == pytest.approx(1.0)
    assert (tmp_path / "postseal_metrics.json").is_file()
    record = json.loads((tmp_path / "postseal_metrics.json").read_text())
    assert record["seal_verified_before_full_gt_load"] is True
    assert record["full_gt_used_for_training"] is False
    assert record["post_gt_mutation_path_present"] is False
