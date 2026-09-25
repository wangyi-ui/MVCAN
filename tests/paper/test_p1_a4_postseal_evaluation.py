import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.paper.formal import p1_a4_postseal_evaluation as evaluation
from release_core.runtime import SealedPredictionPaths, _BUNDLE_KEYS, _SEAL_SCHEMA, _payload_sha256


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fixture(tmp_path, arm):
    root = tmp_path / arm
    root.mkdir()
    sample_ids = np.arange(4, dtype=np.int64)
    predictions = np.array([0, 1, 0, 1], dtype=np.int64)
    q_local = np.ones((4, 2, 2), dtype=np.float32)
    arrays = {"sample_ids": sample_ids, "final_predictions": predictions, "labeled_ids": np.array([0, 1], dtype=np.int64), "q_local": q_local, "q_aligned": q_local.copy(), "M_v": np.repeat(np.eye(2, dtype=np.float32)[None], 2, axis=0), "input_sha256": np.array(["a" * 64], dtype="<U64"), "initial_model_sha256": np.asarray("b" * 64, dtype="<U64"), "final_model_sha256": np.asarray("c" * 64, dtype="<U64")}
    bundle, audit_path, seal_path = root / "pre_gt_bundle.npz", root / "pre_gt_audit.json", root / "pre_gt_seal.json"
    np.savez(bundle, **arrays)
    firewall = {"full_gt_loaded": False, "metrics_computed": False}
    if arm == "OURS_TRUE_U":
        schema = "release-core-pre-gt-audit-v1"
        firewall["full_gt_present_in_bundle"] = False
        audit = {"schema": schema, "scientific_config": {"dataset": "Caltech-6V", "training_seed": 20}, "gt_firewall": firewall}
    else:
        schema = "p1-a3-base-pre-gt-audit-v1"
        audit = {"schema": schema, "scientific_config": {"dataset": "Caltech-6V", "training_seed": 20}, "gt_firewall": firewall, "base": {"action_artifact_loaded": False, "utility_artifact_loaded": False, "semantic_artifact_loaded": False, "phase_a_executed": False, "semantic_optimizer_created": False, "phase_b_executed": True}}
    audit_path.write_text(json.dumps(audit))
    seal = {"schema": _SEAL_SCHEMA, "hash_algorithm": "SHA256", "bundle_sha256": _sha(bundle), "audit_sha256": _sha(audit_path), "bundle_keys": list(_BUNDLE_KEYS), "prediction_logical_sha256": _payload_sha256(predictions), "sample_id_logical_sha256": _payload_sha256(sample_ids), "full_gt_loaded_before_seal": False, "metrics_computed_before_seal": False}
    seal_path.write_text(json.dumps(seal))
    return SealedPredictionPaths(bundle, audit_path, seal_path)


def _loader_counter(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluation, "load_dataset", lambda dataset, path: (calls.append((dataset, path)) or (None, [np.array([0, 1, 1, 0], dtype=np.int64)])))
    return calls


@pytest.mark.parametrize("arm", ("BASE", "OURS_TRUE_U"))
def test_authorized_sealed_schemas_evaluate_after_exact_validation(tmp_path, monkeypatch, arm):
    sealed = _fixture(tmp_path, arm)
    calls = _loader_counter(monkeypatch)
    metric_calls = []
    monkeypatch.setattr(evaluation, "acc", lambda labels, predictions: metric_calls.append("acc") or 0.1)
    monkeypatch.setattr(evaluation, "nmi", lambda labels, predictions: metric_calls.append("nmi") or 0.2)
    monkeypatch.setattr(evaluation, "ari", lambda labels, predictions: metric_calls.append("ari") or 0.3)
    record = evaluation.evaluate_formal_postseal(dataset="Caltech-6V", training_seed=20, arm=arm, full_gt_path="fake.mat", sealed=sealed, output_path=tmp_path / (arm + ".json"))
    assert len(calls) == 1 and metric_calls == ["acc", "nmi", "ari"]
    assert record["schema"] == "paper-formal-postseal-metrics-v1"
    assert record["seal_verified_before_full_gt_load"] is True


@pytest.mark.parametrize("sealed_arm,requested_arm", (("BASE", "OURS_TRUE_U"), ("OURS_TRUE_U", "BASE")))
def test_schema_arm_mismatch_fails_before_gt_load(tmp_path, monkeypatch, sealed_arm, requested_arm):
    calls = _loader_counter(monkeypatch)
    with pytest.raises(RuntimeError):
        evaluation.evaluate_formal_postseal(dataset="Caltech-6V", training_seed=20, arm=requested_arm, full_gt_path="fake.mat", sealed=_fixture(tmp_path, sealed_arm), output_path=tmp_path / "out.json")
    assert calls == []


@pytest.mark.parametrize("failure", ("bundle", "audit", "prediction", "firewall"))
def test_any_seal_or_firewall_failure_blocks_gt_load(tmp_path, monkeypatch, failure):
    sealed = _fixture(tmp_path, "BASE")
    if failure == "bundle":
        sealed.bundle.write_bytes(sealed.bundle.read_bytes() + b"x")
    elif failure == "audit":
        sealed.audit.write_text("{}")
    else:
        seal = json.loads(sealed.seal.read_text())
        if failure == "prediction":
            seal["prediction_logical_sha256"] = "0" * 64
        else:
            audit = json.loads(sealed.audit.read_text())
            audit["gt_firewall"]["full_gt_loaded"] = True
            sealed.audit.write_text(json.dumps(audit))
            seal["audit_sha256"] = _sha(sealed.audit)
        sealed.seal.write_text(json.dumps(seal))
    calls = _loader_counter(monkeypatch)
    with pytest.raises(RuntimeError):
        evaluation.evaluate_formal_postseal(dataset="Caltech-6V", training_seed=20, arm="BASE", full_gt_path="fake.mat", sealed=sealed, output_path=tmp_path / "out.json")
    assert calls == []


def test_evaluator_uses_frozen_release_metric_imports():
    source = Path(evaluation.__file__).read_text(encoding="utf-8")
    assert "from release_core.evaluation import acc, ari, nmi" in source
    assert "def acc(" not in source and "def nmi(" not in source and "def ari(" not in source
