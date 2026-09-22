import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from release_core.evaluation import acc, ari, nmi
from release_core.runtime import _payload_sha256


ARCHIVE = Path("/root/autodl-tmp/CVPR24-MVCAN")
FREEZE = ARCHIVE / "experiment_freeze/f0_a0_multiseed_exact_replay_pass_20260913"
REFERENCES = {
    20: (
        "outputs/final_core/f0_a0_engineering_smoke_seed20",
        "ec78a6102546254c7d0868f4948e1a79a236734ce37ab9d67b350063fefcc010",
        "255234648a40e393059d097bfecba0e0ee822b38f3b6edae1c5f9a40c87302d4",
        "a85828fa1c914194b97df5704c99bfbabef720e33fdf60359d1c4916a870e54f",
        "b3a40a90c5069a2fe6926b48efe406dc51c670b204ccc1bd53c4d322e830d504",
        (0.8614285714285714, 0.7734181958700173, 0.753482395557614),
    ),
    30: (
        "outputs/final_core/f0_a0_exact_replay_seed30",
        "5a2c7819d664ece6d03bab62a669a9973208649ed6a13895296ecc4cf7c2a84a",
        "9253ea739fdf5ae731b75dc4a8fa70f815e8acd5c157d17fd051d95fb655af9e",
        "c24bb1859516dafff26be58b3c38b4f84088781b4ed9c1204d696f389be8e0f6",
        "baf7753b58dba1d30f8fffbe812910f44b31d358560e62580d775600a26941bb",
        (0.8592857142857143, 0.7703779562016145, 0.749533486243257),
    ),
    50: (
        "outputs/final_core/f0_a0_exact_replay_seed50",
        "b2a6aa3c46cc94c3382687d151b2666018e2743181098c761b2c7928ef9721cb",
        "bd83c5ba606b9bb2f01cb2da1cb8b406a7e4f1f666f9aa9966e034076d64a9d4",
        "6082effc877ff3575a805bd3aa53c8d6483c87e55ac679b5defa1cc8d6923c65",
        "d2bfa6ae66d2541f07d8a630f81e69693115ce3fff15ede76140c3cdbd5a6f89",
        (0.8607142857142858, 0.7763024116674858, 0.7538086225874288),
    ),
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.parametrize("seed", [20, 30, 50])
def test_existing_caltech_reference_inventory_and_payload_hash(seed):
    root_name, artifact_hash, audit_hash, seal_hash, prediction_hash, _ = REFERENCES[seed]
    root = ARCHIVE / root_name
    artifact = root / "final_core_pre_gt_artifact.npz"
    audit = root / "final_core_pre_gt_audit.json"
    seal = root / "final_core_pre_gt_seal.json"
    assert _sha256(artifact) == artifact_hash
    assert _sha256(audit) == audit_hash
    assert _sha256(seal) == seal_hash
    with np.load(artifact, allow_pickle=False) as archive:
        assert archive["sample_ids"].dtype == np.int64
        assert np.array_equal(archive["sample_ids"], np.arange(1400, dtype=np.int64))
        assert archive["final_predictions"].shape == (1400,)
        assert _payload_sha256(archive["final_predictions"]) == prediction_hash
        assert archive["q_local"].shape == (1400, 6, 7)
        assert archive["q_aligned"].shape == (1400, 6, 7)
        assert archive["M_v"].shape == (6, 7, 7)
    audit_record = json.loads(audit.read_text(encoding="utf-8"))
    assert audit_record["final_predictions_equal"] is True
    assert audit_record["final_model_hash_equal"] is True
    assert audit_record["GT_loaded_before_pre_gt_seal"] is False


@pytest.mark.parametrize("seed", [20, 30, 50])
def test_existing_predictions_match_frozen_metric_values(seed):
    root_name, _, _, _, _, expected = REFERENCES[seed]
    artifact = ARCHIVE / root_name / "final_core_pre_gt_artifact.npz"
    with np.load(artifact, allow_pickle=False) as archive:
        predictions = np.ascontiguousarray(archive["final_predictions"], dtype=np.int64)
    labels = np.squeeze(sio.loadmat(
        ARCHIVE / "data/Caltech.mat", variable_names=["Y"]
    )["Y"]).astype(np.int64)
    labels[labels == 95] = 5
    actual = (
        float(acc(labels, predictions)),
        float(nmi(labels, predictions)),
        float(ari(labels, predictions)),
    )
    assert actual == expected


def test_multiseed_freeze_declares_exact_payload_not_clean_metadata_identity():
    summary = json.loads((FREEZE / "MULTISEED_SUMMARY.json").read_text())
    assert summary["all_seeds_pass"] is True
    for seed in (20, 30, 50):
        record = summary["seed_results"][str(seed)]
        assert record["pre_gt_exact_replay"] is True
        assert record["prediction_exact_equal"] is True
        assert record["final_model_hash_equal"] is True
    source = Path("release_core/runtime/entrypoint.py").read_text(encoding="utf-8")
    assert '"whole_file_equality_required": False' in source


def test_msrc_evidence_remains_sealed_diagnostic_only():
    manifest = ARCHIVE / (
        "experiment_freeze/g0_b0_msrc_postseal_metrics_diagnostic_20260916/"
        "evaluation_outputs_sha256.txt"
    )
    status = ARCHIVE / (
        "experiment_freeze/g0_b0_msrc_postseal_metrics_diagnostic_20260916/STATUS.txt"
    )
    assert manifest.is_file()
    text = status.read_text(encoding="utf-8")
    assert "DIAGNOSTIC" in text
    assert "NON-GATING" in text


def test_bdgp_evidence_remains_structural_only():
    root = ARCHIVE / "experiment_freeze/g0_b1_bdgp_structural_pilot_pre_gt_pass_20260917"
    assert (root / "structural_pilot_outputs_sha256.txt").is_file()
    text = (root / "STATUS.txt").read_text(encoding="utf-8")
    assert "STRUCTURAL" in text
    assert "Pre-GT" in text
