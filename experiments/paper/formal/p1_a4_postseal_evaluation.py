"""Paper-local seal-first evaluator for authorized formal pre-GT schemas."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from release_core.data import load_dataset, validate_sample_ids
from release_core.evaluation import acc, ari, nmi
from release_core.runtime import SealedPredictionPaths, _BUNDLE_KEYS, _SEAL_SCHEMA, _payload_sha256
from release_core.runtime import evaluation as frozen_evaluation


ROOT = Path("outputs/paper/formal/main")
_BASE_SCHEMA = "p1-a3-base-pre-gt-audit-v1"
_OURS_SCHEMA = "release-core-pre-gt-audit-v1"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "audit or seal root must be an object")
    return value


def _sealed_paths(dataset, training_seed, arm):
    slug = dataset.lower().replace("-", "").replace("_", "")
    root = ROOT / slug / ("seed" + str(training_seed)) / arm
    return SealedPredictionPaths(root / "pre_gt_bundle.npz", root / "pre_gt_audit.json", root / "pre_gt_seal.json")


def _validate_base(sealed):
    for path in (sealed.bundle, sealed.audit, sealed.seal):
        _require(path.is_file(), "sealed prediction file is missing")
    seal = _read(sealed.seal)
    _require(seal.get("schema") == _SEAL_SCHEMA, "seal schema mismatch")
    _require(seal.get("hash_algorithm") == "SHA256", "seal hash algorithm mismatch")
    _require(seal.get("bundle_keys") == list(_BUNDLE_KEYS), "seal bundle schema mismatch")
    _require(seal.get("full_gt_loaded_before_seal") is False, "seal does not preserve the GT firewall")
    _require(seal.get("metrics_computed_before_seal") is False, "seal permits pre-seal metrics")
    _require(_sha(sealed.bundle) == seal.get("bundle_sha256"), "bundle SHA256 mismatch")
    _require(_sha(sealed.audit) == seal.get("audit_sha256"), "audit SHA256 mismatch")
    audit = _read(sealed.audit)
    _require(audit.get("schema") == _BASE_SCHEMA, "audit schema mismatch")
    base = audit.get("base", {})
    _require(all(base.get(name) is expected for name, expected in (("action_artifact_loaded", False), ("utility_artifact_loaded", False), ("semantic_artifact_loaded", False), ("phase_a_executed", False), ("semantic_optimizer_created", False), ("phase_b_executed", True))), "BASE provenance audit mismatch")
    firewall = audit.get("gt_firewall", {})
    _require(firewall.get("full_gt_loaded") is False and firewall.get("metrics_computed") is False, "audit GT firewall mismatch")
    with np.load(sealed.bundle, allow_pickle=False) as archive:
        _require(tuple(archive.files) == _BUNDLE_KEYS, "bundle key whitelist mismatch")
        sample_ids = np.ascontiguousarray(archive["sample_ids"])
        predictions = np.ascontiguousarray(archive["final_predictions"])
        q_local, q_aligned, matrix = np.asarray(archive["q_local"]), np.asarray(archive["q_aligned"]), np.asarray(archive["M_v"])
    _require(sample_ids.dtype == np.int64, "sample IDs must be int64")
    validate_sample_ids(sample_ids, sample_ids.size, require_identity=True)
    _require(predictions.dtype == np.int64 and predictions.shape == sample_ids.shape, "predictions must be int64 [N]")
    _require(q_local.ndim == 3 and q_local.shape[0] == sample_ids.size and q_aligned.shape == q_local.shape, "q state shape mismatch")
    _require(matrix.shape == (q_local.shape[1], q_local.shape[2], q_local.shape[2]), "M_v shape mismatch")
    _require(np.isfinite(q_local).all() and np.isfinite(q_aligned).all() and np.isfinite(matrix).all(), "sealed prediction state is nonfinite")
    _require(_payload_sha256(predictions) == seal.get("prediction_logical_sha256"), "prediction logical hash mismatch")
    _require(_payload_sha256(sample_ids) == seal.get("sample_id_logical_sha256"), "sample-ID logical hash mismatch")
    dataset = audit.get("scientific_config", {}).get("dataset")
    _require(isinstance(dataset, str), "sealed dataset is missing")
    return sample_ids, predictions, dataset, seal


def _validate(arm, sealed):
    if arm == "BASE":
        return _validate_base(sealed)
    if arm == "OURS_TRUE_U":
        sample_ids, predictions, dataset, seal = frozen_evaluation._validate_sealed_prediction(sealed)
        _require(_read(sealed.audit).get("schema") == _OURS_SCHEMA, "audit schema mismatch")
        return sample_ids, predictions, dataset, seal
    raise RuntimeError("FORMAL_POSTSEAL_ARM_NOT_AUTHORIZED")


def evaluate_formal_postseal(*, dataset, training_seed, arm, full_gt_path, sealed=None, output_path=None):
    """Verify arm-bound pre-GT seal completely before the first GT read."""
    sealed = _sealed_paths(dataset, training_seed, arm) if sealed is None else sealed
    _require(isinstance(sealed, SealedPredictionPaths), "sealed must be SealedPredictionPaths")
    sample_ids, predictions, sealed_dataset, seal = _validate(arm, sealed)
    _require(sealed_dataset == dataset, "sealed dataset/request mismatch")
    audit = _read(sealed.audit)
    _require(audit.get("scientific_config", {}).get("training_seed") == training_seed, "sealed training-seed/request mismatch")
    _, label_sets = load_dataset(dataset, full_gt_path)
    _require(isinstance(label_sets, list) and len(label_sets) == 1, "dataset loader returned an invalid label contract")
    labels = np.ascontiguousarray(np.squeeze(label_sets[0]), dtype=np.int64)
    _require(labels.ndim == 1 and labels.shape == sample_ids.shape, "GT/sample alignment mismatch")
    result = {"acc": float(acc(labels, predictions)), "nmi": float(nmi(labels, predictions)), "ari": float(ari(labels, predictions))}
    target = sealed.seal.parent / "postseal_metrics.json" if output_path is None else Path(output_path)
    _require(not target.exists(), "refusing to overwrite metrics output")
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": "paper-formal-postseal-metrics-v1", "dataset": dataset, "arm": arm, "training_seed": training_seed, "metrics": result, "prediction_logical_sha256": seal["prediction_logical_sha256"], "sample_id_logical_sha256": seal["sample_id_logical_sha256"], "pre_gt_bundle_sha256": seal["bundle_sha256"], "pre_gt_audit_sha256": seal["audit_sha256"], "pre_gt_seal_sha256": _sha(sealed.seal), "seal_verified_before_full_gt_load": True, "full_gt_used_for_training": False, "post_gt_mutation_path_present": False}
    with target.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    return record


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("Caltech-6V", "MSRC-v1", "BDGP"))
    parser.add_argument("--training-seed", required=True, type=int, choices=(20, 30, 50))
    parser.add_argument("--arm", required=True, choices=("BASE", "OURS_TRUE_U"))
    parser.add_argument("--full-gt-path", required=True)
    args = parser.parse_args(argv)
    evaluate_formal_postseal(dataset=args.dataset, training_seed=args.training_seed, arm=args.arm, full_gt_path=args.full_gt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
