"""Seal-first post-GT evaluation with no model or training dependency."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from release_core.data import load_dataset, validate_sample_ids
from release_core.evaluation import acc, ari, nmi

from . import (
    Metrics,
    SealedPredictionPaths,
    _BUNDLE_KEYS,
    _SEAL_SCHEMA,
    _payload_sha256,
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError("audit or seal root must be an object")
    return value


def _write_json(path, record):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_sealed_prediction(sealed):
    if not isinstance(sealed, SealedPredictionPaths):
        raise TypeError("sealed must be SealedPredictionPaths")
    for path in (sealed.bundle, sealed.audit, sealed.seal):
        _require(path.is_file(), "sealed prediction file is missing")
    seal = _read_json(sealed.seal)
    _require(seal.get("schema") == _SEAL_SCHEMA, "seal schema mismatch")
    _require(seal.get("hash_algorithm") == "SHA256", "seal hash algorithm mismatch")
    _require(seal.get("bundle_keys") == list(_BUNDLE_KEYS), "seal bundle schema mismatch")
    _require(seal.get("full_gt_loaded_before_seal") is False,
             "seal does not preserve the GT firewall")
    _require(seal.get("metrics_computed_before_seal") is False,
             "seal permits pre-seal metrics")
    _require(_file_sha256(sealed.bundle) == seal.get("bundle_sha256"),
             "bundle SHA256 mismatch")
    _require(_file_sha256(sealed.audit) == seal.get("audit_sha256"),
             "audit SHA256 mismatch")
    audit = _read_json(sealed.audit)
    _require(audit.get("schema") == "release-core-pre-gt-audit-v1",
             "audit schema mismatch")
    firewall = audit.get("gt_firewall", {})
    _require(
        firewall.get("full_gt_loaded") is False
        and firewall.get("metrics_computed") is False
        and firewall.get("full_gt_present_in_bundle") is False,
        "audit GT firewall mismatch",
    )
    with np.load(sealed.bundle, allow_pickle=False) as archive:
        _require(tuple(archive.files) == _BUNDLE_KEYS, "bundle key whitelist mismatch")
        sample_ids = np.ascontiguousarray(archive["sample_ids"])
        predictions = np.ascontiguousarray(archive["final_predictions"])
        q_local = np.asarray(archive["q_local"])
        q_aligned = np.asarray(archive["q_aligned"])
        matrix = np.asarray(archive["M_v"])
    _require(sample_ids.dtype == np.int64, "sample IDs must be int64")
    validate_sample_ids(sample_ids, sample_ids.size, require_identity=True)
    _require(predictions.dtype == np.int64 and predictions.shape == sample_ids.shape,
             "predictions must be int64 [N]")
    _require(q_local.ndim == 3 and q_local.shape[0] == sample_ids.size,
             "q_local must have shape [N,V,K]")
    _require(q_aligned.shape == q_local.shape, "q_aligned shape mismatch")
    _require(
        matrix.shape == (q_local.shape[1], q_local.shape[2], q_local.shape[2]),
        "M_v must have shape [V,K,K]",
    )
    _require(
        np.isfinite(q_local).all() and np.isfinite(q_aligned).all()
        and np.isfinite(matrix).all(),
        "sealed prediction state is nonfinite",
    )
    _require(_payload_sha256(predictions) == seal.get("prediction_logical_sha256"),
             "prediction logical hash mismatch")
    _require(_payload_sha256(sample_ids) == seal.get("sample_id_logical_sha256"),
             "sample-ID logical hash mismatch")
    dataset = audit.get("scientific_config", {}).get("dataset")
    _require(isinstance(dataset, str), "sealed dataset is missing")
    return sample_ids, predictions, dataset, seal


def _load_full_gt(dataset, full_gt_path):
    # This is intentionally the first full-GT access, after complete seal validation.
    _, label_sets = load_dataset(dataset, full_gt_path)
    _require(isinstance(label_sets, list) and len(label_sets) == 1,
             "dataset loader returned an invalid label contract")
    labels = np.ascontiguousarray(np.squeeze(label_sets[0]), dtype=np.int64)
    _require(labels.ndim == 1, "full GT must have shape [N]")
    return labels


def evaluate_postseal(sealed, full_gt_path, *, output_path=None):
    """Verify artifact and audit hashes, then and only then load full GT."""
    sample_ids, predictions, dataset, seal = _validate_sealed_prediction(sealed)
    labels = _load_full_gt(dataset, full_gt_path)
    _require(labels.shape == sample_ids.shape, "GT/sample alignment mismatch")
    result = Metrics(
        acc=float(acc(labels, predictions)),
        nmi=float(nmi(labels, predictions)),
        ari=float(ari(labels, predictions)),
    )
    target = sealed.seal.parent / "postseal_metrics.json" \
        if output_path is None else Path(output_path)
    _require(not target.exists(), "refusing to overwrite metrics output")
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_json(target, {
        "schema": "release-core-postseal-metrics-v1",
        "dataset": dataset,
        "metrics": {"acc": result.acc, "nmi": result.nmi, "ari": result.ari},
        "prediction_logical_sha256": seal["prediction_logical_sha256"],
        "sample_id_logical_sha256": seal["sample_id_logical_sha256"],
        "pre_gt_bundle_sha256": seal["bundle_sha256"],
        "pre_gt_audit_sha256": seal["audit_sha256"],
        "pre_gt_seal_sha256": _file_sha256(sealed.seal),
        "seal_verified_before_full_gt_load": True,
        "full_gt_used_for_training": False,
        "post_gt_mutation_path_present": False,
    })
    return result
