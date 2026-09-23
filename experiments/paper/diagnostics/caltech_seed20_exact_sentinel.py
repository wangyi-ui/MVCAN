"""Gate B: one current-worktree Caltech-6V seed20 exact replay sentinel."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from experiments.paper.transfer_audit.input_artifacts import file_sha256
from release_core.data.weak_quality import ndarray_sha256
from release_core.runtime import (
    ProvenanceConfig,
    RuntimeConfig,
    evaluate_postseal,
    run_pre_gt,
)
from release_core.runtime.evaluation import _validate_sealed_prediction

from . import p0_a3_protocol as protocol


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_frozen_inputs():
    observed = {}
    for path, expected in {
        **protocol.CALTECH_INPUT_FILES,
        **protocol.CALTECH_REFERENCE_FILES,
    }.items():
        _require(path.is_file(), "Caltech frozen file is missing: " + str(path))
        actual = file_sha256(path)
        _require(actual == expected, "Caltech frozen SHA256 mismatch: " + str(path))
        observed[str(path)] = actual
    _require(protocol.CALTECH_FULL_GT.is_file(), "Caltech full GT is missing")
    return observed


def _write_json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _materialize_split_adapter(root):
    artifact = Path(root) / "seed20_sparse_split_adapter.npz"
    audit = Path(root) / "seed20_sparse_split_adapter_audit.json"
    names = ("sample_ids", "labeled_ids", "labeled_targets", "unlabeled_ids")
    with np.load(protocol.CALTECH_ACTION, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]) for name in names}
    np.savez(artifact, **arrays)
    _write_json_exclusive(audit, {
        "schema": "paper-p0-a3-exact-copy-sparse-split-adapter-v1",
        "artifact_path": str(artifact),
        "artifact_sha256": file_sha256(artifact),
        "parent_source_path": str(protocol.CALTECH_ACTION),
        "parent_source_sha256": protocol.CALTECH_INPUT_FILES[protocol.CALTECH_ACTION],
        "copied_keys": list(names),
        "scientific_recomputation": False,
        "label_reselection": False,
        "exact_array_copy": True,
    })
    return artifact, audit


def compare_payload(sealed):
    comparisons = {}
    with np.load(sealed.bundle, allow_pickle=False) as current, np.load(
        protocol.CALTECH_REFERENCE_ARTIFACT, allow_pickle=False
    ) as reference:
        for name in ("sample_ids", "final_predictions", "q_local", "q_aligned", "M_v"):
            left, right = np.asarray(current[name]), np.asarray(reference[name])
            equal = bool(np.array_equal(left, right))
            maximum = None
            if np.issubdtype(left.dtype, np.floating):
                maximum = float(np.max(np.abs(
                    left.astype(np.float64) - right.astype(np.float64)
                )))
            _require(equal, "Caltech scientific payload mismatch: " + name)
            _require(maximum in (None, 0.0), "Caltech float payload is not exact: " + name)
            comparisons[name] = {"array_equal": equal, "max_abs_diff": maximum}
        prediction_hash = ndarray_sha256(current["final_predictions"])
    _require(prediction_hash == protocol.CALTECH_EXPECTED_PREDICTION,
             "Caltech prediction logical SHA256 mismatch")
    comparisons["prediction_logical_sha256"] = prediction_hash
    return comparisons


def run_sentinel(output_dir, device="cuda:0"):
    """Run exactly one seed20 replay; this function intentionally trains."""
    output = Path(output_dir)
    _require(not output.exists(), "refusing to overwrite sentinel output")
    verified = verify_frozen_inputs()
    with tempfile.TemporaryDirectory(prefix="p0_a3_caltech_split_") as temporary:
        split, split_audit = _materialize_split_adapter(temporary)
        expected = dict(protocol.CALTECH_INPUT_FILES)
        expected[split] = file_sha256(split)
        expected[split_audit] = file_sha256(split_audit)
        runtime = RuntimeConfig(
            dataset="Caltech-6V", training_seed=20, epochs=20,
            batch_size=256, learning_rate=1e-4, native_lambda1=0.01,
            refresh_interval=100, label_seed=20, labels_per_class=2,
            device=device,
        )
        provenance = ProvenanceConfig(
            feature_artifact=protocol.CALTECH_FEATURE,
            feature_audit=protocol.CALTECH_FEATURE_AUDIT,
            sparse_split_artifact=split,
            sparse_split_audit=split_audit,
            utility_artifact=protocol.CALTECH_ACTION,
            utility_audit=protocol.CALTECH_ACTION_SEAL,
            semantic_artifact=protocol.CALTECH_ACTION,
            semantic_audit=protocol.CALTECH_ACTION_SEAL,
            checkpoint_paths=protocol.CALTECH_CHECKPOINTS,
            checkpoint_audit=protocol.CALTECH_CHECKPOINT_AUDIT,
            output_root=output,
            strict_replay=True,
            expected_file_sha256=tuple(expected.items()),
            expected_initial_model_sha256=protocol.CALTECH_EXPECTED_INITIAL,
            expected_prediction_sha256=protocol.CALTECH_EXPECTED_PREDICTION,
            historical_audit_sha256=protocol.CALTECH_REFERENCE_FILES[
                protocol.CALTECH_REFERENCE_AUDIT
            ],
            historical_seal_sha256=protocol.CALTECH_REFERENCE_FILES[
                protocol.CALTECH_REFERENCE_SEAL
            ],
        )
        sealed = run_pre_gt(runtime, provenance)
    _validate_sealed_prediction(sealed)
    comparisons = compare_payload(sealed)
    # First full-GT access occurs inside evaluate_postseal, after both gates above.
    metrics = evaluate_postseal(sealed, protocol.CALTECH_FULL_GT)
    actual = {"acc": metrics.acc, "nmi": metrics.nmi, "ari": metrics.ari}
    _require(actual == protocol.CALTECH_EXPECTED_METRICS,
             "Caltech exact metrics mismatch")
    record = {
        "schema": "paper-p0-a3-caltech-seed20-exact-sentinel-v1",
        "result": "PASS",
        "scientific_payload": comparisons,
        "metrics": actual,
        "verified_frozen_file_sha256": verified,
        "full_gt_loaded_before_seal": False,
        "seed30_run": False,
        "seed50_run": False,
    }
    _write_json_exclusive(output / "sentinel_result.json", record)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(run_sentinel(args.output_dir, args.device), sort_keys=True))


if __name__ == "__main__":
    main()
