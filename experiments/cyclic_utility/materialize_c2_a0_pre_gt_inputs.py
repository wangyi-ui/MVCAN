"""Materialize hash-identical frozen C0 inputs for the future C2-A0 stage.

This is an engineering-only, inference-only recovery utility.  It never reads
the post-GT C0 audit, full ground truth, sparse labels, corruption masks, or
Bridge correctness.  No output directory is created until both the recovered
``q_aligned`` and ``p_gen`` exactly match their original sealed logical hashes.
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import (
    c0_complementary_semantic_verification as c0_protocol,
)
from experiments.cyclic_utility import (
    evaluate_bridge_p0_residual_utility as bridge_evaluator,
)
from experiments.cyclic_utility import (
    evaluate_c0_complementary_semantic_verification as c0_evaluator,
)
from experiments.e1_pairwise_utility import (
    train_e1_pairwise_utility as e1_train,
)
from experiments.e4_semantic_memory_bank import (
    evaluate_e4_a0_memory_feasibility as frozen_e1,
)
from irv.b4_information_utility import tensor_sha256
from weak_quality import ndarray_sha256


STAGE = "C2-A0-pre-GT-input-recovery"
SEEDS = (20, 30, 50)
SAMPLE_NUM = 1400
VIEW_NUM = 6
CLASS_NUM = 7
DIRECTION_COUNT = 20

VSA_SOURCE_MANIFEST = (
    REPOSITORY_ROOT
    / "experiment_freeze/vsa_multiseed_provenance_pass_20260829"
    / "source_sha256.txt"
)
BRIDGE_SOURCE_MANIFEST = (
    REPOSITORY_ROOT
    / "experiment_freeze/bridge_p0_gt_loader_bugfix_preregistered_20260830"
    / "source_sha256.txt"
)

AUDITED_P_GEN_SHA256 = {
    20: "e5107adb236aaef9a09513c5fa43ae367c8a95290bcae73d1ccd805bd8ef3264",
    30: "c8b09b1f36a1d6a73a59bb83315b9c783f2d33b00555600f18caf4dad92af937",
    50: "2f6409a0a7d26537ed0a7932e34c5245903003a8296410f093f91e606c4a310a",
}

OUTPUT_ARRAY_NAMES = (
    "sample_ids",
    "p_gen",
    "U_cycle",
    "C_conf",
    "y_gen",
    "generator_subsets",
    "verifier_subsets",
)
FORBIDDEN_OUTPUT_KEYS = {
    "GT",
    "full_GT",
    "correct",
    "correctness",
    "global_mapping",
    "mapping",
    "labels",
    "labeled_targets",
    "corruption_mask",
    "oracle",
}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _display(path):
    value = _resolve(path).resolve()
    try:
        return str(value.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(value)


def _same_path(left, right):
    return _resolve(left).resolve() == _resolve(right).resolve()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def write_json_durable(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def _readonly(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def validate_seed(seed):
    value = int(seed)
    if value not in SEEDS:
        raise ValueError("C2-A0 recovery seed must be one of " + str(SEEDS))
    return value


@dataclass(frozen=True)
class RecoveryPaths:
    vsa_source_manifest: Path
    bridge_source_manifest: Path
    feature_path: Path
    feature_audit_path: Path
    e1_model_dir: Path
    e1_audit_path: Path
    c0_artifact_path: Path
    c0_seal_path: Path
    bridge_input_seal_path: Path
    output_dir: Path


def default_paths(seed, output_dir=None):
    active_seed = validate_seed(seed)
    c0_root = (
        REPOSITORY_ROOT
        / "outputs/cyclic_utility"
        / ("c0_complementary_semantic_verification_seed" + str(active_seed))
    )
    e1_root = (
        REPOSITORY_ROOT
        / "outputs/e1_pairwise_utility"
        / ("lwc_100ep_seed" + str(active_seed))
    )
    bridge_root = (
        REPOSITORY_ROOT
        / "outputs/cyclic_utility"
        / ("bridge_p0_residual_utility_seed" + str(active_seed))
    )
    target = (
        REPOSITORY_ROOT
        / "outputs/cyclic_utility"
        / ("c2_a0_pre_gt_input_seed" + str(active_seed))
        if output_dir is None
        else _resolve(output_dir)
    )
    return RecoveryPaths(
        vsa_source_manifest=VSA_SOURCE_MANIFEST,
        bridge_source_manifest=BRIDGE_SOURCE_MANIFEST,
        feature_path=_resolve(e1_train.DEFAULT_FEATURE_PATH),
        feature_audit_path=_resolve(e1_train.DEFAULT_FEATURE_AUDIT_PATH),
        e1_model_dir=e1_root / "models",
        e1_audit_path=e1_root / "e1_audit.json",
        c0_artifact_path=c0_root / "c0_predictions_and_scores.npz",
        c0_seal_path=c0_root / "c0_prediction_seal.json",
        bridge_input_seal_path=bridge_root / "bridge_input_seal.json",
        output_dir=target,
    )


def verify_sha256_manifest(manifest_path):
    """Verify one repository-relative GNU-style SHA256 manifest."""
    path = _resolve(manifest_path)
    _require(path.is_file(), "source SHA256 manifest is missing: " + str(path))
    records = []
    seen = set()
    with open(path, "r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split(None, 1)
            _require(
                len(fields) == 2 and len(fields[0]) == 64,
                "invalid SHA256 manifest line " + str(line_number),
            )
            expected_hash, relative_name = fields
            relative_name = relative_name.lstrip("*")
            _require(
                relative_name not in seen,
                "duplicate SHA256 manifest path: " + relative_name,
            )
            seen.add(relative_name)
            source_path = (REPOSITORY_ROOT / relative_name).resolve()
            try:
                source_path.relative_to(REPOSITORY_ROOT.resolve())
            except ValueError as error:
                raise RuntimeError(
                    "SHA256 manifest path escapes repository: " + relative_name
                ) from error
            _require(
                source_path.is_file(),
                "manifest source is missing: " + relative_name,
            )
            actual_hash = file_sha256(source_path)
            _require(
                actual_hash == expected_hash,
                "source SHA256 mismatch: " + relative_name,
            )
            records.append({
                "path": relative_name,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "exact_match": True,
            })
    _require(records, "source SHA256 manifest is empty")
    return {
        "path": _display(path),
        "file_sha256": file_sha256(path),
        "entry_count": len(records),
        "entries": records,
        "all_entries_exact_match": True,
    }


def resolve_authoritative_e1_training_seed(requested_seed, e1_audit, c0_seal):
    """Resolve seed identity solely from E1; the legacy C0 field is metadata."""
    requested = validate_seed(requested_seed)
    actual_value = e1_audit.get("training_seed", e1_audit.get("seed"))
    _require(actual_value is not None, "E1 audit training seed is missing")
    actual = validate_seed(actual_value)
    _require(
        actual == requested
        and int(e1_audit.get("seed", actual)) == actual,
        "requested seed and same-seed E1 provenance mismatch",
    )
    return {
        "requested_seed": requested,
        "verified_actual_E1_training_seed": actual,
        "legacy_C0_seed_field": c0_seal.get("seed"),
        "legacy_C0_seed_field_authoritative": False,
        "same_seed_E1_provenance_pass": True,
    }


def verify_canonical_sample_ids(sample_ids, feature_audit, e1_audit):
    values = np.ascontiguousarray(sample_ids, dtype=np.int64)
    canonical = np.arange(SAMPLE_NUM, dtype=np.int64)
    weak_hash = ndarray_sha256(values)
    logical_hash = tensor_sha256(values)
    e1_feature = e1_audit.get("feature_provenance", {})
    _require(
        values.shape == (SAMPLE_NUM,)
        and np.array_equal(values, canonical)
        and feature_audit.get("sample_ids_exact_arange_pass") is True
        and feature_audit.get("sample_ids_dtype") == "int64"
        and feature_audit.get("sample_ids_shape") == [SAMPLE_NUM]
        and feature_audit.get("sample_ids_sha256") == weak_hash
        and e1_feature.get("sample_ids_exact_arange_pass") is True
        and e1_feature.get("sample_ids_sha256") == weak_hash,
        "canonical E0/E1 sample-ID provenance mismatch",
    )
    canonical.setflags(write=False)
    return canonical, {
        "shape": [SAMPLE_NUM],
        "dtype": "int64",
        "exact_arange_1400": True,
        "zero_based": True,
        "E0_E1_provenance_match": True,
        "E0_ndarray_sha256": weak_hash,
        "logical_sha256": logical_hash,
    }


def _verify_e1_provenance(seed, paths, e1_audit):
    _require(
        e1_audit.get("stage") == "E1"
        and e1_audit.get("arm") == "LWC"
        and e1_audit.get("epochs") == 100
        and e1_audit.get("N") == SAMPLE_NUM
        and e1_audit.get("V") == VIEW_NUM
        and e1_audit.get("K") == CLASS_NUM,
        "E1 audit is not the frozen 100-epoch LWC artifact",
    )
    expected_paths = frozen_e1._expected_checkpoint_paths(paths.e1_model_dir)
    output_record = e1_audit.get("final_model_outputs", {})
    recorded_paths = [_resolve(value) for value in output_record.get("paths", ())]
    recorded_hashes = list(output_record.get("file_sha256", ()))
    _require(
        len(expected_paths) == len(recorded_paths) == len(recorded_hashes) == VIEW_NUM
        and all(
            _same_path(expected, recorded)
            for expected, recorded in zip(expected_paths, recorded_paths)
        ),
        "E1 final checkpoint path provenance mismatch",
    )
    actual_hashes = []
    for checkpoint_path, expected_hash in zip(expected_paths, recorded_hashes):
        _require(checkpoint_path.is_file(), "E1 checkpoint is missing")
        actual_hash = file_sha256(checkpoint_path)
        _require(
            actual_hash == expected_hash,
            "E1 checkpoint SHA256 mismatch: " + _display(checkpoint_path),
        )
        actual_hashes.append(actual_hash)
    return {
        "seed": validate_seed(seed),
        "audit_path": _display(paths.e1_audit_path),
        "audit_file_sha256": file_sha256(paths.e1_audit_path),
        "model_dir": _display(paths.e1_model_dir),
        "checkpoint_paths": [_display(path) for path in expected_paths],
        "checkpoint_file_sha256": actual_hashes,
        "all_checkpoint_hashes_exact_match": True,
    }


def _verify_c0_pre_gt_files(seed, paths, c0_seal, bridge_input_seal):
    artifact_hash = file_sha256(paths.c0_artifact_path)
    seal_hash = file_sha256(paths.c0_seal_path)
    _require(
        c0_seal.get("stage") == "C0"
        and _same_path(c0_seal.get("artifact_path", ""), paths.c0_artifact_path)
        and c0_seal.get("artifact_file_sha256") == artifact_hash
        and c0_seal.get("scores_completed_before_GT") is True
        and c0_seal.get("scores_saved_before_GT") is True
        and c0_seal.get("scores_hashed_before_GT") is True
        and c0_seal.get("scores_reloaded_before_GT") is True
        and c0_seal.get("full_GT_loaded_before_seal") is False
        and c0_seal.get("R_loaded_before_seal") is False
        and c0_seal.get("sparse_labels_loaded_before_seal") is False
        and c0_seal.get("corruption_mask_loaded_before_seal") is False
        and c0_seal.get("oracle_loaded_before_seal") is False,
        "C0 pre-GT prediction seal boundary mismatch",
    )
    _require(
        bridge_input_seal.get("stage") == "Bridge-P0"
        and bridge_input_seal.get("seed") == validate_seed(seed)
        and bridge_input_seal.get("E1_training_seed") == validate_seed(seed)
        and bridge_input_seal.get("inputs_verified_before_GT") is True
        and bridge_input_seal.get("GT_loaded_before_bridge_input_seal") is False
        and _same_path(
            bridge_input_seal.get("C0_artifact_path", ""),
            paths.c0_artifact_path,
        )
        and bridge_input_seal.get("C0_artifact_file_sha256") == artifact_hash
        and _same_path(
            bridge_input_seal.get("C0_seal_path", ""), paths.c0_seal_path
        )
        and bridge_input_seal.get("C0_seal_file_sha256") == seal_hash,
        "Bridge pre-GT C0 file provenance mismatch",
    )
    p_gen_hash = c0_seal.get("p_gen_logical_sha256")
    q_hash = c0_seal.get("q_aligned_provenance", {}).get("logical_sha256")
    _require(
        p_gen_hash == AUDITED_P_GEN_SHA256[validate_seed(seed)]
        and isinstance(q_hash, str)
        and len(q_hash) == 64,
        "sealed q_aligned/p_gen logical SHA256 boundary mismatch",
    )
    return {
        "artifact_path": _display(paths.c0_artifact_path),
        "artifact_file_sha256": artifact_hash,
        "prediction_seal_path": _display(paths.c0_seal_path),
        "prediction_seal_file_sha256": seal_hash,
        "bridge_input_seal_path": _display(paths.bridge_input_seal_path),
        "bridge_input_seal_file_sha256": file_sha256(
            paths.bridge_input_seal_path
        ),
        "q_aligned_original_logical_sha256": q_hash,
        "p_gen_original_logical_sha256": p_gen_hash,
        "C0_artifact_content_loaded_before_exact_hash_gates": False,
        "C0_prediction_seal_pre_GT_pass": True,
    }


def preflight_recovery(seed, paths=None):
    """Complete every source/file/seed check required before model forward."""
    active_seed = validate_seed(seed)
    active_paths = default_paths(active_seed) if paths is None else paths
    _require(
        not active_paths.output_dir.exists(),
        "refusing to overwrite C2-A0 recovered input output",
    )

    source_manifests = {
        "VSA_multiseed": verify_sha256_manifest(
            active_paths.vsa_source_manifest
        ),
        "Bridge_GT_loader_bugfix": verify_sha256_manifest(
            active_paths.bridge_source_manifest
        ),
    }
    bridge_frozen_sources = bridge_evaluator.verify_frozen_source_integrity()

    required_files = (
        active_paths.e1_audit_path,
        active_paths.feature_path,
        active_paths.feature_audit_path,
        active_paths.c0_artifact_path,
        active_paths.c0_seal_path,
        active_paths.bridge_input_seal_path,
    )
    _require(all(path.is_file() for path in required_files), "recovery input missing")
    e1_audit = read_json(active_paths.e1_audit_path)
    c0_seal = read_json(active_paths.c0_seal_path)
    bridge_input_seal = read_json(active_paths.bridge_input_seal_path)

    seed_audit = resolve_authoritative_e1_training_seed(
        active_seed, e1_audit, c0_seal
    )
    e1_provenance = _verify_e1_provenance(
        active_seed, active_paths, e1_audit
    )
    _require(
        _same_path(
            bridge_input_seal.get("E1_audit_path", ""),
            active_paths.e1_audit_path,
        )
        and bridge_input_seal.get("E1_audit_file_sha256")
        == e1_provenance["audit_file_sha256"],
        "Bridge/E1 same-seed audit provenance mismatch",
    )
    c0_provenance = _verify_c0_pre_gt_files(
        active_seed, active_paths, c0_seal, bridge_input_seal
    )

    _, feature_sample_ids, feature_audit = (
        e1_train.load_frozen_feature_artifact(
            active_paths.feature_path, active_paths.feature_audit_path
        )
    )
    _require(
        _same_path(
            e1_audit.get("feature_provenance", {}).get("path", ""),
            active_paths.feature_path,
        )
        and e1_audit.get("feature_provenance", {}).get("file_sha256")
        == file_sha256(active_paths.feature_path)
        == bridge_input_seal.get("fixed_feature_file_sha256"),
        "E0/E1/Bridge frozen feature provenance mismatch",
    )
    sample_ids, sample_id_audit = verify_canonical_sample_ids(
        feature_sample_ids, feature_audit, e1_audit
    )

    return {
        "seed": active_seed,
        "paths": active_paths,
        "source_manifests": source_manifests,
        "bridge_frozen_sources": bridge_frozen_sources,
        "seed_audit": seed_audit,
        "e1_provenance": e1_provenance,
        "c0_provenance": c0_provenance,
        "c0_seal": c0_seal,
        "sample_ids": sample_ids,
        "sample_id_audit": sample_id_audit,
        "preflight_completed_before_forward": True,
        "GT_loaded": False,
        "labels_loaded": False,
        "corruption_mask_loaded": False,
    }


def reconstruct_frozen_posteriors(preflight, device="cpu"):
    """Reuse the exact frozen E1/C0 inference and posterior implementation."""
    _require(device == "cpu", "hash recovery is restricted to deterministic CPU")
    paths = preflight["paths"]
    with torch.inference_mode():
        q_aligned, _, sample_ids, representation_audit = (
            c0_evaluator.load_frozen_e1_aligned_q(
                feature_path=paths.feature_path,
                feature_audit_path=paths.feature_audit_path,
                model_dir=paths.e1_model_dir,
                model_audit_path=paths.e1_audit_path,
                device=device,
            )
        )
        posteriors = c0_protocol.build_complementary_posteriors(q_aligned)

    checkpoint = representation_audit.get("checkpoint", {})
    expected_checkpoint = preflight["e1_provenance"]
    _require(
        checkpoint.get("checkpoint_file_sha256")
        == expected_checkpoint["checkpoint_file_sha256"]
        and all(
            _same_path(left, right)
            for left, right in zip(
                checkpoint.get("checkpoint_paths", ()),
                expected_checkpoint["checkpoint_paths"],
            )
        ),
        "inference checkpoint provenance changed after preflight",
    )
    _require(
        np.array_equal(sample_ids, preflight["sample_ids"]),
        "inference sample IDs differ from verified canonical IDs",
    )
    q_values = _readonly(q_aligned.detach().cpu().numpy())
    p_gen = _readonly(
        posteriors["p_gen"].detach().cpu().numpy(), dtype=np.float32
    )
    generator = _readonly(
        posteriors["generator_subsets"].detach().cpu().numpy(),
        dtype=np.int64,
    )
    verifier = _readonly(
        posteriors["verifier_subsets"].detach().cpu().numpy(),
        dtype=np.int64,
    )
    _require(
        not q_values.flags.writeable
        and not p_gen.flags.writeable
        and q_values.shape == (SAMPLE_NUM, VIEW_NUM, CLASS_NUM)
        and p_gen.shape == (SAMPLE_NUM, DIRECTION_COUNT, CLASS_NUM),
        "recovered frozen posterior boundary mismatch",
    )
    return {
        "q_aligned": q_values,
        "p_gen": p_gen,
        "generator_subsets": generator,
        "verifier_subsets": verifier,
        "representation_audit": representation_audit,
        "inference_only": True,
    }


def require_exact_logical_hash(name, value, expected_hash):
    actual_hash = tensor_sha256(value)
    _require(
        actual_hash == expected_hash,
        name + " exact logical SHA256 mismatch",
    )
    return actual_hash


def verify_exact_recovery_hashes(preflight, recovered):
    q_expected = preflight["c0_provenance"][
        "q_aligned_original_logical_sha256"
    ]
    p_expected = preflight["c0_provenance"][
        "p_gen_original_logical_sha256"
    ]
    q_actual = require_exact_logical_hash(
        "q_aligned", recovered["q_aligned"], q_expected
    )
    p_actual = require_exact_logical_hash(
        "p_gen", recovered["p_gen"], p_expected
    )
    return {
        "q_aligned_original_logical_sha256": q_expected,
        "q_aligned_recovered_logical_sha256": q_actual,
        "exact_q_aligned_hash_match": True,
        "p_gen_original_logical_sha256": p_expected,
        "p_gen_recovered_logical_sha256": p_actual,
        "exact_p_gen_hash_match": True,
        "hash_tolerance_used": False,
    }


def _load_sealed_array(archive, seal, name, dtype, shape):
    _require(name in archive.files, "C0 artifact missing array: " + name)
    value = np.array(archive[name], copy=True, order="C")
    record = seal.get("arrays", {}).get(name, {})
    _require(
        value.dtype == np.dtype(dtype)
        and value.shape == shape
        and record.get("dtype") == str(value.dtype)
        and record.get("shape") == list(shape)
        and record.get("logical_sha256") == tensor_sha256(value),
        "C0 sealed array mismatch: " + name,
    )
    value.setflags(write=False)
    return value


def crosscheck_recovered_p_gen(p_gen, c0_arrays):
    """Check recovered p_gen against existing pre-GT C0 predictions only."""
    values = _readonly(p_gen, dtype=np.float32)
    _require(
        values.shape == (SAMPLE_NUM, DIRECTION_COUNT, CLASS_NUM)
        and np.isfinite(values).all()
        and np.all(values >= 0.0)
        and np.all(values <= 1.0 + c0_protocol.BOUND_ATOL)
        and np.allclose(
            values.sum(axis=-1),
            np.ones((SAMPLE_NUM, DIRECTION_COUNT), dtype=np.float32),
            rtol=0.0,
            atol=c0_protocol.PROBABILITY_ATOL,
        ),
        "recovered p_gen probability boundary mismatch",
    )
    predicted = np.ascontiguousarray(values.argmax(axis=-1), dtype=np.int64)
    confidence = np.ascontiguousarray(values.max(axis=-1), dtype=np.float32)
    y_gen = c0_arrays["y_gen"]
    c_conf = c0_arrays["C_conf"]
    conf_gen = c0_arrays["conf_gen"]
    agreement_count = int(np.count_nonzero(predicted == y_gen))
    argmax_pass = bool(agreement_count == SAMPLE_NUM * DIRECTION_COUNT)
    c_conf_pass = bool(np.allclose(
        confidence,
        c_conf,
        rtol=0.0,
        atol=c0_protocol.PROBABILITY_ATOL,
    ))
    conf_gen_pass = bool(np.allclose(
        confidence,
        conf_gen,
        rtol=0.0,
        atol=c0_protocol.PROBABILITY_ATOL,
    ))
    _require(argmax_pass, "argmax(p_gen) does not exactly match y_gen")
    _require(
        c_conf_pass and conf_gen_pass,
        "max(p_gen) does not match frozen generator confidence",
    )
    return {
        "p_gen_shape": list(values.shape),
        "p_gen_dtype": str(values.dtype),
        "p_gen_finite_pass": True,
        "p_gen_probability_mass_pass": True,
        "argmax_agreement_count": agreement_count,
        "argmax_agreement_total": SAMPLE_NUM * DIRECTION_COUNT,
        "p_gen_y_gen_argmax_agreement": argmax_pass,
        "p_gen_C_conf_max_probability_agreement": c_conf_pass,
        "p_gen_conf_gen_max_probability_agreement": conf_gen_pass,
        "C_conf_max_abs_difference": float(
            np.max(np.abs(confidence.astype(np.float64) - c_conf))
        ),
        "confidence_atol": c0_protocol.PROBABILITY_ATOL,
        "confidence_rtol": 0.0,
    }


def load_and_crosscheck_pre_gt_artifact(preflight, recovered):
    """Load C0 NPZ only after exact q/p hash gates have already passed."""
    paths = preflight["paths"]
    seal = preflight["c0_seal"]
    _require(
        file_sha256(paths.c0_artifact_path)
        == preflight["c0_provenance"]["artifact_file_sha256"],
        "C0 artifact changed after preflight",
    )
    with np.load(paths.c0_artifact_path, allow_pickle=False) as archive:
        y_gen = _load_sealed_array(
            archive, seal, "y_gen", np.int64, (SAMPLE_NUM, DIRECTION_COUNT)
        )
        u_cycle = _load_sealed_array(
            archive,
            seal,
            "U_cycle",
            np.float32,
            (SAMPLE_NUM, DIRECTION_COUNT),
        )
        c_conf = _load_sealed_array(
            archive,
            seal,
            "C_conf",
            np.float32,
            (SAMPLE_NUM, DIRECTION_COUNT),
        )
        conf_gen = _load_sealed_array(
            archive,
            seal,
            "conf_gen",
            np.float32,
            (SAMPLE_NUM, DIRECTION_COUNT),
        )
        generator = _load_sealed_array(
            archive,
            seal,
            "generator_subsets",
            np.int64,
            (DIRECTION_COUNT, 3),
        )
        verifier = _load_sealed_array(
            archive,
            seal,
            "verifier_subsets",
            np.int64,
            (DIRECTION_COUNT, 3),
        )
        archive_fields = list(archive.files)

    _require(
        np.array_equal(c_conf, conf_gen),
        "frozen C_conf and conf_gen differ",
    )
    split_audit = c0_protocol.validate_complementary_splits(
        generator, verifier
    )
    expected_generator, expected_verifier = (
        c0_protocol.enumerate_complementary_splits()
    )
    _require(
        np.array_equal(generator, expected_generator)
        and np.array_equal(verifier, expected_verifier)
        and np.array_equal(generator, recovered["generator_subsets"])
        and np.array_equal(verifier, recovered["verifier_subsets"]),
        "frozen complementary direction definitions mismatch",
    )
    arrays = {
        "sample_ids": preflight["sample_ids"],
        "p_gen": recovered["p_gen"],
        "U_cycle": u_cycle,
        "C_conf": c_conf,
        "y_gen": y_gen,
        "generator_subsets": generator,
        "verifier_subsets": verifier,
    }
    _require(
        tuple(arrays) == OUTPUT_ARRAY_NAMES
        and set(arrays).isdisjoint(FORBIDDEN_OUTPUT_KEYS),
        "C2 pre-GT output array schema mismatch",
    )
    crosscheck = crosscheck_recovered_p_gen(recovered["p_gen"], {
        "y_gen": y_gen,
        "C_conf": c_conf,
        "conf_gen": conf_gen,
    })
    return arrays, {
        **crosscheck,
        "C0_archive_fields": archive_fields,
        "C0_artifact_loaded_after_exact_hash_gates": True,
        "C_conf_exactly_equals_conf_gen": True,
        "generator_verifier_splits_exact_frozen_match": True,
        "split_audit": split_audit,
    }


def _array_records(arrays):
    return {
        name: {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "logical_sha256": tensor_sha256(array),
            "read_only_in_memory": not array.flags.writeable,
        }
        for name, array in arrays.items()
    }


def persist_recovery_bundle(preflight, arrays, hash_audit, crosscheck):
    """Persist the self-contained PRE-GT bundle after all hard gates pass."""
    output_root = preflight["paths"].output_dir
    _require(not output_root.exists(), "refusing to overwrite recovery output")
    _require(
        hash_audit.get("exact_q_aligned_hash_match") is True
        and hash_audit.get("exact_p_gen_hash_match") is True
        and crosscheck.get("p_gen_y_gen_argmax_agreement") is True
        and crosscheck.get("p_gen_C_conf_max_probability_agreement") is True,
        "recovery output gate did not pass",
    )
    for array in arrays.values():
        _require(not array.flags.writeable, "output array is not read-only")

    output_root.mkdir(parents=True, exist_ok=False)
    npz_path = output_root / "c2_pre_gt_inputs.npz"
    seal_path = output_root / "c2_input_recovery_seal.json"
    audit_path = output_root / "c2_input_recovery_audit.json"
    with open(npz_path, "wb") as output_file:
        np.savez(output_file, **arrays)
        output_file.flush()
        os.fsync(output_file.fileno())

    with np.load(npz_path, allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == OUTPUT_ARRAY_NAMES
            and set(archive.files).isdisjoint(FORBIDDEN_OUTPUT_KEYS),
            "persisted C2 pre-GT NPZ schema mismatch",
        )
        for name, expected in arrays.items():
            actual = np.array(archive[name], copy=True, order="C")
            _require(
                np.array_equal(actual, expected)
                and tensor_sha256(actual) == tensor_sha256(expected),
                "persisted C2 input differs after reload: " + name,
            )

    seed_audit = preflight["seed_audit"]
    e1_provenance = preflight["e1_provenance"]
    c0_provenance = preflight["c0_provenance"]
    common = {
        "stage": STAGE,
        "engineering_only": True,
        "requested_seed": seed_audit["requested_seed"],
        "verified_actual_E1_training_seed": seed_audit[
            "verified_actual_E1_training_seed"
        ],
        "legacy_C0_seed_field": seed_audit["legacy_C0_seed_field"],
        "legacy_C0_seed_field_authoritative": False,
        "E1_audit_path": e1_provenance["audit_path"],
        "E1_audit_file_sha256": e1_provenance["audit_file_sha256"],
        "E1_model_dir": e1_provenance["model_dir"],
        "E1_checkpoint_paths": e1_provenance["checkpoint_paths"],
        "E1_checkpoint_SHA256": e1_provenance[
            "checkpoint_file_sha256"
        ],
        "C0_prediction_NPZ_path": c0_provenance["artifact_path"],
        "C0_prediction_NPZ_file_SHA256": c0_provenance[
            "artifact_file_sha256"
        ],
        "C0_prediction_seal_path": c0_provenance[
            "prediction_seal_path"
        ],
        "C0_prediction_seal_file_SHA256": c0_provenance[
            "prediction_seal_file_sha256"
        ],
        **hash_audit,
        "sample_ids_logical_SHA256": tensor_sha256(arrays["sample_ids"]),
        "generator_subsets_logical_SHA256": tensor_sha256(
            arrays["generator_subsets"]
        ),
        "verifier_subsets_logical_SHA256": tensor_sha256(
            arrays["verifier_subsets"]
        ),
        "U_cycle_logical_SHA256": tensor_sha256(arrays["U_cycle"]),
        "C_conf_logical_SHA256": tensor_sha256(arrays["C_conf"]),
        "y_gen_logical_SHA256": tensor_sha256(arrays["y_gen"]),
        "p_gen_y_gen_argmax_agreement": crosscheck[
            "p_gen_y_gen_argmax_agreement"
        ],
        "argmax_agreement_count": crosscheck["argmax_agreement_count"],
        "argmax_agreement_total": crosscheck["argmax_agreement_total"],
        "p_gen_C_conf_max_probability_agreement": crosscheck[
            "p_gen_C_conf_max_probability_agreement"
        ],
        "GT_loaded": False,
        "full_GT_loader_called": False,
        "c0_post_GT_audit_loaded": False,
        "Bridge_correctness_loaded": False,
        "global_GT_mapping_loaded": False,
        "labels_loaded": False,
        "sparse_label_loader_called": False,
        "corruption_mask_loaded": False,
        "training_performed": False,
        "backward_called": False,
        "optimizer_step_called": False,
        "weak_quality_scores_regenerated": False,
        "C2_scientific_calibration_performed": False,
    }
    records = _array_records(arrays)
    seal = {
        **common,
        "output_NPZ_path": _display(npz_path),
        "output_NPZ_file_SHA256": file_sha256(npz_path),
        "arrays": records,
        "NPZ_saved_fsynced_reloaded_pass": True,
    }
    write_json_durable(seal_path, seal)
    _require(read_json(seal_path) == seal, "recovery seal durable reload mismatch")
    audit = {
        **common,
        "RECOVERY_AUDIT_PASS": True,
        "source_manifests": preflight["source_manifests"],
        "bridge_frozen_source_integrity": preflight[
            "bridge_frozen_sources"
        ],
        "sample_id_audit": preflight["sample_id_audit"],
        "crosscheck": crosscheck,
        "output_array_names": list(OUTPUT_ARRAY_NAMES),
        "forbidden_output_keys_absent": True,
        "output_NPZ_path": _display(npz_path),
        "output_NPZ_file_SHA256": file_sha256(npz_path),
        "recovery_seal_path": _display(seal_path),
        "recovery_seal_file_SHA256": file_sha256(seal_path),
        "recovery_seal_reloaded_pass": True,
    }
    write_json_durable(audit_path, audit)
    _require(
        read_json(audit_path) == audit,
        "recovery audit durable reload mismatch",
    )
    return {
        "output_dir": output_root,
        "npz_path": npz_path,
        "seal_path": seal_path,
        "audit_path": audit_path,
        "seal": seal,
        "audit": audit,
        "arrays": arrays,
    }


def materialize(seed, output_dir=None, device="cpu"):
    """Run one requested frozen recovery; persist only after every hard gate."""
    active_seed = validate_seed(seed)
    paths = default_paths(active_seed, output_dir=output_dir)
    preflight = preflight_recovery(active_seed, paths=paths)
    recovered = reconstruct_frozen_posteriors(preflight, device=device)
    hash_audit = verify_exact_recovery_hashes(preflight, recovered)
    arrays, crosscheck = load_and_crosscheck_pre_gt_artifact(
        preflight, recovered
    )
    return persist_recovery_bundle(
        preflight, arrays, hash_audit, crosscheck
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = materialize(
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
    )
    audit = result["audit"]
    print("Q_ALIGNED_EXPECTED_SHA256=" + audit[
        "q_aligned_original_logical_sha256"
    ])
    print("Q_ALIGNED_RECOVERED_SHA256=" + audit[
        "q_aligned_recovered_logical_sha256"
    ])
    print("EXACT_Q_ALIGNED_HASH_MATCH=True")
    print("P_GEN_EXPECTED_SHA256=" + audit[
        "p_gen_original_logical_sha256"
    ])
    print("P_GEN_RECOVERED_SHA256=" + audit[
        "p_gen_recovered_logical_sha256"
    ])
    print("EXACT_P_GEN_HASH_MATCH=True")
    print(
        "ARGMAX_AGREEMENT="
        + str(audit["argmax_agreement_count"])
        + "/"
        + str(audit["argmax_agreement_total"])
    )
    print("C_CONF_AGREEMENT=True")
    print("GT_LOADED=False")
    print("LABELS_LOADED=False")
    print("TRAINING_PERFORMED=False")
    print("Saved: " + _display(result["output_dir"]))
    print("C2_A0_INPUT_RECOVERY_SEED" + str(args.seed) + "_PASS=True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
