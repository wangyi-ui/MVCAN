import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments.b7_sparse_supervision import (
    b7_sparse_anchor_protocol as b7_protocol,
)
from experiments.cyclic_utility import (
    evaluate_c0_complementary_semantic_verification as c0_evaluator,
)
from experiments.cyclic_utility import (
    materialize_c2_a0_pre_gt_inputs as recovery,
)
from experiments.e2_sparse_semantic_utility import (
    train_e2_sparse_semantic_utility as e2_train,
)
from experiments.e4_semantic_memory_bank import (
    evaluate_e4_a0_memory_feasibility as frozen_e1,
)
from irv.b4_information_utility import tensor_sha256
from weak_quality import ndarray_sha256


def _readonly(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


@pytest.fixture(scope="module")
def synthetic_values():
    rows = np.arange(recovery.SAMPLE_NUM, dtype=np.int64)[:, None]
    directions = np.arange(recovery.DIRECTION_COUNT, dtype=np.int64)[None, :]
    classes = (rows + directions) % recovery.CLASS_NUM
    p_gen = np.full(
        (
            recovery.SAMPLE_NUM,
            recovery.DIRECTION_COUNT,
            recovery.CLASS_NUM,
        ),
        0.05,
        dtype=np.float32,
    )
    np.put_along_axis(p_gen, classes[..., None], 0.7, axis=-1)
    y_gen = np.ascontiguousarray(p_gen.argmax(axis=-1), dtype=np.int64)
    confidence = np.ascontiguousarray(p_gen.max(axis=-1), dtype=np.float32)
    q_aligned = np.full(
        (recovery.SAMPLE_NUM, recovery.VIEW_NUM, recovery.CLASS_NUM),
        np.float32(1.0 / recovery.CLASS_NUM),
        dtype=np.float32,
    )
    generator, verifier = recovery.c0_protocol.enumerate_complementary_splits()
    arrays = {
        "sample_ids": _readonly(
            np.arange(recovery.SAMPLE_NUM), dtype=np.int64
        ),
        "p_gen": _readonly(p_gen, dtype=np.float32),
        "U_cycle": _readonly(confidence * np.float32(0.5), dtype=np.float32),
        "C_conf": _readonly(confidence, dtype=np.float32),
        "y_gen": _readonly(y_gen, dtype=np.int64),
        "generator_subsets": _readonly(generator, dtype=np.int64),
        "verifier_subsets": _readonly(verifier, dtype=np.int64),
    }
    return {
        "q_aligned": _readonly(q_aligned, dtype=np.float32),
        "p_gen": arrays["p_gen"],
        "generator_subsets": arrays["generator_subsets"],
        "verifier_subsets": arrays["verifier_subsets"],
        "arrays": arrays,
    }


def _synthetic_preflight(paths, values, q_hash=None, p_hash=None):
    q_expected = tensor_sha256(values["q_aligned"]) if q_hash is None else q_hash
    p_expected = tensor_sha256(values["p_gen"]) if p_hash is None else p_hash
    return {
        "seed": 20,
        "paths": paths,
        "source_manifests": {},
        "bridge_frozen_sources": {
            "all_existing_scientific_sources_unchanged_pass": True,
        },
        "seed_audit": {
            "requested_seed": 20,
            "verified_actual_E1_training_seed": 20,
            "legacy_C0_seed_field": 20,
            "legacy_C0_seed_field_authoritative": False,
        },
        "e1_provenance": {
            "audit_path": "synthetic/e1_audit.json",
            "audit_file_sha256": "a" * 64,
            "model_dir": "synthetic/models",
            "checkpoint_paths": ["synthetic/model.pth"],
            "checkpoint_file_sha256": ["b" * 64],
        },
        "c0_provenance": {
            "artifact_path": "synthetic/c0_predictions_and_scores.npz",
            "artifact_file_sha256": "c" * 64,
            "prediction_seal_path": "synthetic/c0_prediction_seal.json",
            "prediction_seal_file_sha256": "d" * 64,
            "q_aligned_original_logical_sha256": q_expected,
            "p_gen_original_logical_sha256": p_expected,
        },
        "sample_ids": values["arrays"]["sample_ids"],
        "sample_id_audit": {
            "exact_arange_1400": True,
            "logical_sha256": tensor_sha256(
                values["arrays"]["sample_ids"]
            ),
        },
    }


def _synthetic_recovered(values):
    return {
        "q_aligned": values["q_aligned"],
        "p_gen": values["p_gen"],
        "generator_subsets": values["generator_subsets"],
        "verifier_subsets": values["verifier_subsets"],
        "representation_audit": {},
        "inference_only": True,
    }


def _synthetic_crosscheck(values):
    confidence = values["p_gen"].max(axis=-1)
    return {
        "p_gen_y_gen_argmax_agreement": True,
        "p_gen_C_conf_max_probability_agreement": True,
        "p_gen_conf_gen_max_probability_agreement": True,
        "argmax_agreement_count": 28000,
        "argmax_agreement_total": 28000,
        "C_conf_max_abs_difference": float(
            np.max(np.abs(confidence - values["arrays"]["C_conf"]))
        ),
    }


def _patch_valid_pipeline(monkeypatch, values):
    def fake_preflight(seed, paths=None):
        return _synthetic_preflight(paths, values)

    monkeypatch.setattr(recovery, "preflight_recovery", fake_preflight)
    monkeypatch.setattr(
        recovery,
        "reconstruct_frozen_posteriors",
        lambda preflight, device="cpu": _synthetic_recovered(values),
    )
    monkeypatch.setattr(
        recovery,
        "load_and_crosscheck_pre_gt_artifact",
        lambda preflight, recovered: (
            values["arrays"],
            _synthetic_crosscheck(values),
        ),
    )


def _forbidden_call(name):
    def fail(*args, **kwargs):
        raise AssertionError(name + " must not be called")

    return fail


def test_01_no_full_gt_loader_is_called(
    monkeypatch, tmp_path, synthetic_values
):
    _patch_valid_pipeline(monkeypatch, synthetic_values)
    monkeypatch.setattr(
        c0_evaluator,
        "load_full_ground_truth_after_seal",
        _forbidden_call("C0 full-GT loader"),
    )
    monkeypatch.setattr(
        frozen_e1,
        "load_full_ground_truth",
        _forbidden_call("full-GT loader"),
    )
    result = recovery.materialize(20, output_dir=tmp_path / "no_gt")
    assert result["audit"]["GT_loaded"] is False
    assert result["audit"]["full_GT_loader_called"] is False


def test_02_no_sparse_label_loader_is_called(
    monkeypatch, tmp_path, synthetic_values
):
    _patch_valid_pipeline(monkeypatch, synthetic_values)
    monkeypatch.setattr(
        e2_train,
        "load_sparse_training_labels",
        _forbidden_call("sparse-label loader"),
    )
    monkeypatch.setattr(
        b7_protocol,
        "load_fixed_label_split",
        _forbidden_call("fixed-label loader"),
    )
    result = recovery.materialize(20, output_dir=tmp_path / "no_labels")
    assert result["audit"]["labels_loaded"] is False
    assert result["audit"]["sparse_label_loader_called"] is False


def test_03_no_corruption_mask_is_read(
    monkeypatch, tmp_path, synthetic_values
):
    _patch_valid_pipeline(monkeypatch, synthetic_values)
    monkeypatch.setattr(
        b7_protocol,
        "load_oracle_corruption_mask",
        _forbidden_call("corruption-mask loader"),
    )
    result = recovery.materialize(20, output_dir=tmp_path / "no_mask")
    assert result["audit"]["corruption_mask_loaded"] is False


def test_04_exact_q_aligned_hash_mismatch_hard_fails(
    monkeypatch, tmp_path, synthetic_values
):
    output_dir = tmp_path / "bad_q"

    def fake_preflight(seed, paths=None):
        return _synthetic_preflight(paths, synthetic_values, q_hash="0" * 64)

    monkeypatch.setattr(recovery, "preflight_recovery", fake_preflight)
    monkeypatch.setattr(
        recovery,
        "reconstruct_frozen_posteriors",
        lambda preflight, device="cpu": _synthetic_recovered(
            synthetic_values
        ),
    )
    with pytest.raises(RuntimeError, match="q_aligned exact logical SHA256"):
        recovery.materialize(20, output_dir=output_dir)
    assert not output_dir.exists()


def test_05_exact_p_gen_hash_mismatch_hard_fails(
    monkeypatch, tmp_path, synthetic_values
):
    output_dir = tmp_path / "bad_p"

    def fake_preflight(seed, paths=None):
        return _synthetic_preflight(paths, synthetic_values, p_hash="0" * 64)

    monkeypatch.setattr(recovery, "preflight_recovery", fake_preflight)
    monkeypatch.setattr(
        recovery,
        "reconstruct_frozen_posteriors",
        lambda preflight, device="cpu": _synthetic_recovered(
            synthetic_values
        ),
    )
    with pytest.raises(RuntimeError, match="p_gen exact logical SHA256"):
        recovery.materialize(20, output_dir=output_dir)
    assert not output_dir.exists()


@pytest.mark.parametrize("failed_name", ["q_aligned", "p_gen"])
def test_06_no_output_persisted_after_any_hash_mismatch(
    failed_name, tmp_path, synthetic_values
):
    paths = recovery.default_paths(20, output_dir=tmp_path / failed_name)
    kwargs = {failed_name[0] + "_hash": "f" * 64}
    if failed_name == "q_aligned":
        preflight = _synthetic_preflight(
            paths, synthetic_values, q_hash="f" * 64
        )
    else:
        preflight = _synthetic_preflight(
            paths, synthetic_values, p_hash="f" * 64
        )
    recovered = _synthetic_recovered(synthetic_values)
    with pytest.raises(RuntimeError, match="exact logical SHA256"):
        recovery.verify_exact_recovery_hashes(preflight, recovered)
    assert not paths.output_dir.exists()
    assert kwargs


def test_07_wrong_same_seed_e1_provenance_hard_fails():
    with pytest.raises(RuntimeError, match="same-seed E1 provenance"):
        recovery.resolve_authoritative_e1_training_seed(
            20,
            {"seed": 30, "training_seed": 30},
            {"seed": 20},
        )


def test_08_legacy_c0_seed_is_not_authoritative():
    audit = recovery.resolve_authoritative_e1_training_seed(
        30,
        {"seed": 30, "training_seed": 30},
        {"seed": 20},
    )
    assert audit["verified_actual_E1_training_seed"] == 30
    assert audit["legacy_C0_seed_field"] == 20
    assert audit["legacy_C0_seed_field_authoritative"] is False


def test_09_argmax_p_gen_must_equal_y_gen(synthetic_values):
    arrays = synthetic_values["arrays"]
    bad_y = np.array(arrays["y_gen"], copy=True)
    bad_y[0, 0] = (bad_y[0, 0] + 1) % recovery.CLASS_NUM
    with pytest.raises(RuntimeError, match=r"argmax\(p_gen\)"):
        recovery.crosscheck_recovered_p_gen(
            synthetic_values["p_gen"],
            {
                "y_gen": bad_y,
                "C_conf": arrays["C_conf"],
                "conf_gen": arrays["C_conf"],
            },
        )


def test_10_max_p_gen_must_match_c_conf(synthetic_values):
    arrays = synthetic_values["arrays"]
    bad_confidence = np.array(arrays["C_conf"], copy=True)
    bad_confidence[0, 0] += 0.01
    with pytest.raises(RuntimeError, match=r"max\(p_gen\)"):
        recovery.crosscheck_recovered_p_gen(
            synthetic_values["p_gen"],
            {
                "y_gen": arrays["y_gen"],
                "C_conf": bad_confidence,
                "conf_gen": bad_confidence,
            },
        )


def test_11_sample_ids_must_equal_verified_canonical_arange():
    canonical = np.arange(recovery.SAMPLE_NUM, dtype=np.int64)
    weak_hash = ndarray_sha256(canonical)
    feature_audit = {
        "sample_ids_exact_arange_pass": True,
        "sample_ids_dtype": "int64",
        "sample_ids_shape": [recovery.SAMPLE_NUM],
        "sample_ids_sha256": weak_hash,
    }
    e1_audit = {
        "feature_provenance": {
            "sample_ids_exact_arange_pass": True,
            "sample_ids_sha256": weak_hash,
        }
    }
    verified, audit = recovery.verify_canonical_sample_ids(
        canonical, feature_audit, e1_audit
    )
    assert np.array_equal(verified, canonical)
    assert not verified.flags.writeable
    assert audit["exact_arange_1400"] is True

    wrong = canonical.copy()
    wrong[[0, 1]] = wrong[[1, 0]]
    with pytest.raises(RuntimeError, match="sample-ID provenance"):
        recovery.verify_canonical_sample_ids(wrong, feature_audit, e1_audit)


def test_12_output_npz_contains_only_pre_gt_keys(
    monkeypatch, tmp_path, synthetic_values
):
    _patch_valid_pipeline(monkeypatch, synthetic_values)
    result = recovery.materialize(20, output_dir=tmp_path / "schema")
    with np.load(result["npz_path"], allow_pickle=False) as archive:
        assert tuple(archive.files) == recovery.OUTPUT_ARRAY_NAMES
        assert set(archive.files).isdisjoint(recovery.FORBIDDEN_OUTPUT_KEYS)
    assert result["audit"]["forbidden_output_keys_absent"] is True
    assert result["audit"]["global_GT_mapping_loaded"] is False
    assert result["audit"]["c0_post_GT_audit_loaded"] is False


def test_13_no_backward_optimizer_or_training_call_path_exists():
    source = inspect.getsource(recovery)
    tree = ast.parse(source)
    forbidden_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"backward", "step", "train"}
    }
    assert forbidden_attributes == set()
    assert "torch.optim" not in source
    assert "optimizer_step_called\": False" in source
    assert "training_performed\": False" in source
