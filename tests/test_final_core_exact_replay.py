"""Regression tests for F0-A0 exact-replay fail-closed boundaries.

These tests never run training and never load full ground truth.
"""

import inspect
from collections import OrderedDict

import numpy as np
import pytest
import torch

from experiments.final_core import evaluate_final_core as evaluator
from experiments.final_core import final_core_protocol as f0
from experiments.final_core import train_final_core as runner


def test_preregistered_protocol_hash_mismatch_hard_fails(tmp_path):
    path = tmp_path / "PROTOCOL.txt"
    path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="PREREG"):
        f0.verify_pinned_file(
            path, f0.PREREGISTERED_PROTOCOL_SHA256,
            "F0_A0_PREREG_HASH_FAIL_CLOSED",
        )


def test_parent_hash_mismatch_hard_fails():
    summary = {
        "all_parent_hashes_pass": True,
        "frozen_C3_B0_source_hashes_pass": True,
        "frozen_C3_B0_formal_output_hashes_pass": True,
        "frozen_carrier_source_hashes_pass": True,
        "frozen_carrier_output_hashes_pass": False,
    }
    with pytest.raises(RuntimeError, match="PARENT_HASH"):
        f0.validate_parent_hash_summary(summary)


def test_wrong_sample_ids_hard_fail():
    expected = np.arange(f0.N, dtype=np.int64)
    actual = expected.copy()
    actual[-1] = actual[-2]
    with pytest.raises(RuntimeError, match="SAMPLE_IDS"):
        f0.validate_sample_ids(actual, expected)


def test_wrong_labeled_ids_hard_fail():
    expected = np.arange(f0.L, dtype=np.int64)
    actual = expected.copy()
    actual[-1] += 1
    with pytest.raises(RuntimeError, match="LABELED_IDS"):
        f0.validate_labeled_ids(actual, expected)


def test_wrong_utility_shape_hard_fail():
    utility = np.ones((f0.N, f0.S - 1), dtype=np.float64)
    with pytest.raises(RuntimeError, match="UTILITY_SHAPE"):
        f0.validate_utility(utility, f0.c3a0_parent_logical_sha256(utility))


def test_wrong_utility_logical_hash_hard_fail():
    utility = np.ones((f0.N, f0.S), dtype=np.float64)
    with pytest.raises(RuntimeError, match="UTILITY_LOGICAL_HASH"):
        f0.validate_utility(utility, "0" * 64)


def test_parent_ndarray_hash_namespace_validates_utility():
    utility = np.arange(f0.N * f0.S, dtype=np.float64).reshape(f0.N, f0.S)
    expected = f0.c3a0_parent_logical_sha256(utility)
    record = f0.validate_utility(utility, expected)
    assert record["parent_logical_sha256"] == expected
    assert record["parent_logical_sha256_equal"] is True
    assert record["utility_logical_sha256_equal"] is True


def test_distinct_internal_hash_namespace_does_not_false_fail():
    utility = np.arange(f0.N * f0.S, dtype=np.float64).reshape(f0.N, f0.S)
    parent_hash = f0.c3a0_parent_logical_sha256(utility)
    internal_hash = f0.logical_sha256(utility)
    assert internal_hash != parent_hash
    record = f0.validate_utility(utility, parent_hash)
    assert record["parent_logical_sha256"] == parent_hash
    assert record["f0_logical_sha256"] == internal_hash
    assert record["parent_logical_sha256_algorithm"] == (
        "weak_quality.ndarray_sha256"
    )


def test_changed_utility_content_fails_parent_hash_gate():
    utility = np.ones((f0.N, f0.S), dtype=np.float64)
    expected = f0.c3a0_parent_logical_sha256(utility)
    changed = utility.copy()
    changed[0, 0] = 2.0
    with pytest.raises(RuntimeError, match="UTILITY_LOGICAL_HASH"):
        f0.validate_utility(changed, expected)


def test_mapping_transpose_error_hard_fails():
    q_local = torch.arange(
        1, f0.N * f0.V * f0.K + 1, dtype=torch.float32
    ).reshape(f0.N, f0.V, f0.K)
    q_local = q_local / q_local.sum(dim=2, keepdim=True)
    permutation = torch.zeros((f0.K, f0.K), dtype=torch.float32)
    for index in range(f0.K):
        permutation[index, (index + 1) % f0.K] = 1.0
    M_v = permutation.unsqueeze(0).repeat(f0.V, 1, 1)
    wrong = torch.stack([
        q_local[:, view, :] @ M_v[view] for view in range(f0.V)
    ], dim=1).detach()
    with pytest.raises(RuntimeError, match="COORDINATE_MAPPING"):
        f0.validate_coordinate_mapping(q_local, wrong, M_v)


def _relation_inputs():
    target = np.zeros((f0.NU, f0.L, f0.S), dtype=np.bool_)
    balance = np.ones((f0.NU, f0.L, f0.S), dtype=np.float64)
    return target, balance


def test_relation_target_mismatch_hard_fails():
    target, balance = _relation_inputs()
    wrong = target.copy()
    wrong[0, 0, 0] = True
    with pytest.raises(RuntimeError, match="RELATION_TARGET"):
        f0.validate_relation_inputs(wrong, target, balance, balance)


def test_relation_balance_weights_mismatch_hard_fails():
    target, balance = _relation_inputs()
    wrong = balance.copy()
    wrong[0, 0, 0] = 2.0
    with pytest.raises(RuntimeError, match="RELATION_BALANCE_WEIGHTS"):
        f0.validate_relation_inputs(target, target, wrong, balance)


@pytest.mark.parametrize("flag", f0.FORBIDDEN_FLAGS)
def test_each_forbidden_path_hard_fails(flag):
    flags = f0.default_forbidden_flags()
    flags[flag] = True
    with pytest.raises(RuntimeError, match="FORBIDDEN_PATH"):
        f0.validate_forbidden_flags(flags)


def test_default_forbidden_paths_are_all_false():
    flags = f0.validate_forbidden_flags(f0.default_forbidden_flags())
    assert tuple(flags) == f0.FORBIDDEN_FLAGS
    assert not any(flags.values())


def test_gt_loader_is_not_called_before_a_valid_seal(monkeypatch, tmp_path):
    called = {"value": False}

    def forbidden_loader(*args, **kwargs):
        called["value"] = True
        raise AssertionError("GT loader must not be called")

    monkeypatch.setattr(
        evaluator.e1_train, "load_labels_after_predictions", forbidden_loader
    )
    with pytest.raises(RuntimeError, match="VALID_PRE_GT_SEAL_REQUIRED"):
        evaluator.evaluate(
            20,
            tmp_path / "missing.npz",
            tmp_path / "missing_audit.json",
            tmp_path / "missing_seal.json",
            tmp_path / "GT.mat",
            tmp_path / "metrics.json",
        )
    assert called["value"] is False


def test_prediction_mismatch_detection():
    ids = np.arange(f0.N, dtype=np.int64)
    expected = np.zeros(f0.N, dtype=np.int64)
    actual = expected.copy()
    actual[3] = 1
    model_hash = {"aggregate": "same", "per_view": ["x"] * f0.V}
    with pytest.raises(RuntimeError, match="REPLAY_PARITY"):
        f0.validate_replay_parity(
            model_hash, model_hash, expected, actual, ids, ids
        )


@pytest.mark.parametrize("metric", ("ACC", "NMI", "ARI"))
def test_metric_mismatch_detection(metric):
    expected = f0.frozen_metrics(20)
    actual = dict(expected)
    actual[metric] += 1e-12
    with pytest.raises(RuntimeError, match=metric + "_MISMATCH"):
        f0.validate_metric_parity(actual, expected)


@pytest.mark.parametrize("seed", f0.SEEDS)
def test_seed_specific_parent_routing(seed):
    routes = runner.parent_routes(seed)
    marker = "seed" + str(seed)
    assert routes["seed"] == seed
    assert marker in str(routes["C0_artifact"])
    assert marker in str(routes["C3_A0_action_npz"])
    assert marker in str(routes["E1_model_dir"])
    assert marker in str(routes["C3_B0_reference_predictions"])
    assert marker in str(routes["coordinate_carrier_artifact"])


def test_scientific_lineage_excludes_c4_and_c5():
    assert f0.SCIENTIFIC_LINEAGE == (
        "C0", "C3-A0", "C3-B0 TRUE_U", "F0-A0"
    )
    assert all("C4" not in stage and "C5" not in stage
               for stage in f0.SCIENTIFIC_LINEAGE)


def test_runner_delegates_exact_training_without_reimplementing_loss():
    source = inspect.getsource(runner.run_exact_replay)
    assert "replay.replay_frozen_true_u(" in source
    assert "c3_train.build_arm_optimizers(" in source
    assert ".backward(" not in source
    assert "torch.optim" not in source


def test_pre_gt_schema_contains_no_full_gt_or_metrics():
    assert "labeled_ids" in f0.PRE_GT_ARRAY_KEYS
    lowered = [name.lower() for name in f0.PRE_GT_ARRAY_KEYS]
    assert "gt" not in lowered
    assert not any(name in ("acc", "nmi", "ari", "metrics") for name in lowered)


def test_pre_gt_schema_records_coordinate_tensors_and_parent_hashes():
    assert {"q_local", "q_aligned", "M_v"}.issubset(f0.PRE_GT_ARRAY_KEYS)
    assert "frozen_input_logical_hashes" in f0.PRE_GT_ARRAY_KEYS
    assert "parent_lineage_hashes" in f0.PRE_GT_ARRAY_KEYS


def test_training_utility_is_selected_by_canonical_sample_id():
    full = np.arange(f0.N * f0.S, dtype=np.float64).reshape(f0.N, f0.S)
    labeled = np.arange(f0.L, dtype=np.int64)
    unlabeled = np.setdiff1d(np.arange(f0.N, dtype=np.int64), labeled)
    record = f0.validate_training_action_view(full, unlabeled, full[unlabeled])
    assert record["selected_by_canonical_sample_id"] is True


def test_training_utility_batch_position_substitution_fails():
    full = np.arange(f0.N * f0.S, dtype=np.float64).reshape(f0.N, f0.S)
    labeled = np.arange(f0.L, dtype=np.int64)
    unlabeled = np.setdiff1d(np.arange(f0.N, dtype=np.int64), labeled)
    wrong = full[:f0.NU]
    with pytest.raises(RuntimeError, match="SAMPLE_ID_MAPPING"):
        f0.validate_training_action_view(full, unlabeled, wrong)


def test_positive_coordinate_mapping_uses_M_transpose():
    q_local = torch.rand((f0.N, f0.V, f0.K), dtype=torch.float32)
    q_local = q_local / q_local.sum(dim=2, keepdim=True)
    M_v = torch.eye(f0.K).unsqueeze(0).repeat(f0.V, 1, 1)
    q_aligned = torch.stack([
        q_local[:, view, :] @ M_v[view].T for view in range(f0.V)
    ], dim=1).detach()
    record = f0.validate_coordinate_mapping(q_local, q_aligned, M_v)
    assert record["mapping_formula"].endswith("@ M_v[v].T")


def test_frozen_metric_constants_match_frozen_reference_files():
    for seed in f0.SEEDS:
        expected, provenance = evaluator.load_frozen_metric_reference(seed)
        assert expected == f0.FROZEN_METRICS[seed]
        assert provenance["frozen_reference_matches_audited_constants"] is True


def test_real_parent_integrity_manifests_pass():
    record = runner.verify_parent_integrity()
    assert record["all_parent_hashes_pass"] is True
    assert record["summary"]["frozen_carrier_output_hashes_pass"] is True
