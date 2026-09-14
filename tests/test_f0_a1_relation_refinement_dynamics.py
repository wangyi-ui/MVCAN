"""F0-A1 implementation tests; no test runs training or loads full GT."""

import inspect
import json
import random
from collections import OrderedDict

import numpy as np
import pytest
import torch

from experiments.cyclic_utility import c3_b0_relation_action_protocol as c3b0
from experiments.final_core import f0_a1_relation_refinement_dynamics_protocol as protocol
from experiments.final_core import run_f0_a1_relation_refinement_dynamics as runner


SAMPLE_IDS = np.arange(protocol.N, dtype=np.int64)
LABELED_IDS = np.arange(protocol.L, dtype=np.int64)
UNLABELED_IDS = np.arange(protocol.L, protocol.N, dtype=np.int64)


class TinyAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, protocol.K, bias=False)

    def clustering(self, latent):
        return torch.softmax(latent, dim=1)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoders = torch.nn.ModuleList([
            TinyAutoencoder() for _ in range(protocol.V)
        ])


@pytest.fixture(scope="module")
def snapshot_case():
    torch.manual_seed(123)
    model = TinyModel()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    views = [
        torch.arange(protocol.N * 3, dtype=torch.float32).reshape(
            protocol.N, 3
        ) / float(1000 + view)
        for view in range(protocol.V)
    ]
    q_local, audit = protocol.full_data_q_local_snapshot(
        model, views, SAMPLE_IDS, torch.device("cpu"),
        optimizers=[optimizer], return_audit=True,
    )
    return model, optimizer, views, q_local, audit


@pytest.fixture(scope="module")
def objective_case():
    base = torch.arange(
        protocol.N * protocol.V * protocol.K, dtype=torch.float64
    ).reshape(protocol.N, protocol.V, protocol.K)
    q_local = torch.softmax((base % 13) / 7.0, dim=2)
    target = np.zeros((protocol.NU, protocol.L, protocol.S), dtype=np.bool_)
    target[:, ::2, ::2] = True
    cycle = np.linspace(
        0.1, 1.0, protocol.NU * protocol.S, dtype=np.float64
    ).reshape(protocol.NU, protocol.S)
    balance = np.linspace(
        0.2, 1.2, protocol.NU * protocol.L * protocol.S,
        dtype=np.float64,
    ).reshape(protocol.NU, protocol.L, protocol.S)
    return q_local, target, cycle, balance


def valid_arrays(arm="TRUE_U", fill_by_snapshot=False):
    epochs, phases = protocol.snapshot_schedule(arm)
    count = len(epochs)
    q = np.zeros((count, protocol.N, protocol.V, protocol.K), dtype=np.float32)
    q[..., 0] = 1.0
    if fill_by_snapshot:
        for index in range(count):
            q[index].fill(index / 100.0)
    return OrderedDict((
        ("sample_ids", SAMPLE_IDS.copy()),
        ("labeled_ids", LABELED_IDS.copy()),
        ("unlabeled_ids", UNLABELED_IDS.copy()),
        ("snapshot_epoch", epochs),
        ("snapshot_phase", phases),
        ("q_local_snapshots", q),
        ("final_predictions", np.zeros(protocol.N, dtype=np.int64)),
        ("final_model_hash", np.asarray("1" * 64, dtype="<U64")),
        ("frozen_U_logical_hash", np.asarray("2" * 64, dtype="<U64")),
        ("relation_target_logical_hash", np.asarray("3" * 64, dtype="<U64")),
        ("relation_balance_weights_logical_hash", np.asarray(
            "4" * 64, dtype="<U64"
        )),
        ("arm", np.asarray(arm, dtype="<U12")),
        ("seed", np.asarray(20, dtype=np.int64)),
    ))


def minimal_audit(arrays):
    arm = str(np.asarray(arrays["arm"]).item())
    count = 21 if arm == "BASE" else 41
    audit = {
        "stage": protocol.STAGE,
        "arm": arm,
        "seed": int(np.asarray(arrays["seed"]).item()),
        "preregistered_protocol_sha256": protocol.PREREGISTERED_PROTOCOL_SHA256,
        "preregistered_protocol_hash_pass": True,
        "all_parent_hashes_pass": True,
        "sample_ids_equal": True,
        "labeled_ids_equal": True,
        "unlabeled_ids_equal": True,
        "snapshot_count_expected": count,
        "snapshot_count_actual": count,
        "snapshot_count_equal": True,
        "snapshot_shape_pass": True,
        "snapshot_order_pass": True,
        "snapshot_finite_pass": True,
        "q_local_primary_state": True,
        "q_aligned_primary_state": False,
        "extra_refresh_for_snapshot": False,
        "refresh_native_target_expected_count": 2,
        "refresh_native_target_actual_count": 2,
        "trajectory_used_for_training": False,
        "trajectory_gradient_enabled": False,
        "GT_loaded_during_trajectory": False,
        "GT_used_for_checkpoint_selection": False,
        "scientific_metric_loaded_during_runner": False,
        "full_GT_present_in_pre_gt_artifact": False,
        "final_sample_ids_equal": True,
        "final_predictions_equal": True,
        "final_model_hash_supported": True,
        "final_model_hash_equal": True,
        "model_state_changed_by_snapshot": False,
        "optimizer_state_changed_by_snapshot": False,
        "rng_state_changed_by_snapshot": False,
        "forbidden_flags": dict(protocol.default_forbidden_flags()),
        "arrays": protocol.array_records(arrays),
    }
    audit.update(protocol.default_forbidden_flags())
    return audit


def write_valid_bundle(tmp_path, arm="BASE"):
    arrays = valid_arrays(arm)
    artifact = tmp_path / "f0_a1_trajectory_pre_gt.npz"
    np.savez(artifact, **arrays)
    audit = minimal_audit(arrays)
    audit_path = tmp_path / "f0_a1_trajectory_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    seal = protocol.build_pre_gt_seal(audit, artifact, audit_path)
    seal_path = tmp_path / "f0_a1_trajectory_seal.json"
    seal_path.write_text(
        json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact, audit_path, seal_path


def test_preregistered_protocol_sha256_exact():
    record = protocol.verify_preregistered_protocol()
    assert record["actual_sha256"] == (
        "022963687054f2b520219d555d40ee248b8fe9bcefdfd77c83cfd0721c300904"
    )
    assert record["exact_match"] is True


def test_only_three_approved_arms_are_accepted():
    assert protocol.ARMS == ("BASE", "TRUE_UNIFORM", "TRUE_U")
    for forbidden in ("SHUFFLE_U", "PERMUTED_U", "ORACLE", "OTHER"):
        with pytest.raises(ValueError):
            protocol.validate_arm(forbidden)


def test_seed_set_is_exactly_20_30_50():
    assert protocol.SEEDS == (20, 30, 50)
    for forbidden in (0, 10, 40, 60):
        with pytest.raises(ValueError):
            protocol.validate_seed(forbidden)


@pytest.mark.parametrize(
    "arm,expected_count", (("BASE", 21), ("TRUE_UNIFORM", 41), ("TRUE_U", 41))
)
def test_snapshot_schedule_counts(arm, expected_count):
    epochs, phases = protocol.snapshot_schedule(arm)
    assert len(epochs) == len(phases) == expected_count


def test_base_has_no_post_a_snapshot():
    _, phases = protocol.snapshot_schedule("BASE")
    assert "POST_A" not in phases.tolist()
    assert phases.tolist() == ["INITIAL"] + ["POST_B"] * protocol.EPOCHS


def test_snapshot_epoch_phase_order_is_deterministic():
    first = protocol.snapshot_schedule("TRUE_U")
    second = protocol.snapshot_schedule("TRUE_U")
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[0].tolist()[:5] == [0, 1, 1, 2, 2]
    assert first[1].tolist()[:5] == ["INITIAL", "POST_A", "POST_B", "POST_A", "POST_B"]


def test_q_local_snapshot_exact_shape_dtype_device_and_detach(snapshot_case):
    _, _, _, q_local, _ = snapshot_case
    assert q_local.shape == (protocol.N, protocol.V, protocol.K)
    assert q_local.dtype == torch.float32
    assert q_local.device.type == "cpu"
    assert q_local.is_contiguous()
    assert q_local.requires_grad is False and q_local.grad_fn is None


def test_q_local_snapshot_does_not_alter_model_hash(snapshot_case):
    _, _, _, _, audit = snapshot_case
    assert audit["model_parameters_unchanged"] is True
    assert audit["model_buffers_unchanged"] is True
    assert audit["model_aggregate_hash_equal"] is True
    assert audit["model_hash_before"] == audit["model_hash_after"]


def test_q_local_snapshot_does_not_alter_training_mode(snapshot_case):
    model, _, _, _, audit = snapshot_case
    assert model.training is True
    assert all(module.training for module in model.modules())
    assert audit["model_training_flags_unchanged"] is True
    assert audit["model_mode_changed_for_snapshot"] is False


def test_q_local_snapshot_does_not_alter_optimizer(snapshot_case):
    _, _, _, _, audit = snapshot_case
    assert audit["optimizer_state_unchanged"] is True


def test_q_local_snapshot_does_not_alter_rng_state(snapshot_case):
    _, _, _, _, audit = snapshot_case
    assert audit["rng_state_unchanged"] is True
    assert audit["RNG_restored_after_snapshot"] is True


def test_snapshot_helper_never_calls_native_refresh(monkeypatch, snapshot_case):
    model, optimizer, views, _, _ = snapshot_case
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("snapshot called native refresh")

    monkeypatch.setattr(runner.e1_train, "refresh_native_target", forbidden)
    protocol.full_data_q_local_snapshot(
        model, views, SAMPLE_IDS, torch.device("cpu"), optimizers=[optimizer]
    )
    assert calls == []


def test_relation_snapshot_shape_and_frozen_formula(objective_case):
    q_local = objective_case[0]
    relation = protocol.relation_state_from_q_local(
        q_local, SAMPLE_IDS, UNLABELED_IDS, LABELED_IDS
    )
    assert relation.shape == (protocol.NU, protocol.L, protocol.V)
    expected = q_local[UNLABELED_IDS, 2, :] @ q_local[LABELED_IDS, 2, :].T
    assert torch.equal(relation[:, :, 2], expected)
    assert relation.requires_grad is False


def test_relation_probability_wrapper_numerically_equals_frozen_helper():
    sample = torch.softmax(torch.arange(21, dtype=torch.float64).reshape(3, 7), 1)
    anchor = torch.softmax(torch.arange(98, dtype=torch.float64).reshape(14, 7), 1)
    actual = c3b0.relation_probability(sample, anchor)
    assert torch.equal(actual, sample @ anchor.detach().T)


@pytest.mark.parametrize("arm", ("TRUE_U", "TRUE_UNIFORM"))
def test_diagnostic_objective_equals_frozen_c3_b0(arm, objective_case):
    q_local, target, cycle, balance = objective_case
    actual, audit = protocol.diagnostic_relation_objective(
        q_local, SAMPLE_IDS, UNLABELED_IDS, LABELED_IDS,
        target, cycle, balance, arm,
    )
    query = torch.from_numpy(UNLABELED_IDS)
    anchor = torch.from_numpy(LABELED_IDS)
    q_samples = [q_local[query, view, :] for view in range(protocol.V)]
    q_anchors = [q_local[anchor, view, :] for view in range(protocol.V)]
    expected, expected_audit = c3b0.relation_semantic_loss(
        q_samples, q_anchors, target, cycle, balance, arm
    )
    assert actual == expected.item()
    assert audit["denominator"] == expected_audit["denominator"]
    assert audit["six_view_arithmetic_mean"] is True


def test_anchor_side_is_detached_but_query_side_retains_gradient():
    sample = torch.softmax(torch.randn(2, 7, dtype=torch.float64), 1)
    anchor = torch.softmax(torch.randn(14, 7, dtype=torch.float64), 1)
    sample.requires_grad_(True)
    anchor.requires_grad_(True)
    c3b0.relation_probability(sample, anchor).sum().backward()
    assert sample.grad is not None
    assert anchor.grad is None


def fake_schedule(monkeypatch, arm):
    calls = {"refresh": 0, "semantic": 0, "native": 0, "snapshot": 0}

    def snapshot(*args, **kwargs):
        calls["snapshot"] += 1
        q = torch.zeros((protocol.N, protocol.V, protocol.K), dtype=torch.float32)
        q[..., 0] = 1.0
        audit = {
            "model_parameters_unchanged": True,
            "model_buffers_unchanged": True,
            "model_aggregate_hash_equal": True,
            "model_training_flags_unchanged": True,
            "optimizer_state_unchanged": True,
            "rng_state_unchanged": True,
        }
        return q, audit

    def semantic(*args, **kwargs):
        calls["semantic"] += 1
        return {"executed": True}

    def native(*args, **kwargs):
        calls["native"] += 1
        return {"executed": True}

    def refresh(*args, **kwargs):
        calls["refresh"] += 1
        return (
            torch.zeros((protocol.N, protocol.K)),
            torch.zeros((protocol.V, protocol.K, protocol.K)),
            np.zeros(protocol.N, dtype=np.int64),
            [1.0] * protocol.V,
        )

    monkeypatch.setattr(runner.c3_train, "relation_semantic_phase", semantic)
    monkeypatch.setattr(runner.c3_train, "native_consolidation_phase", native)
    monkeypatch.setattr(runner.e1_train, "refresh_native_target", refresh)
    semantic_optimizers = None if arm == "BASE" else [object()] * protocol.V
    result = runner.run_instrumented_schedule(
        arm=arm,
        model=object(),
        semantic_optimizers=semantic_optimizers,
        native_optimizers=[object()] * protocol.V,
        full_views=[torch.zeros((protocol.N, 1)) for _ in range(protocol.V)],
        sample_ids=SAMPLE_IDS,
        action_arrays=None,
        orders={
            "semantic_orders": [None] * protocol.EPOCHS,
            "native_orders": [None] * protocol.EPOCHS,
        },
        device=torch.device("cpu"),
        training_seed=20,
        snapshot_fn=snapshot,
    )
    return result, calls


@pytest.mark.parametrize(
    "arm,expected_snapshots,expected_semantic",
    (("BASE", 21, 0), ("TRUE_UNIFORM", 41, 20), ("TRUE_U", 41, 20)),
)
def test_instrumented_20_epoch_schedule_counts(
    monkeypatch, arm, expected_snapshots, expected_semantic,
):
    result, calls = fake_schedule(monkeypatch, arm)
    assert result["q_local_snapshots"].shape[0] == expected_snapshots
    assert calls == {
        "refresh": 2,
        "semantic": expected_semantic,
        "native": 20,
        "snapshot": expected_snapshots,
    }
    assert result["runtime"]["refresh_native_target_total_count"] == 2
    assert result["runtime"]["extra_refresh_for_snapshot"] is False


def test_phase_a_and_phase_b_drift_formulas(objective_case):
    q0 = objective_case[0].to(torch.float32)
    qa = q0.clone()
    qb = q0.clone()
    qa[protocol.L:, :, 0] += 0.01
    qb[protocol.L:, :, 1] += 0.02
    r0 = protocol.relation_state_from_q_local(
        q0, SAMPLE_IDS, UNLABELED_IDS, LABELED_IDS
    )
    ra = protocol.relation_state_from_q_local(
        qa, SAMPLE_IDS, UNLABELED_IDS, LABELED_IDS
    )
    rb = protocol.relation_state_from_q_local(
        qb, SAMPLE_IDS, UNLABELED_IDS, LABELED_IDS
    )
    d_a = protocol.relation_drift(
        q0, qa, SAMPLE_IDS, UNLABELED_IDS, LABELED_IDS
    )
    d_b = protocol.relation_drift(
        qa, qb, SAMPLE_IDS, UNLABELED_IDS, LABELED_IDS
    )
    assert d_a == torch.mean(torch.abs(ra - r0)).item()
    assert d_b == torch.mean(torch.abs(rb - ra)).item()


def test_semantic_purification_response_formula(monkeypatch):
    arrays = valid_arrays("TRUE_U", fill_by_snapshot=True)
    monkeypatch.setattr(protocol, "relation_drift", lambda *args: 0.0)
    monkeypatch.setattr(protocol, "posterior_drift", lambda *args: 0.0)
    monkeypatch.setattr(
        protocol,
        "diagnostic_relation_objective",
        lambda q, *args: (-float(np.asarray(q).mean()), {}),
    )
    result = protocol.compute_trajectory_diagnostics(
        arrays,
        np.zeros((protocol.NU, protocol.L, protocol.S), dtype=np.bool_),
        np.ones((protocol.NU, protocol.S), dtype=np.float64),
        np.ones((protocol.NU, protocol.L, protocol.S), dtype=np.float64),
    )
    expected = [0.01] * protocol.EPOCHS
    assert result["G"] == pytest.approx(expected, abs=2e-7)
    assert result["G_sum"] == pytest.approx(sum(expected), abs=2e-6)


def test_gate3_does_not_require_every_epoch_positive():
    epoch_gains = np.array([0.2, -0.1] + [0.01] * 18)
    assert epoch_gains.sum() > 0 and np.any(epoch_gains < 0)
    summary = protocol.summarize_gate3({20: epoch_gains.sum(), 30: -0.01, 50: 0.2})
    assert summary["gate3_pass"] is True
    assert summary["positive_seed_count"] == 2
    assert summary["all_epochs_positive_required"] is False


def test_u_vs_uniform_divergence_formula_and_exact_q0(monkeypatch):
    true_u = valid_arrays("TRUE_U", fill_by_snapshot=True)
    uniform = valid_arrays("TRUE_UNIFORM", fill_by_snapshot=False)
    true_u["q_local_snapshots"][0] = uniform["q_local_snapshots"][0]
    uniform["q_local_snapshots"][1:] = 0.0

    def scalar_relation(q, *args):
        return torch.tensor([[[float(np.asarray(q).mean())]]])

    monkeypatch.setattr(protocol, "relation_state_from_q_local", scalar_relation)
    result = protocol.compare_true_u_and_uniform_trajectories(true_u, uniform)
    assert result["D_U_UNI_0"] == 0.0
    assert result["D_U_UNI_A"][0] == pytest.approx(0.01)
    assert result["D_U_UNI_B"][-1] == pytest.approx(0.40)
    assert result["Q0_exact_equal"] is True


def test_u_vs_uniform_comparison_requires_exact_ids():
    true_u = valid_arrays("TRUE_U")
    uniform = valid_arrays("TRUE_UNIFORM")
    uniform["sample_ids"] = uniform["sample_ids"].copy()
    uniform["sample_ids"][[0, 1]] = uniform["sample_ids"][[1, 0]]
    with pytest.raises(RuntimeError):
        protocol.compare_true_u_and_uniform_trajectories(true_u, uniform)


def test_u_vs_uniform_comparison_requires_exact_q0():
    true_u = valid_arrays("TRUE_U")
    uniform = valid_arrays("TRUE_UNIFORM")
    uniform["q_local_snapshots"][0, 0, 0, :] = 1.0 / protocol.K
    with pytest.raises(RuntimeError, match="Q0"):
        protocol.compare_true_u_and_uniform_trajectories(true_u, uniform)


@pytest.mark.parametrize("arm", protocol.ARMS)
@pytest.mark.parametrize("seed", protocol.SEEDS)
def test_historical_reference_loader_supports_three_arms_by_three_seeds(seed, arm):
    reference = protocol.load_frozen_c3_b0_reference(seed, arm)
    assert reference["seed"] == seed and reference["arm"] == arm
    assert reference["predictions"].shape == (protocol.N,)
    assert np.array_equal(reference["sample_ids"], SAMPLE_IDS)
    assert reference["canonical_freeze_root"] == str(protocol.C3_FINAL_FREEZE)
    assert "outputs/" not in reference["prediction_path"].replace(
        "formal_outputs/", ""
    )


def test_artifact_overwrite_hard_fails_before_parent_loading(monkeypatch, tmp_path):
    target = tmp_path / "exists"
    target.mkdir()

    def forbidden():
        raise AssertionError("parent loading happened before overwrite refusal")

    monkeypatch.setattr(runner, "verify_parent_integrity", forbidden)
    with pytest.raises(RuntimeError, match="overwrite"):
        runner.run_trajectory(20, "TRUE_U", "cpu", target)


def test_pre_gt_seal_detects_modified_artifact(tmp_path):
    artifact, audit, seal = write_valid_bundle(tmp_path)
    protocol.validate_pre_gt_seal(artifact, audit, seal)
    with open(artifact, "ab") as output_file:
        output_file.write(b"modified")
    with pytest.raises(RuntimeError, match="SEAL"):
        protocol.validate_pre_gt_seal(artifact, audit, seal)


def test_pre_gt_seal_detects_modified_audit(tmp_path):
    artifact, audit, seal = write_valid_bundle(tmp_path)
    protocol.validate_pre_gt_seal(artifact, audit, seal)
    with open(audit, "a", encoding="utf-8") as output_file:
        output_file.write(" \n")
    with pytest.raises(RuntimeError, match="SEAL"):
        protocol.validate_pre_gt_seal(artifact, audit, seal)


def test_no_full_gt_or_scientific_metric_key_in_artifact():
    lowered = [name.lower() for name in protocol.ARTIFACT_KEYS]
    assert not any("gt" in name for name in lowered)
    assert not any(name in ("acc", "nmi", "ari", "metrics") for name in lowered)
    assert "q_local_snapshots" in protocol.ARTIFACT_KEYS
    assert "q_aligned" not in protocol.ARTIFACT_KEYS


def test_all_forbidden_flags_are_false_and_fail_closed():
    flags = protocol.validate_forbidden_flags(protocol.default_forbidden_flags())
    assert tuple(flags) == protocol.FORBIDDEN_FLAGS
    assert not any(flags.values())
    changed = dict(flags)
    changed[protocol.FORBIDDEN_FLAGS[0]] = True
    with pytest.raises(RuntimeError, match="FORBIDDEN"):
        protocol.validate_forbidden_flags(changed)


def test_runner_cli_has_no_full_gt_path_and_no_epoch_control():
    parser_source = inspect.getsource(runner.parse_args)
    assert "full-gt-path" not in parser_source.lower()
    assert "--epochs" not in parser_source
    args = runner.parse_args([
        "--seed", "20", "--arm", "TRUE_U", "--device", "cpu",
        "--output-dir", "unused",
    ])
    assert vars(args) == {
        "seed": 20, "arm": "TRUE_U", "device": "cpu", "output_dir": "unused"
    }


def test_runner_has_no_gt_loader_or_metric_evaluation_invocation():
    source = inspect.getsource(runner)
    forbidden_fragments = (
        "load_" + "labels_after_predictions",
        "evaluate_" + "predictions",
        "full_" + "gt_path",
    )
    assert all(fragment not in source.lower() for fragment in forbidden_fragments)


def test_runner_reuses_frozen_training_and_provenance_helpers():
    source = inspect.getsource(runner)
    for call in (
        "c3_train.build_arm_optimizers(",
        "c3_train.relation_semantic_phase(",
        "c3_train.native_consolidation_phase(",
        "c3_train.load_frozen_c3a0_action_bundle(",
        "e1_train.load_frozen_feature_artifact(",
        "vsa_train.load_lineage_aware_e1_lwc_model(",
        "vsa.precompute_epoch_orders(",
        "hash_backbone(",
        "e1_train.refresh_native_target(",
    ):
        assert call in source


def test_snapshot_rng_restoration_is_real_not_audit_only():
    torch.manual_seed(77)
    np.random.seed(77)
    random.seed(77)
    model = TinyModel()
    views = [torch.zeros((protocol.N, 3)) for _ in range(protocol.V)]
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    protocol.full_data_q_local_snapshot(
        model, views, SAMPLE_IDS, torch.device("cpu")
    )
    assert torch.equal(torch_before, torch.get_rng_state())
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert python_before == random.getstate()
