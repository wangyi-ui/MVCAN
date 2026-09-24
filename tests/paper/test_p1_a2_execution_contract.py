from pathlib import Path
import subprocess

import numpy as np
import pytest

from experiments.paper.formal import audit_p1_a2_execution_contract as audit
from experiments.paper.formal import p1_a0_formal_protocol as p1_a0
from experiments.paper.formal import p1_a1_action_materialization as actions
from experiments.paper.formal import p1_a1_runner as runner
from experiments.paper.formal import p1_a2_execution_contract as contract


def test_generator_is_single_cpu_training_seed_stream_without_reset():
    values = contract.NATIVE_DATALOADER_GENERATOR_CONTRACT
    assert values == {"generator_device": "cpu", "seed_source": "formal training_seed",
                      "created_once_per_native_preparation": True,
                      "shared_across_ae_and_native_loops": True,
                      "reset_between_stages": False, "reset_each_epoch": False}
    assert {item.native_config_seed for item in p1_a0.FORMAL_DATASETS} == {1, 5, 20}
    assert p1_a0.FORMAL_TRAINING_SEEDS == (20, 30, 50)


def test_retained_carrier_aligns_post_final_q_without_any_refresh():
    q = np.array([[[.8, .2], [.1, .9]]], dtype=np.float64)
    retained = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=np.float64)
    aligned = contract.align_with_retained_matrix(q, retained)
    assert np.array_equal(aligned, np.array([[[.2, .8], [.1, .9]]]))
    source = Path(contract.__file__).read_text(encoding="utf-8")
    assert "native_refresh" not in source
    assert contract.ACTION_CARRIER_CONTRACT["fresh_post_final_refresh"] is False
    assert contract.ACTION_CARRIER_CONTRACT["reset_incoming_view_weights_to_ones"] is False


def test_caltech_historical_shuffle_targets_preserve_histogram_and_are_not_global_permutation():
    original = np.repeat(np.arange(7), 2)
    shuffled = np.asarray(contract.CALTECH_FIXED_SHUFFLED_TARGETS)
    assert np.array_equal(np.bincount(shuffled, minlength=7), np.full(7, 2))
    assert any(np.unique(shuffled[original == label]).size > 1 for label in range(7))


def test_shuffle_keeps_true_u_and_changes_relation_not_utility():
    selected = actions.select_arm_utility({"artifact": Path("fake.npz")}, "SHUFFLE_U",
                                          p1_a0.ARM_PROTOCOL["ablation"], dataset="Caltech-6V")
    assert selected["utility"] == "true U_cycle"
    assert selected["kind"] == "shuffle_relation"
    with pytest.raises(RuntimeError, match="UNRESOLVED_FOR_MSRC"):
        actions.select_arm_utility({"artifact": Path("fake.npz")}, "SHUFFLE_U",
                                   p1_a0.ARM_PROTOCOL["ablation"], dataset="MSRC-v1")


def test_arm_scoped_capability_blocks_only_cross_dataset_shuffle(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path / "formal")
    main = runner.FormalRun("MSRC-v1", 30, "OURS_TRUE_U", "cpu")
    shuffle = runner.FormalRun("MSRC-v1", 30, "SHUFFLE_U", "cpu")
    assert runner.plan(main)["execution_capability"] == "AUTHORIZED"
    blocked = runner.plan(shuffle)
    assert blocked["execution_capability"] == "BLOCKED"
    assert blocked["blocked_reason"] == "SHUFFLE_SEMANTICS_UNRESOLVED_FOR_MSRC"
    assert not list(tmp_path.rglob("*"))


def test_uniform_preserves_true_relation_contract_and_main_preflight_is_complete(tmp_path):
    assert contract.TRUE_UNIFORM_CONTRACT["utility"] == "ones_like(true U_cycle)"
    assert "unchanged" in contract.TRUE_UNIFORM_CONTRACT["pred_relation"]
    report = audit.build_report()
    assert report["cell_count"] == 36
    assert report["main_authorized_count"] == 18
    assert report["main_decision"] == "FORMAL_MAIN_RUNNER_READY"
    assert report["ablation_decision"] == "FORMAL_ABLATION_PARTIALLY_BLOCKED"
    target = audit.write_report(tmp_path / "preflight.json")
    assert target.is_file()


def test_execution_contract_and_protected_sources_are_unchanged():
    assert contract.validate_execution_contract() is True
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["git", "diff", "--name-only", "--", "release_core", "experiments/paper/formal/p1_a0_formal_protocol.py"], cwd=root, check=True, text=True, stdout=subprocess.PIPE)
    assert result.stdout == ""
    historical = subprocess.run(["git", "-C", "/root/autodl-tmp/CVPR24-MVCAN", "status", "--short"], check=True, text=True, stdout=subprocess.PIPE)
    assert historical.stdout == ""
