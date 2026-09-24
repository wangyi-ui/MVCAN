import inspect
from pathlib import Path

import pytest

from experiments.paper.formal import p1_a0_formal_protocol as protocol
from experiments.paper.formal import p1_a1_action_materialization as actions
from experiments.paper.formal import p1_a1_base_runtime as base
from experiments.paper.formal import p1_a1_runner as runner


def make_run(dataset="Caltech-6V", seed=20, arm="BASE"):
    return runner.FormalRun(dataset, seed, arm, "cpu")


def test_protocol_is_the_only_resolved_scientific_source():
    assert protocol.validate_formal_protocol() is True
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "epochs=20" not in source
    assert "--lambda1" not in source
    assert runner.runtime_config(make_run()).epochs == protocol.ALTERNATING_PROTOCOL.epochs


def test_cli_exposes_only_authorized_selectors_and_plan_flag():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    for forbidden in ("--epochs", "--lr", "--lambda1", "--batch-size", "--refresh-interval", "--snr", "--label-seed", "--labels-per-class"):
        assert forbidden not in source
    args = runner.parse_args(["--dataset", "BDGP", "--training-seed", "50", "--arm", "OURS_TRUE_U", "--device", "cpu", "--plan-only"])
    assert args.plan_only is True


def test_only_frozen_dataset_seed_and_arm_names_are_accepted():
    assert runner.formal_arms() == ("BASE", "OURS_TRUE_U", "TRUE_UNIFORM", "SHUFFLE_U")
    for run in (make_run("Caltech-6V", 20), make_run("MSRC-v1", 30), make_run("BDGP", 50)):
        assert runner.validate_request(run) is run
    with pytest.raises(RuntimeError, match="SEED"):
        runner.validate_request(make_run(seed=21))


def test_native_config_and_training_seed_are_separate_and_plan_is_pure(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path / "formal")
    before = list(tmp_path.rglob("*"))
    caltech20 = runner.plan(make_run("Caltech-6V", 20))
    caltech30 = runner.plan(make_run("Caltech-6V", 30))
    assert caltech20["native_config_seed"] == 5
    assert caltech20["training_seed"] == 20
    assert caltech30["kmeans_seed"] == caltech30["order_seed"] == 30
    assert caltech20["weak_quality_seed"] == caltech30["weak_quality_seed"] == 20
    assert caltech20["paths"]["features"] == caltech30["paths"]["features"]
    assert caltech20["paths"]["split"] == caltech30["paths"]["split"]
    assert caltech20["paths"]["initialization"] != caltech30["paths"]["initialization"]
    assert list(tmp_path.rglob("*")) == before


def test_same_seed_arm_paths_share_initialization_and_true_action_state():
    base_plan = runner.plan(make_run(arm="BASE"))
    ours_plan = runner.plan(make_run(arm="OURS_TRUE_U"))
    uniform_plan = runner.plan(make_run(arm="TRUE_UNIFORM"))
    assert base_plan["paths"]["initialization"] == ours_plan["paths"]["initialization"]
    assert ours_plan["paths"]["action"] == uniform_plan["paths"]["action"]
    assert base.base_audit_contract()["phase_a_executed"] is False
    assert base.base_audit_contract()["semantic_optimizer_created"] is False


def test_action_seed_namespace_and_undeclared_shuffle_semantics_fail_closed(tmp_path):
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        actions.verify_true_action(tmp_path / "seed20", dataset="Caltech-6V", training_seed=30, initial_model_sha256="x")
    with pytest.raises(RuntimeError, match="SEMANTICS_UNRESOLVED"):
        actions.select_arm_utility({"artifact": tmp_path / "a"}, "SHUFFLE_U", protocol.ARM_PROTOCOL["ablation"])


def test_runtime_injects_frozen_r5_schedule_and_final_prediction_contract():
    runtime = runner.runtime_config(make_run("BDGP", 50, "OURS_TRUE_U"))
    assert (runtime.epochs, runtime.refresh_interval, runtime.batch_size, runtime.learning_rate) == (20, 100, 256, 1e-4)
    assert runtime.native_lambda1 == 10.0
    assert protocol.ALTERNATING_PROTOCOL.phase_order == ("Phase A", "target refresh if zero-based epoch % T2 == 0", "Phase B")
    assert protocol.EVALUATION_PROTOCOL["q_argmax_is_final_prediction"] is False


def test_missing_inputs_fail_before_training_and_write_no_success_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path / "formal")
    monkeypatch.setattr(runner, "run_pre_gt", lambda *_: pytest.fail("training called"))
    run = make_run(arm="OURS_TRUE_U")
    with pytest.raises(RuntimeError, match="FORMAL_INPUT_NOT_MATERIALIZED"):
        runner.run_formal(run)
    output = Path(runner.paths_for(run)["output"])
    assert not output.exists()
    assert Path(str(output) + ".failed.json").is_file()
    assert not (output / "pre_gt_seal.json").exists()


def test_runner_has_no_gt_or_metric_api_and_protected_sources_are_unchanged():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "evaluate_postseal" not in source and "full_gt_path" not in source
    assert "release_core.evaluation" not in source and "from release_core.runtime import evaluate_postseal" not in source
    root = Path(__file__).resolve().parents[2]
    assert not __import__("subprocess").run(["git", "diff", "--name-only", "--", "release_core", "experiments/paper/formal/p1_a0_formal_protocol.py"], cwd=root, text=True, stdout=__import__("subprocess").PIPE, check=True).stdout
