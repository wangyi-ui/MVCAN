import inspect

import numpy as np

from experiments.paper.transfer_diagnostics import msrc_p0_a2_protocol as p
from experiments.paper.transfer_diagnostics import run_msrc_p0_a2 as runner
from experiments.paper.transfer_diagnostics import summarize_msrc_p0_a2 as summarizer


def test_all_arms_bind_one_shared_initialization_namespace():
    assert p.ARMS == ("BASE", "TRUE_U", "UNIFORM")
    assert p.INITIALIZATION_DIR == p.OUTPUT_ROOT / "init"
    source = inspect.getsource(runner.run_arm)
    assert "verify_initialization(init_dir)" in source
    assert "initialization[\"model_sha256\"]" in source
    summary_source = inspect.getsource(summarizer.summarize)
    assert '"checkpoint_sha256"' in summary_source
    assert '"initial_model_sha256"' in summary_source


def test_true_u_and_uniform_only_change_selected_utility():
    true_u = np.arange(210 * 20, dtype=np.float64).reshape(210, 20) / 1000.0
    selected_true = runner.select_arm_utility(true_u, "TRUE_U")
    selected_uniform = runner.select_arm_utility(true_u, "UNIFORM")
    assert np.array_equal(selected_true, true_u)
    assert np.array_equal(selected_uniform, np.ones_like(true_u))
    source = inspect.getsource(runner.build_action_artifacts)
    assert source.count("build_relation_semantics") == 1
    assert "uniform_changes_pred_relation" in source
    assert "select_arm_utility(true_u, active_arm)" in source


def test_base_reuses_authoritative_native_only_orchestration():
    assert p.ARM_SEMANTICS["BASE"] == (
        "no R4 objective; native Phase B only in every epoch"
    )
    source = inspect.getsource(runner.run_arm)
    assert "p0_runner._run_base_pre_gt" in source
    assert "run_relation_refinement_phase" not in source


def test_diagnostic_case_rules_are_frozen_without_thresholds():
    assert summarizer._diagnostic_case(
        {"BASE": 0.5, "UNIFORM": 0.6, "TRUE_U": 0.7}
    ) == "A"
    assert summarizer._diagnostic_case(
        {"BASE": 0.5, "UNIFORM": 0.7, "TRUE_U": 0.6}
    ) == "B"
    assert summarizer._diagnostic_case(
        {"BASE": 0.7, "UNIFORM": 0.6, "TRUE_U": 0.65}
    ) == "C"
