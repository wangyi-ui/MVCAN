import inspect

import numpy as np
import pytest

from experiments.paper.transfer_audit import msrc_p0_protocol as p
from experiments.paper.transfer_audit import run_msrc_p0_transfer as runner


def test_arm_identities_are_exact_and_not_aliased():
    assert p.ARMS == ("BASE", "TRUE_U", "UNIFORM")
    assert len(set(p.ARM_DIRECTORY_NAMES.values())) == 3
    assert p.ARM_DIRECTORY_NAMES == {
        "BASE": "base", "TRUE_U": "true_u", "UNIFORM": "uniform"
    }
    with pytest.raises(ValueError):
        p.validate_arm("TRUE_UNIFORM")


def test_uniform_replaces_only_u_carrier_and_does_not_mutate_true_u():
    true_u = np.arange(210 * 20, dtype=np.float64).reshape(210, 20) / 10000.0
    before = true_u.copy()
    uniform = runner.select_arm_utility(true_u, "UNIFORM")
    selected_true = runner.select_arm_utility(true_u, "TRUE_U")
    assert np.array_equal(true_u, before)
    assert np.array_equal(uniform, np.ones_like(true_u))
    assert np.array_equal(selected_true, true_u)
    assert not np.shares_memory(uniform, true_u)


def test_uniform_contract_retains_true_predrelation_and_balance():
    assert p.ARM_SEMANTICS["UNIFORM"] == (
        "ones_like(U_cycle) + true PredRelation + true balance"
    )
    source = inspect.getsource(runner.build_action_artifacts)
    assert "pred_relation =" in source
    assert "balance =" in source
    assert "select_arm_utility" in source
    assert "uniform_changes_pred_relation" in source


def test_base_is_historical_native_only_control():
    assert p.ARM_SEMANTICS["BASE"] == (
        "no R4 objective; native Phase B only in every epoch"
    )
    source = inspect.getsource(runner._run_base_pre_gt)
    assert "run_native_consolidation_phase" in source
    assert "PHASE_A_SKIPPED" in source
    assert "run_relation_refinement_phase" not in source
    assert "run_alternating_training" not in source


def test_output_overwrite_fails_before_materialized_input_access(tmp_path):
    output = tmp_path / "arm"
    output.mkdir()
    with pytest.raises(RuntimeError, match="overwrite"):
        runner.run_arm(
            arm="TRUE_U", input_dir=tmp_path / "missing", output_dir=output,
            device="cpu", training_seed=20, epochs=20, batch_size=256,
            learning_rate=1e-4, refresh_interval=100,
        )
