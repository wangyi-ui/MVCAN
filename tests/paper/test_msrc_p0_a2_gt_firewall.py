import inspect

from experiments.paper.transfer_audit.input_artifacts import file_sha256
from experiments.paper.transfer_diagnostics import evaluate_msrc_p0_a2 as evaluator
from experiments.paper.transfer_diagnostics import (
    materialize_msrc_current_condition_init as initializer,
)
from experiments.paper.transfer_diagnostics import msrc_p0_a2_protocol as p
from experiments.paper.transfer_diagnostics import run_msrc_p0_a2 as runner


def test_p0_a1_diagnostic_outputs_remain_byte_exact():
    paths = {
        "summary": p.P0_A1_ROOT / "seed20_three_arm_summary.json",
    }
    for arm in ("base", "true_u", "uniform"):
        paths[arm + "_manifest"] = p.P0_A1_ROOT / arm / "run_manifest.json"
        paths[arm + "_audit"] = p.P0_A1_ROOT / arm / "pre_gt_audit.json"
        paths[arm + "_seal"] = p.P0_A1_ROOT / arm / "pre_gt_seal.json"
        paths[arm + "_metrics"] = p.P0_A1_ROOT / arm / "metrics.json"
    assert {name: file_sha256(path) for name, path in paths.items()} == p.P0_A1_RESULT_HASHES


def test_initializer_and_pre_gt_runner_have_no_full_gt_boundary():
    init_signature = inspect.signature(initializer.materialize)
    run_signature = inspect.signature(runner.run_arm)
    assert "dataset_path" not in init_signature.parameters
    assert "dataset_path" not in run_signature.parameters
    init_source = inspect.getsource(initializer)
    run_source = inspect.getsource(runner)
    assert "evaluate_postseal" not in init_source
    assert "evaluate_postseal" not in run_source
    assert "load_dataset" not in init_source
    assert "load_dataset" not in run_source


def test_evaluator_is_the_only_postseal_gt_boundary():
    source = inspect.getsource(evaluator)
    assert "evaluate_postseal" in source
    assert "run_pre_gt" not in source
    assert "optimizer" not in source.lower()
    assert "torch" not in source


def test_output_roots_cannot_alias_preserved_p0_a1():
    assert p.OUTPUT_ROOT.resolve() != p.P0_A1_ROOT.resolve()
    for arm in p.ARMS:
        assert p.default_output_dir(arm).parent == p.OUTPUT_ROOT
