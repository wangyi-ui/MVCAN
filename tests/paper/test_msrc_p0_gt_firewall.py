import inspect

from experiments.paper.transfer_audit import evaluate_msrc_p0_transfer as evaluator
from experiments.paper.transfer_audit import input_artifacts
from experiments.paper.transfer_audit import run_msrc_p0_transfer as runner


def test_training_runner_has_no_dataset_or_gt_argument():
    signature = inspect.signature(runner.run_arm)
    assert "dataset_path" not in signature.parameters
    source = inspect.getsource(runner)
    assert "load_dataset" not in source
    assert "load_msrc_v1" not in source
    assert "evaluate_postseal" not in source


def test_gt_free_input_loader_has_no_dataset_loader_dependency():
    source = inspect.getsource(input_artifacts)
    assert "load_dataset" not in source
    assert "load_msrc_v1" not in source
    assert "scipy" not in source


def test_evaluator_uses_release_seal_first_boundary():
    source = inspect.getsource(evaluator)
    assert "evaluate_postseal" in source
    assert "run_pre_gt" not in source
    assert "optimizer" not in source.lower()
    assert "torch" not in source


def test_required_arm_outputs_are_named_exactly():
    source = inspect.getsource(runner)
    for name in ("pre_gt_bundle.npz", "pre_gt_audit.json", "pre_gt_seal.json"):
        # Names are owned by the release runtime and represented by SealedPredictionPaths.
        assert name in inspect.getsource(runner.runtime_entrypoint._persist_pre_gt)
    assert "run_manifest.json" in source
    assert "GT-loaded-before-seal" in source
