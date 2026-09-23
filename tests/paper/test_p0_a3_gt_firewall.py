import inspect

from experiments.paper.diagnostics import caltech_seed20_exact_sentinel as sentinel
from experiments.paper.diagnostics import msrc_g0b0_parity_audit as parity
from experiments.paper.diagnostics import relation_utility_quality as quality


def test_gate_a_has_no_full_gt_loader_or_evaluation():
    source = inspect.getsource(parity)
    assert "load_dataset" not in source
    assert "evaluate_postseal" not in source
    assert '"full_gt_loaded": False' in source


def test_gate_b_and_c_validate_before_first_gt_access():
    sentinel_source = inspect.getsource(sentinel.run_sentinel)
    assert sentinel_source.index("_validate_sealed_prediction") < sentinel_source.index(
        "evaluate_postseal"
    )
    msrc_source = inspect.getsource(quality.run_msrc)
    assert msrc_source.index("validate_reconstruction_against_manifest") < msrc_source.index(
        "_load_labels"
    )
    caltech_source = inspect.getsource(quality.run_caltech)
    assert caltech_source.index("_verify_files") < caltech_source.index("_load_labels")

