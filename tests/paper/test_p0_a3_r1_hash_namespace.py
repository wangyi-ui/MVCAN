import numpy as np

from experiments.paper.diagnostics import caltech_seed20_exact_sentinel as sentinel
from experiments.paper.diagnostics import p0_a3_protocol as protocol
from release_core.data.weak_quality import ndarray_sha256


def test_historical_reference_prediction_uses_canonical_payload_namespace():
    with np.load(protocol.CALTECH_REFERENCE_ARTIFACT, allow_pickle=False) as archive:
        prediction = np.ascontiguousarray(archive["final_predictions"])
    canonical = sentinel.canonical_prediction_payload_sha256(prediction)
    diagnostic = ndarray_sha256(prediction)
    assert canonical == protocol.CALTECH_EXPECTED_PREDICTION
    assert diagnostic != protocol.CALTECH_EXPECTED_PREDICTION


def test_failed_wrapper_output_was_value_exact_before_wrong_hash_gate():
    bundle = protocol.OUTPUT_ROOT / (
        "caltech_seed20_exact_sentinel/pre_gt_bundle.npz"
    )
    assert bundle.is_file()
    sealed = type("Paths", (), {"bundle": bundle})()
    comparisons = sentinel.compare_payload(sealed)
    assert all(comparisons[name]["array_equal"] for name in (
        "sample_ids", "final_predictions", "q_local", "q_aligned", "M_v"
    ))
    hashes = comparisons["prediction_hashes"]
    assert hashes["canonical_prediction_payload_sha256"] == (
        protocol.CALTECH_EXPECTED_PREDICTION
    )
    assert hashes["ndarray_sha256_diagnostic_only"] != (
        protocol.CALTECH_EXPECTED_PREDICTION
    )

