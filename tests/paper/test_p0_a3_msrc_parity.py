import numpy as np

from experiments.paper.diagnostics import msrc_g0b0_parity_audit as audit


def _comparison(equal=True, close=True):
    return {"array_equal": equal, "allclose_1e-8": close}


def test_comparison_reports_required_fields_and_exact_integer_policy():
    result = audit.compare_arrays(
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
    )
    assert result["array_equal"] is True
    assert result["comparison_policy"] == "exact"
    assert result["allclose_1e-12"] is None
    assert set(result) == {
        "shape", "dtype", "logical_sha256", "array_equal",
        "comparison_policy", "allclose_1e-12", "allclose_1e-8",
        "max_abs_diff",
    }


def test_preregistered_parity_decision_order():
    names = (
        "generator_membership", "verifier_membership", "final_predictions",
        "U_cycle", "PredRelation_true", "relation_balance_weights_true",
        "final_q_local", "final_q_aligned", "final_M_v", "q_aligned", "y_gen",
    )
    comparisons = {name: _comparison() for name in names}
    comparisons["generator_membership"] = _comparison(False)
    assert audit.classify(comparisons).startswith("FAIL_CLOSED_ACTION")
    comparisons["generator_membership"] = _comparison()
    assert audit.classify(comparisons) == "PASS_HISTORICAL_G0_B0_BEHAVIOR_PARITY"

