from experiments.paper.diagnostics.msrc_pre_r2_carrier_temporal_alignment import (
    classify_carrier_alignment,
)


BASE = dict(historical_exact=True, current_exact=True,
            post_model_and_q_local_exact=True, t_equals_h=True,
            w_equals_h=False, f_equals_w=False, h_equals_f=False)


def test_exact_parity_rule():
    assert classify_carrier_alignment(**{**BASE, "h_equals_f": True}) == (
        "PRE_R2_CARRIER_EXACT_PARITY"
    )


def test_view_weight_reset_rule():
    assert classify_carrier_alignment(**{**BASE, "w_equals_h": True}) == (
        "VIEW_WEIGHT_CARRIER_RESET_MISMATCH"
    )


def test_temporal_state_rule():
    assert classify_carrier_alignment(**{**BASE, "f_equals_w": True}) == (
        "FINAL_REFRESH_TEMPORAL_STATE_MISMATCH"
    )


def test_combined_rule():
    assert classify_carrier_alignment(**BASE) == (
        "COMBINED_FINAL_REFRESH_CARRIER_MISMATCH"
    )


def test_fail_closed_preconditions():
    assert classify_carrier_alignment(**{**BASE, "historical_exact": False}) == (
        "HISTORICAL_CARRIER_REPLAY_NOT_EXACT"
    )
    assert classify_carrier_alignment(**{**BASE, "current_exact": False}) == (
        "CURRENT_RECONSTRUCTION_NOT_EXACT"
    )
    assert classify_carrier_alignment(**{**BASE, "post_model_and_q_local_exact": False}) == (
        "POST_NATIVE_MODEL_OR_QLOCAL_CONTRADICTION"
    )
    assert classify_carrier_alignment(**{**BASE, "t_equals_h": False}) == (
        "FINAL_REFRESH_REPLAY_NOT_EXACT"
    )
