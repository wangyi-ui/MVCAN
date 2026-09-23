import types

import numpy as np
import pytest

from experiments.paper.diagnostics import msrc_pre_r2_carrier_temporal_alignment as audit


def test_historical_carrier_gate_uses_frozen_hash_namespace():
    carrier = {
        "U_cycle": np.zeros((1,), dtype=np.float32),
        "PredRelation_true": np.zeros((1,), dtype=np.bool_),
        "relation_balance_weights_true": np.zeros((1,), dtype=np.float64),
    }
    exact, observed = audit._historical_h5_exact(carrier)
    assert exact is False
    assert set(observed) == set(audit.EXPECTED_HISTORICAL_ACTION_HASHES)


def test_current_f_gate_requires_all_frozen_provenance_records():
    assert audit.EXPECTED_SEED == 20
    assert audit.ARM_ARRAYS[:3] == ("q_local", "q_aligned", "M_v")
    assert "U_cycle" in audit.ARM_ARRAYS


def test_final_refresh_instrumentation_restores_legacy_wrappers():
    legacy = types.SimpleNamespace()
    calls = []

    def native_refresh(*args):
        calls.append("refresh")
        return (np.zeros((1, 1)), np.zeros((1, 1, 1)), np.zeros(1), [1.0])

    def coordinate_snapshot(*args):
        calls.append("coordinate")
        return np.zeros((1, 1, 1)), np.zeros((1, 1, 1))

    legacy.native_refresh = native_refresh
    legacy.coordinate_snapshot = coordinate_snapshot
    original_refresh, original_coordinate = legacy.native_refresh, legacy.coordinate_snapshot
    with audit.instrument_historical_final_refresh(
        legacy, audit.HistoricalFinalRefreshCapture()
    ):
        assert legacy.native_refresh is not original_refresh
        assert legacy.coordinate_snapshot is not original_coordinate
    assert legacy.native_refresh is original_refresh
    assert legacy.coordinate_snapshot is original_coordinate



def test_final_refresh_instrumentation_restores_on_exception():
    legacy = types.SimpleNamespace()

    def native_refresh(*args):
        raise RuntimeError("synthetic legacy failure")

    def coordinate_snapshot(*args):
        return np.zeros((1, 1, 1)), np.zeros((1, 1, 1))

    legacy.native_refresh = native_refresh
    legacy.coordinate_snapshot = coordinate_snapshot
    original_refresh, original_coordinate = legacy.native_refresh, legacy.coordinate_snapshot
    with pytest.raises(RuntimeError, match="synthetic legacy failure"):
        with audit.instrument_historical_final_refresh(
            legacy, audit.HistoricalFinalRefreshCapture()
        ):
            legacy.native_refresh(None, None, None, None, None)
    assert legacy.native_refresh is original_refresh
    assert legacy.coordinate_snapshot is original_coordinate

def test_static_carrier_semantics_are_retained_refresh_not_stale():
    source = audit.__file__
    text = open(source, encoding="utf-8").read()
    assert "H4_final_q_aligned_retained_M" in text
    assert "post-final q_local aligned with retained epoch-1000 pre-update refresh M_v" in text
    assert text.count("legacy.prepare_native_backbone(") == 1

def test_synthetic_historical_h5_gate_and_t_reproduction(monkeypatch):
    carrier = {
        "U_cycle": np.zeros((2,), dtype=np.float32),
        "PredRelation_true": np.zeros((2,), dtype=np.bool_),
        "relation_balance_weights_true": np.ones((2,), dtype=np.float64),
    }
    expected = {name: audit.ndarray_sha256(value) for name, value in carrier.items()}
    monkeypatch.setattr(audit, "EXPECTED_HISTORICAL_ACTION_HASHES", expected)
    assert audit._historical_h5_exact(carrier)[0] is True

def test_current_f_and_t_comparison_gates_are_exact_for_same_carrier():
    carrier = {
        "q_local": np.zeros((1, 5, 7), dtype=np.float32),
        "q_aligned": np.zeros((1, 5, 7), dtype=np.float32),
        "M_v": np.eye(7, dtype=np.float32)[None, ...].repeat(5, axis=0),
        "U_cycle": np.zeros((1,), dtype=np.float32),
        "y_gen": np.zeros((1,), dtype=np.int64),
        "PredRelation_true": np.zeros((1,), dtype=np.bool_),
        "relation_balance_weights_true": np.ones((1,), dtype=np.float64),
    }
    assert audit.arm_exact(audit.compare_arm(carrier, carrier)) is True
