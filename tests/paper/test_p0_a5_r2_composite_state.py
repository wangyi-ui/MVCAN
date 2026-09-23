import copy
import types

import pytest
import torch
from torch import nn

from experiments.paper.diagnostics import msrc_pre_r2_carrier_temporal_alignment as audit
from experiments.paper.diagnostics.state_hashing import model_state_snapshot


class FakeAutoencoder(nn.Module):
    def __init__(self, value):
        super().__init__()
        self._encoder = nn.Linear(3, 2)
        self._decoder = nn.Linear(2, 3)
        self._cluster_layer = nn.Parameter(torch.full((2, 2), float(value)))
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(float(value))


class CompositeModel:
    """MvCAN-shaped composite: deliberately not an nn.Module."""
    def __init__(self):
        self.autoencoders = [FakeAutoencoder(1), FakeAutoencoder(2)]


def _fixture():
    source = CompositeModel()
    for autoencoder in source.autoencoders:
        autoencoder.eval()
    expected = model_state_snapshot(source)
    states = audit._state_dict_cpu(source)
    return source, expected, states


def test_per_view_capture_and_restore_do_not_use_composite_module_api():
    source, expected, states = _fixture()
    assert not hasattr(source, "state_dict")
    assert not hasattr(source, "load_state_dict")
    assert not hasattr(source, "cpu")
    assert not hasattr(source, "train")
    assert isinstance(states, tuple) and len(states) == 2
    assert all(state for state in states)
    restored = audit._build_cpu_clone(source, states, expected)
    assert audit._snapshot_exact(model_state_snapshot(restored), expected)


def test_per_view_order_strict_restore_hashes_cpu_and_training_are_exact():
    source, expected, states = _fixture()
    assert states[0]["_cluster_layer"].ne(states[1]["_cluster_layer"]).all()
    with torch.no_grad():
        for autoencoder in source.autoencoders:
            for parameter in autoencoder.parameters():
                parameter.zero_()
    restored = audit._build_cpu_clone(source, states, expected)
    restored_snapshot = model_state_snapshot(restored)
    assert restored_snapshot["aggregate_hash"] == expected["aggregate_hash"]
    for actual, frozen, state, autoencoder in zip(
        restored_snapshot["per_view"], expected["per_view"], states,
        restored.autoencoders,
    ):
        assert actual["view_id"] == frozen["view_id"]
        for field in ("model_hash", "encoder_parameter_hash",
                      "decoder_parameter_hash", "cluster_centers_hash"):
            assert actual[field] == frozen[field]
        assert torch.equal(autoencoder._cluster_layer, state["_cluster_layer"])
        assert autoencoder.training is True
        assert all(value.device.type == "cpu" for value in (
            tuple(autoencoder.parameters()) + tuple(autoencoder.buffers())
        ))


def test_mismatched_view_count_fails_closed():
    source, expected, states = _fixture()
    with pytest.raises(RuntimeError, match="state count mismatch"):
        audit._build_cpu_clone(source, states[:1], expected)


@pytest.mark.parametrize("mutation", ("missing", "corrupt"))
def test_missing_or_corrupt_per_view_state_fails_closed(mutation):
    source, expected, states = _fixture()
    altered = [dict(state) for state in states]
    if mutation == "missing":
        altered[0].pop("_cluster_layer")
    else:
        altered[0]["_cluster_layer"] = torch.zeros((1,))
    with pytest.raises(RuntimeError):
        audit._build_cpu_clone(source, tuple(altered), expected)


def test_composite_wrapper_normal_and_exception_restoration():
    legacy = types.SimpleNamespace()

    def native_refresh(*args):
        return (None, None, None, [1.0])

    def coordinate_snapshot(*args):
        return None, None

    legacy.native_refresh = native_refresh
    legacy.coordinate_snapshot = coordinate_snapshot
    original_refresh, original_coordinate = legacy.native_refresh, legacy.coordinate_snapshot
    with audit.instrument_historical_final_refresh(
        legacy, audit.HistoricalFinalRefreshCapture()
    ):
        assert legacy.native_refresh is not original_refresh
    assert legacy.native_refresh is original_refresh
    assert legacy.coordinate_snapshot is original_coordinate

    def failing_refresh(*args):
        raise RuntimeError("synthetic failure")

    legacy.native_refresh = failing_refresh
    original_refresh = legacy.native_refresh
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with audit.instrument_historical_final_refresh(
            legacy, audit.HistoricalFinalRefreshCapture()
        ):
            legacy.native_refresh(None, None, None, None, None)
    assert legacy.native_refresh is original_refresh
