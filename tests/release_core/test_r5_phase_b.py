import numpy as np
import pytest
import torch

import release_core.training.alternating as alternating
from release_core.training import build_decoupled_optimizers
from release_core.training import initial_native_target_state
from release_core.training import run_native_consolidation_phase
from tests.release_core.test_r5_optimizer_topology import make_case
from tests.release_core.test_r5_optimizer_topology import make_native_state


def _run(case, *, order=None, state=None, batch_size=3):
    _, native, _ = build_decoupled_optimizers(case["model"], 1e-4)
    if order is None:
        order = torch.from_numpy(case["sample_ids"].copy())
    if state is None:
        state = make_native_state(case)
    audit = run_native_consolidation_phase(
        case["model"], native, case["views"], case["sample_ids"], state,
        order, batch_size, 0.01, case["device"],
    )
    return native, audit


@pytest.mark.parametrize("view_count", [2, 5, 6])
def test_phase_b_full_coverage_sum_steps_and_gradient_cleanup(view_count):
    case = make_case(view_count)
    native, audit = _run(case)
    expected_batches = 3
    assert audit.phase == "native_consolidation"
    assert audit.sample_count == case["sample_ids"].size
    assert audit.batch_count == audit.backward_count == expected_batches
    assert audit.optimizer_step_count == view_count * expected_batches
    assert audit.encoder_gradient_path
    assert audit.decoder_gradient_path
    assert audit.cluster_gradient_path
    assert audit.gradients_clean_at_end
    assert all(optimizer.state for optimizer in native)
    assert all(parameter.grad is None for ae in case["model"].autoencoders for parameter in ae.parameters())


def test_phase_b_calls_clean_r1_per_view_and_sums_losses(monkeypatch):
    case = make_case(2)
    original = alternating.native_objective
    recorded = []

    def wrapped(*args, **kwargs):
        loss, rec, clu = original(*args, **kwargs)
        assert not args[3].requires_grad
        recorded.append(float(loss.detach().item()))
        return loss, rec, clu

    monkeypatch.setattr(alternating, "native_objective", wrapped)
    _, audit = _run(case, batch_size=8)
    assert len(recorded) == 2
    assert audit.loss_sum == pytest.approx(sum(recorded), rel=1e-6)


def test_noncontiguous_ids_select_p_global_by_canonical_row(monkeypatch):
    case = make_case(2, noncontiguous=True)
    state = make_native_state(case)
    order = torch.from_numpy(case["sample_ids"][::-1].copy())
    expected = state.p_global.flip(0)
    observed = []
    original = alternating.native_objective

    def wrapped(reconstruction, inputs, q_local, p_local, lambda1):
        if len(observed) == 0:
            observed.append(p_local.detach().clone())
        return original(reconstruction, inputs, q_local, p_local, lambda1)

    monkeypatch.setattr(alternating, "native_objective", wrapped)
    _run(case, order=order, state=state, batch_size=8)
    assert torch.equal(observed[0], expected)


def test_missing_native_state_fails_closed():
    case = make_case(2)
    missing = initial_native_target_state(2)
    with pytest.raises(RuntimeError, match="missing"):
        _run(case, state=missing)


def test_nonfinite_native_loss_fails_closed(monkeypatch):
    case = make_case(2)

    def nonfinite(reconstruction, *_args, **_kwargs):
        value = reconstruction.sum() * torch.tensor(float("nan"))
        return value, value, value

    monkeypatch.setattr(alternating, "native_objective", nonfinite)
    with pytest.raises(RuntimeError, match="finite scalar"):
        _run(case, batch_size=8)
