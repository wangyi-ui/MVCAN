import pytest
import torch

import release_core.training.alternating as alternating
from release_core.training import build_decoupled_optimizers
from release_core.training import initial_native_target_state
from release_core.training import precompute_training_orders
from release_core.training import run_alternating_epoch
from release_core.training import run_alternating_training
from tests.release_core.test_r5_optimizer_topology import make_case
from tests.release_core.test_r5_optimizer_topology import make_native_state


def _epoch(case, state, epoch, refresh_interval, monkeypatch=None):
    semantic, native, _ = build_decoupled_optimizers(case["model"], 1e-4)
    orders = precompute_training_orders(case["sample_ids"], epoch + 1, seed=20)
    return run_alternating_epoch(
        case["model"], semantic, native, case["views"], case["sample_ids"],
        case["labeled_ids"], case["unlabeled_ids"], case["u_cycle"],
        case["pred_relation"], case["balance"], orders.semantic_orders[epoch],
        orders.native_orders[epoch], state, epoch=epoch, batch_size=8,
        refresh_interval=refresh_interval, native_lambda1=0.01, seed=20,
        device=case["device"],
    )


def test_refresh_epoch_event_order_and_post_phase_a_state(monkeypatch):
    case = make_case(2)
    initial_parameter = next(case["model"].autoencoders[0].parameters()).detach().clone()
    seen = {}

    def fake_refresh(model, views, *, epoch, refresh_interval, previous_state, seed, device):
        seen["changed"] = not torch.equal(
            next(model.autoencoders[0].parameters()).detach(), initial_parameter
        )
        state = make_native_state(case)
        audit = alternating.TargetRefreshAudit(
            epoch=epoch, executed=True, refresh_count=1,
            p_global_shape=tuple(state.p_global.shape),
            match_shape=tuple(state.matches.shape),
            view_weights=state.view_weights, used_post_phase_a_model=True,
        )
        return state, audit

    monkeypatch.setattr(alternating, "refresh_native_state_if_due", fake_refresh)
    state, audit = _epoch(case, initial_native_target_state(2), 0, 100)
    assert seen["changed"]
    assert audit.event_sequence == (
        "PHASE_A_START", "PHASE_A_END", "REFRESH_START", "REFRESH_END",
        "PHASE_B_START", "PHASE_B_END",
    )
    assert audit.target_refresh.executed
    assert audit.gradient_clean_transition_pass
    assert state.p_global is not None


def test_nonrefresh_epoch_reuses_target_and_view_weights():
    case = make_case(2)
    previous = make_native_state(case)
    state, audit = _epoch(case, previous, 1, 100)
    assert state is previous
    assert not audit.target_refresh.executed
    assert audit.event_sequence == (
        "PHASE_A_START", "PHASE_A_END", "PHASE_B_START", "PHASE_B_END",
    )
    assert state.view_weights is previous.view_weights


def test_twenty_synthetic_epochs_have_one_in_training_refresh():
    case = make_case(2)
    _, state, audit = run_alternating_training(
        case["model"], case["views"], case["sample_ids"],
        case["labeled_ids"], case["unlabeled_ids"], case["u_cycle"],
        case["pred_relation"], case["balance"], epochs=20, batch_size=8,
        seed=20, refresh_interval=100, learning_rate=1e-4,
        native_lambda1=0.01, device=case["device"],
    )
    assert state.refresh_count == audit.refresh_count == 1
    assert audit.epoch_count == 20
    assert audit.phase_a_backward_count == 20
    assert audit.phase_b_backward_count == 20
    assert not audit.final_prediction_refresh_executed
    assert audit.frozen_inputs_unchanged
    assert all(item.epoch == index for index, item in enumerate(audit.epoch_audits))


@pytest.mark.parametrize("field,value", [("epochs", 0), ("batch_size", 0), ("refresh_interval", 0)])
def test_invalid_training_cadence_fails_closed(field, value):
    case = make_case(2)
    kwargs = dict(
        epochs=1, batch_size=8, seed=20, refresh_interval=100,
        learning_rate=1e-4, native_lambda1=0.01, device=case["device"],
    )
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        run_alternating_training(
            case["model"], case["views"], case["sample_ids"],
            case["labeled_ids"], case["unlabeled_ids"], case["u_cycle"],
            case["pred_relation"], case["balance"], **kwargs,
        )
