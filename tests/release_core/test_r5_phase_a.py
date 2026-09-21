import numpy as np
import pytest
import torch

import release_core.training.alternating as alternating
from release_core.training import build_decoupled_optimizers
from release_core.training import run_relation_refinement_phase
from tests.release_core.test_r5_optimizer_topology import make_case


def _run(case, *, batch_size=4, order=None):
    semantic, _, _ = build_decoupled_optimizers(case["model"], 1e-4)
    if order is None:
        order = torch.from_numpy(case["sample_ids"].copy())
    audit = run_relation_refinement_phase(
        case["model"], semantic, case["views"], case["sample_ids"],
        case["labeled_ids"], case["unlabeled_ids"], case["u_cycle"],
        case["pred_relation"], case["balance"], order, batch_size,
        case["device"],
    )
    return semantic, audit


@pytest.mark.parametrize("view_count", [2, 5, 6])
def test_phase_a_population_steps_gradients_and_cleanup(view_count):
    case = make_case(view_count)
    before = tuple(value.copy() for value in (
        case["u_cycle"], case["pred_relation"], case["balance"]
    ))
    semantic, audit = _run(case)
    assert audit.phase == "relation_refinement"
    assert audit.sample_count == case["unlabeled_ids"].size
    assert audit.batch_count == audit.backward_count == 2
    assert audit.optimizer_step_count == view_count * 2
    assert audit.anchor_recomputation_count == 2
    assert audit.encoder_gradient_path
    assert audit.cluster_gradient_path
    assert not audit.decoder_gradient_path
    assert audit.gradients_clean_at_end
    assert all(parameter.grad is None for ae in case["model"].autoencoders for parameter in ae.parameters())
    assert all(optimizer.state for optimizer in semantic)
    assert all(np.array_equal(old, new) for old, new in zip(
        before, (case["u_cycle"], case["pred_relation"], case["balance"])
    ))


def test_anchor_is_recomputed_without_gradient_and_decoder_is_never_called():
    case = make_case(2, labeled_count=1)
    anchor_outputs = []
    handles = []
    for autoencoder in case["model"].autoencoders:
        handles.append(autoencoder._encoder.register_forward_hook(
            lambda _module, inputs, output: anchor_outputs.append(
                (int(inputs[0].shape[0]), output.detach().clone(), output.requires_grad)
            )
        ))
        handles.append(autoencoder._decoder.register_forward_hook(
            lambda *_args: pytest.fail("decoder executed during Phase A")
        ))
    try:
        _, audit = _run(case)
    finally:
        for handle in handles:
            handle.remove()
    per_view_anchor = [entry for entry in anchor_outputs if entry[0] == 1]
    assert len(per_view_anchor) == 2 * audit.batch_count
    assert all(not entry[2] for entry in per_view_anchor)
    assert any(
        not torch.equal(per_view_anchor[index][1], per_view_anchor[index + 2][1])
        for index in range(2)
    )


def test_phase_a_calls_clean_r4_once_per_batch_without_rescaling(monkeypatch):
    case = make_case(2)
    original = alternating.utility_conditioned_relation_loss
    returned = []

    def wrapped(*args, **kwargs):
        loss, audit = original(*args, **kwargs)
        returned.append(float(loss.detach().item()))
        return loss, audit

    monkeypatch.setattr(alternating, "utility_conditioned_relation_loss", wrapped)
    _, audit = _run(case, batch_size=case["sample_ids"].size)
    assert len(returned) == 1
    assert audit.loss_sum == pytest.approx(returned[0], rel=0.0, abs=0.0)


def test_noncontiguous_global_ids_index_actions_correctly():
    case = make_case(2, noncontiguous=True)
    order = torch.from_numpy(case["sample_ids"][::-1].copy())
    _, audit = _run(case, order=order)
    assert audit.sample_count == case["unlabeled_ids"].size


def test_empty_filtered_batch_fails_closed():
    case = make_case(2, labeled_count=2)
    order = torch.from_numpy(case["sample_ids"].copy())
    semantic, _, _ = build_decoupled_optimizers(case["model"], 1e-4)
    with pytest.raises(RuntimeError, match="empty"):
        run_relation_refinement_phase(
            case["model"], semantic, case["views"], case["sample_ids"],
            case["labeled_ids"], case["unlabeled_ids"], case["u_cycle"],
            case["pred_relation"], case["balance"], order, 1, case["device"],
        )


def test_nonfinite_relation_loss_fails_closed(monkeypatch):
    case = make_case(2)

    def nonfinite(q_query_views, *_args, **_kwargs):
        return q_query_views[0].sum() * torch.tensor(float("nan")), None

    monkeypatch.setattr(alternating, "utility_conditioned_relation_loss", nonfinite)
    with pytest.raises(RuntimeError, match="finite scalar"):
        _run(case, batch_size=8)
