import numpy as np
import torch

from experiments.cyclic_utility.evaluate_c0_complementary_semantic_verification import (
    load_frozen_e1_aligned_q,
)
from experiments.generic_contract.action_space import build_action_space
from experiments.generic_contract.generic_cycle_utility import build_cycle_utility
from irv.b4_information_utility import tensor_sha256
from weak_quality import ndarray_sha256


def _posterior(N=8, V=5, K=3):
    generator = torch.Generator().manual_seed(20)
    logits = torch.rand((N, V, K), generator=generator)
    return logits / logits.sum(dim=-1, keepdim=True)


def test_v5_shapes_and_range():
    result = build_cycle_utility(_posterior())
    assert result["U_cycle"].shape == (8, 20)
    assert result["y_gen"].shape == (8, 20)
    assert torch.isfinite(result["U_cycle"]).all()
    assert torch.all((result["U_cycle"] >= 0) & (result["U_cycle"] <= 1))


def test_v5_two_three_and_three_two_use_independent_means():
    q = _posterior(N=2)
    result = build_cycle_utility(q)
    assert torch.equal(result["p_gen"][:, 0], q[:, (0, 1), :].mean(1))
    assert torch.equal(result["p_ver"][:, 0], q[:, (2, 3, 4), :].mean(1))
    assert torch.equal(result["p_gen"][:, 10], q[:, (0, 1, 2), :].mean(1))
    assert torch.equal(result["p_ver"][:, 10], q[:, (3, 4), :].mean(1))
    assert result["cardinality_correction_used"] is False


def test_utility_exact_formula_and_zero_on_disagreement():
    q = torch.tensor([[
        [0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.1, 0.9]
    ]], dtype=torch.float64)
    result = build_cycle_utility(q)
    expected = result["closure"].to(q.dtype) * torch.sqrt(
        result["conf_gen"] * result["support_ver"]
    )
    assert torch.equal(result["U_cycle"], expected)
    assert torch.all(result["U_cycle"][~result["closure"]] == 0)


def test_utility_is_detached():
    q = _posterior().requires_grad_(True)
    result = build_cycle_utility(q)
    assert result["U_cycle"].requires_grad is False
    assert result["U_cycle"].grad_fn is None
    assert result["y_gen"].requires_grad is False


def test_v5_action_order():
    actions = build_action_space(5)
    assert actions.generator_subsets == (
        (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
        (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
        (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4),
        (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4),
    )
    assert actions.verifier_subsets[0] == (2, 3, 4)
    assert actions.verifier_subsets[10] == (3, 4)


def test_caltech_v6_frozen_exact_utility_parity():
    q_aligned, _, _, _ = load_frozen_e1_aligned_q(device="cpu")
    generic = build_cycle_utility(q_aligned)
    generic_float32 = generic["U_cycle"].numpy()
    with np.load(
        "outputs/cyclic_utility/c0_complementary_semantic_verification_seed20/"
        "c0_predictions_and_scores.npz",
        allow_pickle=False,
    ) as archive:
        frozen_float32 = np.array(archive["U_cycle"], copy=True)
    with np.load(
        "outputs/cyclic_utility/c3_a0_utility_conditioned_action_granularity_seed20/"
        "c3_a0_action_pre_gt.npz",
        allow_pickle=False,
    ) as archive:
        frozen_float64 = np.array(archive["U_cycle"], copy=True)
    generic_float64 = generic_float32.astype(np.float64)
    assert np.array_equal(generic_float32, frozen_float32)
    assert np.array_equal(generic_float64, frozen_float64)
    assert ndarray_sha256(generic_float64) == (
        "fb09bc2a69ef255868b27a0c036c4717c575b4403cbcaeef2f93d8702b6ffeeb"
    )
    assert tensor_sha256(generic_float64) == (
        "66686413301567b85170a4ffe7ef1f32b88dbad1d96bf34e485bbd1dea5ffdff"
    )
