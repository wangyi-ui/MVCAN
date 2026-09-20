import numpy as np
import pytest
import torch

from experiments.generic_contract.generic_cycle_utility import build_cycle_utility
from release_core.utility import (
    build_directional_actions,
    compute_directional_cycle_utility,
)


def _posterior(num_views, dtype, n_samples=7, n_clusters=4):
    generator = torch.Generator().manual_seed(2000 + num_views)
    values = torch.rand(
        (n_samples, num_views, n_clusters), generator=generator, dtype=dtype
    )
    return values / values.sum(dim=-1, keepdim=True)


@pytest.mark.parametrize("num_views", (2, 5, 6))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_all_synthetic_intermediates_are_exact(num_views, dtype):
    q_aligned = _posterior(num_views, dtype)
    clean = compute_directional_cycle_utility(q_aligned)
    reference = build_cycle_utility(q_aligned)
    for name in (
        "p_gen", "p_ver", "conf_gen", "y_gen", "support_ver", "closure",
        "U_cycle",
    ):
        assert torch.equal(clean[name], reference[name]), name
    assert torch.equal(
        torch.argmax(clean["p_ver"], dim=-1),
        torch.argmax(reference["p_ver"], dim=-1),
    )
    assert np.array_equal(
        clean["generator_membership"], reference["generator_membership"]
    )
    assert np.array_equal(
        clean["verifier_membership"], reference["verifier_membership"]
    )


@pytest.mark.parametrize("num_views,actions", ((2, 2), (5, 20), (6, 20)))
def test_output_shapes(num_views, actions):
    result = compute_directional_cycle_utility(
        _posterior(num_views, torch.float32, n_samples=3, n_clusters=5)
    )
    assert result["p_gen"].shape == (3, actions, 5)
    assert result["p_ver"].shape == (3, actions, 5)
    for name in ("conf_gen", "support_ver", "y_gen", "closure", "U_cycle"):
        assert result[name].shape == (3, actions)
    assert result["generator_membership"].shape == (actions, num_views)
    assert result["verifier_membership"].shape == (actions, num_views)


def test_agreement_disagreement_support_and_tie_semantics():
    q_aligned = torch.tensor(
        [
            [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]],
            [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]],
            [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]],
        ],
        dtype=torch.float64,
    )
    result = compute_directional_cycle_utility(q_aligned)
    assert result["closure"][0, 0].item() is True
    assert result["closure"][1, 0].item() is False
    assert result["U_cycle"][1, 0].item() == 0.0
    assert result["support_ver"][1, 0].item() == 0.2
    assert result["support_ver"][1, 0] != result["p_ver"][1, 0].max()
    assert result["y_gen"][2, 0].item() == 0
    assert torch.equal(
        result["U_cycle"],
        result["closure"].to(q_aligned.dtype)
        * torch.sqrt(result["conf_gen"] * result["support_ver"]),
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_uniform_posterior_is_finite_bounded_and_uses_first_tie(dtype):
    q_aligned = torch.full((4, 5, 3), 1.0 / 3.0, dtype=dtype)
    result = compute_directional_cycle_utility(q_aligned)
    assert torch.isfinite(result["U_cycle"]).all()
    assert torch.all(result["U_cycle"] >= 0.0)
    assert torch.all(result["U_cycle"] <= 1.0)
    assert torch.all(result["y_gen"] == 0)
    assert torch.all(result["closure"])


def test_input_and_all_floating_outputs_are_detached():
    original = _posterior(5, torch.float32).requires_grad_(True)
    before = original.detach().clone()
    result = compute_directional_cycle_utility(original)
    assert original.requires_grad is True
    assert torch.equal(original.detach(), before)
    for name in ("U_cycle", "p_gen", "p_ver", "conf_gen", "support_ver"):
        assert result[name].requires_grad is False
        assert result[name].grad_fn is None


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_numpy_input_preserves_dtype_on_cpu(dtype):
    q_aligned = _posterior(5, torch.float64).numpy().astype(dtype)
    result = compute_directional_cycle_utility(q_aligned)
    expected_dtype = torch.float32 if dtype is np.float32 else torch.float64
    assert result["U_cycle"].device.type == "cpu"
    assert result["U_cycle"].dtype == expected_dtype
    assert result["p_gen"].dtype == expected_dtype


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_torch_input_preserves_dtype_and_device(dtype):
    q_aligned = _posterior(6, dtype)
    result = compute_directional_cycle_utility(q_aligned)
    assert result["U_cycle"].dtype == dtype
    assert result["U_cycle"].device == q_aligned.device


def _valid_input():
    return torch.full((2, 2, 3), 1.0 / 3.0, dtype=torch.float64)


@pytest.mark.parametrize(
    "bad_input",
    (
        torch.ones((2, 3), dtype=torch.float32),
        torch.empty((0, 2, 3), dtype=torch.float32),
        torch.ones((2, 1, 3), dtype=torch.float32),
        torch.ones((2, 2, 1), dtype=torch.float32),
        torch.ones((2, 2, 3), dtype=torch.int64),
    ),
)
def test_invalid_shape_or_dtype_is_rejected(bad_input):
    with pytest.raises(ValueError):
        compute_directional_cycle_utility(bad_input)


@pytest.mark.parametrize("kind", ("nan", "inf", "negative", "upper", "mass"))
def test_invalid_values_are_rejected(kind):
    q_aligned = _valid_input()
    if kind == "nan":
        q_aligned[0, 0, 0] = float("nan")
    elif kind == "inf":
        q_aligned[0, 0, 0] = float("inf")
    elif kind == "negative":
        q_aligned[0, 0] = torch.tensor([-0.1, 0.5, 0.6])
    elif kind == "upper":
        q_aligned[0, 0] = torch.tensor([1.000001, 0.0, -0.000001])
    else:
        q_aligned[0, 0] = torch.tensor([0.2, 0.2, 0.2])
    with pytest.raises(ValueError):
        compute_directional_cycle_utility(q_aligned)


def test_unsupported_input_type_is_rejected():
    with pytest.raises(TypeError):
        compute_directional_cycle_utility([[[0.5, 0.5], [0.5, 0.5]]])


def test_action_space_view_mismatch_is_rejected():
    with pytest.raises(ValueError, match="view count"):
        compute_directional_cycle_utility(
            _posterior(5, torch.float32), build_directional_actions(6)
        )
