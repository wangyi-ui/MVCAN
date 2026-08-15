"""B5-A0 unit tests for the protected Gaussian semantic rate channel."""

import hashlib
import inspect
import math
import struct
from pathlib import Path

import pytest
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from irv.b3_audit import hash_state_dict
from irv.b5_shared_semantic_rate import ConditionalPriorBank
from irv.b5_shared_semantic_rate import GaussianSemanticPosteriorBank
from irv.b5_shared_semantic_rate import analytic_kl_diag_gaussian
from irv.b5_shared_semantic_rate import analytic_kl_reference_check
from irv.b5_shared_semantic_rate import canonical_view_tensor_sha256
from irv.b5_shared_semantic_rate import build_conditional_context
from irv.b5_shared_semantic_rate import clamp_logvar
from irv.b5_shared_semantic_rate import conditional_kl_decomposition
from irv.b5_shared_semantic_rate import conditional_rate_views
from irv.b5_shared_semantic_rate import deterministic_semantic
from irv.b5_shared_semantic_rate import fixed_radius_mean
from irv.b5_shared_semantic_rate import gradient_audit
from irv.b5_shared_semantic_rate import isotropic_directional_null_audit
from irv.b5_shared_semantic_rate import isotropic_kl
from irv.b5_shared_semantic_rate import isotropic_kl_decomposition
from irv.b5_shared_semantic_rate import normalize_native_z
from irv.b5_shared_semantic_rate import logvar_audit
from irv.b5_shared_semantic_rate import normalize_semantic
from irv.b5_shared_semantic_rate import per_dimension_rate
from irv.b5_shared_semantic_rate import rate_distribution_audit
from irv.b5_shared_semantic_rate import reparameterized_sample
from irv.b5_shared_semantic_rate import sample_semantic_views
from irv.b5_shared_semantic_rate import shuffle_context_views
from irv.b5_shared_semantic_rate import tensor_list_sha256
from irv.b5_shared_semantic_rate import total_rate_distortion_loss
from irv.b5_shared_semantic_rate import uniform_semantic_distortion
from irv.b5_shared_semantic_rate import z_l2_normalization_audit
from irv.semantic_head import DetachedSemanticHeadBank
from irv.semantic_loss import uniform_cross_view_infonce


def _posterior(view_num=5):
    return GaussianSemanticPosteriorBank(view_num, 10, 10, 1020)


def _z_views(sample_num=12, view_num=5, requires_grad=False):
    generator = torch.Generator().manual_seed(91)
    return [
        torch.randn((sample_num, 10), generator=generator).requires_grad_(
            requires_grad
        )
        for _ in range(view_num)
    ]


def _posterior_outputs(sample_num=12, view_num=5):
    posterior = _posterior(view_num)
    return posterior, posterior(_z_views(sample_num, view_num))


def _random_gaussians(dtype=torch.float64):
    generator = torch.Generator().manual_seed(19)
    mean_q = torch.randn((13, 10), generator=generator, dtype=dtype)
    logvar_q = torch.empty((13, 10), dtype=dtype).uniform_(
        -3.0, 1.0, generator=generator
    )
    mean_p = torch.randn((13, 10), generator=generator, dtype=dtype)
    logvar_p = torch.empty((13, 10), dtype=dtype).uniform_(
        -3.0, 1.0, generator=generator
    )
    return mean_q, logvar_q, mean_p, logvar_p


def test_01_posterior_mean_and_logvar_shapes():
    posterior = _posterior()
    mean, logvar = posterior.forward_one(torch.randn(17, 10), 0)
    assert mean.shape == (17, 10)
    assert logvar.shape == (17, 10)


def test_02_fixed_radius_mean_matches_sqrt_ten_with_tolerance():
    fixed = fixed_radius_mean(torch.randn(31, 10), math.sqrt(10))
    error = torch.max(torch.abs(torch.linalg.vector_norm(fixed, dim=1) - math.sqrt(10)))
    assert error.item() < 1e-4


def test_03_logvar_clamp_range():
    raw = torch.tensor([[-10.0, -6.0, 0.0, 2.0, 9.0]])
    actual = clamp_logvar(raw, -6.0, 2.0)
    assert actual.min().item() == -6.0
    assert actual.max().item() == 2.0


def test_04_reparameterized_sample_and_normalized_semantic_shape_norm():
    mean = fixed_radius_mean(torch.randn(21, 10), math.sqrt(10))
    logvar = torch.zeros_like(mean)
    sample, epsilon = reparameterized_sample(
        mean, logvar, torch.Generator().manual_seed(7)
    )
    semantic = normalize_semantic(sample)
    assert sample.shape == epsilon.shape == semantic.shape == (21, 10)
    assert torch.allclose(
        torch.linalg.vector_norm(semantic, dim=1), torch.ones(21), atol=1e-6
    )


def test_05_deterministic_eval_uses_mean_not_random_sample():
    mean = fixed_radius_mean(torch.randn(9, 10), math.sqrt(10))
    logvar = torch.zeros_like(mean)
    first_sample, _ = reparameterized_sample(
        mean, logvar, torch.Generator().manual_seed(1)
    )
    second_sample, _ = reparameterized_sample(
        mean, logvar, torch.Generator().manual_seed(2)
    )
    expected = F.normalize(mean, dim=1)
    assert torch.equal(deterministic_semantic(mean), expected)
    assert not torch.equal(first_sample, second_sample)


def test_06_isotropic_analytic_kl_matches_torch_distribution():
    mean_q, logvar_q, _, _ = _random_gaussians()
    q = torch.distributions.Independent(
        torch.distributions.Normal(mean_q, torch.exp(0.5 * logvar_q)), 1
    )
    p = torch.distributions.Independent(
        torch.distributions.Normal(
            torch.zeros_like(mean_q), torch.ones_like(mean_q)
        ), 1
    )
    reference = torch.distributions.kl_divergence(q, p)
    error = torch.max(torch.abs(isotropic_kl(mean_q, logvar_q) - reference))
    assert error.item() < 1e-6


def test_07_conditional_analytic_kl_matches_torch_distribution():
    mean_q, logvar_q, mean_p, logvar_p = _random_gaussians()
    q = torch.distributions.Independent(
        torch.distributions.Normal(mean_q, torch.exp(0.5 * logvar_q)), 1
    )
    p = torch.distributions.Independent(
        torch.distributions.Normal(mean_p, torch.exp(0.5 * logvar_p)), 1
    )
    reference = torch.distributions.kl_divergence(q, p)
    actual = analytic_kl_diag_gaussian(mean_q, logvar_q, mean_p, logvar_p)
    assert torch.max(torch.abs(actual - reference)).item() < 1e-6


def test_08_analytic_kl_is_nonnegative_with_numerical_tolerance():
    values = _random_gaussians()
    assert analytic_kl_diag_gaussian(*values).min().item() >= -1e-10
    assert isotropic_kl(values[0], values[1]).min().item() >= -1e-10


def test_09_isotropic_kl_decomposition_is_exact():
    mean_q, logvar_q, _, _ = _random_gaussians()
    decomposition = isotropic_kl_decomposition(mean_q, logvar_q)
    reference = isotropic_kl(mean_q, logvar_q)
    assert torch.allclose(decomposition["total"], reference, atol=1e-12)
    assert torch.allclose(
        decomposition["total"],
        decomposition["mean"] + decomposition["variance"],
        atol=1e-12,
    )


def test_10_rate_per_dimension_divides_kl_by_semantic_dim():
    kl = torch.tensor([10.0, 20.0])
    assert torch.equal(per_dimension_rate(kl, 10), torch.tensor([1.0, 2.0]))


def test_11_semantic_only_total_equals_distortion():
    distortion = torch.tensor(3.5)
    rate = torch.tensor(0.8)
    total = total_rate_distortion_loss(distortion, rate, 0.1, False)
    assert torch.equal(total, distortion)


def test_12_isotropic_total_is_distortion_plus_beta_rate():
    distortion = torch.tensor(3.5)
    rate = torch.tensor(0.8)
    total = total_rate_distortion_loss(distortion, rate, 0.1, True)
    assert total.item() == pytest.approx(3.58)


def test_13_conditional_total_is_distortion_plus_beta_rate():
    distortion = torch.tensor(2.25)
    rate = torch.tensor(1.5)
    total = total_rate_distortion_loss(distortion, rate, 0.1, True)
    assert total.item() == pytest.approx(2.40)


def test_14_conditional_context_excludes_target():
    radius = math.sqrt(10)
    means = [torch.full((3, 10), float(index + 1)) for index in range(5)]
    contexts = build_conditional_context(means, radius)
    for target in range(5):
        expected = torch.stack([
            means[side] / radius for side in range(5) if side != target
        ]).mean(dim=0)
        assert torch.equal(contexts[target], expected)


def test_15_conditional_context_means_exactly_four_side_views():
    radius = math.sqrt(10)
    means = [torch.full((2, 10), float(index)) for index in range(5)]
    contexts = build_conditional_context(means, radius)
    assert torch.equal(
        contexts[2], (means[0] + means[1] + means[3] + means[4]) / (4 * radius)
    )


def test_16_conditional_context_detaches_side_gradient_path():
    sides = [torch.randn(4, 10, requires_grad=True) for _ in range(5)]
    contexts = build_conditional_context(sides, math.sqrt(10))
    assert all(not context.requires_grad for context in contexts)
    prior = ConditionalPriorBank(5, 10, 3020)
    prior_mean, prior_logvar = prior(contexts)
    loss = sum(value.sum() for value in prior_mean + prior_logvar)
    gradients = torch.autograd.grad(loss, sides, allow_unused=True)
    assert all(value is None for value in gradients)


def test_17_conditional_prior_receives_finite_nonzero_gradient():
    posterior, (mean_q, logvar_q) = _posterior_outputs()
    prior = ConditionalPriorBank(5, 10, 3020)
    contexts = build_conditional_context(mean_q, posterior.radius_target)
    mean_p, logvar_p = prior(contexts)
    loss = conditional_rate_views(mean_q, logvar_q, mean_p, logvar_p).mean()
    loss.backward()
    audit = gradient_audit(prior)
    assert audit["finite"] and audit["nonzero"]


def test_18_posterior_receives_finite_nonzero_gradient():
    posterior, (mean, logvar) = _posterior_outputs(sample_num=8, view_num=2)
    semantic, _, _ = sample_semantic_views(
        mean, logvar, torch.Generator().manual_seed(7)
    )
    loss, _ = uniform_semantic_distortion(semantic, 0.2)
    loss.backward()
    audit = gradient_audit(posterior)
    assert audit["finite"] and audit["nonzero"]


def test_19_frozen_z_receives_no_gradient():
    z_views = _z_views(sample_num=8, view_num=2, requires_grad=True)
    posterior = _posterior(view_num=2)
    mean, logvar = posterior(z_views)
    semantic, _, _ = sample_semantic_views(
        mean, logvar, torch.Generator().manual_seed(8)
    )
    uniform_semantic_distortion(semantic, 0.2)[0].backward()
    assert all(value.grad is None for value in z_views)


def test_20_backbone_parameters_keep_grad_none():
    backbones = nn.ModuleList([nn.Linear(6, 10), nn.Linear(6, 10)])
    inputs = [torch.randn(8, 6), torch.randn(8, 6)]
    z_views = [module(value) for module, value in zip(backbones, inputs)]
    posterior = _posterior(view_num=2)
    mean, logvar = posterior(z_views)
    semantic, _, _ = sample_semantic_views(
        mean, logvar, torch.Generator().manual_seed(10)
    )
    uniform_semantic_distortion(semantic, 0.2)[0].backward()
    assert all(parameter.grad is None for parameter in backbones.parameters())


def test_21_same_posterior_template_hashes_all_arms_equal():
    template = _posterior()
    state = template.state_dict()
    hashes = []
    for _ in range(3):
        arm = _posterior()
        arm.load_state_dict(state, strict=True)
        hashes.append(hash_state_dict(arm.state_dict()))
    assert len(set(hashes)) == 1


def test_22_same_sampling_seed_produces_equal_first_epsilon_hashes():
    posterior, (mean, logvar) = _posterior_outputs()
    hashes = []
    for _ in range(3):
        generator = torch.Generator().manual_seed(20260815)
        _, _, epsilon = sample_semantic_views(mean, logvar, generator)
        hashes.append(tensor_list_sha256(epsilon))
    assert len(set(hashes)) == 1


def test_23_local_prior_rng_does_not_perturb_posterior_initialization():
    first = _posterior()
    ConditionalPriorBank(5, 10, 3020)
    second = _posterior()
    assert hash_state_dict(first.state_dict()) == hash_state_dict(second.state_dict())


def test_24_b5_distortion_is_numerically_equivalent_to_b3_uniform_loss():
    generator = torch.Generator().manual_seed(71)
    semantics = [
        F.normalize(torch.randn((12, 10), generator=generator), dim=1)
        for _ in range(5)
    ]
    b5_loss, _ = uniform_semantic_distortion(semantics, 0.2)
    b3_loss, _ = uniform_cross_view_infonce(semantics, 0.2)
    assert torch.equal(b5_loss, b3_loss)


def test_25_five_view_semantic_pair_count_is_ten():
    semantics = [F.normalize(torch.randn(7, 10), dim=1) for _ in range(5)]
    _, diagnostics = uniform_semantic_distortion(semantics, 0.2)
    assert diagnostics["pair_count"] == 10


def test_26_rate_api_has_no_external_weighting_argument_or_import():
    parameters = inspect.signature(conditional_rate_views).parameters
    assert set(parameters) == {
        "posterior_mu_views",
        "posterior_logvar_views",
        "prior_mu_views",
        "prior_logvar_views",
    }
    source = Path("irv/b5_shared_semantic_rate.py").read_text(encoding="utf-8")
    assert "irv.b4_information_utility" not in source


def test_27_posterior_api_has_no_labels():
    parameters = inspect.signature(GaussianSemanticPosteriorBank.forward_one).parameters
    assert set(parameters) == {"self", "z_v", "view_idx"}


def test_28_rate_api_has_no_corruption_mask():
    parameters = inspect.signature(conditional_rate_views).parameters
    assert all("mask" not in name.lower() for name in parameters)


def test_29_conditional_context_shuffle_is_deterministic():
    contexts = [torch.randn(15, 10) for _ in range(5)]
    first = shuffle_context_views(contexts, 20260815)
    second = shuffle_context_views(contexts, 20260815)
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_30_correct_and_shuffled_conditional_rates_are_finite():
    posterior, (mean_q, logvar_q) = _posterior_outputs()
    prior = ConditionalPriorBank(5, 10, 3020)
    contexts = build_conditional_context(mean_q, posterior.radius_target)
    shuffled = shuffle_context_views(contexts, 20260815)
    correct_mean, correct_logvar = prior(contexts)
    shuffled_mean, shuffled_logvar = prior(shuffled)
    correct = conditional_rate_views(
        mean_q, logvar_q, correct_mean, correct_logvar
    )
    shuffled_rate = conditional_rate_views(
        mean_q, logvar_q, shuffled_mean, shuffled_logvar
    )
    assert torch.isfinite(correct).all()
    assert torch.isfinite(shuffled_rate).all()


def test_31_rate_percentiles_are_monotonic():
    audit = rate_distribution_audit(torch.arange(1.0, 101.0).reshape(20, 5))
    values = [
        audit["rate_p50"], audit["rate_p75"], audit["rate_p90"],
        audit["rate_p95"], audit["rate_p99"],
    ]
    assert values == sorted(values)


def test_32_rate_max_to_mean_ratio_is_finite():
    audit = rate_distribution_audit(torch.rand(21, 5) + 0.1)
    assert math.isfinite(audit["rate_max_to_mean_ratio"])


def test_33_logvar_saturation_fractions_are_valid_probabilities():
    audit = logvar_audit([torch.linspace(-6.0, 2.0, 100).reshape(10, 10)])
    assert 0.0 <= audit["posterior_logvar_at_min_fraction"] <= 1.0
    assert 0.0 <= audit["posterior_logvar_at_max_fraction"] <= 1.0


def test_34_full_msrc_tensor_shapes_are_supported():
    posterior = _posterior()
    mean, logvar = posterior(_z_views(sample_num=210, view_num=5))
    semantic, samples, epsilon = sample_semantic_views(
        mean, logvar, torch.Generator().manual_seed(20260815)
    )
    assert len(mean) == len(logvar) == len(semantic) == 5
    assert all(value.shape == (210, 10) for value in mean + logvar)
    assert all(value.shape == (210, 10) for value in semantic + samples + epsilon)


def test_35_posterior_mean_initialization_exactly_matches_b3_heads():
    posterior = _posterior()
    b3 = DetachedSemanticHeadBank(5, 10, 10, 1020)
    for posterior_head, b3_head in zip(posterior.mean_heads, b3.heads):
        assert torch.equal(posterior_head.weight, b3_head.weight)
        assert torch.equal(posterior_head.bias, b3_head.bias)


def test_36_analytic_reference_check_helper_passes_below_tolerance():
    audit = analytic_kl_reference_check()
    assert audit["analytic_kl_check_pass"]
    assert audit["analytic_kl_reference_max_abs_error"] < 1e-6


def test_37_conditional_kl_decomposition_is_exact():
    values = _random_gaussians()
    decomposition = conditional_kl_decomposition(*values)
    reference = analytic_kl_diag_gaussian(*values)
    assert torch.allclose(decomposition["total"], reference, atol=1e-12)
    assert torch.allclose(
        decomposition["total"],
        decomposition["mean_mismatch"] + decomposition["variance"],
        atol=1e-12,
    )


def test_38_tensor_hash_is_deterministic_and_value_sensitive():
    values = [torch.arange(20).reshape(2, 10).float(), torch.ones(2, 10)]
    first = tensor_list_sha256(values)
    second = tensor_list_sha256([value.clone() for value in values])
    changed = tensor_list_sha256([values[0], values[1] + 1.0])
    assert first == second
    assert first != changed


def test_39_frozen_native_z_rows_are_l2_normalized():
    raw_z = torch.randn(210, 10) * torch.linspace(0.5, 3.0, 210).unsqueeze(1)
    normalized_z = normalize_native_z(raw_z)
    norms = torch.linalg.vector_norm(normalized_z, dim=1)
    audit = z_l2_normalization_audit([normalized_z])
    assert torch.max(torch.abs(norms - 1.0)).item() <= 1e-6
    assert audit["z_l2_norm_max_abs_error_from_one"] <= 1e-6
    assert audit["z_normalization_pass"]
    assert not normalized_z.requires_grad


def test_40_canonical_view_tensor_hash_is_deterministic_and_ordered():
    values = [
        torch.arange(20, dtype=torch.float32).reshape(2, 10),
        torch.linspace(-1.0, 1.0, 20, dtype=torch.float32).reshape(2, 10),
    ]
    first = canonical_view_tensor_sha256(values)
    repeated = canonical_view_tensor_sha256([value.clone() for value in values])
    reordered = canonical_view_tensor_sha256(list(reversed(values)))
    assert first == repeated
    assert first != reordered


def test_41_canonical_hash_matches_independent_reference_semantics():
    values = [
        torch.tensor([[1.0, -2.0], [3.5, 4.25]], dtype=torch.float32),
        np.asarray([[7, 8, 9]], dtype=np.int64),
    ]

    def reference_tensor_hash(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(np.asarray(value))
        digest = hashlib.sha256()
        for component in (
                str(array.dtype).encode("ascii"),
                ",".join(str(int(size)) for size in array.shape).encode("ascii"),
                array.tobytes(order="C")):
            digest.update(struct.pack(">Q", len(component)))
            digest.update(component)
        return digest.hexdigest()

    digest = hashlib.sha256()
    for view_id, value in enumerate(values):
        component = (
            str(view_id) + ":" + reference_tensor_hash(value)
        ).encode("ascii")
        digest.update(struct.pack(">Q", len(component)))
        digest.update(component)
    assert canonical_view_tensor_sha256(values) == digest.hexdigest()


def test_42_fixed_radius_isotropic_direction_gradient_is_null():
    audit = isotropic_directional_null_audit()
    assert audit["isotropic_mu_direction_grad_norm"] < 1e-10
    assert audit["isotropic_mu_direction_grad_max_abs"] < 1e-10
    assert audit["B5_A0_ISOTROPIC_DIRECTIONAL_NULL_CONFIRMED"]


def test_43_isotropic_variance_gradient_is_zero_at_zero_logvar():
    audit = isotropic_directional_null_audit()
    assert audit["isotropic_logvar_zero_grad_norm"] < 1e-10
    assert audit["isotropic_zero_logvar_stationary_confirmed"]


def test_44_isotropic_variance_gradient_is_nonzero_away_from_zero():
    audit = isotropic_directional_null_audit()
    assert audit["isotropic_logvar_nonzero_grad_norm"] > 0.0
    assert audit["isotropic_nonzero_logvar_active_confirmed"]

