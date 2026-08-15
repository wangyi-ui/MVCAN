"""Protected Gaussian shared-semantic rate channel for the B5-A0 probe."""

import hashlib
import math
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from irv.semantic_loss import uniform_cross_view_infonce


POSTERIOR_LOGVAR_MIN = -6.0
POSTERIOR_LOGVAR_MAX = 2.0


def _require_matrix(value, name):
    if not torch.is_tensor(value) or value.ndim != 2:
        raise ValueError(name + " must have shape [N,D]")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(name + " dimensions must be positive")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(name + " must be finite")


def _length_prefixed(digest, value):
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def tensor_list_sha256(values):
    """Hash a tensor list with dtype, shape, ordering, and exact bytes."""
    digest = hashlib.sha256()
    for index, value in enumerate(values):
        if not torch.is_tensor(value):
            raise TypeError("all values must be tensors")
        contiguous = value.detach().cpu().contiguous()
        raw = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        shape = ",".join(str(int(size)) for size in contiguous.shape)
        _length_prefixed(digest, str(index).encode("ascii"))
        _length_prefixed(digest, str(contiguous.dtype).encode("ascii"))
        _length_prefixed(digest, shape.encode("ascii"))
        _length_prefixed(digest, raw)
    return digest.hexdigest()


def _canonical_tensor_sha256(value):
    """Hash one tensor using the frozen cross-stage tensor audit semantics."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    for component in (
            str(array.dtype).encode("ascii"),
            ",".join(str(int(size)) for size in array.shape).encode("ascii"),
            array.tobytes(order="C")):
        _length_prefixed(digest, component)
    return digest.hexdigest()


def canonical_view_tensor_sha256(values):
    """Hash ordered views with the canonical B4-compatible audit semantics.

    This is an independent serialization helper. It intentionally has no
    dependency on any reliability or weighting module.
    """
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("values must be a non-empty tensor/array view list")
    digest = hashlib.sha256()
    for view_id, value in enumerate(values):
        component = (
            str(view_id) + ":" + _canonical_tensor_sha256(value)
        ).encode("ascii")
        _length_prefixed(digest, component)
    return digest.hexdigest()


def normalize_native_z(raw_z):
    """Apply the canonical frozen Native-z row normalization and detach."""
    _require_matrix(raw_z, "raw_z")
    # raw_z:        [N, Dz]
    # normalized_z: [N, Dz]
    return F.normalize(raw_z, p=2, dim=1, eps=1e-12).detach()


def z_l2_normalization_audit(z_views):
    """Audit per-view and aggregate row norms of normalized Native z."""
    if not isinstance(z_views, (list, tuple)) or not z_views:
        raise ValueError("z_views must be a non-empty list or tuple")
    per_view = []
    all_norms = []
    for view_id, z_v in enumerate(z_views):
        _require_matrix(z_v, "z_view")
        norms = torch.linalg.vector_norm(z_v.detach(), ord=2, dim=1)
        errors = torch.abs(norms - 1.0)
        all_norms.append(norms)
        per_view.append({
            "view_id": int(view_id),
            "z_l2_norm_mean": float(norms.mean().item()),
            "z_l2_norm_std": float(norms.std(unbiased=False).item()),
            "z_l2_norm_min": float(norms.min().item()),
            "z_l2_norm_max": float(norms.max().item()),
            "z_l2_norm_max_abs_error_from_one": float(errors.max().item()),
        })
    flat = torch.cat(all_norms)
    max_error = float(torch.abs(flat - 1.0).max().item())
    return {
        "z_l2_norm_mean": float(flat.mean().item()),
        "z_l2_norm_std": float(flat.std(unbiased=False).item()),
        "z_l2_norm_min": float(flat.min().item()),
        "z_l2_norm_max": float(flat.max().item()),
        "z_l2_norm_max_abs_error_from_one": max_error,
        "z_l2_norm_per_view": per_view,
        "z_normalization_pass": bool(max_error < 1e-6),
    }


def fixed_radius_mean(raw_mean, radius_target):
    """Remove the posterior/prior mean-norm rate gauge."""
    _require_matrix(raw_mean, "raw_mean")
    radius_target = float(radius_target)
    if not math.isfinite(radius_target) or radius_target <= 0.0:
        raise ValueError("radius_target must be positive and finite")
    # fixed_mean: [N, Ds]
    return radius_target * F.normalize(raw_mean, p=2, dim=-1, eps=1e-12)


def clamp_logvar(raw_logvar, minimum=POSTERIOR_LOGVAR_MIN,
                 maximum=POSTERIOR_LOGVAR_MAX):
    """Clamp diagonal-Gaussian log variance to the preregistered interval."""
    _require_matrix(raw_logvar, "raw_logvar")
    minimum = float(minimum)
    maximum = float(maximum)
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("logvar bounds must be finite")
    if minimum >= maximum:
        raise ValueError("logvar minimum must be below maximum")
    return torch.clamp(raw_logvar, min=minimum, max=maximum)


class GaussianSemanticPosteriorBank(nn.Module):
    """Independent per-view posterior q(h^v | detached z^v)."""

    def __init__(self, view_num, latent_dim, semantic_dim, semantic_seed,
                 logvar_min=POSTERIOR_LOGVAR_MIN,
                 logvar_max=POSTERIOR_LOGVAR_MAX):
        super(GaussianSemanticPosteriorBank, self).__init__()
        self.view_num = int(view_num)
        self.latent_dim = int(latent_dim)
        self.semantic_dim = int(semantic_dim)
        self.semantic_seed = int(semantic_seed)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.radius_target = math.sqrt(self.semantic_dim)
        if self.view_num <= 0 or self.latent_dim <= 0 or self.semantic_dim <= 0:
            raise ValueError("posterior dimensions must be positive")

        # This exactly mirrors B3-A1's isolated Linear mean-head construction.
        # Creating logvar heads cannot advance the caller's global torch RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.semantic_seed)
            mean_heads = [
                nn.Linear(self.latent_dim, self.semantic_dim)
                for _ in range(self.view_num)
            ]
            logvar_heads = [
                nn.Linear(self.latent_dim, self.semantic_dim)
                for _ in range(self.view_num)
            ]
            for head in logvar_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        self.mean_heads = nn.ModuleList(mean_heads)
        self.logvar_heads = nn.ModuleList(logvar_heads)

    def forward_one(self, z_v, view_idx):
        view_idx = int(view_idx)
        if not 0 <= view_idx < self.view_num:
            raise IndexError("view_idx is outside the posterior bank")
        _require_matrix(z_v, "z_v")
        if int(z_v.shape[1]) != self.latent_dim:
            raise ValueError("z_v latent dimension mismatch")

        # z_detached: [N, Dz] -- protected Native boundary.
        z_detached = z_v.detach()
        # mu_raw/logvar_raw: [N, Ds]
        mu_raw = self.mean_heads[view_idx](z_detached)
        logvar_raw = self.logvar_heads[view_idx](z_detached)
        # posterior_mu/posterior_logvar: [N, Ds]
        posterior_mu = fixed_radius_mean(mu_raw, self.radius_target)
        posterior_logvar = clamp_logvar(
            logvar_raw, self.logvar_min, self.logvar_max
        )
        return posterior_mu, posterior_logvar

    def forward(self, z_views):
        if not isinstance(z_views, (list, tuple)):
            raise TypeError("z_views must be a list or tuple")
        if len(z_views) != self.view_num:
            raise ValueError("z_views length must equal view_num")
        posterior_mu_views = []
        posterior_logvar_views = []
        for view_idx, z_v in enumerate(z_views):
            posterior_mu, posterior_logvar = self.forward_one(z_v, view_idx)
            posterior_mu_views.append(posterior_mu)
            posterior_logvar_views.append(posterior_logvar)
        return posterior_mu_views, posterior_logvar_views


class ConditionalPriorBank(nn.Module):
    """Independent p(h^v | detached side-view context) networks."""

    def __init__(self, view_num, semantic_dim, prior_seed,
                 logvar_min=POSTERIOR_LOGVAR_MIN,
                 logvar_max=POSTERIOR_LOGVAR_MAX):
        super(ConditionalPriorBank, self).__init__()
        self.view_num = int(view_num)
        self.semantic_dim = int(semantic_dim)
        self.prior_seed = int(prior_seed)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.radius_target = math.sqrt(self.semantic_dim)
        if self.view_num < 2 or self.semantic_dim <= 0:
            raise ValueError("conditional prior dimensions are invalid")

        # The independent prior seed is RNG-isolated from posterior creation.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.prior_seed)
            mean_heads = [
                nn.Linear(self.semantic_dim, self.semantic_dim)
                for _ in range(self.view_num)
            ]
            logvar_heads = [
                nn.Linear(self.semantic_dim, self.semantic_dim)
                for _ in range(self.view_num)
            ]
        self.mean_heads = nn.ModuleList(mean_heads)
        self.logvar_heads = nn.ModuleList(logvar_heads)

    def forward_one(self, side_context, view_idx):
        view_idx = int(view_idx)
        if not 0 <= view_idx < self.view_num:
            raise IndexError("view_idx is outside the prior bank")
        _require_matrix(side_context, "side_context")
        if int(side_context.shape[1]) != self.semantic_dim:
            raise ValueError("side_context semantic dimension mismatch")
        # The detach is repeated here to make the prior boundary explicit.
        context_detached = side_context.detach()
        # prior_mu_raw/prior_logvar_raw: [N, Ds]
        prior_mu_raw = self.mean_heads[view_idx](context_detached)
        prior_logvar_raw = self.logvar_heads[view_idx](context_detached)
        prior_mu = fixed_radius_mean(prior_mu_raw, self.radius_target)
        prior_logvar = clamp_logvar(
            prior_logvar_raw, self.logvar_min, self.logvar_max
        )
        return prior_mu, prior_logvar

    def forward(self, context_views):
        if not isinstance(context_views, (list, tuple)):
            raise TypeError("context_views must be a list or tuple")
        if len(context_views) != self.view_num:
            raise ValueError("context_views length must equal view_num")
        prior_mu_views = []
        prior_logvar_views = []
        for view_idx, side_context in enumerate(context_views):
            prior_mu, prior_logvar = self.forward_one(side_context, view_idx)
            prior_mu_views.append(prior_mu)
            prior_logvar_views.append(prior_logvar)
        return prior_mu_views, prior_logvar_views


def reparameterized_sample(posterior_mu, posterior_logvar, generator):
    """Draw h = mu + sigma*epsilon using only the supplied local generator."""
    _require_matrix(posterior_mu, "posterior_mu")
    _require_matrix(posterior_logvar, "posterior_logvar")
    if posterior_mu.shape != posterior_logvar.shape:
        raise ValueError("posterior mean and logvar shapes must match")
    if generator is None:
        raise ValueError("a local torch.Generator is required")
    # epsilon/semantic_sample: [N, Ds]
    epsilon = torch.randn(
        posterior_mu.shape,
        dtype=posterior_mu.dtype,
        device=posterior_mu.device,
        generator=generator,
    )
    semantic_sample = posterior_mu + torch.exp(0.5 * posterior_logvar) * epsilon
    return semantic_sample, epsilon


def normalize_semantic(semantic_sample):
    _require_matrix(semantic_sample, "semantic_sample")
    # semantic_normalized: [N, Ds]
    return F.normalize(semantic_sample, p=2, dim=1, eps=1e-12)


def deterministic_semantic(posterior_mu):
    """Deterministic evaluation representation; never draws epsilon."""
    return normalize_semantic(posterior_mu)


def sample_semantic_views(posterior_mu_views, posterior_logvar_views,
                          generator):
    if len(posterior_mu_views) != len(posterior_logvar_views):
        raise ValueError("posterior view lists must align")
    semantic_views = []
    epsilon_views = []
    sample_views = []
    for posterior_mu, posterior_logvar in zip(
            posterior_mu_views, posterior_logvar_views):
        sample, epsilon = reparameterized_sample(
            posterior_mu, posterior_logvar, generator
        )
        sample_views.append(sample)
        epsilon_views.append(epsilon)
        semantic_views.append(normalize_semantic(sample))
    return semantic_views, sample_views, epsilon_views


def uniform_semantic_distortion(semantic_views, temperature):
    """Thin B5 wrapper around frozen B3-A1 uniform symmetric InfoNCE."""
    return uniform_cross_view_infonce(semantic_views, temperature)


def analytic_kl_diag_gaussian(mean_q, logvar_q, mean_p, logvar_p):
    """Analytic KL(q||p), summed over the last diagonal-Gaussian dimension."""
    for value, name in (
            (mean_q, "mean_q"), (logvar_q, "logvar_q"),
            (mean_p, "mean_p"), (logvar_p, "logvar_p")):
        _require_matrix(value, name)
    if not (mean_q.shape == logvar_q.shape == mean_p.shape == logvar_p.shape):
        raise ValueError("all Gaussian parameters must have equal [N,D] shape")
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    # kl_per_sample: [N]
    kl_per_sample = 0.5 * torch.sum(
        logvar_p - logvar_q
        + (var_q + (mean_q - mean_p).pow(2)) / var_p
        - 1.0,
        dim=-1,
    )
    return kl_per_sample


def isotropic_kl(posterior_mu, posterior_logvar):
    """KL(q||N(0,I)), summed over semantic dimensions."""
    _require_matrix(posterior_mu, "posterior_mu")
    _require_matrix(posterior_logvar, "posterior_logvar")
    if posterior_mu.shape != posterior_logvar.shape:
        raise ValueError("posterior mean and logvar shapes must match")
    return 0.5 * torch.sum(
        torch.exp(posterior_logvar)
        + posterior_mu.pow(2)
        - 1.0
        - posterior_logvar,
        dim=-1,
    )


def isotropic_kl_decomposition(posterior_mu, posterior_logvar):
    """Return exact per-sample total, mean, and variance KL contributions."""
    mean_contribution = 0.5 * torch.sum(posterior_mu.pow(2), dim=-1)
    variance_contribution = 0.5 * torch.sum(
        torch.exp(posterior_logvar) - 1.0 - posterior_logvar,
        dim=-1,
    )
    return {
        "total": mean_contribution + variance_contribution,
        "mean": mean_contribution,
        "variance": variance_contribution,
    }


def conditional_kl_decomposition(mean_q, logvar_q, mean_p, logvar_p):
    """Exact conditional KL split into mean-mismatch and variance terms."""
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    mean_contribution = 0.5 * torch.sum(
        (mean_q - mean_p).pow(2) / var_p, dim=-1
    )
    variance_contribution = 0.5 * torch.sum(
        logvar_p - logvar_q + var_q / var_p - 1.0, dim=-1
    )
    return {
        "total": mean_contribution + variance_contribution,
        "mean_mismatch": mean_contribution,
        "variance": variance_contribution,
    }


def per_dimension_rate(kl_per_sample, semantic_dim):
    if not torch.is_tensor(kl_per_sample) or kl_per_sample.ndim != 1:
        raise ValueError("kl_per_sample must have shape [N]")
    semantic_dim = int(semantic_dim)
    if semantic_dim <= 0:
        raise ValueError("semantic_dim must be positive")
    return kl_per_sample / semantic_dim


def build_conditional_context(posterior_mu_views, radius_target):
    """Mean exactly V-1 normalized side means, excluding each target view."""
    if not isinstance(posterior_mu_views, (list, tuple)):
        raise TypeError("posterior_mu_views must be a list or tuple")
    view_num = len(posterior_mu_views)
    if view_num < 2:
        raise ValueError("conditional context requires at least two views")
    first = posterior_mu_views[0]
    _require_matrix(first, "posterior_mu_views[0]")
    for value in posterior_mu_views:
        _require_matrix(value, "posterior_mu_view")
        if value.shape != first.shape:
            raise ValueError("posterior mean views must share shape")
    radius_target = float(radius_target)
    if not math.isfinite(radius_target) or radius_target <= 0.0:
        raise ValueError("radius_target must be positive and finite")
    unit_mean_views = [value / radius_target for value in posterior_mu_views]
    context_views = []
    for target_view in range(view_num):
        side_views = [
            unit_mean_views[side_view]
            for side_view in range(view_num)
            if side_view != target_view
        ]
        # side_context: [N, Ds], with target excluded and prior-path detached.
        side_context = torch.stack(side_views, dim=0).mean(dim=0).detach()
        context_views.append(side_context)
    return context_views


def conditional_rate_views(posterior_mu_views, posterior_logvar_views,
                           prior_mu_views, prior_logvar_views):
    view_num = len(posterior_mu_views)
    if not (
            len(posterior_logvar_views) == view_num
            and len(prior_mu_views) == view_num
            and len(prior_logvar_views) == view_num):
        raise ValueError("Gaussian view lists must align")
    per_view = []
    for view_idx in range(view_num):
        per_view.append(analytic_kl_diag_gaussian(
            posterior_mu_views[view_idx],
            posterior_logvar_views[view_idx],
            prior_mu_views[view_idx],
            prior_logvar_views[view_idx],
        ))
    # rate_per_sample_view: [N,V]
    return torch.stack(per_view, dim=1)


def isotropic_rate_views(posterior_mu_views, posterior_logvar_views):
    if len(posterior_mu_views) != len(posterior_logvar_views):
        raise ValueError("posterior view lists must align")
    # rate_per_sample_view: [N,V]
    return torch.stack([
        isotropic_kl(mean, logvar)
        for mean, logvar in zip(posterior_mu_views, posterior_logvar_views)
    ], dim=1)


def total_rate_distortion_loss(distortion_loss, rate_loss_per_dim, beta,
                               rate_training_enabled):
    if not torch.is_tensor(distortion_loss) or distortion_loss.ndim != 0:
        raise ValueError("distortion_loss must be scalar")
    if not torch.is_tensor(rate_loss_per_dim) or rate_loss_per_dim.ndim != 0:
        raise ValueError("rate_loss_per_dim must be scalar")
    effective_beta = float(beta) if bool(rate_training_enabled) else 0.0
    return distortion_loss + effective_beta * rate_loss_per_dim


def shuffle_context_views(context_views, shuffle_seed):
    """Deterministically shuffle only the sample dimension of all contexts."""
    if not context_views:
        raise ValueError("context_views must not be empty")
    sample_num = int(context_views[0].shape[0])
    generator = torch.Generator(device=context_views[0].device)
    generator.manual_seed(int(shuffle_seed))
    permutation = torch.randperm(
        sample_num, generator=generator, device=context_views[0].device
    )
    return [context[permutation] for context in context_views]


def posterior_mu_radius_audit(posterior_mu_views, radius_target):
    norms = torch.cat([
        torch.linalg.vector_norm(value.detach(), ord=2, dim=1)
        for value in posterior_mu_views
    ])
    error = torch.abs(norms - float(radius_target))
    return {
        "posterior_mu_l2_mean": float(norms.mean().item()),
        "posterior_mu_l2_std": float(norms.std(unbiased=False).item()),
        "posterior_mu_l2_min": float(norms.min().item()),
        "posterior_mu_l2_max": float(norms.max().item()),
        "posterior_mu_radius_target": float(radius_target),
        "posterior_mu_fixed_radius_max_abs_error": float(error.max().item()),
        "posterior_mu_fixed_radius_pass": bool(error.max().item() < 1e-4),
    }


def logvar_audit(logvar_views, minimum=POSTERIOR_LOGVAR_MIN,
                 maximum=POSTERIOR_LOGVAR_MAX):
    flat = torch.cat([value.detach().reshape(-1) for value in logvar_views])
    at_min = float((flat <= float(minimum)).float().mean().item())
    at_max = float((flat >= float(maximum)).float().mean().item())
    result = {
        "posterior_logvar_mean": float(flat.mean().item()),
        "posterior_logvar_std": float(flat.std(unbiased=False).item()),
        "posterior_logvar_min": float(flat.min().item()),
        "posterior_logvar_max": float(flat.max().item()),
        "posterior_logvar_at_min_fraction": at_min,
        "posterior_logvar_at_max_fraction": at_max,
        "logvar_mean": float(flat.mean().item()),
        "logvar_std": float(flat.std(unbiased=False).item()),
        "logvar_min": float(flat.min().item()),
        "logvar_max": float(flat.max().item()),
        "logvar_at_min_fraction": at_min,
        "logvar_at_max_fraction": at_max,
    }
    result["posterior_logvar_finite_pass"] = bool(torch.isfinite(flat).all().item())
    result["POSTERIOR_LOGVAR_SATURATION_WARNING"] = bool(
        at_min > 0.90 or at_max > 0.90
    )
    return result


def rate_distribution_audit(rate_per_sample_view):
    if not torch.is_tensor(rate_per_sample_view) or rate_per_sample_view.ndim != 2:
        raise ValueError("rate_per_sample_view must have shape [N,V]")
    flat = rate_per_sample_view.detach().reshape(-1)
    quantiles = torch.quantile(
        flat,
        torch.tensor(
            [0.50, 0.75, 0.90, 0.95, 0.99],
            dtype=flat.dtype,
            device=flat.device,
        ),
    )
    mean = float(flat.mean().item())
    return {
        "rate_mean": mean,
        "rate_std": float(flat.std(unbiased=False).item()),
        "rate_min": float(flat.min().item()),
        "rate_max": float(flat.max().item()),
        "rate_p50": float(quantiles[0].item()),
        "rate_p75": float(quantiles[1].item()),
        "rate_p90": float(quantiles[2].item()),
        "rate_p95": float(quantiles[3].item()),
        "rate_p99": float(quantiles[4].item()),
        "rate_max_to_mean_ratio": float(flat.max().item() / max(abs(mean), 1e-12)),
    }


def gradient_audit(module):
    squared_norm = 0.0
    finite = True
    nonzero = False
    for parameter in module.parameters():
        if parameter.grad is None:
            finite = False
            continue
        finite = bool(finite and torch.isfinite(parameter.grad).all().item())
        squared_norm += float(parameter.grad.detach().pow(2).sum().item())
        nonzero = bool(nonzero or parameter.grad.detach().abs().sum().item() > 0.0)
    return {
        "norm": math.sqrt(squared_norm),
        "finite": bool(finite),
        "nonzero": bool(nonzero),
    }


def isotropic_directional_null_audit(
        semantic_dim=10, sample_num=17, audit_seed=20260815):
    """Diagnose the fixed-radius isotropic arm's directional-null role."""
    semantic_dim = int(semantic_dim)
    sample_num = int(sample_num)
    if semantic_dim <= 0 or sample_num <= 0:
        raise ValueError("audit dimensions must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(audit_seed))
    raw_mu = torch.randn(
        (sample_num, semantic_dim),
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    radius_target = math.sqrt(semantic_dim)
    posterior_mu = fixed_radius_mean(raw_mu, radius_target)
    zero_logvar = torch.zeros_like(posterior_mu, requires_grad=True)
    zero_logvar_rate = isotropic_kl(posterior_mu, zero_logvar).mean()
    raw_mu_gradient, zero_logvar_gradient = torch.autograd.grad(
        zero_logvar_rate, (raw_mu, zero_logvar)
    )

    nonzero_logvar = torch.full_like(
        posterior_mu.detach(), 0.5, requires_grad=True
    )
    nonzero_logvar_rate = isotropic_kl(
        posterior_mu.detach(), nonzero_logvar
    ).mean()
    nonzero_logvar_gradient = torch.autograd.grad(
        nonzero_logvar_rate, nonzero_logvar
    )[0]

    direction_grad_norm = float(
        torch.linalg.vector_norm(raw_mu_gradient).item()
    )
    direction_grad_max_abs = float(raw_mu_gradient.abs().max().item())
    zero_logvar_grad_norm = float(
        torch.linalg.vector_norm(zero_logvar_gradient).item()
    )
    nonzero_logvar_grad_norm = float(
        torch.linalg.vector_norm(nonzero_logvar_gradient).item()
    )
    return {
        "isotropic_fixed_radius_directional_null": True,
        "isotropic_role": "fixed_radius_directional_null_control",
        "isotropic_mu_direction_grad_norm": direction_grad_norm,
        "isotropic_mu_direction_grad_max_abs": direction_grad_max_abs,
        "isotropic_logvar_zero_grad_norm": zero_logvar_grad_norm,
        "isotropic_logvar_nonzero_grad_norm": nonzero_logvar_grad_norm,
        "B5_A0_ISOTROPIC_DIRECTIONAL_NULL_CONFIRMED": bool(
            direction_grad_norm < 1e-10
        ),
        "isotropic_zero_logvar_stationary_confirmed": bool(
            zero_logvar_grad_norm < 1e-10
        ),
        "isotropic_nonzero_logvar_active_confirmed": bool(
            nonzero_logvar_grad_norm > 0.0
        ),
    }


def analytic_kl_reference_check(check_seed=20260815):
    """Compare B5 analytic KL against torch.distributions in float64."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(check_seed))
    mean_q = torch.randn((17, 10), generator=generator, dtype=torch.float64)
    logvar_q = torch.empty((17, 10), dtype=torch.float64).uniform_(
        -3.0, 1.0, generator=generator
    )
    mean_p = torch.randn((17, 10), generator=generator, dtype=torch.float64)
    logvar_p = torch.empty((17, 10), dtype=torch.float64).uniform_(
        -3.0, 1.0, generator=generator
    )
    actual = analytic_kl_diag_gaussian(mean_q, logvar_q, mean_p, logvar_p)
    q = torch.distributions.Independent(
        torch.distributions.Normal(mean_q, torch.exp(0.5 * logvar_q)), 1
    )
    p = torch.distributions.Independent(
        torch.distributions.Normal(mean_p, torch.exp(0.5 * logvar_p)), 1
    )
    reference = torch.distributions.kl_divergence(q, p)
    conditional_error = float(torch.max(torch.abs(actual - reference)).item())

    iso_actual = isotropic_kl(mean_q, logvar_q)
    p0 = torch.distributions.Independent(
        torch.distributions.Normal(
            torch.zeros_like(mean_q), torch.ones_like(mean_q)
        ),
        1,
    )
    iso_reference = torch.distributions.kl_divergence(q, p0)
    isotropic_error = float(
        torch.max(torch.abs(iso_actual - iso_reference)).item()
    )
    max_error = max(conditional_error, isotropic_error)
    return {
        "analytic_kl_check_pass": bool(max_error < 1e-6),
        "analytic_kl_reference_max_abs_error": max_error,
        "analytic_kl_conditional_max_abs_error": conditional_error,
        "analytic_kl_isotropic_max_abs_error": isotropic_error,
    }
