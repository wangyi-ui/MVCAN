"""Runtime-dimensional directional cycle utility with frozen C0 semantics."""

from types import MappingProxyType

import numpy as np
import torch

from irv.b4_information_utility import tensor_sha256

from .action_space import ActionSpace, build_action_space


PROBABILITY_ATOL = 1e-5
BOUND_ATOL = 1e-7


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def action_membership_arrays(action_space):
    """Return rectangular [S,V] generator/verifier membership arrays."""
    if not isinstance(action_space, ActionSpace):
        raise TypeError("action_space must be an ActionSpace")
    generator = np.zeros((action_space.S, action_space.V), dtype=np.int64)
    verifier = np.zeros_like(generator)
    for action_id, (group, complement) in enumerate(
        zip(action_space.generator_subsets, action_space.verifier_subsets)
    ):
        generator[action_id, list(group)] = 1
        verifier[action_id, list(complement)] = 1
    _require(
        np.array_equal(generator + verifier, np.ones_like(generator)),
        "action membership complement mismatch",
    )
    return generator, verifier


def canonical_action_hashes(action_space):
    generator, verifier = action_membership_arrays(action_space)
    return MappingProxyType({
        "generator_membership_tensor_sha256": tensor_sha256(generator),
        "verifier_membership_tensor_sha256": tensor_sha256(verifier),
    })


def build_cycle_utility(q_aligned, action_space=None):
    """Compute frozen C0 U_cycle for arbitrary N,V,K.

    Each generator and verifier posterior is its own arithmetic mean.  This is
    intentionally unchanged for odd V, where action cardinalities differ.
    """
    source = torch.as_tensor(q_aligned)
    _require(
        source.ndim == 3
        and source.is_floating_point()
        and source.shape[0] > 0
        and source.shape[1] >= 2
        and source.shape[2] >= 2,
        "q_aligned must have floating shape [N,V,K]",
    )
    values = source.detach()
    _require(
        bool(torch.isfinite(values).all().item())
        and bool(torch.all(values >= 0.0).item())
        and bool(torch.all(values <= 1.0 + BOUND_ATOL).item()),
        "q_aligned finite/range boundary mismatch",
    )
    _require(
        bool(torch.allclose(
            values.sum(dim=-1),
            torch.ones_like(values[..., 0]),
            rtol=0.0,
            atol=PROBABILITY_ATOL,
        )),
        "q_aligned probability mass mismatch",
    )
    actions = build_action_space(int(values.shape[1])) if action_space is None else action_space
    _require(actions.V == int(values.shape[1]), "action-space V mismatch")

    p_gen = []
    p_ver = []
    for generator, verifier in zip(
        actions.generator_subsets, actions.verifier_subsets
    ):
        generator_index = torch.as_tensor(
            generator, dtype=torch.long, device=values.device
        )
        verifier_index = torch.as_tensor(
            verifier, dtype=torch.long, device=values.device
        )
        p_gen.append(values.index_select(1, generator_index).mean(dim=1))
        p_ver.append(values.index_select(1, verifier_index).mean(dim=1))
    generator_posterior = torch.stack(p_gen, dim=1).detach()
    verifier_posterior = torch.stack(p_ver, dim=1).detach()
    confidence, y_gen = torch.max(generator_posterior, dim=-1)
    y_ver = torch.argmax(verifier_posterior, dim=-1)
    support = torch.gather(
        verifier_posterior, -1, y_gen.unsqueeze(-1)
    ).squeeze(-1)
    closure = (y_ver == y_gen).detach()
    utility = (
        closure.to(dtype=values.dtype) * torch.sqrt(confidence * support)
    ).detach()
    expected = (int(values.shape[0]), actions.S)
    _require(
        tuple(utility.shape) == tuple(y_gen.shape) == expected
        and bool(torch.isfinite(utility).all().item())
        and bool(torch.all(utility >= 0.0).item())
        and bool(torch.all(utility <= 1.0 + BOUND_ATOL).item())
        and bool(torch.all(utility[~closure] == 0.0).item())
        and not utility.requires_grad
        and utility.grad_fn is None,
        "U_cycle structural boundary mismatch",
    )
    generator_membership, verifier_membership = action_membership_arrays(actions)
    hashes = canonical_action_hashes(actions)
    return MappingProxyType({
        "U_cycle": utility,
        "y_gen": y_gen.detach(),
        "closure": closure,
        "conf_gen": confidence.detach(),
        "support_ver": support.detach(),
        "p_gen": generator_posterior,
        "p_ver": verifier_posterior,
        "action_space": actions,
        "generator_membership": generator_membership,
        "verifier_membership": verifier_membership,
        "action_hashes": hashes,
        "odd_view_mean_semantics": True,
        "cardinality_correction_used": False,
    })
