"""Detached directional cyclic utility with frozen numerical semantics."""

from types import MappingProxyType

import numpy as np
import torch

from . import action_space as _actions


PROBABILITY_ATOL = 1e-5
BOUND_ATOL = 1e-7


def _validate_input(q_aligned):
    if not isinstance(q_aligned, (torch.Tensor, np.ndarray)):
        raise TypeError("q_aligned must be a torch.Tensor or numpy.ndarray")
    values = torch.as_tensor(q_aligned)
    if values.ndim != 3 or not values.is_floating_point():
        raise ValueError("q_aligned must have floating shape [N,V,K]")
    if values.shape[0] <= 0 or values.shape[1] < 2 or values.shape[2] < 2:
        raise ValueError("q_aligned must have N > 0, V >= 2, and K >= 2")
    values = values.detach()
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("q_aligned must contain only finite values")
    if not bool(torch.all(values >= 0.0).item()):
        raise ValueError("q_aligned must be nonnegative")
    if not bool(torch.all(values <= 1.0 + BOUND_ATOL).item()):
        raise ValueError("q_aligned exceeds the probability upper bound")
    if not bool(torch.allclose(
        values.sum(dim=-1),
        torch.ones_like(values[..., 0]),
        rtol=0.0,
        atol=PROBABILITY_ATOL,
    )):
        raise ValueError("q_aligned posterior rows must sum to one")
    return values


def compute_directional_cycle_utility(q_aligned, actions=None):
    """Compute diagnostic action validity from aligned view posteriors."""
    values = _validate_input(q_aligned)
    if actions is None:
        actions = _actions.build_directional_actions(int(values.shape[1]))
    elif not isinstance(actions, _actions.ActionSpace):
        raise TypeError("actions must be an ActionSpace")
    if actions.V != int(values.shape[1]):
        raise ValueError("action-space view count does not match q_aligned")

    generator_posteriors = []
    verifier_posteriors = []
    for generator, verifier in zip(actions.generators, actions.verifiers):
        generator_index = torch.as_tensor(
            generator, dtype=torch.long, device=values.device
        )
        verifier_index = torch.as_tensor(
            verifier, dtype=torch.long, device=values.device
        )
        generator_posteriors.append(
            values.index_select(1, generator_index).mean(dim=1)
        )
        verifier_posteriors.append(
            values.index_select(1, verifier_index).mean(dim=1)
        )

    p_gen = torch.stack(generator_posteriors, dim=1).detach()
    p_ver = torch.stack(verifier_posteriors, dim=1).detach()
    conf_gen, y_gen = torch.max(p_gen, dim=-1)
    y_ver = torch.argmax(p_ver, dim=-1)
    support_ver = torch.gather(
        p_ver, dim=-1, index=y_gen.unsqueeze(-1)
    ).squeeze(-1)
    closure = (y_ver == y_gen).detach()
    utility = (
        closure.to(dtype=values.dtype)
        * torch.sqrt(conf_gen * support_ver)
    ).detach()

    expected_shape = (int(values.shape[0]), actions.S)
    if tuple(utility.shape) != expected_shape:
        raise RuntimeError("cyclic utility output shape mismatch")
    if (
        not bool(torch.isfinite(utility).all().item())
        or not bool(torch.all(utility >= 0.0).item())
        or not bool(torch.all(utility <= 1.0 + BOUND_ATOL).item())
        or not bool(torch.all(utility[~closure] == 0.0).item())
    ):
        raise RuntimeError("cyclic utility output boundary mismatch")

    hashes = _actions.canonical_action_hashes(actions)
    return MappingProxyType({
        "U_cycle": utility,
        "y_gen": y_gen.detach(),
        "closure": closure,
        "conf_gen": conf_gen.detach(),
        "support_ver": support_ver.detach(),
        "p_gen": p_gen,
        "p_ver": p_ver,
        "actions": actions,
        "generator_membership": actions.generator_membership,
        "verifier_membership": actions.verifier_membership,
        "canonical_generator_hash": hashes["generator"],
        "canonical_verifier_hash": hashes["verifier"],
    })
