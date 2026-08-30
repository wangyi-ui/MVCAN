"""Parameter-free memory-specific utility primitives for E4-A1."""

import numpy as np
import torch
import torch.nn.functional as F

from experiments.e4_semantic_memory_bank.semantic_memory_bank import (
    CLASS_NUM,
    LABELED_NUM,
    VIEW_NUM,
    build_classwise_matched_shuffle,
    validate_sparse_labels,
)


NUMERICAL_TOLERANCE = 1e-6
E4A1_ARMS = (
    "ALL_MEMORY",
    "R_MEMORY",
    "S_MEMORY",
    "SHUFFLED_S_MEMORY",
    "U_RS_MEMORY",
    "SHUFFLED_U_RS_MEMORY",
    "ORACLE_CLEAN_MEMORY",
)
NORMAL_E4A1_ARMS = E4A1_ARMS[:-1]


def _frozen_tensor(value, *, dtype=None, device=None):
    if torch.is_tensor(value):
        tensor = value.detach()
        return tensor.to(
            dtype=tensor.dtype if dtype is None else dtype,
            device=tensor.device if device is None else device,
        ).detach()
    array = np.array(value, copy=True, order="C")
    return torch.as_tensor(array, dtype=dtype, device=device).detach()


@torch.no_grad()
def build_sample_anchors(h_labeled):
    """Return one unweighted, all-view semantic anchor per labeled sample."""
    values = _frozen_tensor(h_labeled)
    if not values.is_floating_point():
        raise ValueError("h_labeled must be floating point")
    if values.shape != (LABELED_NUM, VIEW_NUM, CLASS_NUM):
        raise ValueError("h_labeled must have shape [14,6,7]")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("h_labeled must be finite")
    mean_semantic = values.mean(dim=1)
    if not bool(torch.all(torch.linalg.vector_norm(mean_semantic, dim=1) > 0.0)):
        raise ValueError("a sample anchor has zero norm")
    sample_anchor = F.normalize(mean_semantic, dim=-1).detach()
    if sample_anchor.shape != (LABELED_NUM, CLASS_NUM):
        raise RuntimeError("sample_anchor must have shape [14,7]")
    if not bool(torch.isfinite(sample_anchor).all().item()):
        raise RuntimeError("sample_anchor contains non-finite values")
    return sample_anchor


@torch.no_grad()
def build_memory_specific_semantic_usefulness(
    h_labeled,
    labels_labeled,
    numerical_tolerance=NUMERICAL_TOLERANCE,
):
    """Compute the preregistered leave-self-out semantic usefulness S_mem."""
    tolerance = float(numerical_tolerance)
    if tolerance != NUMERICAL_TOLERANCE:
        raise ValueError("E4-A1 numerical tolerance is frozen at 1e-6")
    h_values = _frozen_tensor(h_labeled)
    if not h_values.is_floating_point():
        raise ValueError("h_labeled must be floating point")
    if h_values.shape != (LABELED_NUM, VIEW_NUM, CLASS_NUM):
        raise ValueError("h_labeled must have shape [14,6,7]")
    if not bool(torch.isfinite(h_values).all().item()):
        raise ValueError("h_labeled must be finite")
    targets, _ = validate_sparse_labels(labels_labeled)
    targets = targets.to(device=h_values.device)

    sample_anchor = build_sample_anchors(h_values)
    normalized_writers = F.normalize(h_values, dim=-1)
    all_similarity = torch.einsum(
        "lvk,jk->lvj", normalized_writers, sample_anchor
    )

    positive_peer_indices = torch.empty(
        LABELED_NUM, dtype=torch.long, device=h_values.device
    )
    positive_peer_count = torch.empty(
        LABELED_NUM, dtype=torch.long, device=h_values.device
    )
    for writer_id in range(LABELED_NUM):
        peers = torch.nonzero(
            (targets == targets[writer_id])
            & (
                torch.arange(LABELED_NUM, device=h_values.device)
                != writer_id
            ),
            as_tuple=False,
        ).flatten()
        positive_peer_count[writer_id] = peers.numel()
        if peers.shape != (1,):
            raise RuntimeError(
                "every labeled writer must have exactly one same-class peer"
            )
        positive_peer_indices[writer_id] = peers[0]

    writer_rows = torch.arange(LABELED_NUM, device=h_values.device)
    positive_similarity = all_similarity[
        writer_rows, :, positive_peer_indices
    ].detach()
    positive_reference = sample_anchor[positive_peer_indices].detach()

    negative_mask = targets[:, None] != targets[None, :]
    negative_count = negative_mask.sum(dim=1)
    expected_negative_count = torch.full(
        (LABELED_NUM,), 12, dtype=torch.long, device=h_values.device
    )
    if not torch.equal(negative_count, expected_negative_count):
        raise RuntimeError(
            "every writer must have exactly twelve wrong-class anchors"
        )
    expanded_negative_mask = negative_mask[:, None, :].expand(
        LABELED_NUM, VIEW_NUM, LABELED_NUM
    )
    negative_similarity = all_similarity.masked_fill(
        ~expanded_negative_mask, -torch.inf
    )
    hard_negative_similarity = negative_similarity.max(dim=2).values.detach()
    semantic_margin = (
        positive_similarity - hard_negative_similarity
    ).detach()
    if semantic_margin.shape != (LABELED_NUM, VIEW_NUM):
        raise RuntimeError("semantic_margin must have shape [14,6]")
    if not bool(torch.isfinite(semantic_margin).all().item()):
        raise RuntimeError("semantic_margin contains non-finite values")
    margin_min = float(semantic_margin.min().item())
    margin_max = float(semantic_margin.max().item())
    if (
        margin_min < -2.0 - tolerance
        or margin_max > 2.0 + tolerance
    ):
        raise RuntimeError("semantic_margin lies outside [-2,2]")

    S_mem = ((semantic_margin + 2.0) / 4.0).detach()
    if S_mem.shape != (LABELED_NUM, VIEW_NUM):
        raise RuntimeError("S_mem must have shape [14,6]")
    if not bool(torch.isfinite(S_mem).all().item()):
        raise RuntimeError("S_mem contains non-finite values")
    S_min = float(S_mem.min().item())
    S_max = float(S_mem.max().item())
    if S_min < -tolerance or S_max > 1.0 + tolerance:
        raise RuntimeError("S_mem lies outside [0,1]")
    if S_mem.requires_grad or S_mem.grad_fn is not None:
        raise RuntimeError("S_mem did not stop gradients")

    writer_ids = torch.arange(LABELED_NUM, device=h_values.device)
    self_exclusion = bool(
        torch.all(positive_peer_indices != writer_ids).item()
    )
    positive_class_match = bool(
        torch.equal(targets[positive_peer_indices], targets)
    )
    negative_true_class_excluded = bool(
        torch.all(
            ~negative_mask[
                writer_ids,
                positive_peer_indices,
            ]
        ).item()
        and torch.all(~torch.diagonal(negative_mask)).item()
    )
    if not (
        self_exclusion
        and positive_class_match
        and negative_true_class_excluded
    ):
        raise RuntimeError("positive/negative reference isolation failed")
    return {
        "sample_anchor": sample_anchor.detach(),
        "positive_reference": positive_reference.detach(),
        "positive_peer_indices": positive_peer_indices.detach(),
        "positive_similarity": positive_similarity.detach(),
        "hard_negative_similarity": hard_negative_similarity.detach(),
        "semantic_margin": semantic_margin.detach(),
        "S_mem": S_mem.detach(),
        "negative_mask": negative_mask.detach(),
        "audit": {
            "h_labeled_shape": list(h_values.shape),
            "sample_anchor_shape": list(sample_anchor.shape),
            "positive_reference_shape": list(positive_reference.shape),
            "positive_similarity_shape": list(positive_similarity.shape),
            "hard_negative_similarity_shape": list(
                hard_negative_similarity.shape
            ),
            "semantic_margin_shape": list(semantic_margin.shape),
            "S_mem_shape": list(S_mem.shape),
            "positive_peer_count": positive_peer_count.cpu().tolist(),
            "positive_peer_indices": positive_peer_indices.cpu().tolist(),
            "positive_reference_self_exclusion_pass": self_exclusion,
            "exactly_one_positive_peer_per_writer_pass": bool(
                torch.equal(
                    positive_peer_count,
                    torch.ones_like(positive_peer_count),
                )
            ),
            "positive_reference_same_class_pass": positive_class_match,
            "negative_true_class_exclusion_pass": (
                negative_true_class_excluded
            ),
            "negative_count_per_writer": negative_count.cpu().tolist(),
            "negative_count_exactly_12_pass": True,
            "semantic_margin_min": margin_min,
            "semantic_margin_max": margin_max,
            "semantic_margin_range_pass": True,
            "S_mem_min": S_min,
            "S_mem_max": S_max,
            "S_mem_finite_range_pass": True,
            "S_mem_stop_gradient_pass": True,
            "unlabeled_GT_used_for_S": False,
            "R_used_for_sample_anchor": False,
            "oracle_used_for_sample_anchor": False,
        },
    }


@torch.no_grad()
def build_memory_information_utility(R_labeled, S_mem):
    """Return the fixed first-version formula U_mem = R_labeled * S_mem."""
    reliability = _frozen_tensor(R_labeled)
    usefulness = _frozen_tensor(
        S_mem, dtype=reliability.dtype, device=reliability.device
    )
    if reliability.shape != (LABELED_NUM, VIEW_NUM):
        raise ValueError("R_labeled must have shape [14,6]")
    if usefulness.shape != (LABELED_NUM, VIEW_NUM):
        raise ValueError("S_mem must have shape [14,6]")
    if not bool(
        torch.isfinite(reliability).all().item()
        and torch.isfinite(usefulness).all().item()
    ):
        raise ValueError("R_labeled and S_mem must be finite")
    if not bool(
        torch.all((reliability >= 0.0) & (reliability <= 1.0)).item()
        and torch.all((usefulness >= 0.0) & (usefulness <= 1.0)).item()
    ):
        raise ValueError("R_labeled and S_mem must lie within [0,1]")
    U_mem = (reliability * usefulness).detach()
    if not bool(
        torch.isfinite(U_mem).all().item()
        and torch.all((U_mem >= 0.0) & (U_mem <= 1.0)).item()
    ):
        raise RuntimeError("U_mem finite/range boundary failed")
    if U_mem.requires_grad or U_mem.grad_fn is not None:
        raise RuntimeError("U_mem did not stop gradients")
    return U_mem


@torch.no_grad()
def build_normal_e4a1_writer_weights(
    R_labeled,
    S_mem,
    U_mem,
    labels_labeled,
):
    """Build all six normal writer arms without accepting oracle data."""
    reliability = _frozen_tensor(R_labeled)
    usefulness = _frozen_tensor(
        S_mem, dtype=reliability.dtype, device=reliability.device
    )
    information = _frozen_tensor(
        U_mem, dtype=reliability.dtype, device=reliability.device
    )
    validate_sparse_labels(labels_labeled)
    for name, value in (
        ("R_labeled", reliability),
        ("S_mem", usefulness),
        ("U_mem", information),
    ):
        if value.shape != (LABELED_NUM, VIEW_NUM):
            raise ValueError(name + " must have shape [14,6]")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(name + " must be finite")

    expected_information = reliability * usefulness
    if not torch.equal(information, expected_information):
        raise ValueError("U_mem must equal R_labeled * S_mem exactly")
    S_shuffle, S_shuffle_audit = build_classwise_matched_shuffle(
        usefulness, labels_labeled
    )
    U_shuffle, U_shuffle_audit = build_classwise_matched_shuffle(
        information, labels_labeled
    )
    normal_weights = {
        "ALL_MEMORY": torch.ones_like(reliability).detach(),
        "R_MEMORY": reliability.detach(),
        "S_MEMORY": usefulness.detach(),
        "SHUFFLED_S_MEMORY": S_shuffle.detach(),
        "U_RS_MEMORY": information.detach(),
        "SHUFFLED_U_RS_MEMORY": U_shuffle.detach(),
    }
    if tuple(normal_weights) != NORMAL_E4A1_ARMS:
        raise RuntimeError("normal E4-A1 arm order mismatch")
    return normal_weights, {
        "S_matched_shuffle": S_shuffle_audit,
        "U_matched_shuffle": U_shuffle_audit,
        "U_shuffle_source": "matched class-view swap of U_mem",
        "U_shuffle_built_from_R_times_shuffled_S": False,
    }


@torch.no_grad()
def attach_oracle_writer_weights(normal_weights, oracle_clean_weights):
    """Attach the isolated diagnostic oracle after all normal arms are fixed."""
    if tuple(normal_weights) != NORMAL_E4A1_ARMS:
        raise ValueError("normal writer arm set/order mismatch")
    reference = normal_weights["R_MEMORY"]
    oracle = _frozen_tensor(
        oracle_clean_weights,
        dtype=reference.dtype,
        device=reference.device,
    )
    if oracle.shape != (LABELED_NUM, VIEW_NUM):
        raise ValueError("oracle_clean_weights must have shape [14,6]")
    if not bool(
        torch.isfinite(oracle).all().item()
        and torch.all((oracle == 0.0) | (oracle == 1.0)).item()
    ):
        raise ValueError("oracle writer weights must be finite binary values")
    writer_weights = dict(normal_weights)
    writer_weights["ORACLE_CLEAN_MEMORY"] = oracle.detach()
    if tuple(writer_weights) != E4A1_ARMS:
        raise RuntimeError("E4-A1 writer arm order mismatch")
    return writer_weights
