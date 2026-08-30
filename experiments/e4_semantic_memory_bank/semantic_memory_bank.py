"""Deterministic frozen semantic memories for the E4-A0 diagnostic."""

import numpy as np
import torch
import torch.nn.functional as F


SAMPLE_NUM = 1400
VIEW_NUM = 6
CLASS_NUM = 7
LABELED_NUM = 14
LABELS_PER_CLASS = 2

E4_ARMS = (
    "ALL_MEMORY",
    "R_MEMORY",
    "SHUFFLED_R_MEMORY",
    "ORACLE_CLEAN_MEMORY",
)


def _frozen_tensor(value, *, dtype=None, device=None):
    """Copy array inputs and detach every tensor at the memory boundary."""
    if torch.is_tensor(value):
        tensor = value.detach()
        if dtype is not None or device is not None:
            tensor = tensor.to(
                dtype=tensor.dtype if dtype is None else dtype,
                device=tensor.device if device is None else device,
            )
        return tensor.detach()
    array = np.array(value, copy=True, order="C")
    return torch.as_tensor(array, dtype=dtype, device=device).detach()


@torch.no_grad()
def validate_sparse_labels(labels_labeled):
    """Require the frozen E4-A0 protocol: fourteen labels, exactly two/class."""
    targets = _frozen_tensor(labels_labeled, dtype=torch.long)
    if targets.shape != (LABELED_NUM,):
        raise ValueError("labels_labeled must have shape [14]")
    expected_classes = torch.arange(CLASS_NUM, device=targets.device)
    observed = torch.unique(targets, sorted=True)
    counts = torch.bincount(targets, minlength=CLASS_NUM)
    if not torch.equal(observed, expected_classes):
        raise ValueError("labels_labeled must contain every class 0..6")
    if not torch.equal(
        counts, torch.full_like(counts, LABELS_PER_CLASS)
    ):
        raise ValueError("every class must have exactly two labeled samples")
    return targets.detach(), counts.detach()


@torch.no_grad()
def build_class_memory(h_labeled, labels_labeled, writer_weights):
    """Build seven normalized class memories from all fourteen sparse labels.

    Args:
        h_labeled: Frozen aligned semantic carriers with shape ``[14,6,7]``.
        labels_labeled: Sparse class IDs with shape ``[14]``.
        writer_weights: Non-negative sample-view weights with shape ``[14,6]``.

    Returns:
        A detached, finite, unit-row-norm tensor with shape ``[7,7]``.
    """
    h_values = _frozen_tensor(h_labeled)
    if not h_values.is_floating_point():
        raise ValueError("h_labeled must be floating point")
    if h_values.shape != (LABELED_NUM, VIEW_NUM, CLASS_NUM):
        raise ValueError("h_labeled must have shape [14,6,7]")
    if not bool(torch.isfinite(h_values).all().item()):
        raise ValueError("h_labeled must be finite")

    targets, _ = validate_sparse_labels(labels_labeled)
    targets = targets.to(device=h_values.device)
    weights = _frozen_tensor(
        writer_weights, dtype=h_values.dtype, device=h_values.device
    )
    if weights.shape != (LABELED_NUM, VIEW_NUM):
        raise ValueError("writer_weights must have shape [14,6]")
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("writer_weights must be finite")
    if not bool(torch.all((weights >= 0.0) & (weights <= 1.0)).item()):
        raise ValueError("writer_weights must lie within [0,1]")

    memories = []
    for class_id in range(CLASS_NUM):
        class_rows = targets == class_id
        class_weights = weights[class_rows]
        denominator = class_weights.sum()
        if not bool((denominator > 0.0).item()):
            raise ValueError("every class memory denominator must be positive")
        weighted_sum = (
            class_weights.unsqueeze(-1) * h_values[class_rows]
        ).sum(dim=(0, 1))
        class_memory = weighted_sum / denominator
        if not bool((torch.linalg.vector_norm(class_memory) > 0.0).item()):
            raise ValueError("a class memory has zero norm")
        memories.append(class_memory)

    prototypes = F.normalize(torch.stack(memories, dim=0), dim=-1).detach()
    if prototypes.shape != (CLASS_NUM, CLASS_NUM):
        raise RuntimeError("class memory output must have shape [7,7]")
    if not bool(torch.isfinite(prototypes).all().item()):
        raise RuntimeError("class memory output contains non-finite values")
    if not bool(
        torch.allclose(
            torch.linalg.vector_norm(prototypes, dim=1),
            torch.ones(CLASS_NUM, device=prototypes.device, dtype=prototypes.dtype),
        )
    ):
        raise RuntimeError("class memories are not unit norm")
    if prototypes.requires_grad or prototypes.grad_fn is not None:
        raise RuntimeError("class memories did not stop gradients")
    return prototypes


@torch.no_grad()
def build_classwise_matched_shuffle(R_labeled, labels_labeled):
    """Swap the two reliability rows inside every class, for every view."""
    values = _frozen_tensor(R_labeled)
    if not values.is_floating_point():
        values = values.to(dtype=torch.float32)
    if values.shape != (LABELED_NUM, VIEW_NUM):
        raise ValueError("R_labeled must have shape [14,6]")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("R_labeled must be finite")
    if not bool(torch.all((values >= 0.0) & (values <= 1.0)).item()):
        raise ValueError("R_labeled must lie within [0,1]")
    targets, _ = validate_sparse_labels(labels_labeled)
    targets = targets.to(device=values.device)

    shuffled = values.clone()
    class_view_sum_pass = True
    class_denominator_pass = True
    row_pairs = []
    for class_id in range(CLASS_NUM):
        class_rows = torch.nonzero(
            targets == class_id, as_tuple=False
        ).flatten()
        if class_rows.shape != (LABELS_PER_CLASS,):
            raise RuntimeError("matched shuffle requires exactly two rows/class")
        first, second = int(class_rows[0].item()), int(class_rows[1].item())
        shuffled[first] = values[second]
        shuffled[second] = values[first]
        before_view_sums = values[class_rows].sum(dim=0)
        after_view_sums = shuffled[class_rows].sum(dim=0)
        class_view_sum_pass = bool(
            class_view_sum_pass and torch.equal(before_view_sums, after_view_sums)
        )
        class_denominator_pass = bool(
            class_denominator_pass
            and torch.equal(before_view_sums.sum(), after_view_sums.sum())
        )
        row_pairs.append([first, second])

    changed_cell_count = int(torch.count_nonzero(shuffled != values).item())
    same_global_multiset = bool(
        torch.equal(
            torch.sort(values.reshape(-1)).values,
            torch.sort(shuffled.reshape(-1)).values,
        )
    )
    if not (
        shuffled.shape == values.shape
        and same_global_multiset
        and class_view_sum_pass
        and class_denominator_pass
        and changed_cell_count > 0
    ):
        raise RuntimeError("classwise matched reliability shuffle audit failed")
    return shuffled.detach(), {
        "same_shape_pass": True,
        "same_global_multiset_pass": same_global_multiset,
        "class_view_weight_sum_exact_pass": class_view_sum_pass,
        "class_total_denominator_exact_pass": class_denominator_pass,
        "changed_cell_count": changed_cell_count,
        "changed_cell_count_positive_pass": True,
        "deterministic_no_rng_pass": True,
        "class_row_pairs": row_pairs,
    }


@torch.no_grad()
def build_normal_writer_weights(R_labeled, labels_labeled):
    """Build the three non-oracle writer policies without oracle inputs."""
    values = _frozen_tensor(R_labeled)
    if not values.is_floating_point():
        values = values.to(dtype=torch.float32)
    validate_sparse_labels(labels_labeled)
    if values.shape != (LABELED_NUM, VIEW_NUM):
        raise ValueError("R_labeled must have shape [14,6]")
    shuffled, shuffle_audit = build_classwise_matched_shuffle(
        values, labels_labeled
    )
    return {
        "ALL_MEMORY": torch.ones_like(values).detach(),
        "R_MEMORY": values.detach(),
        "SHUFFLED_R_MEMORY": shuffled.detach(),
    }, shuffle_audit


@torch.no_grad()
def build_shared_query(h_sem):
    """Build the sole E4-A0 query: normalized unweighted mean across views."""
    values = _frozen_tensor(h_sem)
    if not values.is_floating_point() or values.ndim != 3:
        raise ValueError("h_sem must be a floating tensor with shape [N,6,7]")
    if values.shape[1:] != (VIEW_NUM, CLASS_NUM):
        raise ValueError("h_sem must have shape [N,6,7]")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("h_sem must be finite")
    mean_semantic = values.mean(dim=1)
    if not bool(torch.all(torch.linalg.vector_norm(mean_semantic, dim=1) > 0.0)):
        raise ValueError("a shared query has zero norm")
    return F.normalize(mean_semantic, dim=-1).detach()


@torch.no_grad()
def predict_from_memory(h_query, prototypes):
    """Return fixed prototype scores and assignments without any label input."""
    queries = _frozen_tensor(h_query)
    centers = _frozen_tensor(
        prototypes, dtype=queries.dtype, device=queries.device
    )
    if queries.ndim != 2 or queries.shape[1] != CLASS_NUM:
        raise ValueError("h_query must have shape [N,7]")
    if centers.shape != (CLASS_NUM, CLASS_NUM):
        raise ValueError("prototypes must have shape [7,7]")
    if not bool(torch.isfinite(queries).all() and torch.isfinite(centers).all()):
        raise ValueError("query/prototype inputs must be finite")
    scores = (queries @ centers.T).detach()
    if scores.shape != (queries.shape[0], CLASS_NUM):
        raise RuntimeError("prototype scores must have shape [N,7]")
    return torch.argmax(scores, dim=1).detach(), scores
