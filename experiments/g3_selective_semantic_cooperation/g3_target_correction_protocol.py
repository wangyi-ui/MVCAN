"""Frozen-utility target primitives for G3-A0.

The functions in this module construct stop-gradient semantic targets only.
They do not load labels, fit Utility, own an optimizer, or update a model.
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from experiments.b7_sparse_supervision import b7_sparse_anchor_protocol as b7_protocol
from irv.b4_information_utility import tensor_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STAGE = "G3-A0"
DATASET_NAME = "Caltech-6V"
SAMPLE_NUM = 1400
VIEW_NUM = 6
CLASS_NUM = 7
LATENT_DIM = 10
TOP_K = 3
MODEL_SEED = 20
CONTROL_SEED = 20
T1 = 2
T2 = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.0001
COOPERATION_LAMBDA = 0.01
PROBABILITY_ATOL = 1e-6
ARMS = ("BASE", "U_CORRECTION", "SHUFFLED_U_CORRECTION")
EXPECTED_U_SHA256 = b7_protocol.EXPECTED_U_SHA256
EXPECTED_UTILITY_FILE_SHA256 = b7_protocol.EXPECTED_UTILITY_FILE_SHA256
DEFAULT_D2_DIR = b7_protocol.DEFAULT_D2_DIR
DEFAULT_BACKBONE_DIR = (
    REPOSITORY_ROOT / "outputs/d1_caltech6v/snr2p5_k3_seed20/models"
)
DEFAULT_DATA_PATH = REPOSITORY_ROOT / "data/Caltech.mat"
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "outputs/g3_selective_semantic_cooperation"
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def resolve_path(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def display_path(path):
    value = resolve_path(path).resolve()
    try:
        return str(value.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(value)


def load_frozen_u_only(d2_dir=DEFAULT_D2_DIR):
    """Load the closed D2 artifact without exposing a refit or oracle path."""
    return b7_protocol.load_frozen_u_only(d2_dir)


def numpy_to_torch_boundary(value, dtype=None, device=None):
    """Make a writable copy only when a read-only NumPy array crosses to Torch."""
    boundary_value = value
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        boundary_value = np.array(value, copy=True, order="C")
    tensor = torch.as_tensor(boundary_value, dtype=dtype, device=device)
    return tensor.detach()


def compute_rho(utility):
    """Compute the preregistered rho_i = u_(3) - u_(4), with shape [N]."""
    values = np.asarray(utility, dtype=np.float64)
    _require(
        values.ndim == 2
        and values.shape[1] == VIEW_NUM
        and np.isfinite(values).all(),
        "Utility must be finite with shape [N,6]",
    )
    _require(
        float(values.min()) >= 0.0 and float(values.max()) <= 1.0,
        "Utility must lie in [0,1]",
    )
    descending = np.sort(values, axis=1)[:, ::-1]
    rho = np.ascontiguousarray(descending[:, TOP_K - 1] - descending[:, TOP_K])
    _require(
        rho.shape == (values.shape[0],)
        and np.isfinite(rho).all()
        and float(rho.min()) >= 0.0
        and float(rho.max()) <= 1.0,
        "rho boundary mismatch",
    )
    return rho


def build_utility_policy(utility, arm, control_seed=CONTROL_SEED):
    """Build real or full-row-shuffled U, Top-3 admission, and rho."""
    arm = str(arm).upper()
    _require(arm in ARMS, "unknown G3 arm")
    values = np.asarray(utility, dtype=np.float64)
    _require(
        values.ndim == 2
        and values.shape[0] > 1
        and values.shape[1] == VIEW_NUM
        and np.isfinite(values).all(),
        "frozen Utility must have shape [N,6]",
    )
    identity = np.arange(values.shape[0], dtype=np.int64)
    original_top3 = b7_protocol.frozen_top3_mask(values)
    original_rho = compute_rho(values)

    if arm == "SHUFFLED_U_CORRECTION":
        permutation = b7_protocol.deterministic_permutation(
            values.shape[0], control_seed
        )
        selected_utility = np.ascontiguousarray(values[permutation], dtype=np.float64)
    else:
        permutation = identity
        selected_utility = np.ascontiguousarray(values, dtype=np.float64)

    admission = b7_protocol.frozen_top3_mask(selected_utility)
    rho = compute_rho(selected_utility)
    changed_row_count = int(np.any(selected_utility != values, axis=1).sum())
    changed_top3_row_count = int(np.any(admission != original_top3, axis=1).sum())

    _require(
        admission.shape == values.shape
        and np.all(admission.sum(axis=1) == TOP_K),
        "every utility admission row must contain exactly three views",
    )
    if arm == "SHUFFLED_U_CORRECTION":
        _require(
            np.array_equal(selected_utility, values[permutation]),
            "SHUFFLED_U must permute complete Utility rows",
        )
        _require(
            np.array_equal(np.sort(rho), np.sort(original_rho)),
            "SHUFFLED_U changed the rho multiset",
        )
        _require(
            np.array_equal(admission.sum(axis=0), original_top3.sum(axis=0)),
            "SHUFFLED_U changed Top-3 column counts",
        )
        _require(
            changed_row_count > 0 and changed_top3_row_count > 0,
            "SHUFFLED_U control is not informative",
        )

    selected_utility.setflags(write=False)
    admission.setflags(write=False)
    rho.setflags(write=False)
    permutation.setflags(write=False)
    return {
        "arm": arm,
        "utility": selected_utility,
        "admission": admission,
        "rho": rho,
        "row_permutation": permutation,
        "changed_row_count": changed_row_count,
        "changed_top3_row_count": changed_top3_row_count,
        "rho_multiset_preserved": bool(
            np.array_equal(np.sort(rho), np.sort(original_rho))
        ),
        "top3_column_counts_preserved": bool(
            np.array_equal(admission.sum(axis=0), original_top3.sum(axis=0))
        ),
    }


def validate_alignment_matrices(alignment_matrices, class_num=CLASS_NUM):
    matrices = np.asarray(alignment_matrices)
    class_num = int(class_num)
    _require(
        matrices.shape == (VIEW_NUM, class_num, class_num)
        and np.isfinite(matrices).all(),
        "alignment matrices must have shape [6,K,K]",
    )
    expected = np.ones((VIEW_NUM, class_num), dtype=np.int64)
    _require(
        np.all((matrices == 0) | (matrices == 1))
        and np.array_equal(matrices.sum(axis=1), expected)
        and np.array_equal(matrices.sum(axis=2), expected),
        "each alignment matrix must be a complete permutation",
    )
    return True


def align_detached_q(q_local, alignment_matrices):
    """Map local q to global coordinates as q_v @ M_v.T, with no gradient."""
    _require(torch.is_tensor(q_local), "q_local must be a Torch tensor")
    _require(
        q_local.ndim == 3
        and q_local.shape[1] == VIEW_NUM
        and q_local.shape[2] == CLASS_NUM,
        "q_local must have shape [N,6,7]",
    )
    validate_alignment_matrices(alignment_matrices)
    q_detached = q_local.detach()
    matrices = numpy_to_torch_boundary(
        alignment_matrices,
        dtype=q_detached.dtype,
        device=q_detached.device,
    )
    aligned_q = torch.stack(
        [
            torch.mm(q_detached[:, view_id, :], matrices[view_id].t())
            for view_id in range(VIEW_NUM)
        ],
        dim=1,
    ).detach()
    _require(
        tuple(aligned_q.shape) == tuple(q_detached.shape)
        and bool(torch.isfinite(aligned_q).all().item())
        and bool(
            torch.allclose(
                aligned_q.sum(dim=2),
                q_detached.sum(dim=2),
                rtol=0.0,
                atol=PROBABILITY_ATOL,
            )
        ),
        "aligned q probability boundary mismatch",
    )
    _require(
        not q_detached.requires_grad
        and not aligned_q.requires_grad
        and aligned_q.grad_fn is None
        and not matrices.requires_grad,
        "q alignment must be stop-gradient",
    )
    return {
        "q_detached": q_detached,
        "alignment_matrices": matrices,
        "aligned_q": aligned_q,
    }


def build_high_utility_target(
    aligned_q,
    admission_mask,
    fusion_weights_used_for_p_all,
):
    """Construct P_highU using actual current-P_all view weights."""
    _require(torch.is_tensor(aligned_q), "aligned_q must be a Torch tensor")
    q = aligned_q.detach()
    sample_num = int(q.shape[0])
    _require(
        tuple(q.shape) == (sample_num, VIEW_NUM, CLASS_NUM),
        "aligned_q must have shape [N,6,7]",
    )
    admission = numpy_to_torch_boundary(
        admission_mask,
        dtype=torch.bool,
        device=q.device,
    )
    weights = numpy_to_torch_boundary(
        fusion_weights_used_for_p_all,
        dtype=q.dtype,
        device=q.device,
    )
    _require(
        tuple(admission.shape) == (sample_num, VIEW_NUM)
        and bool(torch.all(admission.sum(dim=1) == TOP_K).item()),
        "Top-3 admission boundary mismatch",
    )
    _require(
        tuple(weights.shape) == (VIEW_NUM,)
        and bool(torch.isfinite(weights).all().item())
        and bool(torch.all(weights > 0.0).item()),
        "fusion weights must be finite positive [6] values",
    )
    weighted_admission = admission.to(q.dtype) * weights.unsqueeze(0)
    denominator = weighted_admission.sum(dim=1, keepdim=True)
    _require(bool(torch.all(denominator > 0.0).item()), "empty P_highU row")
    p_high_u = (
        (q * weighted_admission.unsqueeze(2)).sum(dim=1) / denominator
    ).detach()
    _validate_probability_target(p_high_u, "P_highU")
    _require(
        not p_high_u.requires_grad and p_high_u.grad_fn is None,
        "P_highU must be stop-gradient",
    )
    return {
        "P_highU": p_high_u,
        "weighted_admission": weighted_admission.detach(),
        "denominator": denominator.squeeze(1).detach(),
    }


def _validate_probability_target(target, name):
    _require(torch.is_tensor(target) and target.ndim == 2, name + " must be [N,K]")
    _require(
        target.shape[1] == CLASS_NUM
        and bool(torch.isfinite(target).all().item())
        and float(target.min().item()) >= 0.0
        and bool(
            torch.allclose(
                target.sum(dim=1),
                torch.ones(target.shape[0], dtype=target.dtype, device=target.device),
                rtol=0.0,
                atol=PROBABILITY_ATOL,
            )
        ),
        name + " probability boundary mismatch",
    )


def build_corrected_target(p_all, p_high_u, rho):
    """Construct exactly (1-rho)*P_all + rho*P_highU and detach it."""
    _require(torch.is_tensor(p_all) and torch.is_tensor(p_high_u), "targets must be tensors")
    _require(tuple(p_all.shape) == tuple(p_high_u.shape), "target shape mismatch")
    p_all_detached = p_all.detach()
    p_high_detached = p_high_u.detach()
    rho_tensor = numpy_to_torch_boundary(
        rho,
        dtype=p_all_detached.dtype,
        device=p_all_detached.device,
    )
    _require(
        tuple(rho_tensor.shape) == (p_all_detached.shape[0],)
        and bool(torch.isfinite(rho_tensor).all().item())
        and float(rho_tensor.min().item()) >= 0.0
        and float(rho_tensor.max().item()) <= 1.0,
        "rho tensor boundary mismatch",
    )
    rho_column = rho_tensor.unsqueeze(1)
    p_util = (
        (1.0 - rho_column) * p_all_detached
        + rho_column * p_high_detached
    ).detach()
    _validate_probability_target(p_all_detached, "P_all")
    _validate_probability_target(p_high_detached, "P_highU")
    _validate_probability_target(p_util, "P_util")
    _require(
        not p_util.requires_grad and p_util.grad_fn is None,
        "P_util must be stop-gradient",
    )
    return p_util


def select_training_target(arm, p_all, p_high_u=None, rho=None):
    """Select the arm target; BASE is an exact object-identity branch."""
    arm = str(arm).upper()
    _require(arm in ARMS, "unknown G3 arm")
    _require(torch.is_tensor(p_all), "P_all must be a Torch tensor")
    _require(not p_all.requires_grad and p_all.grad_fn is None, "P_all must be detached")
    if arm == "BASE":
        return p_all
    _require(p_high_u is not None and rho is not None, "correction inputs are missing")
    return build_corrected_target(p_all, p_high_u, rho)


def build_local_targets(p_train, alignment_matrices):
    """Map the global target to each local coordinate as P_train @ M_v."""
    _validate_probability_target(p_train, "P_train")
    validate_alignment_matrices(alignment_matrices)
    matrices = numpy_to_torch_boundary(
        alignment_matrices,
        dtype=p_train.dtype,
        device=p_train.device,
    )
    local_targets = tuple(
        torch.mm(p_train, matrices[view_id]).detach()
        for view_id in range(VIEW_NUM)
    )
    _require(
        all(
            tuple(target.shape) == tuple(p_train.shape)
            and not target.requires_grad
            and target.grad_fn is None
            for target in local_targets
        ),
        "local targets must be detached [N,K] tensors",
    )
    return local_targets


def build_training_dataset(view_tensors, sample_ids_full, p_train):
    """Create X1..X6, sample_id, P_train items for shuffled continuation."""
    _require(len(view_tensors) == VIEW_NUM, "G3 requires six view tensors")
    sample_ids = numpy_to_torch_boundary(
        sample_ids_full,
        dtype=torch.long,
        device=p_train.device,
    )
    sample_num = int(p_train.shape[0])
    _require(
        tuple(sample_ids.shape) == (sample_num,)
        and torch.equal(
            sample_ids,
            torch.arange(sample_num, dtype=torch.long, device=p_train.device),
        ),
        "sample_ids_full must be arange(N)",
    )
    _require(
        all(int(view.shape[0]) == sample_num for view in view_tensors),
        "view/sample target row mismatch",
    )
    return TensorDataset(*view_tensors, sample_ids, p_train)


def batch_sample_target_alignment_pass(sample_ids_batch, p_batch, p_train):
    """Verify shuffled dataset rows carry their explicit ID and target together."""
    ids = sample_ids_batch.to(device=p_train.device, dtype=torch.long)
    return bool(torch.equal(p_batch, p_train.index_select(0, ids)))


def target_diagnostics(p_all, p_high_u, p_util, rho):
    """Return the preregistered label-free scalar diagnostics."""
    for name, target in (
        ("P_all", p_all),
        ("P_highU", p_high_u),
        ("P_util", p_util),
    ):
        _validate_probability_target(target, name)
    rho_array = np.asarray(rho, dtype=np.float64)
    _require(
        rho_array.shape == (p_all.shape[0],) and np.isfinite(rho_array).all(),
        "rho diagnostic boundary mismatch",
    )

    def entropy_mean(target):
        values = target.detach()
        tiny = torch.finfo(values.dtype).tiny
        entropy = -torch.sum(values * torch.log(torch.clamp(values, min=tiny)), dim=1)
        return float(entropy.mean().item())

    row_mass_error = float(
        torch.max(torch.abs(p_util.sum(dim=1) - 1.0)).item()
    )
    l1_high_all = torch.abs(p_high_u - p_all).sum(dim=1).mean()
    l1_util_all = torch.abs(p_util - p_all).sum(dim=1).mean()
    diagnostics = {
        "rho_mean": float(np.mean(rho_array)),
        "rho_std": float(np.std(rho_array)),
        "rho_p10": float(np.percentile(rho_array, 10)),
        "rho_p50": float(np.percentile(rho_array, 50)),
        "rho_p90": float(np.percentile(rho_array, 90)),
        "rho_max": float(np.max(rho_array)),
        "P_all_entropy_mean": entropy_mean(p_all),
        "P_highU_entropy_mean": entropy_mean(p_high_u),
        "P_util_entropy_mean": entropy_mean(p_util),
        "mean_L1_P_highU_minus_P_all": float(l1_high_all.item()),
        "mean_L1_P_util_minus_P_all": float(l1_util_all.item()),
        "target_row_mass_error": row_mass_error,
        "target_min": float(p_util.min().item()),
        "target_max": float(p_util.max().item()),
    }
    _require(
        all(math.isfinite(value) for value in diagnostics.values()),
        "target diagnostics must be finite",
    )
    return diagnostics


def target_hashes(p_all, alignment_matrices, q_detached, aligned_q, p_high_u, rho, p_util):
    """Hash each refresh tensor using the existing logical content hash."""
    return {
        "P_all_sha256": tensor_sha256(p_all),
        "M_sha256": tensor_sha256(alignment_matrices),
        "q_sha256": tensor_sha256(q_detached),
        "aligned_q_sha256": tensor_sha256(aligned_q),
        "P_highU_sha256": tensor_sha256(p_high_u),
        "rho_sha256": tensor_sha256(rho),
        "P_util_sha256": tensor_sha256(p_util),
    }
