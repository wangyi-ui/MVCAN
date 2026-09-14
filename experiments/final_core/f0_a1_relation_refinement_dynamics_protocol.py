"""Fail-closed protocol helpers for the pre-GT F0-A1 dynamics audit.

F0-A1 instruments the frozen C3-B0 training path.  This module adds only
read-only trajectory observation and deterministic diagnostics; it defines no
training loss, utility, coefficient, threshold, or ground-truth evaluation.
"""

import hashlib
import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from experiments.cyclic_utility import c3_b0_relation_action_protocol as c3b0
from experiments.cyclic_utility import train_c3_b0_relation_action_pilot as c3_train
from experiments.final_core import final_core_protocol as f0
from irv.b3_audit import hash_backbone


STAGE = "F0-A1"
PURPOSE = "UTILITY_CONDITIONED_RELATION_REFINEMENT_DYNAMICS_AUDIT_PRE_GT"
ARMS = ("BASE", "TRUE_UNIFORM", "TRUE_U")
SEEDS = (20, 30, 50)
EPOCHS = 20
N = 1400
V = 6
K = 7
L = 14
NU = 1386
S = 20

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PREREGISTERED_PROTOCOL_PATH = REPOSITORY_ROOT / (
    "experiment_freeze/f0_a1_closed_loop_dynamics_preregistered_20260914/"
    "PROTOCOL.txt"
)
PREREGISTERED_PROTOCOL_SHA256 = (
    "022963687054f2b520219d555d40ee248b8fe9bcefdfd77c83cfd0721c300904"
)
PARENT_FREEZES_MANIFEST = PREREGISTERED_PROTOCOL_PATH.parent / (
    "parent_freezes_sha256.txt"
)
C3_FINAL_FREEZE = REPOSITORY_ROOT / (
    "experiment_freeze/c3_b0_multiseed_pass_20260912"
)
C3_FORMAL_OUTPUTS = C3_FINAL_FREEZE / "formal_outputs"
C3_FORMAL_MANIFEST = C3_FINAL_FREEZE / "formal_outputs_sha256.txt"

FORBIDDEN_FLAGS = (
    "new_loss_used",
    "new_loss_coefficient_used",
    "new_utility_used",
    "scalar_U_recalibration_used",
    "continuous_U_weighting_used",
    "pseudo_label_used",
    "pseudo_CE_used",
    "memory_used",
    "prototype_memory_used",
    "feature_gate_used",
    "fusion_gate_used",
    "recursive_expansion_used",
    "c4_memory_used",
    "c5_a0_training_used",
    "c5_b0_used",
)

ARTIFACT_KEYS = (
    "sample_ids",
    "labeled_ids",
    "unlabeled_ids",
    "snapshot_epoch",
    "snapshot_phase",
    "q_local_snapshots",
    "final_predictions",
    "final_model_hash",
    "frozen_U_logical_hash",
    "relation_target_logical_hash",
    "relation_balance_weights_logical_hash",
    "arm",
    "seed",
)


def _require(condition, message="F0_A1_FAIL_CLOSED"):
    if not condition:
        raise RuntimeError(message)


def validate_arm(arm):
    if arm not in ARMS:
        raise ValueError("F0-A1 arm must be one of " + str(ARMS))
    return arm


def validate_seed(seed):
    value = int(seed)
    if value not in SEEDS:
        raise ValueError("F0-A1 seed must be one of " + str(SEEDS))
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with open(Path(path), "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_preregistered_protocol():
    return f0.verify_pinned_file(
        PREREGISTERED_PROTOCOL_PATH,
        PREREGISTERED_PROTOCOL_SHA256,
        "F0_A1_PREREGISTRATION_HASH_FAIL_CLOSED",
    )


def _manifest_entries(path):
    entries = OrderedDict()
    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            value = line.strip()
            if value:
                expected, name = value.split(None, 1)
                entries[name.strip()] = expected
    _require(entries, "F0_A1_EMPTY_HASH_MANIFEST_FAIL_CLOSED")
    return entries


def verify_parent_freezes_manifest():
    """Verify the preregistered parent pins relative to the repository root."""
    records = OrderedDict()
    for name, expected in _manifest_entries(PARENT_FREEZES_MANIFEST).items():
        target = REPOSITORY_ROOT / (name[2:] if name.startswith("./") else name)
        _require(target.is_file(), "F0_A1_PARENT_FILE_MISSING_FAIL_CLOSED")
        actual = file_sha256(target)
        _require(actual == expected, "F0_A1_PARENT_HASH_FAIL_CLOSED")
        records[name] = actual
    return {
        "manifest_path": str(PARENT_FREEZES_MANIFEST),
        "manifest_file_sha256": file_sha256(PARENT_FREEZES_MANIFEST),
        "entries": dict(records),
        "all_parent_hashes_pass": True,
    }


def default_forbidden_flags():
    return OrderedDict((name, False) for name in FORBIDDEN_FLAGS)


def validate_forbidden_flags(flags):
    _require(
        isinstance(flags, dict) and set(flags) == set(FORBIDDEN_FLAGS),
        "F0_A1_FORBIDDEN_FLAG_SCHEMA_FAIL_CLOSED",
    )
    _require(
        all(flags[name] is False for name in FORBIDDEN_FLAGS),
        "F0_A1_FORBIDDEN_PATH_ENABLED_FAIL_CLOSED",
    )
    return OrderedDict((name, False) for name in FORBIDDEN_FLAGS)


def snapshot_schedule(arm):
    """Return the only preregistered epoch/phase schedule."""
    active_arm = validate_arm(arm)
    epochs = [0]
    phases = ["INITIAL"]
    for epoch in range(1, EPOCHS + 1):
        if active_arm != "BASE":
            epochs.append(epoch)
            phases.append("POST_A")
        epochs.append(epoch)
        phases.append("POST_B")
    return (
        np.ascontiguousarray(epochs, dtype=np.int64),
        np.ascontiguousarray(phases, dtype="<U7"),
    )


def _clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return value


def _tree_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left) and torch.is_tensor(right)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict) and isinstance(right, dict)
            and tuple(left) == tuple(right)
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, type(right)) and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
            and left.dtype == right.dtype and np.array_equal(left, right)
        )
    return left == right


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [value.clone() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []
        ),
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _rng_equal(left, right):
    numpy_equal = (
        left["numpy"][0] == right["numpy"][0]
        and np.array_equal(left["numpy"][1], right["numpy"][1])
        and left["numpy"][2:] == right["numpy"][2:]
    )
    return (
        left["python"] == right["python"]
        and numpy_equal
        and torch.equal(left["torch_cpu"], right["torch_cpu"])
        and len(left["torch_cuda"]) == len(right["torch_cuda"])
        and all(torch.equal(a, b) for a, b in zip(
            left["torch_cuda"], right["torch_cuda"]
        ))
    )


def _model_hash(model):
    _require(
        hasattr(model, "autoencoders"),
        "F0_A1_SNAPSHOT_MODEL_BOUNDARY_FAIL_CLOSED",
    )
    return hash_backbone(model.autoencoders)

def _backbone_modules(model):
    """Return the six frozen per-view nn.Module backbones, fail closed."""
    _require(
        hasattr(model, "autoencoders"),
        "F0_A1_SNAPSHOT_MODEL_BOUNDARY_FAIL_CLOSED",
    )
    autoencoders = model.autoencoders
    _require(
        hasattr(autoencoders, "__len__") and len(autoencoders) == V,
        "F0_A1_SNAPSHOT_BACKBONE_VIEW_COUNT_FAIL_CLOSED",
    )
    modules = tuple(autoencoders)
    _require(
        all(isinstance(module, torch.nn.Module) for module in modules),
        "F0_A1_SNAPSHOT_BACKBONE_MODULE_API_FAIL_CLOSED",
    )
    return modules


def _iter_backbone_named_parameters(model):
    """Yield stable ``viewN.*`` parameter names without using wrapper APIs."""
    for view_id, module in enumerate(_backbone_modules(model)):
        for name, parameter in module.named_parameters():
            yield "view" + str(view_id) + "." + name, parameter


def _iter_backbone_named_buffers(model):
    """Yield stable ``viewN.*`` buffer names without using wrapper APIs."""
    for view_id, module in enumerate(_backbone_modules(model)):
        for name, buffer in module.named_buffers():
            yield "view" + str(view_id) + "." + name, buffer


def _iter_backbone_named_modules(model):
    """Yield stable root/submodule names for all six view backbones."""
    for view_id, module in enumerate(_backbone_modules(model)):
        prefix = "view" + str(view_id)
        for name, child in module.named_modules():
            yield prefix if name == "" else prefix + "." + name, child


def full_data_q_local_snapshot(
    model, full_views, sample_ids, device, optimizers=(), return_audit=False,
):
    """Observe q_local[N,V,K] without changing training, model, or RNG state.

    The helper deliberately leaves ``model.training`` untouched.  RNG state is
    restored after the q-only forward so observation cannot advance any random
    stream, even if a future architecture accidentally contains a stochastic
    forward operation.
    """
    ids = np.asarray(sample_ids, dtype=np.int64)
    _require(
        ids.shape == (N,) and np.array_equal(ids, np.arange(N)),
        "F0_A1_SNAPSHOT_SAMPLE_IDS_FAIL_CLOSED",
    )
    _require(
        len(full_views) == V and all(len(view) == N for view in full_views),
        "F0_A1_SNAPSHOT_VIEW_BOUNDARY_FAIL_CLOSED",
    )
    optimizer_list = list(optimizers)
    parameters_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in _iter_backbone_named_parameters(model)
    }
    buffers_before = {
        name: buffer.detach().cpu().clone()
        for name, buffer in _iter_backbone_named_buffers(model)
    }
    modes_before = tuple(
        (name, module.training) for name, module in _iter_backbone_named_modules(model)
    )
    optimizers_before = [_clone_tree(item.state_dict()) for item in optimizer_list]
    model_hash_before = _model_hash(model)
    rng_before = _capture_rng_state()
    q_local = None
    try:
        # q_views: six [N,K] view-local posteriors. no_grad makes the
        # trajectory observational and excludes it from every backward graph.
        with torch.no_grad():
            q_views = []
            for view_id in range(V):
                x_view = torch.as_tensor(full_views[view_id]).to(device)
                latent = model.autoencoders[view_id].encoder(x_view)
                q_view = model.autoencoders[view_id].clustering(latent)
                _require(
                    tuple(q_view.shape) == (N, K),
                    "F0_A1_SNAPSHOT_Q_VIEW_SHAPE_FAIL_CLOSED",
                )
                q_views.append(q_view)
            # q_local: [N,V,K], detached float32 contiguous on CPU.
            q_local = torch.stack(q_views, dim=1).detach().to(
                device="cpu", dtype=torch.float32
            ).contiguous()
    finally:
        _restore_rng_state(rng_before)

    parameters_after = {
        name: parameter.detach().cpu().clone()
        for name, parameter in _iter_backbone_named_parameters(model)
    }
    buffers_after = {
        name: buffer.detach().cpu().clone()
        for name, buffer in _iter_backbone_named_buffers(model)
    }
    modes_after = tuple(
        (name, module.training) for name, module in _iter_backbone_named_modules(model)
    )
    optimizers_after = [_clone_tree(item.state_dict()) for item in optimizer_list]
    model_hash_after = _model_hash(model)
    rng_after = _capture_rng_state()
    checks = {
        "model_parameters_unchanged": _tree_equal(
            parameters_before, parameters_after
        ),
        "model_buffers_unchanged": _tree_equal(buffers_before, buffers_after),
        "model_aggregate_hash_equal": (
            model_hash_before["aggregate"] == model_hash_after["aggregate"]
        ),
        "model_training_flags_unchanged": modes_before == modes_after,
        "optimizer_state_unchanged": _tree_equal(
            optimizers_before, optimizers_after
        ),
        "rng_state_unchanged": _rng_equal(rng_before, rng_after),
        "snapshot_requires_grad": bool(q_local.requires_grad),
        "snapshot_grad_fn_absent": q_local.grad_fn is None,
        "snapshot_float32_cpu_contiguous": (
            q_local.dtype == torch.float32 and q_local.device.type == "cpu"
            and q_local.is_contiguous()
        ),
        "refresh_native_target_called": False,
    }
    _require(
        checks["model_parameters_unchanged"]
        and checks["model_buffers_unchanged"]
        and checks["model_aggregate_hash_equal"]
        and checks["model_training_flags_unchanged"]
        and checks["optimizer_state_unchanged"]
        and checks["rng_state_unchanged"]
        and not checks["snapshot_requires_grad"]
        and checks["snapshot_grad_fn_absent"]
        and checks["snapshot_float32_cpu_contiguous"],
        "F0_A1_SNAPSHOT_NON_INTERFERENCE_FAIL_CLOSED",
    )
    audit = {
        **checks,
        "model_hash_before": model_hash_before["aggregate"],
        "model_hash_after": model_hash_after["aggregate"],
        "q_local_shape": [N, V, K],
        "torch_no_grad_used": True,
        "model_mode_changed_for_snapshot": False,
        "RNG_restored_after_snapshot": True,
    }
    return (q_local, audit) if return_audit else q_local


def _validated_q(q_local):
    tensor = torch.as_tensor(q_local).detach().cpu()
    _require(
        tuple(tensor.shape) == (N, V, K)
        and tensor.dtype in (torch.float32, torch.float64)
        and bool(torch.isfinite(tensor).all().item()),
        "F0_A1_Q_LOCAL_BOUNDARY_FAIL_CLOSED",
    )
    return tensor


def _validate_sparse_ids(sample_ids, unlabeled_ids, labeled_ids):
    samples = np.asarray(sample_ids, dtype=np.int64)
    unlabeled = np.asarray(unlabeled_ids, dtype=np.int64)
    labeled = np.asarray(labeled_ids, dtype=np.int64)
    _require(
        samples.shape == (N,) and np.array_equal(samples, np.arange(N))
        and unlabeled.shape == (NU,) and labeled.shape == (L,)
        and np.unique(unlabeled).size == NU and np.unique(labeled).size == L
        and np.intersect1d(unlabeled, labeled).size == 0
        and np.array_equal(
            np.sort(np.concatenate((unlabeled, labeled))), samples
        ),
        "F0_A1_SPARSE_SAMPLE_ID_BOUNDARY_FAIL_CLOSED",
    )
    return samples, unlabeled, labeled


def relation_state_from_q_local(
    q_local, sample_ids, unlabeled_ids, labeled_ids,
):
    """Compute frozen C3-B0 R(Q)[Nu,L,V] by canonical sample ID."""
    q_tensor = _validated_q(q_local)
    _, unlabeled, labeled = _validate_sparse_ids(
        sample_ids, unlabeled_ids, labeled_ids
    )
    query_ids = torch.from_numpy(unlabeled)
    anchor_ids = torch.from_numpy(labeled)
    with torch.no_grad():
        relation_views = []
        for view_id in range(V):
            # q_query: [Nu,K]; q_anchor: [L,K], detached by the frozen helper.
            q_query = q_tensor[query_ids, view_id, :]
            q_anchor = q_tensor[anchor_ids, view_id, :]
            relation_views.append(c3b0.relation_probability(q_query, q_anchor))
        # relation_state: [Nu,L,V].
        relation = torch.stack(relation_views, dim=2).detach().contiguous()
    _require(
        tuple(relation.shape) == (NU, L, V)
        and not relation.requires_grad and relation.grad_fn is None,
        "F0_A1_RELATION_STATE_FAIL_CLOSED",
    )
    return relation


def diagnostic_relation_objective(
    q_local, sample_ids, unlabeled_ids, labeled_ids,
    PredRelation_true, U_cycle, relation_balance_weights_true, arm,
):
    """Thin no-grad wrapper around the frozen C3-B0 semantic objective."""
    active_arm = validate_arm(arm)
    _require(active_arm != "BASE", "F0_A1_BASE_HAS_NO_SEMANTIC_OBJECTIVE")
    q_tensor = _validated_q(q_local)
    _, unlabeled, labeled = _validate_sparse_ids(
        sample_ids, unlabeled_ids, labeled_ids
    )
    validated = c3b0.validate_action_arrays({
        "U_cycle": U_cycle,
        "unlabeled_ids": unlabeled,
        "labeled_ids": labeled,
        "PredRelation_true": PredRelation_true,
        "PredRelation_shuffle": PredRelation_true,
        "relation_balance_weights_true": relation_balance_weights_true,
        "relation_balance_weights_shuffle": relation_balance_weights_true,
    })
    query_ids = torch.from_numpy(unlabeled)
    anchor_ids = torch.from_numpy(labeled)
    # q_sample_views: six [Nu,K]; q_anchor_views: six [L,K].
    q_sample_views = [q_tensor[query_ids, view, :] for view in range(V)]
    q_anchor_views = [q_tensor[anchor_ids, view, :] for view in range(V)]
    with torch.no_grad():
        loss, audit = c3b0.relation_semantic_loss(
            q_sample_views,
            q_anchor_views,
            validated["PredRelation_true"],
            validated["U_cycle"],
            validated["relation_balance_weights_true"],
            active_arm,
        )
    return float(loss.item()), audit


def relation_drift(previous_q, current_q, sample_ids, unlabeled_ids, labeled_ids):
    previous = relation_state_from_q_local(
        previous_q, sample_ids, unlabeled_ids, labeled_ids
    )
    current = relation_state_from_q_local(
        current_q, sample_ids, unlabeled_ids, labeled_ids
    )
    return float(torch.mean(torch.abs(current - previous)).item())


def posterior_drift(previous_q, current_q):
    previous = _validated_q(previous_q)
    current = _validated_q(current_q)
    # Per [i,v], 0.5 * L1 across K; then mean across [N,V].
    return float((0.5 * torch.abs(current - previous).sum(dim=2)).mean().item())


def compute_trajectory_diagnostics(
    artifact, PredRelation_true, U_cycle, relation_balance_weights_true,
):
    """Compute deterministic per-arm pre-GT diagnostics after training."""
    arrays = load_artifact_arrays(artifact) if isinstance(
        artifact, (str, Path)
    ) else artifact
    validate_artifact_arrays(arrays)
    arm = str(np.asarray(arrays["arm"]).item())
    sample_ids = arrays["sample_ids"]
    unlabeled_ids = arrays["unlabeled_ids"]
    labeled_ids = arrays["labeled_ids"]
    snapshots = arrays["q_local_snapshots"]
    if arm == "BASE":
        native = [
            relation_drift(
                snapshots[index - 1], snapshots[index], sample_ids,
                unlabeled_ids, labeled_ids,
            )
            for index in range(1, snapshots.shape[0])
        ]
        posterior = [
            posterior_drift(snapshots[index - 1], snapshots[index])
            for index in range(1, snapshots.shape[0])
        ]
        return {
            "arm": arm,
            "semantic_phase_executed": False,
            "D_A": None,
            "D_B": None,
            "G": None,
            "native_boundary_relation_drift_secondary": native,
            "native_boundary_posterior_drift_secondary": posterior,
        }

    d_a = []
    d_b = []
    d_q_a = []
    d_q_b = []
    objective_pre = []
    objective_post = []
    gains = []
    for epoch_index in range(EPOCHS):
        previous_b = snapshots[2 * epoch_index]
        post_a = snapshots[2 * epoch_index + 1]
        post_b = snapshots[2 * epoch_index + 2]
        d_a.append(relation_drift(
            previous_b, post_a, sample_ids, unlabeled_ids, labeled_ids
        ))
        d_b.append(relation_drift(
            post_a, post_b, sample_ids, unlabeled_ids, labeled_ids
        ))
        d_q_a.append(posterior_drift(previous_b, post_a))
        d_q_b.append(posterior_drift(post_a, post_b))
        pre, _ = diagnostic_relation_objective(
            previous_b, sample_ids, unlabeled_ids, labeled_ids,
            PredRelation_true, U_cycle, relation_balance_weights_true, arm,
        )
        post, _ = diagnostic_relation_objective(
            post_a, sample_ids, unlabeled_ids, labeled_ids,
            PredRelation_true, U_cycle, relation_balance_weights_true, arm,
        )
        objective_pre.append(pre)
        objective_post.append(post)
        gains.append(pre - post)
    return {
        "arm": arm,
        "semantic_phase_executed": True,
        "D_A": d_a,
        "D_B": d_b,
        "D_Q_A_secondary": d_q_a,
        "D_Q_B_secondary": d_q_b,
        "E_pre": objective_pre,
        "E_post": objective_post,
        "G": gains,
        "G_sum": float(np.sum(gains, dtype=np.float64)),
    }


def summarize_gate3(g_u_sum_by_seed):
    _require(
        set(g_u_sum_by_seed) == set(SEEDS)
        and all(np.isfinite(g_u_sum_by_seed[seed]) for seed in SEEDS),
        "F0_A1_GATE3_SEED_SCHEMA_FAIL_CLOSED",
    )
    values = [float(g_u_sum_by_seed[seed]) for seed in SEEDS]
    mean_value = float(np.mean(values))
    positive_count = int(sum(value > 0.0 for value in values))
    return {
        "mean_seed_G_U_sum": mean_value,
        "positive_seed_count": positive_count,
        "all_epochs_positive_required": False,
        "gate3_pass": mean_value > 0.0 and positive_count >= 2,
    }


def _artifact_mapping(value):
    if isinstance(value, (str, Path)):
        return load_artifact_arrays(value)
    return value


def compare_true_u_and_uniform_trajectories(true_u_artifact, uniform_artifact):
    """Compare compatible TRUE_U and TRUE_UNIFORM relation trajectories."""
    true_u = _artifact_mapping(true_u_artifact)
    uniform = _artifact_mapping(uniform_artifact)
    validate_artifact_arrays(true_u)
    validate_artifact_arrays(uniform)
    _require(
        str(np.asarray(true_u["arm"]).item()) == "TRUE_U"
        and str(np.asarray(uniform["arm"]).item()) == "TRUE_UNIFORM",
        "F0_A1_TRAJECTORY_COMPARISON_ARM_FAIL_CLOSED",
    )
    for name in ("sample_ids", "labeled_ids", "unlabeled_ids"):
        _require(
            np.array_equal(true_u[name], uniform[name]),
            "F0_A1_TRAJECTORY_COMPARISON_IDS_FAIL_CLOSED",
        )
    for name in ("snapshot_epoch", "snapshot_phase"):
        _require(
            np.array_equal(true_u[name], uniform[name]),
            "F0_A1_TRAJECTORY_COMPARISON_SCHEDULE_FAIL_CLOSED",
        )
    q_u = true_u["q_local_snapshots"]
    q_uni = uniform["q_local_snapshots"]
    _require(
        q_u.shape == q_uni.shape == (41, N, V, K),
        "F0_A1_TRAJECTORY_COMPARISON_SHAPE_FAIL_CLOSED",
    )
    _require(
        np.array_equal(q_u[0], q_uni[0])
        and f0.logical_sha256(q_u[0]) == f0.logical_sha256(q_uni[0]),
        "F0_A1_Q0_EXACT_EQUALITY_FAIL_CLOSED",
    )
    relation_u = [relation_state_from_q_local(
        q, true_u["sample_ids"], true_u["unlabeled_ids"],
        true_u["labeled_ids"],
    ) for q in q_u]
    relation_uni = [relation_state_from_q_local(
        q, uniform["sample_ids"], uniform["unlabeled_ids"],
        uniform["labeled_ids"],
    ) for q in q_uni]
    divergence = [
        float(torch.mean(torch.abs(left - right)).item())
        for left, right in zip(relation_u, relation_uni)
    ]
    _require(divergence[0] == 0.0, "F0_A1_Q0_DIVERGENCE_FAIL_CLOSED")
    return {
        "D_U_UNI_0": divergence[0],
        "D_U_UNI_A": divergence[1::2],
        "D_U_UNI_B": divergence[2::2],
        "Q0_exact_equal": True,
        "Q0_logical_hash_equal": True,
        "sample_ids_exact_equal": True,
        "snapshot_schedule_exact_compatible": True,
    }


def _c3_reference_paths(seed, arm):
    active_seed = validate_seed(seed)
    active_arm = validate_arm(arm)
    directory = {
        "BASE": "base",
        "TRUE_UNIFORM": "true_uniform",
        "TRUE_U": "true_u",
    }[active_arm]
    root = C3_FORMAL_OUTPUTS / (
        "c3_b0_relation_action_pilot_seed" + str(active_seed) + "_" + directory
    )
    return root / "audit.json", root / "final_predictions.npz"


def load_frozen_c3_b0_reference(seed, arm):
    """Load one of 3x3 references only from the canonical C3-B0 freeze."""
    active_seed = validate_seed(seed)
    active_arm = validate_arm(arm)
    f0.verify_pinned_file(
        C3_FORMAL_MANIFEST,
        f0.C3_FORMAL_MANIFEST_SHA256,
        "F0_A1_C3_FORMAL_MANIFEST_HASH_FAIL_CLOSED",
    )
    audit_path, prediction_path = _c3_reference_paths(active_seed, active_arm)
    expected = _manifest_entries(C3_FORMAL_MANIFEST)
    relative_audit = str(audit_path.relative_to(C3_FINAL_FREEZE))
    relative_prediction = str(prediction_path.relative_to(C3_FINAL_FREEZE))
    _require(
        audit_path.is_file() and prediction_path.is_file()
        and relative_audit in expected and relative_prediction in expected
        and file_sha256(audit_path) == expected[relative_audit]
        and file_sha256(prediction_path) == expected[relative_prediction],
        "F0_A1_CANONICAL_C3_REFERENCE_HASH_FAIL_CLOSED",
    )
    with open(audit_path, "r", encoding="utf-8") as input_file:
        audit = json.load(input_file)
    _require(
        audit.get("stage") == c3b0.STAGE
        and audit.get("arm") == active_arm
        and int(audit.get("seed", -1)) == active_seed
        and isinstance(audit.get("final_model_hash"), dict)
        and isinstance(audit["final_model_hash"].get("aggregate"), str)
        and audit.get("GT_loaded_during_training") is False,
        "F0_A1_CANONICAL_C3_AUDIT_BOUNDARY_FAIL_CLOSED",
    )
    with np.load(prediction_path, allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == ("predictions", "sample_ids"),
            "F0_A1_CANONICAL_C3_PREDICTION_SCHEMA_FAIL_CLOSED",
        )
        predictions = np.ascontiguousarray(archive["predictions"], dtype=np.int64)
        sample_ids = np.ascontiguousarray(archive["sample_ids"], dtype=np.int64)
    _require(
        predictions.shape == (N,) and sample_ids.shape == (N,)
        and np.array_equal(sample_ids, np.arange(N)),
        "F0_A1_CANONICAL_C3_REFERENCE_ARRAY_FAIL_CLOSED",
    )
    return {
        "seed": active_seed,
        "arm": active_arm,
        "canonical_freeze_root": str(C3_FINAL_FREEZE),
        "audit_path": str(audit_path),
        "audit_file_sha256": file_sha256(audit_path),
        "prediction_path": str(prediction_path),
        "prediction_file_sha256": file_sha256(prediction_path),
        "historical_audit_provenance": {
            "stage": audit["stage"],
            "seed": audit["seed"],
            "arm": audit["arm"],
            "GT_loaded_during_training": False,
        },
        "final_model_hash": audit["final_model_hash"],
        "predictions": predictions,
        "sample_ids": sample_ids,
    }


def validate_artifact_arrays(arrays):
    _require(
        isinstance(arrays, dict) and tuple(arrays) == ARTIFACT_KEYS,
        "F0_A1_ARTIFACT_SCHEMA_FAIL_CLOSED",
    )
    arm = validate_arm(str(np.asarray(arrays["arm"]).item()))
    seed = validate_seed(int(np.asarray(arrays["seed"]).item()))
    del seed
    expected_epoch, expected_phase = snapshot_schedule(arm)
    expected_count = 21 if arm == "BASE" else 41
    _validate_sparse_ids(
        arrays["sample_ids"], arrays["unlabeled_ids"], arrays["labeled_ids"]
    )
    q_snapshots = np.asarray(arrays["q_local_snapshots"])
    hashes = (
        "final_model_hash", "frozen_U_logical_hash",
        "relation_target_logical_hash", "relation_balance_weights_logical_hash",
    )
    _require(
        np.asarray(arrays["final_predictions"]).shape == (N,)
        and q_snapshots.shape == (expected_count, N, V, K)
        and q_snapshots.dtype == np.float32 and np.isfinite(q_snapshots).all()
        and np.array_equal(arrays["snapshot_epoch"], expected_epoch)
        and np.array_equal(arrays["snapshot_phase"], expected_phase)
        and all(
            np.asarray(arrays[name]).shape == ()
            and len(str(np.asarray(arrays[name]).item())) == 64
            for name in hashes
        ),
        "F0_A1_ARTIFACT_ARRAY_BOUNDARY_FAIL_CLOSED",
    )
    return True


def load_artifact_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == ARTIFACT_KEYS,
            "F0_A1_ARTIFACT_SCHEMA_FAIL_CLOSED",
        )
        arrays = OrderedDict(
            (name, np.array(archive[name], copy=True, order="C"))
            for name in ARTIFACT_KEYS
        )
    validate_artifact_arrays(arrays)
    return arrays


def array_records(arrays):
    validate_artifact_arrays(arrays)
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
            "logical_sha256": f0.logical_sha256(np.asarray(value)),
        }
        for name, value in arrays.items()
    }


def build_pre_gt_seal(audit, artifact_path, audit_path):
    validate_forbidden_flags(audit.get("forbidden_flags"))
    required_true = (
        "preregistered_protocol_hash_pass", "all_parent_hashes_pass",
        "sample_ids_equal", "labeled_ids_equal", "unlabeled_ids_equal",
        "snapshot_count_equal", "snapshot_shape_pass", "snapshot_order_pass",
        "snapshot_finite_pass", "q_local_primary_state",
        "final_sample_ids_equal", "final_predictions_equal",
        "final_model_hash_supported", "final_model_hash_equal",
    )
    required_false = (
        "q_aligned_primary_state", "extra_refresh_for_snapshot",
        "trajectory_used_for_training", "trajectory_gradient_enabled",
        "GT_loaded_during_trajectory", "GT_used_for_checkpoint_selection",
        "scientific_metric_loaded_during_runner",
        "full_GT_present_in_pre_gt_artifact",
        "model_state_changed_by_snapshot",
        "optimizer_state_changed_by_snapshot", "rng_state_changed_by_snapshot",
    )
    _require(
        audit.get("stage") == STAGE
        and audit.get("preregistered_protocol_sha256")
        == PREREGISTERED_PROTOCOL_SHA256
        and all(audit.get(name) is True for name in required_true)
        and all(audit.get(name) is False for name in required_false)
        and all(audit.get(name) is False for name in FORBIDDEN_FLAGS)
        and audit.get("refresh_native_target_expected_count") == 2
        and audit.get("refresh_native_target_actual_count") == 2,
        "F0_A1_PRE_GT_SEAL_AUDIT_FAIL_CLOSED",
    )
    artifact = Path(artifact_path)
    audit_file = Path(audit_path)
    _require(
        artifact.is_file() and audit_file.is_file(),
        "F0_A1_PRE_GT_SEAL_INPUT_MISSING",
    )
    return {
        "stage": STAGE,
        "purpose": PURPOSE,
        "arm": validate_arm(audit["arm"]),
        "seed": validate_seed(audit["seed"]),
        "pre_gt_seal_valid": True,
        "artifact_path": str(artifact),
        "artifact_file_sha256": file_sha256(artifact),
        "audit_path": str(audit_file),
        "audit_file_sha256": file_sha256(audit_file),
        "preregistered_protocol_sha256": PREREGISTERED_PROTOCOL_SHA256,
        "all_parent_hashes_pass": True,
        "final_predictions_equal": True,
        "final_model_hash_equal": True,
        "snapshot_integrity_pass": True,
        "GT_loaded_before_pre_gt_seal": False,
        "forbidden_flags": dict(validate_forbidden_flags(
            audit["forbidden_flags"]
        )),
        "artifact_keys": list(ARTIFACT_KEYS),
        "arrays": audit["arrays"],
    }


def validate_pre_gt_seal(artifact_path, audit_path, seal_path):
    artifact = Path(artifact_path)
    audit_file = Path(audit_path)
    seal_file = Path(seal_path)
    _require(
        artifact.is_file() and audit_file.is_file() and seal_file.is_file(),
        "F0_A1_VALID_PRE_GT_SEAL_REQUIRED",
    )
    with open(audit_file, "r", encoding="utf-8") as input_file:
        audit = json.load(input_file)
    with open(seal_file, "r", encoding="utf-8") as input_file:
        seal = json.load(input_file)
    _require(
        seal.get("stage") == STAGE
        and seal.get("arm") == audit.get("arm")
        and int(seal.get("seed", -1)) == int(audit.get("seed", -2))
        and seal.get("pre_gt_seal_valid") is True
        and seal.get("preregistered_protocol_sha256")
        == PREREGISTERED_PROTOCOL_SHA256
        and seal.get("all_parent_hashes_pass") is True
        and seal.get("final_predictions_equal") is True
        and seal.get("final_model_hash_equal") is True
        and seal.get("snapshot_integrity_pass") is True
        and seal.get("GT_loaded_before_pre_gt_seal") is False
        and seal.get("artifact_keys") == list(ARTIFACT_KEYS)
        and file_sha256(artifact) == seal.get("artifact_file_sha256")
        and file_sha256(audit_file) == seal.get("audit_file_sha256"),
        "F0_A1_PRE_GT_SEAL_FAIL_CLOSED",
    )
    validate_forbidden_flags(audit.get("forbidden_flags"))
    validate_forbidden_flags(seal.get("forbidden_flags"))
    arrays = load_artifact_arrays(artifact)
    _require(
        seal.get("arm") == str(np.asarray(arrays["arm"]).item())
        and int(seal.get("seed", -1)) == int(np.asarray(arrays["seed"]).item())
        and
        audit.get("arrays") == array_records(arrays)
        and seal.get("arrays") == audit.get("arrays"),
        "F0_A1_PRE_GT_SEAL_ARRAY_RECORD_FAIL_CLOSED",
    )
    build_pre_gt_seal(audit, artifact, audit_file)
    return {
        "stage": STAGE,
        "arm": seal["arm"],
        "seed": seal["seed"],
        "pre_gt_seal_valid": True,
        "GT_loaded": False,
    }
