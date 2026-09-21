"""Clean alternating orchestration over the frozen R1 and R4 objectives."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from release_core.action import utility_conditioned_relation_loss
from release_core.backbone.clustering import native_refresh_from_latents
from release_core.backbone.native_objective import native_objective
from release_core.data.weak_quality import ndarray_sha256


@dataclass(frozen=True, eq=False)
class NativeTargetState:
    """Detached native target objects carried between training epochs."""

    p_global: object
    matches: object
    view_weights: tuple
    refresh_count: int = 0
    last_refresh_epoch: object = None


@dataclass(frozen=True)
class OptimizerTopologyAudit:
    view_count: int
    semantic_optimizer_count: int
    native_optimizer_count: int
    parameter_counts: tuple
    separate_optimizer_objects: bool
    separate_state_mappings: bool
    identical_parameter_objects_per_view: bool
    exact_adam_configuration: bool


@dataclass(frozen=True)
class TrainingOrders:
    semantic_orders: tuple
    native_orders: tuple
    generator_objects_distinct: bool
    corresponding_values_identical: bool


@dataclass(frozen=True)
class PhaseAudit:
    phase: str
    batch_count: int
    sample_count: int
    sample_id_sha256: str
    backward_count: int
    optimizer_step_count: int
    loss_sum: float
    gradients_clean_at_end: bool
    encoder_gradient_path: bool
    decoder_gradient_path: bool
    cluster_gradient_path: bool
    anchor_recomputation_count: int = 0


@dataclass(frozen=True)
class TargetRefreshAudit:
    epoch: int
    executed: bool
    refresh_count: int
    p_global_shape: object
    match_shape: object
    view_weights: tuple
    used_post_phase_a_model: bool


@dataclass(frozen=True)
class EpochAudit:
    epoch: int
    phase_a: PhaseAudit
    target_refresh: TargetRefreshAudit
    phase_b: PhaseAudit
    gradient_clean_transition_pass: bool
    event_sequence: tuple


@dataclass(frozen=True)
class TrainingAudit:
    epoch_count: int
    view_count: int
    optimizer_topology: OptimizerTopologyAudit
    epoch_audits: tuple
    refresh_count: int
    phase_a_backward_count: int
    phase_a_optimizer_step_count: int
    phase_b_backward_count: int
    phase_b_optimizer_step_count: int
    input_hashes: tuple
    output_hashes: tuple
    frozen_inputs_unchanged: bool
    final_prediction_refresh_executed: bool


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(name + " must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(name + " must be positive")
    return value


def _finite_positive_float(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(name + " must be numeric")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(name + " must be positive and finite")
    return value


def _model_views(model):
    autoencoders = getattr(model, "autoencoders", None)
    if not isinstance(autoencoders, (list, tuple)) or not autoencoders:
        raise ValueError("model must expose a non-empty autoencoders sequence")
    if not all(isinstance(module, nn.Module) for module in autoencoders):
        raise TypeError("every per-view model must be a torch module")
    return tuple(autoencoders)


def _validate_stateless_training_model(model):
    forbidden_dropout = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    for view_id, autoencoder in enumerate(_model_views(model)):
        autoencoder.train()
        for name, module in autoencoder.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                raise ValueError("stateful BatchNorm is not supported")
            if isinstance(module, forbidden_dropout):
                raise ValueError("stochastic Dropout is not supported")
            if tuple(module.named_buffers(recurse=False)):
                raise ValueError(
                    "unexpected mutable buffer in view " + str(view_id)
                    + ": " + (name or "<root>")
                )


def _parameter_ids(optimizer):
    return tuple(
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def _optimizer_has_exact_configuration(optimizer, learning_rate):
    expected = {
        "lr": learning_rate,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
    }
    return all(optimizer.defaults.get(key) == value for key, value in expected.items())


def build_decoupled_optimizers(model, lr):
    """Build two independent per-view Adam collections over identical tensors."""
    learning_rate = _finite_positive_float(lr, "lr")
    autoencoders = _model_views(model)
    semantic_optimizers = []
    native_optimizers = []
    parameter_counts = []
    for autoencoder in autoencoders:
        parameters = list(autoencoder.parameters())
        if not parameters:
            raise ValueError("per-view model has no parameters")
        semantic_optimizers.append(torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            amsgrad=False,
            maximize=False,
            foreach=None,
            capturable=False,
            differentiable=False,
            fused=None,
        ))
        native_optimizers.append(torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            amsgrad=False,
            maximize=False,
            foreach=None,
            capturable=False,
            differentiable=False,
            fused=None,
        ))
        parameter_counts.append(len(parameters))

    separate_objects = all(
        semantic is not native
        for semantic, native in zip(semantic_optimizers, native_optimizers)
    )
    separate_states = all(
        semantic.state is not native.state
        for semantic, native in zip(semantic_optimizers, native_optimizers)
    )
    identical_parameters = all(
        _parameter_ids(semantic) == _parameter_ids(native)
        for semantic, native in zip(semantic_optimizers, native_optimizers)
    )
    exact_configuration = all(
        _optimizer_has_exact_configuration(optimizer, learning_rate)
        for optimizer in semantic_optimizers + native_optimizers
    )
    if not (
        separate_objects
        and separate_states
        and identical_parameters
        and exact_configuration
        and all(not optimizer.state for optimizer in semantic_optimizers)
        and all(not optimizer.state for optimizer in native_optimizers)
    ):
        raise RuntimeError("optimizer topology mismatch")
    audit = OptimizerTopologyAudit(
        view_count=len(autoencoders),
        semantic_optimizer_count=len(semantic_optimizers),
        native_optimizer_count=len(native_optimizers),
        parameter_counts=tuple(parameter_counts),
        separate_optimizer_objects=True,
        separate_state_mappings=True,
        identical_parameter_objects_per_view=True,
        exact_adam_configuration=True,
    )
    return tuple(semantic_optimizers), tuple(native_optimizers), audit


def _int64_ids(values, name, *, nonempty=True):
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            values = values.detach().cpu()
        array = values.detach().numpy()
    else:
        array = np.asarray(values)
    if array.ndim != 1 or (nonempty and array.size == 0):
        raise ValueError(name + " must be a non-empty rank-1 sequence")
    if array.dtype != np.dtype(np.int64):
        raise TypeError(name + " must have dtype int64")
    array = np.ascontiguousarray(array)
    if np.unique(array).size != array.size:
        raise ValueError(name + " must contain unique IDs")
    return array


def _validate_order(order, sample_ids, name):
    values = _int64_ids(order, name)
    if values.shape != sample_ids.shape or not np.array_equal(
        np.sort(values), np.sort(sample_ids)
    ):
        raise ValueError(name + " must contain every sample ID exactly once")
    return values


def precompute_training_orders(sample_ids, epochs, seed):
    """Precompute separate but value-identical full-ID order streams."""
    ids = _int64_ids(sample_ids, "sample_ids")
    epoch_count = _positive_integer(epochs, "epochs")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer")
    semantic_generator = torch.Generator(device="cpu")
    native_generator = torch.Generator(device="cpu")
    semantic_generator.manual_seed(int(seed))
    native_generator.manual_seed(int(seed))
    ids_tensor = torch.from_numpy(ids)
    semantic_orders = tuple(
        ids_tensor[torch.randperm(ids.size, generator=semantic_generator)].clone()
        for _ in range(epoch_count)
    )
    native_orders = tuple(
        ids_tensor[torch.randperm(ids.size, generator=native_generator)].clone()
        for _ in range(epoch_count)
    )
    for epoch, (semantic, native) in enumerate(
        zip(semantic_orders, native_orders)
    ):
        _validate_order(semantic, ids, "semantic order " + str(epoch))
        _validate_order(native, ids, "native order " + str(epoch))
        if semantic is native or semantic.data_ptr() == native.data_ptr():
            raise RuntimeError("order tensors must not share identity or storage")
        if not torch.equal(semantic, native):
            raise RuntimeError("historical order values must be identical")
    return TrainingOrders(
        semantic_orders=semantic_orders,
        native_orders=native_orders,
        generator_objects_distinct=semantic_generator is not native_generator,
        corresponding_values_identical=True,
    )


def _validate_views(model, views, sample_count):
    autoencoders = _model_views(model)
    if not isinstance(views, (tuple, list)) or len(views) != len(autoencoders):
        raise ValueError("views must match the per-view model count")
    checked = []
    for view_id, values in enumerate(views):
        if not isinstance(values, torch.Tensor):
            raise TypeError("view " + str(view_id) + " must be a torch tensor")
        if values.ndim != 2 or values.shape[0] != sample_count:
            raise ValueError("each view must have shape [N,D_v]")
        if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
            raise ValueError("each view must be finite floating data")
        checked.append(values)
    return autoencoders, tuple(checked)


def _validate_partition(sample_ids, labeled_ids, unlabeled_ids):
    labeled = _int64_ids(labeled_ids, "labeled_ids")
    unlabeled = _int64_ids(unlabeled_ids, "unlabeled_ids")
    if np.intersect1d(labeled, unlabeled).size:
        raise ValueError("labeled and unlabeled IDs overlap")
    combined = np.concatenate((labeled, unlabeled))
    if combined.size != sample_ids.size or not np.array_equal(
        np.sort(combined), np.sort(sample_ids)
    ):
        raise ValueError("labeled and unlabeled IDs must partition sample_ids")
    return labeled, unlabeled


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return np.ascontiguousarray(value.detach().cpu().numpy())
    return np.ascontiguousarray(np.asarray(value))


def _logical_hash(value):
    return ndarray_sha256(_as_numpy(value))


def _row_lookup(sample_ids):
    return {int(sample_id): row for row, sample_id in enumerate(sample_ids)}


def _positions(ids, lookup, name):
    try:
        rows = [lookup[int(sample_id)] for sample_id in ids]
    except KeyError as error:
        raise ValueError(name + " contains an unknown sample ID") from error
    return np.ascontiguousarray(rows, dtype=np.int64)


def _select_rows(values, rows, device):
    index = torch.as_tensor(rows, dtype=torch.long, device=values.device)
    return values[index].to(device)


def _zero_gradients(optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def _all_gradients_none(model):
    return all(
        parameter.grad is None
        for autoencoder in _model_views(model)
        for parameter in autoencoder.parameters()
    )


def _family_parameters(autoencoder):
    encoder = tuple(autoencoder._encoder.parameters())
    decoder = tuple(autoencoder._decoder.parameters())
    cluster = (autoencoder._cluster_layer,)
    if not encoder or not decoder:
        raise ValueError("per-view parameter families are incomplete")
    return encoder, decoder, cluster


def _gradient_family_status(model):
    family_status = {"encoder": [], "decoder": [], "cluster": []}
    finite = True
    for autoencoder in _model_views(model):
        encoder, decoder, cluster = _family_parameters(autoencoder)
        for name, parameters in (
            ("encoder", encoder), ("decoder", decoder), ("cluster", cluster)
        ):
            gradients = tuple(parameter.grad for parameter in parameters)
            present = any(gradient is not None for gradient in gradients)
            family_status[name].append(present)
            finite = finite and all(
                gradient is None or bool(torch.isfinite(gradient).all().item())
                for gradient in gradients
            )
    return (
        all(family_status["encoder"]),
        all(family_status["decoder"]),
        all(family_status["cluster"]),
        bool(finite),
    )


def _validate_action_inputs(
    sample_ids, labeled_ids, unlabeled_ids, u_cycle, pred_relation, balance
):
    N = sample_ids.size
    Nu = unlabeled_ids.size
    L = labeled_ids.size
    cycle = _as_numpy(u_cycle)
    target = _as_numpy(pred_relation)
    weights = _as_numpy(balance)
    if cycle.ndim != 2 or cycle.shape[0] not in (N, Nu) or cycle.shape[1] == 0:
        raise ValueError("u_cycle must have shape [Nu,S] or [N,S]")
    S = cycle.shape[1]
    if target.shape != (Nu, L, S) or weights.shape != (Nu, L, S):
        raise ValueError("relation tensors must have shape [Nu,L,S]")
    if not np.isfinite(cycle).all() or np.any(cycle < 0):
        raise ValueError("u_cycle must be finite and nonnegative")
    if not np.isfinite(target).all() or not np.all((target == 0) | (target == 1)):
        raise ValueError("pred_relation must be finite and binary")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("balance must be finite and positive")
    return cycle.shape[0] == N


def run_relation_refinement_phase(
    model,
    semantic_optimizers,
    views,
    sample_ids,
    labeled_ids,
    unlabeled_ids,
    u_cycle,
    pred_relation,
    balance,
    order,
    batch_size,
    device,
):
    """Run Phase A over unlabeled query rows with current labeled anchors."""
    ids = _int64_ids(sample_ids, "sample_ids")
    labeled, unlabeled = _validate_partition(
        ids,
        labeled_ids,
        unlabeled_ids,
    )
    autoencoders, full_views = _validate_views(model, views, ids.size)
    if len(semantic_optimizers) != len(autoencoders):
        raise ValueError("semantic optimizer count must equal view count")
    batch_limit = _positive_integer(batch_size, "batch_size")
    active_order = _validate_order(order, ids, "semantic order")
    full_cycle_rows = _validate_action_inputs(
        ids, labeled, unlabeled, u_cycle, pred_relation, balance
    )
    sample_rows = _row_lookup(ids)
    action_rows = _row_lookup(unlabeled)
    anchor_rows = _positions(labeled, sample_rows, "labeled_ids")
    visited = []
    loss_sum = 0.0
    backward_count = 0
    optimizer_step_count = 0
    anchor_recomputation_count = 0
    encoder_path = True
    decoder_path = False
    cluster_path = True

    for start in range(0, ids.size, batch_limit):
        full_batch = active_order[start:start + batch_limit]
        query_ids = full_batch[np.isin(full_batch, unlabeled)]
        if query_ids.size == 0:
            raise RuntimeError("semantic batch is empty after labeled-ID filtering")
        query_rows = _positions(query_ids, sample_rows, "semantic query")
        selected_action_rows = _positions(
            query_ids, action_rows, "semantic query"
        )
        visited.append(query_ids)
        _zero_gradients(semantic_optimizers)
        if not _all_gradients_none(model):
            raise RuntimeError("semantic gradients were not cleared")

        q_query_views = []
        q_anchor_views = []
        for view_id, autoencoder in enumerate(autoencoders):
            # query_x:[B_u,D_v] -> query_z:[B_u,D_z] -> q_query:[B_u,K]
            query_x = _select_rows(full_views[view_id], query_rows, device)
            query_z = autoencoder.encoder(query_x)
            q_query_views.append(autoencoder.clustering(query_z))
            with torch.no_grad():
                # anchor_x:[L,D_v] -> anchor_z:[L,D_z] -> q_anchor:[L,K]
                anchor_x = _select_rows(full_views[view_id], anchor_rows, device)
                anchor_z = autoencoder.encoder(anchor_x)
                q_anchor = autoencoder.clustering(anchor_z)
            q_anchor_views.append(q_anchor.detach())
        anchor_recomputation_count += 1

        cycle_rows = query_rows if full_cycle_rows else selected_action_rows
        # U_batch:[B_u,S], target/balance:[B_u,L,S].
        relation_loss, _ = utility_conditioned_relation_loss(
            q_query_views,
            q_anchor_views,
            _as_numpy(pred_relation)[selected_action_rows],
            _as_numpy(u_cycle)[cycle_rows],
            _as_numpy(balance)[selected_action_rows],
        )
        if relation_loss.ndim != 0 or not bool(torch.isfinite(relation_loss).item()):
            raise RuntimeError("relation loss must be a finite scalar")
        relation_loss.backward()
        backward_count += 1
        encoder, decoder, cluster, finite = _gradient_family_status(model)
        if not finite or not encoder or decoder or not cluster:
            raise RuntimeError("Phase-A gradient-family contract failed")
        for optimizer in semantic_optimizers:
            optimizer.step()
            optimizer_step_count += 1
        loss_sum += float(relation_loss.detach().item())

    _zero_gradients(semantic_optimizers)
    clean_end = _all_gradients_none(model)
    if not clean_end:
        raise RuntimeError("Phase-A gradients survived the phase transition")
    visited_ids = np.concatenate(visited).astype(np.int64, copy=False)
    if not (
        visited_ids.size == unlabeled.size
        and np.unique(visited_ids).size == unlabeled.size
        and np.array_equal(np.sort(visited_ids), np.sort(unlabeled))
        and np.intersect1d(visited_ids, labeled).size == 0
    ):
        raise RuntimeError("Phase-A sample coverage mismatch")
    return PhaseAudit(
        phase="relation_refinement",
        batch_count=backward_count,
        sample_count=int(visited_ids.size),
        sample_id_sha256=ndarray_sha256(np.ascontiguousarray(visited_ids)),
        backward_count=backward_count,
        optimizer_step_count=optimizer_step_count,
        loss_sum=float(loss_sum),
        gradients_clean_at_end=True,
        encoder_gradient_path=encoder_path,
        decoder_gradient_path=decoder_path,
        cluster_gradient_path=cluster_path,
        anchor_recomputation_count=anchor_recomputation_count,
    )


def initial_native_target_state(view_count):
    count = _positive_integer(view_count, "view_count")
    return NativeTargetState(
        p_global=None,
        matches=None,
        view_weights=tuple(1.0 for _ in range(count)),
    )


def refresh_native_state_if_due(
    model,
    views,
    *,
    epoch,
    refresh_interval,
    previous_state,
    seed,
    device,
):
    """Refresh detached R1 target state on the zero-based native cadence."""
    if isinstance(epoch, (bool, np.bool_)) or not isinstance(
        epoch, (int, np.integer)
    ) or int(epoch) < 0:
        raise ValueError("epoch must be a nonnegative integer")
    interval = _positive_integer(refresh_interval, "refresh_interval")
    autoencoders = _model_views(model)
    if not isinstance(previous_state, NativeTargetState):
        raise TypeError("previous_state must be NativeTargetState")
    if len(previous_state.view_weights) != len(autoencoders):
        raise ValueError("view-weight count must equal view count")
    if int(epoch) % interval != 0:
        audit = TargetRefreshAudit(
            epoch=int(epoch),
            executed=False,
            refresh_count=previous_state.refresh_count,
            p_global_shape=(
                None if previous_state.p_global is None
                else tuple(previous_state.p_global.shape)
            ),
            match_shape=(
                None if previous_state.matches is None
                else tuple(previous_state.matches.shape)
            ),
            view_weights=previous_state.view_weights,
            used_post_phase_a_model=True,
        )
        return previous_state, audit

    _, full_views = _validate_views(model, views, int(views[0].shape[0]))
    latent_views = []
    q_local_views = []
    with torch.no_grad():
        for view_id, autoencoder in enumerate(autoencoders):
            # full_x:[N,D_v] -> full_z:[N,D_z] -> q_local:[N,K].
            full_x = full_views[view_id].to(device)
            full_z = autoencoder.encoder(full_x)
            q_local = autoencoder.clustering(full_z)
            latent_views.append(full_z.detach().cpu().numpy())
            q_local_views.append(q_local.detach().cpu().numpy())
        p_numpy, match_numpy, _, weights, _ = native_refresh_from_latents(
            latent_views,
            q_local_views,
            previous_state.view_weights,
            n_clusters=int(getattr(model, "n_clusters")),
            random_state=int(seed),
        )
    # P_global:[N,K]; Match:[V,K,K]. Both are frozen target tensors.
    p_global = torch.from_numpy(np.asarray(p_numpy)).float().detach()
    matches = torch.from_numpy(np.asarray(match_numpy)).float().to(device).detach()
    if (
        p_global.ndim != 2
        or p_global.shape[0] != full_views[0].shape[0]
        or matches.ndim != 3
        or matches.shape[0] != len(autoencoders)
        or matches.shape[1:] != (p_global.shape[1], p_global.shape[1])
        or not bool(torch.isfinite(p_global).all().item())
        or not bool(torch.isfinite(matches).all().item())
    ):
        raise RuntimeError("native target refresh returned invalid state")
    state = NativeTargetState(
        p_global=p_global,
        matches=matches,
        view_weights=tuple(float(value) for value in weights),
        refresh_count=previous_state.refresh_count + 1,
        last_refresh_epoch=int(epoch),
    )
    audit = TargetRefreshAudit(
        epoch=int(epoch),
        executed=True,
        refresh_count=state.refresh_count,
        p_global_shape=tuple(p_global.shape),
        match_shape=tuple(matches.shape),
        view_weights=state.view_weights,
        used_post_phase_a_model=True,
    )
    return state, audit


def _validate_native_state(state, sample_count, view_count):
    if not isinstance(state, NativeTargetState):
        raise TypeError("native state must be NativeTargetState")
    if not isinstance(state.p_global, torch.Tensor) or not isinstance(
        state.matches, torch.Tensor
    ):
        raise RuntimeError("native target state is missing")
    if (
        state.p_global.ndim != 2
        or state.p_global.shape[0] != sample_count
        or state.matches.shape
        != (view_count, state.p_global.shape[1], state.p_global.shape[1])
        or state.p_global.requires_grad
        or state.p_global.grad_fn is not None
        or state.matches.requires_grad
        or state.matches.grad_fn is not None
    ):
        raise RuntimeError("native target state violates shape/detach contract")


def run_native_consolidation_phase(
    model,
    native_optimizers,
    views,
    sample_ids,
    native_state,
    order,
    batch_size,
    native_lambda1,
    device,
):
    """Run Phase B over every canonical sample with the frozen R1 objective."""
    ids = _int64_ids(sample_ids, "sample_ids")
    autoencoders, full_views = _validate_views(model, views, ids.size)
    if len(native_optimizers) != len(autoencoders):
        raise ValueError("native optimizer count must equal view count")
    _validate_native_state(native_state, ids.size, len(autoencoders))
    batch_limit = _positive_integer(batch_size, "batch_size")
    lambda1 = _finite_positive_float(native_lambda1, "native_lambda1")
    active_order = _validate_order(order, ids, "native order")
    sample_rows = _row_lookup(ids)
    visited = []
    loss_sum = 0.0
    backward_count = 0
    optimizer_step_count = 0

    for start in range(0, ids.size, batch_limit):
        batch_ids = active_order[start:start + batch_limit]
        rows = _positions(batch_ids, sample_rows, "native batch")
        visited.append(batch_ids)
        _zero_gradients(native_optimizers)
        if not _all_gradients_none(model):
            raise RuntimeError("native gradients were not cleared")
        p_rows = torch.as_tensor(rows, dtype=torch.long)
        # P_global:[N,K] -> P_batch:[B,K], detached.
        p_batch = native_state.p_global[p_rows].to(device).detach()
        view_losses = []
        for view_id, autoencoder in enumerate(autoencoders):
            # x_v:[B,D_v] -> z_v:[B,D_z], reconstruction_v:[B,D_v], q_v:[B,K].
            x_view = _select_rows(full_views[view_id], rows, device)
            reconstruction, _, q_local = autoencoder(x_view)
            match = native_state.matches[view_id].to(
                device=device, dtype=q_local.dtype
            ).detach()
            # P_local_v:[B,K], derived only from detached native targets.
            p_local = p_batch.to(dtype=q_local.dtype) @ match
            view_loss, _, _ = native_objective(
                reconstruction,
                x_view,
                q_local,
                p_local.detach(),
                lambda1,
            )
            view_losses.append(view_loss)
        native_loss = torch.stack(view_losses).sum()
        if native_loss.ndim != 0 or not bool(torch.isfinite(native_loss).item()):
            raise RuntimeError("native loss must be a finite scalar")
        native_loss.backward()
        backward_count += 1
        encoder, decoder, cluster, finite = _gradient_family_status(model)
        if not finite or not encoder or not decoder or not cluster:
            raise RuntimeError("Phase-B gradient-family contract failed")
        for optimizer in native_optimizers:
            optimizer.step()
            optimizer_step_count += 1
        loss_sum += float(native_loss.detach().item())

    _zero_gradients(native_optimizers)
    clean_end = _all_gradients_none(model)
    if not clean_end:
        raise RuntimeError("Phase-B gradients survived the epoch boundary")
    visited_ids = np.concatenate(visited).astype(np.int64, copy=False)
    if not (
        visited_ids.size == ids.size
        and np.unique(visited_ids).size == ids.size
        and np.array_equal(np.sort(visited_ids), np.sort(ids))
    ):
        raise RuntimeError("Phase-B sample coverage mismatch")
    return PhaseAudit(
        phase="native_consolidation",
        batch_count=backward_count,
        sample_count=int(visited_ids.size),
        sample_id_sha256=ndarray_sha256(np.ascontiguousarray(visited_ids)),
        backward_count=backward_count,
        optimizer_step_count=optimizer_step_count,
        loss_sum=float(loss_sum),
        gradients_clean_at_end=True,
        encoder_gradient_path=True,
        decoder_gradient_path=True,
        cluster_gradient_path=True,
    )


def run_alternating_epoch(
    model,
    semantic_optimizers,
    native_optimizers,
    views,
    sample_ids,
    labeled_ids,
    unlabeled_ids,
    u_cycle,
    pred_relation,
    balance,
    semantic_order,
    native_order,
    native_state,
    *,
    epoch,
    batch_size,
    refresh_interval,
    native_lambda1,
    seed,
    device,
):
    """Execute one complete A, optional refresh, then complete B epoch."""
    events = ["PHASE_A_START"]
    phase_a = run_relation_refinement_phase(
        model,
        semantic_optimizers,
        views,
        sample_ids,
        labeled_ids,
        unlabeled_ids,
        u_cycle,
        pred_relation,
        balance,
        semantic_order,
        batch_size,
        device,
    )
    events.append("PHASE_A_END")
    if not _all_gradients_none(model):
        raise RuntimeError("gradient leakage at Phase-A transition")
    refresh_due = int(epoch) % _positive_integer(
        refresh_interval, "refresh_interval"
    ) == 0
    if refresh_due:
        events.append("REFRESH_START")
    native_state, refresh_audit = refresh_native_state_if_due(
        model,
        views,
        epoch=epoch,
        refresh_interval=refresh_interval,
        previous_state=native_state,
        seed=seed,
        device=device,
    )
    if refresh_due:
        events.append("REFRESH_END")
    if not _all_gradients_none(model):
        raise RuntimeError("target refresh created model gradients")
    events.append("PHASE_B_START")
    phase_b = run_native_consolidation_phase(
        model,
        native_optimizers,
        views,
        sample_ids,
        native_state,
        native_order,
        batch_size,
        native_lambda1,
        device,
    )
    events.append("PHASE_B_END")
    clean_transition = _all_gradients_none(model)
    if not clean_transition:
        raise RuntimeError("gradient leakage after Phase B")
    return native_state, EpochAudit(
        epoch=int(epoch),
        phase_a=phase_a,
        target_refresh=refresh_audit,
        phase_b=phase_b,
        gradient_clean_transition_pass=True,
        event_sequence=tuple(events),
    )


def _frozen_input_hashes(
    u_cycle, pred_relation, balance, labeled_ids, unlabeled_ids
):
    return (
        ("u_cycle", _logical_hash(u_cycle)),
        ("pred_relation", _logical_hash(pred_relation)),
        ("balance", _logical_hash(balance)),
        ("labeled_ids", _logical_hash(labeled_ids)),
        ("unlabeled_ids", _logical_hash(unlabeled_ids)),
    )


def run_alternating_training(
    model,
    views,
    sample_ids,
    labeled_ids,
    unlabeled_ids,
    u_cycle,
    pred_relation,
    balance,
    *,
    epochs,
    batch_size,
    seed,
    refresh_interval,
    learning_rate,
    native_lambda1,
    device,
):
    """Run in-process alternating training and return no prediction or metric."""
    epoch_count = _positive_integer(epochs, "epochs")
    ids = _int64_ids(sample_ids, "sample_ids")
    labeled, unlabeled = _validate_partition(ids, labeled_ids, unlabeled_ids)
    _validate_views(model, views, ids.size)
    _validate_action_inputs(
        ids, labeled, unlabeled, u_cycle, pred_relation, balance
    )
    _validate_stateless_training_model(model)
    input_hashes = _frozen_input_hashes(
        u_cycle, pred_relation, balance, labeled_ids, unlabeled_ids
    )
    semantic_optimizers, native_optimizers, topology = (
        build_decoupled_optimizers(model, learning_rate)
    )
    orders = precompute_training_orders(ids, epoch_count, seed)
    if (
        len(orders.semantic_orders) != epoch_count
        or len(orders.native_orders) != epoch_count
    ):
        raise RuntimeError("training order count mismatch")
    state = initial_native_target_state(len(_model_views(model)))
    epoch_audits = []
    for epoch in range(epoch_count):
        state, audit = run_alternating_epoch(
            model,
            semantic_optimizers,
            native_optimizers,
            views,
            ids,
            labeled,
            unlabeled,
            u_cycle,
            pred_relation,
            balance,
            orders.semantic_orders[epoch],
            orders.native_orders[epoch],
            state,
            epoch=epoch,
            batch_size=batch_size,
            refresh_interval=refresh_interval,
            native_lambda1=native_lambda1,
            seed=seed,
            device=device,
        )
        epoch_audits.append(audit)
    output_hashes = _frozen_input_hashes(
        u_cycle, pred_relation, balance, labeled_ids, unlabeled_ids
    )
    if input_hashes != output_hashes:
        raise RuntimeError("frozen training inputs were mutated")
    training_audit = TrainingAudit(
        epoch_count=epoch_count,
        view_count=len(_model_views(model)),
        optimizer_topology=topology,
        epoch_audits=tuple(epoch_audits),
        refresh_count=state.refresh_count,
        phase_a_backward_count=sum(
            item.phase_a.backward_count for item in epoch_audits
        ),
        phase_a_optimizer_step_count=sum(
            item.phase_a.optimizer_step_count for item in epoch_audits
        ),
        phase_b_backward_count=sum(
            item.phase_b.backward_count for item in epoch_audits
        ),
        phase_b_optimizer_step_count=sum(
            item.phase_b.optimizer_step_count for item in epoch_audits
        ),
        input_hashes=input_hashes,
        output_hashes=output_hashes,
        frozen_inputs_unchanged=True,
        final_prediction_refresh_executed=False,
    )
    return model, state, training_audit
