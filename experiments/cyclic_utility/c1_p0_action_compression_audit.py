"""Read-only primitives for the frozen C1-P0 action-compression audit.

This module diagnoses the existing C1-A1 path.  It does not define a new
target, loss, action, gate, or training rule.  Target construction and both
loss branches are delegated to the frozen C1/E1 helpers.
"""

import hashlib
import math
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from experiments.cyclic_utility import c1_frozen_pseudo_supervision as c1
from experiments.cyclic_utility import train_c1_frozen_pseudo_supervision as c1_train
from irv.b3_audit import hash_backbone
from irv.b4_information_utility import tensor_sha256


STAGE = "C1-P0"
PSEUDO_ARMS = ("UNIFORM", "CONF", "CYCLE", "SHUFFLED_CYCLE")
COMPARATORS = ("UNIFORM", "CONF", "SHUFFLED_CYCLE")
EPSILON = 1e-12
ALLOWED_DESCRIPTIVE_LABELS = (
    "TARGET_AGGREGATION_COMPRESSION_EVIDENCE",
    "GRADIENT_CARRIER_COMPRESSION_EVIDENCE",
    "NATIVE_GRADIENT_DOMINANCE_EVIDENCE",
    "NO_CLEAR_COMPRESSION_LOCATION",
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _as_numpy(value, dtype=None):
    array = c1.as_numpy(value, dtype=dtype)
    return np.ascontiguousarray(array)


def _finite_float(value):
    number = float(value)
    _require(math.isfinite(number), "diagnostic scalar is non-finite")
    return number


def directional_hypothesis_structure(y_gen, class_num=c1.CLASS_NUM):
    """Compute the frozen Layer-0 vote structure without any labels."""
    hypotheses = c1.frozen_tensor(y_gen, dtype=torch.long, device="cpu")
    _require(
        hypotheses.ndim == 2
        and hypotheses.shape[1] == c1.DIRECTION_COUNT
        and bool(torch.all((hypotheses >= 0) & (hypotheses < class_num)).item()),
        "y_gen directional boundary mismatch",
    )
    vote_count = torch.nn.functional.one_hot(
        hypotheses, num_classes=int(class_num)
    ).sum(dim=1)
    vote_purity = vote_count.max(dim=1).values.to(torch.float64) / float(
        hypotheses.shape[1]
    )
    unique_count = (vote_count > 0).sum(dim=1)
    histogram = {
        str(value): int((unique_count == value).sum().item())
        for value in range(1, int(class_num) + 1)
    }
    return {
        "stage": STAGE,
        "layer": 0,
        "GT_used": False,
        "y_gen_shape": list(hypotheses.shape),
        "vote_count_shape": list(vote_count.shape),
        "vote_count_logical_sha256": tensor_sha256(vote_count.numpy()),
        "vote_purity": {
            "mean": _finite_float(vote_purity.mean().item()),
            "std": _finite_float(vote_purity.std(unbiased=False).item()),
            "median": _finite_float(vote_purity.median().item()),
            "min": _finite_float(vote_purity.min().item()),
            "max": _finite_float(vote_purity.max().item()),
            "fraction_equal_1_0": _finite_float(
                (vote_purity == 1.0).double().mean().item()
            ),
            "fraction_ge_0_95": _finite_float(
                (vote_purity >= 0.95).double().mean().item()
            ),
            "fraction_ge_0_90": _finite_float(
                (vote_purity >= 0.90).double().mean().item()
            ),
            "fraction_ge_0_80": _finite_float(
                (vote_purity >= 0.80).double().mean().item()
            ),
        },
        "directional_unique_class_count_per_sample": unique_count.tolist(),
        "directional_unique_class_count_histogram": histogram,
        "directional_unique_class_count_logical_sha256": tensor_sha256(
            unique_count.numpy()
        ),
    }


def _distribution_stats(value):
    flat = _as_numpy(value, dtype=np.float64).reshape(-1)
    _require(flat.size > 0 and np.isfinite(flat).all(), "invalid distribution")
    return {
        "mean": _finite_float(np.mean(flat)),
        "std": _finite_float(np.std(flat)),
        "min": _finite_float(np.min(flat)),
        "max": _finite_float(np.max(flat)),
        "p10": _finite_float(np.percentile(flat, 10)),
        "p25": _finite_float(np.percentile(flat, 25)),
        "p50": _finite_float(np.percentile(flat, 50)),
        "p75": _finite_float(np.percentile(flat, 75)),
        "p90": _finite_float(np.percentile(flat, 90)),
    }


def _correlation_record(left, right, kind):
    x = _as_numpy(left, dtype=np.float64).reshape(-1)
    y = _as_numpy(right, dtype=np.float64).reshape(-1)
    _require(x.shape == y.shape and x.size > 1, "correlation shape mismatch")
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return {
            "value": None,
            "defined": False,
            "reason": "constant_input",
        }
    if kind == "pearson":
        value = pearsonr(x, y).statistic
    elif kind == "spearman":
        value = spearmanr(x, y).statistic
    else:
        raise ValueError("unknown correlation kind")
    return {"value": _finite_float(value), "defined": True, "reason": None}


def pairwise_weight_metrics(cycle, comparator, eps=EPSILON):
    reference = _as_numpy(cycle, dtype=np.float64)
    candidate = _as_numpy(comparator, dtype=np.float64)
    _require(reference.shape == candidate.shape, "weight pair shape mismatch")
    difference = reference - candidate
    exact_mask = reference == candidate
    reference_norm = np.linalg.norm(reference.reshape(-1))
    return {
        "exact_equality": bool(np.array_equal(reference, candidate)),
        "elementwise_exact_equality_count": int(exact_mask.sum()),
        "elementwise_exact_equality_fraction": _finite_float(exact_mask.mean()),
        "mean_absolute_difference": _finite_float(np.mean(np.abs(difference))),
        "max_absolute_difference": _finite_float(np.max(np.abs(difference))),
        "RMSE": _finite_float(np.sqrt(np.mean(np.square(difference)))),
        "normalized_frobenius_distance": _finite_float(
            np.linalg.norm(difference.reshape(-1)) / (reference_norm + eps)
        ),
        "pearson": _correlation_record(reference, candidate, "pearson"),
        "spearman": _correlation_record(reference, candidate, "spearman"),
    }


def build_weight_diagnostics(c0_arrays):
    weights = OrderedDict()
    provenance = OrderedDict()
    for arm in PSEUDO_ARMS:
        value, audit = c1.build_arm_weights(arm, c0_arrays)
        weights[arm] = value.detach()
        provenance[arm] = audit
    comparisons = OrderedDict(
        (
            "CYCLE_vs_" + arm,
            pairwise_weight_metrics(weights["CYCLE"], weights[arm]),
        )
        for arm in COMPARATORS
    )
    return weights, {
        "stage": STAGE,
        "layer": 1,
        "formula_modified": False,
        "arm_statistics": {
            arm: {
                **_distribution_stats(weights[arm]),
                "shape": list(weights[arm].shape),
                "logical_sha256": tensor_sha256(weights[arm].numpy()),
                "source": provenance[arm]["weight_source"],
            }
            for arm in PSEUDO_ARMS
        },
        "comparisons": comparisons,
    }


def reconstruct_frozen_targets(c0_arrays, M0, c1a1_output_dir):
    """Call the frozen C1 target helper and bind every available C1-A1 hash."""
    frozen_dir = Path(c1a1_output_dir)
    root_audit_path = frozen_dir / "c1_audit.json"
    _require(root_audit_path.is_file(), "frozen C1-A1 audit is missing")
    root_audit = c1.read_json(root_audit_path)
    _require(root_audit.get("C1_AUDIT_PASS") is True, "C1-A1 audit did not pass")
    targets = OrderedDict()
    provenance = OrderedDict()
    for arm in PSEUDO_ARMS:
        target, current_audit = c1.build_frozen_pseudo_target(
            arm, c0_arrays, M0
        )
        frozen_path = frozen_dir / arm / "pseudo_target_audit.json"
        _require(frozen_path.is_file(), "frozen C1-A1 target audit is missing: " + arm)
        frozen_audit = c1.read_json(frozen_path)
        expected_hashes = frozen_audit.get("logical_sha256", {})
        actual_hashes = current_audit["logical_sha256"]
        checked = {
            name: bool(actual_hashes[name] == expected_hash)
            for name, expected_hash in expected_hashes.items()
            if name in actual_hashes
        }
        _require(checked and all(checked.values()), "C1-A1 target hash mismatch: " + arm)
        _require(
            frozen_audit.get("M0_audit", {}).get("matches_C0_seal_pass") is True,
            "C1-A1 M0 seal mismatch: " + arm,
        )
        targets[arm] = target
        provenance[arm] = {
            "frozen_audit_path": str(frozen_path),
            "frozen_audit_file_sha256": c1.file_sha256(frozen_path),
            "available_hashes_checked": sorted(checked),
            "hash_equality": checked,
            "all_available_hashes_equal": True,
            "actual_logical_sha256": actual_hashes,
        }
    return targets, provenance


def pairwise_target_metrics(cycle, comparator, eps=EPSILON):
    reference = _as_numpy(cycle, dtype=np.float64)
    candidate = _as_numpy(comparator, dtype=np.float64)
    _require(
        reference.shape == candidate.shape and reference.ndim == 2,
        "global target pair shape mismatch",
    )
    difference = reference - candidate
    sample_l1 = np.abs(difference).sum(axis=1)
    sample_l2 = np.linalg.norm(difference, axis=1)
    argmax_equal = np.argmax(reference, axis=1) == np.argmax(candidate, axis=1)
    reference_hash = tensor_sha256(_as_numpy(cycle))
    candidate_hash = tensor_sha256(_as_numpy(comparator))
    return {
        "exact_equality": bool(np.array_equal(reference, candidate)),
        "cycle_logical_sha256": reference_hash,
        "comparator_logical_sha256": candidate_hash,
        "logical_sha256_equality": bool(reference_hash == candidate_hash),
        "argmax_agreement_fraction": _finite_float(argmax_equal.mean()),
        "changed_argmax_count": int((~argmax_equal).sum()),
        "mean_samplewise_L1_distance": _finite_float(sample_l1.mean()),
        "median_samplewise_L1_distance": _finite_float(np.median(sample_l1)),
        "max_samplewise_L1_distance": _finite_float(sample_l1.max()),
        "mean_samplewise_L2_distance": _finite_float(sample_l2.mean()),
        "max_samplewise_L2_distance": _finite_float(sample_l2.max()),
        "normalized_frobenius_distance": _finite_float(
            np.linalg.norm(difference.reshape(-1))
            / (np.linalg.norm(reference.reshape(-1)) + eps)
        ),
    }


def pairwise_strength_metrics(cycle, comparator, eps=EPSILON):
    reference = _as_numpy(cycle, dtype=np.float64).reshape(-1)
    candidate = _as_numpy(comparator, dtype=np.float64).reshape(-1)
    _require(reference.shape == candidate.shape, "strength pair shape mismatch")
    difference = reference - candidate
    return {
        "exact_equality": bool(np.array_equal(reference, candidate)),
        "mean_absolute_difference": _finite_float(np.mean(np.abs(difference))),
        "max_absolute_difference": _finite_float(np.max(np.abs(difference))),
        "RMSE": _finite_float(np.sqrt(np.mean(np.square(difference)))),
        "pearson": _correlation_record(reference, candidate, "pearson"),
        "spearman": _correlation_record(reference, candidate, "spearman"),
        "cycle_std": _finite_float(np.std(reference)),
        "comparator_std": _finite_float(np.std(candidate)),
        "cycle_mean": _finite_float(np.mean(reference)),
        "comparator_mean": _finite_float(np.mean(candidate)),
        "normalized_L2_distance": _finite_float(
            np.linalg.norm(difference) / (np.linalg.norm(reference) + eps)
        ),
    }


def effective_action_tensor(sample_strength, target_local):
    """Expose the existing a_i * T_local target side; never used for training."""
    strength = c1.frozen_tensor(sample_strength, dtype=torch.float32, device="cpu")
    target = c1.frozen_tensor(target_local, dtype=torch.float32, device="cpu")
    _require(
        strength.ndim == 1
        and target.ndim == 3
        and target.shape[0] == strength.shape[0]
        and target.shape[1:] == (c1.VIEW_NUM, c1.CLASS_NUM),
        "effective action shape mismatch",
    )
    action = (strength[:, None, None] * target).detach()
    _require(
        tuple(action.shape) == (strength.shape[0], c1.VIEW_NUM, c1.CLASS_NUM)
        and not action.requires_grad
        and action.grad_fn is None,
        "effective action did not remain diagnostic-only",
    )
    return action


def pairwise_effective_action_metrics(cycle, comparator, eps=EPSILON):
    reference = _as_numpy(cycle, dtype=np.float64)
    candidate = _as_numpy(comparator, dtype=np.float64)
    _require(reference.shape == candidate.shape and reference.ndim == 3, "action pair shape mismatch")
    difference = reference - candidate
    sample_l1 = np.abs(difference).sum(axis=(1, 2))
    argmax_equal = np.argmax(reference, axis=2) == np.argmax(candidate, axis=2)
    return {
        "exact_equality": bool(np.array_equal(reference, candidate)),
        "mean_absolute_difference": _finite_float(np.mean(np.abs(difference))),
        "max_absolute_difference": _finite_float(np.max(np.abs(difference))),
        "normalized_frobenius_distance": _finite_float(
            np.linalg.norm(difference.reshape(-1))
            / (np.linalg.norm(reference.reshape(-1)) + eps)
        ),
        "per_sample_L1_mean": _finite_float(sample_l1.mean()),
        "per_sample_L1_median": _finite_float(np.median(sample_l1)),
        "per_sample_L1_max": _finite_float(sample_l1.max()),
        "effective_action_argmax_agreement": _finite_float(argmax_equal.mean()),
    }


def build_target_action_diagnostics(targets, target_provenance, M0_audit):
    target_pairs = OrderedDict()
    strength_pairs = OrderedDict()
    action_pairs = OrderedDict()
    actions = OrderedDict()
    for arm in PSEUDO_ARMS:
        actions[arm] = effective_action_tensor(
            targets[arm]["sample_strength"], targets[arm]["target_local"]
        )
    for arm in COMPARATORS:
        key = "CYCLE_vs_" + arm
        target_pairs[key] = pairwise_target_metrics(
            targets["CYCLE"]["target_global"], targets[arm]["target_global"]
        )
        strength_pairs[key] = pairwise_strength_metrics(
            targets["CYCLE"]["sample_strength"], targets[arm]["sample_strength"]
        )
        action_pairs[key] = pairwise_effective_action_metrics(
            actions["CYCLE"], actions[arm]
        )
    target_record = {
        "stage": STAGE,
        "layer": "2A",
        "C1_formula_modified": False,
        "global_to_local_formula": "T_global @ M0",
        "M0_audit": M0_audit,
        "reconstruction_provenance": target_provenance,
        "comparisons": target_pairs,
    }
    strength_record = {
        "stage": STAGE,
        "layer": "2B",
        "mass_normalization_contract": "a = m / (mean(m) + 1e-12)",
        "arm_mean_strength": {
            arm: _finite_float(targets[arm]["sample_strength"].mean().item())
            for arm in PSEUDO_ARMS
        },
        "comparisons": strength_pairs,
    }
    action_record = {
        "stage": STAGE,
        "layer": 3,
        "diagnostic_only": True,
        "used_for_training": False,
        "formula": "A[i,v,c] = a[i] * T_local[i,v,c]",
        "arm_shape": {arm: list(actions[arm].shape) for arm in PSEUDO_ARMS},
        "arm_logical_sha256": {
            arm: tensor_sha256(actions[arm].numpy()) for arm in PSEUDO_ARMS
        },
        "comparisons": action_pairs,
    }
    return actions, target_record, strength_record, action_record


def c1_epoch1_batches(sample_ids, batch_size=c1.BATCH_SIZE, seed=c1.SEED):
    """Replay DataLoader's exact first-epoch shuffle and explicit row IDs."""
    ids = _as_numpy(sample_ids, dtype=np.int64).reshape(-1)
    _require(
        np.array_equal(ids, np.arange(ids.size, dtype=np.int64)),
        "sample IDs are not canonical",
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(ids))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
        generator=generator,
        num_workers=0,
    )
    batches = [packed[0].numpy().astype(np.int64, copy=True) for packed in loader]
    order = np.concatenate(batches)
    _require(
        order.size == ids.size
        and np.array_equal(np.sort(order), ids)
        and np.unique(order).size == ids.size,
        "sample order does not cover canonical IDs exactly once",
    )
    return batches, {
        "sample_ids_canonical": True,
        "sample_count": int(ids.size),
        "batch_size": int(batch_size),
        "batch_count": len(batches),
        "batch_boundaries": [int(batch.size) for batch in batches],
        "sample_order_sha256": tensor_sha256(order),
        "sample_order": order,
    }


def training_sensitive_module_report(model):
    """Find layers that make C1 train-mode forwards stochastic or stateful."""
    dropout_types = (
        torch.nn.Dropout,
        torch.nn.Dropout1d,
        torch.nn.Dropout2d,
        torch.nn.Dropout3d,
        torch.nn.AlphaDropout,
        torch.nn.FeatureAlphaDropout,
    )
    records = []
    for view_id, autoencoder in enumerate(model.autoencoders):
        for name, module in autoencoder.named_modules():
            reason = None
            if isinstance(module, dropout_types):
                reason = "dropout"
            elif isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                reason = "batchnorm_running_state"
            elif isinstance(module, torch.nn.RNNBase) and float(module.dropout) > 0.0:
                reason = "recurrent_dropout"
            elif (
                hasattr(module, "track_running_stats")
                and bool(getattr(module, "track_running_stats"))
                and hasattr(module, "running_mean")
            ):
                reason = "tracked_running_state"
            if reason is not None:
                records.append(
                    {
                        "view_id": int(view_id),
                        "module_name": name,
                        "module_class": module.__class__.__name__,
                        "reason": reason,
                    }
                )
    return {
        "C1_runner_model_mode": "train",
        "training_sensitive_module_count": len(records),
        "training_sensitive_modules": records,
        "hard_fail_required": bool(records),
        "safe_to_replay_C1_train_mode": not records,
    }


def require_c1_model_mode_safe(model):
    report = training_sensitive_module_report(model)
    _require(
        not report["hard_fail_required"],
        "Dropout/BatchNorm/training-sensitive module forbids read-only replay",
    )
    _require(
        all(autoencoder.training for autoencoder in model.autoencoders),
        "model mode differs from the C1 runner train mode",
    )
    return report


def named_trainable_autoencoder_parameters(model):
    records = []
    for view_id, autoencoder in enumerate(model.autoencoders):
        for name, parameter in autoencoder.named_parameters():
            if parameter.requires_grad:
                records.append((int(view_id), "view%d.%s" % (view_id, name), parameter))
    _require(records, "model has no trainable autoencoder parameters")
    _require(len({id(record[2]) for record in records}) == len(records), "duplicate parameter")
    return records


def new_cpu_float64_accumulators(parameter_records):
    return [
        torch.zeros_like(parameter.detach(), device="cpu", dtype=torch.float64)
        for _, _, parameter in parameter_records
    ]


def accumulate_batch_gradient(
    loss,
    parameter_records,
    accumulators,
    batch_sample_count,
    full_sample_count,
):
    """Accumulate one autograd.grad result with B/N full-mean weighting."""
    parameters = [record[2] for record in parameter_records]
    _require(len(parameters) == len(accumulators), "gradient accumulator mismatch")
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    weight = float(batch_sample_count) / float(full_sample_count)
    unused = []
    for index, (gradient, accumulator) in enumerate(zip(gradients, accumulators)):
        if gradient is None:
            unused.append(index)
        else:
            accumulator.add_(gradient.detach().to(device="cpu", dtype=torch.float64), alpha=weight)
    return unused


def _forward_model(model, x_views):
    reconstructions = []
    q_views = []
    for view_id, x_view in enumerate(x_views):
        reconstruction, _, q_local = model.autoencoders[view_id](x_view)
        reconstructions.append(reconstruction)
        q_views.append(q_local)
    return reconstructions, q_views, torch.stack(q_views, dim=1)


def _batch_views(views, sample_ids_batch, device):
    ids = _as_numpy(sample_ids_batch, dtype=np.int64)
    return [
        torch.from_numpy(np.ascontiguousarray(view[ids])).to(device)
        for view in views
    ]


def _gradient_hash(names, accumulators):
    digest = hashlib.sha256()
    for name, value in zip(names, accumulators):
        raw = value.detach().cpu().contiguous().numpy().tobytes(order="C")
        shape = ",".join(str(int(size)) for size in value.shape).encode("ascii")
        for field in (
            name.encode("utf-8"),
            str(value.dtype).encode("ascii"),
            shape,
            raw,
        ):
            digest.update(struct.pack(">Q", len(field)))
            digest.update(field)
    return digest.hexdigest()


def gradient_norm_record(accumulators, view_ids):
    _require(len(accumulators) == len(view_ids), "gradient/view mismatch")
    total_square = sum(float(value.square().sum().item()) for value in accumulators)
    per_view = {}
    for view_id in range(c1.VIEW_NUM):
        square = sum(
            float(value.square().sum().item())
            for value, owner in zip(accumulators, view_ids)
            if owner == view_id
        )
        per_view[str(view_id)] = _finite_float(math.sqrt(square))
    return {
        "norm": _finite_float(math.sqrt(total_square)),
        "per_view_norm": per_view,
    }


def gradient_pair_metrics(cycle, comparator, view_ids, eps=EPSILON):
    _require(len(cycle) == len(comparator) == len(view_ids), "gradient pair mismatch")

    def components(indices):
        dot = 0.0
        left_square = 0.0
        right_square = 0.0
        difference_square = 0.0
        for index in indices:
            left = cycle[index]
            right = comparator[index]
            dot += float((left * right).sum().item())
            left_square += float(left.square().sum().item())
            right_square += float(right.square().sum().item())
            difference_square += float((left - right).square().sum().item())
        left_norm = math.sqrt(left_square)
        right_norm = math.sqrt(right_square)
        difference_norm = math.sqrt(difference_square)
        cosine = None
        if left_norm > 0.0 and right_norm > 0.0:
            cosine = _finite_float(dot / (left_norm * right_norm + eps))
        return {
            "cosine": cosine,
            "cycle_norm": _finite_float(left_norm),
            "comparator_norm": _finite_float(right_norm),
            "difference_norm": _finite_float(difference_norm),
            "relative_gradient_difference": _finite_float(
                difference_norm / (left_norm + eps)
            ),
            "norm_ratio_to_cycle": _finite_float(right_norm / (left_norm + eps)),
        }

    overall = components(range(len(cycle)))
    overall["per_view_cosine"] = {
        str(view_id): components(
            [index for index, owner in enumerate(view_ids) if owner == view_id]
        )["cosine"]
        for view_id in range(c1.VIEW_NUM)
    }
    return overall


def analytic_total_gradient_metrics(native, cycle, comparator, eps=EPSILON):
    _require(len(native) == len(cycle) == len(comparator), "total gradient mismatch")
    dot = 0.0
    cycle_square = 0.0
    comparator_square = 0.0
    difference_square = 0.0
    for native_value, cycle_value, comparator_value in zip(native, cycle, comparator):
        total_cycle = native_value + c1.LAMBDA_PSEUDO * cycle_value
        total_comparator = native_value + c1.LAMBDA_PSEUDO * comparator_value
        dot += float((total_cycle * total_comparator).sum().item())
        cycle_square += float(total_cycle.square().sum().item())
        comparator_square += float(total_comparator.square().sum().item())
        difference_square += float((total_cycle - total_comparator).square().sum().item())
    cycle_norm = math.sqrt(cycle_square)
    comparator_norm = math.sqrt(comparator_square)
    difference_norm = math.sqrt(difference_square)
    cosine = None
    if cycle_norm > 0.0 and comparator_norm > 0.0:
        cosine = _finite_float(dot / (cycle_norm * comparator_norm + eps))
    return {
        "formula": "g_total_X = g_native + 0.01 * g_pseudo_X",
        "lambda_pseudo": c1.LAMBDA_PSEUDO,
        "cosine": cosine,
        "cycle_total_norm": _finite_float(cycle_norm),
        "comparator_total_norm": _finite_float(comparator_norm),
        "difference_norm": _finite_float(difference_norm),
        "relative_total_gradient_difference": _finite_float(
            difference_norm / (cycle_norm + eps)
        ),
    }


def pseudo_loss_and_gradient(model, views, batches, pseudo_target, device):
    """Layer 4/5 diagnostic using autograd.grad only and no state mutation."""
    mode_audit = require_c1_model_mode_safe(model)
    parameter_records = named_trainable_autoencoder_parameters(model)
    names = [record[1] for record in parameter_records]
    view_ids = [record[0] for record in parameter_records]
    accumulators = new_cpu_float64_accumulators(parameter_records)
    model_hash_before = hash_backbone(model.autoencoders)
    _require(
        model_hash_before["aggregate"] == c1.EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256,
        "initial E1 aggregate hash mismatch",
    )
    _require(all(record[2].grad is None for record in parameter_records), "pre-existing gradients")
    full_count = sum(len(batch) for batch in batches)
    loss_mean = 0.0
    unused_parameter_indices = set()
    for batch in batches:
        x_views = _batch_views(views, batch, device)
        _, _, q_local = _forward_model(model, x_views)
        target_batch, strength_batch = c1.pseudo_batch_by_sample_ids(
            pseudo_target, torch.from_numpy(batch)
        )
        loss = c1.soft_pseudo_cross_entropy(q_local, target_batch, strength_batch)
        batch_weight = float(len(batch)) / float(full_count)
        loss_mean += float(loss.detach().item()) * batch_weight
        unused_parameter_indices.update(
            accumulate_batch_gradient(
                loss, parameter_records, accumulators, len(batch), full_count
            )
        )
    model_hash_after = hash_backbone(model.autoencoders)
    _require(model_hash_before == model_hash_after, "pseudo diagnostic mutated model")
    _require(all(record[2].grad is None for record in parameter_records), "autograd.grad populated .grad")
    norm = gradient_norm_record(accumulators, view_ids)
    return accumulators, {
        "full_dataset_mean_pseudo_loss": _finite_float(loss_mean),
        "gradient": {
            **norm,
            "logical_sha256": _gradient_hash(names, accumulators),
            "parameter_count": len(parameter_records),
            "unused_parameter_count": len(unused_parameter_indices),
            "all_trainable_autoencoder_parameters_included": True,
        },
        "parameter_names": names,
        "parameter_view_ids": view_ids,
        "model_hash_before": model_hash_before,
        "model_hash_after": model_hash_after,
        "model_hash_before_after_equal": True,
        "initial_aggregate_hash_pass": True,
        "parameter_grad_fields_untouched": True,
        "autograd_grad_used_for_diagnostic_only": True,
        "batch_weighting": "batch_sample_count / 1400",
        "model_mode_audit": mode_audit,
    }


def native_loss_and_gradient(model, views, batches, device):
    """Layer 6 exact epoch-1 REC + 0.01 CLU + 0.01 LWC gradient."""
    mode_audit = require_c1_model_mode_safe(model)
    parameter_records = named_trainable_autoencoder_parameters(model)
    names = [record[1] for record in parameter_records]
    view_ids = [record[0] for record in parameter_records]
    accumulators = new_cpu_float64_accumulators(parameter_records)
    model_hash_before = hash_backbone(model.autoencoders)
    _require(
        model_hash_before["aggregate"] == c1.EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256,
        "initial E1 aggregate hash mismatch",
    )
    _require(all(record[2].grad is None for record in parameter_records), "pre-existing gradients")
    full_views = [torch.from_numpy(view) for view in views]
    view_weights = [1.0] * c1.VIEW_NUM
    native_p_all, native_matches, _, _ = c1_train.e1_train.refresh_native_target(
        model, full_views, view_weights, device
    )
    _require(
        hash_backbone(model.autoencoders) == model_hash_before,
        "native target refresh mutated model",
    )
    full_count = sum(len(batch) for batch in batches)
    loss_sums = {"REC": 0.0, "CLU": 0.0, "LWC": 0.0, "native": 0.0}
    unused_parameter_indices = set()
    for batch in batches:
        x_views = _batch_views(views, batch, device)
        reconstructions, q_views, q_local = _forward_model(model, x_views)
        p_batch = native_p_all[torch.from_numpy(batch)].to(device)
        native_view_losses, native_diagnostics = c1_train.e1_train.native_mvcan_losses(
            x_views, reconstructions, q_views, p_batch, native_matches
        )
        rec_total = torch.stack([record[0] for record in native_diagnostics]).sum()
        clu_total = torch.stack([record[1] for record in native_diagnostics]).sum()
        rec_clu_total = torch.stack(native_view_losses).sum()
        lwc_loss, _ = c1_train.build_lwc_loss(q_local, native_matches)
        native_loss = rec_clu_total + c1_train.e1_train.LAMBDA1 * lwc_loss
        batch_weight = float(len(batch)) / float(full_count)
        for key, value in (
            ("REC", rec_total),
            ("CLU", clu_total),
            ("LWC", lwc_loss),
            ("native", native_loss),
        ):
            loss_sums[key] += float(value.detach().item()) * batch_weight
        unused_parameter_indices.update(
            accumulate_batch_gradient(
                native_loss, parameter_records, accumulators, len(batch), full_count
            )
        )
    model_hash_after = hash_backbone(model.autoencoders)
    _require(model_hash_before == model_hash_after, "native diagnostic mutated model")
    _require(all(record[2].grad is None for record in parameter_records), "autograd.grad populated .grad")
    _require(not unused_parameter_indices, "native objective omitted a trainable parameter")
    norm = gradient_norm_record(accumulators, view_ids)
    return accumulators, {
        "native_objective": "REC + 0.01 CLU + 0.01 LWC",
        "full_dataset_mean_losses": {
            key: _finite_float(value) for key, value in loss_sums.items()
        },
        "gradient": {
            **norm,
            "logical_sha256": _gradient_hash(names, accumulators),
            "parameter_count": len(parameter_records),
            "unused_parameter_count": 0,
            "all_trainable_autoencoder_parameters_included": True,
        },
        "parameter_names": names,
        "parameter_view_ids": view_ids,
        "model_hash_before": model_hash_before,
        "model_hash_after": model_hash_after,
        "model_hash_before_after_equal": True,
        "initial_aggregate_hash_pass": True,
        "parameter_grad_fields_untouched": True,
        "autograd_grad_used_for_diagnostic_only": True,
        "batch_weighting": "batch_sample_count / 1400",
        "native_target_refresh_count": 1,
        "native_target_refresh_protocol": "C1-A1 epoch-1 exact refresh",
        "native_P_all_rewritten": False,
        "P_corr_used": False,
        "P_util_used": False,
        "model_mode_audit": mode_audit,
    }


def gradient_scale_metrics(native, cycle, comparator, view_ids, eps=EPSILON):
    pseudo_pair = gradient_pair_metrics(cycle, comparator, view_ids, eps=eps)
    native_norm = gradient_norm_record(native, view_ids)["norm"]
    cycle_norm = pseudo_pair["cycle_norm"]
    comparator_norm = pseudo_pair["comparator_norm"]
    difference_norm = pseudo_pair["difference_norm"]
    return {
        "pseudo_pair": pseudo_pair,
        "cycle_pseudo_to_native_gradient_ratio": _finite_float(
            c1.LAMBDA_PSEUDO * cycle_norm / (native_norm + eps)
        ),
        "comparator_pseudo_to_native_gradient_ratio": _finite_float(
            c1.LAMBDA_PSEUDO * comparator_norm / (native_norm + eps)
        ),
        "pseudo_difference_to_native": _finite_float(
            c1.LAMBDA_PSEUDO * difference_norm / (native_norm + eps)
        ),
        "total_update": analytic_total_gradient_metrics(
            native, cycle, comparator, eps=eps
        ),
    }


def audit_boundary_record():
    """Static contract written beside successful diagnostic results."""
    return {
        "stage": STAGE,
        "training_used": False,
        "optimizer_created": False,
        "optimizer_step_used": False,
        "parameter_update_used": False,
        "autograd_grad_used_for_diagnostic_only": True,
        "GT_loaded": False,
        "R_loaded": False,
        "sparse_labels_loaded": False,
        "corruption_mask_loaded": False,
        "oracle_used": False,
        "C0_scores_recomputed": False,
        "C1_formula_modified": False,
        "native_P_all_rewritten": False,
        "P_corr_used": False,
        "P_util_used": False,
        "Memory_used": False,
        "dynamic_cycle_used": False,
        "threshold_used": False,
        "top_k_used": False,
        "lambda_sweep_used": False,
        "new_semantic_formula_used": False,
        "new_pseudo_label_formula_used": False,
        "scientific_gate_used": False,
        "automatic_descriptive_label_selected": False,
        "allowed_descriptive_labels_for_human_review": list(
            ALLOWED_DESCRIPTIVE_LABELS
        ),
    }
