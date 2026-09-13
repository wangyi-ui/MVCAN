"""Frozen read-only protocol for C5-A0 pseudo-semantic feasibility.

Sparse labels determine semantic direction; frozen directional ``U_cycle``
determines action validity.  C5 never trains a model and never feeds a result
back into MVCAN.  Its PERMUTED_U control is a preregistered deterministic
complete-row permutation, not C3-B0's historical semantic-shuffle arm.
"""

import hashlib
from collections import OrderedDict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from experiments.cyclic_utility import c0_complementary_semantic_verification as c0
from experiments.cyclic_utility import c3_a0_utility_conditioned_action_granularity_protocol as c3a0
from experiments.cyclic_utility import c3_b0_relation_action_protocol as c3b0
from irv.b4_information_utility import tensor_sha256


STAGE = "C5-A0"
SEEDS = (20, 30, 50)
ARMS = ("UNIFORM", "TRUE_U", "PERMUTED_U")
N = 1400
V = 6
K = 7
S = 20
L = 14
NU = 1386
GENERATOR_SIZE = 3
PERMUTATION_NAMESPACE = "C5A0_UTILITY_PERMUTE_ELIGIBLE_V2:"
EPSILON = 1e-12
ROW_SUM_ATOL = 1e-6
EXPECTED_GENERATOR_SUBSETS_SHA256 = (
    "1da78797e059f952bf6ae3749fc94ca2dd5bebf83d76b1e8eeb6a03583fe36da"
)
EXPECTED_VERIFIER_SUBSETS_SHA256 = (
    "aa726d58832d5eb17ff952049d6be1b3c6eb7295b6fcba6b656c3c1ead80baca"
)
COVERAGES = (0.10, 0.20, 0.30, 0.50, 1.00)
PRE_GT_ARRAY_KEYS = (
    "sample_ids",
    "labeled_ids",
    "unlabeled_ids",
    "eligible_mask",
    "eligible_ids",
    "abstained_ids",
    "utility_mass",
    "generator_subsets",
    "verifier_subsets",
    "permutation_source_ids",
    "U_true",
    "U_true_eligible",
    "U_permuted",
    "directional_evidence",
    "directional_posterior",
    "directional_pred",
    "directional_margin",
    "eligible_directional_posterior",
    "posterior_UNIFORM",
    "posterior_TRUE_U",
    "posterior_PERMUTED_U",
    "prediction_UNIFORM",
    "prediction_TRUE_U",
    "prediction_PERMUTED_U",
    "confidence_UNIFORM",
    "confidence_TRUE_U",
    "confidence_PERMUTED_U",
)


def _require(condition, message="C5_A0_PROTOCOL_FAIL_CLOSED"):
    if not condition:
        raise RuntimeError(message)


def _readonly(value, dtype=None):
    array = np.array(np.asarray(value, dtype=dtype), copy=True, order="C")
    array.setflags(write=False)
    return array


def validate_seed(seed):
    value = int(seed)
    if value not in SEEDS:
        raise ValueError("C5-A0 seed must be one of " + str(SEEDS))
    return value


def logical_sha256(value):
    return tensor_sha256(value)


def frozen_direction_definitions():
    generator, verifier = c0.enumerate_complementary_splits()
    return validate_direction_definitions(generator, verifier)


def validate_direction_definitions(generator_subsets, verifier_subsets):
    generator = _readonly(generator_subsets, np.int64)
    verifier = _readonly(verifier_subsets, np.int64)
    _require(
        generator.shape == verifier.shape == (S, GENERATOR_SIZE)
        and np.array_equal(
            np.sort(np.concatenate((generator, verifier), axis=1)),
            np.tile(np.arange(V, dtype=np.int64), (S, 1)),
        )
        and logical_sha256(generator) == EXPECTED_GENERATOR_SUBSETS_SHA256
        and logical_sha256(verifier) == EXPECTED_VERIFIER_SUBSETS_SHA256,
        "C5_A0_DIRECTION_DEFINITION_FAIL_CLOSED",
    )
    return generator, verifier


def validate_sparse_inputs(labeled_ids, labeled_targets, unlabeled_ids):
    split = c3a0.validate_fixed_sparse_split(
        labeled_ids, unlabeled_ids, labeled_targets
    )
    labeled = _readonly(split["labeled_sample_ids"], np.int64)
    targets = _readonly(split["labeled_targets"], np.int64)
    unlabeled = _readonly(split["unlabeled_ids"], np.int64)
    _require(
        labeled.shape == (L,)
        and targets.shape == (L,)
        and unlabeled.shape == (NU,)
        and all(np.count_nonzero(targets == class_id) == 2 for class_id in range(K)),
        "C5_A0_ALL_14_ANCHORS_REQUIRED_FAIL_CLOSED",
    )
    return labeled, targets, unlabeled


def build_directional_semantic_candidates(
    q_aligned,
    labeled_ids,
    labeled_targets,
    unlabeled_ids,
    generator_subsets,
):
    """Build generator-only pseudo-semantic candidates.

    Parameters
    ----------
    q_aligned : torch.Tensor [N,V,K] = [1400,6,7]
    labeled_ids : [L] = [14]
    labeled_targets : [L] = [14]
    unlabeled_ids : [Nu] = [1386]
    generator_subsets : [S,3] = [20,3]

    Returns
    -------
    directional_evidence : [Nu,S,K] = [1386,20,7]
    directional_posterior : [Nu,S,K] = [1386,20,7]
    directional_pred : [Nu,S] = [1386,20]
    directional_margin : [Nu,S] = [1386,20]
    """
    _require(
        torch.is_tensor(q_aligned)
        and tuple(q_aligned.shape) == (N, V, K)
        and bool(torch.isfinite(q_aligned).all().item())
        and bool((q_aligned >= 0).all().item()),
        "C5_A0_Q_ALIGNED_BOUNDARY_FAIL_CLOSED",
    )
    frozen_generator, _ = frozen_direction_definitions()
    generator = _readonly(generator_subsets, np.int64)
    _require(
        generator.shape == (S, GENERATOR_SIZE)
        and np.array_equal(generator, frozen_generator),
        "C5_A0_GENERATOR_DEFINITION_FAIL_CLOSED",
    )
    labeled, targets, unlabeled = validate_sparse_inputs(
        labeled_ids, labeled_targets, unlabeled_ids
    )
    device = q_aligned.device
    labeled_index = torch.as_tensor(
        np.array(labeled, copy=True), device=device, dtype=torch.long
    )
    unlabeled_index = torch.as_tensor(
        np.array(unlabeled, copy=True), device=device, dtype=torch.long
    )
    with torch.no_grad():
        q = q_aligned.detach()
        anchor_q = q[labeled_index]
        query_q = q[unlabeled_index]
        relation_by_view = [
            c3b0.relation_probability(query_q[:, view_id, :], anchor_q[:, view_id, :]).detach()
            for view_id in range(V)
        ]
        evidence = torch.empty((NU, S, K), dtype=q.dtype, device=device)
        for direction_id in range(S):
            views = generator[direction_id]
            for class_id in range(K):
                anchor_positions = np.flatnonzero(targets == class_id)
                pieces = [
                    relation_by_view[int(view_id)][:, anchor_positions]
                    for view_id in views
                ]
                evidence[:, direction_id, class_id] = torch.stack(
                    pieces, dim=1
                ).mean(dim=(1, 2))
        evidence = evidence.detach()
        denominator = evidence.sum(dim=2, keepdim=True)
        _require(
            bool(torch.isfinite(evidence).all().item())
            and bool((evidence >= 0).all().item())
            and bool((denominator > EPSILON).all().item()),
            "C5_A0_DIRECTIONAL_EVIDENCE_FAIL_CLOSED",
        )
        posterior = (evidence / (denominator + EPSILON)).detach()
        top_two = torch.topk(posterior, 2, dim=2).values
        prediction = torch.argmax(posterior, dim=2).detach()
        margin = (top_two[:, :, 0] - top_two[:, :, 1]).detach()
    validate_posterior(posterior, expected_shape=(NU, S, K))
    _require(
        tuple(prediction.shape) == (NU, S)
        and tuple(margin.shape) == (NU, S)
        and bool(torch.isfinite(margin).all().item()),
        "C5_A0_DIRECTIONAL_OUTPUT_FAIL_CLOSED",
    )
    return {
        "directional_evidence": evidence,
        "directional_posterior": posterior,
        "directional_pred": prediction,
        "directional_margin": margin,
        "all_14_sparse_anchors_used": True,
        "anchors_per_class": 2,
        "generator_views_only": True,
        "verifier_views_used_for_candidate": False,
    }


def validate_action_weights(action_weights):
    _require(torch.is_tensor(action_weights), "C5_A0_ACTION_WEIGHT_FAIL_CLOSED")
    weights = action_weights.detach()
    _require(
        weights.ndim == 2
        and weights.shape[0] > 0
        and weights.shape[1] == S
        and bool(torch.isfinite(weights).all().item())
        and bool((weights >= 0).all().item())
        and bool((weights.sum(dim=1) > EPSILON).all().item()),
        "C5_A0_ACTION_WEIGHT_FAIL_CLOSED",
    )
    return weights


def validate_posterior(posterior, expected_shape=(NU, K)):
    _require(torch.is_tensor(posterior), "C5_A0_POSTERIOR_FAIL_CLOSED")
    values = posterior.detach()
    _require(
        tuple(values.shape) == tuple(expected_shape)
        and not values.requires_grad
        and values.grad_fn is None
        and bool(torch.isfinite(values).all().item())
        and bool((values >= 0).all().item())
        and torch.allclose(
            values.sum(dim=-1), torch.ones_like(values[..., 0]),
            rtol=0.0, atol=ROW_SUM_ATOL,
        ),
        "C5_A0_POSTERIOR_FAIL_CLOSED",
    )
    return True


def aggregate_pseudo_semantics(directional_posterior, action_weights):
    """Aggregate eligible [Ne,S,K] candidates with positive [Ne,S] weights."""
    _require(
        torch.is_tensor(directional_posterior)
        and directional_posterior.ndim == 3
        and directional_posterior.shape[0] > 0
        and tuple(directional_posterior.shape[1:]) == (S, K),
        "C5_A0_DIRECTIONAL_POSTERIOR_SHAPE_FAIL_CLOSED",
    )
    directional = directional_posterior.detach()
    row_count = int(directional.shape[0])
    validate_posterior(directional, expected_shape=(row_count, S, K))
    weights = validate_action_weights(action_weights).to(
        device=directional.device, dtype=directional.dtype
    )
    _require(int(weights.shape[0]) == row_count, "C5_A0_ACTION_WEIGHT_FAIL_CLOSED")
    with torch.no_grad():
        weighted = directional * weights.unsqueeze(-1)
        numerator = weighted.sum(dim=1)
        denominator = weights.sum(dim=1, keepdim=True)
        posterior = (numerator / (denominator + EPSILON)).detach()
        top_two = torch.topk(posterior, 2, dim=1).values
        prediction = torch.argmax(posterior, dim=1).detach()
        confidence = (top_two[:, 0] - top_two[:, 1]).detach()
    validate_posterior(posterior, expected_shape=(row_count, K))
    _require(
        tuple(prediction.shape) == tuple(confidence.shape) == (row_count,)
        and bool(torch.isfinite(confidence).all().item()),
        "C5_A0_AGGREGATE_OUTPUT_FAIL_CLOSED",
    )
    return {
        "posterior": posterior,
        "prediction": prediction,
        "confidence": confidence,
    }


def extract_unlabeled_utility(U_cycle, sample_ids, unlabeled_ids):
    sample = _readonly(sample_ids, np.int64)
    cycle = _readonly(U_cycle, np.float64)
    unlabeled = _readonly(unlabeled_ids, np.int64)
    _require(
        sample.shape == (N,)
        and np.array_equal(sample, np.arange(N, dtype=np.int64))
        and cycle.shape == (N, S)
        and np.isfinite(cycle).all()
        and np.all(cycle >= 0)
        and unlabeled.shape == (NU,)
        and np.unique(unlabeled).size == NU
        and np.all((unlabeled >= 0) & (unlabeled < N)),
        "C5_A0_U_CYCLE_BOUNDARY_FAIL_CLOSED",
    )
    return _readonly(cycle[unlabeled], np.float64)


def build_zero_mass_abstention(U_true, unlabeled_ids):
    """Freeze one common eligible set before constructing any C5 arm."""
    true = _readonly(U_true, np.float64)
    ids = _readonly(unlabeled_ids, np.int64)
    _require(
        true.shape == (NU, S)
        and ids.shape == (NU,)
        and np.unique(ids).size == NU
        and np.isfinite(true).all()
        and np.all(true >= 0),
        "C5_A0_ABSTENTION_BOUNDARY_FAIL_CLOSED",
    )
    mass = np.ascontiguousarray(true.sum(axis=1), dtype=np.float64)
    eligible_mask = np.ascontiguousarray(mass > EPSILON, dtype=np.bool_)
    eligible_ids = np.ascontiguousarray(ids[eligible_mask], dtype=np.int64)
    abstained_ids = np.ascontiguousarray(ids[~eligible_mask], dtype=np.int64)
    eligible_true = np.ascontiguousarray(true[eligible_mask], dtype=np.float64)
    eligible_count = int(eligible_ids.size)
    _require(
        eligible_count > 1
        and eligible_true.shape == (eligible_count, S)
        and np.all(eligible_true.sum(axis=1) > EPSILON),
        "C5_A0_NO_ELIGIBLE_ACTION_MASS_FAIL_CLOSED",
    )
    for value in (mass, eligible_mask, eligible_ids, abstained_ids, eligible_true):
        value.setflags(write=False)
    return {
        "eligible_mask": eligible_mask,
        "eligible_ids": eligible_ids,
        "abstained_ids": abstained_ids,
        "utility_mass": mass,
        "U_true_eligible": eligible_true,
        "eligible_count": eligible_count,
        "abstention_count": int(abstained_ids.size),
        "eligibility_rate": float(eligible_count / NU),
        "abstention_rate": float(abstained_ids.size / NU),
        "eligibility_rule": "sum_s U_true[i,s] > EPSILON",
        "same_eligible_set_for_all_arms": True,
        "uniform_fallback_for_abstained_rows": False,
    }


def _row_multiset(values):
    contiguous = np.ascontiguousarray(values)
    return sorted(contiguous[index].tobytes() for index in range(contiguous.shape[0]))


def build_permuted_utility(U_true, unlabeled_ids):
    """Cyclically shift complete utility rows in fixed SHA256 sample-ID order."""
    true = _readonly(U_true, np.float64)
    ids = _readonly(unlabeled_ids, np.int64)
    eligible_count = int(ids.size)
    _require(
        true.ndim == 2
        and true.shape == (eligible_count, S)
        and eligible_count > 1
        and ids.shape == (eligible_count,)
        and np.unique(ids).size == eligible_count
        and np.isfinite(true).all()
        and np.all(true >= 0)
        and np.all(true.sum(axis=1) > EPSILON),
        "C5_A0_PERMUTED_U_BOUNDARY_FAIL_CLOSED",
    )
    ranked = np.asarray(sorted(
        (int(sample_id) for sample_id in ids),
        key=lambda sample_id: (
            hashlib.sha256(
                (PERMUTATION_NAMESPACE + str(sample_id)).encode("utf-8")
            ).hexdigest(),
            sample_id,
        ),
    ), dtype=np.int64)
    source_ranked = np.roll(ranked, -1)
    row_for_id = {int(sample_id): row for row, sample_id in enumerate(ids)}
    source_ids = np.empty(eligible_count, dtype=np.int64)
    for target_id, source_id in zip(ranked, source_ranked):
        source_ids[row_for_id[int(target_id)]] = int(source_id)
    source_rows = np.asarray([row_for_id[int(value)] for value in source_ids], dtype=np.int64)
    permuted = np.ascontiguousarray(true[source_rows])
    direction_marginals = all(
        np.array_equal(np.sort(permuted[:, direction]), np.sort(true[:, direction]))
        for direction in range(S)
    )
    complete_rows = _row_multiset(permuted) == _row_multiset(true)
    _require(
        np.all(source_ids != ids)
        and not np.array_equal(permuted, true)
        and direction_marginals
        and complete_rows
        and np.all(permuted.sum(axis=1) > EPSILON),
        "C5_A0_PERMUTED_U_INVARIANT_FAIL_CLOSED",
    )
    source_ids.setflags(write=False)
    permuted.setflags(write=False)
    return {
        "U_permuted": permuted,
        "permutation_source_ids": source_ids,
        "permutation_namespace": PERMUTATION_NAMESPACE,
        "permutation_mapping_orientation": "permutation_source_ids[row_of_target_unlabeled_id] = source_sample_id",
        "permutation_logical_sha256": logical_sha256(source_ids),
        "U_true_logical_sha256": logical_sha256(true),
        "U_permuted_logical_sha256": logical_sha256(permuted),
        "complete_row_multiset_preserved": True,
        "every_direction_marginal_preserved": True,
        "within_row_direction_structure_preserved": True,
        "no_fixed_sample_id": True,
        "U_permuted_not_globally_equal_U_true": bool(not np.array_equal(permuted, true)),
        "every_true_row_has_positive_action_mass": True,
        "every_permuted_row_has_positive_action_mass": True,
        "eligible_count": eligible_count,
        "RNG_used": False,
        "GT_used": False,
    }


def build_three_arm_outputs(directional_posterior, U_true, U_permuted):
    true = torch.from_numpy(np.array(U_true, copy=True, order="C")).to(
        device=directional_posterior.device, dtype=directional_posterior.dtype
    ).detach()
    permuted = torch.from_numpy(np.array(U_permuted, copy=True, order="C")).to(
        device=directional_posterior.device, dtype=directional_posterior.dtype
    ).detach()
    uniform = torch.ones_like(true).detach()
    return OrderedDict((
        ("UNIFORM", aggregate_pseudo_semantics(directional_posterior, uniform)),
        ("TRUE_U", aggregate_pseudo_semantics(directional_posterior, true)),
        ("PERMUTED_U", aggregate_pseudo_semantics(directional_posterior, permuted)),
    ))


def validate_pre_gt_array_keys(arrays):
    keys = tuple(arrays.keys())
    forbidden = ("gt", "truth", "correct", "acc", "nmi", "ari", "metric")
    _require(
        keys == PRE_GT_ARRAY_KEYS
        and not any(token in key.lower() for key in keys for token in forbidden),
        "C5_A0_PRE_GT_WHITELIST_FAIL_CLOSED",
    )
    return True


def binary_auc_fail_closed(correct_action, scores):
    correct = np.asarray(correct_action, dtype=np.int8).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    _require(
        correct.shape == values.shape
        and np.unique(correct).size == 2
        and np.isfinite(values).all(),
        "C5_A0_AUC_UNDEFINED_FAIL_CLOSED",
    )
    result = float(roc_auc_score(correct, values))
    _require(np.isfinite(result), "C5_A0_AUC_UNDEFINED_FAIL_CLOSED")
    return result


def deterministic_coverage_selection(confidence, sample_ids, coverage):
    scores = np.asarray(confidence, dtype=np.float64)
    ids = np.asarray(sample_ids, dtype=np.int64)
    gamma = float(coverage)
    _require(
        scores.ndim == ids.ndim == 1
        and scores.shape == ids.shape
        and scores.size > 0
        and np.isfinite(scores).all()
        and np.unique(ids).size == ids.size
        and gamma in COVERAGES,
        "C5_A0_COVERAGE_BOUNDARY_FAIL_CLOSED",
    )
    count = int(np.floor(gamma * ids.size))
    order = np.lexsort((ids, -scores))
    selected = np.ascontiguousarray(ids[order[:count]], dtype=np.int64)
    selected.setflags(write=False)
    return selected
