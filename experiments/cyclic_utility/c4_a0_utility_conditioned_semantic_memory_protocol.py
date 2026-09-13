"""Frozen pre-GT primitives for C4-A0 semantic-memory diagnostics.

Sparse labels determine semantic direction.  Frozen ``U_cycle`` determines
directional action validity.  Every function in this module is a detached,
read-only sidecar operation; no value produced here is a training target.
"""

import hashlib
from collections import OrderedDict

import numpy as np
import torch

from experiments.cyclic_utility import c0_complementary_semantic_verification as c0
from experiments.cyclic_utility import c3_a0_utility_conditioned_action_granularity_protocol as c3a0
from experiments.cyclic_utility import c3_b0_relation_action_protocol as c3b0
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    align_semantic_probabilities,
)
from irv.b4_information_utility import tensor_sha256
from weak_quality import ndarray_sha256


STAGE = "C4-A0"
SEEDS = (20, 30, 50)
CARRIER_ARM = "TRUE_U"
MEMORY_ARMS = (
    "LABEL_ONLY", "UNIFORM_WRITE", "TRUE_U_WRITE", "SHUFFLE_U_WRITE",
)
N = SAMPLE_NUM = 1400
V = VIEW_NUM = 6
K = CLASS_NUM = 7
S = DIRECTION_COUNT = 20
LABELED_COUNT = 14
UNLABELED_COUNT = 1386
HOLDOUT_COUNT = 277
WRITER_POOL_COUNT = 1109
WRITE_FRACTION = 0.10
WRITERS_PER_CLASS = 15
TRAIN_EPOCHS = 20
HOLDOUT_NAMESPACE = "C4A0_HOLDOUT_V1:"
EXPECTED_HOLDOUT_SHA256 = (
    "f7359be4ed4870f7455160f7548f22ee45c42127d69a82bcf8ebde664e37af8e"
)
EXPECTED_WRITER_POOL_SHA256 = (
    "bbd6276d02532fd0449f7d9b3e89cb3a4a8e81d1edbe19a9100f40c118fd6248"
)
EXPECTED_GENERATOR_SUBSETS_SHA256 = (
    "1da78797e059f952bf6ae3749fc94ca2dd5bebf83d76b1e8eeb6a03583fe36da"
)
EXPECTED_VERIFIER_SUBSETS_SHA256 = (
    "aa726d58832d5eb17ff952049d6be1b3c6eb7295b6fcba6b656c3c1ead80baca"
)
PROBABILITY_ATOL = c0.PROBABILITY_ATOL
PRE_GT_ARRAY_WHITELIST = (
    "sample_ids", "labeled_ids", "unlabeled_ids", "H", "W", "E_c",
    "generator_subsets", "verifier_subsets", "epoch1_M_v", "epoch1_A0",
    "epoch1_Z0", "epoch1_M0", "semantic_margins", "writer_predicted_class",
    "class_pool_counts", "c4_shuffle_permutation",
    "final_memory_LABEL_ONLY", "final_memory_UNIFORM_WRITE",
    "final_memory_TRUE_U_WRITE", "final_memory_SHUFFLE_U_WRITE",
    "epoch20_holdout_z", "prediction_LABEL_ONLY",
    "prediction_UNIFORM_WRITE", "prediction_TRUE_U_WRITE",
    "prediction_SHUFFLE_U_WRITE", "similarity_LABEL_ONLY",
    "similarity_UNIFORM_WRITE", "similarity_TRUE_U_WRITE",
    "similarity_SHUFFLE_U_WRITE",
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _readonly(value, dtype=None):
    array = np.array(np.asarray(value, dtype=dtype), copy=True, order="C")
    array.setflags(write=False)
    return array


def validate_seed(seed):
    value = int(seed)
    if value not in SEEDS:
        raise ValueError("C4-A0 seed must be one of " + str(SEEDS))
    return value


def frozen_direction_definitions():
    """Return and hash-check the exact C0 20-direction ordering."""
    generator, verifier = c0.enumerate_complementary_splits()
    _require(
        generator.shape == verifier.shape == (S, 3)
        and tensor_sha256(generator) == EXPECTED_GENERATOR_SUBSETS_SHA256
        and tensor_sha256(verifier) == EXPECTED_VERIFIER_SUBSETS_SHA256,
        "C4-A0 frozen direction definition/hash mismatch",
    )
    return _readonly(generator, np.int64), _readonly(verifier, np.int64)


def align_q_readonly(q_local, native_matches):
    """Map [N,6,7] local q into M_v[global,local] coordinates."""
    _require(torch.is_tensor(q_local) and torch.is_tensor(native_matches),
             "C4-A0 q/M must be tensors")
    _require(tuple(q_local.shape) == (N, V, K), "C4-A0 q_local shape mismatch")
    _require(tuple(native_matches.shape) == (V, K, K), "C4-A0 M_v shape mismatch")
    with torch.no_grad():
        q_aligned, _ = align_semantic_probabilities(q_local, native_matches)
        q_aligned = q_aligned.detach()
    _require(
        not q_aligned.requires_grad and q_aligned.grad_fn is None
        and bool(torch.isfinite(q_aligned).all().item())
        and bool(torch.allclose(
            q_aligned.sum(dim=-1), torch.ones_like(q_aligned[..., 0]),
            rtol=0.0, atol=PROBABILITY_ATOL,
        )),
        "C4-A0 aligned q boundary mismatch",
    )
    return q_aligned


def build_directional_payload(q_aligned, generator_subsets=None):
    """Build detached generator-view means [N,20,7], never verifier means."""
    _require(torch.is_tensor(q_aligned), "C4-A0 q_aligned must be a tensor")
    _require(tuple(q_aligned.shape) == (N, V, K), "C4-A0 q_aligned shape mismatch")
    frozen_generator, _ = frozen_direction_definitions()
    generator = frozen_generator if generator_subsets is None else _readonly(
        generator_subsets, np.int64
    )
    _require(
        generator.shape == (S, 3)
        and tensor_sha256(generator) == EXPECTED_GENERATOR_SUBSETS_SHA256,
        "C4-A0 generator subsets mismatch",
    )
    index = torch.as_tensor(
        np.array(generator, copy=True), dtype=torch.long, device=q_aligned.device
    ).detach()
    with torch.no_grad():
        payload = q_aligned.detach()[:, index, :].mean(dim=2).detach()
    _require(
        tuple(payload.shape) == (N, S, K)
        and bool(torch.isfinite(payload).all().item())
        and not payload.requires_grad and payload.grad_fn is None,
        "C4-A0 directional payload boundary mismatch",
    )
    return payload


def fixed_memory_split(unlabeled_ids):
    """Create the seed-independent SHA256-ranked H/W memory-only split."""
    ids = _readonly(unlabeled_ids, np.int64)
    expected_unlabeled = np.setdiff1d(
        np.arange(N, dtype=np.int64), c3a0.fixed_labeled_ids()
    )
    _require(
        ids.shape == (UNLABELED_COUNT,)
        and np.array_equal(ids, expected_unlabeled),
        "C4-A0 requires the exact frozen unlabeled IDs",
    )
    ranked = sorted(
        (
            hashlib.sha256(
                (HOLDOUT_NAMESPACE + str(int(sample_id))).encode("utf-8")
            ).hexdigest(),
            int(sample_id),
        )
        for sample_id in ids
    )
    holdout = _readonly([item[1] for item in ranked[:HOLDOUT_COUNT]], np.int64)
    writer_pool = _readonly([item[1] for item in ranked[HOLDOUT_COUNT:]], np.int64)
    _require(
        holdout.shape == (HOLDOUT_COUNT,)
        and writer_pool.shape == (WRITER_POOL_COUNT,)
        and np.intersect1d(holdout, writer_pool).size == 0
        and np.array_equal(np.sort(np.concatenate((holdout, writer_pool))), ids)
        and ndarray_sha256(holdout) == EXPECTED_HOLDOUT_SHA256
        and ndarray_sha256(writer_pool) == EXPECTED_WRITER_POOL_SHA256,
        "C4-A0 H/W split/hash mismatch",
    )
    return {
        "H": holdout,
        "W": writer_pool,
        "H_logical_sha256": ndarray_sha256(holdout),
        "W_logical_sha256": ndarray_sha256(writer_pool),
        "training_seed_used": False,
        "GT_used": False,
    }


def initialize_label_memory(q_aligned, labeled_ids, labeled_targets):
    """Compute the single epoch-1 label-only A0/Z0/M0 state."""
    _require(torch.is_tensor(q_aligned) and tuple(q_aligned.shape) == (N, V, K),
             "C4-A0 initialization q shape mismatch")
    split = c3a0.validate_fixed_sparse_split(
        labeled_ids,
        np.setdiff1d(np.arange(N, dtype=np.int64), np.asarray(labeled_ids)),
        labeled_targets,
    )
    ids = split["labeled_sample_ids"]
    targets = split["labeled_targets"]
    rows = []
    with torch.no_grad():
        for class_id in range(K):
            class_ids = ids[targets == class_id]
            _require(class_ids.shape == (2,), "C4-A0 requires two anchors per class")
            index = torch.as_tensor(
                np.array(class_ids, copy=True), dtype=torch.long,
                device=q_aligned.device,
            )
            rows.append(q_aligned.detach().index_select(0, index).sum(dim=(0, 1)))
        A0 = torch.stack(rows).detach()
        Z0 = torch.full((K,), 12.0, dtype=q_aligned.dtype,
                        device=q_aligned.device).detach()
        M0 = (A0 / Z0[:, None]).detach()
    _require(
        tuple(A0.shape) == tuple(M0.shape) == (K, K)
        and tuple(Z0.shape) == (K,)
        and bool(torch.isfinite(A0).all().item())
        and bool(torch.isfinite(M0).all().item())
        and bool(torch.all(Z0 == 12.0).item()),
        "C4-A0 label-only initialization failed",
    )
    return {"A": A0, "Z": Z0, "M": M0}


def clone_initial_memory_arms(initial_state):
    """Create four independent, bitwise-identical memory states."""
    arms = OrderedDict()
    for arm in MEMORY_ARMS:
        arms[arm] = {
            name: initial_state[name].detach().clone()
            for name in ("A", "Z", "M")
        }
    reference = arms[MEMORY_ARMS[0]]
    _require(
        all(
            torch.equal(reference[name], arms[arm][name])
            and reference[name].data_ptr() != arms[arm][name].data_ptr()
            for arm in MEMORY_ARMS[1:] for name in ("A", "Z", "M")
        ),
        "C4-A0 memory-arm initialization is not independent/identical",
    )
    return arms


def freeze_epoch1_writer_eligibility(
    q_aligned, writer_pool_ids, labeled_ids, labeled_targets,
):
    """Freeze semantic-only E_c from epoch-1 q.

    Sparse labels determine semantic direction, but a labeled anchor is
    admitted for semantic propagation only when its leave-one-out class
    relation remains stronger than every competing class relation.

    IMPORTANT:
    - this anchor-consistency mask is NOT a new utility;
    - U_cycle is not accepted by this API and is not used here;
    - the mask affects writer semantic-direction estimation only;
    - label-only memory initialization remains defined elsewhere from all
      14 fixed sparse labeled anchors.
    """
    _require(
        torch.is_tensor(q_aligned) and tuple(q_aligned.shape) == (N, V, K),
        "C4-A0 eligibility q shape mismatch",
    )

    writer_ids = _readonly(writer_pool_ids, np.int64)
    _require(
        writer_ids.shape == (WRITER_POOL_COUNT,),
        "C4-A0 W shape mismatch",
    )

    labeled = _readonly(labeled_ids, np.int64)
    targets = _readonly(labeled_targets, np.int64)

    c3a0.validate_fixed_sparse_split(
        labeled,
        np.setdiff1d(np.arange(N, dtype=np.int64), labeled),
        targets,
    )

    writer_index = torch.as_tensor(
        np.array(writer_ids, copy=True),
        dtype=torch.long,
        device=q_aligned.device,
    )
    anchor_index = torch.as_tensor(
        np.array(labeled, copy=True),
        dtype=torch.long,
        device=q_aligned.device,
    )

    with torch.no_grad():
        writers = q_aligned.detach().index_select(0, writer_index)
        anchors = q_aligned.detach().index_select(0, anchor_index)

        # --------------------------------------------------------------
        # 1. Relation tensors for writer -> labeled-anchor semantics
        #
        # relation:
        #   [W, V, L] = [1109, 6, 14]
        # --------------------------------------------------------------
        relation_by_view = []

        for view_id in range(V):
            relation_by_view.append(
                c3b0.relation_probability(
                    writers[:, view_id, :],
                    anchors[:, view_id, :],
                ).detach()
            )

        relation = torch.stack(relation_by_view, dim=1)

        # --------------------------------------------------------------
        # 2. Leave-one-out labeled-anchor semantic consistency
        #
        # anchor_relation:
        #   [L, V, L] = [14, 6, 14]
        #
        # For anchor a with sparse class y_a:
        #
        #   own_score =
        #       mean relation to the OTHER anchor(s) in class y_a
        #
        #   competitor_score[c] =
        #       mean relation to anchors of class c
        #
        #   consistency_gap =
        #       own_score - max competitor_score
        #
        # keep iff gap > 0.
        #
        # This is a semantic-admission mask, not U/reliability.
        # --------------------------------------------------------------
        anchor_relation_by_view = []

        for view_id in range(V):
            anchor_relation_by_view.append(
                c3b0.relation_probability(
                    anchors[:, view_id, :],
                    anchors[:, view_id, :],
                ).detach()
            )

        anchor_relation = torch.stack(
            anchor_relation_by_view,
            dim=1,
        )  # [14, 6, 14]

        anchor_keep = np.zeros((labeled.shape[0],), dtype=np.bool_)
        anchor_gap = np.empty((labeled.shape[0],), dtype=np.float64)
        anchor_own_score = np.empty((labeled.shape[0],), dtype=np.float64)
        anchor_best_other_score = np.empty(
            (labeled.shape[0],),
            dtype=np.float64,
        )
        anchor_best_other_class = np.empty(
            (labeled.shape[0],),
            dtype=np.int64,
        )

        for anchor_pos in range(labeled.shape[0]):
            own_class = int(targets[anchor_pos])

            own_positions_np = np.flatnonzero(targets == own_class)
            own_positions_np = own_positions_np[
                own_positions_np != anchor_pos
            ]

            _require(
                own_positions_np.size >= 1,
                "C4_ANCHOR_CONSISTENCY_FAIL_CLOSED",
            )

            own_positions = torch.as_tensor(
                own_positions_np,
                dtype=torch.long,
                device=q_aligned.device,
            )

            own_score = (
                anchor_relation[anchor_pos]
                .index_select(1, own_positions)
                .mean()
            )

            competitor_scores = []

            for class_id in range(K):
                if class_id == own_class:
                    competitor_scores.append(None)
                    continue

                class_positions_np = np.flatnonzero(targets == class_id)

                _require(
                    class_positions_np.size >= 1,
                    "C4_ANCHOR_CONSISTENCY_FAIL_CLOSED",
                )

                class_positions = torch.as_tensor(
                    class_positions_np,
                    dtype=torch.long,
                    device=q_aligned.device,
                )

                score = (
                    anchor_relation[anchor_pos]
                    .index_select(1, class_positions)
                    .mean()
                )

                competitor_scores.append(score)

            best_other_class = None
            best_other_score = None

            for class_id, value in enumerate(competitor_scores):
                if value is None:
                    continue

                if (
                    best_other_score is None
                    or float(value.item()) > float(best_other_score.item())
                ):
                    best_other_score = value
                    best_other_class = class_id

            _require(
                best_other_score is not None,
                "C4_ANCHOR_CONSISTENCY_FAIL_CLOSED",
            )

            gap = float(
                own_score.item() - best_other_score.item()
            )

            anchor_gap[anchor_pos] = gap
            anchor_own_score[anchor_pos] = float(own_score.item())
            anchor_best_other_score[anchor_pos] = float(
                best_other_score.item()
            )
            anchor_best_other_class[anchor_pos] = int(
                best_other_class
            )

            anchor_keep[anchor_pos] = bool(gap > 0.0)

        # Every semantic class must retain at least one labeled anchor.
        kept_anchor_count_by_class = np.empty((K,), dtype=np.int64)

        for class_id in range(K):
            kept_anchor_count_by_class[class_id] = int(
                np.sum(
                    (targets == class_id)
                    & anchor_keep
                )
            )

            _require(
                kept_anchor_count_by_class[class_id] >= 1,
                "C4_ANCHOR_CONSISTENCY_FAIL_CLOSED",
            )

        # --------------------------------------------------------------
        # 3. Writer class scores use ONLY consistency-admitted anchors.
        #
        # This is the only change relative to C4-A0-v1.
        # No U_cycle enters the calculation.
        # --------------------------------------------------------------
        class_scores = []

        for class_id in range(K):
            class_anchor_positions = np.flatnonzero(
                (targets == class_id)
                & anchor_keep
            )

            _require(
                class_anchor_positions.size >= 1,
                "C4_ANCHOR_CONSISTENCY_FAIL_CLOSED",
            )

            positions = torch.as_tensor(
                class_anchor_positions,
                dtype=torch.long,
                device=q_aligned.device,
            )

            class_scores.append(
                relation
                .index_select(2, positions)
                .mean(dim=(1, 2))
            )

        scores = torch.stack(class_scores, dim=1).detach()

        top_two = torch.topk(
            scores,
            k=2,
            dim=1,
        ).values

        margins = (
            top_two[:, 0]
            - top_two[:, 1]
        ).detach()

        predicted = torch.argmax(
            scores,
            dim=1,
        ).detach()

    scores_np = scores.cpu().numpy()
    margins_np = margins.cpu().numpy()
    predicted_np = (
        predicted.cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )

    # --------------------------------------------------------------
    # 4. Original balanced writer budget remains unchanged:
    #
    # exactly WRITERS_PER_CLASS = 15 per predicted semantic class.
    # --------------------------------------------------------------
    eligible = np.empty(
        (K, WRITERS_PER_CLASS),
        dtype=np.int64,
    )

    pool_counts = np.empty(
        (K,),
        dtype=np.int64,
    )

    for class_id in range(K):
        positions = np.flatnonzero(
            predicted_np == class_id
        )

        pool_counts[class_id] = positions.size

        _require(
            positions.size >= WRITERS_PER_CLASS,
            "PRE_GT_PROTOCOL_FAIL_CLOSED",
        )

        order = np.lexsort(
            (
                writer_ids[positions],
                -margins_np[positions],
            )
        )

        eligible[class_id] = writer_ids[
            positions[
                order[:WRITERS_PER_CLASS]
            ]
        ]

    # Audit arrays are immutable after construction.
    for array in (
        scores_np,
        margins_np,
        predicted_np,
        eligible,
        pool_counts,
        anchor_keep,
        anchor_gap,
        anchor_own_score,
        anchor_best_other_score,
        anchor_best_other_class,
        kept_anchor_count_by_class,
    ):
        array.setflags(write=False)

    return {
        "E_c": eligible,
        "E_c_logical_sha256": ndarray_sha256(eligible),

        "semantic_scores": scores_np,
        "semantic_margins": margins_np,
        "predicted_class": predicted_np,
        "class_pool_counts": pool_counts,

        "anchor_consistency_keep_mask": anchor_keep,
        "anchor_consistency_gap": anchor_gap,
        "anchor_consistency_own_score": anchor_own_score,
        "anchor_consistency_best_other_score": (
            anchor_best_other_score
        ),
        "anchor_consistency_best_other_class": (
            anchor_best_other_class
        ),
        "kept_anchor_count_by_class": (
            kept_anchor_count_by_class
        ),

        "eligibility_epoch": 1,
        "eligibility_frozen_after_epoch1": True,

        "anchor_consistency_gate_used": True,
        "anchor_consistency_rule": (
            "LOO_OWN_MINUS_BEST_OTHER_GT_ZERO"
        ),

        "U_cycle_used": False,
        "GT_used": False,
    }

def lookup_utility_by_sample_ids(U_cycle, unlabeled_ids, sample_ids):
    """Use C3-B0's canonical sample-ID-to-unlabeled-row mapping."""
    cycle = _readonly(U_cycle, np.float64)
    unlabeled = _readonly(unlabeled_ids, np.int64)
    query = _readonly(sample_ids, np.int64)
    _require(cycle.shape == (UNLABELED_COUNT, S), "C4-A0 U_cycle shape mismatch")
    rows = c3b0._rows_for_sample_ids(unlabeled, query.reshape(-1))
    selected = _readonly(cycle[rows].reshape(query.shape + (S,)), np.float64)
    _require(np.isfinite(selected).all(), "C4-A0 selected utility non-finite")
    return selected


def build_c4_writer_utilities(U_cycle, unlabeled_ids, E_c):
    """Build true and class-local cyclic-shift utilities [7,15,20]."""
    eligible = _readonly(E_c, np.int64)
    _require(
        eligible.shape == (K, WRITERS_PER_CLASS)
        and np.unique(eligible).size == K * WRITERS_PER_CLASS,
        "C4-A0 E_c boundary mismatch",
    )
    true_utility = lookup_utility_by_sample_ids(U_cycle, unlabeled_ids, eligible)
    shuffled = np.empty_like(true_utility)
    permutation_ids = np.empty_like(eligible)
    for class_id in range(K):
        sorted_ids = np.sort(eligible[class_id])
        shifted_ids = np.roll(sorted_ids, -1)
        _require(np.all(sorted_ids != shifted_ids), "C4-A0 shuffle fixed point")
        shifted_u = lookup_utility_by_sample_ids(U_cycle, unlabeled_ids, shifted_ids)
        destination = {int(sample_id): position for position, sample_id in enumerate(eligible[class_id])}
        for sorted_position, target_id in enumerate(sorted_ids):
            target_position = destination[int(target_id)]
            shuffled[class_id, target_position] = shifted_u[sorted_position]
            permutation_ids[class_id, target_position] = shifted_ids[sorted_position]
        _require(
            np.array_equal(
                np.sort(true_utility[class_id], axis=0),
                np.sort(shuffled[class_id], axis=0),
            )
            and np.array_equal(
                np.sort(true_utility[class_id], axis=None),
                np.sort(shuffled[class_id], axis=None),
            )
            and np.sum(np.sort(true_utility[class_id], axis=None), dtype=np.float64)
            == np.sum(np.sort(shuffled[class_id], axis=None), dtype=np.float64),
            "C4-A0 shuffle did not preserve utility mass/multisets",
        )
    for value in (true_utility, shuffled, permutation_ids):
        value.setflags(write=False)
    return {
        "true": true_utility,
        "shuffle": shuffled,
        "permutation_ids": permutation_ids,
        "permutation_logical_sha256": ndarray_sha256(permutation_ids),
        "true_utility_logical_sha256": ndarray_sha256(true_utility),
        "shuffle_utility_logical_sha256": ndarray_sha256(shuffled),
        "entire_direction_vector_moves_together": True,
        "RNG_used": False,
    }


def memory_write_weights(arm, writer_utilities, *, dtype, device):
    if arm == "LABEL_ONLY":
        return None
    if arm == "UNIFORM_WRITE":
        value = np.ones((K, WRITERS_PER_CLASS, S), dtype=np.float64)
    elif arm == "TRUE_U_WRITE":
        value = writer_utilities["true"]
    elif arm == "SHUFFLE_U_WRITE":
        value = writer_utilities["shuffle"]
    else:
        raise ValueError("unknown C4-A0 memory arm")
    return torch.as_tensor(
        np.array(value, copy=True), dtype=dtype, device=device
    ).detach()


def directional_write_delta(writer_payload, weights):
    """Apply the exact 1/20 action-count normalization."""
    _require(
        torch.is_tensor(writer_payload) and torch.is_tensor(weights)
        and tuple(writer_payload.shape) == (K, WRITERS_PER_CLASS, S, K)
        and tuple(weights.shape) == (K, WRITERS_PER_CLASS, S),
        "C4-A0 write tensor shape mismatch",
    )
    with torch.no_grad():
        delta_A = torch.sum(
            weights.detach()[..., None] * writer_payload.detach(), dim=(1, 2)
        ).div(float(S)).detach()
        delta_Z = torch.sum(weights.detach(), dim=(1, 2)).div(float(S)).detach()
    _require(
        bool(torch.isfinite(delta_A).all().item())
        and bool(torch.isfinite(delta_Z).all().item()),
        "C4-A0 write delta non-finite",
    )
    return delta_A, delta_Z


def apply_memory_event(memory_arms, directional_payload, E_c, writer_utilities):
    """Return new detached A/Z/M states for one sidecar write event."""
    eligible = _readonly(E_c, np.int64)
    index = torch.as_tensor(
        np.array(eligible, copy=True), dtype=torch.long,
        device=directional_payload.device,
    )
    writer_payload = directional_payload.detach()[index]
    result = OrderedDict()
    for arm in MEMORY_ARMS:
        previous = memory_arms[arm]
        if arm == "LABEL_ONLY":
            result[arm] = {name: previous[name].detach().clone() for name in ("A", "Z", "M")}
            continue
        weights = memory_write_weights(
            arm, writer_utilities, dtype=directional_payload.dtype,
            device=directional_payload.device,
        )
        delta_A, delta_Z = directional_write_delta(writer_payload, weights)
        with torch.no_grad():
            A = (previous["A"].detach() + delta_A).detach()
            Z = (previous["Z"].detach() + delta_Z).detach()
            M = (A / Z[:, None]).detach()
        _require(
            bool(torch.isfinite(A).all().item())
            and bool(torch.isfinite(Z).all().item())
            and bool(torch.isfinite(M).all().item())
            and bool(torch.all(Z > 0).item())
            and bool(torch.allclose(
                M.sum(dim=1), torch.ones_like(Z), rtol=0.0,
                atol=PROBABILITY_ATOL,
            )),
            "C4-A0 persistent memory boundary mismatch",
        )
        result[arm] = {"A": A, "Z": Z, "M": M}
    return result


def holdout_representation(q_aligned, holdout_ids):
    """Uniform six-view read representation; this API accepts no utility."""
    ids = _readonly(holdout_ids, np.int64)
    _require(ids.shape == (HOLDOUT_COUNT,), "C4-A0 holdout shape mismatch")
    index = torch.as_tensor(np.array(ids, copy=True), dtype=torch.long,
                            device=q_aligned.device)
    with torch.no_grad():
        z = q_aligned.detach().index_select(0, index).mean(dim=1).detach()
    _require(tuple(z.shape) == (HOLDOUT_COUNT, K)
             and bool(torch.isfinite(z).all().item()),
             "C4-A0 readout representation mismatch")
    return z


def memory_readout(z, memory_arms):
    """Read all four memories using one shared detached z."""
    _require(torch.is_tensor(z) and tuple(z.shape) == (HOLDOUT_COUNT, K),
             "C4-A0 z shape mismatch")
    output = OrderedDict()
    with torch.no_grad():
        for arm in MEMORY_ARMS:
            scores = (z.detach() @ memory_arms[arm]["M"].detach().transpose(0, 1)).detach()
            predictions = torch.argmax(scores, dim=1).detach()
            output[arm] = {"scores": scores, "predictions": predictions}
    return output


def semantic_identity_safety(final_memory, initial_memory):
    """Fail closed on nonfinite/zero/collapsed/swapped memory prototypes."""
    final = torch.as_tensor(final_memory).detach()
    initial = torch.as_tensor(initial_memory).detach()
    _require(tuple(final.shape) == tuple(initial.shape) == (K, K),
             "C4-A0 drift matrix shape mismatch")
    _require(bool(torch.isfinite(final).all().item())
             and bool(torch.isfinite(initial).all().item()),
             "C4-A0 nonfinite semantic prototype")
    final_norm = torch.linalg.vector_norm(final, dim=1)
    initial_norm = torch.linalg.vector_norm(initial, dim=1)
    _require(bool(torch.all(final_norm > 0).item())
             and bool(torch.all(initial_norm > 0).item()),
             "C4-A0 zero-norm semantic prototype")
    _require(torch.unique(final, dim=0).shape[0] == K,
             "C4-A0 duplicated/collapsed semantic prototype")
    similarity = (
        (final @ initial.transpose(0, 1))
        / (final_norm[:, None] * initial_norm[None, :])
    ).detach()
    _require(bool(torch.isfinite(similarity).all().item()),
             "C4-A0 nonfinite semantic cosine")
    _require(torch.equal(torch.argmax(similarity, dim=1), torch.arange(K, device=final.device)),
             "C4-A0 semantic swap")
    return similarity


def validate_pre_gt_array_whitelist(arrays):
    keys = tuple(arrays.keys())
    _require(set(keys) == set(PRE_GT_ARRAY_WHITELIST),
             "C4-A0 pre-GT artifact whitelist mismatch")
    lowered = tuple(name.lower() for name in keys)
    _require(not any("gt" in name or "label_array" in name for name in lowered),
             "C4-A0 GT field crossed pre-GT boundary")
    return True
