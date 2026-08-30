"""Tests for E4-A1 memory-specific Information Utility identification."""

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.e4_semantic_memory_bank import (
    evaluate_e4_a0_memory_feasibility as e4a0,
)
from experiments.e4_semantic_memory_bank import (
    evaluate_e4_a1_memory_specific_utility as evaluator,
)
from experiments.e4_semantic_memory_bank.e4a1_memory_specific_utility import (
    E4A1_ARMS,
    NORMAL_E4A1_ARMS,
    attach_oracle_writer_weights,
    build_memory_information_utility,
    build_memory_specific_semantic_usefulness,
    build_normal_e4a1_writer_weights,
    build_sample_anchors,
)
from experiments.e4_semantic_memory_bank.semantic_memory_bank import (
    build_class_memory,
    build_shared_query,
    validate_sparse_labels,
)


FROZEN_LABELED_IDS = np.array(
    [67, 82, 90, 111, 200, 365, 440, 513, 536, 983, 1027, 1250, 1316, 1385],
    dtype=np.int64,
)
FROZEN_LABELED_TARGETS = np.array(
    [4, 6, 2, 3, 0, 4, 1, 5, 6, 3, 0, 2, 5, 1],
    dtype=np.int64,
)


def _inputs():
    generator = torch.Generator().manual_seed(812)
    h_sem = F.normalize(
        torch.rand(1400, 6, 7, generator=generator),
        dim=-1,
    )
    R_full = (
        0.1
        + 0.8
        * torch.rand(
            1400, 6, generator=generator, dtype=torch.float64
        )
    )
    oracle = torch.zeros(14, 6, dtype=torch.float64)
    oracle[:, :3] = 1.0
    return (
        h_sem,
        FROZEN_LABELED_IDS.copy(),
        FROZEN_LABELED_TARGETS.copy(),
        R_full,
        oracle,
    )


@pytest.fixture(scope="module")
def fixed_outputs():
    return evaluator.build_fixed_e4a1_outputs(*_inputs())


def _new_sources():
    root = Path(__file__).resolve().parents[1]
    paths = (
        root
        / "experiments/e4_semantic_memory_bank/e4a1_memory_specific_utility.py",
        root
        / (
            "experiments/e4_semantic_memory_bank/"
            "evaluate_e4_a1_memory_specific_utility.py"
        ),
    )
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gate_results(
    all_pair,
    S_pair,
    shuffled_S_pair,
    U_pair,
    shuffled_U_pair,
    oracle_pair=(0.99, 0.99),
):
    pairs = {
        "ALL_MEMORY": all_pair,
        "R_MEMORY": all_pair,
        "S_MEMORY": S_pair,
        "SHUFFLED_S_MEMORY": shuffled_S_pair,
        "U_RS_MEMORY": U_pair,
        "SHUFFLED_U_RS_MEMORY": shuffled_U_pair,
        "ORACLE_CLEAN_MEMORY": oracle_pair,
    }
    return {
        arm: {
            "prototype_assignment_ACC": float(pair[0]),
            "true_class_margin_mean": float(pair[1]),
        }
        for arm, pair in pairs.items()
    }


def _frozen_replay_fixed():
    root = evaluator.DEFAULT_E4A0_OUTPUT_DIR
    fixed = {}
    with np.load(
        root / "memory_prototypes.npz", allow_pickle=False
    ) as prototypes, np.load(
        root / "predictions.npz", allow_pickle=False
    ) as predictions:
        for arm in evaluator.E4A0_REPLAY_ARMS:
            fixed[arm] = {
                "prototypes": torch.from_numpy(
                    np.array(prototypes[arm], copy=True, order="C")
                ),
                "predictions": torch.from_numpy(
                    np.array(
                        predictions[arm],
                        dtype=np.int64,
                        copy=True,
                        order="C",
                    )
                ),
            }
    return fixed


def test_exact_frozen_fourteen_ids_and_targets():
    labeled_ids, labels_labeled, audit = e4a0.load_sparse_label_protocol()

    assert np.array_equal(labeled_ids, FROZEN_LABELED_IDS)
    assert np.array_equal(labels_labeled, FROZEN_LABELED_TARGETS)
    assert audit["labeled_count"] == 14


def test_exactly_two_sparse_labels_per_class():
    targets, counts = validate_sparse_labels(FROZEN_LABELED_TARGETS)

    assert targets.shape == (14,)
    assert counts.tolist() == [2] * 7


def test_representation_and_labeled_tensor_shapes(fixed_outputs):
    assert fixed_outputs["h_sem"].shape == (1400, 6, 7)
    assert fixed_outputs["h_labeled"].shape == (14, 6, 7)
    assert not fixed_outputs["h_sem"].requires_grad
    assert not fixed_outputs["h_labeled"].requires_grad


def test_sample_anchor_shape_formula_and_inputs(fixed_outputs):
    h_labeled = fixed_outputs["h_labeled"]
    sample_anchor = build_sample_anchors(h_labeled)
    expected = F.normalize(h_labeled.mean(dim=1), dim=-1)

    assert sample_anchor.shape == (14, 7)
    assert torch.equal(sample_anchor, expected)
    assert tuple(inspect.signature(build_sample_anchors).parameters) == (
        "h_labeled",
    )


def test_positive_reference_excludes_writer_and_has_one_peer(fixed_outputs):
    components = fixed_outputs["semantic_components"]
    audit = components["audit"]
    peer_ids = components["positive_peer_indices"]
    targets = fixed_outputs["labels_labeled"]

    assert audit["positive_reference_self_exclusion_pass"]
    assert audit["exactly_one_positive_peer_per_writer_pass"]
    assert audit["positive_peer_count"] == [1] * 14
    for writer_id, peer_id in enumerate(peer_ids.tolist()):
        assert peer_id != writer_id
        assert targets[peer_id] == targets[writer_id]
        assert torch.equal(
            components["positive_reference"][writer_id],
            components["sample_anchor"][peer_id],
        )


def test_negatives_exclude_true_class_and_number_exactly_twelve(
    fixed_outputs,
):
    components = fixed_outputs["semantic_components"]
    negative_mask = components["negative_mask"]
    targets = fixed_outputs["labels_labeled"]

    assert negative_mask.shape == (14, 14)
    assert components["audit"]["negative_true_class_exclusion_pass"]
    assert components["audit"]["negative_count_exactly_12_pass"]
    assert components["audit"]["negative_count_per_writer"] == [12] * 14
    for writer_id in range(14):
        assert torch.equal(
            negative_mask[writer_id],
            targets != targets[writer_id],
        )


def test_positive_and_hard_negative_similarity_shapes(fixed_outputs):
    components = fixed_outputs["semantic_components"]

    assert components["positive_similarity"].shape == (14, 6)
    assert components["hard_negative_similarity"].shape == (14, 6)
    assert torch.isfinite(components["positive_similarity"]).all()
    assert torch.isfinite(components["hard_negative_similarity"]).all()


def test_semantic_margin_shape_finite_and_theoretical_range(fixed_outputs):
    margin = fixed_outputs["semantic_components"]["semantic_margin"]

    assert margin.shape == (14, 6)
    assert torch.isfinite(margin).all()
    assert float(margin.min()) >= -2.0
    assert float(margin.max()) <= 2.0
    assert fixed_outputs["semantic_components"]["audit"][
        "semantic_margin_range_pass"
    ]


def test_S_mem_shape_range_and_stop_gradient(fixed_outputs):
    S_mem = fixed_outputs["S_mem"]

    assert S_mem.shape == (14, 6)
    assert torch.isfinite(S_mem).all()
    assert float(S_mem.min()) >= 0.0
    assert float(S_mem.max()) <= 1.0
    assert not S_mem.requires_grad and S_mem.grad_fn is None


def test_S_mem_is_exact_preregistered_mapping(fixed_outputs):
    components = fixed_outputs["semantic_components"]
    expected = (components["semantic_margin"] + 2.0) / 4.0

    assert torch.equal(components["S_mem"], expected)


def test_U_mem_is_exact_R_times_S_without_coefficient(fixed_outputs):
    expected = fixed_outputs["R_labeled"] * fixed_outputs["S_mem"].to(
        dtype=fixed_outputs["R_labeled"].dtype
    )

    assert fixed_outputs["U_mem"].shape == (14, 6)
    assert torch.equal(fixed_outputs["U_mem"], expected)
    assert tuple(
        inspect.signature(build_memory_information_utility).parameters
    ) == ("R_labeled", "S_mem")


def test_S_builder_has_no_R_or_full_GT_argument():
    parameters = tuple(
        inspect.signature(
            build_memory_specific_semantic_usefulness
        ).parameters
    )

    assert parameters == (
        "h_labeled",
        "labels_labeled",
        "numerical_tolerance",
    )
    assert all("GT" not in name for name in parameters)
    assert all(not name.startswith("R_") for name in parameters)


def test_S_matched_shuffle_preserves_class_view_sums_and_changes_cells(
    fixed_outputs,
):
    original = fixed_outputs["writer_weights"]["S_MEMORY"]
    shuffled = fixed_outputs["writer_weights"]["SHUFFLED_S_MEMORY"]
    audit = fixed_outputs["shuffle_audit"]["S_matched_shuffle"]
    targets = fixed_outputs["labels_labeled"]

    assert audit["same_shape_pass"]
    assert audit["same_global_multiset_pass"]
    assert audit["class_view_weight_sum_exact_pass"]
    assert audit["class_total_denominator_exact_pass"]
    assert audit["changed_cell_count"] > 0
    assert not torch.equal(original, shuffled)
    for class_id in range(7):
        rows = targets == class_id
        assert torch.equal(
            original[rows].sum(dim=0), shuffled[rows].sum(dim=0)
        )


def test_U_matched_shuffle_preserves_class_view_sums_and_changes_cells(
    fixed_outputs,
):
    original = fixed_outputs["writer_weights"]["U_RS_MEMORY"]
    shuffled = fixed_outputs["writer_weights"]["SHUFFLED_U_RS_MEMORY"]
    audit = fixed_outputs["shuffle_audit"]["U_matched_shuffle"]
    targets = fixed_outputs["labels_labeled"]

    assert audit["same_shape_pass"]
    assert audit["same_global_multiset_pass"]
    assert audit["class_view_weight_sum_exact_pass"]
    assert audit["class_total_denominator_exact_pass"]
    assert audit["changed_cell_count"] > 0
    assert not torch.equal(original, shuffled)
    for class_id in range(7):
        rows = targets == class_id
        assert torch.equal(
            original[rows].sum(dim=0), shuffled[rows].sum(dim=0)
        )


def test_U_shuffle_is_direct_U_swap_not_R_times_shuffled_S(fixed_outputs):
    R_labeled = fixed_outputs["R_labeled"]
    S_shuffle = fixed_outputs["writer_weights"]["SHUFFLED_S_MEMORY"].to(
        dtype=R_labeled.dtype
    )
    U_shuffle = fixed_outputs["writer_weights"][
        "SHUFFLED_U_RS_MEMORY"
    ]

    assert fixed_outputs["shuffle_audit"][
        "U_shuffle_built_from_R_times_shuffled_S"
    ] is False
    assert not torch.equal(U_shuffle, R_labeled * S_shuffle)


def test_shuffles_are_deterministic_and_do_not_use_global_RNG():
    h_sem, ids, targets, R_full, oracle = _inputs()
    torch.manual_seed(142)
    state_before = torch.random.get_rng_state().clone()

    first = evaluator.build_fixed_e4a1_outputs(
        h_sem, ids, targets, R_full, oracle
    )
    second = evaluator.build_fixed_e4a1_outputs(
        h_sem, ids, targets, R_full, oracle
    )

    assert torch.equal(torch.random.get_rng_state(), state_before)
    for arm in ("SHUFFLED_S_MEMORY", "SHUFFLED_U_RS_MEMORY"):
        assert torch.equal(
            first["writer_weights"][arm],
            second["writer_weights"][arm],
        )
        assert torch.equal(
            first[arm]["predictions"], second[arm]["predictions"]
        )


def test_every_arm_uses_all_labels_without_hard_selection(fixed_outputs):
    targets = fixed_outputs["labels_labeled"]

    assert tuple(fixed_outputs["writer_weights"]) == E4A1_ARMS
    for arm in E4A1_ARMS:
        weights = fixed_outputs["writer_weights"][arm]
        assert weights.shape == (14, 6)
        for class_id in range(7):
            assert float(weights[targets == class_id].sum()) > 0.0
        assert fixed_outputs[arm]["prototypes"].shape == (7, 7)
        assert torch.isfinite(fixed_outputs[arm]["prototypes"]).all()
        assert torch.allclose(
            torch.linalg.vector_norm(
                fixed_outputs[arm]["prototypes"], dim=1
            ),
            torch.ones(7),
        )
        assert not fixed_outputs[arm]["prototypes"].requires_grad


def test_A0_class_memory_helper_is_reused_exactly(fixed_outputs):
    arm = "U_RS_MEMORY"
    expected = build_class_memory(
        fixed_outputs["h_labeled"],
        fixed_outputs["labels_labeled"],
        fixed_outputs["writer_weights"][arm],
    )

    assert torch.equal(fixed_outputs[arm]["prototypes"], expected)


def test_query_is_identical_across_all_seven_arms(fixed_outputs):
    expected = build_shared_query(fixed_outputs["h_sem"])
    query_pointers = {
        fixed_outputs[arm]["h_query"].data_ptr() for arm in E4A1_ARMS
    }

    assert fixed_outputs["h_query"].shape == (1400, 7)
    assert torch.equal(fixed_outputs["h_query"], expected)
    assert query_pointers == {fixed_outputs["h_query"].data_ptr()}
    for arm in E4A1_ARMS:
        assert torch.equal(
            fixed_outputs[arm]["h_query"], fixed_outputs["h_query"]
        )
        assert fixed_outputs[arm]["scores"].shape == (1400, 7)


def test_query_has_no_R_S_or_U_input():
    parameters = tuple(inspect.signature(build_shared_query).parameters)

    assert parameters == ("h_sem",)


def test_alignment_remains_exactly_q_times_M_transpose():
    generator = torch.Generator().manual_seed(51)
    q_local = torch.rand(4, 6, 7, generator=generator)
    matrices = torch.zeros(6, 7, 7)
    for view_id in range(6):
        permutation = torch.roll(torch.arange(7), shifts=view_id + 1)
        matrices[view_id, torch.arange(7), permutation] = 1.0
    expected = torch.stack(
        [
            q_local[:, view_id, :] @ matrices[view_id].T
            for view_id in range(6)
        ],
        dim=1,
    )

    q_aligned, h_sem, audit = e4a0.align_q_to_global(
        q_local, matrices
    )

    assert torch.equal(q_aligned, expected)
    assert torch.allclose(h_sem, F.normalize(expected, dim=-1))
    assert audit["alignment_direction"] == "q_local @ M^T"


def test_no_raw_z_cross_view_average_or_new_alignment():
    source = _new_sources()

    assert "raw_z.mean" not in source
    assert "align_q_to_global" not in source
    assert "load_e1_lwc_semantic_representation" in source


def test_no_disallowed_training_or_selection_constructs():
    source = _new_sources()
    forbidden = (
        "torch.optim",
        ".backward(",
        ".train(",
        "soft" + "max(",
        "top_" + "k",
        "top-" + "k",
        "pseudo" + "-label",
        "temperature" + "_sweep",
    )
    for token in forbidden:
        assert token not in source


def test_prediction_seal_precedes_full_GT_load(tmp_path, fixed_outputs):
    run_source = inspect.getsource(evaluator.run_evaluation)

    assert run_source.index("save_prediction_seal") < run_source.index(
        "load_full_ground_truth"
    )
    seal = evaluator.save_prediction_seal(fixed_outputs, tmp_path)
    assert (tmp_path / "predictions.npz").is_file()
    assert (tmp_path / "prediction_seal.json").is_file()
    assert seal["prediction_saved_before_full_GT"]
    assert seal["prediction_reloaded_before_full_GT"]
    assert seal["prediction_hashed_before_full_GT"]
    assert not seal["full_unlabeled_GT_loaded_before_prediction"]


def test_unlabeled_GT_cannot_enter_S_U_or_memory_builder():
    S_parameters = tuple(
        inspect.signature(
            build_memory_specific_semantic_usefulness
        ).parameters
    )
    U_parameters = tuple(
        inspect.signature(build_memory_information_utility).parameters
    )
    fixed_parameters = tuple(
        inspect.signature(
            evaluator.build_fixed_e4a1_outputs
        ).parameters
    )

    assert "full_GT" not in S_parameters
    assert "full_GT" not in U_parameters
    assert "full_GT" not in fixed_parameters
    assert "unlabeled_ids" not in fixed_parameters


def test_oracle_is_isolated_from_every_normal_arm():
    h_sem, ids, targets, R_full, first_oracle = _inputs()
    second_oracle = torch.zeros_like(first_oracle)
    second_oracle[:, 3:] = 1.0

    first = evaluator.build_fixed_e4a1_outputs(
        h_sem, ids, targets, R_full, first_oracle
    )
    second = evaluator.build_fixed_e4a1_outputs(
        h_sem, ids, targets, R_full, second_oracle
    )

    for arm in NORMAL_E4A1_ARMS:
        assert torch.equal(
            first["writer_weights"][arm],
            second["writer_weights"][arm],
        )
        assert torch.equal(
            first[arm]["prototypes"], second[arm]["prototypes"]
        )
        assert torch.equal(
            first[arm]["predictions"], second[arm]["predictions"]
        )
    assert not torch.equal(
        first["writer_weights"]["ORACLE_CLEAN_MEMORY"],
        second["writer_weights"]["ORACLE_CLEAN_MEMORY"],
    )


def test_normal_writer_builder_has_no_oracle_argument():
    parameters = tuple(
        inspect.signature(
            build_normal_e4a1_writer_weights
        ).parameters
    )
    attach_parameters = tuple(
        inspect.signature(attach_oracle_writer_weights).parameters
    )

    assert parameters == (
        "R_labeled",
        "S_mem",
        "U_mem",
        "labels_labeled",
    )
    assert attach_parameters == (
        "normal_weights",
        "oracle_clean_weights",
    )


def test_E4A0_ALL_prediction_and_prototype_exact_replay():
    replay = evaluator.verify_e4a0_exact_replay(
        _frozen_replay_fixed()
    )

    record = replay["arms"]["ALL_MEMORY"]
    assert record["prediction_exact_replay_pass"]
    assert record["prototype_exact_replay_pass"]
    assert replay["ALL_prediction_exact_replay_pass"]


def test_E4A0_R_prediction_and_prototype_exact_replay():
    replay = evaluator.verify_e4a0_exact_replay(
        _frozen_replay_fixed()
    )

    record = replay["arms"]["R_MEMORY"]
    assert record["prediction_exact_replay_pass"]
    assert record["prototype_exact_replay_pass"]
    assert replay["R_prediction_exact_replay_pass"]


def test_E4A0_Oracle_prediction_and_prototype_exact_replay():
    replay = evaluator.verify_e4a0_exact_replay(
        _frozen_replay_fixed()
    )

    record = replay["arms"]["ORACLE_CLEAN_MEMORY"]
    assert record["prediction_exact_replay_pass"]
    assert record["prototype_exact_replay_pass"]
    assert replay["Oracle_prediction_exact_replay_pass"]


@pytest.mark.parametrize("arm", evaluator.E4A0_REPLAY_ARMS)
def test_E4A0_replay_mismatch_hard_fails_for_every_required_arm(arm):
    fixed = _frozen_replay_fixed()
    fixed[arm]["prototypes"][0, 0] += 1.0

    with pytest.raises(
        RuntimeError, match=evaluator.E4A0_REPLAY_MISMATCH
    ):
        evaluator.verify_e4a0_exact_replay(fixed)


def test_E4A0_replay_lock_runs_before_full_GT():
    source = inspect.getsource(evaluator.run_evaluation)

    assert source.index("verify_e4a0_exact_replay") < source.index(
        "load_full_ground_truth"
    )


def test_gate_decision_A_is_exact():
    results = _gate_results(
        all_pair=(0.50, 0.50),
        S_pair=(0.70, 0.70),
        shuffled_S_pair=(0.60, 0.60),
        U_pair=(0.80, 0.80),
        shuffled_U_pair=(0.70, 0.70),
    )

    gate = evaluator.build_e4a1_gate_decision(results)

    assert gate["S_specificity_pass"]
    assert gate["U_specificity_pass"]
    assert gate["U_net_gain_pass"]
    assert gate["R_contribution_pass"]
    assert gate["final_decision"] == "E4A1_RS_INFORMATION_UTILITY_PASS"


def test_gate_decision_B_is_exact():
    results = _gate_results(
        all_pair=(0.50, 0.50),
        S_pair=(0.80, 0.80),
        shuffled_S_pair=(0.60, 0.60),
        U_pair=(0.70, 0.70),
        shuffled_U_pair=(0.65, 0.65),
    )

    gate = evaluator.build_e4a1_gate_decision(results)

    assert gate["S_specificity_pass"]
    assert gate["S_net_gain_pass"]
    assert not gate["R_contribution_pass"]
    assert gate["final_decision"] == "E4A1_S_ONLY_MEMORY_UTILITY_PASS"


def test_gate_decision_C_is_exact():
    results = _gate_results(
        all_pair=(0.80, 0.80),
        S_pair=(0.70, 0.70),
        shuffled_S_pair=(0.60, 0.60),
        U_pair=(0.75, 0.75),
        shuffled_U_pair=(0.70, 0.70),
    )

    gate = evaluator.build_e4a1_gate_decision(results)

    assert gate["S_specificity_pass"]
    assert gate["U_specificity_pass"]
    assert not gate["S_net_gain_pass"]
    assert not gate["U_net_gain_pass"]
    assert gate["final_decision"] == (
        "E4A1_SPECIFIC_BUT_NO_NET_MEMORY_GAIN"
    )


def test_gate_decision_D_is_exact():
    results = _gate_results(
        all_pair=(0.80, 0.80),
        S_pair=(0.60, 0.60),
        shuffled_S_pair=(0.60, 0.60),
        U_pair=(0.60, 0.60),
        shuffled_U_pair=(0.70, 0.70),
    )

    gate = evaluator.build_e4a1_gate_decision(results)

    assert not gate["S_specificity_pass"]
    assert not gate["U_specificity_pass"]
    assert gate["final_decision"] == (
        "E4A1_MEMORY_UTILITY_IDENTIFICATION_FAIL"
    )


def test_R_contribution_requires_no_decrease_and_one_strict_gain():
    equal = _gate_results(
        (0.4, 0.4), (0.7, 0.7), (0.5, 0.5), (0.7, 0.7), (0.5, 0.5)
    )
    margin_gain = _gate_results(
        (0.4, 0.4), (0.7, 0.7), (0.5, 0.5), (0.7, 0.8), (0.5, 0.5)
    )
    ACC_drop = _gate_results(
        (0.4, 0.4), (0.7, 0.7), (0.5, 0.5), (0.6, 0.8), (0.5, 0.5)
    )

    assert not evaluator.build_e4a1_gate_decision(equal)[
        "R_contribution_pass"
    ]
    assert evaluator.build_e4a1_gate_decision(margin_gain)[
        "R_contribution_pass"
    ]
    assert not evaluator.build_e4a1_gate_decision(ACC_drop)[
        "R_contribution_pass"
    ]


def test_oracle_metrics_never_enter_gate():
    first = _gate_results(
        (0.5, 0.5), (0.7, 0.7), (0.6, 0.6), (0.8, 0.8), (0.7, 0.7),
        oracle_pair=(0.0, 0.0),
    )
    second = _gate_results(
        (0.5, 0.5), (0.7, 0.7), (0.6, 0.6), (0.8, 0.8), (0.7, 0.7),
        oracle_pair=(1.0, 1.0),
    )

    assert evaluator.build_e4a1_gate_decision(first) == (
        evaluator.build_e4a1_gate_decision(second)
    )


def test_unregistered_gate_combination_hard_fails_instead_of_inventing_rule():
    results = _gate_results(
        all_pair=(0.5, 0.5),
        S_pair=(0.7, 0.7),
        shuffled_S_pair=(0.6, 0.6),
        U_pair=(0.8, 0.8),
        shuffled_U_pair=(0.9, 0.9),
    )

    with pytest.raises(
        RuntimeError, match=evaluator.E4A1_DECISION_TREE_GAP
    ):
        evaluator.build_e4a1_gate_decision(results)


def test_primary_metrics_cover_1386_unlabeled_only(fixed_outputs):
    full_GT = np.arange(1400, dtype=np.int64) % 7
    unlabeled_ids = np.setdiff1d(
        np.arange(1400, dtype=np.int64),
        FROZEN_LABELED_IDS,
    )

    results = evaluator.evaluate_unlabeled_fixed_arms(
        fixed_outputs, full_GT, unlabeled_ids
    )

    required = {
        "prototype_assignment_ACC",
        "prototype_assignment_NMI",
        "prototype_assignment_ARI",
        "true_class_margin_mean",
        "prototype_pairwise_cosine_mean",
        "prototype_pairwise_cosine_max",
        "prototype_separation",
    }
    for arm in E4A1_ARMS:
        assert results[arm]["evaluation_sample_count"] == 1386
        assert required.issubset(results[arm])


def test_future_output_contract_and_formal_path_are_frozen():
    assert evaluator.DEFAULT_OUTPUT_DIR == (
        evaluator.REPOSITORY_ROOT
        / "outputs/e4_semantic_memory_bank"
        / "e4a1_memory_specific_utility_seed20"
    )
    run_source = inspect.getsource(evaluator.run_evaluation)
    for artifact in (
        "utility_components.npz",
        "writer_weights.npz",
        "memory_prototypes.npz",
        "e4a1_audit.json",
        "diagnostic_results.json",
    ):
        assert artifact in run_source


def test_E4A0_files_remain_bitwise_frozen():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "experiments/e4_semantic_memory_bank/semantic_memory_bank.py": (
            "8c4f99e1cf31ec03eb69183445076177362e6b4adc6e6dc56350c76f41a0afe6"
        ),
        (
            "experiments/e4_semantic_memory_bank/"
            "evaluate_e4_a0_memory_feasibility.py"
        ): "5f9e26d4ef3d446f22743dc4f7900369e9a0d82bde6f13a170eaa89278e99e4c",
        "tests/test_e4_a0_semantic_memory_bank.py": (
            "4ed15f1bb3f20e4788a07f820eb685bc75de1da5239161ed593db96d54a2e28c"
        ),
    }

    for relative_path, expected_sha in expected.items():
        assert _file_sha256(root / relative_path) == expected_sha
