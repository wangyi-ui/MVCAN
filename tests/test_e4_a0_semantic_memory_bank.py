"""Targeted tests for the corrected E4-A0 semantic-memory protocol."""

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.e4_semantic_memory_bank import (
    evaluate_e4_a0_memory_feasibility as evaluator,
)
from experiments.e4_semantic_memory_bank.semantic_memory_bank import (
    E4_ARMS,
    build_class_memory,
    build_classwise_matched_shuffle,
    build_normal_writer_weights,
    build_shared_query,
    predict_from_memory,
    validate_sparse_labels,
)


def _labels():
    return torch.arange(7, dtype=torch.long).repeat_interleave(2)


def _inputs(sample_num=28):
    generator = torch.Generator().manual_seed(77)
    h_sem = torch.rand(
        sample_num,
        6,
        7,
        generator=generator,
        requires_grad=True,
    )
    h_sem = F.normalize(h_sem, dim=-1)
    R_full = torch.rand(sample_num, 6, generator=generator)
    R_full = 0.1 + 0.8 * R_full
    labeled_ids = np.arange(14, dtype=np.int64)
    oracle = np.ones((14, 6), dtype=np.float32)
    oracle[:, 3:] = 0.0
    return h_sem, labeled_ids, _labels(), R_full, oracle


def _fixed(sample_num=28, oracle=None):
    h_sem, labeled_ids, labels_labeled, R_full, default_oracle = _inputs(
        sample_num
    )
    return evaluator.build_fixed_arm_outputs(
        h_sem=h_sem,
        labeled_ids=labeled_ids,
        labels_labeled=labels_labeled,
        R_full=R_full,
        oracle_clean_weights=default_oracle if oracle is None else oracle,
    )


def _module_sources():
    root = Path(__file__).resolve().parents[1]
    paths = (
        root
        / "experiments/e4_semantic_memory_bank/semantic_memory_bank.py",
        root
        / (
            "experiments/e4_semantic_memory_bank/"
            "evaluate_e4_a0_memory_feasibility.py"
        ),
    )
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_exact_fourteen_labels_and_two_per_class():
    targets, counts = validate_sparse_labels(_labels())

    assert targets.shape == (14,)
    assert counts.tolist() == [2] * 7
    with pytest.raises(ValueError, match="shape"):
        validate_sparse_labels(_labels()[:-1])
    invalid = _labels().clone()
    invalid[-1] = 0
    with pytest.raises(ValueError):
        validate_sparse_labels(invalid)


def test_frozen_sparse_label_artifact_is_exact_protocol():
    labeled_ids, labels_labeled, audit = evaluator.load_sparse_label_protocol()

    assert labeled_ids.shape == labels_labeled.shape == (14,)
    assert np.bincount(labels_labeled, minlength=7).tolist() == [2] * 7
    assert audit["all_14_labels_used_pass"]
    assert audit["exactly_two_labels_per_class_pass"]
    assert not audit["full_GT_loaded"]


def test_obsolete_labeled_admission_design_is_absent():
    source = _module_sources()
    forbidden = (
        "DEFAULT_" + "TOP_K",
        "TOP_" + "K = 5",
        "Random" + "MemoryBank",
        "rand" + "perm(",
        "_select_" + "admission",
        "top-" + "k labeled",
    )
    for token in forbidden:
        assert token not in source


def test_class_memory_is_7_by_7_unit_norm_and_uses_weighted_formula():
    h_sem, _, labels_labeled, R_full, _ = _inputs()
    h_labeled = h_sem[:14]
    weights = R_full[:14]

    prototypes = build_class_memory(
        h_labeled, labels_labeled, weights
    )

    assert prototypes.shape == (7, 7)
    assert torch.isfinite(prototypes).all()
    assert torch.allclose(torch.linalg.vector_norm(prototypes, dim=1), torch.ones(7))
    class_rows = labels_labeled == 0
    weighted = (
        weights[class_rows].unsqueeze(-1) * h_labeled.detach()[class_rows]
    ).sum(dim=(0, 1)) / weights[class_rows].sum()
    assert torch.allclose(prototypes[0], F.normalize(weighted, dim=0))


def test_class_memory_stops_all_gradients_and_rejects_empty_denominator():
    h_sem, _, labels_labeled, R_full, _ = _inputs()
    weights = R_full[:14].clone().requires_grad_(True)

    prototypes = build_class_memory(h_sem[:14], labels_labeled, weights)

    assert not prototypes.requires_grad and prototypes.grad_fn is None
    assert weights.grad is None
    empty_class = weights.detach().clone()
    empty_class[labels_labeled == 3] = 0.0
    with pytest.raises(ValueError, match="denominator"):
        build_class_memory(h_sem[:14], labels_labeled, empty_class)


def test_frozen_D2_R_has_required_key_shape_range_and_provenance():
    R_full, frozen_reliability = evaluator.load_frozen_reliability()

    assert R_full.shape == (1400, 6)
    assert np.isfinite(R_full).all()
    assert float(R_full.min()) >= 0.0
    assert float(R_full.max()) <= 1.0
    assert frozen_reliability["loaded_npz_key"] == "U"
    assert frozen_reliability["loaded_fields"] == ["U"]
    assert frozen_reliability["R_full"]["logical_sha256"] == (
        evaluator.EXPECTED_R_LOGICAL_SHA256
    )
    assert frozen_reliability["raw_npz_sha256"] == (
        evaluator.EXPECTED_R_RAW_NPZ_SHA256
    )


def test_matched_shuffle_preserves_every_class_view_sum_and_denominator():
    _, _, labels_labeled, R_full, _ = _inputs()
    R_labeled = R_full[:14]

    shuffled, audit = build_classwise_matched_shuffle(
        R_labeled, labels_labeled
    )

    assert shuffled.shape == R_labeled.shape
    assert audit["same_global_multiset_pass"]
    assert audit["class_view_weight_sum_exact_pass"]
    assert audit["class_total_denominator_exact_pass"]
    for class_id in range(7):
        rows = labels_labeled == class_id
        assert torch.equal(
            R_labeled[rows].sum(dim=0), shuffled[rows].sum(dim=0)
        )
        assert torch.equal(
            R_labeled[rows].sum(dim=0).sum(),
            shuffled[rows].sum(dim=0).sum(),
        )


def test_matched_shuffle_changes_assignments_and_is_deterministic():
    _, _, labels_labeled, R_full, _ = _inputs()
    R_labeled = R_full[:14]

    first, first_audit = build_classwise_matched_shuffle(
        R_labeled, labels_labeled
    )
    second, second_audit = build_classwise_matched_shuffle(
        R_labeled, labels_labeled
    )

    assert first_audit["changed_cell_count"] > 0
    assert not torch.equal(first, R_labeled)
    assert torch.equal(first, second)
    assert first_audit == second_audit
    for class_id, (first_row, second_row) in enumerate(
        first_audit["class_row_pairs"]
    ):
        assert labels_labeled[first_row] == labels_labeled[second_row] == class_id
        assert torch.equal(first[first_row], R_labeled[second_row])
        assert torch.equal(first[second_row], R_labeled[first_row])


def test_all_arms_share_byte_identical_unweighted_query():
    h_sem, _, _, _, _ = _inputs()
    fixed = _fixed()
    expected = F.normalize(h_sem.detach().mean(dim=1), dim=-1)

    assert torch.equal(fixed["h_query"], expected)
    pointers = {
        fixed[arm]["h_query"].data_ptr() for arm in E4_ARMS
    }
    assert pointers == {fixed["h_query"].data_ptr()}
    for arm in E4_ARMS:
        assert torch.equal(fixed[arm]["h_query"], fixed["h_query"])
        assert fixed[arm]["scores"].shape == (28, 7)


def test_no_old_external_stage_dependency_or_old_cli():
    source = _module_sources().lower()
    args = vars(evaluator.parse_args([]))

    assert ("e3" + "b") not in source
    assert "utility_path" not in args
    assert "encoder_dir" not in args
    assert args["e1_lwc_model_dir"].endswith(
        "outputs/e1_pairwise_utility/lwc_100ep_seed20/models"
    )


def test_prediction_is_saved_and_hashed_before_full_GT_loader(tmp_path):
    run_source = inspect.getsource(evaluator.run_evaluation)
    assert run_source.index("_save_predictions_before_full_GT") < (
        run_source.index("load_full_ground_truth")
    )

    fixed = _fixed()
    seal = evaluator._save_predictions_before_full_GT(fixed, tmp_path)

    assert (tmp_path / "predictions.npz").is_file()
    assert (tmp_path / "prediction_seal.json").is_file()
    assert seal["prediction_saved_before_full_GT"]
    assert seal["prediction_hashed_before_full_GT"]
    assert not seal["full_unlabeled_GT_loaded_before_prediction"]


def test_prediction_and_memory_apis_have_no_full_or_unlabeled_GT():
    prediction_parameters = tuple(
        inspect.signature(predict_from_memory).parameters
    )
    fixed_parameters = tuple(
        inspect.signature(evaluator.build_fixed_arm_outputs).parameters
    )

    assert prediction_parameters == ("h_query", "prototypes")
    assert all("label" not in name for name in prediction_parameters)
    assert "full_GT" not in fixed_parameters
    assert "unlabeled" not in " ".join(fixed_parameters)


def test_evaluator_contains_no_fitting_optimizer_or_backward_path():
    source = _module_sources()
    for forbidden in (
        "torch.optim",
        ".backward(",
        ".train(",
        "fit_base_kmeans",
        "KMeans(",
    ):
        assert forbidden not in source


def test_fixed_output_is_deterministic_without_global_rng_use():
    h_sem, labeled_ids, labels_labeled, R_full, oracle = _inputs()
    torch.manual_seed(1234)
    state_before = torch.random.get_rng_state().clone()

    first = evaluator.build_fixed_arm_outputs(
        h_sem, labeled_ids, labels_labeled, R_full, oracle
    )
    second = evaluator.build_fixed_arm_outputs(
        h_sem, labeled_ids, labels_labeled, R_full, oracle
    )

    assert torch.equal(torch.random.get_rng_state(), state_before)
    for arm in E4_ARMS:
        assert torch.equal(
            first[arm]["prototypes"], second[arm]["prototypes"]
        )
        assert torch.equal(
            first[arm]["predictions"], second[arm]["predictions"]
        )


def test_oracle_never_changes_R_ALL_or_shuffle_outputs():
    first_oracle = np.ones((14, 6), dtype=np.float32)
    second_oracle = np.zeros((14, 6), dtype=np.float32)
    second_oracle[:, :2] = 1.0
    h_sem, labeled_ids, labels_labeled, R_full, _ = _inputs()

    first = evaluator.build_fixed_arm_outputs(
        h_sem, labeled_ids, labels_labeled, R_full, first_oracle
    )
    second = evaluator.build_fixed_arm_outputs(
        h_sem, labeled_ids, labels_labeled, R_full, second_oracle
    )

    for arm in E4_ARMS[:3]:
        assert torch.equal(
            first["writer_weights"][arm], second["writer_weights"][arm]
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
        inspect.signature(build_normal_writer_weights).parameters
    )
    assert parameters == ("R_labeled", "labels_labeled")


def test_alignment_direction_is_q_times_M_transpose():
    generator = torch.Generator().manual_seed(19)
    q_local = torch.rand(5, 6, 7, generator=generator)
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

    q_aligned, h_sem, audit = evaluator.align_q_to_global(
        q_local, matrices
    )

    assert torch.equal(q_aligned, expected)
    assert torch.allclose(h_sem, F.normalize(expected, dim=-1))
    assert audit["alignment_direction"] == "q_local @ M^T"
    assert audit["M_permutation_pass"]
    assert audit["q_alignment_mass_preserved_pass"]


def test_invalid_alignment_matrix_hard_fails():
    q_local = torch.rand(3, 6, 7)
    invalid = torch.eye(7).repeat(6, 1, 1)
    invalid[0, 0] = invalid[0, 1]

    with pytest.raises(ValueError, match="permutation|row/column"):
        evaluator.align_q_to_global(q_local, invalid)


def test_no_raw_unaligned_cross_view_z_averaging():
    representation_source = inspect.getsource(
        evaluator.load_e1_lwc_semantic_representation
    )
    query_source = inspect.getsource(build_shared_query)

    assert "raw_z.mean" not in representation_source
    assert "q_local" in representation_source
    assert "h_sem" in query_source
    assert "z" not in tuple(inspect.signature(build_shared_query).parameters)


def test_primary_metric_names_and_gate_are_frozen():
    h_sem, labeled_ids, labels_labeled, R_full, oracle = _inputs(1400)
    fixed = evaluator.build_fixed_arm_outputs(
        h_sem, labeled_ids, labels_labeled, R_full, oracle
    )
    full_GT = np.arange(1400, dtype=np.int64) % 7
    unlabeled_ids = np.arange(14, 1400, dtype=np.int64)

    results = evaluator.evaluate_unlabeled_fixed_arms(
        fixed, full_GT, unlabeled_ids
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
    for arm in E4_ARMS:
        assert required.issubset(results[arm])
        assert results[arm]["evaluation_sample_count"] == 1386
    gate = evaluator.build_gate_decision(results)
    assert set(
        (
            gate["R_specificity_pass"],
            gate["final_decision"],
        )
    )
    assert "R_vs_ALL_ACC_delta" in gate
    assert "R_vs_ALL_margin_delta" in gate


def test_E1_LWC_checkpoint_path_is_exact_and_complete():
    assert evaluator.DEFAULT_E1_LWC_MODEL_DIR == (
        evaluator.REPOSITORY_ROOT
        / "outputs/e1_pairwise_utility/lwc_100ep_seed20/models"
    )
    expected_paths = evaluator._expected_checkpoint_paths(
        evaluator.DEFAULT_E1_LWC_MODEL_DIR
    )
    assert len(expected_paths) == 6
    assert all(path.is_file() for path in expected_paths)


def test_missing_E1_LWC_checkpoint_has_required_hard_fail(tmp_path, capsys):
    with pytest.raises(RuntimeError, match=evaluator.CHECKPOINT_NOT_FOUND):
        evaluator._fail_if_lwc_checkpoint_missing(
            tmp_path / "missing_models",
            tmp_path / "missing_audit.json",
        )
    assert evaluator.CHECKPOINT_NOT_FOUND in capsys.readouterr().err
