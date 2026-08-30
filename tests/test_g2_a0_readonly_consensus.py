"""Targeted protocol tests for G2-A0 read-only semantic consensus."""

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from experiments.g2_utility_semantic_consensus import g2_consensus_protocol as protocol
from irv.b3_audit import hash_backbone
from irv.b4_information_utility import tensor_sha256
from model import MvCAN


class _RawTrackingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.arange(1, 11, dtype=torch.float32))
        self.received_by_clustering = None

    def encoder(self, features):
        return features * self.scale[None, :]

    def clustering(self, latent):
        self.received_by_clustering = latent.detach().clone()
        return torch.softmax(latent[:, :7], dim=1)


def _utility(sample_num=40):
    return np.random.RandomState(8).uniform(
        size=(sample_num, protocol.VIEW_NUM)
    )


def _oracle_corruption():
    corruption = np.zeros(
        (protocol.SAMPLE_NUM, protocol.VIEW_NUM), dtype=bool
    )
    for sample_id in range(protocol.SAMPLE_NUM):
        selected = (sample_id + np.arange(protocol.CORRUPTION_K)) % protocol.VIEW_NUM
        corruption[sample_id, selected] = True
    return corruption


def _synthetic_q_and_global(sample_num=14):
    global_ids = np.arange(sample_num, dtype=np.int64) % protocol.CLASS_NUM
    q = np.zeros(
        (sample_num, protocol.VIEW_NUM, protocol.CLASS_NUM), dtype=np.float32
    )
    for view_id in range(protocol.VIEW_NUM):
        local_ids = (global_ids + view_id) % protocol.CLASS_NUM
        q[np.arange(sample_num), view_id, local_ids] = 1.0
    return q, global_ids


def test_native_extraction_shapes_and_q_uses_raw_z():
    autoencoders = [_RawTrackingAutoencoder() for _ in range(protocol.VIEW_NUM)]
    views = [
        np.full((5, protocol.LATENT_DIM), view_id + 1.0, dtype=np.float32)
        for view_id in range(protocol.VIEW_NUM)
    ]

    extracted = protocol.extract_native_z_q(autoencoders, views)

    assert extracted["raw_z_stack"].shape == (5, 6, 10)
    assert extracted["q_stack"].shape == (5, 6, 7)
    for view_id, autoencoder in enumerate(autoencoders):
        raw_z = extracted["raw_z_stack"][:, view_id, :]
        normalized_z = extracted["z_stack"][:, view_id, :]
        assert torch.equal(autoencoder.received_by_clustering, raw_z)
        assert not torch.equal(autoencoder.received_by_clustering, normalized_z)


def test_alignment_matrices_are_complete_k_by_k_permutations():
    q, global_ids = _synthetic_q_and_global()

    aligned = protocol.build_alignment(
        q,
        global_ids,
        lambda local, global_: MvCAN.Match(None, local, global_),
    )

    matrices = aligned["alignment_matrix"]
    assert matrices.shape == (6, 7, 7)
    assert np.array_equal(matrices.sum(axis=1), np.ones((6, 7), dtype=np.int64))
    assert np.array_equal(matrices.sum(axis=2), np.ones((6, 7), dtype=np.int64))
    assert np.allclose(aligned["aligned_q"].sum(axis=2), 1.0)
    assert np.array_equal(
        np.argmax(aligned["aligned_q"], axis=2),
        np.repeat(global_ids[:, None], protocol.VIEW_NUM, axis=1),
    )


def test_synthetic_q_times_m_transpose_orientation_is_explicit():
    q = np.zeros((1, protocol.VIEW_NUM, protocol.CLASS_NUM), dtype=np.float32)
    q[:, :, 1] = 1.0
    matrix = np.eye(protocol.CLASS_NUM, dtype=np.int64)
    matrix[[0, 1]] = matrix[[1, 0]]
    matrices = np.repeat(matrix[None, :, :], protocol.VIEW_NUM, axis=0)

    aligned = protocol.align_q_to_global(q, matrices)

    assert np.all(np.argmax(aligned, axis=2) == 0)
    assert np.array_equal(aligned[:, 0, :], q[:, 0, :] @ matrix.T)


def test_alignment_hard_fails_when_q_argmax_misses_cluster():
    q, global_ids = _synthetic_q_and_global()
    q[:, 0, :] = 0.0
    q[:, 0, 0] = 1.0

    with pytest.raises(RuntimeError, match="missing a cluster"):
        protocol.build_alignment(
            q,
            global_ids,
            lambda local, global_: MvCAN.Match(None, local, global_),
        )


def test_alignment_preserves_arbitrary_soft_probability_mass():
    rng = np.random.RandomState(19)
    q = rng.uniform(size=(9, protocol.VIEW_NUM, protocol.CLASS_NUM))
    q = q / q.sum(axis=2, keepdims=True)
    matrices = []
    for view_id in range(protocol.VIEW_NUM):
        permutation = np.roll(np.arange(protocol.CLASS_NUM), view_id)
        matrix = np.zeros((protocol.CLASS_NUM, protocol.CLASS_NUM), dtype=np.int64)
        matrix[np.arange(protocol.CLASS_NUM), permutation] = 1
        matrices.append(matrix)

    aligned = protocol.align_q_to_global(q, np.stack(matrices))

    assert np.allclose(aligned.sum(axis=2), q.sum(axis=2), atol=1e-12)


def test_all_u_top3_and_shuffled_u_admission_contracts():
    result = protocol.build_normal_admissions(_utility())
    admissions = result["admissions"]
    original = admissions["U_TOP3"]
    shuffled = admissions["SHUFFLED_U"]
    permutation = result["shuffled_u_row_permutation"]

    assert admissions["ALL"].shape == (40, 6)
    assert admissions["ALL"].all()
    assert np.all(original.sum(axis=1) == 3)
    assert np.all(shuffled.sum(axis=1) == 3)
    assert np.array_equal(shuffled, original[permutation])
    assert np.array_equal(shuffled.sum(axis=0), original.sum(axis=0))
    assert result["changed_row_count"] > 0


def test_oracle_admits_exactly_the_three_clean_views():
    corruption = _oracle_corruption()

    oracle = protocol.build_oracle_admission(corruption)

    assert oracle.shape == (1400, 6)
    assert np.all(oracle.sum(axis=1) == 3)
    assert np.array_equal(oracle, ~corruption)


def test_normal_builder_api_cannot_receive_oracle_data():
    parameters = inspect.signature(protocol.build_normal_admissions).parameters

    assert set(parameters) == {"utility", "control_seed"}
    assert protocol.normal_api_oracle_isolation_pass()
    assert all(
        token not in name.lower()
        for name in parameters
        for token in ("oracle", "corrupt", "clean", "mask")
    )


def test_consensus_uses_fixed_native_weight_formula_and_shapes():
    q = np.zeros((4, protocol.VIEW_NUM, protocol.CLASS_NUM), dtype=np.float64)
    for view_id in range(protocol.VIEW_NUM):
        q[:, view_id, view_id % protocol.CLASS_NUM] = 1.0
    admission = np.zeros((4, protocol.VIEW_NUM), dtype=bool)
    admission[:, :3] = True
    weights = np.arange(1, protocol.VIEW_NUM + 1, dtype=np.float64)

    result = protocol.semantic_consensus(q, admission, weights)
    expected = np.zeros((4, protocol.CLASS_NUM), dtype=np.float64)
    expected[:, 0] = 1.0 / 6.0
    expected[:, 1] = 2.0 / 6.0
    expected[:, 2] = 3.0 / 6.0

    assert result["weighted_mask"].shape == (4, 6)
    assert result["denominator"].shape == (4,)
    assert result["consensus_score"].shape == (4, 7)
    assert result["prediction"].shape == (4,)
    assert np.allclose(result["consensus_score"], expected)
    assert np.all(result["prediction"] == 2)


def test_consensus_probability_diagnostics_are_label_free_and_finite():
    rng = np.random.RandomState(31)
    q = rng.uniform(size=(12, protocol.VIEW_NUM, protocol.CLASS_NUM))
    q = q / q.sum(axis=2, keepdims=True)
    admission = np.ones((12, protocol.VIEW_NUM), dtype=bool)

    result = protocol.semantic_consensus(q, admission, np.ones(6))

    assert set(result["diagnostics"]) == {
        "mean_entropy",
        "mean_top1_top2_margin",
        "mean_max_confidence",
    }
    assert all(np.isfinite(value) for value in result["diagnostics"].values())


class _FakeKMeans:
    def __init__(self):
        self.call_count = 0
        self.cluster_centers_ = None

    def fit_predict(self, fused):
        self.call_count += 1
        prediction = np.arange(fused.shape[0], dtype=np.int64) % protocol.CLASS_NUM
        self.cluster_centers_ = np.stack([
            fused[prediction == cluster_id].mean(axis=0)
            for cluster_id in range(protocol.CLASS_NUM)
        ])
        return prediction


def test_native_replay_is_full_n_t1_two_and_uses_checkpoint_q_assignments(monkeypatch):
    rng = np.random.RandomState(23)
    raw_z = rng.normal(
        size=(protocol.SAMPLE_NUM, protocol.VIEW_NUM, protocol.LATENT_DIM)
    ).astype(np.float32)
    q = np.zeros(
        (protocol.SAMPLE_NUM, protocol.VIEW_NUM, protocol.CLASS_NUM),
        dtype=np.float32,
    )
    base = np.arange(protocol.SAMPLE_NUM) % protocol.CLASS_NUM
    for view_id in range(protocol.VIEW_NUM):
        q[np.arange(protocol.SAMPLE_NUM), view_id, (base + view_id) % 7] = 1.0
    fake = _FakeKMeans()
    monkeypatch.setattr(protocol, "_make_global_kmeans", lambda: fake)

    reference = protocol.replay_native_global_reference(raw_z, q)

    assert fake.call_count == protocol.FUSION_UPDATES == 2
    assert reference["view_assignments"].shape == (1400, 6)
    assert np.array_equal(reference["view_assignments"], np.argmax(q, axis=2))
    assert reference["latent_fusion"].shape == (1400, 60)
    assert reference["global_centers"].shape == (7, 60)
    assert reference["native_nmi_weights"].shape == (6,)


def test_all_policies_share_one_reference_hash_record():
    q, global_ids = _synthetic_q_and_global()
    matrices = np.repeat(
        np.eye(protocol.CLASS_NUM, dtype=np.int64)[None, :, :],
        protocol.VIEW_NUM,
        axis=0,
    )
    reference = {
        "global_prediction": global_ids,
        "global_centers": np.zeros((7, 60), dtype=np.float64),
        "native_nmi_weights": np.ones(6, dtype=np.float64),
        "latent_fusion": np.zeros((14, 60), dtype=np.float64),
    }
    alignment = {
        "alignment_matrix": matrices,
        "aligned_q": q,
    }

    shared = protocol.reference_hashes(reference, alignment)
    records = {policy: dict(shared) for policy in protocol.CONSENSUS_POLICIES}

    assert all(record == shared for record in records.values())


def test_changing_ground_truth_cannot_change_scores_predictions_or_hashes():
    rng = np.random.RandomState(41)
    q = rng.uniform(size=(21, protocol.VIEW_NUM, protocol.CLASS_NUM))
    q = q / q.sum(axis=2, keepdims=True)
    admission = np.ones((21, protocol.VIEW_NUM), dtype=bool)
    fixed = protocol.semantic_consensus(q, admission, np.ones(6))
    score_hash = tensor_sha256(fixed["consensus_score"])
    prediction_hash = tensor_sha256(fixed["prediction"])
    labels_a = np.arange(21, dtype=np.int64) % 7
    labels_b = labels_a[::-1].copy()

    metrics_a = protocol.metrics_from_fixed_prediction(labels_a, fixed["prediction"])
    metrics_b = protocol.metrics_from_fixed_prediction(labels_b, fixed["prediction"])

    assert not np.array_equal(labels_a, labels_b)
    assert tensor_sha256(fixed["consensus_score"]) == score_hash
    assert tensor_sha256(fixed["prediction"]) == prediction_hash


def test_freezing_and_extraction_leave_backbone_hash_and_grad_state_unchanged():
    autoencoders = [_RawTrackingAutoencoder() for _ in range(protocol.VIEW_NUM)]
    views = [
        np.ones((4, protocol.LATENT_DIM), dtype=np.float32)
        for _ in range(protocol.VIEW_NUM)
    ]
    protocol.freeze_autoencoders(autoencoders)
    before = hash_backbone(autoencoders)

    with torch.no_grad():
        protocol.extract_native_z_q(autoencoders, views)
    after = hash_backbone(autoencoders)
    audit = protocol.frozen_parameter_audit(autoencoders)

    assert before == after
    assert audit["all_requires_grad_false"]
    assert audit["all_grad_none"]


def test_smoke_selection_cannot_change_full_reference_hashes():
    q, global_ids = _synthetic_q_and_global(sample_num=1400)
    identity = np.repeat(
        np.eye(protocol.CLASS_NUM, dtype=np.int64)[None, :, :],
        protocol.VIEW_NUM,
        axis=0,
    )
    reference = {
        "global_prediction": global_ids,
        "global_centers": np.zeros((7, 60), dtype=np.float64),
        "native_nmi_weights": np.ones(6, dtype=np.float64),
        "latent_fusion": np.zeros((1400, 60), dtype=np.float64),
    }
    alignment = {"alignment_matrix": identity, "aligned_q": q}
    before = protocol.reference_hashes(reference, alignment)

    smoke_ids = np.arange(protocol.select_evaluation_count(256), dtype=np.int64)
    _ = q[smoke_ids]
    after = protocol.reference_hashes(reference, alignment)

    assert smoke_ids.shape == (256,)
    assert before == after


def test_frozen_u_loader_verifies_both_logical_and_file_hashes():
    frozen = protocol.load_frozen_u_only()

    assert frozen["U"].shape == (1400, 6)
    assert frozen["U"].flags.writeable is False
    assert frozen["U_sha256"] == protocol.EXPECTED_U_SHA256
    assert frozen["utility_file_sha256"] == protocol.EXPECTED_UTILITY_FILE_SHA256


def test_g2_sources_have_no_update_training_or_utility_refit_calls():
    root = Path(__file__).resolve().parents[1]
    paths = (
        root
        / "experiments/g2_utility_semantic_consensus/g2_consensus_protocol.py",
        root
        / "experiments/g2_utility_semantic_consensus/evaluate_g2_a0_readonly_consensus.py",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    for forbidden in (
        "torch.optim",
        ".backward(",
        ".train(",
        "compute_transfer_scores(",
        "oof_ridge_predictability(",
        "compute_information_utility(",
    ):
        assert forbidden not in source


def test_fixed_constructor_has_no_label_loader_and_run_orders_labels_after_seal():
    from experiments.g2_utility_semantic_consensus import (
        evaluate_g2_a0_readonly_consensus as evaluator,
    )

    constructor_source = inspect.getsource(evaluator._construct_fixed_outputs)
    run_source = inspect.getsource(evaluator.run_evaluation)

    assert "load_caltech_labels_only" not in constructor_source
    assert run_source.index("_construct_fixed_outputs(") < run_source.index(
        "_metrics_after_fixed_output_seal("
    )


def test_evaluator_replays_reference_and_alignment_only_once_for_all_arms():
    from experiments.g2_utility_semantic_consensus import (
        evaluate_g2_a0_readonly_consensus as evaluator,
    )

    source = inspect.getsource(evaluator._construct_fixed_outputs)

    assert source.count("replay_native_global_reference(") == 1
    assert source.count("build_alignment(") == 1
    assert "for policy_name in protocol.CONSENSUS_POLICIES" in source
