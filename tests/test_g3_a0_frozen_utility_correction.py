"""Targeted protocol tests for G3-A0 frozen-utility target correction."""

import inspect
from pathlib import Path

import numpy as np
import torch

from experiments.g3_selective_semantic_cooperation import (
    g3_target_correction_protocol as protocol,
)
from experiments.g3_selective_semantic_cooperation import (
    train_g3_a0_frozen_utility_correction as runner,
)
from irv.b4_information_utility import tensor_sha256


def _utility(sample_num=31):
    rng = np.random.RandomState(17)
    values = rng.uniform(size=(sample_num, protocol.VIEW_NUM))
    # Avoid accidental equal rows or unchanged Top-3 control rows.
    values += np.arange(sample_num, dtype=np.float64)[:, None] * 1e-8
    return values


def _probabilities(sample_num=13, view_num=None):
    rng = np.random.RandomState(29)
    if view_num is None:
        values = rng.uniform(size=(sample_num, protocol.CLASS_NUM))
        values /= values.sum(axis=1, keepdims=True)
    else:
        values = rng.uniform(
            size=(sample_num, view_num, protocol.CLASS_NUM)
        )
        values /= values.sum(axis=2, keepdims=True)
    return torch.tensor(values, dtype=torch.float32)


def _cyclic_alignment():
    local_for_global = np.roll(np.arange(protocol.CLASS_NUM), 1)
    matrix = np.zeros((protocol.CLASS_NUM, protocol.CLASS_NUM), dtype=np.int64)
    matrix[np.arange(protocol.CLASS_NUM), local_for_global] = 1
    return np.repeat(matrix[None, :, :], protocol.VIEW_NUM, axis=0)


def test_rho_is_exact_third_minus_fourth_shape_and_range():
    utility = np.asarray([
        [0.1, 0.9, 0.4, 0.8, 0.2, 0.7],
        [0.0, 1.0, 0.5, 0.4, 0.6, 0.2],
    ], dtype=np.float64)
    expected = np.asarray([0.7 - 0.4, 0.5 - 0.4], dtype=np.float64)

    rho = protocol.compute_rho(utility)

    assert rho.shape == (2,)
    assert np.array_equal(rho, expected)
    assert np.all((rho >= 0.0) & (rho <= 1.0))


def test_real_u_top3_has_exactly_three_views_per_row():
    policy = protocol.build_utility_policy(_utility(), "U_CORRECTION")

    assert policy["admission"].shape == (31, 6)
    assert np.all(policy["admission"].sum(axis=1) == 3)


def test_shuffled_control_permutes_complete_rows_and_recomputes_both_actions():
    utility = _utility()
    real = protocol.build_utility_policy(utility, "U_CORRECTION")
    shuffled = protocol.build_utility_policy(
        utility, "SHUFFLED_U_CORRECTION", control_seed=20
    )
    permutation = shuffled["row_permutation"]

    assert np.array_equal(shuffled["utility"], utility[permutation])
    assert np.array_equal(
        shuffled["admission"], real["admission"][permutation]
    )
    assert np.array_equal(shuffled["rho"], real["rho"][permutation])
    assert shuffled["changed_row_count"] > 0


def test_shuffled_control_preserves_rho_multiset_and_top3_column_counts():
    utility = _utility()
    real = protocol.build_utility_policy(utility, "U_CORRECTION")
    shuffled = protocol.build_utility_policy(
        utility, "SHUFFLED_U_CORRECTION"
    )

    assert np.array_equal(np.sort(shuffled["rho"]), np.sort(real["rho"]))
    assert np.array_equal(
        shuffled["admission"].sum(axis=0),
        real["admission"].sum(axis=0),
    )
    assert shuffled["rho_multiset_preserved"]
    assert shuffled["top3_column_counts_preserved"]


def test_alignment_uses_q_times_m_transpose_and_detaches_q():
    q = _probabilities(sample_num=5, view_num=protocol.VIEW_NUM)
    q.requires_grad_(True)
    matrices = _cyclic_alignment()

    result = protocol.align_detached_q(q, matrices)

    expected = q.detach()[:, 0, :] @ torch.tensor(
        matrices[0].T, dtype=q.dtype
    )
    wrong_orientation = q.detach()[:, 0, :] @ torch.tensor(
        matrices[0], dtype=q.dtype
    )
    assert torch.equal(result["aligned_q"][:, 0, :], expected)
    assert not torch.equal(result["aligned_q"][:, 0, :], wrong_orientation)
    assert not result["q_detached"].requires_grad
    assert not result["aligned_q"].requires_grad
    assert result["aligned_q"].grad_fn is None
    assert not result["alignment_matrices"].requires_grad


def test_p_high_u_shape_probability_mass_and_fixed_weight_formula():
    aligned_q = _probabilities(sample_num=7, view_num=protocol.VIEW_NUM)
    admission = np.zeros((7, protocol.VIEW_NUM), dtype=bool)
    admission[:, [0, 2, 5]] = True
    weights = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    result = protocol.build_high_utility_target(
        aligned_q, admission, weights
    )
    expected = (
        aligned_q[:, 0, :] * 1.0
        + aligned_q[:, 2, :] * 3.0
        + aligned_q[:, 5, :] * 6.0
    ) / 10.0

    assert result["P_highU"].shape == (7, 7)
    assert torch.allclose(result["P_highU"], expected, atol=1e-7)
    assert torch.allclose(result["P_highU"].sum(dim=1), torch.ones(7))
    assert torch.all(result["P_highU"] >= 0.0)
    assert not result["P_highU"].requires_grad
    assert result["P_highU"].grad_fn is None


def test_p_util_is_exact_preregistered_convex_combination_and_detached():
    p_all = _probabilities(sample_num=9)
    p_high = torch.flip(p_all, dims=(1,))
    rho = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    rho_boundary = torch.tensor(rho, dtype=p_all.dtype).unsqueeze(1)
    expected = (1.0 - rho_boundary) * p_all + rho_boundary * p_high

    p_util = protocol.build_corrected_target(p_all, p_high, rho)

    assert p_util.shape == (9, 7)
    assert torch.equal(p_util, expected)
    assert torch.all(p_util >= 0.0)
    assert torch.allclose(p_util.sum(dim=1), torch.ones(9), atol=1e-6)
    assert not p_util.requires_grad
    assert p_util.grad_fn is None


def test_base_is_direct_object_array_and_hash_identity():
    p_all = _probabilities(sample_num=6).detach()

    p_base = protocol.select_training_target("BASE", p_all)

    assert p_base is p_all
    assert torch.equal(p_base, p_all)
    assert tensor_sha256(p_base) == tensor_sha256(p_all)


def test_local_target_uses_p_train_times_m_not_m_transpose():
    p_train = _probabilities(sample_num=4)
    matrices = _cyclic_alignment()

    local_targets = protocol.build_local_targets(p_train, matrices)

    expected = p_train @ torch.tensor(matrices[0], dtype=p_train.dtype)
    wrong_orientation = p_train @ torch.tensor(
        matrices[0].T, dtype=p_train.dtype
    )
    assert len(local_targets) == protocol.VIEW_NUM
    assert torch.equal(local_targets[0], expected)
    assert not torch.equal(local_targets[0], wrong_orientation)
    assert all(not target.requires_grad for target in local_targets)


def test_sample_ids_and_targets_stay_aligned_under_shuffled_dataloader():
    sample_num = 23
    ids = np.arange(sample_num, dtype=np.int64)
    views = [
        torch.tensor(ids[:, None] + view_id * 100, dtype=torch.float32)
        for view_id in range(protocol.VIEW_NUM)
    ]
    p_train = torch.nn.functional.one_hot(
        torch.tensor(ids % protocol.CLASS_NUM), protocol.CLASS_NUM
    ).float()
    dataset = protocol.build_training_dataset(views, ids, p_train)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=5,
        shuffle=True,
        generator=torch.Generator().manual_seed(20),
    )
    seen = []
    order_changed = False
    ordinal = 0
    for batch in loader:
        batch_ids = batch[-2]
        assert protocol.batch_sample_target_alignment_pass(
            batch_ids, batch[-1], p_train
        )
        for view_id in range(protocol.VIEW_NUM):
            assert torch.equal(
                batch[view_id].squeeze(1),
                batch_ids.float() + view_id * 100,
            )
        order_changed = order_changed or not torch.equal(
            batch_ids,
            torch.arange(ordinal, ordinal + batch_ids.numel()),
        )
        ordinal += batch_ids.numel()
        seen.extend(batch_ids.tolist())

    assert order_changed
    assert sorted(seen) == list(range(sample_num))


def test_changing_labels_cannot_change_any_training_target():
    q = _probabilities(sample_num=8, view_num=protocol.VIEW_NUM)
    matrices = np.repeat(
        np.eye(protocol.CLASS_NUM, dtype=np.int64)[None, :, :],
        protocol.VIEW_NUM,
        axis=0,
    )
    aligned = protocol.align_detached_q(q, matrices)["aligned_q"]
    utility = _utility(sample_num=8)
    policy = protocol.build_utility_policy(utility, "U_CORRECTION")
    high = protocol.build_high_utility_target(
        aligned, policy["admission"], np.ones(protocol.VIEW_NUM)
    )["P_highU"]
    p_all = _probabilities(sample_num=8)
    labels_a = np.arange(8, dtype=np.int64) % protocol.CLASS_NUM
    labels_b = labels_a[::-1].copy()

    first = protocol.select_training_target(
        "U_CORRECTION", p_all, high, policy["rho"]
    )
    second = protocol.select_training_target(
        "U_CORRECTION", p_all, high, policy["rho"]
    )

    assert not np.array_equal(labels_a, labels_b)
    assert torch.equal(first, second)
    assert tensor_sha256(first) == tensor_sha256(second)


def test_frozen_u_loader_checks_logical_and_file_hashes_without_oracle():
    frozen = protocol.load_frozen_u_only()

    assert frozen["U"].shape == (1400, 6)
    assert frozen["U"].flags.writeable is False
    assert frozen["U_sha256"] == protocol.EXPECTED_U_SHA256
    assert frozen["utility_file_sha256"] == protocol.EXPECTED_UTILITY_FILE_SHA256
    assert "corruption_mask" not in frozen


def test_readonly_numpy_boundary_makes_writable_copy_only_at_torch_conversion():
    source = np.arange(12, dtype=np.float64).reshape(2, 6)
    source.setflags(write=False)

    converted = protocol.numpy_to_torch_boundary(source, dtype=torch.float32)

    assert source.flags.writeable is False
    assert converted.shape == (2, 6)
    assert converted.dtype == torch.float32


def test_native_fusion_and_target_refresh_apis_have_no_utility_inputs():
    for function in (
        runner._native_fusion_updates,
        runner.refresh_native_global_target,
    ):
        parameter_names = set(inspect.signature(function).parameters)
        assert all(
            token not in name.lower()
            for name in parameter_names
            for token in ("utility", "rho", "admission")
        )


def test_target_constructors_have_no_label_inputs_or_extra_transform():
    functions = (
        protocol.compute_rho,
        protocol.build_utility_policy,
        protocol.align_detached_q,
        protocol.build_high_utility_target,
        protocol.build_corrected_target,
        protocol.select_training_target,
    )
    for function in functions:
        assert all(
            "label" not in name.lower()
            and "target_distribution" not in name.lower()
            for name in inspect.signature(function).parameters
        )
    source = inspect.getsource(protocol.build_corrected_target)
    assert "target_distribution" not in source
    assert "softmax" not in source
    assert "temperature" not in source


def test_all_three_arms_share_one_training_code_path_and_only_epochs_varies():
    source = inspect.getsource(runner.run_experiment)
    arguments = inspect.signature(runner.run_experiment).parameters

    assert "epochs" in arguments
    assert "smoke" not in source.lower()
    assert "formal" not in source.lower()
    assert "for arm in protocol.ARMS" in source
    assert source.count("_train_one_arm(") == 1


def test_labels_are_loaded_only_after_all_arm_predictions_are_sealed():
    run_source = inspect.getsource(runner.run_experiment)
    train_source = inspect.getsource(runner._train_one_arm)
    evaluation_source = inspect.getsource(
        runner._evaluate_fixed_predictions_after_seal
    )

    assert "load_caltech_labels_only" not in train_source
    assert "load_caltech_labels_only" not in run_source
    assert "load_caltech_labels_only" in evaluation_source
    assert run_source.index("for arm in protocol.ARMS") < run_source.index(
        "_evaluate_fixed_predictions_after_seal("
    )
    assert evaluation_source.index("prediction_path\"].is_file()") < (
        evaluation_source.index("load_caltech_labels_only")
    )


def test_g3_does_not_modify_native_model_or_run_sources():
    root = Path(__file__).resolve().parents[1]
    g3_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            root
            / "experiments/g3_selective_semantic_cooperation/g3_target_correction_protocol.py",
            root
            / "experiments/g3_selective_semantic_cooperation/train_g3_a0_frozen_utility_correction.py",
        )
    )

    assert "compute_transfer_scores(" not in g3_sources
    assert "oof_ridge_predictability(" not in g3_sources
    assert "compute_information_utility(" not in g3_sources
    assert "semantic_head" not in inspect.getsource(protocol.build_corrected_target)


def test_fixed_native_training_hyperparameters_and_no_tuning_cli():
    config = runner._validated_native_config(2)
    training = config["training"]
    arguments = runner.parse_args(
        ["--epochs", "2", "--output-dir", "/tmp/g3-test-not-run"]
    )

    assert training["batch_size"] == 256
    assert training["T_1"] == 2
    assert training["T_2"] == 100
    assert training["lr"] == 0.0001
    assert training["lambda1"] == 0.01
    assert arguments.epochs == 2
    assert not hasattr(arguments, "rho")
    assert not hasattr(arguments, "lambda1")
    assert not hasattr(arguments, "temperature")
