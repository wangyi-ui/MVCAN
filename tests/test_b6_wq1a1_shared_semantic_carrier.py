"""Tests for the B6-WQ1A-1 shared semantic carrier prototype."""

import copy
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import experiments.b6_weak_quality.audit_b6_wq1a0_admission_feasibility as wq1a0
import experiments.b6_weak_quality.evaluate_b6_wq0_utility_semantic_admission as b6
import experiments.b6_weak_quality.shared_semantic_isolation as isolation
import experiments.b6_weak_quality.train_b6_wq1a1_shared_semantic_carrier as train
from irv.b3_audit import hash_state_dict


@pytest.fixture(scope="module")
def frozen_seed20():
    return train.prepare_frozen_private_inputs(20)


@pytest.fixture(scope="module")
def trained_correct_pair(frozen_seed20, tmp_path_factory):
    output_root = tmp_path_factory.mktemp("b6_wq1a1_train")
    manifest_before = b6.CANONICAL_MANIFEST_PATH.read_bytes()
    train.set_explicit_rng_seeds(20)
    template = isolation.ViewProjectors()
    initial_state = copy.deepcopy(template.state_dict())
    initial_hash = hash_state_dict(initial_state)
    detached_audit = {
        "utility_requires_grad": False,
        "utility_grad_exists": False,
    }
    records = []
    for repeat_id in range(2):
        records.append(train.train_shared_arm(
            "correct_u_shared",
            frozen_seed20["z_stack"],
            frozen_seed20["correct_admission"],
            initial_state,
            initial_hash,
            train.SMOKE_EPOCHS,
            20,
            output_root / ("repeat_" + str(repeat_id)),
            frozen_seed20["models"].autoencoders,
            detached_audit,
            "WQ1A-0 frozen Utility Top3 binary admission",
        ))
    labels = train.load_evaluation_labels(
        frozen_seed20["data_path"]
    )
    for record in records:
        train._evaluate_and_save_metrics(record, labels)
    manifest_after = b6.CANONICAL_MANIFEST_PATH.read_bytes()
    return {
        "records": records,
        "initial_hash": initial_hash,
        "manifest_before": manifest_before,
        "manifest_after": manifest_after,
    }

@pytest.fixture(scope="module")
def formal_dry_validation(tmp_path_factory):
    output_root = tmp_path_factory.mktemp(
        "b6_wq1a1_formal_dry"
    )
    pilot_path = train._resolve(
        train.PILOT_REFERENCE_SUMMARY
    )
    pilot_before = pilot_path.read_bytes()
    result = train.run_formal_dry_validation(
        seed=20,
        output_dir=output_root,
    )
    pilot_after = pilot_path.read_bytes()
    return {
        "result": result,
        "output_root": output_root,
        "pilot_before": pilot_before,
        "pilot_after": pilot_after,
    }



def test_frozen_z_stack_shape_is_210_by_5_by_10(frozen_seed20):
    assert tuple(frozen_seed20["z_stack"].shape) == (210, 5, 10)


def test_frozen_z_requires_grad_is_false(frozen_seed20):
    z_stack = frozen_seed20["z_stack"]
    assert not z_stack.requires_grad
    assert z_stack.grad is None


def test_frozen_utility_shape_is_210_by_5(frozen_seed20):
    assert tuple(frozen_seed20["utility_tensor"].shape) == (210, 5)


def test_frozen_utility_is_detached(frozen_seed20):
    utility = frozen_seed20["utility_tensor"]
    assert not utility.requires_grad
    assert utility.grad is None


def test_correct_u_admission_is_exactly_three_per_sample(frozen_seed20):
    admission = frozen_seed20["correct_admission"]
    assert admission.shape == (210, 5)
    assert np.all(admission.sum(axis=1) == 3)


def test_correct_u_admission_exactly_reproduces_wq1a0_seed20(
    frozen_seed20,
):
    stored = np.load(
        frozen_seed20["formal_wq1a0_mask_path"],
        allow_pickle=False,
    )
    assert frozen_seed20[
        "B6_WQ1A1_WQ1A0_ADMISSION_REPRO_PASS"
    ]
    assert np.array_equal(
        frozen_seed20["correct_admission"],
        stored,
    )


def test_uniform_admission_is_exactly_five_per_sample(frozen_seed20):
    admission = frozen_seed20["uniform_admission"]
    assert np.all(admission)
    assert np.all(admission.sum(axis=1) == 5)


def test_oracle_admission_is_exactly_three_clean_per_sample(
    frozen_seed20,
):
    admission = frozen_seed20["oracle_admission"]
    assert np.all(admission.sum(axis=1) == 3)
    assert not np.any(
        np.logical_and(
            admission,
            frozen_seed20["corruption_mask"],
        )
    )


def test_shuffled_admission_is_exactly_three_per_sample(frozen_seed20):
    masks, hashes = train.make_shuffled_admissions(
        frozen_seed20["utility_array"],
        train.SMOKE_SHUFFLE_REPEATS,
    )
    assert len(masks) == 2
    assert len(hashes) == 2
    assert all(np.all(mask.sum(axis=1) == 3) for mask in masks)


def test_shuffled_utility_preserves_per_view_distribution(frozen_seed20):
    utility = frozen_seed20["utility_array"]
    bank = wq1a0.generate_permutation_bank()
    shuffled = wq1a0.shuffle_utility_within_views(
        utility,
        bank[0],
    )
    for view_id in range(5):
        assert np.array_equal(
            np.sort(shuffled[:, view_id]),
            np.sort(utility[:, view_id]),
        )


def test_projector_input_and_output_are_n_by_10():
    train.set_explicit_rng_seeds(20)
    projector = isolation.ViewProjectors()
    captured = []

    def capture_input(module, inputs, output):
        del module, output
        captured.append(tuple(inputs[0].shape))

    hook = projector.projectors[0][0].register_forward_hook(
        capture_input
    )
    z_stack = torch.randn(7, 5, 10)
    h_stack = projector(z_stack)
    hook.remove()
    assert captured == [(7, 10)]
    assert tuple(h_stack[:, 0, :].shape) == (7, 10)


def test_h_stack_shape_is_210_by_5_by_10(frozen_seed20):
    train.set_explicit_rng_seeds(20)
    projector = isolation.ViewProjectors()
    with torch.no_grad():
        h_stack = projector(frozen_seed20["z_stack"])
    assert tuple(h_stack.shape) == (210, 5, 10)


def test_pair_selected_index_set_is_exact():
    admission = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
    ], dtype=torch.bool)
    indices = isolation.admitted_pair_indices(
        admission,
        0,
        1,
    )
    assert torch.equal(indices, torch.tensor([0, 3]))


def test_pair_logits_shape_is_m_by_m():
    pair_h_v = F.normalize(torch.randn(6, 10), dim=-1)
    pair_h_w = F.normalize(torch.randn(6, 10), dim=-1)
    _, logits = isolation.symmetric_pair_infonce(
        pair_h_v,
        pair_h_w,
        train.TEMPERATURE,
    )
    assert tuple(logits.shape) == (6, 6)


def test_symmetric_infonce_is_finite():
    pair_h_v = F.normalize(torch.randn(6, 10), dim=-1)
    pair_h_w = F.normalize(torch.randn(6, 10), dim=-1)
    loss, _ = isolation.symmetric_pair_infonce(
        pair_h_v,
        pair_h_w,
        train.TEMPERATURE,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_sample_count_weighted_semantic_loss_formula_is_exact():
    torch.manual_seed(20)
    h_stack = F.normalize(torch.randn(8, 5, 10), dim=-1)
    admission = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
    ], dtype=torch.bool)
    loss, details = isolation.admitted_symmetric_infonce(
        h_stack,
        admission,
        train.TEMPERATURE,
    )
    numerator = sum(
        details["pair_sample_counts"][name] * pair_loss
        for name, pair_loss in details["pair_losses"].items()
        if pair_loss is not None
    )
    expected = numerator / details["weighted_sample_count"]
    assert float(loss.item()) == pytest.approx(expected, rel=1e-6)


def test_only_projector_receives_gradient(trained_correct_pair):
    gradient = trained_correct_pair["records"][0]["gradient_audit"]
    assert gradient["projector_grad_tensor_count"] > 0
    assert gradient["projector_grad_all_finite"]
    assert gradient["encoder_grad_tensor_count"] == 0
    assert gradient["B6_WQ1A1_GRADIENT_PATH_PASS"]


def test_encoder_gradient_count_is_zero(trained_correct_pair):
    for record in trained_correct_pair["records"]:
        assert (
            record["gradient_audit"]["encoder_grad_tensor_count"]
            == 0
        )


def test_private_z_gradient_is_absent(trained_correct_pair):
    for record in trained_correct_pair["records"]:
        assert not record["gradient_audit"][
            "private_z_requires_grad"
        ]
        assert not record["gradient_audit"][
            "private_z_grad_exists"
        ]


def test_utility_gradient_is_absent(trained_correct_pair):
    for record in trained_correct_pair["records"]:
        assert not record["gradient_audit"][
            "utility_requires_grad"
        ]
        assert not record["gradient_audit"][
            "utility_grad_exists"
        ]


def test_all_arm_projectors_can_load_one_exact_initial_state():
    train.set_explicit_rng_seeds(20)
    template = isolation.ViewProjectors()
    state = copy.deepcopy(template.state_dict())
    expected_hash = hash_state_dict(state)
    hashes = []
    for _ in range(6):
        projector = isolation.ViewProjectors()
        projector.load_state_dict(copy.deepcopy(state), strict=True)
        hashes.append(hash_state_dict(projector.state_dict()))
    assert set(hashes) == {expected_hash}


def test_shared_semantic_shape_is_210_by_10(trained_correct_pair):
    semantic = trained_correct_pair["records"][0]["semantic"]
    assert semantic.shape == (210, 10)


def test_shared_semantic_is_finite_and_unit_normalized(
    trained_correct_pair,
):
    semantic = trained_correct_pair["records"][0]["semantic"]
    assert np.isfinite(semantic).all()
    assert np.allclose(
        np.linalg.norm(semantic, axis=1),
        1.0,
        rtol=0.0,
        atol=1e-6,
    )


def test_effective_rank_is_finite_and_within_one_and_ten():
    torch.manual_seed(20)
    semantic = F.normalize(torch.randn(210, 10), dim=-1)
    prediction = np.arange(210) % 7
    diagnostics = isolation.collapse_diagnostics(
        semantic,
        prediction,
    )
    effective_rank = diagnostics["semantic_effective_rank"]
    assert np.isfinite(effective_rank)
    assert 1.0 <= effective_rank <= 10.0


def test_offdiag_cosine_excludes_diagonal():
    semantic = torch.zeros(3, 10)
    semantic[0, 0] = 1.0
    semantic[1, 1] = 1.0
    semantic[2, 2] = 1.0
    offdiag = isolation.offdiag_cosine_values(semantic)
    assert tuple(offdiag.shape) == (6,)
    assert torch.equal(offdiag, torch.zeros(6))


def test_kmeans_prediction_shape_is_210(trained_correct_pair):
    for record in trained_correct_pair["records"]:
        assert record["prediction"].shape == (210,)


def test_labels_are_not_available_before_final_evaluation():
    signature = inspect.signature(train.train_shared_arm)
    assert "labels" not in signature.parameters
    run_source = inspect.getsource(train.run_experiment)
    assert (
        run_source.rfind("train_shared_arm(")
        < run_source.index("load_evaluation_labels(")
    )
    assert "labels_used_for_training" in run_source
    dry_source = inspect.getsource(
        train.run_formal_dry_validation
    )
    for forbidden_call in (
        "train_shared_arm(",
        "load_evaluation_labels(",
        ".fit_predict(",
        ".backward(",
        "torch.optim.",
    ):
        assert forbidden_call not in dry_source



def test_corruption_mask_is_unavailable_to_correct_u_training_path():
    parameters = inspect.signature(
        train.train_shared_arm
    ).parameters
    assert "corruption_mask" not in parameters
    assert "utility" not in parameters


def test_oracle_path_is_explicitly_diagnostic_only():
    metadata = train._arm_metadata(
        "oracle_clean_shared",
        20,
        train.SMOKE_EPOCHS,
        "evaluation-only clean-mask upper bound",
        "a" * 64,
        "b" * 64,
    )
    assert metadata["oracle_diagnostic_only"]


def test_smoke_protocol_is_exactly_three_epochs_and_two_shuffles():
    assert train._mode_protocol("smoke") == (3, 2)
    assert train.SMOKE_EPOCHS == 3
    assert train.SMOKE_SHUFFLE_REPEATS == 2


def test_pilot_protocol_is_exactly_100_epochs_and_20_shuffles():
    assert train._mode_protocol("pilot") == (100, 20)
    assert train.PILOT_EPOCHS == 100
    assert train.PILOT_SHUFFLE_REPEATS == 20


def test_cli_cannot_override_scientific_hyperparameters():
    for option in (
        "--temperature",
        "--tau",
        "--top-k",
        "--semantic-dim",
        "--hidden-dim",
        "--epochs",
        "--shuffle-repeats",
        "--learning-rate",
        "--lr",
        "--weight-decay",
    ):
        with pytest.raises(SystemExit):
            train.parse_args([
                "--mode", "formal",
                "--seed", "20",
                option, "999",
            ])


def test_formal_supports_only_seed20_at_present(
    formal_dry_validation,
):
    assert formal_dry_validation["result"]["model_seed"] == 20
    _, pilot_path, pilot_hash = (
        train.load_validated_pilot_reference(20)
    )
    assert pilot_path.is_file()
    assert pilot_hash == train.EXPECTED_PILOT_REFERENCE_SHA256
    for seed in (30, 50):
        with pytest.raises(
            RuntimeError,
            match="formal supports only seed20",
        ):
            train.run_formal_dry_validation(seed=seed)


def test_formal_protocol_is_exactly_100_epochs_and_200_shuffles():
    assert train._mode_protocol("formal") == (100, 200)
    assert train.FORMAL_SHUFFLE_REPEATS == 200


def test_formal_hyperparameters_are_identical_to_pilot(
    formal_dry_validation,
):
    protocol = formal_dry_validation["result"]["fixed_protocol"]
    assert train._mode_protocol("formal")[0] == train._mode_protocol(
        "pilot"
    )[0]
    assert protocol == {
        "epochs": 100,
        "shuffled_repeats": 200,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "temperature": 0.2,
        "semantic_dim": 10,
        "hidden_dim": 32,
        "top_k": 3,
        "full_batch_size": 210,
        "kmeans_n_init": 100,
        "kmeans_random_state": 20,
    }


def test_formal_empirical_p_formula_is_exact():
    shuffled_acc = np.concatenate((
        np.zeros(190, dtype=np.float64),
        np.ones(10, dtype=np.float64),
    ))
    audit = train.formal_acc_null_audit(
        shuffled_acc,
        correct_acc=0.5,
    )
    assert audit["count_shuffled_ge_correct"] == 10
    assert audit["one_sided_empirical_p"] == 11 / 201


def test_formal_p95_gate_is_strict_and_correct():
    passing_null = np.concatenate((
        np.zeros(195, dtype=np.float64),
        np.ones(5, dtype=np.float64),
    ))
    passing = train.formal_acc_null_audit(
        passing_null,
        correct_acc=0.5,
    )
    assert passing["formal_shuffled_acc_p95"] == 0.0
    assert passing["one_sided_empirical_p"] < 0.05
    assert passing["B6_WQ1A1_FORMAL_NULL_PASS"]

    tied = train.formal_acc_null_audit(
        np.zeros(200, dtype=np.float64),
        correct_acc=0.0,
    )
    assert tied["formal_shuffled_acc_p95"] == 0.0
    assert not tied["B6_WQ1A1_FORMAL_NULL_PASS"]


def test_formal_seed20_gate_is_exact_conjunction():
    assert train.formal_seed20_gate(*([True] * 9))
    for index in range(9):
        values = [True] * 9
        values[index] = False
        assert not train.formal_seed20_gate(*values)


def test_formal_dry_validation_checks_all_200_initial_states(
    formal_dry_validation,
):
    result = formal_dry_validation["result"]
    hashes = result[
        "shuffled_projector_initial_state_sha256"
    ]
    assert len(hashes) == 200
    assert set(hashes) == {
        result["canonical_projector_initial_state_sha256"]
    }
    assert len(result["shuffled_utility_sha256"]) == 200
    assert result["B6_WQ1A1_PROJECTOR_INIT_MATCH_PASS"]
    assert result["B6_WQ1A1_FORMAL_ADMISSION_COUNT_PASS"]


def test_formal_dry_validation_performs_no_scientific_run(
    formal_dry_validation,
):
    result = formal_dry_validation["result"]
    assert result["B6_WQ1A1_FORMAL_DRY_VALIDATION_PASS"]
    assert not result["training_performed"]
    assert not result["optimizer_constructed"]
    assert not result["backward_performed"]
    assert not result["kmeans_fit_performed"]
    assert not result["labels_loaded"]
    assert not result["formal_scientific_run_performed"]
    assert not result[
        "multiseed_scientific_conclusion_performed"
    ]


def test_pilot_artifact_is_byte_exact_after_formal_dry_validation(
    formal_dry_validation,
):
    assert (
        formal_dry_validation["pilot_before"]
        == formal_dry_validation["pilot_after"]
    )
    assert hashlib.sha256(
        formal_dry_validation["pilot_after"]
    ).hexdigest() == train.EXPECTED_PILOT_REFERENCE_SHA256
    assert formal_dry_validation["result"][
        "B6_WQ1A1_PROTECTED_ARTIFACTS_UNCHANGED_PASS"
    ]
    assert (
        formal_dry_validation["output_root"]
        / "formal_dry_validation.json"
    ).is_file()


def test_formal_artifact_contract_is_wired_without_running_science():
    source = inspect.getsource(train.run_experiment)
    assert "b6_wq1a1_seed20_formal_summary.json" in source
    assert '"shuffled_" + metric + ".npy"' in source


def test_dry_run_is_available_only_for_formal_mode():
    with pytest.raises(
        RuntimeError,
        match="supported only with --mode formal",
    ):
        train.main([
            "--mode", "smoke",
            "--seed", "20",
            "--dry-run",
        ])


def test_seed30_and_seed50_scientific_execution_are_rejected():
    for seed in (30, 50):
        with pytest.raises(
            RuntimeError,
            match="supports only seed20",
        ):
            train.run_experiment("smoke", seed)


def test_training_uses_only_preregistered_semantic_loss():
    source = inspect.getsource(train.train_shared_arm)
    assert "admitted_symmetric_infonce" in source
    for forbidden in (
        "reconstruction",
        "cluster_loss",
        "prototype",
        "entropy_loss",
        "VICReg",
        "Barlow",
        "rate_loss",
        "kl_loss",
    ):
        assert forbidden not in source


def test_direct_utility_feature_scaling_is_not_reintroduced():
    source = inspect.getsource(train)
    assert "b6.admission_weights(" not in source
    assert "fused_representation_from_admission(" not in source


def test_deterministic_correct_u_training_is_numerically_exact(
    trained_correct_pair,
):
    first, second = trained_correct_pair["records"]
    assert first["initial_state_sha256"] == second[
        "initial_state_sha256"
    ]
    assert np.array_equal(first["admission"], second["admission"])
    assert first["history"] == second["history"]
    assert first["semantic_sha256"] == second["semantic_sha256"]
    assert first["prediction_sha256"] == second["prediction_sha256"]
    assert first["metrics"] == second["metrics"]


def test_each_trained_arm_writes_required_artifacts(trained_correct_pair):
    required = {
        "metadata.json",
        "training_history.json",
        "final_semantic.npy",
        "prediction.npy",
        "admission_mask.npy",
        "metrics.json",
        "collapse_diagnostics.json",
        "gradient_audit.json",
        "pair_sample_counts.json",
    }
    for record in trained_correct_pair["records"]:
        assert required <= {
            path.name for path in Path(record["arm_dir"]).iterdir()
        }
        assert len(record["initial_state_sha256"]) == 64
        assert len(record["final_state_sha256"]) == 64
        assert len(record["semantic_sha256"]) == 64
        assert len(record["prediction_sha256"]) == 64


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_canonical_manifest_and_protected_scientific_files_are_unchanged(
    trained_correct_pair,
):
    assert (
        trained_correct_pair["manifest_before"]
        == trained_correct_pair["manifest_after"]
    )
    expected = {
        "model.py": "3f866536857f0a5df5d451dec93232c77f0805ff216db60a2dd894b0c1a357c9",
        "run.py": "3a766742ca331471e7ece7f111921b7ffc44c6e883728e43ac7583741427fdc4",
        "ClusteringTest.py": "5dd576ecd2f9fef161741608f5c94429e86c28fcb5d38cb443e22b8db7e092b4",
        "configure.py": "20ea7aee71a66e3ef741056197afba731e96fa8ef4d95b09607fdc4dc601ec7a",
        "datasets.py": "52ec339eac819b5a913d3c4c7b6dbdb13dcd75a493f2da0efbf66303ba908d80",
        "weak_quality.py": "2fd09439e9375392859db875dbd2dd03275ccfe383013b8717967b68eec9c83d",
        "experiments/b6_weak_quality/evaluate_b6_wq0_utility_semantic_admission.py": "10cc2c09d6bfd4b675853b29601af75a875cd7c2483f301f06a459eb62251bda",
        "experiments/b6_weak_quality/audit_b6_wq1a0_admission_feasibility.py": "5dc407e4c7e87e269fddf867308093a84417ab4e3d871d388bf96ea98f0989e0",
        "experiments/b6_weak_quality/shared_semantic_isolation.py": "7beb1d95319f5939acf04738865072932446ab1678cc30c039f0139cb1ec4fe4",
        "outputs/b6_weak_quality/wq1a1_shared_semantic_carrier/pilot_seed20/b6_wq1a1_seed20_summary.json": "662cec723a3499b4e309677f9f6ff3338e3d9d565e6c46a02c237c86087043ba",
        "experiments/b6_weak_quality/b6_wq0_frozen_baseline_manifest.json": "f7ba5bd25dfe4dfdbbb3db1f4956c601db33bf3fffce8f456ff2447c4ffa187e",
    }
    for relative_path, expected_hash in expected.items():
        assert _sha256(
            train.REPOSITORY_ROOT / relative_path
        ) == expected_hash
