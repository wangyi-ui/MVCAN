"""Unit-only tests for E1 optimization-seed parameterization."""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.e1_pairwise_utility import (
    train_e1_pairwise_utility as train,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
UNRELATED_FROZEN_SOURCE_SHA256 = {
    "experiments/cyclic_utility/c0_complementary_semantic_verification.py": (
        "d52fbf0816557ba57a11fc490ead1a26598b35e68bac7808b7077c448bc7a4a9"
    ),
    "experiments/cyclic_utility/evaluate_c0_complementary_semantic_verification.py": (
        "adb420bbc59833ff7eca58f3b5d54162a45b4106227d8f6540fa69fc22ac691b"
    ),
    "experiments/cyclic_utility/vsa_decoupled_action_protocol.py": (
        "4dc1bdf75895319f41131ed446eb310192c42948cecc6f2234870cbf0eda8091"
    ),
    "experiments/cyclic_utility/train_vsa_a0_decoupled_action.py": (
        "c441fc4d7aaba9f192b19d0fe6138f1c5503132fa3a61928604a3e3a517fd3ee"
    ),
    "experiments/cyclic_utility/summarize_vsa_a0_decoupled_action.py": (
        "80e63642d2041178ab82fc494a5f708a2569ad761c4e1a30bee46a5d23b0888f"
    ),
}


def test_parser_default_seed_is_20():
    args = train.parse_args(["--arm", "LWC", "--epochs", "100"])
    assert args.seed == 20


@pytest.mark.parametrize("training_seed", (20, 30, 50))
def test_parser_accepts_controlled_training_seeds(training_seed):
    args = train.parse_args([
        "--arm",
        "LWC",
        "--epochs",
        "100",
        "--seed",
        str(training_seed),
    ])
    assert args.seed == training_seed


def test_parser_rejects_unsupported_seed():
    with pytest.raises(SystemExit):
        train.parse_args([
            "--arm", "LWC", "--epochs", "100", "--seed", "40"
        ])


@pytest.mark.parametrize(
    "training_seed,expected_name",
    (
        (20, "lwc_100ep_seed20"),
        (30, "lwc_100ep_seed30"),
        (50, "lwc_100ep_seed50"),
    ),
)
def test_default_output_name_uses_training_seed(
    training_seed, expected_name
):
    assert train.default_output_dir(
        "LWC", 100, training_seed
    ).name == expected_name


def test_set_global_seed_receives_requested_training_seed(monkeypatch):
    calls = {}
    monkeypatch.setenv("PYTHONHASHSEED", "test-original")
    monkeypatch.setattr(train.random, "seed", lambda seed: calls.setdefault("random", seed))
    monkeypatch.setattr(train.np.random, "seed", lambda seed: calls.setdefault("numpy", seed))
    monkeypatch.setattr(train.torch, "manual_seed", lambda seed: calls.setdefault("torch", seed))
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)

    train.set_global_seed(30)

    assert calls == {"random": 30, "numpy": 30, "torch": 30}
    assert train.os.environ["PYTHONHASHSEED"] == "30"


@pytest.mark.parametrize("training_seed", (20, 30, 50))
def test_kmeans_random_state_receives_training_seed(
    monkeypatch, training_seed
):
    captured = {}

    class ConstructorReached(Exception):
        pass

    def fake_kmeans(**kwargs):
        captured.update(kwargs)
        raise ConstructorReached

    monkeypatch.setattr(train, "KMeans", fake_kmeans)
    with pytest.raises(ConstructorReached):
        train.refresh_native_target(
            None, None, None, "cpu", training_seed=training_seed
        )
    assert captured["random_state"] == training_seed
    assert captured["n_init"] == train.KMEANS_N_INIT


@pytest.mark.parametrize("training_seed", (20, 30, 50))
def test_explicit_torch_generator_uses_training_seed(training_seed):
    generator = train.build_sample_order_generator(training_seed)
    assert generator.initial_seed() == training_seed


def test_sample_order_rng_changes_with_training_seed():
    orders = {
        seed: torch.randperm(
            train.SAMPLE_NUM,
            generator=train.build_sample_order_generator(seed),
        )
        for seed in train.TRAINING_SEED_CHOICES
    }
    assert not torch.equal(orders[20], orders[30])
    assert not torch.equal(orders[20], orders[50])
    assert not torch.equal(orders[30], orders[50])


@pytest.mark.parametrize("training_seed", (20, 30, 50))
def test_audit_seed_fields_use_requested_training_seed(training_seed):
    audit = train.training_seed_audit_fields(training_seed)
    assert audit["seed"] == training_seed
    assert audit["training_seed"] == training_seed
    assert audit["sample_order_rng_seed"] == training_seed


def test_d1_checkpoint_default_remains_fixed_seed20():
    assert str(train.DEFAULT_CHECKPOINT_DIR).endswith(
        "outputs/d1_caltech6v/snr2p5_k3_seed20/models"
    )


def test_d2_reliability_defaults_remain_fixed_seed20():
    assert str(train.DEFAULT_RELIABILITY_PATH).endswith(
        "outputs/d2_caltech6v/a0_utility_transfer_seed20/utility_scores.npz"
    )
    assert str(train.DEFAULT_D2_AUDIT_PATH).endswith(
        "outputs/d2_caltech6v/a0_utility_transfer_seed20/utility_transfer.json"
    )


def test_e0_feature_defaults_remain_fixed_seed20():
    assert str(train.DEFAULT_FEATURE_PATH).endswith(
        "outputs/e0_glgc_adapter/caltech6v_snr2p5_k3_seed20.npz"
    )
    assert str(train.DEFAULT_FEATURE_AUDIT_PATH).endswith(
        "outputs/e0_glgc_adapter/export_audit.json"
    )


def test_d2_condition_check_remains_fixed_seed20():
    source = inspect.getsource(train.load_frozen_reliability)
    assert 'd2_audit.get("condition") == "snr2p5_k3_seed20"' in source
    assert "training_seed" not in inspect.signature(
        train.load_frozen_reliability
    ).parameters


@pytest.mark.parametrize("training_seed", (20, 30, 50))
def test_weak_quality_realization_is_not_tied_to_training_seed(training_seed):
    audit = train.seed_semantics_audit(training_seed)
    assert audit["training_seed"] == training_seed
    assert audit["weak_quality_condition_fixed"] == "snr2p5_k3_seed20"
    assert audit["weak_quality_realization_varied"] is False


def test_model_constructor_receives_requested_training_seed(
    monkeypatch, tmp_path
):
    captured = {}

    class ConstructorReached(Exception):
        pass

    def fake_model(**kwargs):
        captured.update(kwargs)
        raise ConstructorReached

    monkeypatch.setattr(train, "MvCAN", fake_model)
    with pytest.raises(ConstructorReached):
        train.build_model_from_frozen_d1(
            tmp_path, torch.device("cpu"), training_seed=50
        )
    assert captured["seed"] == 50


def test_training_continuation_threads_seed_to_all_stochastic_helpers():
    source = inspect.getsource(train.train_continuation)
    assert "build_sample_order_generator(training_seed)" in source
    assert "reliability, seed=training_seed" in source
    assert source.count("training_seed=training_seed") == 2


def test_no_algorithm_or_loss_coefficient_changed():
    assert train.LEARNING_RATE == 1e-4
    assert train.LAMBDA1 == 0.01
    assert train.TARGET_REFRESH_INTERVAL == 100
    assert train.TARGET_WEIGHT_UPDATES == 2
    assert train.KMEANS_N_INIT == 100
    native_source = inspect.getsource(train.native_mvcan_losses)
    continuation_source = inspect.getsource(train.train_continuation)
    assert "reconstruction_loss + LAMBDA1 * clustering_loss" in native_source
    assert "native_total + LAMBDA1 * pair_loss" in continuation_source


@pytest.mark.parametrize(
    "relative_path,expected_hash",
    tuple(UNRELATED_FROZEN_SOURCE_SHA256.items()),
)
def test_c0_and_vsa_sources_are_unmodified(relative_path, expected_hash):
    assert train.file_sha256(REPOSITORY_ROOT / relative_path) == expected_hash


def test_seed20_backward_compatibility_of_parser_output_and_order():
    old = train.parse_args(["--arm", "LWC", "--epochs", "100"])
    explicit = train.parse_args([
        "--arm", "LWC", "--epochs", "100", "--seed", "20"
    ])
    assert vars(old) == vars(explicit)
    assert train.default_output_dir("LWC", 100) == train.default_output_dir(
        "LWC", 100, 20
    )
    old_order = torch.randperm(
        train.SAMPLE_NUM,
        generator=train.build_sample_order_generator(),
    )
    explicit_order = torch.randperm(
        train.SAMPLE_NUM,
        generator=train.build_sample_order_generator(20),
    )
    assert torch.equal(old_order, explicit_order)


def test_historical_call_signatures_default_to_seed20():
    assert inspect.signature(
        train.build_model_from_frozen_d1
    ).parameters["training_seed"].default == 20
    assert inspect.signature(
        train.refresh_native_target
    ).parameters["training_seed"].default == 20
    assert inspect.signature(
        train.train_continuation
    ).parameters["training_seed"].default == 20


def test_parser_description_no_longer_claims_seed20_only():
    source = inspect.getsource(train.parse_args)
    assert "controlled multi-seed continuation" in source
    assert "seed20 only" not in source


def test_argument_parsing_and_output_naming_create_no_directories(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(train, "REPOSITORY_ROOT", tmp_path)
    args = train.parse_args([
        "--arm", "LWC", "--epochs", "100", "--seed", "30"
    ])
    output = train.default_output_dir(args.arm, args.epochs, args.seed)
    assert output.name == "lwc_100ep_seed30"
    assert not output.exists()


def test_overwrite_refusal_precedes_output_directory_creation():
    source = inspect.getsource(train.main)
    assert source.index("refusing to overwrite E1 output") < source.index(
        "output_dir.mkdir"
    )


def test_python38_compatible_syntax_without_ast_unparse():
    source = Path(train.__file__).read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 8))
    assert "ast.unparse" not in source
