import hashlib
import inspect
from pathlib import Path

import numpy as np
import scipy.io as sio

from configure import get_default_config
from experiments.generic_contract.action_space import (
    build_action_space,
    historical_action_logical_sha256,
)
from experiments.generic_contract.dataset_contract import DatasetContract
from experiments.generic_contract.generic_final_core_adapter import (
    BDGP_RUNTIME_SPEC,
    FEATURE_FIELDS,
    MSRC_RUNTIME_SPEC,
    PRE_GT_FIELDS,
    run_final_core,
    validate_materialized_inputs,
)
from experiments.generic_contract.generic_relation_action import (
    build_relation_semantics,
)
from experiments.generic_contract.generic_weak_quality import (
    generate_half_corruption_mask,
)
from experiments.generic_contract.materialize_g0_b1_bdgp_inputs import (
    DATASET_PATH,
    DATASET_SHA256,
    FEATURE_FIELDS as BDGP_FEATURE_FIELDS,
    load_authoritative_bdgp,
)
from experiments.generic_contract.run_g0_b1_bdgp_structural_pilot import (
    main as bdgp_runner_main,
)
from experiments.generic_contract.sparse_label_contract import (
    materialize_hash_ranked_sparse_split,
)


EXPECTED_GENERATOR_SHA = (
    "8a779fd09d0f7a4ff666981de44701140fed6eb7ee405b7927b6813ce777be80"
)
EXPECTED_VERIFIER_SHA = (
    "2424701ad3aa3b1a8c2c8f8aefab6929471ee2bae87b885df1c074c307923105"
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_bdgp_dataset_sha_and_authoritative_two_view_metadata():
    assert _sha256(DATASET_PATH) == DATASET_SHA256
    fields = {name: (shape, dtype) for name, shape, dtype in sio.whosmat(DATASET_PATH)}
    assert fields["X1"] == ((2500, 1750), "double")
    assert fields["X2"] == ((2500, 79), "uint8")
    assert fields["X3"] == ((2500, 10), "double")
    assert fields["Y"] == ((1, 2500), "int64")
    assert BDGP_FEATURE_FIELDS == ("X1", "X2", "sample_ids")
    assert "X3" not in BDGP_FEATURE_FIELDS


def test_bdgp_loader_requests_only_x1_x2_y_and_preserves_native_orientation(
    monkeypatch,
):
    calls = []

    def fake_loadmat(path, variable_names):
        calls.append((Path(path), variable_names))
        return {
            "X1": np.zeros((2500, 1750), dtype=np.float64),
            "X2": np.zeros((2500, 79), dtype=np.uint8),
            "Y": np.repeat(np.arange(5, dtype=np.int64), 500)[None, :],
            "X3": np.ones((2500, 10), dtype=np.float64),
        }

    monkeypatch.setattr(
        "experiments.generic_contract.materialize_g0_b1_bdgp_inputs.sio.loadmat",
        fake_loadmat,
    )
    views, labels = load_authoritative_bdgp()
    assert calls == [(DATASET_PATH, ["X1", "X2", "Y"])]
    assert [view.shape for view in views] == [(2500, 1750), (2500, 79)]
    assert all(view.dtype == np.float32 for view in views)
    assert labels.shape == (2500,)
    assert labels.dtype == np.int64


def test_bdgp_runtime_contract_exact():
    spec = BDGP_RUNTIME_SPEC
    assert (spec.N, spec.V, spec.K, spec.view_dims) == (2500, 2, 5, (1750, 79))
    assert (spec.labels_per_class, spec.L, spec.N_u, spec.S) == (2, 10, 2490, 2)
    contract = DatasetContract(
        spec.dataset_name,
        spec.N,
        spec.V,
        spec.K,
        spec.view_dims,
        spec.labels_per_class,
    )
    assert contract.tensor_shapes["q_local"] == (2500, 2, 5)
    assert contract.tensor_shapes["q_aligned"] == (2500, 2, 5)
    assert (spec.V, spec.K, spec.K) == (2, 5, 5)
    assert contract.tensor_shapes["U_cycle"] == (2500, 2)
    assert contract.tensor_shapes["PredRelation"] == (2490, 10, 2)
    assert contract.tensor_shapes["relation_balance_weights"] == (2490, 10, 2)
    assert contract.tensor_shapes["final_predictions"] == (2500,)


def test_v2_action_order_and_hashes_exact():
    actions = build_action_space(2)
    assert actions.S == 2
    assert actions.generator_subsets == ((0,), (1,))
    assert actions.verifier_subsets == ((1,), (0,))
    assert (
        historical_action_logical_sha256(actions.generator_subsets)
        == EXPECTED_GENERATOR_SHA
    )
    assert (
        historical_action_logical_sha256(actions.verifier_subsets)
        == EXPECTED_VERIFIER_SHA
    )


def test_v2_weak_quality_mask_exact_deterministic_contract():
    first, first_audit = generate_half_corruption_mask(2500, 2, 20)
    second, second_audit = generate_half_corruption_mask(2500, 2, 20)
    assert first.shape == (2500, 2)
    assert np.array_equal(first, second)
    assert np.array_equal(first.sum(axis=1), np.ones(2500, dtype=np.int64))
    assert int(first.sum()) == 2500
    assert first.size == 5000
    assert int(first.sum(axis=0).max()) - int(first.sum(axis=0).min()) <= 1
    assert first_audit["mask_sha256"] == second_audit["mask_sha256"]


def test_bdgp_sparse_hash_ranking_selects_exactly_two_per_class():
    labels = np.repeat(np.arange(5, dtype=np.int64), 500)
    split = materialize_hash_ranked_sparse_split(
        labels,
        dataset_name="BDGP",
        label_seed=20,
        labels_per_class=2,
    )
    assert split.labeled_ids.shape == split.labeled_targets.shape == (10,)
    assert split.unlabeled_ids.shape == (2490,)
    assert np.array_equal(
        np.bincount(split.labeled_targets, minlength=5),
        np.full(5, 2, dtype=np.int64),
    )


def test_bdgp_native_config_and_seed_distinction_exact():
    config = get_default_config("BDGP")
    assert config == BDGP_RUNTIME_SPEC.expected_config
    assert config["training"] == {
        "seed": 1,
        "batch_size": 256,
        "init_epoch": 200,
        "T_1": 2,
        "T_2": 100,
        "epoch": 1000,
        "lr": 0.0001,
        "lambda1": 10,
    }
    assert BDGP_RUNTIME_SPEC.native_config_seed == 1
    assert BDGP_RUNTIME_SPEC.training_seed == 20
    assert BDGP_RUNTIME_SPEC.native_lambda1 == 10


def test_final_core_phase_b_uses_native_lambda1_and_phase_a_does_not():
    source = inspect.getsource(run_final_core)
    loop = source.split("for epoch in range(20):", 1)[1]
    phase_a, phase_b = loop.split("if epoch % native_refresh_interval == 0:", 1)
    assert "relation_semantic_loss(" in phase_a
    assert "native_lambda1" not in phase_a
    assert "native_lambda1 * clu" in phase_b
    assert "relation_semantic_loss(" not in phase_b
    assert '"phase_A_native_lambda1_used": False' in source
    assert '"phase_B_native_lambda1_used": True' in source


def test_generic_relation_shapes_for_bdgp_contract():
    y_gen = np.tile(np.arange(2, dtype=np.int64), (2500, 1))
    labeled_ids = np.arange(10, dtype=np.int64)
    labeled_targets = np.repeat(np.arange(5, dtype=np.int64), 2)
    unlabeled_ids = np.arange(10, 2500, dtype=np.int64)
    relation = build_relation_semantics(
        y_gen,
        labeled_ids,
        labeled_targets,
        unlabeled_ids,
        class_count=5,
        labels_per_class=2,
    )
    assert relation["PredRelation_true"].shape == (2490, 10, 2)
    assert relation["relation_balance_weights_true"].shape == (2490, 10, 2)


def test_training_runner_has_no_gt_or_dataset_loader():
    module_source = inspect.getsource(
        __import__(
            "experiments.generic_contract.run_g0_b1_bdgp_structural_pilot",
            fromlist=["unused"],
        )
    )
    combined = inspect.getsource(bdgp_runner_main) + module_source
    assert "datasets.load_data" not in combined
    assert "load_data(" not in combined
    assert "scipy.io.loadmat" not in combined
    assert "loadmat(" not in combined


def test_feature_and_pre_gt_schemas_exclude_full_gt():
    assert BDGP_FEATURE_FIELDS == ("X1", "X2", "sample_ids")
    assert not {"X3", "Y", "GT", "gt", "labels"}.intersection(BDGP_FEATURE_FIELDS)
    lowered = {name.lower() for name in PRE_GT_FIELDS}
    assert not {"y", "gt", "labels", "metrics", "acc", "nmi", "ari"}.intersection(
        lowered
    )
    assert tuple(PRE_GT_FIELDS) == (
        "sample_ids",
        "labeled_ids",
        "unlabeled_ids",
        "final_predictions",
        "q_local",
        "q_aligned",
        "M_v",
        "U_cycle",
        "PredRelation_true",
        "relation_balance_weights_true",
        "generator_membership",
        "verifier_membership",
    )


def test_msrc_default_runtime_boundary_is_backward_compatible():
    assert FEATURE_FIELDS == ("X1", "X2", "X3", "X4", "X5", "sample_ids")
    assert MSRC_RUNTIME_SPEC.native_lambda1 == 0.01
    assert (
        MSRC_RUNTIME_SPEC.N,
        MSRC_RUNTIME_SPEC.V,
        MSRC_RUNTIME_SPEC.K,
        MSRC_RUNTIME_SPEC.L,
        MSRC_RUNTIME_SPEC.N_u,
        MSRC_RUNTIME_SPEC.S,
    ) == (210, 5, 7, 14, 196, 20)
    assert (
        inspect.signature(validate_materialized_inputs)
        .parameters["runtime_spec"]
        .default
        is MSRC_RUNTIME_SPEC
    )


def test_bdgp_stage_and_artifact_names_are_isolated():
    spec = BDGP_RUNTIME_SPEC
    assert spec.stage_name == "G0-B1"
    assert spec.gate_field == "Gate7_A_through_O_pass"
    assert spec.feature_artifact_name == "bdgp_trainable_features.npz"
    assert spec.split_artifact_name == "bdgp_sparse_split.npz"
    assert spec.output_artifact_name == "bdgp_structural_pre_gt_artifact.npz"
    assert spec.output_audit_name == "bdgp_structural_pre_gt_audit.json"
    assert spec.output_seal_name == "bdgp_structural_pre_gt_seal.json"
