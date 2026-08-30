"""Regression tests for VSA-B multi-seed provenance parameterization."""

import copy
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.cyclic_utility import (
    c1_frozen_pseudo_supervision as c1,
)
from experiments.cyclic_utility import (
    summarize_vsa_a0_decoupled_action as summarize,
)
from experiments.cyclic_utility import (
    train_vsa_a0_decoupled_action as train,
)
from experiments.cyclic_utility import (
    vsa_decoupled_action_protocol as vsa,
)
from experiments.e1_pairwise_utility import (
    train_e1_pairwise_utility as e1_train,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_E1 = {
    20: c1.EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256,
    30: "606d94ee66396505ba0b0bd446420774e68e5763d0f2cd6a510e3188614d997a",
    50: "41c98952c143c149d639451a16a5902fd25281013a6a937559f9835365ed1ca5",
}
EXPECTED_C0 = {
    20: c1.EXPECTED_C0_ARTIFACT_SHA256,
    30: "0b7f878b7a1902a82f6c088a5e4d764d796e57add5b050e3ade5268c75d921f4",
    50: "f8b7dfd89ae0f00f1e35c040e2d896e47e430146fe1158850b8e4aa0f85c2bce",
}
EXPECTED_M0 = {
    20: c1.EXPECTED_M0_LOGICAL_SHA256,
    30: "62bd52eccc44b40aa1e77efa9d91cff08afa3a771da8873d4297de06ade177a5",
    50: "9db02dd48005726c923bb180e0fea388fe4884d0ac1c11f18199b9f30013e9da",
}


def _e1_dir(seed):
    return ROOT / "outputs/e1_pairwise_utility" / (
        "lwc_100ep_seed" + str(seed)
    )


def _c0_dir(seed):
    return ROOT / "outputs/cyclic_utility" / (
        "c0_complementary_semantic_verification_seed" + str(seed)
    )


@pytest.fixture(scope="module")
def upstream():
    views, sample_ids, feature = e1_train.load_frozen_feature_artifact(
        train.DEFAULT_FEATURE_PATH, train.DEFAULT_FEATURE_AUDIT_PATH
    )
    fixed = train.validate_fixed_feature_realization(
        train.DEFAULT_FEATURE_PATH,
        train.DEFAULT_FEATURE_AUDIT_PATH,
        feature,
    )
    records = {}
    for seed in vsa.TRAINING_SEED_CHOICES:
        e1_dir = _e1_dir(seed)
        c0_dir = _c0_dir(seed)
        model, _, e1_provenance = (
            train.load_lineage_aware_e1_lwc_model(
                e1_dir / "models",
                e1_dir / "e1_audit.json",
                torch.device("cpu"),
                seed,
            )
        )
        arrays, c0_provenance = (
            train.load_lineage_aware_c0_artifact(
                c0_dir / "c0_predictions_and_scores.npz",
                c0_dir / "c0_prediction_seal.json",
                seed,
                e1_provenance,
                feature,
            )
        )
        M0, M0_audit = train.derive_seed_specific_M0(
            model,
            views,
            arrays["native_global_cluster"],
            torch.device("cpu"),
            seed,
            c0_provenance,
        )
        records[seed] = {
            "e1": e1_provenance,
            "c0": c0_provenance,
            "arrays": arrays,
            "M0": M0,
            "M0_audit": M0_audit,
        }
    return {
        "views": views,
        "sample_ids": sample_ids,
        "feature": feature,
        "fixed": fixed,
        "records": records,
    }


@pytest.fixture(scope="module")
def seed20_bundles():
    return {
        arm: summarize.load_arm_bundle(
            summarize.DEFAULT_INPUT_DIR, arm, expected_seed=20
        )
        for arm in vsa.FORMAL_ARMS
    }


def _bundles_for_seed(seed, seed20_bundles, upstream):
    bundles = copy.deepcopy(seed20_bundles)
    if seed == 20:
        return bundles
    lineage = upstream["records"][seed]
    e1 = lineage["e1"]
    c0_provenance = lineage["c0"]
    M0_audit = lineage["M0_audit"]
    orders = vsa.precompute_epoch_orders(
        vsa.FORMAL_EPOCHS,
        np.arange(vsa.SAMPLE_NUM, dtype=np.int64),
        seed=seed,
    )
    for arm in vsa.FORMAL_ARMS:
        bundle = bundles[arm]
        audit = bundle["train_audit"]
        bundle["training_seed"] = seed
        audit["seed"] = seed
        audit["training_seed"] = seed
        audit["weak_quality_condition_fixed"] = (
            train.WEAK_QUALITY_CONDITION
        )
        audit["feature_realization_unchanged"] = True
        audit["requested_seed_lineage_match_pass"] = True
        audit["C0_E1_lineage_match_pass"] = True
        audit["initial_model_hash"] = copy.deepcopy(
            e1["actual_loaded_model_hash"]
        )
        audit["initial_model_aggregate_sha256"] = e1[
            "actual_loaded_aggregate_sha256"
        ]
        checkpoint = audit["checkpoint_provenance"]
        checkpoint.update(copy.deepcopy(e1))
        audit["E1_audit_path"] = e1["source_audit_path"]
        audit["E1_actual_loaded_aggregate_hash"] = e1[
            "actual_loaded_aggregate_sha256"
        ]
        audit["E1_expected_from_own_audit_aggregate_hash"] = e1[
            "expected_aggregate_sha256_from_own_audit"
        ]
        audit["C0_artifact_provenance"] = copy.deepcopy(
            c0_provenance
        )
        audit["M0_audit"] = copy.deepcopy(M0_audit)
        audit["native_order_sha256_per_epoch"] = copy.deepcopy(
            orders["native_order_sha256_per_epoch"]
        )
        audit["semantic_order_sha256_per_epoch"] = copy.deepcopy(
            orders["semantic_order_sha256_per_epoch"]
        )
        if arm in vsa.PSEUDO_ARMS:
            bundle["semantic_target_audit"]["logical_sha256"][
                "y_gen"
            ] = c0_provenance["array_logical_sha256"]["y_gen"]
    return bundles


def _expect_summary_rejection(bundles, seed):
    with pytest.raises(RuntimeError):
        summarize.validate_formal_bundles(bundles, expected_seed=seed)


def test_default_seed20_backward_compatibility():
    assert vsa.SEED == 20
    assert train.parse_args(["--arm", "BASE"]).seed == 20


@pytest.mark.parametrize("seed", (20, 30, 50))
def test_accepts_supported_training_seeds(seed):
    assert vsa.validate_training_seed(seed) == seed
    assert train.parse_args(["--arm", "BASE", "--seed", str(seed)]).seed == seed


@pytest.mark.parametrize("seed", (-1, 0, 21, 40, 51))
def test_rejects_unsupported_training_seed(seed):
    with pytest.raises(ValueError):
        vsa.validate_training_seed(seed)


def test_seed20_e0_identity_contract_unchanged():
    result = train.verify_saved_e0_identity()
    assert result["VSA_E0_IDENTITY_PASS"] is True


def test_e0_rejects_non_seed20_before_loading(tmp_path):
    with pytest.raises(RuntimeError, match="E0 exact identity"):
        train.run_arm(
            "BASE", run_kind="E0", seed=30,
            output_dir=tmp_path / "unused", device="cpu"
        )


def test_seed20_historical_c0_hashes_accepted(upstream):
    c0_provenance = upstream["records"][20]["c0"]
    assert c0_provenance["artifact_file_sha256"] == EXPECTED_C0[20]
    assert c0_provenance["seed20_historical_hashes_pass"] is True


def test_seed20_historical_e1_hash_accepted(upstream):
    e1 = upstream["records"][20]["e1"]
    assert e1["actual_loaded_aggregate_sha256"] == EXPECTED_E1[20]
    assert e1["seed20_historical_hash_pass"] is True


@pytest.mark.parametrize("seed", (30, 50))
def test_multiseed_e1_own_audit_model_hash_accepted(upstream, seed):
    e1 = upstream["records"][seed]["e1"]
    assert e1["actual_loaded_aggregate_sha256"] == EXPECTED_E1[seed]
    assert e1["model_matches_own_audit_pass"] is True


@pytest.mark.parametrize("requested,audit_seed", ((30, 20), (50, 30)))
def test_cross_seed_e1_mismatch_hard_fails(requested, audit_seed):
    directory = _e1_dir(audit_seed)
    with pytest.raises(RuntimeError, match="training seed mismatch"):
        train.load_lineage_aware_e1_lwc_model(
            directory / "models",
            directory / "e1_audit.json",
            torch.device("cpu"),
            requested,
        )


@pytest.mark.parametrize("seed", (30, 50))
def test_multiseed_c0_own_seal_accepted(upstream, seed):
    provenance = upstream["records"][seed]["c0"]
    assert provenance["artifact_file_sha256"] == EXPECTED_C0[seed]
    assert provenance["own_seal_audit_validation_pass"] is True


@pytest.mark.parametrize("seed", (30, 50))
def test_multiseed_c0_does_not_require_seed20_tensor_hash(upstream, seed):
    hashes = upstream["records"][seed]["c0"]["array_logical_sha256"]
    assert hashes["y_gen"] != c1.EXPECTED_C0_ARRAY_SHA256["y_gen"]
    assert hashes["U_cycle"] != c1.EXPECTED_C0_ARRAY_SHA256["U_cycle"]


@pytest.mark.parametrize("requested,c0_seed", ((30, 20), (50, 30)))
def test_cross_seed_c0_mismatch_hard_fails(upstream, requested, c0_seed):
    c0_dir = _c0_dir(c0_seed)
    with pytest.raises(RuntimeError, match="C0 and E1 model lineage mismatch"):
        train.load_lineage_aware_c0_artifact(
            c0_dir / "c0_predictions_and_scores.npz",
            c0_dir / "c0_prediction_seal.json",
            requested,
            upstream["records"][requested]["e1"],
            upstream["feature"],
        )


@pytest.mark.parametrize("c0_seed,e1_seed", ((30, 50), (50, 30)))
def test_c0_e1_lineage_mismatch_hard_fails(upstream, c0_seed, e1_seed):
    c0_dir = _c0_dir(c0_seed)
    with pytest.raises(RuntimeError, match="C0 and E1 model lineage mismatch"):
        train.load_lineage_aware_c0_artifact(
            c0_dir / "c0_predictions_and_scores.npz",
            c0_dir / "c0_prediction_seal.json",
            e1_seed,
            upstream["records"][e1_seed]["e1"],
            upstream["feature"],
        )


def test_seed20_M0_historical_hash_unchanged(upstream):
    audit = upstream["records"][20]["M0_audit"]
    assert audit["logical_sha256"] == c1.EXPECTED_M0_LOGICAL_SHA256


@pytest.mark.parametrize("seed", (30, 50))
def test_multiseed_M0_may_differ_from_seed20(upstream, seed):
    audit = upstream["records"][seed]["M0_audit"]
    assert audit["logical_sha256"] == EXPECTED_M0[seed]
    assert audit["logical_sha256"] != c1.EXPECTED_M0_LOGICAL_SHA256


@pytest.mark.parametrize("seed", (20, 30, 50))
def test_seed_specific_M0_is_full_detached_permutation(upstream, seed):
    M0 = upstream["records"][seed]["M0"]
    audit = upstream["records"][seed]["M0_audit"]
    assert tuple(M0.shape) == (6, 7, 7)
    assert torch.equal(M0.sum(dim=1), torch.ones(6, 7))
    assert torch.equal(M0.sum(dim=2), torch.ones(6, 7))
    assert not M0.requires_grad
    assert audit["training_M0_is_exact_01_cast_of_int64_pass"] is True


@pytest.mark.parametrize("seed", (20, 30, 50))
def test_seed_specific_M0_conserves_C0_alignment(upstream, seed):
    record = upstream["records"][seed]
    assert record["M0_audit"]["logical_sha256"] == record["c0"][
        "M0_logical_sha256_from_C0_seal"
    ]
    assert record["M0_audit"]["alignment_conservation_pass"] is True


@pytest.mark.parametrize("seed", (20, 30, 50))
def test_training_seed_controls_both_precomputed_orders(seed):
    orders = vsa.precompute_epoch_orders(
        2, np.arange(vsa.SAMPLE_NUM, dtype=np.int64), seed=seed
    )
    assert orders["seed"] == seed
    assert orders["semantic_order_sha256_per_epoch"] == orders[
        "native_order_sha256_per_epoch"
    ]


def test_training_seeds_produce_distinct_sample_orders():
    ids = np.arange(vsa.SAMPLE_NUM, dtype=np.int64)
    hashes = {
        tuple(vsa.precompute_epoch_orders(1, ids, seed=seed)[
            "native_order_sha256_per_epoch"
        ])
        for seed in vsa.TRAINING_SEED_CHOICES
    }
    assert len(hashes) == 3


@pytest.mark.parametrize("seed", (20, 30, 50))
def test_fixed_e0_noisy_feature_stays_seed20(upstream, seed):
    e1 = upstream["records"][seed]["e1"]
    assert e1["weak_quality_condition_fixed"] == "snr2p5_k3_seed20"
    assert upstream["fixed"]["feature_realization_unchanged_pass"] is True


def test_training_seed_cannot_substitute_feature_realization(upstream, tmp_path):
    with pytest.raises(RuntimeError, match="must remain seed20"):
        train.validate_fixed_feature_realization(
            tmp_path / "different.npz",
            train.DEFAULT_FEATURE_AUDIT_PATH,
            upstream["feature"],
        )


def test_c0_loader_exposes_no_R_sparse_labels_or_corruption(upstream):
    for seed in vsa.TRAINING_SEED_CHOICES:
        provenance = upstream["records"][seed]["c0"]
        assert provenance["R_loaded"] is False
        assert provenance["sparse_labels_loaded"] is False
        assert provenance["corruption_mask_loaded"] is False
        assert provenance["GT_derived_correctness_loaded"] is False


def test_runner_honors_explicit_c0_and_e1_paths():
    source = inspect.getsource(train.run_arm)
    assert "load_lineage_aware_c0_artifact(\n        c0_artifact_path" in source
    assert "c0_seal_path" in source
    assert "load_lineage_aware_e1_lwc_model(\n            model_dir" in source
    assert "model_audit_path" in source


def test_vsa_semantic_formula_remains_frozen_C1_builder():
    assert vsa.SEMANTIC_OBJECTIVE == (
        "mean_{i,v}(a_i * CE(T_local[i,v], q_local[i,v]))"
    )
    source = inspect.getsource(vsa.build_frozen_semantic_target)
    assert "c1.build_frozen_pseudo_target" in source


def test_semantic_and_native_phases_remain_separated():
    source = inspect.getsource(train.train_vsa_arm)
    assert source.index("_semantic_phase(") < source.index("_native_phase(")
    assert "semantic_before_native_every_epoch" in source


def test_no_additive_native_pseudo_loss():
    source = inspect.getsource(train.train_vsa_arm)
    assert "combine_native_and_pseudo_loss" not in source
    assert "native_e1_lwc_loss +" not in source


def test_optimizer_states_remain_separate():
    source = inspect.getsource(vsa.build_decoupled_optimizers)
    assert "semantic_optimizers =" in source
    assert "native_optimizers =" in source
    assert "state_mappings_separate" in source


@pytest.mark.parametrize("seed", (20, 30, 50))
def test_per_seed_summarizer_validation_works(
    seed, seed20_bundles, upstream
):
    bundles = _bundles_for_seed(seed, seed20_bundles, upstream)
    checks = summarize.validate_formal_bundles(
        bundles, expected_seed=seed
    )
    assert checks["training_seed"] == seed
    assert checks["same_E1_lineage_all_arms"] is True
    assert checks["same_frozen_C0_all_arms"] is True


def test_same_seed_five_arm_provenance_required(seed20_bundles, upstream):
    bundles = _bundles_for_seed(30, seed20_bundles, upstream)
    bundles["CONF"]["training_seed"] = 50
    with pytest.raises(RuntimeError, match="mix training seeds"):
        summarize.validate_formal_bundles(bundles)


@pytest.mark.parametrize(
    "mutation",
    (
        "E1_audit",
        "C0_artifact",
        "C0_seal",
        "y_gen",
        "M0",
        "initialization",
        "native_order",
        "semantic_order",
    ),
)
def test_summarizer_rejects_cross_arm_provenance_drift(
    mutation, seed20_bundles, upstream
):
    bundles = _bundles_for_seed(30, seed20_bundles, upstream)
    audit = bundles["CONF"]["train_audit"]
    if mutation == "E1_audit":
        audit["E1_audit_path"] = "wrong/e1_audit.json"
    elif mutation == "C0_artifact":
        audit["C0_artifact_provenance"]["artifact_file_sha256"] = "bad"
    elif mutation == "C0_seal":
        audit["C0_artifact_provenance"]["seal_file_sha256"] = "bad"
    elif mutation == "y_gen":
        bundles["CONF"]["semantic_target_audit"]["logical_sha256"][
            "y_gen"
        ] = "bad"
    elif mutation == "M0":
        audit["M0_audit"]["logical_sha256"] = "bad"
    elif mutation == "initialization":
        audit["initial_model_hash"]["aggregate"] = "bad"
    elif mutation == "native_order":
        audit["native_order_sha256_per_epoch"][0] = "bad"
    elif mutation == "semantic_order":
        audit["semantic_order_sha256_per_epoch"][0] = "bad"
    _expect_summary_rejection(bundles, 30)


def test_seed20_summarizer_historical_checks_remain_exact(seed20_bundles):
    checks = summarize.validate_formal_bundles(
        copy.deepcopy(seed20_bundles), expected_seed=20
    )
    assert checks["seed20_historical_initialization_pass"] is True
    assert checks["seed20_historical_C0_pass"] is True
    assert checks["seed20_historical_M0_pass"] is True
    assert checks["seed20_historical_y_gen_pass"] is True


def test_requested_seed_is_forwarded_to_native_target_refresh():
    source = inspect.getsource(train.train_vsa_arm)
    assert "training_seed = vsa.validate_training_seed(training_seed)" in source
    assert source.count("training_seed=training_seed") == 2


def test_train_audit_declares_required_lineage_metadata():
    source = inspect.getsource(train.run_arm)
    for field in (
        "training_seed",
        "weak_quality_condition_fixed",
        "E1_audit_path",
        "E1_training_seed",
        "E1_actual_loaded_aggregate_hash",
        "E1_expected_from_own_audit_aggregate_hash",
        "C0_artifact_path",
        "C0_seal_path",
        "C0_artifact_file_hash",
        "C0_source_model_provenance_hash",
        "C0_E1_lineage_match_pass",
        "requested_seed_lineage_match_pass",
        "M0_seed_specific_logical_hash",
        "feature_realization_unchanged",
        "GT_loaded_before_prediction_seal",
    ):
        assert '"' + field + '"' in source


def test_no_vsa_b_aggregate_gate_was_added():
    source = inspect.getsource(summarize)
    assert "VSA_B_AGGREGATE" not in source
    assert "build_vsa_a0_decision" in source


def test_python38_compatibility_does_not_use_ast_unparse():
    for module in (vsa, train, summarize):
        assert "ast.unparse" not in inspect.getsource(module)
