"""Tests for the B6-WQ0-R1 cross-path canonical replay audit."""

import hashlib
import inspect
import json

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

import experiments.b6_weak_quality.audit_b6_wq0_cross_path_replay as cross_path
import experiments.b6_weak_quality.evaluate_b6_wq0_utility_semantic_admission as b6

ORIGINAL_KMEANS = b6._kmeans
ORIGINAL_PREPARE_SEED = b6.prepare_seed
ORIGINAL_RUN_BASE = b6.run_base


@pytest.fixture(scope="module")
def seed20_cross_path(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("b6_cross_path")
    return cross_path.run_cross_path_audit(
        seed=20,
        output_dir=output_dir,
    )


def _run_real_provenance_path(seed):
    manifest_before = b6.CANONICAL_MANIFEST_PATH.read_bytes()
    state = cross_path.run_provenance_path(
        seed=seed,
        ambient_seed=seed + cross_path.AMBIENT_SEED_PATH_A_OFFSET,
    )
    manifest_after = b6.CANONICAL_MANIFEST_PATH.read_bytes()
    return {
        "state": state,
        "manifest_before": manifest_before,
        "manifest_after": manifest_after,
    }


@pytest.fixture(scope="module")
def real_provenance_seed20():
    return _run_real_provenance_path(20)


@pytest.fixture(scope="module")
def real_provenance_seed30():
    return _run_real_provenance_path(30)


@pytest.fixture(scope="module")
def real_provenance_seed50():
    return _run_real_provenance_path(50)


@pytest.fixture(scope="module")
def seed30_cross_path(tmp_path_factory):
    return cross_path.run_cross_path_audit(
        seed=30,
        output_dir=tmp_path_factory.mktemp("b6_cross_path_seed30"),
    )


@pytest.fixture(scope="module")
def seed50_cross_path(tmp_path_factory):
    return cross_path.run_cross_path_audit(
        seed=50,
        output_dir=tmp_path_factory.mktemp("b6_cross_path_seed50"),
    )


@pytest.mark.parametrize("seed", (20, 30, 50))
def test_supported_seed_is_accepted_by_cli_and_validation(seed):
    args = cross_path.parse_args(["--seed", str(seed)])
    assert args.seed == seed
    assert cross_path._validate_seed(args.seed) == seed


def test_unsupported_seed_is_rejected():
    with pytest.raises(RuntimeError, match="unsupported R1 model seed"):
        cross_path._validate_seed(40)


def _assert_seed_specific_registered_provenance(seed):
    record, source_path = cross_path._load_registered_record(seed)
    expected = b6.SEED_PROVENANCE[seed]
    assert record["model_seed"] == seed
    assert record["backbone_dir"] == expected["backbone_dir"]
    assert record["backbone_hash"] == expected["backbone_hash"]
    assert record["normalized_z_sha256"] == expected["z_hash"]
    assert source_path.name == "seed" + str(seed) + "_provenance.json"


def test_seed30_uses_seed30_backbone_and_registered_provenance():
    _assert_seed_specific_registered_provenance(30)


def test_seed50_uses_seed50_backbone_and_registered_provenance():
    _assert_seed_specific_registered_provenance(50)


def test_reference_model_seed_mismatch_is_rejected(tmp_path):
    reference_path = tmp_path / "reference.npz"
    np.savez(
        reference_path,
        critical_metadata_json=np.asarray(json.dumps({"model_seed": 20})),
    )
    with pytest.raises(
        RuntimeError,
        match="reference model_seed does not match requested seed",
    ):
        cross_path.compare_reference(
            reference_path,
            {"prepared": {"seed": 30}},
        )


def _assert_real_provenance_path_completed(audit, seed):
    state = audit["state"]
    record = state["provenance_record"]
    assert state["prepared"]["seed"] == seed
    assert record["model_seed"] == seed
    assert all(np.isfinite(value) for value in state["metrics"].values())
    assert len(
        state["prepared"]["fusion_inputs"]["raw_z_views"]
    ) == b6.VIEW_NUM
    assert not record["optimizer_created"]
    assert not record["backward_performed"]
    assert not record["parameter_updates"]
    assert state["backbone_hash_before"] == state["backbone_hash_after"]
    assert state["backbone_unchanged"]
    for autoencoder, checkpoint_path in zip(
        state["prepared"]["models"].autoencoders,
        state["prepared"]["checkpoint_paths"],
    ):
        checkpoint_cluster = b6._load_checkpoint(
            checkpoint_path
        )["_cluster_layer"]
        assert np.array_equal(
            autoencoder._cluster_layer.detach().cpu().numpy(),
            checkpoint_cluster.detach().cpu().numpy(),
        )
    assert audit["manifest_before"] == audit["manifest_after"]
    assert b6._kmeans is ORIGINAL_KMEANS
    assert b6.prepare_seed is ORIGINAL_PREPARE_SEED
    assert b6.run_base is ORIGINAL_RUN_BASE


def test_real_provenance_path_seed20_completes(real_provenance_seed20):
    _assert_real_provenance_path_completed(real_provenance_seed20, 20)


def test_old_r1_snapshot_is_preserved_as_diagnostic(seed20_cross_path):
    snapshot_path = cross_path._resolve(
        cross_path.OLD_R1_SNAPSHOT_ARTIFACT
    )
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    diagnostic = seed20_cross_path["old_r1_snapshot_diagnostic"]
    assert hashlib.sha256(snapshot_bytes).hexdigest() == (
        "844afa7000ccae05b79fbaa050572a02ef9095f6be7512ec00a7e90b726bfb0a"
    )
    assert (
        diagnostic["old_r1_snapshot_reference_kind"]
        == "historical-diagnostic-only"
    )
    assert not diagnostic["old_r1_snapshot_is_scientific_gate"]
    old_final = snapshot["live_cross_path_comparison"]["final"]
    assert diagnostic["old_r1_expected_fused_hash"] == (
        old_final["final_fused"]["hash_a"]
    )
    assert diagnostic["old_r1_expected_prediction_hash"] == (
        old_final["final_prediction"]["raw_hash_a"]
    )
    assert diagnostic["old_r1_expected_metrics"] == (
        old_final["metrics_path_a"]
    )
    assert snapshot_path.read_bytes() == snapshot_bytes


def test_current_seed20_live_paths_match_registered_canonical(
    seed20_cross_path,
):
    assert seed20_cross_path["cross_path_fused_hash_exact"]
    assert seed20_cross_path["cross_path_prediction_array_exact"]
    assert seed20_cross_path["cross_path_metrics_exact"]
    assert seed20_cross_path[
        "B6_WQ0_PATH_A_REGISTERED_CANONICAL_MATCH"
    ]
    assert seed20_cross_path[
        "B6_WQ0_PATH_B_REGISTERED_CANONICAL_MATCH"
    ]
    assert seed20_cross_path[
        "B6_WQ0_LIVE_PATHS_REGISTERED_CANONICAL_PASS"
    ]
    assert seed20_cross_path[
        "registered_vs_live_first_divergence_stage"
    ] is None
    assert seed20_cross_path[
        "cross_path_first_divergence_stage"
    ] is None


def test_real_provenance_path_seed30_completes(real_provenance_seed30):
    _assert_real_provenance_path_completed(real_provenance_seed30, 30)


def test_real_provenance_path_seed50_completes(real_provenance_seed50):
    _assert_real_provenance_path_completed(real_provenance_seed50, 50)


def _assert_real_cross_path_smoke(result, seed):
    assert result["model_seed"] == seed
    for key in (
        "cross_path_fused_hash_exact",
        "cross_path_prediction_array_exact",
        "cross_path_metrics_exact",
        "registered_vs_live_first_divergence_stage",
    ):
        assert key in result
    assert not result["optimizer_created"]
    assert not result["backward_performed"]
    assert not result["parameter_updates"]
    assert not result["formal_manifest_modified"]


def test_real_cross_path_seed30_completes(seed30_cross_path):
    _assert_real_cross_path_smoke(seed30_cross_path, 30)


def test_real_cross_path_seed50_completes(seed50_cross_path):
    _assert_real_cross_path_smoke(seed50_cross_path, 50)


def test_provenance_path_uses_python38_compatible_context_managers():
    source = inspect.getsource(cross_path.run_provenance_path)
    assert "with ExitStack() as stack:" in source
    assert source.count("stack.enter_context(") == 3
    assert "with (" not in source


def test_cross_path_audit_creates_no_optimizer():
    source = inspect.getsource(cross_path)
    assert ".train(" not in source
    assert "torch.optim" not in source


def test_cross_path_audit_performs_no_backward():
    source = inspect.getsource(cross_path)
    assert ".backward(" not in source


def test_cross_path_audit_performs_no_parameter_update():
    source = inspect.getsource(cross_path)
    assert "optimizer.step(" not in source


def test_cross_path_reports_no_training_actions(seed20_cross_path):
    assert not seed20_cross_path["optimizer_created"]
    assert not seed20_cross_path["backward_performed"]
    assert not seed20_cross_path["parameter_updates"]


def test_two_real_paths_use_independent_models_and_preserve_backbone(
    seed20_cross_path,
):
    assert seed20_cross_path["independent_model_objects"]
    assert seed20_cross_path["path_a"]["backbone_unchanged"]
    assert seed20_cross_path["path_b"]["backbone_unchanged"]
    assert seed20_cross_path["backbone_unchanged"]


def test_compare_arrays_detects_synthetic_float_perturbation():
    original = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    perturbed = original.copy()
    perturbed[7] = np.nextafter(
        perturbed[7], np.float32(np.inf), dtype=np.float32
    )
    comparison = cross_path.compare_arrays(original, perturbed)
    assert not comparison["hash_exact"]
    assert not comparison["array_exact"]
    assert comparison["max_abs_error"] > 0.0
    assert comparison["mean_abs_error"] > 0.0


def test_partition_equivalence_accepts_pure_label_permutation():
    prediction_a = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    prediction_b = np.asarray([2, 2, 0, 0, 1, 1], dtype=np.int64)
    assert cross_path.partition_equivalent(prediction_a, prediction_b)
    assert adjusted_rand_score(prediction_a, prediction_b) == 1.0


def test_raw_prediction_hash_still_detects_label_permutation():
    prediction_a = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    prediction_b = np.asarray([2, 2, 0, 0, 1, 1], dtype=np.int64)
    comparison = cross_path.compare_partitions(
        prediction_a, prediction_b
    )
    assert not comparison["raw_hash_exact"]
    assert not comparison["array_exact"]
    assert comparison["ari"] == 1.0
    assert comparison["partition_equivalent"]


def test_partition_equivalence_rejects_split_or_merge():
    prediction_a = np.asarray([0, 0, 1, 1], dtype=np.int64)
    prediction_b = np.asarray([0, 1, 1, 1], dtype=np.int64)
    assert not cross_path.partition_equivalent(
        prediction_a, prediction_b
    )


def test_first_divergence_locator_returns_first_failed_layer():
    layers = [
        {"stage": "corrupted_views", "equal": True},
        {"stage": "raw_z_view0", "equal": True},
        {"stage": "scaled_z_view1", "equal": False},
        {"stage": "prediction_update0", "equal": False},
    ]
    assert (
        cross_path.first_divergence_from_layers(layers)
        == "scaled_z_view1"
    )
    assert cross_path.first_divergence_from_layers(
        [{"stage": "raw_z_view0", "equal": True}]
    ) == "none"


def test_seed20_reference_divergence_scopes_are_independent(
    seed20_cross_path,
):
    assert seed20_cross_path[
        "live_cross_path_comparison"
    ]["first_divergence_stage"] == "none"
    assert seed20_cross_path[
        "cross_path_first_divergence_stage"
    ] is None
    assert seed20_cross_path[
        "registered_vs_live_first_divergence_stage"
    ] is None
    if seed20_cross_path["B6_WQ0_OLD_R1_SNAPSHOT_MATCH"]:
        assert seed20_cross_path[
            "old_r1_vs_live_first_divergence_stage"
        ] is None
    else:
        assert seed20_cross_path[
            "old_r1_vs_live_first_divergence_stage"
        ] is not None
    assert seed20_cross_path["cross_path_fused_hash_exact"]
    assert seed20_cross_path["cross_path_fused_max_abs_error"] == 0.0
    assert seed20_cross_path["cross_path_prediction_hash_exact"]
    assert seed20_cross_path["cross_path_prediction_array_exact"]
    assert seed20_cross_path[
        "cross_path_prediction_partition_equivalent"
    ]
    assert seed20_cross_path["cross_path_prediction_ARI"] == 1.0
    assert seed20_cross_path["cross_path_metrics_exact"]


def test_ambient_rng_probe_and_special_mismatch_flags(seed20_cross_path):
    assert not seed20_cross_path["B6_WQ0_AMBIENT_RNG_DEPENDENCY"]
    assert not seed20_cross_path[
        "B6_WQ0_FUSED_HASH_BYTEWISE_ONLY_MISMATCH"
    ]
    assert not seed20_cross_path[
        "B6_WQ0_PREDICTION_HASH_LABEL_PERMUTATION_ONLY"
    ]
    path_a_runtime = seed20_cross_path["path_a"]["runtime_metadata"]
    path_b_runtime = seed20_cross_path["path_b"]["runtime_metadata"]
    assert (
        path_a_runtime["python_random_seed"]
        != path_b_runtime["python_random_seed"]
    )
    assert path_a_runtime["kmeans_random_state"] == "explicit model_seed"
    assert path_b_runtime["kmeans_n_init"] == 100
    assert (
        path_a_runtime["mini_batch_kmeans_condition"]
        == "N>10000 (false for N=210)"
    )
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "torch_num_threads",
    ):
        assert name in path_a_runtime["thread_settings"]


def test_representation_and_assignment_layers_are_audited(seed20_cross_path):
    comparison = seed20_cross_path["live_cross_path_comparison"]
    for name in ("raw_z", "normalized_z", "scaled_z"):
        rows = comparison["representations"][name]
        assert len(rows) == 5
        assert all(row["hash_exact"] for row in rows)
        assert all(row["max_abs_error"] == 0.0 for row in rows)
    for name in (
        "checkpoint_native",
        "kmeans_initialized",
        "temporary_q_initialized",
    ):
        rows = comparison["prefusion_assignments"][name]
        assert len(rows) == 5
        assert all(row["array_exact"] for row in rows)
        assert all(row["ari"] == 1.0 for row in rows)


def test_registered_manifest_is_not_modified(seed20_cross_path):
    assert (
        seed20_cross_path["formal_manifest_sha256_before"]
        == seed20_cross_path["formal_manifest_sha256_after"]
    )
    assert not seed20_cross_path["formal_manifest_modified"]


def test_scientific_comparator_and_gate_are_unchanged():
    comparator_source = inspect.getsource(
        b6.canonical_base_reproduction_audit
    ).encode("utf-8")
    gate_source = inspect.getsource(b6.scientific_action_gate).encode(
        "utf-8"
    )
    assert hashlib.sha256(comparator_source).hexdigest() == (
        "bf8532f9140d91d2cf9772f74e9a7cdc046a006385dcef1a5bf2c74af77eff76"
    )
    assert hashlib.sha256(gate_source).hexdigest() == (
        "8ba45a550728ed161aaf906748ee87d50d86b2c946d1add89122ad1fbc88e2a2"
    )
