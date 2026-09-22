import dataclasses
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import release_core.runtime.entrypoint as entrypoint
from release_core.runtime import ProvenanceConfig, RuntimeConfig, run_pre_gt
from release_core.training import NativeTargetState


class FakeAutoencoder:
    def __init__(self, cluster_count):
        self.cluster_count = cluster_count

    def encoder(self, values):
        return values[:, :self.cluster_count]

    def clustering(self, latent):
        return torch.softmax(latent, dim=1)

    def train(self):
        return self


class FakeModel:
    def __init__(self, view_count, cluster_count):
        self.autoencoders = [FakeAutoencoder(cluster_count) for _ in range(view_count)]
        self.n_clusters = cluster_count
        self._state = torch.arange(4, dtype=torch.float32)

    def state_dicts(self):
        return tuple({"state": self._state} for _ in self.autoencoders)


@dataclasses.dataclass(frozen=True)
class FakeTrainingAudit:
    final_prediction_refresh_executed: bool = False
    synthetic: bool = True


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _save_bound_npz(path, audit_path, **arrays):
    np.savez(path, **arrays)
    _write_json(audit_path, {"artifact_sha256": _sha256(path)})


def make_runtime_case(tmp_path, dataset):
    tmp_path.mkdir(parents=True, exist_ok=True)
    view_count = {"BDGP": 2, "MSRC-v1": 5, "Caltech-6V": 6}[dataset]
    cluster_count = {"BDGP": 5, "MSRC-v1": 7, "Caltech-6V": 7}[dataset]
    labeled_count = cluster_count * 2
    sample_count = labeled_count + 4
    sample_ids = np.arange(sample_count, dtype=np.int64)
    labeled_ids = sample_ids[:labeled_count]
    unlabeled_ids = sample_ids[labeled_count:]
    labeled_targets = np.repeat(np.arange(cluster_count, dtype=np.int64), 2)
    split_path = tmp_path / "split.npz"
    split_audit = tmp_path / "split.json"
    _save_bound_npz(
        split_path,
        split_audit,
        sample_ids=sample_ids,
        labeled_ids=labeled_ids,
        labeled_targets=labeled_targets,
        unlabeled_ids=unlabeled_ids,
    )

    feature_path = tmp_path / "features.npz"
    feature_audit = tmp_path / "features.json"
    feature_arrays = {
        "view_" + str(index): np.ascontiguousarray(
            np.arange(sample_count * cluster_count, dtype=np.float32).reshape(
                sample_count, cluster_count
            ) + index
        )
        for index in range(view_count)
    }
    feature_arrays["sample_ids"] = sample_ids
    np.savez(feature_path, **feature_arrays)
    _write_json(feature_audit, {
        "artifact_sha256": _sha256(feature_path),
        "dataset": dataset,
        "trainable_artifact_forbidden_fields_absent": True,
    })

    action_count = 3
    utility_path = tmp_path / "utility.npz"
    utility_audit = tmp_path / "utility.json"
    _save_bound_npz(
        utility_path,
        utility_audit,
        u_cycle=np.ones((unlabeled_ids.size, action_count), dtype=np.float64),
        unlabeled_ids=unlabeled_ids,
    )
    semantic_path = tmp_path / "semantic.npz"
    semantic_audit = tmp_path / "semantic.json"
    relation = np.zeros(
        (unlabeled_ids.size, labeled_ids.size, action_count), dtype=np.bool_
    )
    relation[:, ::2, :] = True
    _save_bound_npz(
        semantic_path,
        semantic_audit,
        pred_relation=relation,
        balance=np.ones(relation.shape, dtype=np.float64),
        labeled_ids=labeled_ids,
        unlabeled_ids=unlabeled_ids,
    )
    checkpoints = []
    checkpoint_hashes = []
    for index in range(view_count):
        path = tmp_path / ("checkpoint_" + str(index) + ".pt")
        path.write_bytes(("checkpoint-" + str(index)).encode("ascii"))
        checkpoints.append(path)
        checkpoint_hashes.append(_sha256(path))
    checkpoint_audit = tmp_path / "checkpoints.json"
    _write_json(checkpoint_audit, {"checkpoint_sha256": checkpoint_hashes})

    runtime = RuntimeConfig(dataset=dataset, training_seed=20, epochs=1)
    provenance = ProvenanceConfig(
        feature_artifact=feature_path,
        feature_audit=feature_audit,
        sparse_split_artifact=split_path,
        sparse_split_audit=split_audit,
        utility_artifact=utility_path,
        utility_audit=utility_audit,
        semantic_artifact=semantic_path,
        semantic_audit=semantic_audit,
        checkpoint_paths=tuple(checkpoints),
        checkpoint_audit=checkpoint_audit,
        output_root=tmp_path / "run",
    )
    return runtime, provenance, sample_count, view_count, cluster_count


def install_synthetic_execution(monkeypatch, events, view_count, cluster_count):
    model = FakeModel(view_count, cluster_count)

    def fake_load_model(runtime, provenance, contract, device):
        events.append("model_loaded")
        return model, entrypoint._model_sha256(model), tuple(
            "a" * 64 for _ in range(view_count)
        ), "b" * 64

    def fake_training(model_arg, views, sample_ids, labeled_ids, unlabeled_ids,
                      cycle, relation, balance, **kwargs):
        events.append("r5_returned")
        state = NativeTargetState(
            p_global=torch.ones(sample_ids.size, cluster_count),
            matches=torch.eye(cluster_count).repeat(view_count, 1, 1),
            view_weights=tuple(1.0 for _ in range(view_count)),
            refresh_count=1,
            last_refresh_epoch=0,
        )
        return model_arg, state, FakeTrainingAudit()

    def fake_refresh(latent_views, q_views, weights, n_clusters, random_state):
        events.append("final_refresh")
        sample_count = latent_views[0].shape[0]
        predictions = np.arange(sample_count, dtype=np.int64) % cluster_count
        matrices = np.repeat(
            np.eye(cluster_count, dtype=np.int64)[None, :, :], view_count, axis=0
        )
        return (
            np.full((sample_count, cluster_count), 1.0 / cluster_count),
            matrices,
            predictions,
            np.ones(view_count, dtype=np.float64),
            np.zeros((cluster_count, cluster_count)),
        )

    monkeypatch.setattr(entrypoint, "_load_initial_model", fake_load_model)
    monkeypatch.setattr(entrypoint, "run_alternating_training", fake_training)
    monkeypatch.setattr(entrypoint, "native_refresh_from_latents", fake_refresh)


def test_public_api_is_neutral_and_configs_are_immutable(tmp_path):
    import release_core.runtime as runtime_api

    assert set(runtime_api.__all__) == {
        "RuntimeConfig", "ProvenanceConfig", "SealedPredictionPaths", "Metrics",
        "run_pre_gt", "evaluate_postseal",
    }
    assert not any(name in runtime_api.__all__ for name in (
        "TRUE_U", "TRUE_UNIFORM", "SHUFFLE_U", "BASE", "C3", "F0", "pilot",
    ))
    config = RuntimeConfig("BDGP", 20)
    with pytest.raises(ValueError):
        RuntimeConfig("BDGP", 20, learning_rate=float("nan"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.epochs = 3
    runtime, provenance, _, _, _ = make_runtime_case(tmp_path, "BDGP")
    with pytest.raises(dataclasses.FrozenInstanceError):
        provenance.strict_replay = True
    assert runtime.epochs == 1


def test_run_pre_gt_has_no_gt_or_metric_argument():
    names = set(inspect.signature(run_pre_gt).parameters)
    names.update(inspect.signature(entrypoint.run_pre_gt).parameters)
    assert not names.intersection({"full_gt_path", "ground_truth", "labels", "metrics"})


def test_determinism_matches_frozen_runtime_contract():
    _, audit = entrypoint._configure_determinism(31, "cpu")
    first = (random_value := np.random.rand(), torch.rand(1).item())
    entrypoint._configure_determinism(31, "cpu")
    second = (np.random.rand(), torch.rand(1).item())
    assert first == second
    assert audit["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert set(audit["thread_environment"].values()) == {"1"}
    assert audit["deterministic_algorithms"] is True
    assert audit["cuda_matmul_tf32"] is False
    assert audit["cudnn_tf32"] is False
    assert audit["pythonhashseed_added"] is False
    assert random_value == second[0]


@pytest.mark.parametrize("dataset", ["BDGP", "MSRC-v1", "Caltech-6V"])
def test_synthetic_v2_v5_v6_execution_and_exact_call_order(
    tmp_path, monkeypatch, dataset
):
    runtime, provenance, sample_count, view_count, cluster_count = make_runtime_case(
        tmp_path, dataset
    )
    events = []
    install_synthetic_execution(monkeypatch, events, view_count, cluster_count)
    sealed = entrypoint.run_pre_gt(runtime, provenance)
    assert events == ["model_loaded", "r5_returned", "final_refresh"]
    assert sealed.bundle.name == "pre_gt_bundle.npz"
    assert sealed.audit.name == "pre_gt_audit.json"
    assert sealed.seal.name == "pre_gt_seal.json"
    with np.load(sealed.bundle, allow_pickle=False) as archive:
        assert tuple(archive.files) == entrypoint._BUNDLE_KEYS
        assert archive["final_predictions"].shape == (sample_count,)
        assert archive["final_predictions"].dtype == np.int64
        assert archive["q_local"].shape == (sample_count, view_count, cluster_count)
        assert archive["q_aligned"].shape == archive["q_local"].shape
        assert archive["M_v"].shape == (view_count, cluster_count, cluster_count)
        assert not any(key.lower() in {"y", "gt", "labels", "full_gt"}
                       for key in archive.files)
    audit = json.loads(sealed.audit.read_text(encoding="utf-8"))
    assert audit["r5_invocation_count"] == 1
    assert audit["r5_training_audit"]["final_prediction_refresh_executed"] is False
    assert audit["final_refresh"]["refresh_count"] == 1
    assert audit["final_refresh"]["executed_after_r5"] is True
    assert audit["gt_firewall"]["full_gt_loaded"] is False
    seal = json.loads(sealed.seal.read_text(encoding="utf-8"))
    assert seal["bundle_sha256"] == _sha256(sealed.bundle)
    assert seal["audit_sha256"] == _sha256(sealed.audit)


def test_feature_only_artifact_rejects_labels(tmp_path):
    runtime, provenance, _, _, _ = make_runtime_case(tmp_path, "BDGP")
    np.savez(
        provenance.feature_artifact,
        view_0=np.ones((4, 2), dtype=np.float32),
        view_1=np.ones((4, 2), dtype=np.float32),
        sample_ids=np.arange(4, dtype=np.int64),
        labels=np.arange(4, dtype=np.int64),
    )
    _write_json(provenance.feature_audit, {
        "artifact_sha256": _sha256(provenance.feature_artifact)
    })
    with pytest.raises(RuntimeError, match="GT field"):
        entrypoint._load_feature_only(runtime, provenance)


def test_sample_id_sparse_and_action_alignment_fail_closed(tmp_path):
    runtime, provenance, _, _, _ = make_runtime_case(tmp_path, "BDGP")
    with np.load(provenance.feature_artifact, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["sample_ids"] = arrays["sample_ids"][::-1].copy()
    np.savez(provenance.feature_artifact, **arrays)
    _write_json(provenance.feature_audit, {
        "artifact_sha256": _sha256(provenance.feature_artifact)
    })
    with pytest.raises(ValueError, match="canonical identity"):
        entrypoint._load_feature_only(runtime, provenance)

    runtime, provenance, _, _, _ = make_runtime_case(tmp_path / "second", "BDGP")
    views, sample_ids, contract, _ = entrypoint._load_feature_only(runtime, provenance)
    split, _ = entrypoint._load_sparse_split(
        runtime, provenance, sample_ids, contract.n_clusters
    )
    with np.load(provenance.utility_artifact, allow_pickle=False) as archive:
        utility = {key: archive[key] for key in archive.files}
    utility["unlabeled_ids"] = utility["unlabeled_ids"][::-1].copy()
    np.savez(provenance.utility_artifact, **utility)
    _write_json(provenance.utility_audit, {
        "artifact_sha256": _sha256(provenance.utility_artifact)
    })
    with pytest.raises(RuntimeError, match="query IDs"):
        entrypoint._load_action_inputs(provenance, split, sample_ids.size)


def test_checkpoint_loader_is_cpu_and_strict_loading_is_present(tmp_path):
    path = tmp_path / "state.pt"
    torch.save({"state_dict": {"weight": torch.ones(2)}}, path)
    state = entrypoint._load_torch_state(path)
    assert set(state) == {"weight"}
    source = inspect.getsource(entrypoint._load_initial_model)
    assert 'map_location="cpu"' in inspect.getsource(entrypoint._load_torch_state)
    assert "strict=True" in source


def test_runtime_contains_no_scientific_formula_duplication_or_closed_routes():
    source = inspect.getsource(entrypoint)
    assert "experiments." not in source
    for fragment in (
        "MinMaxScaler", "KMeans(", "linear_sum_assignment", "optimizer.step",
        ".backward(", "U_tilde", "pseudo_anchor", "prototype_memory",
    ):
        assert fragment not in source
    assert source.count("run_alternating_training(") == 1
    assert source.count("native_refresh_from_latents(") == 1
