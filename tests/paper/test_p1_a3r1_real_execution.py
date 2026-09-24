import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.generic_contract.sparse_label_contract import frozen_caltech_sparse_split, materialize_hash_ranked_sparse_split
from experiments.paper.formal import p1_a3_input_materialization as inputs
from release_core.data.weak_quality import ndarray_sha256
from release_core.semantics import validate_sparse_label_split
from experiments.paper.formal import p1_a1_action_materialization as actions
from experiments.paper.formal import p1_a1_base_runtime as base
from experiments.paper.formal import p1_a1_native_preparation as preparation
from experiments.paper.formal import p1_a1_runner as runner
from experiments.paper.formal import p1_a3_real_execution as real
from experiments.paper.formal import p1_a3_runtime_wiring as wiring
from release_core.runtime import RuntimeConfig
from release_core.semantics import SparseLabelSplit


def _tiny_spec():
    return SimpleNamespace(name="Tiny", n_samples=6, n_views=2, n_clusters=2, view_dims=(2, 2), native_config_seed=5)


def _tiny_config():
    return {"Autoencoder": {"arch": [2], "channal": [1], "activations": "relu", "batchnorm": False, "FCN": True}, "training": {"seed": 5, "batch_size": 6, "init_epoch": 1, "T_1": 2, "T_2": 1, "epoch": 1, "lr": 1e-4, "lambda1": 0.01}}


def test_synthetic_writer_seals_adapter_and_dispatches_without_caltech(tmp_path, monkeypatch):
    spec = _tiny_spec()
    monkeypatch.setattr(real, "item", lambda dataset: spec)
    monkeypatch.setattr(real, "get_native_config", lambda dataset: _tiny_config())
    monkeypatch.setattr(real.p2, "validate_execution_contract", lambda: True)
    monkeypatch.setattr(preparation.protocol, "FORMAL_DATASETS", (spec,))
    feature = tmp_path / "features.npz"
    np.savez(feature, view_1=np.ascontiguousarray(np.arange(12, dtype=np.float32).reshape(6, 2)), view_2=np.ascontiguousarray(np.arange(12, 24, dtype=np.float32).reshape(6, 2)), sample_ids=np.arange(6, dtype=np.int64))
    init_root = tmp_path / "initialization"
    real.build_initialization(dataset="Tiny", training_seed=20, feature_path=feature, output_dir=init_root, device="cpu")
    initialization = preparation.verify_initialization(init_root, dataset="Tiny", training_seed=20)
    assert len(initialization["checkpoint_paths"]) == 2
    assert all(path.is_file() for path in initialization["checkpoint_paths"])
    assert (init_root / "carrier_state.npz").is_file()
    assert json.loads((init_root / "initialization_seal.json").read_text())["seal_valid"] is True
    split_path = tmp_path / "split.npz"
    np.savez(split_path, sample_ids=np.arange(6, dtype=np.int64), labeled_ids=np.array([0, 1, 3, 4], dtype=np.int64), labeled_targets=np.array([0, 0, 1, 1], dtype=np.int64), unlabeled_ids=np.array([2, 5], dtype=np.int64))
    action_root = tmp_path / "action"
    real.build_action(dataset="Tiny", training_seed=20, carrier_state_path=init_root / "carrier_state.npz", split_path=split_path, output_dir=action_root, initial_model_sha256=initialization["initial_model_sha256"])
    action = actions.verify_true_action(action_root, dataset="Tiny", training_seed=20, initial_model_sha256=initialization["initial_model_sha256"])
    assert action["artifact"].is_file()
    assert json.loads((action_root / "action_seal.json").read_text())["seal_valid"] is True
    adapter = wiring.materialize_ours_true_u_adapter(true_action=action, output_dir=tmp_path / "adapter")
    for path in (adapter["utility"], adapter["utility_audit"], adapter["semantic"], adapter["semantic_audit"]):
        assert path.is_file()
    feature_audit, split_audit = tmp_path / "feature_audit.json", tmp_path / "split_audit.json"
    feature_audit.write_text("{}")
    split_audit.write_text("{}")
    provenance = wiring.arm_provenance(feature=feature, feature_audit=feature_audit, split=split_path, split_audit=split_audit, utility=adapter["utility"], utility_audit=adapter["utility_audit"], semantic=adapter["semantic"], semantic_audit=adapter["semantic_audit"], initialization=initialization, output=tmp_path / "ours")
    assert provenance.strict_replay is True and provenance.expected_initial_model_sha256 == initialization["initial_model_sha256"]
    assert "selected[\"provenance\"]" not in Path(runner.__file__).read_text()
    dispatched = {}
    monkeypatch.setattr(runner, "run_pre_gt", lambda runtime, value: dispatched.setdefault("ours", value))
    assert runner.run_pre_gt(RuntimeConfig(dataset="Caltech-6V", training_seed=20, device="cpu"), provenance) is provenance
    contract = SimpleNamespace(n_views=2, n_clusters=2)
    split = SparseLabelSplit(np.arange(6, dtype=np.int64), np.array([0, 1, 3, 4], dtype=np.int64), np.array([0, 0, 1, 1], dtype=np.int64), np.array([2, 5], dtype=np.int64), 2, 2, 20, "Caltech-6V")
    model = real.MultiViewBackbone(_tiny_config(), 2, (2, 2), 2, seed=5).to_device(torch.device("cpu"))
    monkeypatch.setattr(base.entry, "_verify_expected_files", lambda value: None)
    monkeypatch.setattr(base.entry, "_load_feature_only", lambda runtime, value: ((torch.zeros(6, 2).numpy().astype(np.float32), torch.ones(6, 2).numpy().astype(np.float32)), np.arange(6, dtype=np.int64), contract, "f" * 64))
    monkeypatch.setattr(base.entry, "_load_sparse_split", lambda runtime, value, ids, count: (split, "s" * 64))
    monkeypatch.setattr(base.entry, "_configure_determinism", lambda seed, device: (torch.device("cpu"), {}))
    monkeypatch.setattr(base.entry, "_load_initial_model", lambda runtime, value, loaded_contract, device: (model, initialization["initial_model_sha256"], tuple("c" * 64 for _ in range(2)), "a" * 64))
    monkeypatch.setattr(base.entry, "_final_prediction_state", lambda model, views, state, seed, device: (np.zeros(6, dtype=np.int64), np.zeros((6, 2, 2), dtype=np.float32), np.zeros((6, 2, 2), dtype=np.float32), np.zeros((2, 2, 2), dtype=np.float32), {}))
    monkeypatch.setattr(base.entry, "_persist_pre_gt", lambda output, arrays, audit: (arrays, audit))
    result = base.run_base_pre_gt(RuntimeConfig(dataset="Caltech-6V", training_seed=20, epochs=1, batch_size=6, refresh_interval=100, device="cpu"), provenance)
    assert result[1]["base"]["phase_a_executed"] is False
    assert result[1]["base"]["semantic_optimizer_created"] is False


def test_real_execution_stub_names_are_absent_and_protected_sources_stay_clean():
    root = Path(__file__).resolve().parents[2]
    source = "\n".join(path.read_text(encoding="utf-8") for path in (Path(real.__file__), Path(base.__file__), Path(wiring.__file__), Path(runner.__file__)))
    for token in ("FORMAL_NATIVE_PREPARATION_BUILDER_NOT_IMPLEMENTED", "FORMAL_ACTION_MATERIALIZATION_BUILDER_NOT_IMPLEMENTED", "FORMAL_BASE_REQUIRES_SEALED_INITIALIZATION", "P1_A3_REAL_MATERIALIZATION_REQUIRES_EXPLICIT_NEXT_STAGE_AUTHORIZATION", "selected[\"provenance\"]"):
        assert token not in source
    import subprocess
    assert not subprocess.run(["git", "diff", "--name-only", "--", "release_core", "experiments/paper/formal/p1_a0_formal_protocol.py", "experiments/paper/formal/p1_a2_execution_contract.py"], cwd=root, text=True, stdout=subprocess.PIPE, check=True).stdout


def test_generic_sparse_split_crosses_runtime_type_boundary_without_semantic_change(tmp_path, monkeypatch):
    mask = np.zeros((6, 2), dtype=np.bool_)
    tiny = SimpleNamespace(name="Tiny", n_samples=6, n_views=2, n_clusters=2, view_dims=(2, 2), native_config_seed=5, weak_quality_mask_sha256=ndarray_sha256(mask))
    monkeypatch.setattr(inputs.protocol, "FORMAL_DATASETS", (tiny,))
    raw_split = materialize_hash_ranked_sparse_split(np.array([0, 0, 0, 1, 1, 1], dtype=np.int64), dataset_name="Tiny", label_seed=20, labels_per_class=2)
    runtime_split = inputs._runtime_split_from_generic(raw_split, sample_ids=np.arange(6, dtype=np.int64), class_count=2)
    validated = validate_sparse_label_split(runtime_split)
    assert runtime_split.digest == raw_split.split_sha256
    assert np.array_equal(runtime_split.labeled_ids, raw_split.labeled_ids)
    assert np.array_equal(runtime_split.labeled_targets, raw_split.labeled_targets)
    assert np.array_equal(runtime_split.unlabeled_ids, raw_split.unlabeled_ids)
    assert (runtime_split.labels_per_class, runtime_split.label_seed) == (2, 20)
    output = inputs.materialize_inputs(dataset="Tiny", views=(np.zeros((6, 2), dtype=np.float32), np.ones((6, 2), dtype=np.float32)), corruption_mask=mask, sparse_split=validated, output_dir=tmp_path / "sealed")
    with np.load(output / "sparse_split.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"sample_ids", "labeled_ids", "labeled_targets", "unlabeled_ids"}
        assert np.array_equal(archive["labeled_ids"], raw_split.labeled_ids)
        assert np.array_equal(archive["labeled_targets"], raw_split.labeled_targets)
        assert np.array_equal(archive["unlabeled_ids"], raw_split.unlabeled_ids)
    audit = json.loads((output / "sparse_split_audit.json").read_text())
    assert audit["full_gt_persisted"] is False and audit["unlabeled_gt_persisted"] is False
    frozen = frozen_caltech_sparse_split()
    frozen_runtime = inputs._runtime_split_from_generic(frozen, sample_ids=np.arange(1400, dtype=np.int64), class_count=7)
    assert frozen_runtime.digest == frozen.split_sha256
    assert np.array_equal(frozen_runtime.labeled_ids, frozen.labeled_ids)
    assert np.array_equal(frozen_runtime.labeled_targets, frozen.labeled_targets)
    assert np.array_equal(frozen_runtime.unlabeled_ids, frozen.unlabeled_ids)


def test_caltech_path_uses_only_frozen_split_authority_and_truthful_audit(tmp_path, monkeypatch):
    import release_core.data as release_data
    import release_core.data.weak_quality as weak_quality
    import experiments.generic_contract.sparse_label_contract as sparse_contract
    class FakePath:
        def is_file(self):
            return True
    seen = {}
    monkeypatch.setattr(inputs, "Path", lambda value: FakePath())
    monkeypatch.setattr(inputs, "_sha", lambda path: "72fa848269b663f819a8e9bd441628ece1955c654d98e2b10a85be1cd2613d5a")
    monkeypatch.setattr(release_data, "load_caltech", lambda path: ((), np.zeros((1, 1400), dtype=np.int64)))
    monkeypatch.setattr(weak_quality, "apply_half_gaussian_corruption", lambda views, snr, seed: (("view",), None))
    monkeypatch.setattr(weak_quality, "generate_half_corruption_mask", lambda n, v, seed: (np.zeros((n, v), dtype=np.bool_), None))
    monkeypatch.setattr(sparse_contract, "materialize_hash_ranked_sparse_split", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Caltech must not invoke hash-ranked selection")))
    monkeypatch.setattr(inputs, "materialize_inputs", lambda **kwargs: seen.update(kwargs) or "sealed")
    assert inputs.materialize_caltech_inputs(tmp_path / "unused") == "sealed"
    split = seen["sparse_split"]
    assert np.array_equal(split.labeled_ids, np.array([67, 82, 90, 111, 200, 365, 440, 513, 536, 983, 1027, 1250, 1316, 1385], dtype=np.int64))
    assert np.array_equal(split.labeled_targets, np.array([4, 6, 2, 3, 0, 4, 1, 5, 6, 3, 0, 2, 5, 1], dtype=np.int64))
    assert split.digest == frozen_caltech_sparse_split().split_sha256
    assert (split.labels_per_class, split.label_seed) == (2, 20)
    assert seen["sparse_split_source"] == "frozen Caltech backward-compatibility split"
    assert seen["full_gt_loaded_at_input_boundary"] is True
    assert seen["full_gt_used_to_select_sparse_ids"] is False
    for dataset in ("MSRC-v1", "BDGP"):
        generic = materialize_hash_ranked_sparse_split(np.array([0, 0, 0, 1, 1, 1], dtype=np.int64), dataset_name=dataset, label_seed=20, labels_per_class=2)
        assert generic.labels_per_class == 2 and generic.label_seed == 20
