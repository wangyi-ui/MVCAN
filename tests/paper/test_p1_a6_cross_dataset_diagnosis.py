import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from experiments.paper.diagnostics import d5_r1_refresh_cadence as d5
from experiments.paper.formal import p1_a3_input_materialization as inputs


def test_bdgp_formal_boundary_is_generic_exact_and_gt_free(tmp_path, monkeypatch):
    source = tmp_path / "BDGP2V_N.mat"
    source.write_bytes(b"synthetic source")
    views = (np.full((2500, 1750), 1.0, dtype=np.float32), np.full((2500, 79), 2.0, dtype=np.float32))
    labels = np.repeat(np.arange(5, dtype=np.int64), 500)
    monkeypatch.setattr(inputs, "_bdgp_source_path", lambda: source)
    import release_core.data as release_data
    import experiments.generic_contract.sparse_label_contract as sparse
    monkeypatch.setattr(release_data, "load_bdgp", lambda path: (views, [labels]))
    monkeypatch.setattr(sparse, "frozen_caltech_sparse_split", lambda: (_ for _ in ()).throw(AssertionError("BDGP must not use Caltech split")))
    root = inputs.materialize_bdgp_inputs(tmp_path / "bdgp")
    weak = json.loads((root / "weak_quality_audit.json").read_text())
    split = json.loads((root / "sparse_split_audit.json").read_text())
    assert weak == {"mask_logical_sha256": "a3c882c29f2f5064d552e60bdc1a2d585764317e7ba78aa055cd1c3c22e5de07", "snr_db": 2.5, "corrupted_pair_count": 2500, "per_sample_corrupted_count_unique": [1], "per_view_corrupted_counts": [1250, 1250]}
    assert split["sparse_split_source"] == "generic SHA256-ranked sparse split"
    with np.load(root / "features.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"view_1", "view_2", "sample_ids"}
        assert archive["view_1"].shape == (2500, 1750) and archive["view_2"].shape == (2500, 79)
        assert not any("label" in key.lower() or key.lower() in {"y", "gt"} for key in archive.files)
    with np.load(root / "sparse_split.npz", allow_pickle=False) as archive:
        assert archive["labeled_ids"].size == 10
        assert np.array_equal(np.bincount(archive["labeled_targets"], minlength=5), np.full(5, 2))


def test_d5_runtime_changes_only_refresh_interval_and_refreshes_every_epoch():
    v1, diagnostic = d5.diagnostic_runtime(20, "cpu")
    assert v1.refresh_interval == 100 and diagnostic.refresh_interval == 1
    assert {key for key in asdict(v1) if getattr(v1, key) != getattr(diagnostic, key)} == {"refresh_interval"}
    assert d5.refresh_epochs(diagnostic) == tuple(range(20))


def test_d5_output_namespace_is_disjoint_and_invalid_seed_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(d5, "ROOT", tmp_path / "diagnostics")
    path = d5.paths_for(20)
    assert not d5._under(path["root"], d5.FORMAL_ROOT)
    with pytest.raises(RuntimeError, match="SEED_NOT_AUTHORIZED"):
        d5.paths_for(99)


def test_d5_v1_verification_is_read_only_and_does_not_train(tmp_path, monkeypatch):
    monkeypatch.setattr(d5, "ROOT", tmp_path / "diagnostics")
    paths = d5.paths_for(20)
    watched = (paths["inputs"] / "features.npz", paths["initialization"] / "initialization_seal.json", paths["action"] / "true_action_state.npz", paths["base"] / "pre_gt_seal.json", paths["ours_v1"] / "pre_gt_seal.json")
    before = {path: d5._sha(path) for path in watched}
    plan = d5.prepare(20, "cpu")
    assert plan["runtime"].refresh_interval == 1
    assert json.loads(Path(plan["adapter"]["utility_audit"]).read_text())["source_action_sha256"] == plan["action"]["artifact_sha256"]
    assert plan["paths"]["output"].exists() is False
    assert plan["baselines"]["BASE_v1"]["metrics"]
    assert plan["baselines"]["OURS_TRUE_U_v1_refresh100"]["metrics"]
    assert {path: d5._sha(path) for path in watched} == before
    assert not (tmp_path / "formal").exists()


def test_d5_existing_diagnostic_namespace_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(d5, "ROOT", tmp_path / "diagnostics")
    d5.paths_for(20)["root"].mkdir(parents=True)
    with pytest.raises(RuntimeError, match="OUTPUT_ALREADY_EXISTS"):
        d5.prepare(20, "cpu")


def test_d5_source_has_no_gt_loader_or_base_retrain_call():
    source = Path(d5.__file__).read_text(encoding="utf-8")
    assert "load_dataset(" not in source and "run_base_pre_gt(" not in source
    assert "run_pre_gt(" in source and "postseal._validate" in source
    assert "u_or_relation_regenerated\": False" in source
    assert "base_or_v1_ours_retrained\": False" in source
