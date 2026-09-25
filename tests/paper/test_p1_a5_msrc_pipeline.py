import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.paper.formal import p1_a3_input_materialization as inputs
from experiments.paper.formal import run_formal_pipeline as pipeline


def test_msrc_formal_input_boundary_uses_frozen_odd_view_and_generic_split(tmp_path, monkeypatch):
    source = tmp_path / "MSRC-v1.mat"
    source.write_bytes(b"source")
    views = tuple(np.full((210, dim), index + 1, dtype=np.float32) for index, dim in enumerate((24, 576, 512, 256, 254)))
    labels = np.repeat(np.arange(7, dtype=np.int64), 30)
    monkeypatch.setattr(inputs, "_msrc_source_path", lambda: source)
    import release_core.data as release_data
    import experiments.generic_contract.sparse_label_contract as sparse
    monkeypatch.setattr(release_data, "load_msrc_v1", lambda path: (views, [labels]))
    monkeypatch.setattr(sparse, "frozen_caltech_sparse_split", lambda: (_ for _ in ()).throw(AssertionError("MSRC must not use Caltech split")))
    root = inputs.materialize_msrc_inputs(tmp_path / "msrc")
    audit = json.loads((root / "weak_quality_audit.json").read_text())
    split_audit = json.loads((root / "sparse_split_audit.json").read_text())
    assert audit["corrupted_pair_count"] == 525
    assert audit["per_sample_corrupted_count_unique"] == [2, 3]
    assert audit["per_view_corrupted_counts"] == [105] * 5
    assert audit["mask_logical_sha256"] == "d64f5fbf95518e5a77e44a1fe18a15dee9924b8c2a341e930f86f20ce5ac9e9d"
    assert split_audit["sparse_split_source"] == "generic SHA256-ranked sparse split"
    with np.load(root / "features.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"view_1", "view_2", "view_3", "view_4", "view_5", "sample_ids"}
        assert tuple(archive["view_" + str(i + 1)].shape for i in range(5)) == tuple((210, d) for d in (24, 576, 512, 256, 254))
        assert not any("label" in name.lower() or name == "gt" for name in archive.files)
    with np.load(root / "sparse_split.npz", allow_pickle=False) as archive:
        assert archive["labeled_ids"].size == 14
        assert np.array_equal(np.bincount(archive["labeled_targets"], minlength=7), np.full(7, 2))


def test_pipeline_stage_order_resume_and_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path / "formal")
    (pipeline.ROOT / "main" / "msrcv1").mkdir(parents=True)
    events = []
    monkeypatch.setattr(pipeline, "_existing_or_run", lambda label, target, verify, build, resume: events.append(label))
    monkeypatch.setattr(pipeline.preparation, "verify_initialization", lambda *args, **kwargs: {"initial_model_sha256": "a" * 64})
    monkeypatch.setattr(pipeline, "_summary", lambda dataset, seeds, arms: {"dataset": dataset, "rows": [], "aggregate": {}})
    pipeline.run_pipeline(dataset="MSRC-v1", training_seeds=(20, 30, 50), arms=("BASE", "OURS_TRUE_U"), device="cpu", full_gt_path="unused", resume=True)
    expected = ["[INPUTS]"]
    for seed in (20, 30, 50):
        expected.extend(["[seed%d][INITIALIZATION]" % seed, "[seed%d][ACTIONS]" % seed, "[seed%d][BASE]" % seed, "[seed%d][OURS_TRUE_U]" % seed, "[seed%d][BASE][EVALUATION]" % seed, "[seed%d][OURS_TRUE_U][EVALUATION]" % seed])
    assert events == expected
    with pytest.raises(RuntimeError, match="SEED_ORDER"):
        pipeline.run_pipeline(dataset="MSRC-v1", training_seeds=(30, 20), arms=("BASE", "OURS_TRUE_U"), device="cpu", full_gt_path="unused", resume=True)


def test_existing_invalid_artifact_fails_without_build(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    called = []
    with pytest.raises(RuntimeError, match="bad"):
        pipeline._existing_or_run("[X]", target, lambda path: (_ for _ in ()).throw(RuntimeError("bad")), lambda: called.append(True), True)
    assert called == []


def test_valid_resume_verifies_and_skips_without_build(tmp_path, capsys):
    target = tmp_path / "valid"
    target.mkdir()
    calls = []
    pipeline._existing_or_run("[VALID]", target, lambda path: calls.append(path), lambda: (_ for _ in ()).throw(AssertionError("must not build")), True)
    text = capsys.readouterr().out
    assert calls == [target]
    assert "[VALID]/VERIFY START" in text
    assert "[VALID]/VERIFY PASS" in text
    assert "[VALID] SKIP" in text


def test_summary_is_exact_and_resume_verifies_all_summary_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path / "formal")
    root = pipeline.ROOT / "main" / "msrcv1"
    root.mkdir(parents=True)
    expected = {"dataset": "MSRC-v1", "seeds": [20], "rows": [{"training_seed": 20, "BASE": {"acc": 0.1, "nmi": 0.2, "ari": 0.3}, "OURS_TRUE_U": {"acc": 0.2, "nmi": 0.3, "ari": 0.4}, "delta": {"acc": 0.1, "nmi": 0.1, "ari": 0.1}}], "aggregate": {"BASE": {}, "OURS_TRUE_U": {}, "delta": {}}}
    monkeypatch.setattr(pipeline, "_summary", lambda dataset, seeds, arms: expected)
    result = pipeline._write_or_verify_summary("MSRC-v1", (20,), ("BASE", "OURS_TRUE_U"), False)
    assert result == expected
    assert json.loads((root / "formal_summary.json").read_text()) == expected
    assert (root / "formal_summary.txt").read_text() == json.dumps(expected, sort_keys=True, indent=2) + "\n"
    assert pipeline._write_or_verify_summary("MSRC-v1", (20,), ("BASE", "OURS_TRUE_U"), True) == expected
    assert "[SUMMARY] SKIP" in capsys.readouterr().out


@pytest.mark.parametrize("failed_label", ("[seed20][INITIALIZATION]", "[seed20][ACTIONS]", "[seed20][BASE]", "[seed20][OURS_TRUE_U]"))
def test_pipeline_stops_before_any_downstream_stage_on_stage_failure(tmp_path, monkeypatch, failed_label):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path / "formal")
    (pipeline.ROOT / "main" / "msrcv1").mkdir(parents=True)
    seen = []
    def stage(label, target, verify, build, resume):
        seen.append(label)
        if label == failed_label:
            raise RuntimeError("deliberate stage failure")
    monkeypatch.setattr(pipeline, "_existing_or_run", stage)
    monkeypatch.setattr(pipeline.preparation, "verify_initialization", lambda *args, **kwargs: {"initial_model_sha256": "a" * 64})
    with pytest.raises(RuntimeError, match="deliberate stage failure"):
        pipeline.run_pipeline(dataset="MSRC-v1", training_seeds=(20,), arms=("BASE", "OURS_TRUE_U"), device="cpu", full_gt_path="unused", resume=True)
    assert failed_label in seen
    downstream = {
        "[seed20][ACTIONS]": ("[seed20][BASE]", "[seed20][OURS_TRUE_U]", "[seed20][BASE][EVALUATION]", "[seed20][OURS_TRUE_U][EVALUATION]"),
        "[seed20][BASE]": ("[seed20][OURS_TRUE_U]", "[seed20][BASE][EVALUATION]", "[seed20][OURS_TRUE_U][EVALUATION]"),
        "[seed20][OURS_TRUE_U]": ("[seed20][BASE][EVALUATION]", "[seed20][OURS_TRUE_U][EVALUATION]"),
        "[seed20][INITIALIZATION]": ("[seed20][ACTIONS]", "[seed20][BASE]", "[seed20][OURS_TRUE_U]"),
    }[failed_label]
    assert all(label not in seen for label in downstream)


def test_pipeline_cli_has_no_scientific_override_and_tee_log(tmp_path, monkeypatch):
    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    for forbidden in ("--epochs", "--lr", "--lambda1", "--batch-size", "--refresh-interval"):
        assert forbidden not in source
    monkeypatch.setattr(pipeline, "run_pipeline", lambda **kwargs: print("[SUMMARY] FINAL METRICS {}", flush=True) or kwargs)
    log = tmp_path / "pipeline.log"
    result = pipeline.main(["--dataset", "MSRC-v1", "--training-seeds", "20", "30", "50", "--arms", "BASE", "OURS_TRUE_U", "--device", "cpu", "--full-gt-path", "unused", "--resume", "--log-file", str(log)])
    assert result["training_seeds"] == (20, 30, 50)
    assert "FINAL METRICS" in log.read_text()
