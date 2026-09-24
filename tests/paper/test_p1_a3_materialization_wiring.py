from pathlib import Path

from experiments.paper.formal import p1_a1_base_runtime as base
from experiments.paper.formal import p1_a1_native_preparation as initialization
from experiments.paper.formal import p1_a3_materialize as cli
from experiments.paper.formal import p1_a3_runtime_wiring as wiring


def test_materialization_cli_is_stage_only_and_has_no_scientific_override():
    args = cli.parse_args(["--dataset", "Caltech-6V", "--training-seed", "20", "--device", "cpu", "--stage", "inputs", "--plan-only"])
    assert args.stage == "inputs" and args.plan_only
    source = Path(cli.__file__).read_text(encoding="utf-8")
    for forbidden in ("--epochs", "--lr", "--lambda1", "--snr", "--label-seed"):
        assert forbidden not in source


def test_public_builders_are_wired_and_base_contract_remains_native_only():
    assert "NOT_IMPLEMENTED" not in Path(initialization.__file__).read_text(encoding="utf-8")
    contract = base.base_audit_contract()
    assert contract["phase_a_executed"] is False
    assert contract["semantic_optimizer_created"] is False
    assert contract["final_prediction_source"].endswith("KMeans prediction IDs")


def test_provenance_wiring_uses_release_config_without_metric_or_gt_import(tmp_path):
    names = ("feature", "feature_audit", "split", "split_audit", "utility", "utility_audit", "semantic", "semantic_audit", "checkpoint_audit", "checkpoint")
    paths = {}
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"x")
        paths[name] = path
    init = {"audit": paths["checkpoint_audit"], "checkpoint_paths": (paths["checkpoint"],), "initial_model_sha256": "a" * 64}
    provenance = wiring.arm_provenance(feature=paths["feature"], feature_audit=paths["feature_audit"], split=paths["split"], split_audit=paths["split_audit"], utility=paths["utility"], utility_audit=paths["utility_audit"], semantic=paths["semantic"], semantic_audit=paths["semantic_audit"], initialization=init, output=tmp_path / "output")
    assert provenance.strict_replay is True
    assert provenance.expected_initial_model_sha256 == "a" * 64
