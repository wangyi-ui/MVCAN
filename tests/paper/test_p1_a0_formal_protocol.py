import json
import subprocess
from pathlib import Path

import pytest

from experiments.paper.formal import audit_p1_a0_formal_protocol as audit
from experiments.paper.formal import p1_a0_formal_protocol as protocol


def test_native_tuple_values_are_config_seeds_not_epochs():
    source = Path("release_core/config/native.py").read_text(encoding="utf-8")
    assert 'seed, lambda1 = _TRAINING[data_name]' in source
    assert {item.name: item.native_config_seed for item in protocol.FORMAL_DATASETS} == {
        "Caltech-6V": 5, "MSRC-v1": 20, "BDGP": 1,
    }
    assert protocol.NATIVE_PREPARATION_PROTOCOL["epoch"] == 1000
    assert protocol.NATIVE_PREPARATION_PROTOCOL["init_epoch"] == 200


def test_native_schedule_and_dataset_lambda_are_frozen():
    native = protocol.NATIVE_PREPARATION_PROTOCOL
    assert native["batch_size"] == 256
    assert native["T_2"] == 100
    assert native["lr"] == 1e-4
    assert {item.name: item.native_lambda1 for item in protocol.FORMAL_DATASETS} == {
        "Caltech-6V": 0.01, "MSRC-v1": 0.01, "BDGP": 10.0,
    }


def test_formal_seeds_and_weak_label_contracts_are_separate():
    assert protocol.FORMAL_TRAINING_SEEDS == (20, 30, 50)
    assert protocol.WEAK_QUALITY_PROTOCOL["realization_seed"] == 20
    assert protocol.WEAK_QUALITY_PROTOCOL["training_seed_changes_realization"] is False
    assert protocol.SPARSE_LABEL_PROTOCOL["label_seed"] == 20
    assert protocol.SPARSE_LABEL_PROTOCOL["labels_per_class"] == 2


def test_dataset_weak_quality_contracts_cover_even_odd_and_minimal_view_cases():
    contracts = {item.name: item.weak_quality_mask_contract for item in protocol.FORMAL_DATASETS}
    assert contracts["Caltech-6V"] == "exactly 3 of 6 views per sample"
    assert contracts["MSRC-v1"] == "105 samples x2 and 105 samples x3 corrupted views"
    assert contracts["BDGP"] == "exactly 1 of 2 views per sample"
    assert all(item.weak_quality_mask_sha256 for item in protocol.FORMAL_DATASETS)
    assert {item.name: item.weak_quality_mask_sha256 for item in protocol.FORMAL_DATASETS}["BDGP"] == "a3c882c29f2f5064d552e60bdc1a2d585764317e7ba78aa055cd1c3c22e5de07"


def test_action_generation_is_same_condition_once_per_dataset_seed_and_frozen_for_r5():
    action = protocol.ACTION_GENERATION_PROTOCOL
    assert action["state_dependent"] is True
    assert action["per_dataset_training_seed"] is True
    assert action["during_r5"] == "detached and frozen; never dynamically regenerated per epoch"


def test_alternating_order_refresh_and_final_prediction_contract():
    alternating = protocol.ALTERNATING_PROTOCOL
    assert alternating.epochs == 20
    assert alternating.phase_order[0] == "Phase A"
    assert "zero-based epoch % T2 == 0" in alternating.phase_order[1]
    assert alternating.phase_order[-1] == "Phase B"
    assert alternating.final_prediction_source == "final native refresh second-pass KMeans prediction IDs"
    assert protocol.EVALUATION_PROTOCOL["q_argmax_is_final_prediction"] is False


def test_gt_firewall_and_forbidden_mechanisms_are_explicit():
    assert protocol.EVALUATION_PROTOCOL["pre_gt_full_gt_loaded"] is False
    assert protocol.EVALUATION_PROTOCOL["postseal_only"] == ("ACC", "NMI", "ARI")
    forbidden = set(protocol.FORBIDDEN_SCIENCE)
    assert {"U_tilde", "memory bank", "pseudo-label CE", "feature gating", "fusion gating", "additive joint loss", "result-based tuning"} <= forbidden


def test_main_and_ablation_arms_are_separate():
    assert set(protocol.ARM_PROTOCOL["main"]) == {"BASE", "OURS_TRUE_U"}
    assert set(protocol.ARM_PROTOCOL["ablation"]) == {"TRUE_UNIFORM", "SHUFFLE_U"}


def test_unresolved_critical_field_fails_closed(monkeypatch):
    monkeypatch.setattr(protocol, "FORMAL_TRAINING_SEEDS", (20,))
    with pytest.raises(RuntimeError, match="unresolved"):
        protocol.validate_formal_protocol()


def test_report_is_protocol_only_and_resolved(tmp_path):
    report = audit.build_report()
    assert report["decision"] == "FORMAL_PROTOCOL_RESOLVED"
    assert report["formal_training_executed"] is False
    target = audit.write_report(tmp_path / "report.json")
    assert json.loads(target.read_text(encoding="utf-8"))["unresolved_fields"] == []


def test_protected_sources_and_historical_archive_are_unchanged():
    root = Path(__file__).resolve().parents[2]
    diff = subprocess.run(
        ["git", "diff", "--name-only", "--", "release_core"], cwd=root,
        check=True, text=True, stdout=subprocess.PIPE,
    )
    historical = subprocess.run(
        ["git", "-C", "/root/autodl-tmp/CVPR24-MVCAN", "status", "--short"],
        check=True, text=True, stdout=subprocess.PIPE,
    )
    assert diff.stdout == ""
    assert historical.stdout == ""
