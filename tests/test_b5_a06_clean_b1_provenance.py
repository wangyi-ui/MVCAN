"""Tests for the B5-A0.6 canonical clean B1 provenance bridge."""

import hashlib
import inspect
import json
from pathlib import Path

import pytest

import experiments.b5_semantic_rate.audit_b5_a06_clean_b1_provenance as a06
import experiments.b5_semantic_rate.train_b5_a0_shared_semantic_rate as train
from configure import get_default_config
from datasets import load_data
from irv.b3_audit import hash_backbone as canonical_hash_backbone
from irv.b4_information_utility import (
    tensor_view_list_sha256 as canonical_tensor_view_list_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT / "experiments/b5_semantic_rate/b5_a0_condition_manifest.json"
)
A06_OUTPUT = ROOT / "outputs/b5_semantic_rate/a06_clean_b1_provenance"
NOISY_ENTRY_SHA256 = {
    20: "a41a6b7391d40034efe9f7cafb62f0845940b7e307ce653ad9ee733391d729b5",
    30: "3824a2cd5540ce728ad4fd1c763c0ed3653ceae7f0f905726c82e273065843de",
    50: "28510d550a416b69341e161bd423881f8bcf0506e853cc5b4f3b5845e19f39e8",
}


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _entry_sha256(entry):
    encoded = json.dumps(
        entry, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_clean_entry(model_seed):
    return train._load_and_validate_manifest(
        MANIFEST_PATH,
        model_seed=model_seed,
        requested_conditions=("clean",),
    )["clean"]


@pytest.fixture(scope="module")
def manifest_value():
    return _read_json(MANIFEST_PATH)


@pytest.fixture(scope="module")
def clean_manifest_preflight():
    config = get_default_config("MSRC-v1")
    config["dataset"] = "MSRC-v1"
    clean_views = load_data(config)[0]
    result = {}
    for seed in (20, 30, 50):
        entry = _load_clean_entry(seed)
        models, backbone_hash = train._load_frozen_backbone(
            entry, config, clean_views
        )
        z_views = train._extract_frozen_z(models, clean_views)
        result[seed] = {
            "backbone_hash": backbone_hash["aggregate"],
            "z_hash": canonical_tensor_view_list_sha256(z_views),
        }
    return result


@pytest.fixture(scope="module")
def audit_result(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("b5_a06_provenance")
    summary, per_seed = a06.run_provenance_audit(
        (20, 30, 50), output_dir
    )
    return output_dir, summary, per_seed


def test_all_three_seeds_have_five_checkpoints():
    for seed in (20, 30, 50):
        paths = a06.checkpoint_paths_for(
            a06.B1_BACKBONE_TEMPLATE.format(seed=seed)
        )
        assert len(paths) == 5


def test_all_checkpoint_files_exist():
    for seed in (20, 30, 50):
        paths = a06.checkpoint_paths_for(
            a06.B1_BACKBONE_TEMPLATE.format(seed=seed)
        )
        assert all(path.is_file() for path in paths)


def test_seed20_b1_and_b3_model_loading_succeeds(audit_result):
    _, _, per_seed = audit_result
    assert per_seed[20]["checkpoint_count"] == 5
    assert per_seed[20]["b3_reference_audit"]["checkpoint_count"] == 5


def test_hash_backbone_is_the_canonical_b3_helper():
    assert a06.hash_backbone is canonical_hash_backbone


def test_z_hash_is_the_canonical_b4_helper():
    assert a06.tensor_view_list_sha256 is canonical_tensor_view_list_sha256


def test_each_seed_has_five_z_views(audit_result):
    _, _, per_seed = audit_result
    assert all(len(per_seed[seed]["z_shape"]) == 5 for seed in (20, 30, 50))


def test_each_z_view_has_shape_210_by_10(audit_result):
    _, _, per_seed = audit_result
    expected = [[210, 10]] * 5
    assert all(per_seed[seed]["z_shape"] == expected for seed in (20, 30, 50))


def test_each_z_view_is_l2_normalized(audit_result):
    _, _, per_seed = audit_result
    assert all(
        per_seed[seed]["z_l2_norm_max_abs_error_from_one"] <= 1e-6
        for seed in (20, 30, 50)
    )


def test_seed20_historical_backbone_hash_exact_match(audit_result):
    _, _, per_seed = audit_result
    assert (
        per_seed[20]["b3_backbone_hash"]
        == a06.EXPECTED_HISTORICAL_BACKBONE_HASH
    )
    assert per_seed[20]["b3_historical_backbone_hash_exact_match"]


def test_seed20_historical_z_hash_exact_match(audit_result):
    _, _, per_seed = audit_result
    assert (
        per_seed[20]["b3_z_hash"]
        == a06.EXPECTED_HISTORICAL_Z_HASH
    )
    assert per_seed[20]["b3_historical_z_hash_exact_match"]


def test_seed20_b1_b3_backbone_hash_exact_comparison(audit_result):
    _, _, per_seed = audit_result
    assert per_seed[20]["backbone_hash_exact_match"] == (
        per_seed[20]["b1_backbone_hash"]
        == per_seed[20]["b3_backbone_hash"]
    )


def test_seed20_b1_b3_z_hash_exact_comparison(audit_result):
    _, _, per_seed = audit_result
    assert per_seed[20]["z_hash_exact_match"] == (
        per_seed[20]["b1_z_hash"] == per_seed[20]["b3_z_hash"]
    )
    assert per_seed[20]["B5_A06_B1_B3_SEED20_EQUIVALENCE_PASS"] == (
        per_seed[20]["backbone_hash_exact_match"]
        and per_seed[20]["z_hash_exact_match"]
    )


def test_seed30_provenance_is_deterministic(audit_result):
    _, _, per_seed = audit_result
    result = per_seed[30]
    assert result["backbone_hash_exact_repeat"]
    assert result["z_hash_exact_repeat"]
    assert result["max_z_tensor_difference"] == 0.0
    assert result["deterministic_pass"]


def test_seed50_provenance_is_deterministic(audit_result):
    _, _, per_seed = audit_result
    result = per_seed[50]
    assert result["backbone_hash_exact_repeat"]
    assert result["z_hash_exact_repeat"]
    assert result["max_z_tensor_difference"] == 0.0
    assert result["deterministic_pass"]


def test_labels_are_not_used(audit_result):
    _, summary, per_seed = audit_result
    assert summary["labels_used"] is False
    assert all(per_seed[seed]["labels_used"] is False for seed in (20, 30, 50))
    public_parameters = (
        inspect.signature(a06.load_frozen_backbone).parameters,
        inspect.signature(a06.extract_normalized_native_z).parameters,
        inspect.signature(a06.audit_one_backbone).parameters,
    )
    assert all(
        "label" not in name.lower()
        for parameters in public_parameters
        for name in parameters
    )


def test_corruption_mask_is_not_used(audit_result):
    _, summary, per_seed = audit_result
    assert summary["corruption_mask_used"] is False
    assert all(
        per_seed[seed]["corruption_mask_used"] is False
        for seed in (20, 30, 50)
    )


def test_no_backward_is_present_or_reported(audit_result):
    _, summary, per_seed = audit_result
    source = inspect.getsource(a06)
    assert ".backward(" not in source
    assert summary["backward_performed"] is False
    assert all(
        per_seed[seed]["backward_performed"] is False
        for seed in (20, 30, 50)
    )


def test_no_optimizer_is_created(audit_result):
    _, summary, per_seed = audit_result
    source = inspect.getsource(a06)
    assert "torch.optim" not in source
    assert summary["optimizer_created"] is False
    assert all(
        per_seed[seed]["optimizer_created"] is False
        for seed in (20, 30, 50)
    )


def test_no_parameters_change(audit_result):
    _, summary, per_seed = audit_result
    assert summary["parameter_updates"] is False
    for seed in (20, 30, 50):
        result = per_seed[seed]
        assert result["parameter_updates"] is False
        assert result["parameters_unchanged_exact"]
        assert result["backbone_hash"] == result["backbone_hash_after"]


def test_manifest_candidate_schema_is_complete(audit_result):
    _, summary, _ = audit_result
    candidates = summary["manifest_candidate_entries"]
    assert [entry["model_seed"] for entry in candidates] == [20, 30, 50]
    required = {
        "condition",
        "dataset",
        "model_seed",
        "corruption_seed",
        "corruption_mode",
        "corruption_k",
        "snr_db",
        "backbone_dir",
        "checkpoint_paths",
        "source_audit",
        "expected_backbone_hash",
        "expected_z_hash_b4_compatible",
    }
    for entry in candidates:
        assert set(entry) == required
        assert entry["condition"] == "clean"
        assert entry["dataset"] == "MSRC-v1"
        assert entry["corruption_mode"] == "none"
        assert entry["corruption_seed"] is None
        assert entry["corruption_k"] is None
        assert entry["snr_db"] is None
        assert len(entry["checkpoint_paths"]) == 5


def test_four_requested_json_outputs_are_written(audit_result):
    output_dir, summary, _ = audit_result
    expected = [
        output_dir / ("clean_seed" + str(seed) + "_provenance.json")
        for seed in (20, 30, 50)
    ] + [output_dir / "clean_multiseed_provenance_summary.json"]
    assert all(path.is_file() for path in expected)
    for path in expected:
        with open(path, "r", encoding="utf-8") as input_file:
            value = json.load(input_file)
        assert value["stage"] == "B5-A0.6-preflight"
        assert value["dataset"] == "MSRC-v1"
        assert value["condition"] == "clean"
    assert summary["formal_manifest_modified"] is False


def test_final_provenance_flags_report_result_and_completion(audit_result):
    _, summary, _ = audit_result
    reference = summary["seed20_reference_checks"]
    assert summary["B5_A06_B1_B3_SEED20_EQUIVALENCE_PASS"] == (
        reference["backbone_hash_exact_match"]
        and reference["z_hash_exact_match"]
    )
    assert summary["seed20_b3_historical_reference_pass"]
    assert summary["B5_A06_PROVENANCE_DETERMINISTIC_PASS"]
    assert summary["B5_A06_CLEAN_PROVENANCE_COMPLETE"]


def test_formal_manifest_is_not_a_write_target():
    source = inspect.getsource(a06)
    manifest_path = Path("experiments/b5_semantic_rate/b5_a0_condition_manifest.json")
    assert str(manifest_path) not in source


def test_manifest_clean_seeds_are_exactly_20_30_50(manifest_value):
    clean = [entry for entry in manifest_value if entry["condition"] == "clean"]
    assert {entry["model_seed"] for entry in clean} == {20, 30, 50}
    assert len(clean) == 3


def test_each_clean_manifest_seed_has_exactly_five_checkpoints(manifest_value):
    clean = [entry for entry in manifest_value if entry["condition"] == "clean"]
    assert all(len(entry["checkpoint_paths"]) == 5 for entry in clean)


def test_all_clean_manifest_checkpoints_exist(manifest_value):
    clean = [entry for entry in manifest_value if entry["condition"] == "clean"]
    assert all(
        (ROOT / checkpoint).is_file()
        for entry in clean
        for checkpoint in entry["checkpoint_paths"]
    )


def test_all_clean_source_audits_exist(manifest_value):
    clean = [entry for entry in manifest_value if entry["condition"] == "clean"]
    assert all((ROOT / entry["source_audit"]).is_file() for entry in clean)


def test_clean_source_audit_model_seed_is_exact(manifest_value):
    clean = [entry for entry in manifest_value if entry["condition"] == "clean"]
    for entry in clean:
        source = _read_json(ROOT / entry["source_audit"])
        assert source["model_seed"] == entry["model_seed"]


def test_clean_source_audit_corruption_mode_is_none(manifest_value):
    clean = [entry for entry in manifest_value if entry["condition"] == "clean"]
    for entry in clean:
        source = _read_json(ROOT / entry["source_audit"])
        assert source["corruption_mode"] == "none"


def test_manifest_backbone_hashes_equal_a06_provenance(manifest_value):
    clean = [entry for entry in manifest_value if entry["condition"] == "clean"]
    for entry in clean:
        source = _read_json(ROOT / entry["source_audit"])
        assert entry["expected_backbone_hash"] == source["backbone_hash"]["aggregate"]


def test_manifest_z_hashes_equal_a06_provenance(manifest_value):
    clean = [entry for entry in manifest_value if entry["condition"] == "clean"]
    for entry in clean:
        source = _read_json(ROOT / entry["source_audit"])
        assert entry["expected_z_hash_b4_compatible"] == source["z_hash_b4_compatible"]


def test_formal_clean_manifest_exactly_matches_a06_candidates(manifest_value):
    summary = _read_json(A06_OUTPUT / "clean_multiseed_provenance_summary.json")
    clean = [entry for entry in manifest_value if entry["condition"] == "clean"]
    assert clean == summary["manifest_candidate_entries"]


def test_old_b3_seed20_path_is_not_a_formal_clean_entry(manifest_value):
    clean = [entry for entry in manifest_value if entry["condition"] == "clean"]
    assert all(
        entry["backbone_dir"]
        != "outputs/b3_semantic/a1_uniform_clean_seed20/models"
        for entry in clean
    )


def test_noisy_manifest_entries_are_exact_regressions(manifest_value):
    noisy = {
        entry["model_seed"]: entry
        for entry in manifest_value
        if entry["condition"] == "snr2p5_k2"
    }
    assert set(noisy) == {20, 30, 50}
    assert {
        seed: _entry_sha256(entry) for seed, entry in noisy.items()
    } == NOISY_ENTRY_SHA256


def test_loader_accepts_clean_seed20():
    assert _load_clean_entry(20)["model_seed"] == 20


def test_loader_accepts_clean_seed30():
    assert _load_clean_entry(30)["model_seed"] == 30


def test_loader_accepts_clean_seed50():
    assert _load_clean_entry(50)["model_seed"] == 50


def test_loaded_clean_backbone_hashes_match_manifest(clean_manifest_preflight):
    for seed, result in clean_manifest_preflight.items():
        entry = _load_clean_entry(seed)
        assert result["backbone_hash"] == entry["expected_backbone_hash"]


def test_loaded_clean_normalized_z_hashes_match_manifest(clean_manifest_preflight):
    for seed, result in clean_manifest_preflight.items():
        entry = _load_clean_entry(seed)
        assert result["z_hash"] == entry["expected_z_hash_b4_compatible"]
