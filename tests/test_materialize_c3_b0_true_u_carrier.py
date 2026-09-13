import hashlib
import inspect
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.cyclic_utility import c3_b0_true_u_carrier_protocol as carrier
from experiments.cyclic_utility import materialize_c3_b0_true_u_carrier as runner
from experiments.cyclic_utility import train_c3_b0_relation_action_pilot as c3_train


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def identity_matches():
    return torch.eye(carrier.K, dtype=torch.float64).repeat(carrier.V, 1, 1)


@pytest.fixture(scope="module")
def q_local():
    classes = torch.arange(carrier.N) % carrier.K
    q = torch.nn.functional.one_hot(classes, num_classes=carrier.K)
    return q.to(torch.float64)[:, None, :].repeat(1, carrier.V, 1)


def _model_hash(value="same"):
    return {"aggregate": value, "per_view": [value] * carrier.V}


def _valid_audit(seed=20):
    return {
        "seed": seed,
        "all_parent_hashes_pass": True,
        "final_model_hash_equal": True,
        "final_predictions_equal": True,
        "final_prediction_sample_ids_equal": True,
        "coordinate_mapping_pass": True,
        "GT_loaded_for_carrier_materialization": False,
        "C4_used": False,
    }


def test_01_q_local_to_q_aligned_shape(q_local, identity_matches):
    output = carrier.align_q_readonly(q_local, identity_matches)
    assert output.shape == (carrier.N, carrier.V, carrier.K)
    assert torch.equal(output, q_local)
    assert not output.requires_grad


def test_02_mapping_uses_M_transpose(q_local, identity_matches):
    mapping = identity_matches.clone()
    mapping[0] = torch.eye(carrier.K, dtype=torch.float64)[
        [1, 2, 0, 3, 4, 5, 6]
    ]
    output = carrier.align_q_readonly(q_local, mapping)
    assert torch.equal(output[:, 0], q_local[:, 0] @ mapping[0].T)
    assert not torch.equal(output[:, 0], q_local[:, 0] @ mapping[0])


def test_03_permutation_validation_passes(identity_matches):
    audit = carrier.validate_permutation_matrices(identity_matches)
    assert audit["shape"] == [6, 7, 7]
    assert audit["all_permutation_checks_pass"] is True


@pytest.mark.parametrize("kind", ["shape", "duplicate", "fractional", "nan"])
def test_04_invalid_permutation_fails_closed(identity_matches, kind):
    mapping = identity_matches.clone()
    if kind == "shape":
        mapping = mapping[:5]
    elif kind == "duplicate":
        mapping[0, 1] = mapping[0, 0]
    elif kind == "fractional":
        mapping[0, 0, 0] = 0.5
    else:
        mapping[0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="COORDINATE_MAPPING_FAIL_CLOSED"):
        carrier.validate_permutation_matrices(mapping)


def test_05_sample_ids_must_be_exact_arange():
    valid = np.arange(carrier.N, dtype=np.int64)
    assert np.array_equal(carrier.validate_sample_ids(valid), valid)
    invalid = valid.copy()
    invalid[[0, 1]] = invalid[[1, 0]]
    with pytest.raises(RuntimeError, match="SAMPLE_IDS_FAIL_CLOSED"):
        carrier.validate_sample_ids(invalid)


def test_06_same_snapshot_metadata_validation():
    audit = carrier.validate_snapshot_metadata(
        carrier.SNAPSHOT_STAGE, carrier.SNAPSHOT_EPOCH, carrier.SNAPSHOT_POSITION
    )
    assert audit["q_and_M_v_same_refresh_lifetime"] is True
    with pytest.raises(RuntimeError, match="SNAPSHOT_MISMATCH_FAIL_CLOSED"):
        carrier.validate_snapshot_metadata(
            carrier.SNAPSHOT_STAGE, 19, carrier.SNAPSHOT_POSITION
        )


def test_07_model_hash_mismatch_fails_closed():
    ids = np.arange(carrier.N)
    pred = np.zeros(carrier.N, dtype=np.int64)
    with pytest.raises(RuntimeError, match="REPLAY_PARITY_FAIL_CLOSED"):
        carrier.validate_replay_parity(
            _model_hash("expected"), _model_hash("actual"),
            pred, pred, ids, ids,
        )


def test_08_prediction_mismatch_fails_closed():
    ids = np.arange(carrier.N)
    expected = np.zeros(carrier.N, dtype=np.int64)
    actual = expected.copy()
    actual[0] = 1
    with pytest.raises(RuntimeError, match="REPLAY_PARITY_FAIL_CLOSED"):
        carrier.validate_replay_parity(
            _model_hash(), _model_hash(), expected, actual, ids, ids,
        )


def test_09_prediction_sample_ids_mismatch_fails_closed():
    ids = np.arange(carrier.N)
    wrong = ids.copy()
    wrong[[0, 1]] = wrong[[1, 0]]
    pred = np.zeros(carrier.N, dtype=np.int64)
    with pytest.raises(RuntimeError, match="REPLAY_PARITY_FAIL_CLOSED"):
        carrier.validate_replay_parity(
            _model_hash(), _model_hash(), pred, pred, ids, wrong,
        )


def test_10_replay_parity_pass_record_is_exact():
    ids = np.arange(carrier.N)
    pred = ids % carrier.K
    audit = carrier.validate_replay_parity(
        _model_hash(), _model_hash(), pred, pred.copy(), ids, ids.copy()
    )
    assert audit["final_model_hash_equal"] is True
    assert audit["final_predictions_equal"] is True
    assert audit["final_prediction_sample_ids_equal"] is True


def test_11_parent_hash_mismatch_fails_closed():
    good = {"all_parent_hashes_pass": True}
    manifest = {"all_entries_exact_match": True}
    with pytest.raises(RuntimeError, match="PARENT_HASH_FAIL_CLOSED"):
        carrier.validate_parent_hash_audits(
            good, manifest, {"all_entries_exact_match": False}
        )


def test_12_C4_is_forbidden_as_parent():
    with pytest.raises(RuntimeError, match="PARENT_FAIL_CLOSED"):
        carrier.validate_scientific_parent("C4-A0")
    assert carrier.validate_scientific_parent("C3-B0")["C4_used"] is False
    source = inspect.getsource(runner)
    assert "import c4_" not in source
    assert "from experiments.cyclic_utility import c4_" not in source


def test_13_full_GT_loader_is_absent_from_materializer():
    source = inspect.getsource(runner)
    assert "load_labels_after_predictions" not in source
    assert "DEFAULT_FULL_GT_PATH" not in source
    assert "evaluate_predictions" not in source


def test_14_artifact_key_whitelist_forbids_GT():
    valid = OrderedDict((
        ("sample_ids", np.arange(carrier.N)),
        ("q_local", np.empty((carrier.N, carrier.V, carrier.K))),
        ("q_aligned", np.empty((carrier.N, carrier.V, carrier.K))),
        ("M_v", np.empty((carrier.V, carrier.K, carrier.K))),
    ))
    assert carrier.validate_artifact_keys(valid)
    invalid = OrderedDict(valid)
    invalid["full_GT"] = np.empty(carrier.N)
    with pytest.raises(RuntimeError, match="WHITELIST_FAIL_CLOSED"):
        carrier.validate_artifact_keys(invalid)


def test_15_invalid_q_shape_fails_closed(identity_matches):
    q = torch.zeros((carrier.N - 1, carrier.V, carrier.K))
    with pytest.raises(RuntimeError, match="Q_SHAPE_OR_FINITE_FAIL_CLOSED"):
        carrier.align_q_readonly(q, identity_matches.float())


def test_16_nonfinite_q_fails_closed(q_local, identity_matches):
    q = q_local.clone()
    q[0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="Q_SHAPE_OR_FINITE_FAIL_CLOSED"):
        carrier.align_q_readonly(q, identity_matches)


def test_17_coordinate_snapshot_requires_exact_mapping(q_local, identity_matches):
    aligned = carrier.align_q_readonly(q_local, identity_matches)
    bad = aligned.clone()
    bad[0, 0] = torch.roll(bad[0, 0], 1)
    with pytest.raises(RuntimeError, match="COORDINATE_MAPPING_FAIL_CLOSED"):
        carrier.validate_coordinate_snapshot(q_local, bad.detach(), identity_matches)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("all_parent_hashes_pass", False),
        ("final_model_hash_equal", False),
        ("final_predictions_equal", False),
        ("final_prediction_sample_ids_equal", False),
        ("coordinate_mapping_pass", False),
        ("GT_loaded_for_carrier_materialization", True),
        ("C4_used", True),
    ],
)
def test_18_seal_requires_every_parity_and_safety_gate(tmp_path, field, bad_value):
    artifact = tmp_path / "carrier.npz"
    audit_path = tmp_path / "audit.json"
    artifact.write_bytes(b"artifact")
    audit_path.write_text("{}", encoding="utf-8")
    audit = _valid_audit()
    audit[field] = bad_value
    with pytest.raises(RuntimeError, match="SEAL_FAIL_CLOSED"):
        carrier.build_valid_seal(audit, artifact, audit_path)


def test_19_valid_seal_is_downstream_readonly(tmp_path):
    artifact = tmp_path / "carrier.npz"
    audit_path = tmp_path / "audit.json"
    artifact.write_bytes(b"artifact")
    audit_path.write_text(json.dumps(_valid_audit()), encoding="utf-8")
    seal = carrier.build_valid_seal(_valid_audit(), artifact, audit_path)
    assert seal["carrier_valid_for_downstream_readonly_use"] is True
    assert seal["GT_loaded_for_carrier_materialization"] is False
    assert seal["C4_used"] is False


def test_20_output_directory_created_only_after_parity():
    source = inspect.getsource(runner.materialize)
    parity_position = source.index("carrier.validate_replay_parity(")
    mkdir_position = source.index("target.mkdir(")
    assert parity_position < mkdir_position


def test_21_snapshot_matches_frozen_C3_B0_control_flow():
    source = inspect.getsource(c3_train.train_relation_action_arm)
    phase_b = source.index("native_consolidation_phase(")
    final_refresh = source.rindex("e1_train.refresh_native_target(")
    assert phase_b < final_refresh
    assert "return predictions" in source[final_refresh:]
    assert carrier.SNAPSHOT_EPOCH == 20
    assert "after_epoch20_Phase_B" in carrier.SNAPSHOT_POSITION
    assert "after_final_native_refresh" in carrier.SNAPSHOT_POSITION
    assert "before_final_prediction_seal" in carrier.SNAPSHOT_POSITION


def test_22_runner_uses_exact_frozen_training_helpers():
    source = inspect.getsource(runner.replay_frozen_true_u)
    assert "c3_train.relation_semantic_phase(" in source
    assert "c3_train.native_consolidation_phase(" in source
    assert source.count("e1_train.refresh_native_target(") == 2
    assert "carrier.FORMAL_EPOCHS" in source
    assert 'carrier.CARRIER_ARM' in source


def test_23_seed_specific_E1_paths_and_lineage():
    for seed in carrier.SEEDS:
        model_dir, audit = runner.seed_specific_e1_paths(seed)
        assert model_dir.parent.name == "lwc_100ep_seed" + str(seed)
        assert model_dir.name == "models"
        assert audit == model_dir.parent / "e1_audit.json"
        lineage = runner.validate_seed_lineage(seed, {
            "source_seed": seed,
            "source_training_seed": seed,
            "requested_seed_lineage_match_pass": True,
            "model_matches_own_audit_pass": True,
        }, {"seed": seed})
        assert lineage["all_seed_lineage_equal_pass"] is True


def test_24_frozen_parent_sources_unchanged():
    expected = {
        "experiments/cyclic_utility/c0_complementary_semantic_verification.py":
            "d52fbf0816557ba57a11fc490ead1a26598b35e68bac7808b7077c448bc7a4a9",
        "experiments/cyclic_utility/c3_a0_utility_conditioned_action_granularity_protocol.py":
            "8182370acd4cda507bfa425a3f40e05779945e49359d0b6c3c80b88472cb1115",
        "experiments/cyclic_utility/c3_b0_relation_action_protocol.py":
            "9ae55b93c2803d8dea594fed1cf3f26e32e8963779a87ee5027cdf9ae3340b1d",
        "experiments/cyclic_utility/train_c3_b0_relation_action_pilot.py":
            "0a657cb6e5faeaf21478c6bb7bd626bbd8385343408185d3730eb4b366aa8d50",
    }
    for relative, digest in expected.items():
        actual = hashlib.sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest()
        assert actual == digest


def test_25_cli_requires_one_seed_and_has_no_GT_argument():
    args = runner.parse_args(["--seed", "20", "--output-dir", "out"])
    assert args.seed == 20 and args.output_dir == "out"
    assert not hasattr(args, "full_gt_path")
    with pytest.raises(SystemExit):
        runner.parse_args([])
