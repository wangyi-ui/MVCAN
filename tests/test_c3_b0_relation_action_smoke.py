import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.cyclic_utility import c3_b0_relation_action_protocol as c3b0
from experiments.cyclic_utility import smoke_c3_b0_relation_action_pilot as smoke
from experiments.cyclic_utility import train_c3_b0_relation_action_pilot as train


class TinySmokeModel:
    def __init__(self):
        self.autoencoders = [TinyAutoencoder() for _ in range(c3b0.VIEW_NUM)]


class TinyAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(2, 3)
        self.clustering = torch.nn.Sequential(
            torch.nn.Linear(3, c3b0.CLASS_NUM), torch.nn.Softmax(dim=1)
        )


def _gradient():
    return {"gradient_tensor_count": 1, "gradient_finite_pass": True,
            "gradient_l2": 1.0, "gradient_nonzero": True}


def _phase_a(executed=True):
    if not executed:
        return {"phase": "relation_semantic", "executed": False,
                "reason": "BASE has no Phase A"}
    return {
        "phase": "relation_semantic",
        "executed": True,
        "anchor_count": c3b0.LABEL_COUNT,
        "relation_loss_mean": 0.25,
        "all_expected_sample_ids_exactly_once_pass": True,
        "batch_records": [{
            "query_count": 200,
            "relation_loss": 0.25,
            "denominator": 8.0,
            "view_count": c3b0.VIEW_NUM,
            "gradient": _gradient(),
        }],
    }


def _phase_b():
    return {
        "phase": "native_consolidation",
        "executed": True,
        "all_expected_sample_ids_exactly_once_pass": True,
        "batch_records": [{
            "native_loss": 0.5,
            "reconstruction_loss_sum_views": 0.4,
            "clustering_loss_sum_views": 0.1,
            "gradient": _gradient(),
        }],
    }


def _optimizer_audit(arm):
    return {
        "semantic_optimizer_count": 0 if arm == "BASE" else c3b0.VIEW_NUM,
        "native_optimizer_count": c3b0.VIEW_NUM,
    }


def _optimizers(model, arm):
    native = [
        torch.optim.SGD(autoencoder.parameters(), lr=0.01)
        for autoencoder in model.autoencoders
    ]
    semantic = None if arm == "BASE" else [
        torch.optim.SGD(autoencoder.parameters(), lr=0.01)
        for autoencoder in model.autoencoders
    ]
    return semantic, native


def _true_u_shapes():
    batch = 242
    return {
        "batch_unlabeled_count": batch,
        "expected_shapes": {
            "U_batch": [batch, 20], "PredRelation": [batch, 14, 20],
            "balance": [batch, 14, 20], "q_query": [batch, 7],
            "q_anchor": [14, 7], "relation_prob": [batch, 14],
        },
        "observed_shapes": {
            "U_batch": [batch, 20], "PredRelation": [batch, 14, 20],
            "balance": [batch, 14, 20], "q_query": [batch, 7],
            "q_anchor": [14, 7], "relation_prob": [batch, 14],
        },
        "per_view_posterior_shapes": [{}] * c3b0.VIEW_NUM,
        "view_count": c3b0.VIEW_NUM,
        "sample_ID_lookup_pass": True,
        "runtime_tensor_shapes_pass": True,
        "forbidden_N_by_N_tensor_created": False,
        "forbidden_N_by_N_by_any_tensor_created": False,
    }


def test_smoke_scope_is_exactly_four_arms_seed20_and_one_epoch():
    assert smoke.SMOKE_ARMS == (
        "BASE", "TRUE_U", "TRUE_UNIFORM", "SHUFFLE_U"
    )
    assert smoke.SMOKE_ARMS == c3b0.ARMS
    assert smoke.SMOKE_SEED == 20
    assert smoke.SMOKE_EPOCH_COUNT == 1


def test_independent_smoke_output_paths_are_exact_and_not_formal_paths():
    expected = {
        "BASE": "c3_b0_relation_action_engineering_smoke_seed20_base",
        "TRUE_U": "c3_b0_relation_action_engineering_smoke_seed20_true_u",
        "TRUE_UNIFORM": (
            "c3_b0_relation_action_engineering_smoke_seed20_true_uniform"
        ),
        "SHUFFLE_U": (
            "c3_b0_relation_action_engineering_smoke_seed20_shuffle_u"
        ),
    }
    paths = {arm: smoke.default_smoke_output_dir(arm) for arm in smoke.SMOKE_ARMS}
    assert {arm: path.name for arm, path in paths.items()} == expected
    assert len({path.resolve() for path in paths.values()}) == 4
    for arm, path in paths.items():
        assert path.parent == smoke.REPOSITORY_ROOT / "outputs/cyclic_utility"
        assert path.resolve() != train.default_output_dir(20, arm).resolve()


def test_per_arm_output_schema_contains_only_engineering_json_files():
    assert smoke.OUTPUT_FILENAMES == (
        "smoke_audit.json", "smoke_runtime.json"
    )


def test_existing_smoke_output_is_hard_failure(monkeypatch, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    monkeypatch.setattr(smoke, "default_smoke_output_dir", lambda arm: existing)
    monkeypatch.setattr(
        smoke, "verify_preformal_source_hashes",
        lambda: (_ for _ in ()).throw(AssertionError("hash should follow guard")),
    )
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        smoke.run_smoke_arm("BASE", device="cpu")


def test_base_runtime_has_only_one_native_phase_and_no_semantic_activity():
    model = TinySmokeModel()
    semantic, native = _optimizers(model, "BASE")
    runtime = smoke.build_smoke_runtime(
        "BASE", _optimizer_audit("BASE"), _phase_a(False), _phase_b(),
        semantic, native,
    )
    assert runtime["C3_B0_SMOKE_PASS"]
    assert runtime["phase_a_execution_count"] == 0
    assert runtime["phase_b_execution_count"] == 1
    assert runtime["semantic_optimizer_count"] == 0
    assert runtime["native_optimizer_count"] == 6
    assert runtime["semantic_backward_count"] == 0
    assert runtime["semantic_step_count"] == 0
    assert runtime["anchor_relation_forward_count"] == 0


@pytest.mark.parametrize("arm", ("TRUE_U", "TRUE_UNIFORM", "SHUFFLE_U"))
def test_non_base_runtime_executes_phase_a_and_phase_b_with_finite_checks(arm):
    model = TinySmokeModel()
    semantic, native = _optimizers(model, arm)
    runtime = smoke.build_smoke_runtime(
        arm, _optimizer_audit(arm), _phase_a(), _phase_b(), semantic, native,
        _true_u_shapes() if arm == "TRUE_U" else None,
    )
    assert runtime["C3_B0_SMOKE_PASS"]
    assert runtime["phase_a_execution_count"] == 1
    assert runtime["phase_b_execution_count"] == 1
    assert runtime["semantic_optimizer_count"] == 6
    assert runtime["native_optimizer_count"] == 6
    assert runtime["semantic_backward_count"] > 0
    assert runtime["native_backward_count"] > 0
    assert runtime["relation_loss_finite"]
    assert runtime["native_REC_finite"]
    assert runtime["native_CLU_finite"]
    assert runtime["native_loss_finite"]
    assert runtime["gradient_finite"]
    assert runtime["parameter_finite"]
    assert runtime["denominator_finite_and_above_epsilon"]
    assert runtime["sample_id_alignment_pass"]
    assert runtime["anchor_count"] == 14
    assert runtime["view_count"] == 6


def test_non_finite_loss_and_gradient_do_not_pass():
    phase_a = _phase_a()
    phase_a["batch_records"][0]["relation_loss"] = np.nan
    phase_a["batch_records"][0]["gradient"]["gradient_finite_pass"] = False
    model = TinySmokeModel()
    semantic, native = _optimizers(model, "TRUE_U")
    runtime = smoke.build_smoke_runtime(
        "TRUE_U", _optimizer_audit("TRUE_U"), phase_a, _phase_b(),
        semantic, native, _true_u_shapes(),
    )
    assert runtime["relation_loss_finite"] is False
    assert runtime["gradient_finite"] is False
    assert runtime["C3_B0_SMOKE_PASS"] is False


def test_orchestration_container_without_parameters_api_is_supported():
    model = TinySmokeModel()
    assert not hasattr(model, "parameters")
    semantic, native = _optimizers(model, "TRUE_U")
    assert smoke._all_optimizer_parameters_finite(semantic, native) is True


def test_base_finite_checker_uses_native_optimizers_only():
    model = TinySmokeModel()
    semantic, native = _optimizers(model, "BASE")
    assert semantic is None
    assert smoke._all_optimizer_parameters_finite(semantic, native) is True


    assert smoke._all_optimizer_parameters_finite([], native) is True
def test_non_base_finite_checker_uses_semantic_and_native_union():
    model = TinySmokeModel()
    semantic, native = _optimizers(model, "TRUE_U")
    assert smoke._all_optimizer_parameters_finite(semantic, native) is True
    expected = {
        id(parameter)
        for autoencoder in model.autoencoders
        for parameter in autoencoder.parameters()
    }
    actual = smoke._unique_optimizer_parameters(semantic, native)
    assert {id(parameter) for parameter in actual} == expected


def test_optimizer_parameter_nan_returns_false():
    parameter = torch.nn.Parameter(torch.tensor([1.0, float("nan")]))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    assert smoke._all_optimizer_parameters_finite(None, [optimizer]) is False


def test_optimizer_parameter_inf_returns_false():
    parameter = torch.nn.Parameter(torch.tensor([1.0, float("inf")]))
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    assert smoke._all_optimizer_parameters_finite(None, [optimizer]) is False


def test_shared_semantic_native_parameters_are_deduplicated_by_id():
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([2.0]))
    semantic = [torch.optim.SGD([first, second], lr=0.01)]
    native = [torch.optim.SGD([second, first], lr=0.01)]
    parameters = smoke._unique_optimizer_parameters(semantic, native)
    assert [id(parameter) for parameter in parameters] == [id(first), id(second)]
    assert smoke._all_optimizer_parameters_finite(semantic, native) is True


def test_finite_checker_does_not_call_or_require_model_parameters():
    source = inspect.getsource(smoke._all_optimizer_parameters_finite)
    assert "model.parameters" not in source
    assert "model" not in inspect.signature(
        smoke._all_optimizer_parameters_finite
    ).parameters
    assert "parameter.data" in source


def test_true_u_runtime_tensor_shapes_are_observed_without_dense_pairwise(
    monkeypatch,
):
    model = TinySmokeModel()
    views = [torch.zeros((c3b0.SAMPLE_NUM, 2), dtype=torch.float32)
             for _ in range(c3b0.VIEW_NUM)]
    action_arrays = {
        "unlabeled_ids": np.arange(14, c3b0.SAMPLE_NUM, dtype=np.int64),
        "labeled_ids": np.arange(14, dtype=np.int64),
    }
    monkeypatch.setattr(c3b0, "validate_action_arrays", lambda arrays: arrays)

    def select(arrays, arm, query_ids):
        batch = len(query_ids)
        return {
            "U_cycle": np.ones((batch, 20)),
            "PredRelation": np.zeros((batch, 14, 20), dtype=np.bool_),
            "balance_weight": np.ones((batch, 14, 20)),
            "source_rows": query_ids - 14,
        }

    monkeypatch.setattr(c3b0, "action_batch_by_sample_ids", select)
    result = smoke.audit_true_u_runtime_tensor_shapes(
        model, views, action_arrays, np.arange(c3b0.SAMPLE_NUM),
        torch.device("cpu"),
    )
    batch = c3b0.BATCH_SIZE - c3b0.LABEL_COUNT
    assert result["observed_shapes"] == {
        "U_batch": [batch, 20],
        "PredRelation": [batch, 14, 20],
        "balance": [batch, 14, 20],
        "q_query": [batch, 7],
        "q_anchor": [14, 7],
        "relation_prob": [batch, 14],
    }
    assert result["view_count"] == 6
    assert result["sample_ID_lookup_pass"]
    assert result["forbidden_N_by_N_tensor_created"] is False
    assert result["forbidden_N_by_N_by_any_tensor_created"] is False


def test_sample_id_mapping_is_by_canonical_id_not_batch_position():
    unlabeled = np.arange(14, 1400, dtype=np.int64)[::-1].copy()
    query = np.array([1399, 14, 100], dtype=np.int64)
    rows = c3b0._rows_for_sample_ids(unlabeled, query)
    assert rows.tolist() == [0, 1385, 1299]
    assert np.array_equal(unlabeled[rows], query)


@pytest.mark.parametrize("arm", smoke.SMOKE_ARMS)
def test_one_epoch_core_calls_formal_phase_helpers_with_expected_counts(
    monkeypatch, arm,
):
    calls = []
    model = TinySmokeModel()
    views = [np.zeros((c3b0.SAMPLE_NUM, 2), dtype=np.float32)
             for _ in range(c3b0.VIEW_NUM)]
    sample_ids = np.arange(c3b0.SAMPLE_NUM, dtype=np.int64)

    def relation(*args, **kwargs):
        calls.append("formal_phase_a")
        return _phase_a()

    def refresh(*args, **kwargs):
        calls.append("formal_refresh")
        return (torch.zeros((1400, 7)), torch.zeros((6, 7, 7)), None,
                [1.0] * 6)

    def native(*args, **kwargs):
        calls.append("formal_phase_b")
        return _phase_b()

    monkeypatch.setattr(train, "relation_semantic_phase", relation)
    monkeypatch.setattr(train.e1_train, "refresh_native_target", refresh)
    monkeypatch.setattr(train, "native_consolidation_phase", native)
    monkeypatch.setattr(
        smoke, "audit_true_u_runtime_tensor_shapes", lambda *args: _true_u_shapes()
    )
    semantic, native_optimizers = _optimizers(model, arm)
    runtime = smoke.run_one_epoch_training_core(
        arm, model, semantic, native_optimizers,
        _optimizer_audit(arm), views, sample_ids, {}, torch.device("cpu"),
    )
    assert runtime["epoch_count"] == 1
    assert calls.count("formal_phase_a") == (0 if arm == "BASE" else 1)
    assert calls.count("formal_refresh") == 1
    assert calls.count("formal_phase_b") == 1


def test_smoke_wrapper_has_no_gt_or_scientific_evaluation_call():
    source = inspect.getsource(smoke.run_smoke_arm)
    assert "load_labels_after_predictions" not in source
    assert "evaluate_predictions" not in source
    assert "save_predictions_before_GT" not in source
    assert "run_arm(" not in source
    assert "full_gt" not in source.lower()


def test_smoke_does_not_read_or_modify_formal_gate():
    source = inspect.getsource(smoke)
    assert "summarize_multiseed_metrics" not in source
    assert "C3_B0_RELATION_ACTION_PILOT_PASS" not in source
    assert "PASS_DECISION" not in source
    assert "FAIL_DECISION" not in source


def test_repair_failure_is_classified_as_container_api_mismatch():
    assert smoke.REPAIR_FAILURE_CLASSIFICATION == (
        "ENGINEERING_SMOKE_AUDIT_CONTAINER_API_MISMATCH"
    )


def test_formal_source_hashes_remain_exactly_frozen():
    audit = smoke.verify_preformal_source_hashes()
    assert audit["all_entries_exact_match"] is True
    assert audit["entry_count"] == 4
    assert set(audit["entries"]) == {
        "experiments/cyclic_utility/c3_b0_relation_action_protocol.py",
        "experiments/cyclic_utility/train_c3_b0_relation_action_pilot.py",
        "experiments/cyclic_utility/summarize_c3_b0_relation_action_pilot.py",
        "tests/test_c3_b0_relation_action_pilot.py",
    }


def test_only_authorized_new_files_define_the_smoke():
    assert Path(smoke.__file__).resolve() == (
        smoke.REPOSITORY_ROOT
        / "experiments/cyclic_utility/smoke_c3_b0_relation_action_pilot.py"
    )
    assert Path(__file__).resolve() == (
        smoke.REPOSITORY_ROOT / "tests/test_c3_b0_relation_action_smoke.py"
    )
