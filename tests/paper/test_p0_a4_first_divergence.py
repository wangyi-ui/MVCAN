import copy

from experiments.paper.diagnostics.msrc_native_stagewise_parity import (
    ALLOWED_DECISIONS,
    STAGES,
    classify_first_divergence,
    static_audit_table,
)


def _trace():
    return {
        "snapshots": [
            {
                "stage": stage,
                "model": {"aggregate_hash": "model"},
                "optimizer": {"aggregate_hash": "optimizer"},
                "rng": {
                    "numpy_rng_hash": "numpy",
                    "torch_cpu_rng_hash": "cpu",
                    "torch_cuda_rng_hash": "cuda",
                    "dataloader_generator_state_hash": "loader",
                    "python_random_hash": "excluded-from-scientific-comparison",
                },
                "native_refresh": {"p_all_hash": "refresh"},
                "representation": {
                    "latent_aggregate_hash": "latent",
                    "q_local_aggregate_hash": "q",
                },
            }
            for stage in STAGES
        ]
    }


def _mutate(stage, key_path):
    left, right = _trace(), _trace()
    target = right["snapshots"][STAGES.index(stage)]
    for key in key_path[:-1]:
        target = target[key]
    target[key_path[-1]] = "different"
    return classify_first_divergence(left, right)


def test_exact_and_each_stage_family_use_only_preregistered_decisions():
    exact = classify_first_divergence(_trace(), _trace())
    assert exact == {
        "decision": "EXACT_INITIALIZATION_PARITY",
        "last_exact_stage": STAGES[-1],
        "first_divergent_stage": None,
        "first_divergent_components": [],
        "first_divergent_paths": [],
    }
    cases = (
        (STAGES[0], ("model", "aggregate_hash"), "MODEL_CONSTRUCTION_DIVERGENCE"),
        (STAGES[1], ("model", "aggregate_hash"), "AE_OPTIMIZATION_DIVERGENCE"),
        (STAGES[5], ("model", "aggregate_hash"), "KMEANS_INITIALIZATION_DIVERGENCE"),
        (STAGES[6], ("model", "aggregate_hash"), "NATIVE_REFRESH_DIVERGENCE"),
        (STAGES[7], ("model", "aggregate_hash"), "NATIVE_UPDATE_ORCHESTRATION_DIVERGENCE"),
        (STAGES[12], ("native_refresh", "p_all_hash"), "NATIVE_REFRESH_DIVERGENCE"),
        (STAGES[0], ("rng", "numpy_rng_hash"), "NUMERICAL_NONDETERMINISM"),
    )
    for stage, path, expected in cases:
        result = _mutate(stage, path)
        assert result["decision"] == expected
        assert result["decision"] in ALLOWED_DECISIONS
        assert result["first_divergent_stage"] == stage


def test_optimizer_continuity_requires_exact_ae_model_and_native0_model_drift():
    left, right = _trace(), _trace()
    right["snapshots"][4]["optimizer"]["aggregate_hash"] = "different"
    right["snapshots"][7]["model"]["aggregate_hash"] = "different"
    result = classify_first_divergence(left, right)
    assert result["first_divergent_stage"] == STAGES[4]
    assert result["decision"] == "OPTIMIZER_STATE_CONTINUITY_DIVERGENCE"
    assert result["first_divergent_components"] == ["optimizer_state"]



def test_dataloader_difference_has_priority_and_python_rng_is_diagnostic_only():
    left, right = _trace(), _trace()
    right["snapshots"][1]["rng"]["dataloader_generator_state_hash"] = "different"
    result = classify_first_divergence(left, right)
    assert result["decision"] == "DATALOADER_RNG_TRAJECTORY_DIVERGENCE"
    assert result["first_divergent_components"] == ["dataloader_generator"]

    right = copy.deepcopy(left)
    right["snapshots"][0]["rng"]["python_random_hash"] = "different"
    assert classify_first_divergence(left, right)["decision"] == "EXACT_INITIALIZATION_PARITY"


def test_static_audit_is_complete_a_through_h():
    rows = static_audit_table()
    assert [row["id"] for row in rows] == list("ABCDEFGH")
    assert rows[2]["status"] == "NO_OBVIOUS_DIFFERENCE"
    assert rows[3]["status"] == "NO_OBVIOUS_DIFFERENCE"
