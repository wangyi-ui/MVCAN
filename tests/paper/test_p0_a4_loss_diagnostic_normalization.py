import copy

import numpy as np

from experiments.paper.diagnostics.msrc_native_stagewise_parity import (
    STAGES,
    canonical_native_total,
    classify_first_divergence,
)


RECONSTRUCTION = np.float32(2303.939697265625)
CLUSTERING = np.float32(0.038957979530096054)
LAMBDA1 = 0.01


def _trace(raw_total, canonical_total):
    losses = {
        "reconstruction": float(RECONSTRUCTION),
        "clustering": float(CLUSTERING),
        "canonical_total": canonical_total,
        "raw_total_tensor_value": raw_total,
    }
    return {
        "snapshots": [
            {
                "stage": stage,
                "model": {"aggregate_hash": "model"},
                "optimizer": {"aggregate_hash": "optimizer"},
                "rng": {"dataloader_generator_state_hash": "generator"},
                "representation": {"latent_aggregate_hash": "latent"},
                "native_update": [{"losses": copy.deepcopy(losses)}],
            }
            for stage in STAGES
        ]
    }


def test_python_component_total_can_differ_from_float32_tensor_style_total():
    component_total = float(RECONSTRUCTION) + LAMBDA1 * float(CLUSTERING)
    tensor_style_total = float(np.float32(RECONSTRUCTION + LAMBDA1 * CLUSTERING))
    assert component_total == 2303.9400868454204
    assert tensor_style_total == 2303.940185546875
    assert component_total != tensor_style_total


def test_both_paths_use_the_same_canonical_total_helper():
    historical = canonical_native_total(RECONSTRUCTION, CLUSTERING, LAMBDA1)
    current = canonical_native_total(
        float(RECONSTRUCTION), float(CLUSTERING), float(LAMBDA1)
    )
    assert historical == current == 2303.9400868454204


def test_raw_total_is_diagnostic_only_but_canonical_total_is_scientific():
    canonical = canonical_native_total(RECONSTRUCTION, CLUSTERING, LAMBDA1)
    historical = _trace(canonical, canonical)
    current = _trace(float(np.float32(canonical)), canonical)
    result = classify_first_divergence(historical, current)
    assert result["decision"] == "EXACT_INITIALIZATION_PARITY"

    current["snapshots"][7]["native_update"][0]["losses"][
        "canonical_total"
    ] = canonical + 1.0
    result = classify_first_divergence(historical, current)
    assert result["decision"] == "NATIVE_UPDATE_ORCHESTRATION_DIVERGENCE"
    assert result["first_divergent_stage"] == "S7_POST_NATIVE_0"
    assert result["first_divergent_paths"] == [
        "native_update[0].losses.canonical_total"
    ]
