from pathlib import Path

from experiments.paper.diagnostics import msrc_pre_r2_carrier_temporal_alignment as audit


def test_p0_a5_wrapper_has_no_full_gt_loader_or_metric_entrypoint():
    text = Path(audit.__file__).read_text(encoding="utf-8").lower()
    forbidden = ("ground_truth", "load_full_gt", "evaluate_clustering", "acc_score")
    assert not any(token in text for token in forbidden)
    assert "full_gt_loaded\": false" in text


def test_p0_a5_does_not_own_release_or_historical_scientific_sources():
    text = Path(audit.__file__).read_text(encoding="utf-8")
    assert "native_refresh_from_latents" in text
    assert "load_frozen_legacy_adapter" in text
    assert "compute_directional_cycle_utility" in text
