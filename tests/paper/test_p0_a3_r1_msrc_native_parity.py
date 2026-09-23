import inspect

import numpy as np

from experiments.paper.diagnostics import msrc_native_initializer_parity_replay as replay


def _input(value=1.0):
    sample_ids = np.arange(4, dtype=np.int64)
    split = {
        "labeled_ids": np.asarray([0, 1], dtype=np.int64),
        "labeled_targets": np.asarray([0, 1], dtype=np.int64),
        "unlabeled_ids": np.asarray([2, 3], dtype=np.int64),
    }
    return {
        "sample_ids": sample_ids,
        "views": tuple(
            np.full((4, width), value, dtype=np.float32) for width in range(1, 6)
        ),
        "split": split,
    }


def test_scientific_input_comparison_is_array_based_and_fail_closed():
    current, historical = _input(), _input()
    mask = np.asarray([
        [1, 1, 0, 0, 0], [0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0], [0, 1, 0, 1, 0],
    ], dtype=np.bool_)
    records, equal = replay.compare_input_arrays(
        current, historical, mask, mask.copy()
    )
    assert equal is True
    assert records["view_1"]["array_equal"] is True
    historical["views"][0][0, 0] += np.float32(1e-3)
    _, equal = replay.compare_input_arrays(current, historical, mask, mask.copy())
    assert equal is False


def test_all_five_preregistered_a6_decisions():
    classify = replay.classify_a6
    assert classify(inputs_equal=False, historical_replay_exact=False,
                    state_equal=False, r2_cross_equal=False,
                    r3_cross_equal=False) == "INPUT_MATERIALIZATION_DIVERGENCE"
    assert classify(inputs_equal=True, historical_replay_exact=False,
                    state_equal=False, r2_cross_equal=False,
                    r3_cross_equal=False).startswith("HISTORICAL_REPLAY_")
    assert classify(inputs_equal=True, historical_replay_exact=True,
                    state_equal=True, r2_cross_equal=False,
                    r3_cross_equal=False) == "R2_MIGRATION_DIVERGENCE"
    assert classify(inputs_equal=True, historical_replay_exact=True,
                    state_equal=False, r2_cross_equal=True,
                    r3_cross_equal=False) == "R3_MIGRATION_DIVERGENCE"
    assert classify(inputs_equal=True, historical_replay_exact=True,
                    state_equal=False, r2_cross_equal=True,
                    r3_cross_equal=True) == "NATIVE_INITIALIZATION_MIGRATION_DIVERGENCE"


def test_a6_wraps_original_legacy_initializer_and_has_no_gt_boundary():
    source = inspect.getsource(replay.run_replay)
    assert "legacy.prepare_native_backbone(" in source
    assert "torch.optim" not in source
    assert "DataLoader" not in source
    assert "load_dataset" not in inspect.getsource(replay)
    assert "evaluate_postseal" not in inspect.getsource(replay)

