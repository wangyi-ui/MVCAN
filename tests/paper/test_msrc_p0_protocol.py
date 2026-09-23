import json

import numpy as np
import pytest

from experiments.paper.transfer_audit import msrc_p0_protocol as p
from release_core.utility import action_count, build_directional_actions


def test_msrc_frozen_protocol_constants_are_exact():
    assert (p.DATASET, p.N, p.V, p.K, p.VIEW_DIMS) == (
        "MSRC-v1", 210, 5, 7, (24, 576, 512, 256, 254)
    )
    assert (p.LABELS_PER_CLASS, p.L, p.N_U, p.ACTION_COUNT) == (2, 14, 196, 20)
    assert (p.LABEL_SEED, p.WEAK_QUALITY_SEED, p.TRAINING_SEED) == (20, 20, 20)
    assert (p.SNR_DB, p.EPOCHS, p.BATCH_SIZE, p.LEARNING_RATE, p.REFRESH_INTERVAL) == (
        2.5, 20, 256, 1e-4, 100
    )
    assert p.DATASET_SHA256 == "38d89aa41ae984f0b026f4baa8b5dcb7c569c471e2fb42b73a0f6743398e9103"
    assert p.CORRUPTION_MASK_SHA256 == "d64f5fbf95518e5a77e44a1fe18a15dee9924b8c2a341e930f86f20ce5ac9e9d"
    assert p.SPLIT_SHA256 == "4f16e8b1b8213591a7ea1d36115335ccee43bf568d32bbcb007f929bc9b15231"
    assert p.LABELED_IDS == (17, 20, 30, 42, 60, 81, 96, 107, 139, 142, 153, 175, 182, 201)
    assert p.LABELED_TARGETS == (0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6)


def test_v5_frozen_action_space_has_twenty_ordered_actions():
    actions = build_directional_actions(5)
    assert action_count(5) == actions.S == 20
    assert actions.generators[:2] == ((0, 1), (0, 2))
    assert actions.generators[10:12] == ((0, 1, 2), (0, 1, 3))


def test_training_configuration_rejects_auto_tuning():
    p.validate_frozen_training_values(
        training_seed=20, epochs=20, batch_size=256,
        learning_rate=1e-4, refresh_interval=100,
    )
    with pytest.raises(ValueError, match="frozen"):
        p.validate_frozen_training_values(
            training_seed=20, epochs=21, batch_size=256,
            learning_rate=1e-4, refresh_interval=100,
        )


def test_checkpoint_reference_is_five_view_and_exact():
    record = json.loads(p.CHECKPOINT_AUDIT.read_text(encoding="utf-8"))
    assert record["checkpoint_count"] == 5
    assert tuple(record["checkpoint_sha256"]) == p.CHECKPOINT_SHA256
    assert record["release_runtime_initial_model_sha256"] == p.INITIAL_MODEL_SHA256
    assert record["source_condition"] == "snr2p5_k2_seed20"
    assert record["source_is_transfer_initialization_only"] is True
    assert record["source_mask_is_new_half_mask"] is False
