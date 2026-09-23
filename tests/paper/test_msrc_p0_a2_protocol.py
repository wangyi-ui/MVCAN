import pytest

from experiments.paper.transfer_diagnostics import msrc_p0_a2_protocol as p


def test_p0_a2_primary_condition_and_input_identity_are_exact():
    assert (p.DATASET, p.N, p.V, p.K, p.VIEW_DIMS) == (
        "MSRC-v1", 210, 5, 7, (24, 576, 512, 256, 254)
    )
    assert p.DATASET_SHA256 == "38d89aa41ae984f0b026f4baa8b5dcb7c569c471e2fb42b73a0f6743398e9103"
    assert p.FEATURE_ARTIFACT_SHA256 == "2a225229de5b4c4d87ac530860691c351a0e8e3c1a015f1ecd290673f663755d"
    assert p.CORRUPTION_MASK_SHA256 == "d64f5fbf95518e5a77e44a1fe18a15dee9924b8c2a341e930f86f20ce5ac9e9d"
    assert p.SPLIT_SHA256 == "4f16e8b1b8213591a7ea1d36115335ccee43bf568d32bbcb007f929bc9b15231"


def test_historical_native_schedule_is_200_then_1001_with_11_refreshes():
    assert p.NATIVE_INIT_EPOCHS == 200
    assert p.NATIVE_CONFIG_EPOCH == 1000
    assert p.NATIVE_LOOP_ITERATIONS == 1001
    assert p.NATIVE_REFRESH_INTERVAL == 100
    assert p.NATIVE_REFRESH_EPOCHS == tuple(range(0, 1001, 100))
    assert p.NATIVE_REFRESH_COUNT == 11
    assert (p.KMEANS_N_INIT, p.KMEANS_RANDOM_STATE) == (100, 20)


def test_p0_a2_paths_are_disjoint_from_preserved_p0_a1():
    assert p.OUTPUT_ROOT != p.P0_A1_ROOT
    assert p.P0_A1_ROOT not in p.OUTPUT_ROOT.parents
    assert p.OUTPUT_ROOT not in p.P0_A1_ROOT.parents
    assert p.INITIALIZATION_DIR.parent == p.OUTPUT_ROOT
    assert len(set(p.default_output_dir(arm) for arm in p.ARMS)) == 3


def test_initialization_and_arm_tuning_fail_closed():
    p.validate_initialization_values(
        seed=20, init_epochs=200, native_config_epoch=1000,
        refresh_interval=100, learning_rate=1e-4,
        native_lambda1=0.01, batch_size=256,
    )
    with pytest.raises(ValueError, match="frozen"):
        p.validate_initialization_values(
            seed=20, init_epochs=199, native_config_epoch=1000,
            refresh_interval=100, learning_rate=1e-4,
            native_lambda1=0.01, batch_size=256,
        )
    with pytest.raises(ValueError, match="frozen"):
        p.validate_arm_training_values(
            training_seed=20, epochs=20, batch_size=256,
            learning_rate=2e-4, refresh_interval=100,
        )
