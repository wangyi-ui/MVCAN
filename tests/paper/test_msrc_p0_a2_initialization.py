import inspect

import numpy as np
import pytest

from experiments.paper.transfer_diagnostics import (
    materialize_msrc_current_condition_init as initializer,
)
from experiments.paper.transfer_diagnostics import msrc_p0_a2_protocol as p
from release_core.backbone import MultiViewBackbone
from release_core.config import get_native_config
import release_core.runtime.entrypoint as runtime_entrypoint


def test_feature_only_initializer_reuses_exact_p0_a1_artifact():
    record = initializer.load_current_condition_features(p.P0_A1_INPUT_DIR)
    assert record["feature_sha256"] == p.FEATURE_ARTIFACT_SHA256
    assert len(record["views"]) == 5
    assert tuple(view.shape for view in record["views"]) == tuple(
        (210, size) for size in p.VIEW_DIMS
    )
    assert np.array_equal(record["sample_ids"], np.arange(210, dtype=np.int64))


def test_initializer_has_no_gt_split_utility_or_relation_dependency():
    source = inspect.getsource(initializer)
    assert "load_dataset" not in source
    assert "load_msrc_v1" not in source
    assert "build_relation_semantics" not in source
    assert "compute_directional_cycle_utility" not in source
    assert "utility_conditioned_relation_loss" not in source
    loader_source = inspect.getsource(initializer.load_current_condition_features)
    assert "msrc_sparse_split" not in loader_source
    assert "labeled_ids" not in loader_source
    assert "unlabeled_ids" not in loader_source


def test_initializer_uses_release_primitives_and_preserves_historical_loop():
    source = inspect.getsource(initializer)
    assert "MultiViewBackbone" in source
    assert "initialize_kmeans_centers" in source
    assert "native_refresh_from_latents" in source
    assert "native_objective" in source
    assert "from model import" not in source
    assert "from sklearn" not in source
    loop = inspect.getsource(initializer._run_historical_native_initialization)
    assert "range(frozen.init_epochs)" in loop
    assert "range(frozen.native_loop_iterations)" in loop
    assert "generator=generator" in loop


def test_five_checkpoint_states_roundtrip_strictly(tmp_path):
    model = MultiViewBackbone(
        get_native_config(p.DATASET), p.V, p.VIEW_DIMS,
        n_clusters=p.K, seed=p.TRAINING_SEED,
    )
    paths = tuple(tmp_path / name for name in p.INITIALIZATION_CHECKPOINT_NAMES)
    expected = runtime_entrypoint._model_sha256(model)
    actual = initializer._checkpoint_roundtrip(model, paths)
    assert actual == expected
    assert len(paths) == 5 and all(path.is_file() for path in paths)


def test_initialization_overwrite_fails_before_feature_or_training_access(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(RuntimeError, match="overwrite"):
        initializer.materialize(
            input_dir=tmp_path / "missing", output_dir=output, device="cpu",
            seed=20, init_epochs=200, native_config_epoch=1000,
            refresh_interval=100, learning_rate=1e-4,
            native_lambda1=0.01, batch_size=256,
        )
