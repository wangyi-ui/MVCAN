from types import SimpleNamespace

import pytest
import json
import sys
import inspect
import subprocess

from experiments.paper.diagnostics import msrc_native_stagewise_parity as stagewise


def test_runner_uses_equal_frozen_current_and_historical_seed20():
    source = inspect.getsource(stagewise)
    assert "p0_a3_protocol.TRAINING_SEED" not in source
    assert stagewise.current_initializer.protocol.TRAINING_SEED == 20

    command = (
        "import json; "
        "from experiments.paper.diagnostics import "
        "msrc_native_stagewise_parity as s; "
        "from experiments.paper.diagnostics.msrc_native_initializer_parity_replay "
        "import load_frozen_legacy_adapter; "
        "print(json.dumps(s.validate_frozen_training_seeds("
        "load_frozen_legacy_adapter())))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=stagewise.current_initializer.protocol.REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    current_seed, historical_seed = json.loads(completed.stdout)
    assert current_seed == 20
    assert historical_seed == 20
    assert current_seed == historical_seed


def test_wrapper_repair_does_not_modify_release_core():
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "release_core"],
        cwd=stagewise.current_initializer.protocol.REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""

def test_frozen_seed_validation_fails_closed_on_either_mismatch(monkeypatch):
    bad_historical = SimpleNamespace(
        MSRC_RUNTIME_SPEC=SimpleNamespace(
            expected_config={"training": {"seed": 21}}
        )
    )
    with pytest.raises(RuntimeError, match="historical training seed"):
        stagewise.validate_frozen_training_seeds(bad_historical)

    good_historical = SimpleNamespace(
        MSRC_RUNTIME_SPEC=SimpleNamespace(
            expected_config={"training": {"seed": 20}}
        )
    )
    monkeypatch.setattr(stagewise.current_initializer.protocol, "TRAINING_SEED", 21)
    with pytest.raises(RuntimeError, match="current P0-A2 training seed"):
        stagewise.validate_frozen_training_seeds(good_historical)
