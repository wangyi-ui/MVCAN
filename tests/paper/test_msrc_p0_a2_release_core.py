import hashlib

import pytest

from experiments.paper.transfer_diagnostics import msrc_p0_a2_protocol as p
from experiments.paper.transfer_diagnostics import run_msrc_p0_a2 as runner


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_release_core_matches_p0_a2_frozen_hash_manifest():
    manifest = (
        p.REPOSITORY_ROOT
        / "experiment_freeze/p0_a2_msrc_same_condition_init_preregistered_20260923/release_core_sha256.txt"
    )
    records = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(None, 1)
        records.append((expected, relative.strip()))
    assert records
    assert all(_sha256(p.REPOSITORY_ROOT / relative) == expected
               for expected, relative in records)


def test_a2_arm_output_overwrite_fails_before_input_or_training(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(p, "OUTPUT_ROOT", tmp_path / "a2")
    output = p.default_output_dir("BASE")
    output.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="overwrite"):
        runner.run_arm(
            arm="BASE",
            input_dir=tmp_path / "missing-input",
            init_dir=tmp_path / "missing-init",
            output_dir=output,
            device="cpu",
            training_seed=20,
            epochs=20,
            batch_size=256,
            learning_rate=1e-4,
            refresh_interval=100,
        )
