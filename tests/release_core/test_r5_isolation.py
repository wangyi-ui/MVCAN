import ast
import hashlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from release_core.training import PhaseAudit


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = (
    ROOT / "release_core/training/__init__.py",
    ROOT / "release_core/training/alternating.py",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_imports_only_clean_runtime_dependencies():
    forbidden_import_roots = {
        "experiments", "evaluation", "argparse", "pathlib",
    }
    for path in PRODUCTION:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name.split(".")[0] in forbidden_import_roots for name in imports)
        assert all(
            not name.startswith("release_core.evaluation") for name in imports
        )


def test_forbidden_production_concepts_are_absent():
    forbidden = (
        "TRUE_U", "TRUE_UNIFORM", "SHUFFLE_U", "BASE", "U_tilde",
        "pseudo", "memory", "ACC", "NMI", "ARI", "ground_truth",
        "sklearn.metrics", "pilot", "final_core", "C3_B0", "F0",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in PRODUCTION)
    assert all(token not in source for token in forbidden)


def test_public_api_contains_required_neutral_functions():
    import release_core.training as training

    expected = {
        "build_decoupled_optimizers",
        "precompute_training_orders",
        "run_relation_refinement_phase",
        "refresh_native_state_if_due",
        "run_native_consolidation_phase",
        "run_alternating_epoch",
        "run_alternating_training",
    }
    assert expected <= set(training.__all__)
    assert "arm" not in training.__all__


def test_audit_records_are_frozen_and_small():
    audit = PhaseAudit(
        phase="x", batch_count=1, sample_count=2, sample_id_sha256="0" * 64,
        backward_count=1, optimizer_step_count=2, loss_sum=1.0,
        gradients_clean_at_end=True, encoder_gradient_path=True,
        decoder_gradient_path=False, cluster_gradient_path=True,
    )
    with pytest.raises(FrozenInstanceError):
        audit.batch_count = 2
    assert not hasattr(audit, "tensor")


def test_r1_r4_consumed_sources_remain_exact():
    expected = {
        "release_core/backbone/clustering.py": "aefb0f8f6c596721e21e75332166d877ae0e73b130cd6927831b17ddd6a12474",
        "release_core/backbone/native_objective.py": "f7756312fd5ea301e0503e354afff4f3e5de1c63bbc1aec73df6d20a124e786c",
        "release_core/action/relation.py": "813da34bb736f485cb060b86137c0775b690a4ef1df1ac90ce6782fadde390e8",
    }
    assert all(_sha256(ROOT / name) == digest for name, digest in expected.items())


def test_only_two_production_modules_exist():
    assert sorted(path.name for path in (ROOT / "release_core/training").glob("*.py")) == [
        "__init__.py", "alternating.py",
    ]
