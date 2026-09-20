import ast
import inspect
from pathlib import Path

import release_core.utility as utility


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_FILES = (
    ROOT / "release_core/utility/__init__.py",
    ROOT / "release_core/utility/action_space.py",
    ROOT / "release_core/utility/cyclic.py",
)
FORBIDDEN_IMPORT_ROOTS = {
    "datasets", "model", "configure", "weak_quality", "ClusteringTest",
    "experiments", "irv",
}
FORBIDDEN_ACTIVE_NAMES = {
    "Parameter", "Optimizer", "backward", "Autoencoder", "PredRelation",
    "relation_balance_weights", "pseudo_label", "semantic_memory",
    "prototype_memory", "alternating_trainer", "U_tilde",
}
FORBIDDEN_SOURCE_PHRASES = (
    "dataset loading", "weak-quality corruption", "native refresh",
    "alignment estimation", "phase a", "phase b", "feature gating",
    "fusion gating", "pseudo ce", "pseudo-anchor expansion",
    "continuous u-weighted posterior aggregation", "scalar utility correction",
)


def _trees():
    return tuple(
        (path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        for path in PRODUCTION_FILES
    )


def test_r2_production_imports_are_clean_and_release_native():
    for _, tree in _trees():
        for node in ast.walk(tree):
            imported = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module]
            for name in imported:
                assert name.split(".")[0] not in FORBIDDEN_IMPORT_ROOTS


def test_r2_has_no_training_or_later_stage_executable_symbols():
    for _, tree in _trees():
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names.update(
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        )
        names.update(
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        assert names.isdisjoint(FORBIDDEN_ACTIVE_NAMES)


def test_r2_has_no_closed_route_or_later_stage_source_phrases():
    source = "\n".join(
        path.read_text(encoding="utf-8").lower() for path in PRODUCTION_FILES
    )
    for phrase in FORBIDDEN_SOURCE_PHRASES:
        assert phrase not in source


def test_utility_api_accepts_only_aligned_posteriors_and_actions():
    signature = inspect.signature(utility.compute_directional_cycle_utility)
    assert tuple(signature.parameters) == ("q_aligned", "actions")
    assert signature.parameters["actions"].default is None


def test_exactly_three_r2_production_files_exist():
    actual = tuple(sorted((ROOT / "release_core/utility").glob("*.py")))
    assert actual == tuple(sorted(PRODUCTION_FILES))
