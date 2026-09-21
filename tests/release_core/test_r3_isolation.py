import ast
import inspect
from pathlib import Path

import release_core.semantics as semantics


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_FILES = (
    ROOT / "release_core/semantics/__init__.py",
    ROOT / "release_core/semantics/sparse_labels.py",
    ROOT / "release_core/semantics/relations.py",
)
FORBIDDEN_IMPORT_ROOTS = {
    "datasets", "configure", "experiments", "weak_quality", "irv",
}
FORBIDDEN_ACTIVE_NAMES = {
    "compute_directional_cycle_utility", "Optimizer", "backward", "grad",
    "binary_cross_entropy", "pseudo_label", "semantic_memory",
    "prototype_memory", "semantic_expansion", "U_tilde",
}
FORBIDDEN_PARAMETERS = {
    "U_cycle", "full_y", "ground_truth", "unlabeled_targets",
    "evaluation_labels", "utility_weight", "utility_threshold",
}
FORBIDDEN_PHRASES = (
    "phase a", "phase b", "alternating trainer", "feature gating",
    "fusion gating", "pseudo-anchor", "utility recalibration",
)


def _trees():
    return tuple(
        (path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        for path in PRODUCTION_FILES
    )


def test_exactly_three_r3_production_files_exist():
    actual = tuple(sorted((ROOT / "release_core/semantics").glob("*.py")))
    assert actual == tuple(sorted(PRODUCTION_FILES))


def test_r3_imports_are_release_native_and_only_reuse_action_metadata():
    imports = set()
    for _, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
    assert all(name.split(".")[0] not in FORBIDDEN_IMPORT_ROOTS for name in imports)
    assert "release_core.utility.action_space" in imports
    assert "release_core.utility.cyclic" not in imports


def test_r3_has_no_later_stage_active_symbols_or_parameters():
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
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parameters = {argument.arg for argument in node.args.args}
                parameters.update(argument.arg for argument in node.args.kwonlyargs)
                assert parameters.isdisjoint(FORBIDDEN_PARAMETERS)


def test_r3_has_no_closed_route_or_later_stage_source_phrases():
    source = "\n".join(
        path.read_text(encoding="utf-8").lower() for path in PRODUCTION_FILES
    )
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in source
    assert "u_cycle" not in source
    assert "bce" not in source
    assert "optimizer" not in source


def test_public_semantics_api_is_only_sparse_targets_and_raw_relation():
    signatures = {
        name: tuple(inspect.signature(getattr(semantics, name)).parameters)
        for name in (
            "validate_sparse_label_split", "build_vote_semantic_state",
            "fit_sparse_mapping", "build_relation_semantics",
            "build_relation_balance_weights", "compute_view_relations",
        )
    }
    assert signatures == {
        "validate_sparse_label_split": ("split",),
        "build_vote_semantic_state": ("y_gen", "class_count"),
        "fit_sparse_mapping": ("vote_state", "split"),
        "build_relation_semantics": ("y_gen", "split", "actions"),
        "build_relation_balance_weights": ("pred_relation",),
        "compute_view_relations": ("q_query", "q_anchor"),
    }

