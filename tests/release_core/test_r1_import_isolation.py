import ast
from pathlib import Path

FORBIDDEN_IMPORTS = {
    "model", "datasets", "configure", "weak_quality", "ClusteringTest",
    "TSNE", "matplotlib", "generic_cycle_utility", "action_space",
    "sparse_label_contract", "generic_relation_action", "final_core",
}
FORBIDDEN_SYMBOLS = {
    "U_cycle", "PredRelation", "relation_balance_weights", "pseudo_label",
    "memory", "semantic_ce", "utility_threshold", "utility_temperature",
    "utility_exponent", "relation_coefficient",
}


def _production_files():
    root = Path(__file__).resolve().parents[2] / "release_core"
    return sorted(root.rglob("*.py"))


def test_production_imports_are_isolated():
    for path in _production_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module]
            for name in imported:
                assert not name.startswith(("experiments", "irv"))
                assert name.split(".")[0] not in FORBIDDEN_IMPORTS


def test_later_stage_symbols_are_absent_from_executable_ast():
    for path in _production_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        executable_names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        executable_names.update(
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        assert executable_names.isdisjoint(FORBIDDEN_SYMBOLS)

