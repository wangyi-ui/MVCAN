import ast
import hashlib
import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import release_core.action as action


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_FILES = (
    ROOT / "release_core/action/__init__.py",
    ROOT / "release_core/action/relation.py",
)
TEST_FILES = (
    ROOT / "tests/release_core/test_r4_relation_action.py",
    ROOT / "tests/release_core/test_r4_generic_parity.py",
    ROOT / "tests/release_core/test_r4_frozen_objective_parity.py",
    ROOT / "tests/release_core/test_r4_isolation.py",
)
FORBIDDEN_ACTIVE_NAMES = {
    "Optimizer", "Adam", "SGD", "zero_grad", "backward", "grad", "step",
    "DataLoader", "REC", "CLU", "P_global", "P_local", "Match",
    "refresh_native_target", "pseudo_label", "memory_bank", "prototype",
    "semantic_expansion", "feature_gate", "latent_gate", "posterior_gate",
    "view_gate", "fusion_gate", "threshold", "top_k", "U_tilde",
}
FORBIDDEN_PARAMETERS = {
    "arm", "full_gt", "ground_truth", "unlabeled_gt", "evaluation_labels",
    "generator_views", "verifier_views", "epsilon", "temperature", "margin",
}
FORBIDDEN_SOURCE_PHRASES = (
    "true_u", "true_uniform", "shuffle_u", "base", "c3_b0", "pilot", "f0",
    "final_core", "optimizer", "phase a", "phase b", "native target",
    "u_new", "u_tilde", "u_refined", "semantic_u", "recalibrated_u",
)


def _trees():
    return tuple(
        (path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        for path in PRODUCTION_FILES
    )


def _manifest_entries(path):
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(None, 1)
        relative = relative.lstrip("* ")
        entries.append((digest, ROOT / relative))
    return entries


def _file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exactly_two_production_and_four_test_files_exist():
    assert tuple(sorted((ROOT / "release_core/action").glob("*.py"))) == tuple(
        sorted(PRODUCTION_FILES)
    )
    actual_tests = tuple(sorted((ROOT / "tests/release_core").glob("test_r4_*.py")))
    assert actual_tests == tuple(sorted(TEST_FILES))


def test_r4_imports_only_torch_standard_library_and_r3_primitive():
    imports = set()
    for _, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
    assert imports <= {
        "dataclasses", "torch", "release_core.semantics.relations", "relation",
    }
    assert "release_core.semantics.relations" in imports


def test_no_r5_closed_route_or_control_arm_active_symbols():
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
    source = "\n".join(
        path.read_text(encoding="utf-8").lower() for path in PRODUCTION_FILES
    )
    for phrase in FORBIDDEN_SOURCE_PHRASES:
        assert phrase not in source
    assert ".repeat(" not in source
    assert "sample_ids" not in source


def test_clean_public_api_and_immutable_audit_contract():
    assert action.__all__ == (
        "RelationActionAudit", "build_action_weight", "relation_bce",
        "utility_conditioned_relation_loss",
    )
    assert tuple(inspect.signature(action.build_action_weight).parameters) == (
        "u_cycle", "balance_weight", "like",
    )
    assert tuple(inspect.signature(action.relation_bce).parameters) == (
        "relation_probability", "pred_relation",
    )
    assert tuple(
        inspect.signature(action.utility_conditioned_relation_loss).parameters
    ) == (
        "q_query_views", "q_anchor_views", "pred_relation", "u_cycle",
        "balance_weight",
    )
    audit = action.RelationActionAudit(
        2, (1.0, 2.0), 3.0, True, True, True, True, True, True, True, True
    )
    with pytest.raises(FrozenInstanceError):
        audit.view_count = 3


def test_r1_r2_r3_frozen_manifests_remain_exact():
    manifests = (
        ("r1_final_backbone_data_foundation_pass_20260919", "r1_production_source_sha256.txt", 15),
        ("r1_final_backbone_data_foundation_pass_20260919", "r1_test_sha256.txt", 11),
        ("r2_final_directional_cyclic_utility_pass_20260920", "r2_source_sha256.txt", 3),
        ("r2_final_directional_cyclic_utility_pass_20260920", "r2_test_sha256.txt", 4),
        ("r3_final_sparse_relation_semantics_pass_20260921", "r3_production_sha256.txt", 3),
        ("r3_final_sparse_relation_semantics_pass_20260921", "r3_test_sha256.txt", 4),
    )
    for directory, filename, count in manifests:
        entries = _manifest_entries(ROOT / "experiment_freeze" / directory / filename)
        assert len(entries) == count
        assert all(path.is_file() and _file_sha256(path) == digest for digest, path in entries)
