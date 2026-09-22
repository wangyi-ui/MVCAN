from tools.release_validation.r7_a1_release_composition_audit import (
    audit_dependency_closure,
    audit_ownership_closure,
)


def test_r7_unique_ownership_and_direction():
    dependency = audit_dependency_closure()
    result = audit_ownership_closure(dependency)
    assert result["pass"] is True
    assert result["failures"] == []
    assert all(item["declared_owner"] in {"R1", "R2", "R3", "R4", "R5", "R6"} for item in result["files"])


def test_r7_explicit_scientific_responsibility_checks():
    result = audit_ownership_closure()
    assert all(result["explicit_checks"].values())
    assert result["explicit_checks"]["r7_release_core_files_absent"] is True
