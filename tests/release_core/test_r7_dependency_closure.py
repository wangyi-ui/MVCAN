from tools.release_validation.r7_a1_release_composition_audit import (
    audit_dependency_closure,
)


def test_r7_static_and_runtime_dependency_closure():
    result = audit_dependency_closure()
    assert result["pass"] is True
    assert result["static_module_count"] > 0
    assert result["failures"] == []
    assert any(item["check"] == "static" for item in result["static_records"])
    assert any(item["check"] == "runtime" for item in result["runtime_records"])


def test_r7_runtime_loads_no_repo_python_outside_release_core():
    result = audit_dependency_closure()
    forbidden = [
        item for item in result["runtime_records"]
        if item["origin_classification"] == "repo_forbidden"
    ]
    assert forbidden == []
