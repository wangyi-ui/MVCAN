from tools.release_validation.r7_a1_release_composition_audit import (
    audit_lineage_and_source_identity,
    audit_r5_training_dynamics,
)


def test_r7_lineage_freezes_sources_and_tests_are_exact():
    result = audit_lineage_and_source_identity()
    assert result["pass"] is True
    assert result["lineage_integrity"] is True
    assert result["source_identity"] is True
    assert result["test_identity"] is True
    assert set(result["tags"]) == {"R1", "R2", "R3", "R4", "R5", "R6", "R7-A0"}


def test_r7_inherits_r5_dynamics_without_training():
    result = audit_r5_training_dynamics(source_identity=True)
    assert result["pass"] is True
    assert result["checks"]["training_rerun"] is False
    assert result["checks"]["trajectory_exact"] is True
    assert result["checks"]["phase_schedule_exact"] is True
    assert result["checks"]["optimizer_topology_exact"] is True
