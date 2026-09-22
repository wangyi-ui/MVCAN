import pytest

from tools.release_validation.r7_a1_release_composition_audit import (
    AuditFailure,
    audit_caltech_behavior,
    audit_evidence_boundary,
    audit_gt_firewall,
    validate_evidence_boundary,
)


def test_r7_caltech_three_seed_full_chain_exactness():
    result = audit_caltech_behavior()
    assert result["pass"] is True
    assert result["exact_seed_count"] == result["required_seed_count"] == 3
    assert result["metrics_only_parity_is_sufficient"] is False
    for seed in ("20", "30", "50"):
        assert all(item["array_equal"] for item in result["seeds"][seed]["arrays"].values())
        assert result["seeds"][seed]["arrays"]["q_local"]["max_abs_diff"] == 0.0
        assert result["seeds"][seed]["arrays"]["q_aligned"]["max_abs_diff"] == 0.0
        assert result["seeds"][seed]["arrays"]["M_v"]["max_abs_diff"] == 0.0


def test_r7_gt_firewall_three_of_three():
    result = audit_gt_firewall()
    assert result["pass"] is True
    assert result["pass_count"] == result["required_count"] == 3


def test_r7_dataset_evidence_boundary_is_fail_closed():
    result = audit_evidence_boundary()
    assert result["Caltech-6V"] == "EXACT_FULL_CHAIN_3_OF_3"
    assert result["MSRC-v1"] == "BOUNDED_ONLY"
    assert result["BDGP"] == "BOUNDED_ONLY"
    upgraded = dict(result)
    upgraded["MSRC-v1"] = "EXACT_FULL_CHAIN"
    with pytest.raises(AuditFailure, match="MSRC-v1 evidence was upgraded"):
        validate_evidence_boundary(upgraded)
