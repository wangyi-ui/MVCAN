"""Generic BASE dispatch boundary.  BASE never creates a semantic optimizer."""

def base_audit_contract():
    return {
        "phase_a_executed": False,
        "semantic_optimizer_created": False,
        "phase_b_executed": True,
        "phase_order": ("PHASE_A_SKIPPED", "REFRESH", "PHASE_B"),
        "full_gt_loaded": False,
        "final_prediction_source": "final native refresh second-pass KMeans prediction IDs",
    }


def run_base_pre_gt(*_args, **_kwargs):
    """Real BASE must be supplied only after the native generator contract is frozen."""
    raise RuntimeError("FORMAL_NATIVE_GENERATOR_SEMANTICS_UNRESOLVED")
