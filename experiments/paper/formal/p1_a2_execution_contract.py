"""P1-A2 execution clarification derived from frozen historical lineage.

This is execution provenance only: it introduces no scientific parameter.
"""

from dataclasses import dataclass

import numpy as np

from . import p1_a0_formal_protocol as protocol


NATIVE_DATALOADER_GENERATOR_CONTRACT = {
    "generator_device": "cpu", "seed_source": "formal training_seed",
    "created_once_per_native_preparation": True,
    "shared_across_ae_and_native_loops": True,
    "reset_between_stages": False, "reset_each_epoch": False,
}
ACTION_CARRIER_CONTRACT = {
    "q_local": "post-final model posterior",
    "M_v": "retained last scheduled native-refresh matrix before final native update",
    "q_aligned": "post_final_q_local @ retained_M_v.T",
    "fresh_post_final_refresh": False,
    "reset_incoming_view_weights_to_ones": False,
    "p_global_used_by_r2": False,
}
TRUE_UNIFORM_CONTRACT = {
    "utility": "ones_like(true U_cycle)",
    "pred_relation": "PredRelation_true unchanged",
    "balance": "relation_balance_weights_true unchanged",
}
CALTECH_FIXED_SHUFFLED_TARGETS = (6, 2, 3, 0, 4, 1, 5, 6, 3, 0, 2, 5, 1, 4)
SOURCE_PROVENANCE = {
    "generator_and_carrier": "experiments/generic_contract/generic_final_core_adapter.py: prepare_native_backbone, coordinate_snapshot",
    "shuffle_relation": "experiments/cyclic_utility/c3_b0_relation_action_protocol.py and c3_a0_utility_conditioned_action_granularity_protocol.py",
    "caltech_shuffle": "experiments/cyclic_utility/c2_a0_sparse_label_utility_protocol.py: FIXED_SHUFFLED_TARGETS",
    "p0_carrier_evidence": "experiment_freeze/p0_a5_r6_final_refresh_replay_validity_preregistered_20260924 and p0_a5_r7_hfw_carrier_root_cause_preregistered_20260924",
}


def arm_capability(dataset, arm):
    if dataset not in tuple(item.name for item in protocol.FORMAL_DATASETS):
        raise RuntimeError("FORMAL_DATASET_NOT_AUTHORIZED")
    if arm in ("BASE", "OURS_TRUE_U", "TRUE_UNIFORM"):
        return "AUTHORIZED", None
    if arm == "SHUFFLE_U" and dataset == "Caltech-6V":
        return "AUTHORIZED", None
    if arm == "SHUFFLE_U":
        return "BLOCKED", "SHUFFLE_SEMANTICS_UNRESOLVED_FOR_" + dataset.split("-")[0].upper()
    raise RuntimeError("FORMAL_ARM_NOT_AUTHORIZED")


def align_with_retained_matrix(post_final_q_local, retained_m_v):
    """The only R2/R3 alignment path: never invokes a native refresh."""
    q = np.ascontiguousarray(np.asarray(post_final_q_local))
    matrix = np.ascontiguousarray(np.asarray(retained_m_v))
    if q.ndim != 3 or matrix.shape != (q.shape[1], q.shape[2], q.shape[2]):
        raise ValueError("retained carrier shape mismatch")
    return np.ascontiguousarray(np.stack([
        q[:, view, :] @ matrix[view].T for view in range(q.shape[1])
    ], axis=1))


def validate_execution_contract():
    if protocol.validate_formal_protocol() is not True:
        raise RuntimeError("FORMAL_PROTOCOL_UNRESOLVED")
    if NATIVE_DATALOADER_GENERATOR_CONTRACT["seed_source"] != "formal training_seed":
        raise RuntimeError("NATIVE_GENERATOR_CONTINUITY_UNRESOLVED")
    if ACTION_CARRIER_CONTRACT["fresh_post_final_refresh"]:
        raise RuntimeError("ACTION_CARRIER_SEMANTICS_UNRESOLVED")
    return True
