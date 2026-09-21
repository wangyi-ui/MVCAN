"""Public alternating-training orchestration API."""

from .alternating import (
    EpochAudit,
    NativeTargetState,
    OptimizerTopologyAudit,
    PhaseAudit,
    TargetRefreshAudit,
    TrainingAudit,
    TrainingOrders,
    build_decoupled_optimizers,
    initial_native_target_state,
    precompute_training_orders,
    refresh_native_state_if_due,
    run_alternating_epoch,
    run_alternating_training,
    run_native_consolidation_phase,
    run_relation_refinement_phase,
)

__all__ = (
    "EpochAudit",
    "NativeTargetState",
    "OptimizerTopologyAudit",
    "PhaseAudit",
    "TargetRefreshAudit",
    "TrainingAudit",
    "TrainingOrders",
    "build_decoupled_optimizers",
    "initial_native_target_state",
    "precompute_training_orders",
    "refresh_native_state_if_due",
    "run_alternating_epoch",
    "run_alternating_training",
    "run_native_consolidation_phase",
    "run_relation_refinement_phase",
)
