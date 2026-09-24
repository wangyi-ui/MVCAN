"""Recovered immutable P1-A0 main-experiment protocol constants.

This module defines protocol only.  It intentionally has no data loading,
model construction, optimizer, utility generation, or training code.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetProtocol:
    name: str
    n_samples: int
    n_views: int
    n_clusters: int
    view_dims: tuple
    native_config_seed: int
    native_lambda1: float
    weak_quality_mask_contract: str
    weak_quality_mask_sha256: str


@dataclass(frozen=True)
class AlternatingProtocol:
    epochs: int
    batch_size: int
    learning_rate: float
    refresh_interval: int
    phase_order: tuple
    r2_r3_timing: str
    r2_r3_frozen_during_r5: bool
    final_prediction_source: str


FORMAL_DATASETS = (
    DatasetProtocol("Caltech-6V", 1400, 6, 7, (48, 40, 254, 1984, 512, 928),
                    5, 0.01, "exactly 3 of 6 views per sample",
                    "e6250535db6e33b9263102cf335fcf9e8552a24cae4b70ec5a94887f4e58bf0b"),
    DatasetProtocol("MSRC-v1", 210, 5, 7, (24, 576, 512, 256, 254),
                    20, 0.01, "105 samples x2 and 105 samples x3 corrupted views",
                    "d64f5fbf95518e5a77e44a1fe18a15dee9924b8c2a341e930f86f20ce5ac9e9d"),
    DatasetProtocol("BDGP", 2500, 2, 5, (1750, 79), 1, 10.0,
                    "exactly 1 of 2 views per sample",
                    "a3c882c29f2f5064d552e60bdc1a2d585764317e7ba78aa055cd1c3c22e5de07"),
)
FORMAL_TRAINING_SEEDS = (20, 30, 50)
WEAK_QUALITY_PROTOCOL = {
    "snr_db": 2.5, "realization_seed": 20,
    "noise_seed_semantics": "derived deterministically from realization seed",
    "training_seed_changes_realization": False,
}
SPARSE_LABEL_PROTOCOL = {
    "labels_per_class": 2, "label_seed": 20,
    "Caltech-6V_total": 14, "MSRC-v1_total": 14, "BDGP_total": 10,
    "labels_determine": "semantic direction/content only",
}
NATIVE_PREPARATION_PROTOCOL = {
    "batch_size": 256, "init_epoch": 200, "T_1": 2, "T_2": 100,
    "epoch": 1000, "lr": 1e-4,
    "implementation": "same-condition native-prepared MVCAN state",
    "checkpoint_generation": "native preparation creates the per-run initial checkpoint/state before R2/R3",
    "weak_quality_preparation": "same fixed seed20 realization for every training seed",
    "kmeans_random_state": "training_seed for each formal run",
    "dataloader_order_seed": "training_seed for each formal run",
}
ACTION_GENERATION_PROTOCOL = {
    "state_dependent": True,
    "per_dataset_training_seed": True,
    "generation_timing": "once after same-condition native preparation",
    "during_r5": "detached and frozen; never dynamically regenerated per epoch",
    "utility_role": "action validity", "label_role": "semantic direction/content",
}
ALTERNATING_PROTOCOL = AlternatingProtocol(
    epochs=20, batch_size=256, learning_rate=1e-4, refresh_interval=100,
    phase_order=("Phase A", "target refresh if zero-based epoch % T2 == 0", "Phase B"),
    r2_r3_timing=ACTION_GENERATION_PROTOCOL["generation_timing"],
    r2_r3_frozen_during_r5=True,
    final_prediction_source="final native refresh second-pass KMeans prediction IDs",
)
EVALUATION_PROTOCOL = {
    "pre_gt_full_gt_loaded": False,
    "pre_gt_outputs": ("pre_gt_bundle", "audit", "seal"),
    "pre_gt_seal_order": "bundle -> audit -> seal -> evaluate_postseal",
    "sample_id_alignment": "exact canonical sample_ids; final prediction length equals N",
    "postseal_only": ("ACC", "NMI", "ARI"),
    "q_argmax_is_final_prediction": False,
}
ARM_PROTOCOL = {
    "main": {"BASE": "native Phase B only", "OURS_TRUE_U": "utility-conditioned semantic closed loop"},
    "ablation": {"TRUE_UNIFORM": "uniform action-validity weighting", "SHUFFLE_U": "historical correspondence-specificity control"},
}
FORBIDDEN_SCIENCE = (
    "U_tilde", "scalar utility calibration", "label-gated features",
    "pseudo-label CE", "memory bank", "prototype write", "feature gating",
    "fusion gating", "posterior gating", "target rewriting", "additive joint loss",
    "result-based tuning",
)
SOURCE_PROVENANCE = {
    "native_config": "release_core/config/native.py",
    "runtime": "release_core/runtime/__init__.py and entrypoint.py",
    "alternating": "release_core/training/alternating.py",
    "native_reference": "experiments/generic_contract/generic_final_core_adapter.py",
    "formal_epochs": "experiment_freeze/f0_a1_multiseed_formal_pass_20260914/MULTISEED_DYNAMICS_SUMMARY.json (B20)",
    "generic_contract": "experiment_freeze/g0_p0_generic_scientific_contract_preregistered_20260915/PROTOCOL.txt",
    "bdgp_mask": "experiment_freeze/g0_b1_bdgp_structural_pilot_pre_gt_pass_20260917/structural_logical_hashes.txt",
    "weak_quality": "release_core/data/weak_quality.py; G0-B1 structural pilot freeze",
    "r2_r3_r4_r5": "experiment_freeze/r2_final_directional_cyclic_utility_pass_20260920; r3_final_sparse_relation_semantics_pass_20260921; r4_final_utility_conditioned_relation_action_pass_20260921; r5_final_alternating_trainer_pass_20260922",
    "r6_r7": "experiment_freeze/r6_final_runtime_entrypoint_pass_20260922; r7_final_extraction_closure_pass_20260923",
}


FORMAL_FIELD_STATUS = {
    "dataset_matrix": "RESOLVED",
    "native_preparation_schedule": "RESOLVED",
    "formal_training_seeds": "RESOLVED",
    "weak_quality": "RESOLVED",
    "sparse_labels": "RESOLVED",
    "r2_r3_generation": "RESOLVED",
    "alternating_epochs": "RESOLVED",
    "refresh_schedule": "RESOLVED",
    "final_prediction": "RESOLVED",
    "gt_firewall": "RESOLVED",
    "arms": "RESOLVED",
    "no_tuning": "RESOLVED",
}
UNRESOLVED_FIELDS = ()

def validate_formal_protocol():
    """Fail closed if any critical recovered formal field is absent or altered."""
    if FORMAL_TRAINING_SEEDS != (20, 30, 50):
        raise RuntimeError("formal training seeds unresolved")
    if ALTERNATING_PROTOCOL.epochs != 20:
        raise RuntimeError("formal alternating epoch count unresolved")
    if NATIVE_PREPARATION_PROTOCOL != {
        "batch_size": 256, "init_epoch": 200, "T_1": 2, "T_2": 100,
        "epoch": 1000, "lr": 1e-4,
        "implementation": "same-condition native-prepared MVCAN state",
    "checkpoint_generation": "native preparation creates the per-run initial checkpoint/state before R2/R3",
    "weak_quality_preparation": "same fixed seed20 realization for every training seed",
        "kmeans_random_state": "training_seed for each formal run",
        "dataloader_order_seed": "training_seed for each formal run",
    }:
        raise RuntimeError("native preparation protocol unresolved")
    if FORMAL_DATASETS[2].weak_quality_mask_sha256 != "a3c882c29f2f5064d552e60bdc1a2d585764317e7ba78aa055cd1c3c22e5de07":
        raise RuntimeError("BDGP weak-quality mask hash unresolved")
    if any(status != "RESOLVED" for status in FORMAL_FIELD_STATUS.values()) or UNRESOLVED_FIELDS:
        raise RuntimeError("formal protocol has unresolved fields")
    if any(not item.weak_quality_mask_sha256 for item in FORMAL_DATASETS):
        raise RuntimeError("weak-quality protocol unresolved")
    if ACTION_GENERATION_PROTOCOL["during_r5"] != "detached and frozen; never dynamically regenerated per epoch":
        raise RuntimeError("R2/R3 timing unresolved")
    return True
