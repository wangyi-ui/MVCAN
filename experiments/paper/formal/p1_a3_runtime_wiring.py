"""P1-A3 runtime adapters over release primitives; no alternate science."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from release_core.backbone import MultiViewBackbone
from release_core.config import get_native_config
from release_core.data.weak_quality import ndarray_sha256
from release_core.runtime import ProvenanceConfig
from release_core.semantics import SparseLabelSplit, build_relation_semantics
from release_core.utility import build_directional_actions, compute_directional_cycle_utility

from . import p1_a0_formal_protocol as protocol
from . import p1_a2_execution_contract as execution


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_initialization(*, dataset, training_seed, feature_path, output_dir, device):
    """Create the non-overwriting namespace; real numerical loop is delegated to release primitives in P1-A3 execution."""
    execution.validate_execution_contract()
    item = next(value for value in protocol.FORMAL_DATASETS if value.name == dataset)
    output = Path(output_dir)
    if output.exists():
        raise RuntimeError("FORMAL_INITIALIZATION_OUTPUT_ALREADY_EXISTS")
    if not Path(feature_path).is_file():
        raise RuntimeError("FORMAL_INPUT_NOT_MATERIALIZED")
    # This constructor deliberately preserves native config seed while the
    # process generator/KMeans/order seed remains training_seed.
    config = get_native_config(dataset)
    if config["training"]["seed"] != item.native_config_seed:
        raise RuntimeError("FORMAL_NATIVE_CONFIG_SEED_MISMATCH")
    with np.load(feature_path, allow_pickle=False) as archive:
        views = tuple(np.ascontiguousarray(archive["view_" + str(i + 1)]) for i in range(item.n_views))
    if tuple(view.shape for view in views) != tuple((item.n_samples, dim) for dim in item.view_dims):
        raise RuntimeError("FORMAL_INPUT_CONTRACT_MISMATCH")
    # A real call is intentionally guarded by this stage's no-training authority.
    # The returned plan is sufficient for first-materialization authorization and
    # captures the exact preserved carrier/output schema.
    return {"dataset": dataset, "training_seed": training_seed, "native_config_seed": item.native_config_seed,
            "feature_sha256": _sha(feature_path), "output_dir": str(output),
            "generator_create_count": 1, "generator_seed": training_seed,
            "generator_reset_count": 0, "shared_across_ae_native": True,
            "native_preparation_epochs": config["training"]["init_epoch"],
            "native_loop_iterations": config["training"]["epoch"] + 1,
            "last_refresh_epoch": config["training"]["epoch"],
            "carrier_schema": ("M_v", "P_global", "view_weights", "q_local", "q_aligned", "sample_ids")}


def build_true_action(*, dataset, training_seed, carrier_state_path, split_path, output_dir):
    """Build canonical R2/R3 directly from sealed retained-carrier q_aligned."""
    execution.validate_execution_contract()
    target = Path(output_dir)
    if target.exists():
        raise RuntimeError("FORMAL_ACTION_OUTPUT_ALREADY_EXISTS")
    if not Path(carrier_state_path).is_file() or not Path(split_path).is_file():
        raise RuntimeError("FORMAL_ACTION_REQUIRES_SEALED_INITIALIZATION_AND_SPLIT")
    with np.load(carrier_state_path, allow_pickle=False) as carrier, np.load(split_path, allow_pickle=False) as split_file:
        q_aligned = np.ascontiguousarray(carrier["q_aligned"])
        ids = np.ascontiguousarray(split_file["sample_ids"], dtype=np.int64)
        split = SparseLabelSplit(ids, split_file["labeled_ids"], split_file["labeled_targets"], split_file["unlabeled_ids"], q_aligned.shape[-1], protocol.SPARSE_LABEL_PROTOCOL["labels_per_class"], protocol.SPARSE_LABEL_PROTOCOL["label_seed"], dataset)
    actions = build_directional_actions(q_aligned.shape[1])
    cycle = compute_directional_cycle_utility(torch.from_numpy(q_aligned), actions)
    semantics = build_relation_semantics(np.asarray(cycle["y_gen"].cpu()), split, actions)
    return {"q_aligned_sha256": ndarray_sha256(q_aligned), "U_cycle": cycle["U_cycle"].detach().cpu().numpy(),
            "y_gen": cycle["y_gen"].detach().cpu().numpy(), "PredRelation_true": semantics.pred_relation,
            "relation_balance_weights_true": semantics.balance_weights, "detached_frozen": True}


def arm_provenance(*, feature, feature_audit, split, split_audit, utility, utility_audit, semantic, semantic_audit, initialization, output):
    """Construct the release ProvenanceConfig without changing its interface."""
    paths = (feature, feature_audit, split, split_audit, utility, utility_audit, semantic, semantic_audit,
             initialization["audit"], *initialization["checkpoint_paths"])
    return ProvenanceConfig(feature_artifact=feature, feature_audit=feature_audit,
        sparse_split_artifact=split, sparse_split_audit=split_audit, utility_artifact=utility,
        utility_audit=utility_audit, semantic_artifact=semantic, semantic_audit=semantic_audit,
        checkpoint_paths=initialization["checkpoint_paths"], checkpoint_audit=initialization["audit"],
        output_root=output, strict_replay=True,
        expected_file_sha256=tuple((Path(path), _sha(path)) for path in paths),
        expected_initial_model_sha256=initialization["initial_model_sha256"])
