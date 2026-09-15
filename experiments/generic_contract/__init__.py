"""Dataset-generic, pre-training contracts for G0."""

from .action_space import (
    ActionSpace,
    action_count,
    build_action_space,
    historical_action_logical_sha256,
)
from .dataset_contract import DatasetContract, infer_dataset_contract
from .generic_weak_quality import (
    apply_half_gaussian_corruption,
    generate_half_corruption_mask,
)
from .sparse_label_contract import (
    SparseLabelSplit,
    frozen_caltech_sparse_split,
    materialize_hash_ranked_sparse_split,
)

__all__ = (
    "ActionSpace",
    "DatasetContract",
    "SparseLabelSplit",
    "action_count",
    "apply_half_gaussian_corruption",
    "build_action_space",
    "frozen_caltech_sparse_split",
    "generate_half_corruption_mask",
    "historical_action_logical_sha256",
    "infer_dataset_contract",
    "materialize_hash_ranked_sparse_split",
)
