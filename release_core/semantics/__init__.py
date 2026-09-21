"""Sparse-label relation semantics for the frozen release core."""

from .relations import (
    RelationSemantics,
    SparseMapping,
    build_relation_balance_weights,
    build_relation_semantics,
    build_vote_semantic_state,
    compute_view_relations,
    fit_sparse_mapping,
)
from .sparse_labels import (
    SparseLabelSplit,
    canonical_split_payload,
    split_sha256,
    validate_sparse_label_split,
)

__all__ = (
    "RelationSemantics",
    "SparseLabelSplit",
    "SparseMapping",
    "build_relation_balance_weights",
    "build_relation_semantics",
    "build_vote_semantic_state",
    "canonical_split_payload",
    "compute_view_relations",
    "fit_sparse_mapping",
    "split_sha256",
    "validate_sparse_label_split",
)
