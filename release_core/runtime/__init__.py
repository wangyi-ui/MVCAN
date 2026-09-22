"""Clean runtime boundary around the frozen R1--R5 scientific core."""

import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


_DATASET_NAMES = ("Caltech-6V", "MSRC-v1", "BDGP")
_BUNDLE_KEYS = (
    "sample_ids",
    "final_predictions",
    "labeled_ids",
    "q_local",
    "q_aligned",
    "M_v",
    "input_sha256",
    "initial_model_sha256",
    "final_model_sha256",
)
_SEAL_SCHEMA = "release-core-pre-gt-seal-v1"


def _payload_sha256(value):
    """Hash dtype, shape, and bytes in the frozen scientific-payload namespace."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    for component in (
        str(array.dtype).encode("ascii"),
        ",".join(str(int(size)) for size in array.shape).encode("ascii"),
        array.tobytes(order="C"),
    ):
        digest.update(struct.pack(">Q", len(component)))
        digest.update(component)
    return digest.hexdigest()


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def _positive_float(value, name):
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(name + " must be positive")
    return result


def _optional_sha256(value, name):
    if value is None:
        return None
    result = str(value).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(name + " must be a lowercase SHA256 digest")
    return result


@dataclass(frozen=True)
class RuntimeConfig:
    """Scientific configuration plus process-only device selection."""

    dataset: str
    training_seed: int
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-4
    native_lambda1: float = 0.01
    refresh_interval: int = 100
    label_seed: int = 20
    labels_per_class: int = 2
    device: str = "cuda:0"

    def __post_init__(self):
        if self.dataset not in _DATASET_NAMES:
            raise ValueError("unsupported dataset: " + str(self.dataset))
        if isinstance(self.training_seed, bool) or not isinstance(self.training_seed, int):
            raise ValueError("training_seed must be an integer")
        if self.training_seed < 0:
            raise ValueError("training_seed must be nonnegative")
        _positive_integer(self.epochs, "epochs")
        _positive_integer(self.batch_size, "batch_size")
        _positive_integer(self.refresh_interval, "refresh_interval")
        _positive_integer(self.labels_per_class, "labels_per_class")
        if isinstance(self.label_seed, bool) or not isinstance(self.label_seed, int):
            raise ValueError("label_seed must be an integer")
        if self.label_seed < 0:
            raise ValueError("label_seed must be nonnegative")
        object.__setattr__(self, "learning_rate", _positive_float(
            self.learning_rate, "learning_rate"
        ))
        object.__setattr__(self, "native_lambda1", _positive_float(
            self.native_lambda1, "native_lambda1"
        ))
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")


@dataclass(frozen=True)
class ProvenanceConfig:
    """Immutable paths and validation gates that cannot alter science."""

    feature_artifact: Path
    feature_audit: Path
    sparse_split_artifact: Path
    sparse_split_audit: Path
    utility_artifact: Path
    utility_audit: Path
    semantic_artifact: Path
    semantic_audit: Path
    checkpoint_paths: Tuple[Path, ...]
    checkpoint_audit: Path
    output_root: Path
    strict_replay: bool = False
    expected_file_sha256: Tuple[Tuple[Path, str], ...] = ()
    expected_initial_model_sha256: Optional[str] = None
    expected_final_model_sha256: Optional[str] = None
    expected_prediction_sha256: Optional[str] = None
    historical_audit_sha256: Optional[str] = None
    historical_seal_sha256: Optional[str] = None

    def __post_init__(self):
        path_fields = (
            "feature_artifact", "feature_audit", "sparse_split_artifact",
            "sparse_split_audit", "utility_artifact", "utility_audit",
            "semantic_artifact", "semantic_audit", "checkpoint_audit",
            "output_root",
        )
        for name in path_fields:
            object.__setattr__(self, name, Path(getattr(self, name)))
        checkpoints = tuple(Path(path) for path in self.checkpoint_paths)
        if not checkpoints:
            raise ValueError("checkpoint_paths must not be empty")
        object.__setattr__(self, "checkpoint_paths", checkpoints)
        if not isinstance(self.strict_replay, bool):
            raise ValueError("strict_replay must be boolean")
        expected = []
        for path, digest in self.expected_file_sha256:
            expected.append((Path(path), _optional_sha256(digest, "expected file hash")))
        object.__setattr__(self, "expected_file_sha256", tuple(expected))
        for name in (
            "expected_initial_model_sha256",
            "expected_final_model_sha256",
            "expected_prediction_sha256",
            "historical_audit_sha256",
            "historical_seal_sha256",
        ):
            object.__setattr__(self, name, _optional_sha256(getattr(self, name), name))
        if self.strict_replay and not expected:
            raise ValueError("strict replay requires expected file hashes")


@dataclass(frozen=True)
class SealedPredictionPaths:
    bundle: Path
    audit: Path
    seal: Path

    def __post_init__(self):
        object.__setattr__(self, "bundle", Path(self.bundle))
        object.__setattr__(self, "audit", Path(self.audit))
        object.__setattr__(self, "seal", Path(self.seal))


@dataclass(frozen=True)
class Metrics:
    acc: float
    nmi: float
    ari: float


def run_pre_gt(runtime: RuntimeConfig, provenance: ProvenanceConfig):
    """Lazily enter the training/prediction process without importing GT code."""
    from .entrypoint import run_pre_gt as implementation

    return implementation(runtime, provenance)


def evaluate_postseal(
    sealed: SealedPredictionPaths,
    full_gt_path,
    *,
    output_path=None,
):
    """Lazily enter the separate, seal-first evaluation process."""
    from .evaluation import evaluate_postseal as implementation

    return implementation(sealed, full_gt_path, output_path=output_path)


__all__ = (
    "RuntimeConfig",
    "ProvenanceConfig",
    "SealedPredictionPaths",
    "Metrics",
    "run_pre_gt",
    "evaluate_postseal",
)
