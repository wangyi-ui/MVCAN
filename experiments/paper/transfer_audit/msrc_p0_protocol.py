"""Exact P0-A0/P0-A1 MSRC-v1 transfer protocol and provenance constants."""

from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_ROOT = Path("/root/autodl-tmp/CVPR24-MVCAN")

DATASET = "MSRC-v1"
DATASET_PATH = ARCHIVE_ROOT / "data/MSRC_v1.mat"
DATASET_SHA256 = "38d89aa41ae984f0b026f4baa8b5dcb7c569c471e2fb42b73a0f6743398e9103"

N = 210
V = 5
K = 7
VIEW_DIMS = (24, 576, 512, 256, 254)
LABELS_PER_CLASS = 2
L = 14
N_U = 196
ACTION_COUNT = 20
LABEL_SEED = 20
WEAK_QUALITY_SEED = 20
SNR_DB = 2.5
TRAINING_SEED = 20
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
REFRESH_INTERVAL = 100
NATIVE_LAMBDA1 = 0.01

CORRUPTED_PAIRS = 525
TOTAL_SAMPLE_VIEW_PAIRS = 1050
ROWS_WITH_TWO = 105
ROWS_WITH_THREE = 105
PER_VIEW_CORRUPTED = (105, 105, 105, 105, 105)
CORRUPTION_MASK_SHA256 = (
    "d64f5fbf95518e5a77e44a1fe18a15dee9924b8c2a341e930f86f20ce5ac9e9d"
)

LABELED_IDS = (17, 20, 30, 42, 60, 81, 96, 107, 139, 142, 153, 175, 182, 201)
LABELED_TARGETS = (0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6)
SPLIT_SHA256 = "4f16e8b1b8213591a7ea1d36115335ccee43bf568d32bbcb007f929bc9b15231"

ARMS = ("BASE", "TRUE_U", "UNIFORM")
ARM_DIRECTORY_NAMES = {"BASE": "base", "TRUE_U": "true_u", "UNIFORM": "uniform"}
ARM_SEMANTICS = {
    "BASE": "no R4 objective; native Phase B only in every epoch",
    "TRUE_U": "true U_cycle + true PredRelation + true balance",
    "UNIFORM": "ones_like(U_cycle) + true PredRelation + true balance",
}

CHECKPOINT_ROOT = ARCHIVE_ROOT / "outputs/b2_weak_quality/snr2p5_k2_seed20/models"
CHECKPOINT_PATHS = tuple(
    CHECKPOINT_ROOT / ("MSRC-v1%dV.pth" % view_id) for view_id in range(1, 6)
)
CHECKPOINT_SHA256 = (
    "eb75aa4c607256acd576fd036306a4654c6a5a748fd9f3e0e36a3a2b2b420c1f",
    "c309f37326f6ccc3e3b6c360b088eda07d1294b463908669b9fc57e8d51a5ca7",
    "003f77ed027da8f8ac235a77a6c5c74ba89f1dfba37cddbf1c69991839bb4d3d",
    "496ad10012120a04d10aef6f74c2c6ab759ec7ffa76a748d384155ac1b99ffa8",
    "ff450b868021daef13b013fc369d59b4fcd16804752bd330b85480290422094c",
)
INITIAL_MODEL_SHA256 = "626eaf912284cc5f8f536b542ecf3d10fe4ae40f61eaecdabcc8e5bf4a438031"
HISTORICAL_BACKBONE_SHA256 = (
    "f4f026f47938c619bfdfb868cfcdb5eac354652a51d71199edbc997c0a5de0ab"
)
CHECKPOINT_SOURCE_CONDITION = "snr2p5_k2_seed20"
CHECKPOINT_SOURCE_MANIFEST = (
    REPOSITORY_ROOT / "experiments/b6_weak_quality/b6_wq0_frozen_baseline_manifest.json"
)
CHECKPOINT_SOURCE_PROVENANCE = (
    ARCHIVE_ROOT
    / "outputs/b6_weak_quality/wq0_frozen_baseline_provenance/seed20_provenance.json"
)
CHECKPOINT_AUDIT = (
    REPOSITORY_ROOT
    / "experiment_freeze/p0_a0_msrc_transfer_preregistered_20260923/checkpoint_reference.json"
)

FEATURE_FIELDS = ("X1", "X2", "X3", "X4", "X5", "sample_ids")
SPLIT_FIELDS = ("sample_ids", "labeled_ids", "labeled_targets", "unlabeled_ids")
FORBIDDEN_FEATURE_FIELDS = frozenset(
    ("y", "gt", "labels", "label", "full_gt", "ground_truth", "y_true")
)


@dataclass(frozen=True)
class FrozenTrainingConfig:
    training_seed: int = TRAINING_SEED
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    refresh_interval: int = REFRESH_INTERVAL
    label_seed: int = LABEL_SEED
    labels_per_class: int = LABELS_PER_CLASS
    weak_quality_seed: int = WEAK_QUALITY_SEED
    snr_db: float = SNR_DB


def validate_arm(arm):
    """Return one canonical arm name or fail closed."""
    if arm not in ARMS:
        raise ValueError("arm must be one of " + str(ARMS))
    return arm


def validate_frozen_training_values(
    *, training_seed, epochs, batch_size, learning_rate, refresh_interval
):
    """Reject all accidental tuning at the public runner boundary."""
    actual = (
        int(training_seed), int(epochs), int(batch_size), float(learning_rate),
        int(refresh_interval),
    )
    expected = (TRAINING_SEED, EPOCHS, BATCH_SIZE, LEARNING_RATE, REFRESH_INTERVAL)
    if actual != expected:
        raise ValueError("P0-A1 training configuration is frozen: " + repr(expected))
    return FrozenTrainingConfig()


def default_input_dir():
    return (
        REPOSITORY_ROOT
        / "outputs/paper/transfer_audit/MSRC-v1/snr2p5_half_seed20/inputs"
    )


def default_output_dir(arm):
    return default_input_dir().parent / ARM_DIRECTORY_NAMES[validate_arm(arm)]
