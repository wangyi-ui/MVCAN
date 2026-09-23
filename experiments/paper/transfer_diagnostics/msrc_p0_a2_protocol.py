"""Frozen P0-A2 same-condition initialization diagnostic contract."""

from dataclasses import dataclass
from pathlib import Path

from experiments.paper.transfer_audit import msrc_p0_protocol as p0


REPOSITORY_ROOT = p0.REPOSITORY_ROOT
DATASET = p0.DATASET
DATASET_PATH = p0.DATASET_PATH
DATASET_SHA256 = p0.DATASET_SHA256
N, V, K = p0.N, p0.V, p0.K
VIEW_DIMS = p0.VIEW_DIMS
FEATURE_FIELDS = p0.FEATURE_FIELDS
FORBIDDEN_FEATURE_FIELDS = p0.FORBIDDEN_FEATURE_FIELDS
CORRUPTION_MASK_SHA256 = p0.CORRUPTION_MASK_SHA256
SPLIT_SHA256 = p0.SPLIT_SHA256
FEATURE_ARTIFACT_SHA256 = (
    "2a225229de5b4c4d87ac530860691c351a0e8e3c1a015f1ecd290673f663755d"
)

P0_A0_COMMIT = "cf59ad7a41fd0b0347c03e63036c558800ba7735"
P0_A0_TAG = "p0-a0-msrc-transfer-preregistered-20260923"
P0_A1_ROOT = p0.default_input_dir().parent
P0_A1_INPUT_DIR = p0.default_input_dir()
P0_A1_SUMMARY_SHA256 = (
    "e20096a6463262aa795874769d3cf062584f8f4f25aa77dfa5a5a1caed911ae1"
)

OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "outputs/paper/transfer_diagnostics/MSRC-v1/p0_a2_same_condition_init_seed20"
)
INITIALIZATION_DIR = OUTPUT_ROOT / "init"
INITIALIZATION_CHECKPOINT_NAMES = tuple(
    "view%d.pth" % view_id for view_id in range(1, V + 1)
)
INITIALIZATION_AUDIT_NAME = "initialization_audit.json"
INITIALIZATION_SEAL_NAME = "initialization_seal.json"
INITIALIZATION_MANIFEST_NAME = "initialization_manifest.json"
INITIALIZATION_SOURCE_CONDITION = "current_msrc_primary_50pct_corruption"

ARMS = p0.ARMS
ARM_DIRECTORY_NAMES = p0.ARM_DIRECTORY_NAMES
ARM_SEMANTICS = p0.ARM_SEMANTICS

TRAINING_SEED = p0.TRAINING_SEED
EPOCHS = p0.EPOCHS
BATCH_SIZE = p0.BATCH_SIZE
LEARNING_RATE = p0.LEARNING_RATE
REFRESH_INTERVAL = p0.REFRESH_INTERVAL
NATIVE_LAMBDA1 = p0.NATIVE_LAMBDA1
LABEL_SEED = p0.LABEL_SEED
LABELS_PER_CLASS = p0.LABELS_PER_CLASS
WEAK_QUALITY_SEED = p0.WEAK_QUALITY_SEED
SNR_DB = p0.SNR_DB

NATIVE_INIT_EPOCHS = 200
NATIVE_CONFIG_EPOCH = 1000
NATIVE_LOOP_ITERATIONS = 1001
NATIVE_REFRESH_INTERVAL = 100
NATIVE_REFRESH_EPOCHS = tuple(range(0, NATIVE_LOOP_ITERATIONS, 100))
NATIVE_REFRESH_COUNT = 11
KMEANS_N_INIT = 100
KMEANS_RANDOM_STATE = 20

P0_A1_RESULT_HASHES = {
    "summary": P0_A1_SUMMARY_SHA256,
    "base_manifest": "ec2aa43804846a14477bb66f2652b253edf5a43e791c6abe8a9e31916444089c",
    "base_audit": "37aba71307e46e6720da734bd9070341b3e6bd119eb64798f37ebfb291b7b753",
    "base_seal": "6d8ad2b28cafd38ccb95e0b7e596fa7d9a5e1f2cb1e95e15b5d961f63f13cec7",
    "base_metrics": "da07ef68f694d0498d6c14ac1d9d1e96e3387be17adff91c47f16ac262608a4e",
    "true_u_manifest": "4fb826e35ab9550901d93b12253b6bcf823fd1c948c1d264ae6e08049fca0cf7",
    "true_u_audit": "91427bc3cb9efdae8a5b0b793fee0c796ea0bea2a5da706855ffff848958bfc6",
    "true_u_seal": "53acdff2d8bc9f8073d0caef88d980793659ca5b9dd99cebdc49511a4c8e055d",
    "true_u_metrics": "99b727e0fa7e2830bbb47d2a4176e8d3d2fc9ca0149c9e0245020223a2169a47",
    "uniform_manifest": "f4e8faa874793badd64fa9373f382417b575d6b4c6074909b514016c22ffc345",
    "uniform_audit": "453822557bff391cf4a84ba9b06a3f6018e714064b363ff1960f941b9ac79bcb",
    "uniform_seal": "96ba14691eba5f3239de83046ffc415727979432bb0fa7d1b3d7a67c742d2503",
    "uniform_metrics": "1acb27d7aab446e1df1458cc71eadced1da2616993842537bf41a7a61e511cba",
}


@dataclass(frozen=True)
class FrozenInitializationConfig:
    seed: int = TRAINING_SEED
    init_epochs: int = NATIVE_INIT_EPOCHS
    native_config_epoch: int = NATIVE_CONFIG_EPOCH
    native_loop_iterations: int = NATIVE_LOOP_ITERATIONS
    refresh_interval: int = NATIVE_REFRESH_INTERVAL
    learning_rate: float = LEARNING_RATE
    native_lambda1: float = NATIVE_LAMBDA1
    batch_size: int = BATCH_SIZE
    kmeans_n_init: int = KMEANS_N_INIT
    kmeans_random_state: int = KMEANS_RANDOM_STATE


def _require_equal(actual, expected, name):
    if actual != expected:
        raise ValueError(name + " is frozen at " + repr(expected))


def validate_initialization_values(
    *, seed, init_epochs, native_config_epoch, refresh_interval,
    learning_rate, native_lambda1, batch_size,
):
    """Reject any initialization tuning at the public CLI boundary."""
    values = {
        "seed": int(seed),
        "init_epochs": int(init_epochs),
        "native_config_epoch": int(native_config_epoch),
        "refresh_interval": int(refresh_interval),
        "learning_rate": float(learning_rate),
        "native_lambda1": float(native_lambda1),
        "batch_size": int(batch_size),
    }
    expected = FrozenInitializationConfig()
    for name, value in values.items():
        _require_equal(value, getattr(expected, name), name)
    return expected


def validate_arm(arm):
    return p0.validate_arm(arm)


def validate_arm_training_values(**values):
    return p0.validate_frozen_training_values(**values)


def default_output_dir(arm):
    return OUTPUT_ROOT / ARM_DIRECTORY_NAMES[validate_arm(arm)]


def initialization_checkpoint_paths(init_dir=INITIALIZATION_DIR):
    root = Path(init_dir)
    return tuple(root / name for name in INITIALIZATION_CHECKPOINT_NAMES)
