import hashlib
import struct
from pathlib import Path

import numpy as np
import torch

from experiments.cyclic_utility.evaluate_c0_complementary_semantic_verification import (
    load_frozen_e1_aligned_q,
)
from release_core.utility import compute_directional_cycle_utility


ROOT = Path(__file__).resolve().parents[2]
FROZEN_OUTPUT = (
    ROOT
    / "outputs/cyclic_utility/c0_complementary_semantic_verification_seed20"
    / "c0_predictions_and_scores.npz"
)
Q_ALIGNED_TENSOR_SHA256 = (
    "c0b48b300a9f041857dbd5a3bcf46aab6b3874397cc4af64f8e4e29b750a2a23"
)
U_TENSOR_SHA256 = (
    "b23dde6bd4f524e5e47748a73df3224f77e93d166d94f5a6c60db8865afefbfe"
)
U_NDARRAY_SHA256 = (
    "52dc7b5354daf9df189cd7d98c4e6cc8110f12182d1f82c24545eb8d11f66fe6"
)
GENERATOR_SHA256 = (
    "1da78797e059f952bf6ae3749fc94ca2dd5bebf83d76b1e8eeb6a03583fe36da"
)
VERIFIER_SHA256 = (
    "aa726d58832d5eb17ff952049d6be1b3c6eb7295b6fcba6b656c3c1ead80baca"
)


def _tensor_sha256(value):
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


def _ndarray_sha256(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(size) for size in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def test_caltech_same_state_frozen_utility_is_exact_without_ground_truth():
    q_aligned, _, sample_ids, audit = load_frozen_e1_aligned_q(device="cpu")
    assert q_aligned.shape == (1400, 6, 7)
    assert q_aligned.dtype == torch.float32
    assert np.array_equal(sample_ids, np.arange(1400, dtype=np.int64))
    assert audit["source_stage"] == "E1"
    assert _tensor_sha256(q_aligned.numpy()) == Q_ALIGNED_TENSOR_SHA256

    clean = compute_directional_cycle_utility(q_aligned)
    clean_array = clean["U_cycle"].numpy()
    with np.load(FROZEN_OUTPUT, allow_pickle=False) as archive:
        frozen_array = np.array(archive["U_cycle"], copy=True)

    assert clean["U_cycle"].shape == (1400, 20)
    assert clean["U_cycle"].dtype == torch.float32
    assert torch.isfinite(clean["U_cycle"]).all()
    assert np.array_equal(clean_array, frozen_array)
    assert _tensor_sha256(clean_array) == U_TENSOR_SHA256
    assert _ndarray_sha256(clean_array) == U_NDARRAY_SHA256
    assert clean["canonical_generator_hash"] == GENERATOR_SHA256
    assert clean["canonical_verifier_hash"] == VERIFIER_SHA256
