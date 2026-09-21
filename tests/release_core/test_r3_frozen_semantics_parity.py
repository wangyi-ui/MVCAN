import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from experiments.generic_contract import generic_relation_action as authoritative
from release_core.semantics import SparseLabelSplit, build_relation_semantics


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path("/root/autodl-tmp/CVPR24-MVCAN")
CALTECH_DIR = (
    ROOT / "outputs/cyclic_utility"
    / "c3_a0_utility_conditioned_action_granularity_seed20"
)
MSRC_ARTIFACT = (
    ROOT / "outputs/generic_contract"
    / "g0_b0_msrc_structural_pilot_seed20_20260916"
    / "msrc_structural_pre_gt_artifact.npz"
)
BDGP_ARTIFACT = (
    ARCHIVE / "outputs/generic_contract"
    / "g0_b1_bdgp_structural_pilot_seed20_20260917"
    / "bdgp_structural_pre_gt_artifact.npz"
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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


def test_caltech_seal_artifact_split_and_y_gen_identities():
    artifact = CALTECH_DIR / "c3_a0_action_pre_gt.npz"
    seal = CALTECH_DIR / "c3_a0_action_seal.json"
    assert _file_sha256(artifact) == (
        "b4f9ff0241b28ec9f6a8ec6c55c0f9da9c4631ffde2b032c0d3550340fdece71"
    )
    assert _file_sha256(seal) == (
        "0e23e9cea448435d04351fc6661ab8020a69b12a660f6e6604ff855ea96bb9d0"
    )
    assert json.loads(seal.read_text(encoding="utf-8"))["label_split_sha256"] == (
        "0463cf7155bc2b90a6133f0a79fd30d8c5a9e78079c6692dd1fa4106cb174487"
    )
    with np.load(artifact, allow_pickle=False) as archive:
        assert _ndarray_sha256(archive["y_gen"]) == (
            "90d6df0ff2583117a20c914b4a3c8583d0f6ff6aad58583a71821901442237b9"
        )


def test_caltech_clean_semantics_are_exact_same_state_replay():
    artifact = CALTECH_DIR / "c3_a0_action_pre_gt.npz"
    with np.load(artifact, allow_pickle=False) as archive:
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in (
                "sample_ids", "labeled_ids", "labeled_targets", "unlabeled_ids",
                "y_gen", "PredRelation_true", "relation_balance_weights_true",
            )
        }
    split = SparseLabelSplit(
        arrays["sample_ids"], arrays["labeled_ids"], arrays["labeled_targets"],
        arrays["unlabeled_ids"], 7, 2, 20, "Caltech-6V",
    )
    clean = build_relation_semantics(arrays["y_gen"], split)
    frozen = authoritative.build_relation_semantics(
        arrays["y_gen"], arrays["labeled_ids"], arrays["labeled_targets"],
        arrays["unlabeled_ids"], class_count=7, labels_per_class=2,
    )
    assert clean.pred_relation.shape == (1386, 14, 20)
    assert np.array_equal(clean.sparse_mapping, frozen["sparse_mapping"])
    assert np.array_equal(clean.class_pred, frozen["class_pred"])
    assert np.array_equal(clean.pred_relation, arrays["PredRelation_true"])
    assert np.array_equal(clean.balance_weights, arrays["relation_balance_weights_true"])
    assert _ndarray_sha256(clean.pred_relation) == (
        "a61460618339930b577c0b53e9fe6f15f4e40650e3522fdab34fa9a43d7bfca5"
    )
    assert _ndarray_sha256(clean.balance_weights) == (
        "1d7f1dcc65f49b37fdd6d0d8947456cdf828dff83936e10ca507cf1093d300aa"
    )


def _assert_sealed_output(
    path, file_hash, shape, pred_array_hash, pred_tensor_hash,
    balance_array_hash, balance_tensor_hash,
):
    assert _file_sha256(path) == file_hash
    with np.load(path, allow_pickle=False) as archive:
        assert "y_gen" not in archive.files
        predicted = np.array(archive["PredRelation_true"], copy=True)
        balance = np.array(archive["relation_balance_weights_true"], copy=True)
    assert predicted.shape == balance.shape == shape
    assert predicted.dtype == np.bool_
    assert balance.dtype == np.float64
    assert _ndarray_sha256(predicted) == pred_array_hash
    assert _tensor_sha256(predicted) == pred_tensor_hash
    assert _ndarray_sha256(balance) == balance_array_hash
    assert _tensor_sha256(balance) == balance_tensor_hash


def test_msrc_sealed_output_integrity_without_replay_claim():
    _assert_sealed_output(
        MSRC_ARTIFACT,
        "b04f107f6279839d534e55500fbb0f9aba94d7686944ab20f255337a511af234",
        (196, 14, 20),
        "cbd2d2de1dfd51abba9b8e80fcc1ee82f4e902df6532ca2af388e1266ff0d2b8",
        "3752fab36abcc607877b3f0083e678a63f2fa88cfd7a491a6e19235e3ca52bd1",
        "2d69642382947465851d69fe9f151df440979ce626481f9ad4c060909f92ff9a",
        "dc730ff68b4b0c719dbeb68a2f2ab8cddff8404fe2ea79302105b7ba3c9f67d6",
    )


def test_bdgp_sealed_output_integrity_without_replay_claim():
    _assert_sealed_output(
        BDGP_ARTIFACT,
        "6e929633526c037e5480ccfac8af496caac02e237489b012988641679f973029",
        (2490, 10, 2),
        "b4df46922ca857bea6d6c8db57bd00d71ec7a5fb3af3b66d1e7119d705bd45d4",
        "ddff6d37ef3474a5c94d23dea188c5457cef731841eb987e0ae1394b14dff404",
        "5e3d6ccf9c93822d96104a3974438a41ac9f236277c804a0ef59f885b9d75adb",
        "29fcd056e5c915d45380303c21bb21cbb155809ecc76856afb962e6a995af9ed",
    )
