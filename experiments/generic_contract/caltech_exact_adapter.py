"""Read-only Caltech-6V validation adapter for future G0-A1 exact replay.

This module connects generic G0 contracts to frozen seed20 references.  It
contains no training, model execution, ground-truth loading, or scientific
mechanism.  Gate6 remains deliberately unexecuted.
"""

import hashlib
import json
import subprocess
from pathlib import Path
from types import MappingProxyType

import numpy as np

from experiments.cyclic_utility import (
    c0_complementary_semantic_verification as c0,
)
from experiments.cyclic_utility import (
    c2_a0_sparse_label_utility_protocol as c2,
)
from experiments.cyclic_utility import (
    materialize_c3_b0_true_u_carrier as carrier_replay,
)
from experiments.cyclic_utility import (
    train_c3_b0_relation_action_pilot as c3_train,
)
from experiments.e1_pairwise_utility import train_e1_pairwise_utility as e1
from experiments.final_core import final_core_protocol as f0
from experiments.final_core import train_final_core as final_core_train
from irv.b4_information_utility import tensor_sha256
from weak_quality import ndarray_sha256

from .action_space import build_action_space, historical_action_logical_sha256
from .dataset_contract import DatasetContract
from .generic_weak_quality import generate_half_corruption_mask
from .sparse_label_contract import frozen_caltech_sparse_split


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

CALTECH_DATASET_NAME = "Caltech-6V"
CALTECH_SEED = 20
CALTECH_N = 1400
CALTECH_V = 6
CALTECH_K = 7
CALTECH_L = 14
CALTECH_NU = 1386
CALTECH_S = 20
CALTECH_VIEW_DIMS = (48, 40, 254, 1984, 512, 928)
CALTECH_SNR_DB = 2.5

G0_A0_PARENT_COMMIT = "71896919db9d108f96e6a5c15f648c030df8c957"
G0_A0_PARENT_TAG = "g0-a0-generic-primitives-presmoke-pass-20260915"
CALTECH_DATASET_FILE_SHA256 = (
    "72fa848269b663f819a8e9bd441628ece1955c654d98e2b10a85be1cd2613d5a"
)
SAMPLE_IDS_NDARRAY_SHA256 = (
    "887cb8dff9918e38f0b0c7f467d3b9aec75dcef9bf821b2307fd076ca5c42237"
)
GENERATOR_TENSOR_SHA256 = (
    "1da78797e059f952bf6ae3749fc94ca2dd5bebf83d76b1e8eeb6a03583fe36da"
)
VERIFIER_TENSOR_SHA256 = (
    "aa726d58832d5eb17ff952049d6be1b3c6eb7295b6fcba6b656c3c1ead80baca"
)
CORRUPTION_MASK_FILE_SHA256 = (
    "d549ab7a8b4f7740b30e28d23638c86c1aaf9b1ce77fb6698f6e708fd2a59a50"
)
CORRUPTION_MASK_NDARRAY_SHA256 = (
    "e6250535db6e33b9263102cf335fcf9e8552a24cae4b70ec5a94887f4e58bf0b"
)
CORRUPTION_AUDIT_FILE_SHA256 = (
    "553d34a722781d63503ee3bf001cf8916abadfefc8cf6fb7127731d0e01e56af"
)
U_CYCLE_NDARRAY_SHA256 = (
    "fb09bc2a69ef255868b27a0c036c4717c575b4403cbcaeef2f93d8702b6ffeeb"
)
U_CYCLE_TENSOR_SHA256 = (
    "66686413301567b85170a4ffe7ef1f32b88dbad1d96bf34e485bbd1dea5ffdff"
)
PRED_RELATION_NDARRAY_SHA256 = (
    "a61460618339930b577c0b53e9fe6f15f4e40650e3522fdab34fa9a43d7bfca5"
)
PRED_RELATION_TENSOR_SHA256 = (
    "4e72d2cdf118391caa8b586f1f8220d5cbf52a2e6bdf5bb5afa6a8a43ec19d5a"
)
BALANCE_WEIGHT_NDARRAY_SHA256 = (
    "1d7f1dcc65f49b37fdd6d0d8947456cdf828dff83936e10ca507cf1093d300aa"
)
BALANCE_WEIGHT_TENSOR_SHA256 = (
    "af9402d1adbe4cc7313277064360a68ff2e7b6a2e2ff88f57f3061b07df227f9"
)
REFERENCE_PREDICTION_FILE_SHA256 = (
    "9836306da4e64703c72ffe4f6959a2a2b51446e56cb6708ecc24e0c852295e2f"
)
PREDICTION_TENSOR_SHA256 = (
    "b3a40a90c5069a2fe6926b48efe406dc51c670b204ccc1bd53c4d322e830d504"
)
FINAL_MODEL_AGGREGATE_SHA256 = (
    "7a69d078e9573452cd9a1e98a9d72bbe618bf547dc0ba2bbdae3423bd053e747"
)
FINAL_MODEL_PER_VIEW_SHA256 = (
    "1f40f183786830a7bd30c1c563c852178a39da729352fa3dc09f7747cd611e1e",
    "28c11a0819bba5d452934d208dfd2e5898e7ebe96421c371337a8247411029b1",
    "251b6179ee81cecd127525a3ff19c9c6b132a8ccf342ff0f627768a8499a0f53",
    "16ce556f0e135e3259000c22bb8fec72124fd064aaef759ccec7f5bb319f30bb",
    "c4a808b548d77ec8c7dd7a04a7b7e09133b2fc237fbc05203e9da99b4175c4ef",
    "412d3c8eb216bbafb6abec3bb14cad0d4dbf212f6ee96e8918490613f98f5130",
)

C3_ACTION_FILE_SHA256 = (
    "b4f9ff0241b28ec9f6a8ec6c55c0f9da9c4631ffde2b032c0d3550340fdece71"
)
C3_ACTION_SEAL_SHA256 = (
    "0e23e9cea448435d04351fc6661ab8020a69b12a660f6e6604ff855ea96bb9d0"
)
CARRIER_ARTIFACT_SHA256 = (
    "a590ac3365b57d871b9b6687b238bdc7380baf4a9d7bba0143ebfaf1b031b696"
)
CARRIER_AUDIT_SHA256 = (
    "33c6988291ff162e224d1190eefc13e4e6cffbd8da7f83427d6d7514d72d9261"
)
CARRIER_SEAL_SHA256 = (
    "9cce5011fe2e85ad6946f7e96e2c3e6cc118b4cacb6a2998c8c9cf6bc95ea2f6"
)
F0_ARTIFACT_SHA256 = (
    "ec78a6102546254c7d0868f4948e1a79a236734ce37ab9d67b350063fefcc010"
)
F0_AUDIT_SHA256 = (
    "255234648a40e393059d097bfecba0e0ee822b38f3b6edae1c5f9a40c87302d4"
)
F0_SEAL_SHA256 = (
    "a85828fa1c914194b97df5704c99bfbabef720e33fdf60359d1c4916a870e54f"
)

CALTECH_DATASET_PATH = REPOSITORY_ROOT / "data/Caltech.mat"
CORRUPTION_MASK_PATH = REPOSITORY_ROOT / (
    "outputs/d1_caltech6v/snr2p5_k3_seed20/audit/corruption_mask.npy"
)
CORRUPTION_AUDIT_PATH = CORRUPTION_MASK_PATH.with_name("corruption_audit.json")
F0_ROOT = REPOSITORY_ROOT / "outputs/final_core/f0_a0_engineering_smoke_seed20"
F0_ARTIFACT_PATH = F0_ROOT / "final_core_pre_gt_artifact.npz"
F0_AUDIT_PATH = F0_ROOT / "final_core_pre_gt_audit.json"
F0_SEAL_PATH = F0_ROOT / "final_core_pre_gt_seal.json"
G0_A0_SOURCE_MANIFEST = REPOSITORY_ROOT / (
    "experiment_freeze/g0_a0_generic_primitives_implementation_presmoke_20260915/"
    "implementation_source_sha256.txt"
)

FUTURE_REPLAY_DELEGATE = (
    "experiments.final_core.train_final_core.run_exact_replay"
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _record(**values):
    return MappingProxyType(values)


def _file_sha256(path):
    target = Path(path)
    _require(target.is_file(), "missing frozen reference: " + str(target))
    digest = hashlib.sha256()
    with open(target, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path, expected, name):
    actual = _file_sha256(path)
    _require(actual == expected, name + " whole-file SHA256 mismatch")
    return actual


def _verify_manifest(path):
    target = Path(path)
    _require(target.is_file(), "missing frozen manifest: " + str(target))
    checked = 0
    with open(target, "r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if not stripped:
                continue
            expected, relative = stripped.split(None, 1)
            actual = _file_sha256(REPOSITORY_ROOT / relative.strip())
            _require(actual == expected, "frozen manifest entry mismatch")
            checked += 1
    _require(checked > 0, "empty frozen manifest")
    return checked


def _routes():
    routes = final_core_train.parent_routes(CALTECH_SEED)
    _require(routes.get("seed") == CALTECH_SEED, "seed20 route mismatch")
    return routes


def build_caltech_dataset_contract():
    """Construct the frozen Caltech shape contract without parsing data."""
    return DatasetContract(
        dataset_name=CALTECH_DATASET_NAME,
        N=CALTECH_N,
        V=CALTECH_V,
        K=CALTECH_K,
        view_dims=CALTECH_VIEW_DIMS,
        labels_per_class=2,
    )


def validate_caltech_dataset_contract():
    contract = build_caltech_dataset_contract()
    expected_shapes = {
        "q_local": (1400, 6, 7),
        "q_aligned": (1400, 6, 7),
        "U_cycle": (1400, 20),
        "labeled_ids": (14,),
        "unlabeled_ids": (1386,),
        "PredRelation": (1386, 14, 20),
        "relation_balance_weights": (1386, 14, 20),
        "final_predictions": (1400,),
    }
    _require(tuple(e1.VIEW_DIMS) == CALTECH_VIEW_DIMS, "historical view dims mismatch")
    _require(
        (contract.L, contract.N_u, contract.S) == (CALTECH_L, CALTECH_NU, CALTECH_S)
        and dict(contract.tensor_shapes) == expected_shapes,
        "Caltech dataset contract mismatch",
    )
    dataset_sha = _verify_file(
        CALTECH_DATASET_PATH, CALTECH_DATASET_FILE_SHA256, "Caltech dataset"
    )
    return _record(
        pass_status=True,
        contract=contract,
        tensor_shapes=MappingProxyType(expected_shapes),
        historical_view_dims_exact=True,
        dataset_file_sha256=dataset_sha,
        dataset_content_parsed=False,
    )


def validate_caltech_action_contract():
    generic = build_action_space(CALTECH_V)
    historical_generator, historical_verifier = c0.enumerate_complementary_splits()
    generator = np.asarray(generic.generator_subsets, dtype=np.int64)
    verifier = np.asarray(generic.verifier_subsets, dtype=np.int64)
    generator_hash = historical_action_logical_sha256(generator)
    verifier_hash = historical_action_logical_sha256(verifier)
    generator_equal = np.array_equal(generator, historical_generator)
    verifier_equal = np.array_equal(verifier, historical_verifier)
    _require(
        generic.S == CALTECH_S
        and generator_equal
        and verifier_equal
        and generator_hash == GENERATOR_TENSOR_SHA256
        and verifier_hash == VERIFIER_TENSOR_SHA256,
        "Caltech action contract mismatch",
    )
    return _record(
        pass_status=True,
        direction_count=generic.S,
        generator_exact_equal=True,
        verifier_exact_equal=True,
        generator_logical_sha256=generator_hash,
        verifier_logical_sha256=verifier_hash,
        action_order_exact=True,
    )


def validate_caltech_weak_quality_contract():
    mask_file_sha = _verify_file(
        CORRUPTION_MASK_PATH,
        CORRUPTION_MASK_FILE_SHA256,
        "corruption mask",
    )
    audit_file_sha = _verify_file(
        CORRUPTION_AUDIT_PATH,
        CORRUPTION_AUDIT_FILE_SHA256,
        "corruption audit",
    )
    authoritative = np.load(CORRUPTION_MASK_PATH, allow_pickle=False)
    generic, _ = generate_half_corruption_mask(CALTECH_N, CALTECH_V, CALTECH_SEED)
    with open(CORRUPTION_AUDIT_PATH, "r", encoding="utf-8") as input_file:
        audit = json.load(input_file)
    row_counts = authoritative.sum(axis=1, dtype=np.int64)
    view_counts = authoritative.sum(axis=0, dtype=np.int64)
    logical_hash = ndarray_sha256(authoritative)
    _require(
        authoritative.shape == (CALTECH_N, CALTECH_V)
        and authoritative.dtype == np.bool_
        and np.array_equal(generic, authoritative)
        and int(authoritative.sum()) == 4200
        and np.array_equal(row_counts, np.full(CALTECH_N, 3))
        and np.array_equal(view_counts, np.full(CALTECH_V, 700))
        and logical_hash == CORRUPTION_MASK_NDARRAY_SHA256
        and audit.get("mask_sha256") == logical_hash
        and audit.get("target_snr_db") == CALTECH_SNR_DB
        and audit.get("corruption_seed") == CALTECH_SEED,
        "Caltech weak-quality contract mismatch",
    )
    return _record(
        pass_status=True,
        mask_exact_equal=True,
        mask_shape=authoritative.shape,
        mask_dtype=str(authoritative.dtype),
        corrupted_pair_count=4200,
        per_row_count=3,
        per_view_counts=tuple(int(value) for value in view_counts),
        mask_file_sha256=mask_file_sha,
        mask_logical_sha256=logical_hash,
        audit_file_sha256=audit_file_sha,
        target_snr_db=CALTECH_SNR_DB,
        weak_quality_seed=CALTECH_SEED,
        gaussian_corruption_applied=False,
    )


def validate_caltech_sparse_label_contract():
    split = frozen_caltech_sparse_split()
    expected_ids = np.asarray(c2.FIXED_LABELED_IDS, dtype=np.int64)
    expected_targets = np.asarray(c2.FIXED_LABELED_TARGETS, dtype=np.int64)
    sample_ids = c2.canonical_sample_ids()
    expected_unlabeled = np.setdiff1d(sample_ids, expected_ids, assume_unique=True)
    counts = np.bincount(split.labeled_targets, minlength=CALTECH_K)
    sample_hash = ndarray_sha256(sample_ids)
    _require(
        np.array_equal(split.labeled_ids, expected_ids)
        and np.array_equal(split.labeled_targets, expected_targets)
        and np.array_equal(split.unlabeled_ids, expected_unlabeled)
        and np.array_equal(counts, np.full(CALTECH_K, 2))
        and split.labeled_ids.size == CALTECH_L
        and split.unlabeled_ids.size == CALTECH_NU
        and sample_hash == SAMPLE_IDS_NDARRAY_SHA256,
        "Caltech sparse-label contract mismatch",
    )
    return _record(
        pass_status=True,
        labeled_ids=tuple(int(value) for value in split.labeled_ids),
        labeled_targets=tuple(int(value) for value in split.labeled_targets),
        class_counts=tuple(int(value) for value in counts),
        unlabeled_count=int(split.unlabeled_ids.size),
        sample_ids_logical_sha256=sample_hash,
        full_ground_truth_loaded=False,
        generic_ranking_used=False,
    )


def load_and_validate_caltech_semantic_references():
    routes = _routes()
    action_path = routes["C3_A0_action_npz"]
    action_seal_path = routes["C3_A0_action_seal"]
    _verify_file(action_path, C3_ACTION_FILE_SHA256, "C3-A0 action artifact")
    _verify_file(action_seal_path, C3_ACTION_SEAL_SHA256, "C3-A0 action seal")
    required = ("U_cycle", "PredRelation_true", "relation_balance_weights_true")
    _require(set(required).issubset(c3_train.ACTION_FIELDS), "historical action fields mismatch")
    with np.load(action_path, allow_pickle=False) as archive:
        _require(set(required).issubset(archive.files), "semantic reference missing")
        arrays = {
            name: np.array(archive[name], copy=True, order="C")
            for name in required
        }
    expected = {
        "U_cycle": (
            (CALTECH_N, CALTECH_S), np.dtype("float64"),
            U_CYCLE_NDARRAY_SHA256, U_CYCLE_TENSOR_SHA256,
        ),
        "PredRelation_true": (
            (CALTECH_NU, CALTECH_L, CALTECH_S), np.dtype("bool"),
            PRED_RELATION_NDARRAY_SHA256, PRED_RELATION_TENSOR_SHA256,
        ),
        "relation_balance_weights_true": (
            (CALTECH_NU, CALTECH_L, CALTECH_S), np.dtype("float64"),
            BALANCE_WEIGHT_NDARRAY_SHA256, BALANCE_WEIGHT_TENSOR_SHA256,
        ),
    }
    records = {}
    for name, array in arrays.items():
        shape, dtype, parent_hash, f0_hash = expected[name]
        actual_parent = ndarray_sha256(array)
        actual_f0 = tensor_sha256(array)
        _require(
            array.shape == shape
            and array.dtype == dtype
            and np.isfinite(array).all()
            and actual_parent == parent_hash
            and actual_f0 == f0_hash,
            name + " exact reference mismatch",
        )
        records[name] = _record(
            shape=array.shape,
            dtype=str(array.dtype),
            parent_logical_sha256=actual_parent,
            f0_logical_sha256=actual_f0,
        )
    _require(
        np.all(arrays["U_cycle"] >= 0.0)
        and np.all(arrays["relation_balance_weights_true"] > 0.0),
        "semantic reference value boundary mismatch",
    )
    return _record(
        pass_status=True,
        action_artifact_path=str(action_path),
        action_artifact_file_sha256=C3_ACTION_FILE_SHA256,
        action_seal_path=str(action_seal_path),
        action_seal_file_sha256=C3_ACTION_SEAL_SHA256,
        arrays=MappingProxyType(records),
        references_reconstructed=False,
    )


def validate_caltech_frozen_prediction_reference():
    reference = carrier_replay.load_frozen_true_u_reference(CALTECH_SEED)
    _require(
        reference["prediction_file_sha256"] == REFERENCE_PREDICTION_FILE_SHA256
        and reference["predictions"].shape == (CALTECH_N,)
        and tensor_sha256(reference["predictions"]) == PREDICTION_TENSOR_SHA256
        and ndarray_sha256(reference["sample_ids"]) == SAMPLE_IDS_NDARRAY_SHA256,
        "frozen prediction reference mismatch",
    )
    with np.load(F0_ARTIFACT_PATH, allow_pickle=False) as archive:
        f0_predictions = np.array(archive["final_predictions"], copy=True)
        f0_sample_ids = np.array(archive["sample_ids"], copy=True)
        recorded_model_hash = str(archive["final_model_hash_aggregate"].item())
    model_hash = reference["final_model_hash"]
    _require(
        np.array_equal(f0_predictions, reference["predictions"])
        and np.array_equal(f0_sample_ids, reference["sample_ids"])
        and model_hash.get("aggregate") == FINAL_MODEL_AGGREGATE_SHA256
        and tuple(model_hash.get("per_view", ())) == FINAL_MODEL_PER_VIEW_SHA256
        and recorded_model_hash == FINAL_MODEL_AGGREGATE_SHA256,
        "frozen prediction/model parity mismatch",
    )
    return _record(
        pass_status=True,
        prediction_path=reference["prediction_path"],
        prediction_file_sha256=reference["prediction_file_sha256"],
        prediction_logical_sha256=PREDICTION_TENSOR_SHA256,
        sample_ids_logical_sha256=SAMPLE_IDS_NDARRAY_SHA256,
        final_predictions_exact_equal=True,
        final_sample_ids_exact_equal=True,
        final_model_hash_supported=True,
        final_model_hash_equal=True,
        final_model_aggregate_sha256=FINAL_MODEL_AGGREGATE_SHA256,
        final_model_per_view_sha256=FINAL_MODEL_PER_VIEW_SHA256,
    )


def validate_caltech_f0_pre_gt_reference():
    _verify_file(F0_ARTIFACT_PATH, F0_ARTIFACT_SHA256, "F0 pre-GT artifact")
    _verify_file(F0_AUDIT_PATH, F0_AUDIT_SHA256, "F0 pre-GT audit")
    _verify_file(F0_SEAL_PATH, F0_SEAL_SHA256, "F0 pre-GT seal")
    _, validation = f0.validate_pre_gt_seal(
        CALTECH_SEED, F0_ARTIFACT_PATH, F0_AUDIT_PATH, F0_SEAL_PATH
    )
    _require(
        validation.get("pre_gt_seal_valid") is True
        and validation.get("pre_gt_seal_verified_before_GT_load") is True,
        "F0 pre-GT seal validation mismatch",
    )
    return _record(
        pass_status=True,
        artifact_path=str(F0_ARTIFACT_PATH),
        artifact_file_sha256=F0_ARTIFACT_SHA256,
        audit_path=str(F0_AUDIT_PATH),
        audit_file_sha256=F0_AUDIT_SHA256,
        seal_path=str(F0_SEAL_PATH),
        seal_file_sha256=F0_SEAL_SHA256,
        pre_gt_seal_valid=True,
        full_ground_truth_loaded=False,
    )


def validate_caltech_parent_integrity():
    tag_commit = subprocess.check_output(
        ["git", "rev-parse", G0_A0_PARENT_TAG + "^{commit}"],
        cwd=REPOSITORY_ROOT,
        text=True,
    ).strip()
    _require(tag_commit == G0_A0_PARENT_COMMIT, "G0-A0 parent tag mismatch")
    generic_entry_count = _verify_manifest(G0_A0_SOURCE_MANIFEST)
    integrity = final_core_train.verify_parent_integrity()
    summary = integrity.get("summary", {})
    required = (
        "all_parent_hashes_pass",
        "frozen_C3_B0_source_hashes_pass",
        "frozen_C3_B0_formal_output_hashes_pass",
        "frozen_carrier_source_hashes_pass",
        "frozen_carrier_output_hashes_pass",
    )
    _require(
        integrity.get("all_parent_hashes_pass") is True
        and all(summary.get(name) is True for name in required),
        "Caltech frozen parent integrity mismatch",
    )
    return _record(
        pass_status=True,
        g0_a0_parent_tag=G0_A0_PARENT_TAG,
        g0_a0_parent_commit=tag_commit,
        generic_source_manifest_entry_count=generic_entry_count,
        all_parent_hashes_pass=True,
        frozen_C3_B0_source_hashes_pass=True,
        frozen_C3_B0_formal_output_hashes_pass=True,
        frozen_carrier_source_hashes_pass=True,
        frozen_carrier_output_hashes_pass=True,
    )


def _gate_readiness_from(records):
    _require(all(record["pass_status"] is True for record in records), "Gate0-Gate5 mismatch")
    return _record(
        Gate0_parent_integrity="PASS",
        Gate1_dataset_contract="PASS",
        Gate2_action_space="PASS",
        Gate3_weak_quality="PASS",
        Gate4_sparse_labels="PASS",
        Gate5_U_and_relation_semantics="PASS",
        Gate6_exact_replay="NOT_RUN",
        Overall_G0_A1="NOT_COMPLETE",
    )


def build_caltech_gate_readiness():
    """Validate Gate0-Gate5 and keep the future exact replay gate closed."""
    records = (
        validate_caltech_parent_integrity(),
        validate_caltech_dataset_contract(),
        validate_caltech_action_contract(),
        validate_caltech_weak_quality_contract(),
        validate_caltech_sparse_label_contract(),
        load_and_validate_caltech_semantic_references(),
    )
    return _gate_readiness_from(records)


def audit_caltech_exact_adapter():
    """Run only read-only pre-training validation; never execute Gate6."""
    parent = validate_caltech_parent_integrity()
    dataset = validate_caltech_dataset_contract()
    action = validate_caltech_action_contract()
    weak_quality = validate_caltech_weak_quality_contract()
    sparse = validate_caltech_sparse_label_contract()
    semantics = load_and_validate_caltech_semantic_references()
    prediction = validate_caltech_frozen_prediction_reference()
    pre_gt = validate_caltech_f0_pre_gt_reference()
    gates = _gate_readiness_from(
        (parent, dataset, action, weak_quality, sparse, semantics)
    )
    return _record(
        stage="G0-A1-A0",
        parent=parent,
        dataset=dataset,
        action=action,
        weak_quality=weak_quality,
        sparse_labels=sparse,
        semantic_references=semantics,
        prediction_reference=prediction,
        f0_pre_gt_reference=pre_gt,
        gate_readiness=gates,
        future_replay_delegate=FUTURE_REPLAY_DELEGATE,
        exact_replay_run=False,
        training_run=False,
        full_ground_truth_loaded=False,
        overall_G0_A1="NOT_COMPLETE",
    )
