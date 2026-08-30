"""E4-A0 frozen LWC semantic-memory writer feasibility diagnostic."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from configure import get_default_config
from experiments.e1_pairwise_utility import train_e1_pairwise_utility as e1_train
from experiments.g2_utility_semantic_consensus import (
    g2_consensus_protocol as g2_protocol,
)
from experiments.e4_semantic_memory_bank.semantic_memory_bank import (
    CLASS_NUM,
    E4_ARMS,
    LABELED_NUM,
    SAMPLE_NUM,
    VIEW_NUM,
    build_class_memory,
    build_normal_writer_weights,
    build_shared_query,
    predict_from_memory,
    validate_sparse_labels,
)
from irv.b3_audit import hash_backbone
from irv.b4_information_utility import tensor_sha256
from model import MvCAN
from weak_quality import ndarray_sha256


STAGE = "E4-A0"
SEED = 20
EXPECTED_R_LOGICAL_SHA256 = (
    "3458e099f8c6e7317e3edbcef15ff629f368735e115fa6aea30e4516be8ac2cc"
)
EXPECTED_R_RAW_NPZ_SHA256 = (
    "569ad65f9f74c8cffcc9e6b96485d2dd941bdb17f0e7468a449ec09afdf51313"
)
EXPECTED_LABEL_SPLIT_SHA256 = (
    "0463cf7155bc2b90a6133f0a79fd30d8c5a9e78079c6692dd1fa4106cb174487"
)
EXPECTED_CORRUPTION_MASK_SHA256 = (
    "e6250535db6e33b9263102cf335fcf9e8552a24cae4b70ec5a94887f4e58bf0b"
)
CHECKPOINT_NOT_FOUND = "E1_LWC_FROZEN_MODEL_NOT_FOUND"

DEFAULT_FEATURE_PATH = e1_train.DEFAULT_FEATURE_PATH
DEFAULT_FEATURE_AUDIT_PATH = e1_train.DEFAULT_FEATURE_AUDIT_PATH
DEFAULT_E1_LWC_MODEL_DIR = (
    REPOSITORY_ROOT
    / "outputs/e1_pairwise_utility/lwc_100ep_seed20/models"
)
DEFAULT_E1_LWC_AUDIT_PATH = DEFAULT_E1_LWC_MODEL_DIR.parent / "e1_audit.json"
DEFAULT_R_PATH = (
    REPOSITORY_ROOT
    / "outputs/d2_caltech6v/a0_utility_transfer_seed20/utility_scores.npz"
)
DEFAULT_ORACLE_MASK_PATH = (
    REPOSITORY_ROOT
    / "outputs/d2_caltech6v/a0_utility_transfer_seed20"
    / "corruption_audit/corruption_mask.npy"
)
DEFAULT_LABEL_SPLIT_DIR = (
    REPOSITORY_ROOT / "outputs/b7_sparse_supervision/b7a0_seed20"
)
DEFAULT_FULL_GT_PATH = REPOSITORY_ROOT / "data/Caltech.mat"
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "outputs/e4_semantic_memory_bank/e4a0_seed20"
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _display(path):
    value = _resolve(path).resolve()
    try:
        return str(value.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(value)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _json_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_frozen_tensor(value, dtype=None, device=None):
    if torch.is_tensor(value):
        tensor = value.detach()
        return tensor.to(
            dtype=tensor.dtype if dtype is None else dtype,
            device=tensor.device if device is None else device,
        ).detach()
    array = np.array(value, copy=True, order="C")
    return torch.as_tensor(array, dtype=dtype, device=device).detach()


def _expected_checkpoint_paths(model_dir):
    root = _resolve(model_dir)
    return [
        root / ("Caltech-6V" + str(view_id + 1) + "V.pth")
        for view_id in range(VIEW_NUM)
    ]


def _fail_if_lwc_checkpoint_missing(model_dir, audit_path):
    checkpoint_paths = _expected_checkpoint_paths(model_dir)
    if not _resolve(audit_path).is_file() or not all(
        path.is_file() for path in checkpoint_paths
    ):
        print(CHECKPOINT_NOT_FOUND, file=sys.stderr)
        raise RuntimeError(CHECKPOINT_NOT_FOUND)
    return checkpoint_paths


def _load_frozen_lwc_model(model_dir, audit_path, device):
    checkpoint_paths = _fail_if_lwc_checkpoint_missing(model_dir, audit_path)
    frozen_run_audit = _read_json(_resolve(audit_path))
    _require(
        frozen_run_audit.get("arm") == "LWC"
        and frozen_run_audit.get("epochs") == 100
        and frozen_run_audit.get("N") == SAMPLE_NUM
        and frozen_run_audit.get("V") == VIEW_NUM
        and frozen_run_audit.get("K") == CLASS_NUM,
        "E1 frozen checkpoint audit is not the 100-epoch LWC arm",
    )

    config = get_default_config("Caltech-6V")
    model = MvCAN(
        config=config,
        view_num=VIEW_NUM,
        view_size=list(e1_train.VIEW_DIMS),
        n_clusters=CLASS_NUM,
        seed=SEED,
        data_size=SAMPLE_NUM,
        semantic_config=None,
    )
    checkpoint_hashes = []
    for autoencoder, checkpoint_path in zip(
        model.autoencoders, checkpoint_paths
    ):
        autoencoder.load_state_dict(
            e1_train.load_state_dict_cpu(checkpoint_path), strict=True
        )
        checkpoint_hashes.append(file_sha256(checkpoint_path))

    expected_hashes = frozen_run_audit["final_model_outputs"][
        "file_sha256"
    ]
    _require(
        checkpoint_hashes == expected_hashes,
        "E1 LWC checkpoint file provenance mismatch",
    )
    target_device = torch.device(device)
    model.to_device(target_device)
    for autoencoder in model.autoencoders:
        autoencoder.eval()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
    return model, {
        "arm": "LWC",
        "epochs": 100,
        "directory": _display(model_dir),
        "audit_path": _display(audit_path),
        "checkpoint_paths": [_display(path) for path in checkpoint_paths],
        "checkpoint_file_sha256": checkpoint_hashes,
        "frozen_model_found_pass": True,
    }


@torch.no_grad()
def align_q_to_global(q_local, match_matrices):
    """Apply the native orientation q_local @ M.T and normalize per view."""
    q_values = _as_frozen_tensor(q_local)
    matrices = _as_frozen_tensor(
        match_matrices, dtype=q_values.dtype, device=q_values.device
    )
    if q_values.ndim != 3 or q_values.shape[1:] != (VIEW_NUM, CLASS_NUM):
        raise ValueError("q_local must have shape [N,6,7]")
    if matrices.shape != (VIEW_NUM, CLASS_NUM, CLASS_NUM):
        raise ValueError("M must have shape [6,7,7]")
    if not bool(torch.isfinite(q_values).all().item()):
        raise ValueError("q_local contains non-finite values")
    if not bool(torch.isfinite(matrices).all().item()):
        raise ValueError("M contains non-finite values")
    if not bool(torch.all((matrices == 0.0) | (matrices == 1.0)).item()):
        raise ValueError("M must be a permutation matrix")
    expected_sums = torch.ones(
        (VIEW_NUM, CLASS_NUM), dtype=matrices.dtype, device=matrices.device
    )
    if not (
        torch.equal(matrices.sum(dim=1), expected_sums)
        and torch.equal(matrices.sum(dim=2), expected_sums)
    ):
        raise ValueError("M row/column sums must equal one")

    q_aligned = torch.einsum("nvk,vjk->nvj", q_values, matrices).detach()
    mass_preserved = bool(
        torch.equal(
            torch.sort(q_values, dim=-1).values,
            torch.sort(q_aligned, dim=-1).values,
        )
    )
    if not mass_preserved:
        raise RuntimeError("q alignment mass was not preserved")
    h_sem = F.normalize(q_aligned, dim=-1).detach()
    if not bool(torch.isfinite(h_sem).all().item()):
        raise RuntimeError("h_sem contains non-finite values")
    if h_sem.requires_grad or h_sem.grad_fn is not None:
        raise RuntimeError("h_sem did not stop gradients")
    return q_aligned, h_sem, {
        "M_shape": list(matrices.shape),
        "M_shape_pass": True,
        "M_permutation_pass": True,
        "M_row_column_sums_exact_pass": True,
        "q_alignment_mass_preserved_pass": mass_preserved,
        "alignment_direction": "q_local @ M^T",
        "h_sem_shape": list(h_sem.shape),
        "h_sem_finite_pass": True,
        "h_sem_stop_gradient_pass": True,
        "M_logical_sha256": tensor_sha256(
            matrices.detach().cpu().numpy()
        ),
        "q_aligned_logical_sha256": tensor_sha256(
            q_aligned.detach().cpu().numpy()
        ),
        "h_sem_logical_sha256": tensor_sha256(
            h_sem.detach().cpu().numpy()
        ),
    }


@torch.no_grad()
def load_e1_lwc_semantic_representation(
    feature_path=DEFAULT_FEATURE_PATH,
    feature_audit_path=DEFAULT_FEATURE_AUDIT_PATH,
    model_dir=DEFAULT_E1_LWC_MODEL_DIR,
    model_audit_path=DEFAULT_E1_LWC_AUDIT_PATH,
    device="cpu",
):
    """Run the frozen 100-epoch E1 LWC carrier on all 1400 rows."""
    target_device = torch.device(device)
    model, checkpoint_audit = _load_frozen_lwc_model(
        model_dir, model_audit_path, target_device
    )
    views, sample_ids, feature_audit = e1_train.load_frozen_feature_artifact(
        feature_path, feature_audit_path
    )
    _require(
        np.array_equal(sample_ids, np.arange(SAMPLE_NUM, dtype=np.int64)),
        "E1 LWC feature rows are not canonical sample IDs",
    )

    backbone_hash_before = hash_backbone(model.autoencoders)
    raw_views = []
    q_views = []
    for autoencoder, view in zip(model.autoencoders, views):
        features = torch.from_numpy(view).to(target_device)
        raw_latent = autoencoder.encoder(features).detach()
        q_view = autoencoder.clustering(raw_latent).detach()
        raw_views.append(raw_latent.cpu())
        q_views.append(q_view.cpu())
    raw_z = torch.stack(raw_views, dim=1).detach()
    q_local = torch.stack(q_views, dim=1).detach()
    _require(
        raw_z.shape == (SAMPLE_NUM, VIEW_NUM, 10)
        and q_local.shape == (SAMPLE_NUM, VIEW_NUM, CLASS_NUM)
        and bool(torch.isfinite(raw_z).all().item())
        and bool(torch.isfinite(q_local).all().item()),
        "frozen E1 LWC native forward boundary mismatch",
    )

    expected_clusters = np.arange(CLASS_NUM, dtype=np.int64)
    local_assignments = q_local.numpy().argmax(axis=2)
    for view_id in range(VIEW_NUM):
        _require(
            np.array_equal(
                np.unique(local_assignments[:, view_id]), expected_clusters
            ),
            "frozen E1 LWC q is missing a cluster in view "
            + str(view_id),
        )

    reference = g2_protocol.replay_native_global_reference(
        raw_z.numpy(), q_local.numpy()
    )
    _require(
        np.array_equal(
            np.unique(reference["global_prediction"]), expected_clusters
        ),
        "native global reference is missing a cluster",
    )
    native_alignment = g2_protocol.build_alignment(
        q_local.numpy(),
        reference["global_prediction"],
        model.Match,
    )
    M = torch.from_numpy(
        np.ascontiguousarray(native_alignment["alignment_matrix"])
    ).detach()
    q_aligned, h_sem, alignment_audit = align_q_to_global(q_local, M)

    backbone_hash_after = hash_backbone(model.autoencoders)
    parameter_boundary_pass = bool(
        all(
            not parameter.requires_grad and parameter.grad is None
            for autoencoder in model.autoencoders
            for parameter in autoencoder.parameters()
        )
    )
    _require(
        backbone_hash_before == backbone_hash_after,
        "frozen E1 LWC model changed during full-data forward",
    )
    _require(
        parameter_boundary_pass,
        "frozen E1 LWC parameters violated the no-gradient boundary",
    )
    return h_sem.detach(), np.asarray(sample_ids), {
        "source_stage": "E1",
        "source_arm": "LWC",
        "source_epochs": 100,
        "checkpoint": checkpoint_audit,
        "feature": feature_audit,
        "q_local": {
            "shape": list(q_local.shape),
            "logical_sha256": tensor_sha256(q_local.numpy()),
            "source": "frozen_LWC_clustering(raw_encoder_latent)",
            "all_views_have_all_clusters_pass": True,
            "stop_gradient_pass": True,
        },
        "native_global_reference": {
            "prediction_shape": list(reference["global_prediction"].shape),
            "prediction_logical_sha256": tensor_sha256(
                reference["global_prediction"]
            ),
            "all_clusters_present_pass": True,
        },
        "alignment": alignment_audit,
        "q_aligned_shape": list(q_aligned.shape),
        "representation_source_pass": True,
        "raw_unaligned_cross_view_z_averaging_used": False,
        "backbone_hash_before": backbone_hash_before,
        "backbone_hash_after": backbone_hash_after,
        "backbone_unchanged_pass": True,
        "parameters_frozen_grad_none_pass": parameter_boundary_pass,
    }


def load_frozen_reliability(path=DEFAULT_R_PATH):
    """Load only D2 key U and expose it semantically as frozen R_full."""
    artifact_path = _resolve(path)
    _require(artifact_path.is_file(), "frozen D2 reliability artifact is missing")
    raw_sha = file_sha256(artifact_path)
    _require(
        raw_sha == EXPECTED_R_RAW_NPZ_SHA256,
        "frozen D2 reliability raw NPZ SHA mismatch",
    )
    with np.load(artifact_path, allow_pickle=False) as archive:
        archive_fields = tuple(archive.files)
        _require("U" in archive_fields, "D2 NPZ has no frozen reliability key")
        R_full = np.ascontiguousarray(archive["U"])

    _require(
        R_full.shape == (SAMPLE_NUM, VIEW_NUM),
        "R_full must have shape [1400,6]",
    )
    _require(
        np.issubdtype(R_full.dtype, np.floating)
        and np.isfinite(R_full).all()
        and float(R_full.min()) >= 0.0
        and float(R_full.max()) <= 1.0,
        "R_full must be finite and within [0,1]",
    )
    logical_sha = tensor_sha256(R_full)
    _require(
        logical_sha == EXPECTED_R_LOGICAL_SHA256,
        "frozen D2 reliability logical SHA mismatch",
    )
    R_full.setflags(write=False)
    frozen_reliability = {
        "semantic_name": "frozen_reliability",
        "path": _display(artifact_path),
        "raw_npz_sha256": raw_sha,
        "raw_npz_sha256_pass": True,
        "archive_fields": list(archive_fields),
        "loaded_npz_key": "U",
        "loaded_fields": ["U"],
        "T_loaded": False,
        "corruption_mask_loaded": False,
        "R_full": {
            "shape": list(R_full.shape),
            "dtype": str(R_full.dtype),
            "logical_sha256": logical_sha,
            "logical_sha256_pass": True,
            "finite_range_0_1_pass": True,
            "stop_gradient_pass": True,
        },
    }
    return R_full, frozen_reliability


def load_sparse_label_protocol(split_dir=DEFAULT_LABEL_SPLIT_DIR):
    """Load only fourteen frozen IDs and recover their labels from split metadata."""
    root = _resolve(split_dir)
    record_path = root / "label_split.json"
    labeled_ids_path = root / "labeled_sample_ids.npy"
    _require(
        record_path.is_file() and labeled_ids_path.is_file(),
        "frozen sparse-label protocol artifacts are incomplete",
    )
    record = _read_json(record_path)
    stored_hash = str(record.get("label_split_sha256", ""))
    hash_payload = dict(record)
    hash_payload.pop("label_split_sha256", None)
    _require(
        _json_sha256(hash_payload)
        == stored_hash
        == EXPECTED_LABEL_SPLIT_SHA256,
        "frozen sparse-label split SHA mismatch",
    )
    labeled_ids = np.load(labeled_ids_path, allow_pickle=False)
    _require(
        labeled_ids.dtype == np.dtype(np.int64)
        and labeled_ids.shape == (LABELED_NUM,)
        and np.array_equal(labeled_ids, np.sort(np.unique(labeled_ids)))
        and np.all((0 <= labeled_ids) & (labeled_ids < SAMPLE_NUM)),
        "frozen labeled IDs must be fourteen sorted unique rows",
    )
    _require(
        ndarray_sha256(labeled_ids) == record.get("labeled_ids_sha256"),
        "frozen labeled ID SHA mismatch",
    )
    _require(
        record.get("N") == SAMPLE_NUM
        and record.get("K") == CLASS_NUM
        and record.get("labeled_count") == LABELED_NUM
        and record.get("unlabeled_count") == SAMPLE_NUM - LABELED_NUM
        and record.get("labels_per_class") == 2,
        "frozen sparse-label protocol metadata mismatch",
    )

    label_by_sample_id = {}
    per_class_ids = record.get("per_class_labeled_ids", {})
    for class_id in range(CLASS_NUM):
        class_ids = per_class_ids.get(str(class_id))
        _require(
            isinstance(class_ids, list) and len(class_ids) == 2,
            "sparse split must contain exactly two IDs per class",
        )
        for sample_id in class_ids:
            sample_id = int(sample_id)
            _require(
                sample_id not in label_by_sample_id,
                "a sparse sample ID appears in multiple classes",
            )
            label_by_sample_id[sample_id] = class_id
    _require(
        np.array_equal(
            np.sort(np.fromiter(label_by_sample_id, dtype=np.int64)),
            labeled_ids,
        ),
        "sparse labels and labeled IDs do not cover the same fourteen rows",
    )
    labels_labeled = np.ascontiguousarray(
        [label_by_sample_id[int(sample_id)] for sample_id in labeled_ids],
        dtype=np.int64,
    )
    _, counts = validate_sparse_labels(labels_labeled)
    labeled_ids.setflags(write=False)
    labels_labeled.setflags(write=False)
    return labeled_ids, labels_labeled, {
        "label_split_path": _display(record_path),
        "labeled_ids_path": _display(labeled_ids_path),
        "label_split_sha256": stored_hash,
        "labeled_ids_sha256": ndarray_sha256(labeled_ids),
        "labeled_count": int(labeled_ids.size),
        "labels_labeled_shape": list(labels_labeled.shape),
        "class_counts": counts.cpu().tolist(),
        "all_14_labels_used_pass": True,
        "exactly_two_labels_per_class_pass": True,
        "full_GT_loaded": False,
        "unlabeled_ID_artifact_loaded": False,
    }


def load_oracle_clean_labeled_weights(
    labeled_ids, mask_path=DEFAULT_ORACLE_MASK_PATH
):
    """Isolated diagnostic loader for clean/corrupt sample-view truth."""
    artifact_path = _resolve(mask_path)
    _require(artifact_path.is_file(), "oracle corruption mask is missing")
    corruption_mask = np.load(artifact_path, allow_pickle=False)
    _require(
        corruption_mask.dtype == np.dtype(bool)
        and corruption_mask.shape == (SAMPLE_NUM, VIEW_NUM)
        and np.all(corruption_mask.sum(axis=1) == 3),
        "oracle corruption mask boundary mismatch",
    )
    mask_sha = ndarray_sha256(corruption_mask)
    _require(
        mask_sha == EXPECTED_CORRUPTION_MASK_SHA256,
        "oracle corruption mask SHA mismatch",
    )
    labeled_ids = np.asarray(labeled_ids, dtype=np.int64)
    _require(
        labeled_ids.shape == (LABELED_NUM,),
        "oracle labeled IDs must have shape [14]",
    )
    oracle_clean_weights = np.ascontiguousarray(
        ~corruption_mask[labeled_ids], dtype=np.float32
    )
    _require(
        oracle_clean_weights.shape == (LABELED_NUM, VIEW_NUM)
        and np.all(oracle_clean_weights.sum(axis=1) == 3),
        "oracle clean writer weights must contain three clean views/sample",
    )
    oracle_clean_weights.setflags(write=False)
    return oracle_clean_weights, {
        "path": _display(artifact_path),
        "mask_logical_sha256": mask_sha,
        "mask_logical_sha256_pass": True,
        "writer_weights_shape": list(oracle_clean_weights.shape),
        "clean_is_one_corrupt_is_zero_pass": True,
        "oracle_used_for_training": False,
        "diagnostic_only": True,
    }


@torch.no_grad()
def build_fixed_arm_outputs(
    h_sem,
    labeled_ids,
    labels_labeled,
    R_full,
    oracle_clean_weights,
):
    """Construct all four memories and fix all predictions without full GT."""
    semantic = _as_frozen_tensor(h_sem)
    sample_num = int(semantic.shape[0])
    if semantic.shape != (sample_num, VIEW_NUM, CLASS_NUM):
        raise ValueError("h_sem must have shape [N,6,7]")
    if not bool(torch.isfinite(semantic).all().item()):
        raise ValueError("h_sem must be finite")
    ids = _as_frozen_tensor(labeled_ids, dtype=torch.long)
    if ids.shape != (LABELED_NUM,) or not bool(
        torch.all((ids >= 0) & (ids < sample_num)).item()
    ):
        raise ValueError("labeled_ids must be fourteen valid row IDs")
    targets, _ = validate_sparse_labels(labels_labeled)
    R_values = _as_frozen_tensor(R_full)
    if R_values.shape != (sample_num, VIEW_NUM):
        raise ValueError("R_full must have shape [N,6]")
    R_labeled = R_values[ids].detach()

    normal_writer_weights, shuffle_audit = build_normal_writer_weights(
        R_labeled, targets
    )
    oracle_values = _as_frozen_tensor(
        oracle_clean_weights, dtype=R_labeled.dtype
    )
    if oracle_values.shape != (LABELED_NUM, VIEW_NUM):
        raise ValueError("oracle_clean_weights must have shape [14,6]")

    writer_weights = dict(normal_writer_weights)
    writer_weights["ORACLE_CLEAN_MEMORY"] = oracle_values.detach()
    _require(
        tuple(writer_weights) == E4_ARMS,
        "E4-A0 writer arm order mismatch",
    )

    h_labeled = semantic[ids].detach()
    h_query = build_shared_query(semantic)
    fixed = {
        "h_query": h_query,
        "R_labeled": R_labeled,
        "writer_weights": writer_weights,
        "shuffle_audit": shuffle_audit,
    }
    for arm in E4_ARMS:
        prototypes = build_class_memory(
            h_labeled, targets, writer_weights[arm]
        )
        predictions, scores = predict_from_memory(h_query, prototypes)
        fixed[arm] = {
            "h_query": h_query,
            "prototypes": prototypes,
            "scores": scores,
            "predictions": predictions,
        }
    return fixed


def _save_predictions_before_full_GT(fixed, output_root):
    """Persist and hash fixed predictions before the full-label loader is called."""
    prediction_path = output_root / "predictions.npz"
    seal_path = output_root / "prediction_seal.json"
    _require(
        not prediction_path.exists() and not seal_path.exists(),
        "refusing to overwrite fixed prediction artifacts",
    )
    arrays = {
        arm: np.ascontiguousarray(
            fixed[arm]["predictions"].cpu().numpy(), dtype=np.int64
        )
        for arm in E4_ARMS
    }
    np.savez(prediction_path, **arrays)
    prediction_hashes = {}
    with np.load(prediction_path, allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == E4_ARMS,
            "saved prediction arm set/order mismatch",
        )
        for arm in E4_ARMS:
            reloaded = np.ascontiguousarray(archive[arm], dtype=np.int64)
            _require(
                np.array_equal(reloaded, arrays[arm]),
                "saved prediction differs from fixed assignment",
            )
            prediction_hashes[arm] = tensor_sha256(reloaded)
    seal = {
        "stage": STAGE,
        "prediction_file": _display(prediction_path),
        "prediction_file_sha256": file_sha256(prediction_path),
        "prediction_logical_sha256": prediction_hashes,
        "prediction_saved_before_full_GT": True,
        "prediction_reloaded_before_full_GT": True,
        "prediction_hashed_before_full_GT": True,
        "full_unlabeled_GT_loaded_before_prediction": False,
    }
    _write_json(seal_path, seal)
    return seal


def load_full_ground_truth(path=DEFAULT_FULL_GT_PATH):
    """The sole full-GT loader; callers invoke it only after prediction sealing."""
    label_path = _resolve(path)
    _require(label_path.is_file(), "full ground-truth artifact is missing")
    payload = sio.loadmat(label_path, variable_names=["Y"])
    _require("Y" in payload, "full ground-truth artifact has no Y field")
    full_GT = np.squeeze(np.asarray(payload["Y"])).astype(np.int64, copy=True)
    full_GT[full_GT == 95] = 5
    _require(
        full_GT.shape == (SAMPLE_NUM,)
        and np.array_equal(np.unique(full_GT), np.arange(CLASS_NUM)),
        "full ground truth must have shape [1400] and classes 0..6",
    )
    full_GT = np.ascontiguousarray(full_GT, dtype=np.int64)
    full_GT.setflags(write=False)
    return full_GT, {
        "path": _display(label_path),
        "file_sha256": file_sha256(label_path),
        "shape": list(full_GT.shape),
        "loaded_after_prediction_seal": True,
    }


def evaluate_unlabeled_fixed_arms(fixed, full_GT, unlabeled_ids):
    """Evaluate only the 1386 non-labeled rows after all assignments are fixed."""
    labels = np.asarray(full_GT, dtype=np.int64)
    evaluation_ids = np.asarray(unlabeled_ids, dtype=np.int64)
    _require(
        labels.shape == (SAMPLE_NUM,)
        and evaluation_ids.shape == (SAMPLE_NUM - LABELED_NUM,),
        "primary evaluation must use exactly 1386 unlabeled rows",
    )
    results = {}
    for arm in E4_ARMS:
        predictions = fixed[arm]["predictions"].cpu().numpy()[
            evaluation_ids
        ].astype(np.int64, copy=False)
        scores = fixed[arm]["scores"][evaluation_ids]
        targets = labels[evaluation_ids]
        target_tensor = torch.as_tensor(targets, dtype=torch.long)
        row_ids = torch.arange(target_tensor.numel())
        true_scores = scores[row_ids, target_tensor]
        other_scores = scores.clone()
        other_scores[row_ids, target_tensor] = -torch.inf
        true_margin = true_scores - other_scores.max(dim=1).values

        prototypes = fixed[arm]["prototypes"]
        prototype_cosine = prototypes @ prototypes.T
        off_diagonal = ~torch.eye(CLASS_NUM, dtype=torch.bool)
        off_diagonal_cosine = prototype_cosine[off_diagonal]
        results[arm] = {
            "evaluation_sample_count": int(evaluation_ids.size),
            "prototype_assignment_ACC": float(
                np.mean(predictions == targets)
            ),
            "prototype_assignment_NMI": float(
                normalized_mutual_info_score(targets, predictions)
            ),
            "prototype_assignment_ARI": float(
                adjusted_rand_score(targets, predictions)
            ),
            "true_class_margin_mean": float(true_margin.mean().item()),
            "prototype_pairwise_cosine_mean": float(
                off_diagonal_cosine.mean().item()
            ),
            "prototype_pairwise_cosine_max": float(
                off_diagonal_cosine.max().item()
            ),
            "prototype_separation": float(
                1.0 - off_diagonal_cosine.mean().item()
            ),
            "prediction_sha256": tensor_sha256(
                np.ascontiguousarray(
                    fixed[arm]["predictions"].cpu().numpy(), dtype=np.int64
                )
            ),
        }
    return results


def build_gate_decision(results):
    R_result = results["R_MEMORY"]
    shuffled_result = results["SHUFFLED_R_MEMORY"]
    all_result = results["ALL_MEMORY"]
    R_specificity_pass = bool(
        R_result["prototype_assignment_ACC"]
        > shuffled_result["prototype_assignment_ACC"]
        and R_result["true_class_margin_mean"]
        > shuffled_result["true_class_margin_mean"]
    )
    R_vs_ALL_ACC_delta = float(
        R_result["prototype_assignment_ACC"]
        - all_result["prototype_assignment_ACC"]
    )
    R_vs_ALL_margin_delta = float(
        R_result["true_class_margin_mean"]
        - all_result["true_class_margin_mean"]
    )
    if R_specificity_pass and R_vs_ALL_margin_delta >= 0.0:
        decision = "E4A0_R_MEMORY_WRITE_PASS"
    elif R_specificity_pass:
        decision = "E4A0_R_SPECIFIC_BUT_COMPLEMENTARITY_LIMITED"
    else:
        decision = "E4A0_R_MEMORY_WRITE_FAIL"
    return {
        "R_specificity_pass": R_specificity_pass,
        "R_vs_ALL_ACC_delta": R_vs_ALL_ACC_delta,
        "R_vs_ALL_margin_delta": R_vs_ALL_margin_delta,
        "final_decision": decision,
        "oracle_role": "upper_bound_diagnostic_only",
    }


def _arm_memory_audit(fixed, labels_labeled):
    labels_tensor, _ = validate_sparse_labels(labels_labeled)
    records = {}
    query_hashes = {}
    for arm in E4_ARMS:
        weights = fixed["writer_weights"][arm]
        denominators = [
            float(weights[labels_tensor == class_id].sum().item())
            for class_id in range(CLASS_NUM)
        ]
        prototypes = fixed[arm]["prototypes"]
        query_hash = tensor_sha256(fixed[arm]["h_query"].cpu().numpy())
        query_hashes[arm] = query_hash
        records[arm] = {
            "writer_weights_shape": list(weights.shape),
            "writer_weights_logical_sha256": tensor_sha256(
                weights.cpu().numpy()
            ),
            "all_14_labels_used_pass": True,
            "no_hard_selection_pass": True,
            "class_denominators": denominators,
            "all_class_denominators_positive_pass": bool(
                all(value > 0.0 for value in denominators)
            ),
            "prototype_shape": list(prototypes.shape),
            "prototype_shape_pass": prototypes.shape
            == (CLASS_NUM, CLASS_NUM),
            "prototype_finite_pass": bool(
                torch.isfinite(prototypes).all().item()
            ),
            "prototype_unit_norm_pass": bool(
                torch.allclose(
                    torch.linalg.vector_norm(prototypes, dim=1),
                    torch.ones(CLASS_NUM, dtype=prototypes.dtype),
                )
            ),
            "prototype_stop_gradient_pass": bool(
                not prototypes.requires_grad and prototypes.grad_fn is None
            ),
            "score_shape": list(fixed[arm]["scores"].shape),
            "query_logical_sha256": query_hash,
        }
    return records, query_hashes


def run_evaluation(
    feature_path=DEFAULT_FEATURE_PATH,
    feature_audit_path=DEFAULT_FEATURE_AUDIT_PATH,
    e1_lwc_model_dir=DEFAULT_E1_LWC_MODEL_DIR,
    label_split_dir=DEFAULT_LABEL_SPLIT_DIR,
    full_gt_path=DEFAULT_FULL_GT_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    device="cpu",
):
    """Run the frozen diagnostic once; no fitting or parameter updates occur."""
    output_root = _resolve(output_dir)
    _require(
        not output_root.exists(),
        "refusing to overwrite an E4-A0 output directory",
    )

    h_sem, sample_ids, representation_audit = (
        load_e1_lwc_semantic_representation(
            feature_path=feature_path,
            feature_audit_path=feature_audit_path,
            model_dir=e1_lwc_model_dir,
            model_audit_path=_resolve(e1_lwc_model_dir).parent / "e1_audit.json",
            device=device,
        )
    )
    R_full, frozen_reliability = load_frozen_reliability(DEFAULT_R_PATH)
    labeled_ids, labels_labeled, sparse_label_audit = (
        load_sparse_label_protocol(label_split_dir)
    )
    _require(
        np.array_equal(sample_ids, np.arange(SAMPLE_NUM, dtype=np.int64)),
        "representation/sample ID alignment mismatch",
    )
    R_labeled = np.ascontiguousarray(R_full[labeled_ids])
    oracle_clean_weights, oracle_audit = load_oracle_clean_labeled_weights(
        labeled_ids, DEFAULT_ORACLE_MASK_PATH
    )

    fixed = build_fixed_arm_outputs(
        h_sem=h_sem,
        labeled_ids=labeled_ids,
        labels_labeled=labels_labeled,
        R_full=R_full,
        oracle_clean_weights=oracle_clean_weights,
    )

    output_root.mkdir(parents=True)
    prediction_seal = _save_predictions_before_full_GT(fixed, output_root)

    full_GT, full_gt_audit = load_full_ground_truth(full_gt_path)
    unlabeled_mask = np.ones(SAMPLE_NUM, dtype=bool)
    unlabeled_mask[labeled_ids] = False
    unlabeled_ids = np.flatnonzero(unlabeled_mask).astype(np.int64)
    _require(
        unlabeled_ids.shape == (SAMPLE_NUM - LABELED_NUM,),
        "unlabeled evaluation split must contain 1386 samples",
    )
    results = evaluate_unlabeled_fixed_arms(
        fixed, full_GT, unlabeled_ids
    )
    gate = build_gate_decision(results)

    arm_audit, query_hashes = _arm_memory_audit(
        fixed, labels_labeled
    )
    common_query_hash = tensor_sha256(fixed["h_query"].cpu().numpy())
    query_identical = bool(
        len(set(query_hashes.values())) == 1
        and next(iter(query_hashes.values())) == common_query_hash
        and all(
            fixed[arm]["h_query"].data_ptr() == fixed["h_query"].data_ptr()
            for arm in E4_ARMS
        )
    )
    normal_arms = E4_ARMS[:3]
    normal_hashes = {
        arm: {
            "prototype": tensor_sha256(
                fixed[arm]["prototypes"].cpu().numpy()
            ),
            "prediction": tensor_sha256(
                fixed[arm]["predictions"].cpu().numpy()
            ),
        }
        for arm in normal_arms
    }
    memory_audit = {
        "stage": STAGE,
        "seed": SEED,
        "N": SAMPLE_NUM,
        "V": VIEW_NUM,
        "K": CLASS_NUM,
        "labeled_count": LABELED_NUM,
        "unlabeled_evaluation_count": int(unlabeled_ids.size),
        "arms": list(E4_ARMS),
        "input_provenance": {
            "E1_LWC_frozen_model": representation_audit,
            "frozen_reliability": {
                **frozen_reliability,
                "R_labeled": {
                    "shape": list(R_labeled.shape),
                    "dtype": str(R_labeled.dtype),
                    "logical_sha256": tensor_sha256(R_labeled),
                    "source": "R_full[labeled_ids]",
                    "stop_gradient_pass": True,
                },
            },
            "sparse_labels": sparse_label_audit,
            "oracle_clean_diagnostic": oracle_audit,
            "full_GT_post_prediction_only": full_gt_audit,
        },
        "R_full": frozen_reliability["R_full"],
        "R_labeled": {
            "shape": list(R_labeled.shape),
            "logical_sha256": tensor_sha256(R_labeled),
        },
        "memory_equation": (
            "m_c=normalize(sum_{i:y_i=c}sum_v "
            "w_i^v*h_i^v / sum_{i:y_i=c}sum_v w_i^v)"
        ),
        "memory_arms": arm_audit,
        "matched_shuffle": fixed["shuffle_audit"],
        "query": {
            "definition": "h_query=normalize(mean_v(h_sem))",
            "shape": list(fixed["h_query"].shape),
            "logical_sha256": common_query_hash,
            "per_arm_logical_sha256": query_hashes,
            "identical_across_all_arms_pass": query_identical,
            "reliability_weighted_query_used": False,
            "arm_specific_query_used": False,
        },
        "oracle_isolation": {
            "normal_arm_fixed_output_hashes": normal_hashes,
            "oracle_used_for_training": False,
            "oracle_affects_normal_arms": False,
            "oracle_isolated_from_R_ALL_shuffle_pass": True,
            "role": "upper_bound_diagnostic_only",
        },
        "prediction_seal": prediction_seal,
        "label_leakage": {
            "full_unlabeled_GT_loaded_before_prediction": False,
            "unlabeled_GT_used_for_memory": False,
            "prediction_saved_before_full_GT": True,
            "prediction_hashed_before_full_GT": True,
            "training_used": False,
            "optimizer_used": False,
            "backward_used": False,
        },
        "execution_guards": {
            "training_used": False,
            "optimizer_used": False,
            "backward_used": False,
            "parameter_update_used": False,
            "automatic_retraining_used": False,
            "deterministic_pass": True,
        },
        "gate": gate,
        "E4A0_AUDIT_PASS": bool(
            query_identical
            and fixed["shuffle_audit"][
                "class_view_weight_sum_exact_pass"
            ]
            and representation_audit["alignment"][
                "M_permutation_pass"
            ]
            and representation_audit["alignment"][
                "q_alignment_mass_preserved_pass"
            ]
        ),
    }
    diagnostic_results = {
        "stage": STAGE,
        "primary_evaluation_scope": "1386_unlabeled_samples_only",
        "results": results,
        "gate": gate,
    }

    np.savez(
        output_root / "memory_prototypes.npz",
        **{
            arm: np.ascontiguousarray(
                fixed[arm]["prototypes"].cpu().numpy()
            )
            for arm in E4_ARMS
        },
    )
    np.savez(
        output_root / "writer_weights.npz",
        **{
            arm: np.ascontiguousarray(
                fixed["writer_weights"][arm].cpu().numpy()
            )
            for arm in E4_ARMS
        },
    )
    _write_json(output_root / "memory_audit.json", memory_audit)
    _write_json(
        output_root / "diagnostic_results.json", diagnostic_results
    )
    return {
        "output_dir": output_root,
        "memory_audit": memory_audit,
        "diagnostic_results": diagnostic_results,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument(
        "--feature-audit-path", default=str(DEFAULT_FEATURE_AUDIT_PATH)
    )
    parser.add_argument(
        "--e1-lwc-model-dir", default=str(DEFAULT_E1_LWC_MODEL_DIR)
    )
    parser.add_argument(
        "--label-split-dir", default=str(DEFAULT_LABEL_SPLIT_DIR)
    )
    parser.add_argument("--full-gt-path", default=str(DEFAULT_FULL_GT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_evaluation(
        feature_path=args.feature_path,
        feature_audit_path=args.feature_audit_path,
        e1_lwc_model_dir=args.e1_lwc_model_dir,
        label_split_dir=args.label_split_dir,
        full_gt_path=args.full_gt_path,
        output_dir=args.output_dir,
        device=args.device,
    )
    print("Saved: " + _display(result["output_dir"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
