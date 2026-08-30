"""Frozen sparse semantic-anchor protocol for B7-A1.

This module contains only deterministic array/tensor construction and read-only
artifact loading.  It does not define a trainable head, loss, optimizer, or
parameter update.
"""

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "Caltech-6V"
SAMPLE_NUM = 1400
VIEW_NUM = 6
CLASS_NUM = 7
LABELED_NUM = 14
UNLABELED_NUM = 1386
TOP_K = 3
CONTROL_SEED = 20
EXPECTED_U_SHA256 = (
    "3458e099f8c6e7317e3edbcef15ff629f368735e115fa6aea30e4516be8ac2cc"
)
EXPECTED_UTILITY_FILE_SHA256 = (
    "569ad65f9f74c8cffcc9e6b96485d2dd941bdb17f0e7468a449ec09afdf51313"
)
EXPECTED_LABEL_SPLIT_SHA256 = (
    "0463cf7155bc2b90a6133f0a79fd30d8c5a9e78079c6692dd1fa4106cb174487"
)
DEFAULT_D2_DIR = (
    REPOSITORY_ROOT / "outputs/d2_caltech6v/a0_utility_transfer_seed20"
)
DEFAULT_LABEL_SPLIT_DIR = (
    REPOSITORY_ROOT / "outputs/b7_sparse_supervision/b7a0_seed20"
)
ANCHOR_POLICIES = (
    "ALL",
    "U_TOP3",
    "U_CLASS_SHUFFLED",
    "SHUFFLED_U",
    "ORACLE",
    "ORACLE_CLASS_SHUFFLED",
    "SHUFFLED_LABEL",
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _torch_boundary(value, dtype=None):
    """Convert at the NumPy->Torch boundary, copying only read-only arrays."""
    boundary_value = value
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        boundary_value = np.array(value, copy=True, order="C")
    return torch.as_tensor(boundary_value, dtype=dtype).detach()


def file_sha256(path):
    """Hash the complete on-disk byte stream of one artifact."""
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_u_only(d2_dir=DEFAULT_D2_DIR):
    """Load frozen D2 T/U without loading or returning the oracle mask."""
    from irv.b4_information_utility import tensor_sha256

    root = _resolve(d2_dir)
    utility_path = root / "utility_scores.npz"
    transfer_path = root / "utility_transfer.json"
    _require(utility_path.is_file(), "frozen D2 utility_scores.npz is missing")
    _require(transfer_path.is_file(), "frozen D2 utility_transfer.json is missing")
    transfer = _read_json(transfer_path)
    _require(
        transfer.get("D2_A0_UTILITY_TRANSFER_PASS") is True,
        "D2-A0 frozen transfer gate is not closed",
    )
    with np.load(utility_path, allow_pickle=False) as archive:
        _require(
            {"T", "U"}.issubset(set(archive.files)),
            "D2 archive does not contain frozen T/U",
        )
        # T/U: [N,V].  The corruption_mask member is deliberately not accessed.
        predictability = np.ascontiguousarray(archive["T"], dtype=np.float64)
        utility = np.ascontiguousarray(archive["U"], dtype=np.float64)
    _require(
        predictability.shape == (SAMPLE_NUM, VIEW_NUM)
        and utility.shape == (SAMPLE_NUM, VIEW_NUM),
        "frozen D2 T/U shape mismatch",
    )
    _require(
        np.isfinite(predictability).all()
        and np.isfinite(utility).all()
        and float(utility.min()) >= 0.0
        and float(utility.max()) <= 1.0,
        "frozen D2 T/U value boundary mismatch",
    )
    u_hash = tensor_sha256(utility)
    archive_hash = file_sha256(utility_path)
    _require(
        u_hash == transfer.get("U_sha256") == EXPECTED_U_SHA256,
        "frozen D2 logical U SHA256 mismatch",
    )
    _require(
        archive_hash == EXPECTED_UTILITY_FILE_SHA256,
        "frozen D2 NPZ file-byte SHA256 mismatch",
    )
    predictability.setflags(write=False)
    utility.setflags(write=False)
    return {
        "T": predictability,
        "U": utility,
        "T_sha256": str(transfer["T_sha256"]),
        "U_sha256": u_hash,
        "utility_file_sha256": archive_hash,
        "utility_path": str(utility_path.resolve()),
        "transfer_path": str(transfer_path.resolve()),
    }


def load_oracle_corruption_mask(d2_dir=DEFAULT_D2_DIR):
    """Load the frozen corruption mask through the isolated oracle boundary."""
    from weak_quality import ndarray_sha256

    root = _resolve(d2_dir)
    mask_path = root / "corruption_audit/corruption_mask.npy"
    audit_path = root / "corruption_audit/corruption_audit.json"
    _require(mask_path.is_file(), "oracle corruption mask is missing")
    _require(audit_path.is_file(), "oracle corruption audit is missing")
    # corruption_mask: [N,V]
    corruption_mask = np.load(mask_path, allow_pickle=False)
    audit = _read_json(audit_path)
    _require(
        corruption_mask.dtype == np.dtype(bool)
        and corruption_mask.shape == (SAMPLE_NUM, VIEW_NUM),
        "oracle corruption mask boundary mismatch",
    )
    _require(
        np.all(corruption_mask.sum(axis=1) == TOP_K),
        "oracle condition is not exactly-k3",
    )
    mask_hash = ndarray_sha256(corruption_mask)
    _require(mask_hash == audit.get("mask_sha256"), "oracle mask SHA256 mismatch")
    corruption_mask.setflags(write=False)
    return {
        "corruption_mask": corruption_mask,
        "corruption_mask_sha256": mask_hash,
        "corruption_mask_path": str(mask_path.resolve()),
        "corruption_audit_path": str(audit_path.resolve()),
    }


def load_fixed_label_split(split_dir=DEFAULT_LABEL_SPLIT_DIR):
    """Read and verify the formal B7-A0 split without regenerating it."""
    from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
        json_sha256,
    )
    from weak_quality import ndarray_sha256

    root = _resolve(split_dir)
    record_path = root / "label_split.json"
    labeled_path = root / "labeled_sample_ids.npy"
    unlabeled_path = root / "unlabeled_sample_ids.npy"
    _require(
        record_path.is_file() and labeled_path.is_file() and unlabeled_path.is_file(),
        "formal B7-A0 label split artifacts are incomplete",
    )
    record = _read_json(record_path)
    stored_split_hash = str(record.get("label_split_sha256", ""))
    hash_payload = dict(record)
    hash_payload.pop("label_split_sha256", None)
    _require(
        json_sha256(hash_payload)
        == stored_split_hash
        == EXPECTED_LABEL_SPLIT_SHA256,
        "formal B7-A0 label_split_sha256 mismatch",
    )
    # labeled_ids/unlabeled_ids: [L] / [N-L]
    labeled_ids = np.load(labeled_path, allow_pickle=False)
    unlabeled_ids = np.load(unlabeled_path, allow_pickle=False)
    all_ids = np.arange(SAMPLE_NUM, dtype=np.int64)
    _require(
        labeled_ids.dtype == np.dtype(np.int64)
        and unlabeled_ids.dtype == np.dtype(np.int64)
        and labeled_ids.shape == (LABELED_NUM,)
        and unlabeled_ids.shape == (UNLABELED_NUM,),
        "formal B7-A0 split array boundary mismatch",
    )
    _require(
        np.intersect1d(labeled_ids, unlabeled_ids).size == 0
        and np.array_equal(
            np.sort(np.concatenate((labeled_ids, unlabeled_ids))), all_ids
        ),
        "formal B7-A0 labeled/unlabeled IDs are not an exact partition",
    )
    _require(
        ndarray_sha256(labeled_ids) == record.get("labeled_ids_sha256"),
        "formal B7-A0 labeled ID SHA256 mismatch",
    )
    labeled_ids.setflags(write=False)
    unlabeled_ids.setflags(write=False)
    return {
        "labeled_sample_ids": labeled_ids,
        "unlabeled_sample_ids": unlabeled_ids,
        "label_split_sha256": stored_split_hash,
        "labeled_ids_sha256": str(record["labeled_ids_sha256"]),
        "record": record,
        "label_split_path": str(record_path.resolve()),
        "labeled_ids_path": str(labeled_path.resolve()),
        "unlabeled_ids_path": str(unlabeled_path.resolve()),
    }


def frozen_top3_mask(utility):
    """Return D2's stable frozen Top-3 admission mask."""
    from experiments.d2_caltech6v.evaluate_d2_a0_utility_transfer import (
        topk_admission,
    )

    values = np.asarray(utility)
    _require(values.ndim == 2 and values.shape[1] == VIEW_NUM, "U must be [N,6]")
    _, mask = topk_admission(values, admitted_view_num=TOP_K)
    _require(np.all(mask.sum(axis=1) == TOP_K), "frozen U Top-3 row mismatch")
    return np.ascontiguousarray(mask, dtype=bool)


def deterministic_permutation(length, control_seed=CONTROL_SEED):
    """Create one local deterministic permutation without touching global RNG."""
    length = int(length)
    _require(length > 1, "permutation length must exceed one")
    permutation = np.random.RandomState(int(control_seed)).permutation(length)
    return np.ascontiguousarray(permutation, dtype=np.int64)


def build_normal_anchor_policies(labeled_utility, control_seed=CONTROL_SEED):
    """Build normal anchor masks; this API has no oracle-mask argument."""
    values = np.asarray(labeled_utility, dtype=np.float64)
    _require(
        values.shape == (LABELED_NUM, VIEW_NUM) and np.isfinite(values).all(),
        "labeled Utility must have shape [14,6]",
    )
    # u_top3: [L,V]
    u_top3 = frozen_top3_mask(values)
    # row_permutation: [L].  Only sample/mask correspondence is broken.
    row_permutation = deterministic_permutation(values.shape[0], control_seed)
    shuffled_u = np.ascontiguousarray(u_top3[row_permutation], dtype=bool)
    _require(
        np.all(u_top3.sum(axis=1) == TOP_K)
        and np.all(shuffled_u.sum(axis=1) == TOP_K)
        and np.array_equal(u_top3.sum(axis=0), shuffled_u.sum(axis=0)),
        "SHUFFLED_U does not preserve Top-3 row/column admission counts",
    )
    all_mask = np.ones_like(u_top3, dtype=bool)
    return {
        "anchor_masks": {
            "ALL": all_mask,
            "U_TOP3": u_top3,
            "SHUFFLED_U": shuffled_u,
            "SHUFFLED_LABEL": u_top3.copy(),
        },
        "shuffled_u_row_permutation": row_permutation,
    }


def build_oracle_anchor_mask(corruption_mask, labeled_sample_ids):
    """Build exactly-k3 clean-view anchors in the isolated oracle path."""
    values = np.asarray(corruption_mask, dtype=bool)
    labeled_ids = np.asarray(labeled_sample_ids, dtype=np.int64)
    _require(values.shape == (SAMPLE_NUM, VIEW_NUM), "corruption mask must be [N,6]")
    _require(labeled_ids.shape == (LABELED_NUM,), "labeled IDs must be [14]")
    # oracle_anchor_mask: [L,V]
    oracle_anchor_mask = np.ascontiguousarray(~values[labeled_ids], dtype=bool)
    _require(
        oracle_anchor_mask.shape == (LABELED_NUM, VIEW_NUM)
        and np.all(oracle_anchor_mask.sum(axis=1) == TOP_K),
        "ORACLE labeled rows must contain exactly three clean views",
    )
    return oracle_anchor_mask


def build_shuffled_labeled_targets(labeled_targets, control_seed=CONTROL_SEED):
    """Permute only the fourteen sparse targets and preserve their histogram."""
    targets = np.asarray(labeled_targets, dtype=np.int64)
    _require(targets.shape == (LABELED_NUM,), "labeled targets must be [14]")
    permutation = deterministic_permutation(targets.size, control_seed)
    shuffled = np.ascontiguousarray(targets[permutation], dtype=np.int64)
    _require(
        np.array_equal(
            np.bincount(targets, minlength=CLASS_NUM),
            np.bincount(shuffled, minlength=CLASS_NUM),
        ),
        "SHUFFLED_LABEL changed the sparse class histogram",
    )
    return shuffled, permutation


def build_classwise_swapped_anchor_mask(
    anchor_mask,
    labeled_targets,
    num_classes,
):
    """Swap the two admission rows within every sparse-label class."""
    base = np.asarray(anchor_mask, dtype=bool)
    targets = np.asarray(labeled_targets, dtype=np.int64)
    num_classes = int(num_classes)
    _require(
        base.shape == (LABELED_NUM, VIEW_NUM)
        and targets.shape == (LABELED_NUM,)
        and num_classes == CLASS_NUM,
        "classwise anchor-swap input boundary mismatch",
    )
    _require(
        np.all(base.sum(axis=1) == TOP_K),
        "classwise anchor swap requires exactly three views per row",
    )
    swapped = np.array(base, copy=True, order="C")
    classwise_swap_audit = []
    identical_mask_pair_count = 0
    for class_id in range(num_classes):
        class_rows = np.flatnonzero(targets == class_id).astype(np.int64)
        _require(
            class_rows.shape == (2,),
            "classwise anchor swap requires exactly two labels per class",
        )
        first_row, second_row = (int(class_rows[0]), int(class_rows[1]))
        pair_identical = bool(np.array_equal(base[first_row], base[second_row]))
        identical_mask_pair_count += int(pair_identical)
        swapped[first_row] = base[second_row]
        swapped[second_row] = base[first_row]
        classwise_swap_audit.append({
            "class_id": class_id,
            "labeled_row_ids": [first_row, second_row],
            "input_pair_identical": pair_identical,
            "first_row_changed": bool(
                not np.array_equal(swapped[first_row], base[first_row])
            ),
            "second_row_changed": bool(
                not np.array_equal(swapped[second_row], base[second_row])
            ),
        })
    changed_row_count = int(np.any(swapped != base, axis=1).sum())
    _require(
        changed_row_count > 0 and not np.array_equal(swapped, base),
        "classwise swapped control is uninformative: admission matrix unchanged",
    )
    _require(
        np.all(swapped.sum(axis=1) == TOP_K),
        "classwise swapped control changed a row admission count",
    )
    for class_id in range(num_classes):
        class_rows = targets == class_id
        _require(
            np.array_equal(
                base[class_rows].sum(axis=0), swapped[class_rows].sum(axis=0)
            ),
            "classwise swapped control changed class-view prototype counts",
        )
    return np.ascontiguousarray(swapped, dtype=bool), {
        "changed_row_count": changed_row_count,
        "identical_mask_pair_count": int(identical_mask_pair_count),
        "classwise_swap_audit": classwise_swap_audit,
        "targets_unchanged": True,
    }


def extract_native_z_q(autoencoders, evaluation_views):
    """Extract normalized z and checkpoint-native q from raw encoder latents."""
    _require(
        len(autoencoders) == VIEW_NUM and len(evaluation_views) == VIEW_NUM,
        "native extraction requires six aligned views",
    )
    normalized_views = []
    raw_views = []
    belief_views = []
    with torch.no_grad():
        for view_id, autoencoder in enumerate(autoencoders):
            autoencoder.eval()
            features = torch.from_numpy(evaluation_views[view_id]).float()
            # raw_z: [N,Dz]
            raw_z = autoencoder.encoder(features).detach()
            # q: [N,K].  clustering receives raw_z, never normalized z.
            q = autoencoder.clustering(raw_z).detach()
            # z_norm: [N,Dz]
            z_norm = F.normalize(raw_z, p=2, dim=1, eps=1e-12).detach()
            raw_views.append(raw_z)
            belief_views.append(q)
            normalized_views.append(z_norm)
    # raw_z_stack/z_stack/q_stack: [N,V,Dz] / [N,V,Dz] / [N,V,K]
    raw_z_stack = torch.stack(raw_views, dim=1).detach()
    z_stack = torch.stack(normalized_views, dim=1).detach()
    q_stack = torch.stack(belief_views, dim=1).detach()
    sample_num = int(raw_z_stack.shape[0])
    _require(
        tuple(raw_z_stack.shape) == (sample_num, VIEW_NUM, 10)
        and tuple(z_stack.shape) == (sample_num, VIEW_NUM, 10)
        and tuple(q_stack.shape) == (sample_num, VIEW_NUM, CLASS_NUM),
        "native z/q shape mismatch",
    )
    _require(
        not raw_z_stack.requires_grad
        and not z_stack.requires_grad
        and not q_stack.requires_grad
        and torch.isfinite(raw_z_stack).all().item()
        and torch.isfinite(z_stack).all().item()
        and torch.isfinite(q_stack).all().item(),
        "native z/q frozen-finite boundary mismatch",
    )
    return {
        "raw_z_stack": raw_z_stack,
        "z_stack": z_stack,
        "q_stack": q_stack,
        "q_input_source": "raw_encoder_latent",
    }


def build_within_view_prototypes(
    labeled_repr,
    labeled_targets,
    anchor_mask,
    num_classes,
):
    """Build class/view prototypes using only explicitly sparse label inputs."""
    values = _torch_boundary(labeled_repr)
    targets = _torch_boundary(labeled_targets, dtype=torch.long)
    admitted = _torch_boundary(anchor_mask, dtype=torch.bool)
    num_classes = int(num_classes)
    _require(values.ndim == 3, "labeled_repr must have shape [L,V,D]")
    labeled_num, view_num, representation_dim = values.shape
    _require(
        labeled_num == LABELED_NUM
        and view_num == VIEW_NUM
        and targets.shape == (labeled_num,)
        and admitted.shape == (labeled_num, view_num),
        "sparse prototype input shape mismatch",
    )
    _require(
        num_classes == CLASS_NUM
        and torch.all((targets >= 0) & (targets < num_classes)).item(),
        "sparse prototype target boundary mismatch",
    )
    # prototypes: [K,V,D]; missing entries remain storage zeros but are invalid.
    prototypes = torch.zeros(
        num_classes, view_num, representation_dim, dtype=values.dtype
    )
    # prototype_count/prototype_valid_mask: [K,V]
    prototype_count = torch.zeros(num_classes, view_num, dtype=torch.long)
    prototype_valid_mask = torch.zeros(num_classes, view_num, dtype=torch.bool)
    for class_id in range(num_classes):
        class_members = targets == class_id
        for view_id in range(view_num):
            selected = torch.logical_and(class_members, admitted[:, view_id])
            count = int(selected.sum().item())
            prototype_count[class_id, view_id] = count
            if count > 0:
                mean_value = values[selected, view_id, :].mean(dim=0)
                norm = torch.linalg.vector_norm(mean_value, ord=2)
                _require(
                    torch.isfinite(norm).item() and float(norm.item()) > 0.0,
                    "valid sparse prototype has zero or non-finite norm",
                )
                prototypes[class_id, view_id, :] = F.normalize(
                    mean_value, p=2, dim=0, eps=1e-12
                )
                prototype_valid_mask[class_id, view_id] = True
    _require(
        torch.equal(prototype_valid_mask, prototype_count > 0),
        "prototype validity/count mismatch",
    )
    # class_coverage/view_coverage: [K] / [V]
    class_coverage = prototype_valid_mask.sum(dim=1, dtype=torch.long)
    view_coverage = prototype_valid_mask.sum(dim=0, dtype=torch.long)
    return {
        "prototypes": prototypes.detach(),
        "prototype_count": prototype_count.detach(),
        "prototype_valid_mask": prototype_valid_mask.detach(),
        "class_coverage": class_coverage.detach(),
        "view_coverage": view_coverage.detach(),
        "class_view_coverage_rate": float(prototype_valid_mask.float().mean().item()),
    }


def score_unlabeled_queries(
    unlabeled_repr,
    prototypes,
    prototype_valid_mask,
    query_mask,
):
    """Aggregate within-view cosine scores over common frozen-U query views."""
    queries = _torch_boundary(unlabeled_repr)
    prototype_values = _torch_boundary(prototypes)
    prototype_valid = _torch_boundary(prototype_valid_mask, dtype=torch.bool)
    admitted = _torch_boundary(query_mask, dtype=torch.bool)
    _require(queries.ndim == 3, "unlabeled_repr must have shape [Nq,V,D]")
    query_num, view_num, representation_dim = queries.shape
    _require(
        tuple(prototype_values.shape)
        == (CLASS_NUM, view_num, representation_dim)
        and prototype_valid.shape == (CLASS_NUM, view_num)
        and admitted.shape == (query_num, view_num),
        "semantic scoring input shape mismatch",
    )
    _require(
        torch.all(admitted.sum(dim=1) == TOP_K).item(),
        "query mask must admit exactly three views per query",
    )
    with torch.no_grad():
        # Normalization here is metric-only; q remains native_q_raw provenance.
        normalized_queries = F.normalize(queries, p=2, dim=-1, eps=1e-12)
        normalized_prototypes = F.normalize(
            prototype_values, p=2, dim=-1, eps=1e-12
        )
        # per_view_scores: [Nq,K,V]
        per_view_scores = torch.einsum(
            "nvd,kvd->nkv", normalized_queries, normalized_prototypes
        )
        # effective_mask: [Nq,K,V]
        effective_mask = torch.logical_and(
            admitted[:, None, :], prototype_valid[None, :, :]
        )
        denominator = effective_mask.sum(dim=2, dtype=torch.long)
        # query_class_score_available: [Nq,K]
        query_class_score_available = denominator > 0
        weighted_sum = torch.sum(
            per_view_scores * effective_mask.to(per_view_scores.dtype), dim=2
        )
        # aggregated_scores: [Nq,K]; unavailable entries are NaN, never zero.
        aggregated_scores = torch.full(
            (query_num, CLASS_NUM),
            float("nan"),
            dtype=per_view_scores.dtype,
        )
        aggregated_scores[query_class_score_available] = (
            weighted_sum[query_class_score_available]
            / denominator[query_class_score_available].to(weighted_sum.dtype)
        )
        any_available = query_class_score_available.any(dim=1)
        _require(
            torch.all(any_available).item(),
            "one or more queries have no available class score",
        )
        prediction_scores = aggregated_scores.masked_fill(
            ~query_class_score_available, -torch.inf
        )
        # predictions: [Nq]
        predictions = torch.argmax(prediction_scores, dim=1)
    return {
        "per_view_scores": per_view_scores.detach(),
        "effective_mask": effective_mask.detach(),
        "effective_view_count": denominator.detach(),
        "aggregated_scores": aggregated_scores.detach(),
        "query_class_score_available": query_class_score_available.detach(),
        "predictions": predictions.detach(),
    }


def evaluate_fixed_predictions(
    evaluation_targets,
    predictions,
    aggregated_scores,
    query_class_score_available,
):
    """Evaluate already-fixed predictions; labels cannot affect prior scoring."""
    targets = _torch_boundary(evaluation_targets, dtype=torch.long)
    predicted = _torch_boundary(predictions, dtype=torch.long)
    scores = _torch_boundary(aggregated_scores)
    available = _torch_boundary(query_class_score_available, dtype=torch.bool)
    query_num = int(targets.numel())
    _require(
        targets.shape == predicted.shape == (query_num,)
        and scores.shape == available.shape == (query_num, CLASS_NUM),
        "fixed prediction evaluation shape mismatch",
    )
    _require(
        torch.all((targets >= 0) & (targets < CLASS_NUM)).item(),
        "evaluation targets are outside the class boundary",
    )
    rows = torch.arange(query_num, dtype=torch.long)
    true_available = available[rows, targets]
    wrong_available = available.clone()
    wrong_available[rows, targets] = False
    has_wrong = wrong_available.any(dim=1)
    margin_valid = torch.logical_and(true_available, has_wrong)
    _require(bool(margin_valid.any().item()), "no query has a valid semantic margin")
    true_scores = scores[rows, targets]
    wrong_max = scores.masked_fill(~wrong_available, -torch.inf).max(dim=1).values
    # margins: only rows with true and at least one wrong score available.
    margins = true_scores[margin_valid] - wrong_max[margin_valid]
    wrong_sum = torch.nansum(scores.masked_fill(~wrong_available, float("nan")), dim=1)
    wrong_count = wrong_available.sum(dim=1)
    wrong_mean = wrong_sum[margin_valid] / wrong_count[margin_valid].to(scores.dtype)
    semantic_gap = true_scores[margin_valid].mean() - wrong_mean.mean()
    samples_available = available.any(dim=1)
    return {
        "nearest_prototype_ACC": float((predicted == targets).float().mean().item()),
        "margin_mean": float(margins.mean().item()),
        "margin_median": float(torch.median(margins).item()),
        "positive_margin_rate": float((margins > 0.0).float().mean().item()),
        "margin_valid_count": int(margin_valid.sum().item()),
        "semantic_gap": float(semantic_gap.item()),
        "true_class_score_available_rate": float(true_available.float().mean().item()),
        "full_class_score_available_rate": float(
            available.all(dim=1).float().mean().item()
        ),
        "samples_with_at_least_one_available_class": int(samples_available.sum().item()),
    }


def prototype_builder_api_pass():
    """Return whether the sparse prototype API has the exact allowed boundary."""
    return tuple(inspect.signature(build_within_view_prototypes).parameters) == (
        "labeled_repr",
        "labeled_targets",
        "anchor_mask",
        "num_classes",
    )
