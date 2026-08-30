"""B7-A0 sparse true-supervision admission experiment.

The Native MVCAN backbone and D2 Utility are frozen inputs.  All five arms use
the same B6-style admitted-only InfoNCE objective and semantic readout.  The
only arm-specific training input is the binary supervised admission policy on
the same fourteen labeled samples.
"""

import argparse
import copy
import hashlib
import inspect
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ClusteringTest import acc as mvcan_acc
from ClusteringTest import ari as mvcan_ari
from ClusteringTest import nmi as mvcan_nmi
from configure import get_default_config
from datasets import load_data
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import ARMS
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
    CLUSTER_NUM,
)
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
    DATASET_NAME,
)
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
    LABELS_PER_CLASS,
)
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
    LABEL_SPLIT_SEED,
)
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
    SAMPLE_NUM,
)
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
    SHUFFLE_SEED,
)
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import TOP_K
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import VIEW_NUM
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
    build_normal_supervised_admission,
)
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
    build_oracle_supervised_admission,
)
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
    deterministic_within_sample_view_shuffle,
)
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
    frozen_u_topk_admission,
)
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
    make_class_balanced_split,
)
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
    save_label_split,
)
from experiments.b7_sparse_supervision.b7_sparse_label_protocol import (
    supervised_channel_count,
)
import experiments.d2_caltech6v.evaluate_d2_a0_utility_transfer as d2
from irv.b3_audit import hash_backbone
from irv.b3_audit import hash_state_dict
from irv.b4_information_utility import tensor_sha256
from weak_quality import apply_weak_quality_protocol
from weak_quality import ndarray_sha256


STAGE = "B7-A0"
MODEL_SEED = 20
CORRUPTION_SEED = 20
CORRUPTION_K = 3
SNR_DB = 2.5
LATENT_DIM = 10
HIDDEN_DIM = 32
SEMANTIC_DIM = 10
TEMPERATURE = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LAMBDA_SUP = 1.0
SMOKE_EPOCHS = 3
PILOT_EPOCHS = 100
KMEANS_N_INIT = 100
EXPECTED_VIEW_DIMS = [48, 40, 254, 1984, 512, 928]
EXPECTED_D2_U_SHA256 = (
    "3458e099f8c6e7317e3edbcef15ff629f368735e115fa6aea30e4516be8ac2cc"
)
EXPECTED_D2_UTILITY_FILE_SHA256 = (
    "569ad65f9f74c8cffcc9e6b96485d2dd941bdb17f0e7468a449ec09afdf51313"
)
EXPECTED_CORRUPTION_MASK_SHA256 = (
    "e6250535db6e33b9263102cf335fcf9e8552a24cae4b70ec5a94887f4e58bf0b"
)
DEFAULT_BACKBONE_DIR = (
    REPOSITORY_ROOT / "outputs/d1_caltech6v/snr2p5_k3_seed20/models"
)
DEFAULT_D2_DIR = (
    REPOSITORY_ROOT / "outputs/d2_caltech6v/a0_utility_transfer_seed20"
)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/b7_sparse_supervision"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def _display(path):
    value = Path(path).resolve()
    try:
        return str(value.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(value)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_explicit_rng_seeds(seed=MODEL_SEED):
    """Freeze initialization and deterministic CPU execution controls."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class B7SemanticCarrier(nn.Module):
    """Six-view extension of the frozen B6/WQ1A1 minimal carrier."""

    def __init__(
        self,
        view_num=VIEW_NUM,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        semantic_dim=SEMANTIC_DIM,
        cluster_num=CLUSTER_NUM,
    ):
        super().__init__()
        self.view_num = int(view_num)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.semantic_dim = int(semantic_dim)
        self.cluster_num = int(cluster_num)
        _require(self.view_num >= 2, "view_num must be at least two")
        _require(
            min(
                self.latent_dim,
                self.hidden_dim,
                self.semantic_dim,
                self.cluster_num,
            )
            > 0,
            "carrier dimensions must be positive",
        )
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.semantic_dim),
            )
            for _ in range(self.view_num)
        ])
        self.classifier = nn.Linear(self.semantic_dim, self.cluster_num)

    def forward(self, z_stack):
        # z_stack: [N,V,D_z]
        _require(
            torch.is_tensor(z_stack)
            and z_stack.ndim == 3
            and tuple(z_stack.shape[1:])
            == (self.view_num, self.latent_dim),
            "z_stack must have shape [N,V,D_z]",
        )
        semantic_views = []
        for view_id, projector in enumerate(self.projectors):
            h_raw = projector(z_stack[:, view_id, :])
            semantic_views.append(F.normalize(h_raw, p=2, dim=-1))
        # h_stack: [N,V,D_h]
        h_stack = torch.stack(semantic_views, dim=1)
        # logits: [N,V,K]
        logits = self.classifier(h_stack)
        return h_stack, logits


def _validate_semantic_admission(admission, h_stack):
    mask = torch.as_tensor(admission, dtype=torch.bool, device=h_stack.device)
    _require(
        mask.shape == h_stack.shape[:2],
        "semantic admission must have shape [N,V]",
    )
    _require(
        bool(torch.all(mask.sum(dim=1) > 0).item()),
        "every sample needs at least one semantic admission",
    )
    return mask


def admitted_symmetric_infonce(h_stack, admission, temperature=TEMPERATURE):
    """Exact B6 sample-count-weighted admitted-only symmetric InfoNCE."""
    _require(h_stack.ndim == 3, "h_stack must have shape [N,V,D_h]")
    temperature = float(temperature)
    _require(temperature == TEMPERATURE, "temperature must remain frozen at 0.2")
    mask = _validate_semantic_admission(admission, h_stack)
    weighted_loss = None
    weighted_sample_count = 0
    pair_sample_counts = {}
    for view_v in range(h_stack.shape[1]):
        for view_w in range(view_v + 1, h_stack.shape[1]):
            selected = torch.logical_and(mask[:, view_v], mask[:, view_w])
            sample_ids = torch.nonzero(selected, as_tuple=False).flatten()
            sample_count = int(sample_ids.numel())
            pair_name = str(view_v) + "-" + str(view_w)
            pair_sample_counts[pair_name] = sample_count
            if sample_count < 2:
                continue
            h_v = h_stack[sample_ids, view_v, :]
            h_w = h_stack[sample_ids, view_w, :]
            pair_logits = torch.matmul(h_v, h_w.T) / temperature
            targets = torch.arange(
                sample_count, dtype=torch.long, device=h_stack.device
            )
            pair_loss = 0.5 * (
                F.cross_entropy(pair_logits, targets)
                + F.cross_entropy(pair_logits.T, targets)
            )
            contribution = sample_count * pair_loss
            weighted_loss = (
                contribution
                if weighted_loss is None
                else weighted_loss + contribution
            )
            weighted_sample_count += sample_count
    _require(
        weighted_loss is not None and weighted_sample_count > 0,
        "no eligible admitted view pair",
    )
    loss = weighted_loss / weighted_sample_count
    _require(bool(torch.isfinite(loss).item()), "unsupervised loss is non-finite")
    return loss, {
        "pair_sample_counts": pair_sample_counts,
        "weighted_sample_count": int(weighted_sample_count),
    }


def sparse_supervised_cross_entropy(
    logits,
    admission_mask,
    labeled_sample_ids,
    labeled_targets,
):
    """Mean hard-admission CE using only explicitly supplied sparse targets."""
    # admission_mask: [N,V]
    # logits: [N,V,K]
    # labels: [N] conceptually, but full labels are deliberately not accepted;
    #         labeled_targets: [L] is the only target tensor in this API.
    # labeled_mask: [N], constructed only from labeled_sample_ids below.
    _require(logits.ndim == 3, "logits must have shape [N,V,K]")
    mask = torch.as_tensor(
        admission_mask, dtype=torch.bool, device=logits.device
    )
    _require(mask.shape == logits.shape[:2], "admission_mask shape mismatch")
    ids = torch.as_tensor(
        labeled_sample_ids, dtype=torch.long, device=logits.device
    )
    targets = torch.as_tensor(
        labeled_targets, dtype=torch.long, device=logits.device
    )
    _require(
        ids.ndim == 1 and targets.shape == ids.shape,
        "sparse IDs and targets must align",
    )
    _require(
        ids.numel() == torch.unique(ids).numel()
        and bool(torch.all((ids >= 0) & (ids < logits.shape[0])).item()),
        "labeled_sample_ids are invalid",
    )
    labeled_mask = torch.zeros(
        logits.shape[0], dtype=torch.bool, device=logits.device
    )
    labeled_mask[ids] = True
    _require(
        not bool(mask[~labeled_mask].any().item()),
        "supervision admission exists outside labeled samples",
    )
    channel_count = int(mask.sum().item())
    if channel_count == 0:
        return logits.sum() * 0.0, 0
    sparse_logits = logits[ids]
    sparse_mask = mask[ids]
    per_channel = F.cross_entropy(
        sparse_logits.reshape(-1, logits.shape[-1]),
        targets[:, None].expand(-1, logits.shape[1]).reshape(-1),
        reduction="none",
    ).reshape(ids.numel(), logits.shape[1])
    loss = torch.sum(per_channel * sparse_mask.to(per_channel.dtype)) / max(
        channel_count, 1
    )
    _require(bool(torch.isfinite(loss).item()), "supervised loss is non-finite")
    return loss, channel_count


def compute_b7_losses(
    model,
    z_stack,
    unsupervised_admission,
    supervised_admission,
    labeled_sample_ids,
    labeled_targets,
    lambda_sup=LAMBDA_SUP,
):
    """Compute B7 losses without any full or unlabeled target vector."""
    _require(not z_stack.requires_grad, "frozen z_stack must be detached")
    h_stack, logits = model(z_stack)
    unsup_loss, unsup_audit = admitted_symmetric_infonce(
        h_stack, unsupervised_admission, TEMPERATURE
    )
    sup_loss, channel_count = sparse_supervised_cross_entropy(
        logits,
        supervised_admission,
        labeled_sample_ids,
        labeled_targets,
    )
    total_loss = unsup_loss + float(lambda_sup) * sup_loss
    _require(bool(torch.isfinite(total_loss).item()), "total loss is non-finite")
    return total_loss, unsup_loss, sup_loss, channel_count, unsup_audit


def shared_semantic_readout(h_stack, admission):
    """Use the common frozen-U semantic admission for every arm's readout."""
    mask = _validate_semantic_admission(admission, h_stack)
    weights = mask.to(h_stack.dtype).unsqueeze(-1)
    semantic = torch.sum(h_stack * weights, dim=1) / torch.sum(weights, dim=1)
    return F.normalize(semantic, p=2, dim=-1)


def cluster_metrics_by_partition(
    evaluation_targets,
    prediction,
    labeled_sample_ids,
    unlabeled_sample_ids,
):
    """Final-evaluation-only ALL/unlabeled/labeled clustering metrics."""
    targets = np.asarray(evaluation_targets, dtype=np.int64)
    predicted = np.asarray(prediction, dtype=np.int64)
    labeled_ids = np.asarray(labeled_sample_ids, dtype=np.int64)
    unlabeled_ids = np.asarray(unlabeled_sample_ids, dtype=np.int64)
    _require(
        targets.ndim == predicted.ndim == 1 and targets.shape == predicted.shape,
        "evaluation target/prediction shape mismatch",
    )

    def metrics(ids):
        return {
            "ACC": float(mvcan_acc(targets[ids], predicted[ids])),
            "NMI": float(mvcan_nmi(targets[ids], predicted[ids])),
            "ARI": float(mvcan_ari(targets[ids], predicted[ids])),
        }

    all_ids = np.arange(targets.size, dtype=np.int64)
    all_metrics = metrics(all_ids)
    unlabeled_metrics = metrics(unlabeled_ids)
    labeled_metrics = metrics(labeled_ids)
    return {
        "ACC_all": all_metrics["ACC"],
        "NMI_all": all_metrics["NMI"],
        "ARI_all": all_metrics["ARI"],
        "ACC_unlabeled": unlabeled_metrics["ACC"],
        "NMI_unlabeled": unlabeled_metrics["NMI"],
        "ARI_unlabeled": unlabeled_metrics["ARI"],
        "ACC_labeled": labeled_metrics["ACC"],
        "NMI_labeled": labeled_metrics["NMI"],
        "ARI_labeled": labeled_metrics["ARI"],
    }


def load_frozen_b7_inputs(
    backbone_dir=DEFAULT_BACKBONE_DIR,
    d2_dir=DEFAULT_D2_DIR,
):
    """Load D1/D2 frozen inputs and return detached [1400,6,10] z."""
    backbone_dir = _resolve(backbone_dir)
    d2_dir = _resolve(d2_dir)
    utility_path = d2_dir / "utility_scores.npz"
    transfer_path = d2_dir / "utility_transfer.json"
    mask_path = d2_dir / "corruption_audit/corruption_mask.npy"
    _require(backbone_dir.is_dir(), "frozen D1 backbone directory is missing")
    _require(
        utility_path.is_file() and transfer_path.is_file() and mask_path.is_file(),
        "frozen D2 inputs are incomplete",
    )
    transfer = _read_json(transfer_path)
    _require(
        transfer.get("D2_A0_UTILITY_TRANSFER_PASS") is True,
        "D2-A0 frozen transfer gate is not closed",
    )
    with np.load(utility_path, allow_pickle=False) as archive:
        _require(set(archive.files) == {"T", "U", "corruption_mask"}, "D2 keys changed")
        utility = np.ascontiguousarray(archive["U"], dtype=np.float64)
        archived_mask = np.asarray(archive["corruption_mask"], dtype=bool)
    stored_mask = np.load(mask_path, allow_pickle=False)
    _require(
        utility.shape == (SAMPLE_NUM, VIEW_NUM)
        and np.isfinite(utility).all()
        and float(utility.min()) >= 0.0
        and float(utility.max()) <= 1.0,
        "frozen Utility boundary mismatch",
    )
    utility_hash = tensor_sha256(utility)
    utility_file_hash = _file_sha256(utility_path)
    _require(
        utility_hash == transfer.get("U_sha256") == EXPECTED_D2_U_SHA256
        and utility_file_hash == EXPECTED_D2_UTILITY_FILE_SHA256,
        "frozen Utility tensor SHA mismatch",
    )
    _require(
        stored_mask.dtype == np.dtype(bool)
        and stored_mask.shape == (SAMPLE_NUM, VIEW_NUM)
        and np.array_equal(archived_mask, stored_mask)
        and np.all(stored_mask.sum(axis=1) == CORRUPTION_K)
        and ndarray_sha256(stored_mask) == EXPECTED_CORRUPTION_MASK_SHA256,
        "D2 corruption mask boundary mismatch",
    )

    clean_views, label_list = load_data({"dataset": DATASET_NAME})
    evaluation_targets = np.asarray(label_list[0], dtype=np.int64)
    _require(
        len(clean_views) == VIEW_NUM
        and [int(view.shape[1]) for view in clean_views] == EXPECTED_VIEW_DIMS
        and evaluation_targets.shape == (SAMPLE_NUM,)
        and np.array_equal(np.unique(evaluation_targets), np.arange(CLUSTER_NUM)),
        "Caltech-6V dataset boundary mismatch",
    )
    evaluation_views, corruption_audit = apply_weak_quality_protocol(
        clean_views,
        mode="heterogeneous_gaussian",
        k=CORRUPTION_K,
        snr_db=SNR_DB,
        corruption_seed=CORRUPTION_SEED,
    )
    _require(
        np.array_equal(corruption_audit["mask"], stored_mask),
        "runtime corruption mask differs from frozen D2 mask",
    )
    config = get_default_config(DATASET_NAME)
    models, latent_views, d2_backbone_audit = d2._load_native_representations(
        config=config,
        evaluation_views=evaluation_views,
        backbone_dir=backbone_dir,
        sample_num=SAMPLE_NUM,
        view_num=VIEW_NUM,
        cluster_num=CLUSTER_NUM,
        model_seed=MODEL_SEED,
    )
    for autoencoder in models.autoencoders:
        autoencoder.eval()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
    # z_v: [N, latent_dim]; z_all: [N,V,latent_dim].
    z_stack = torch.stack(
        [latent.detach() for latent in latent_views], dim=1
    ).detach()
    _require(
        z_stack.shape == (SAMPLE_NUM, VIEW_NUM, LATENT_DIM)
        and not z_stack.requires_grad
        and bool(torch.isfinite(z_stack).all().item()),
        "frozen MVCAN z_stack boundary mismatch",
    )
    backbone_hash = hash_backbone(models.autoencoders)
    _require(
        backbone_hash == d2_backbone_audit["backbone_hash_before"],
        "backbone changed during B7 frozen extraction",
    )
    requires_grad_pass = bool(
        all(
            not parameter.requires_grad and parameter.grad is None
            for module in models.autoencoders
            for parameter in module.parameters()
        )
    )
    utility.setflags(write=False)
    return {
        "models": models,
        "z_stack": z_stack,
        "evaluation_targets": evaluation_targets,
        "utility": utility,
        "utility_sha256": utility_hash,
        "utility_file_sha256": utility_file_hash,
        "utility_source": _display(utility_path),
        "corruption_mask": stored_mask,
        "corruption_mask_sha256": ndarray_sha256(stored_mask),
        "backbone_hash_before": backbone_hash,
        "backbone_requires_grad_pass": requires_grad_pass,
        "checkpoint_paths": d2_backbone_audit["checkpoint_paths"],
        "checkpoint_file_sha256": d2_backbone_audit[
            "checkpoint_file_sha256"
        ],
    }


def build_all_arm_admissions(utility, corruption_mask, labeled_sample_ids):
    """Construct normal policies first, then the isolated oracle policy."""
    unsupervised_admission = frozen_u_topk_admission(utility)
    _require(
        np.all(unsupervised_admission.sum(axis=1) == TOP_K),
        "common unsupervised admission must select exactly Top3",
    )
    admissions = {}
    shuffle_mapping = None
    for arm in ARMS[:-1]:
        admissions[arm], mapping = build_normal_supervised_admission(
            arm,
            utility,
            labeled_sample_ids,
            shuffle_seed=SHUFFLE_SEED,
        )
        if arm == "SHUFFLED_U_LABEL":
            shuffle_mapping = mapping
    # The corruption mask first enters only this diagnostic-only builder.
    admissions["ORACLE_LABEL"] = build_oracle_supervised_admission(
        corruption_mask, labeled_sample_ids
    )
    return unsupervised_admission, admissions, shuffle_mapping


def initial_semantic_state(seed=MODEL_SEED):
    set_explicit_rng_seeds(seed)
    template = B7SemanticCarrier()
    state = copy.deepcopy(template.state_dict())
    return state, hash_state_dict(state)


def semantic_optimizer_step(
    model,
    optimizer,
    z_stack,
    unsupervised_admission,
    supervised_admission,
    labeled_sample_ids,
    labeled_targets,
):
    """One full-batch update, exposed for exact gradient/freeze tests."""
    optimizer.zero_grad(set_to_none=True)
    losses = compute_b7_losses(
        model,
        z_stack,
        unsupervised_admission,
        supervised_admission,
        labeled_sample_ids,
        labeled_targets,
        lambda_sup=LAMBDA_SUP,
    )
    losses[0].backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    gradient_finite_pass = bool(
        gradients
        and all(
            gradient is not None and torch.isfinite(gradient).all().item()
            for gradient in gradients
        )
    )
    _require(gradient_finite_pass, "semantic gradient audit failed")
    optimizer.step()
    return losses, gradient_finite_pass


def train_one_arm(
    arm,
    z_stack,
    unsupervised_admission,
    supervised_admission,
    labeled_sample_ids,
    labeled_targets,
    initial_state,
    initial_state_sha256,
    epochs,
    output_dir,
):
    """Train one arm; this API cannot receive full or unlabeled labels."""
    arm = str(arm).upper()
    _require(arm in ARMS, "unsupported B7 arm")
    epochs = int(epochs)
    _require(epochs in (SMOKE_EPOCHS, PILOT_EPOCHS), "epochs are not preregistered")
    _require(
        z_stack.shape == (SAMPLE_NUM, VIEW_NUM, LATENT_DIM)
        and not z_stack.requires_grad,
        "arm received a non-frozen z_stack",
    )
    supervised_array = np.asarray(supervised_admission, dtype=bool)
    expected_channels = {
        "UNSUP": 0,
        "LABEL_ONLY": 84,
        "U_LABEL": 42,
        "SHUFFLED_U_LABEL": 42,
        "ORACLE_LABEL": 42,
    }[arm]
    _require(
        supervised_channel_count(supervised_array) == expected_channels,
        "supervised channel count mismatch for " + arm,
    )
    model = B7SemanticCarrier()
    model.load_state_dict(copy.deepcopy(initial_state), strict=True)
    arm_initial_hash = hash_state_dict(model.state_dict())
    _require(
        arm_initial_hash == initial_state_sha256,
        "semantic-head initialization mismatch",
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history = []
    gradient_finite_pass = True
    fixed_pair_counts = None
    for epoch_id in range(epochs):
        losses, finite_gradient = semantic_optimizer_step(
            model,
            optimizer,
            z_stack,
            unsupervised_admission,
            supervised_array,
            labeled_sample_ids,
            labeled_targets,
        )
        total_loss, unsup_loss, sup_loss, channel_count, unsup_audit = losses
        gradient_finite_pass = bool(gradient_finite_pass and finite_gradient)
        if fixed_pair_counts is None:
            fixed_pair_counts = dict(unsup_audit["pair_sample_counts"])
        else:
            _require(
                fixed_pair_counts == unsup_audit["pair_sample_counts"],
                "common unsupervised pair counts changed",
            )
        history.append({
            "epoch": int(epoch_id + 1),
            "loss": float(total_loss.detach().item()),
            "unsup_loss": float(unsup_loss.detach().item()),
            "sup_loss": float(sup_loss.detach().item()),
            "supervised_channel_count": int(channel_count),
            "weighted_unsup_pair_sample_count": int(
                unsup_audit["weighted_sample_count"]
            ),
            "loss_finite": bool(torch.isfinite(total_loss).item()),
            "gradient_finite": bool(finite_gradient),
        })

    model.eval()
    with torch.no_grad():
        final_h, _ = model(z_stack)
        final_semantic = shared_semantic_readout(
            final_h, unsupervised_admission
        )
    semantic_array = np.ascontiguousarray(
        final_semantic.detach().cpu().numpy(), dtype=np.float32
    )
    prediction = KMeans(
        n_clusters=CLUSTER_NUM,
        n_init=KMEANS_N_INIT,
        random_state=MODEL_SEED,
    ).fit_predict(semantic_array)
    prediction = np.ascontiguousarray(prediction, dtype=np.int64)
    all_finite_pass = bool(
        np.isfinite(semantic_array).all()
        and all(row["loss_finite"] and row["gradient_finite"] for row in history)
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    np.save(root / "final_semantic.npy", semantic_array)
    np.save(root / "prediction.npy", prediction)
    np.save(root / "supervised_admission_mask.npy", supervised_array)
    np.save(
        root / "unsupervised_admission_mask.npy",
        np.asarray(unsupervised_admission, dtype=bool),
    )
    np.save(
        root / "training_labeled_sample_ids.npy",
        np.asarray(labeled_sample_ids, dtype=np.int64),
    )
    torch.save(model.state_dict(), root / "final_semantic_head.pt")
    _write_json(root / "training_history.json", {"epochs": epochs, "rows": history})
    metadata = {
        "stage": STAGE,
        "arm": arm,
        "dataset": DATASET_NAME,
        "model_seed": MODEL_SEED,
        "corruption_seed": CORRUPTION_SEED,
        "label_split_seed": LABEL_SPLIT_SEED,
        "labeled_count": int(len(labeled_sample_ids)),
        "unlabeled_count": int(SAMPLE_NUM - len(labeled_sample_ids)),
        "labels_per_class": LABELS_PER_CLASS,
        "supervised_channel_count": expected_channels,
        "epochs": epochs,
        "batch_size": SAMPLE_NUM,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "temperature": TEMPERATURE,
        "lambda_sup": LAMBDA_SUP,
        "projector_architecture": (
            "per-view Linear(10,32)-GELU-Linear(32,10)-L2"
        ),
        "classifier_architecture": "shared Linear(10,7)",
        "initial_head_sha256": arm_initial_hash,
        "initial_semantic_head_sha256": arm_initial_hash,
        "final_head_sha256": hash_state_dict(model.state_dict()),
        "semantic_sha256": tensor_sha256(semantic_array),
        "prediction_sha256": tensor_sha256(prediction),
        "training_labeled_ids_sha256": ndarray_sha256(
            np.asarray(labeled_sample_ids, dtype=np.int64)
        ),
        "full_labels_available_to_training_function": False,
        "unlabeled_labels_available_to_training_function": False,
        "utility_values_available_to_training_function": False,
        "corruption_mask_available_to_training_function": False,
        "unsupervised_objective": (
            "B6/WQ1A1 sample-count-weighted admitted symmetric InfoNCE"
        ),
        "unsupervised_admission_identical_across_arms": True,
        "semantic_readout_identical_across_arms": True,
        "gradient_finite_pass": gradient_finite_pass,
        "all_finite_pass": all_finite_pass,
        "final_loss": history[-1]["loss"],
        "final_unsup_loss": history[-1]["unsup_loss"],
        "final_sup_loss": history[-1]["sup_loss"],
    }
    _write_json(root / "metadata.json", metadata)
    return {
        "arm": arm,
        "output_dir": root,
        "metadata": metadata,
        "prediction": prediction,
        "initial_head_sha256": arm_initial_hash,
        "all_finite_pass": all_finite_pass,
    }


def _update_arm_metadata(record, common_audit):
    metadata = dict(record["metadata"])
    metadata.update(common_audit)
    _write_json(record["output_dir"] / "metadata.json", metadata)
    record["metadata"] = metadata


def run_experiment(
    smoke=False,
    output_dir=None,
    backbone_dir=DEFAULT_BACKBONE_DIR,
    d2_dir=DEFAULT_D2_DIR,
):
    """Run smoke (3 epochs) or the frozen 100-epoch B7-A0 pilot."""
    smoke = bool(smoke)
    epochs = SMOKE_EPOCHS if smoke else PILOT_EPOCHS
    if output_dir is None:
        leaf = "b7a0_seed20_smoke" if smoke else "b7a0_seed20"
        output_root = DEFAULT_OUTPUT_ROOT / leaf
    else:
        output_root = _resolve(output_dir)
    _require(not output_root.exists(), "refusing to overwrite a B7 output")
    output_root.mkdir(parents=True)

    set_explicit_rng_seeds(MODEL_SEED)
    frozen = load_frozen_b7_inputs(backbone_dir=backbone_dir, d2_dir=d2_dir)
    utility_hash_before = tensor_sha256(frozen["utility"])
    split = make_class_balanced_split(
        frozen["evaluation_targets"],
        labels_per_class=LABELS_PER_CLASS,
        seed=LABEL_SPLIT_SEED,
    )
    split_record = save_label_split(split, output_root, dataset=DATASET_NAME)
    labeled_ids = split["labeled_sample_ids"]
    unlabeled_ids = split["unlabeled_sample_ids"]
    # Only these fourteen targets cross the training boundary.
    labeled_targets = np.ascontiguousarray(
        frozen["evaluation_targets"][labeled_ids], dtype=np.int64
    )

    unsup_admission, supervised_admissions, shuffle_mapping = (
        build_all_arm_admissions(
            frozen["utility"], frozen["corruption_mask"], labeled_ids
        )
    )
    initial_state, initial_hash = initial_semantic_state(MODEL_SEED)
    torch.save(initial_state, output_root / "initial_semantic_head.pt")
    _write_json(
        output_root / "initial_semantic_head.json",
        {"initial_semantic_head_sha256": initial_hash, "model_seed": MODEL_SEED},
    )

    records = []
    directory_names = {
        "UNSUP": "unsup",
        "LABEL_ONLY": "label_only",
        "U_LABEL": "u_label",
        "SHUFFLED_U_LABEL": "shuffled_u_label",
        "ORACLE_LABEL": "oracle_label",
    }
    for arm in ARMS:
        record = train_one_arm(
            arm=arm,
            z_stack=frozen["z_stack"],
            unsupervised_admission=unsup_admission,
            supervised_admission=supervised_admissions[arm],
            labeled_sample_ids=labeled_ids,
            labeled_targets=labeled_targets,
            initial_state=initial_state,
            initial_state_sha256=initial_hash,
            epochs=epochs,
            output_dir=output_root / directory_names[arm],
        )
        records.append(record)
    np.save(
        output_root / "shuffled_u_label/shuffle_mapping.npy",
        np.asarray(shuffle_mapping, dtype=np.int64),
    )

    # Full labels first re-enter after every arm has completed training.
    for record in records:
        metrics = cluster_metrics_by_partition(
            frozen["evaluation_targets"],
            record["prediction"],
            labeled_ids,
            unlabeled_ids,
        )
        _write_json(record["output_dir"] / "metrics.json", metrics)
        record["metrics"] = metrics

    backbone_hash_after = hash_backbone(frozen["models"].autoencoders)
    backbone_requires_grad_pass = bool(
        frozen["backbone_requires_grad_pass"]
        and all(
            not parameter.requires_grad and parameter.grad is None
            for module in frozen["models"].autoencoders
            for parameter in module.parameters()
        )
    )
    backbone_immutability_pass = bool(
        backbone_hash_after == frozen["backbone_hash_before"]
    )
    utility_hash_after = tensor_sha256(frozen["utility"])
    u_frozen_input_pass = bool(
        not frozen["utility"].flags.writeable
        and utility_hash_before == utility_hash_after == frozen["utility_sha256"]
    )
    initial_hashes = [record["initial_head_sha256"] for record in records]
    same_initialization_pass = bool(
        len(initial_hashes) == len(ARMS)
        and all(value == initial_hash for value in initial_hashes)
    )
    normal_parameters = inspect.signature(
        build_normal_supervised_admission
    ).parameters
    oracle_isolation_pass = bool(
        all(
            token not in name.lower()
            for name in normal_parameters
            for token in ("oracle", "corrupt", "clean", "mask")
        )
        and records[-1]["metadata"][
            "corruption_mask_available_to_training_function"
        ]
        is False
    )
    shuffled_again, mapping_again = deterministic_within_sample_view_shuffle(
        frozen["utility"], shuffle_seed=SHUFFLE_SEED
    )
    shuffle_deterministic_pass = bool(
        np.array_equal(mapping_again, shuffle_mapping)
        and np.array_equal(
            frozen_u_topk_admission(shuffled_again)[labeled_ids],
            supervised_admissions["SHUFFLED_U_LABEL"][labeled_ids],
        )
    )
    label_split_sha = split_record["label_split_sha256"]
    no_unlabeled_label_training_pass = bool(
        all(
            record["metadata"]["full_labels_available_to_training_function"]
            is False
            and record["metadata"][
                "unlabeled_labels_available_to_training_function"
            ]
            is False
            and record["metadata"]["training_labeled_ids_sha256"]
            == split["labeled_ids_sha256"]
            for record in records
        )
    )
    all_finite_pass = bool(
        all(
            record["all_finite_pass"]
            and all(np.isfinite(value) for value in record["metrics"].values())
            for record in records
        )
    )
    common_audit = {
        "backbone_requires_grad_pass": backbone_requires_grad_pass,
        "backbone_parameter_immutability_pass": backbone_immutability_pass,
        "u_frozen_input_pass": u_frozen_input_pass,
        "u_sha256": frozen["utility_sha256"],
        "label_split_sha256": label_split_sha,
        "initial_head_sha256": initial_hash,
        "initial_semantic_head_sha256": initial_hash,
        "same_initialization_pass": same_initialization_pass,
        "no_unlabeled_label_training_pass": no_unlabeled_label_training_pass,
        "oracle_isolation_pass": oracle_isolation_pass,
        "shuffle_deterministic_pass": shuffle_deterministic_pass,
        "all_finite_pass": all_finite_pass,
    }
    for record in records:
        _update_arm_metadata(record, common_audit)

    metadata = {
        "stage": STAGE,
        "mode": "smoke" if smoke else "pilot",
        "dataset": DATASET_NAME,
        "N": SAMPLE_NUM,
        "V": VIEW_NUM,
        "K": CLUSTER_NUM,
        "model_seed": MODEL_SEED,
        "corruption_seed": CORRUPTION_SEED,
        "corruption_k": CORRUPTION_K,
        "snr_db": SNR_DB,
        "label_split_seed": LABEL_SPLIT_SEED,
        "labels_per_class": LABELS_PER_CLASS,
        "labeled_count": int(labeled_ids.size),
        "unlabeled_count": int(unlabeled_ids.size),
        "epochs": epochs,
        "lambda_sup": LAMBDA_SUP,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "temperature": TEMPERATURE,
        "batch_size": SAMPLE_NUM,
        "kmeans_n_init": KMEANS_N_INIT,
        "kmeans_random_state": MODEL_SEED,
        "common_unsupervised_admission": "frozen D2 U Top3",
        "common_unsupervised_admission_sha256": ndarray_sha256(
            unsup_admission
        ),
        "common_semantic_readout": "mean frozen-U-admitted h then L2 normalize",
        "supervised_channel_counts": {
            record["arm"]: record["metadata"]["supervised_channel_count"]
            for record in records
        },
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle_mapping_sha256": ndarray_sha256(shuffle_mapping),
        "shuffle_mapping_source": _display(
            output_root / "shuffled_u_label/shuffle_mapping.npy"
        ),
        "utility_source": frozen["utility_source"],
        "utility_file_sha256": frozen["utility_file_sha256"],
        "corruption_mask_sha256": frozen["corruption_mask_sha256"],
        "backbone_checkpoint_paths": frozen["checkpoint_paths"],
        "backbone_checkpoint_file_sha256": frozen["checkpoint_file_sha256"],
        "backbone_hash_before": frozen["backbone_hash_before"],
        "backbone_hash_after": backbone_hash_after,
        "training_api_accepts_full_labels": False,
        "training_api_accepts_unlabeled_labels": False,
        "training_complete_before_full_label_evaluation": True,
        "oracle_diagnostic_only": True,
        **common_audit,
    }
    _write_json(output_root / "metadata.json", metadata)

    from experiments.b7_sparse_supervision.summarize_b7_a0_sparse_supervision import (
        summarize_outputs,
    )
    from experiments.b7_sparse_supervision.audit_b7_a0_outputs import (
        audit_outputs,
    )

    summarize_outputs(output_root)
    audit_outputs(output_root)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the fixed three-epoch structural smoke instead of 100 epochs",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--backbone-dir", default=str(DEFAULT_BACKBONE_DIR))
    parser.add_argument("--d2-dir", default=str(DEFAULT_D2_DIR))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    metadata = run_experiment(
        smoke=args.smoke,
        output_dir=args.output_dir,
        backbone_dir=args.backbone_dir,
        d2_dir=args.d2_dir,
    )
    print("B7_A0_ENGINEERING_PASS=" + str(all((
        metadata["backbone_requires_grad_pass"],
        metadata["backbone_parameter_immutability_pass"],
        metadata["u_frozen_input_pass"],
        metadata["same_initialization_pass"],
        metadata["no_unlabeled_label_training_pass"],
        metadata["oracle_isolation_pass"],
        metadata["shuffle_deterministic_pass"],
        metadata["all_finite_pass"],
    ))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
