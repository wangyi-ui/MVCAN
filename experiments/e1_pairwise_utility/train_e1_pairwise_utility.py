"""Controlled E1 continuation from the frozen D1 noisy MVCAN checkpoint.

The trainable boundary contains only six noisy feature views, canonical sample
IDs, MVCAN's own targets, and frozen D2 reliability.  Labels are loaded only
after final predictions have been saved and hashed.
"""

import argparse
import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ClusteringTest import acc as clustering_accuracy
from configure import get_default_config
from irv.b3_audit import hash_backbone
from irv.b4_information_utility import tensor_sha256
from model import MvCAN
from weak_quality import ndarray_sha256

from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import ARMS
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    EXPECTED_RELIABILITY_SHA256,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    GLGC_CONTRASTIVE_TEMPERATURE,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    GLGC_ETA,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    GLGC_GRAPH_TEMPERATURE,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    GLGC_RELEASED_COMMIT,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    align_semantic_probabilities,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    pairwise_semantic_cooperation_loss,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    robust_inter_affinity,
)
from experiments.e1_pairwise_utility.pairwise_semantic_cooperation import (
    whole_row_reliability_shuffle,
)


STAGE = "E1"
DATASET = "Caltech-6V"
TRAINING_SEED_CHOICES = (20, 30, 50)
DEFAULT_TRAINING_SEED = 20
# Backward-compatible public default used by historical callers.
SEED = DEFAULT_TRAINING_SEED
WEAK_QUALITY_CONDITION = "snr2p5_k3_seed20"
SAMPLE_NUM = 1400
VIEW_NUM = 6
CLUSTER_NUM = 7
VIEW_DIMS = (48, 40, 254, 1984, 512, 928)
LEARNING_RATE = 1e-4
LAMBDA1 = 0.01
TARGET_REFRESH_INTERVAL = 100
TARGET_WEIGHT_UPDATES = 2
KMEANS_N_INIT = 100

DEFAULT_CHECKPOINT_DIR = (
    REPOSITORY_ROOT / "outputs/d1_caltech6v/snr2p5_k3_seed20/models"
)
DEFAULT_RELIABILITY_PATH = (
    REPOSITORY_ROOT
    / "outputs/d2_caltech6v/a0_utility_transfer_seed20/utility_scores.npz"
)
DEFAULT_D2_AUDIT_PATH = DEFAULT_RELIABILITY_PATH.with_name(
    "utility_transfer.json"
)
DEFAULT_FEATURE_PATH = (
    REPOSITORY_ROOT
    / "outputs/e0_glgc_adapter/caltech6v_snr2p5_k3_seed20.npz"
)
DEFAULT_FEATURE_AUDIT_PATH = (
    REPOSITORY_ROOT / "outputs/e0_glgc_adapter/export_audit.json"
)
DEFAULT_GLGC_REPOSITORY = Path("/root/autodl-tmp/GLGC-CVPR26")
DEFAULT_LABEL_PATH = REPOSITORY_ROOT / "data/Caltech.mat"

EXPECTED_GLGC_FILE_SHA256 = {
    "loss.py": "98fa224968429fe725fdfc88264a1d43760f14f9d451f4885865eca46cacda78",
    "util.py": "9530695c20c690a666b87407e2c832d8183b090a87e4277e067dfc726d8ff341",
}
EXPECTED_FEATURE_KEYS = tuple("X" + str(index) for index in range(1, 7)) + (
    "sample_ids",
)


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


def validate_training_seed(seed):
    training_seed = int(seed)
    if training_seed not in TRAINING_SEED_CHOICES:
        raise ValueError("E1 training seed must be one of 20, 30, 50")
    return training_seed


def default_output_dir(
    arm, epochs, training_seed=DEFAULT_TRAINING_SEED
):
    seed = validate_training_seed(training_seed)
    return (
        REPOSITORY_ROOT
        / "outputs/e1_pairwise_utility"
        / (
            str(arm).lower()
            + "_"
            + str(int(epochs))
            + "ep_seed"
            + str(seed)
        )
    )


def seed_semantics_audit(training_seed):
    seed = validate_training_seed(training_seed)
    return {
        "training_seed": seed,
        "weak_quality_condition_fixed": WEAK_QUALITY_CONDITION,
        "weak_quality_realization_varied": False,
    }


def training_seed_audit_fields(training_seed):
    seed = validate_training_seed(training_seed)
    return {
        "seed": seed,
        "training_seed": seed,
        "sample_order_rng_seed": seed,
        "weak_quality_condition": WEAK_QUALITY_CONDITION,
        "seed_semantics": seed_semantics_audit(seed),
    }


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_global_seed(seed=SEED):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_state_dict_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def verify_glgc_released_logic(glgc_repository=DEFAULT_GLGC_REPOSITORY):
    """Read-only verification of the two released files ported by E1."""
    repository = Path(glgc_repository).resolve()
    _require(repository.is_dir(), "GLGC reference repository is missing")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repository), text=True
    ).strip()
    hashes = {
        name: file_sha256(repository / name)
        for name in EXPECTED_GLGC_FILE_SHA256
    }
    _require(head == GLGC_RELEASED_COMMIT, "GLGC reference commit mismatch")
    _require(
        hashes == EXPECTED_GLGC_FILE_SHA256,
        "GLGC released loss.py/util.py content mismatch",
    )
    return {
        "repository": str(repository),
        "commit": head,
        "released_file_sha256": hashes,
        "released_logic_match_pass": True,
        "ported_logic": [
            "util.py:robust_affinity noise-path local cross-view graph",
            "loss.py:Loss.forward_inter_noise positive-logit weighting",
        ],
        "fixed_graph_temperature": GLGC_GRAPH_TEMPERATURE,
        "fixed_contrastive_temperature": GLGC_CONTRASTIVE_TEMPERATURE,
        "fixed_eta": GLGC_ETA,
    }


def load_frozen_feature_artifact(
    feature_path=DEFAULT_FEATURE_PATH,
    feature_audit_path=DEFAULT_FEATURE_AUDIT_PATH,
):
    """Load only X1..X6 and sample_ids; no labels or oracle masks exist."""
    feature_path = _resolve(feature_path)
    feature_audit_path = _resolve(feature_audit_path)
    _require(feature_path.is_file(), "frozen noisy feature artifact is missing")
    _require(feature_audit_path.is_file(), "E0 feature audit is missing")
    audit = _read_json(feature_audit_path)
    with np.load(feature_path, allow_pickle=False) as archive:
        keys = tuple(archive.files)
        _require(keys == EXPECTED_FEATURE_KEYS, "feature artifact key mismatch")
        views = [
            np.ascontiguousarray(archive["X" + str(view_id + 1)])
            for view_id in range(VIEW_NUM)
        ]
        sample_ids = np.ascontiguousarray(archive["sample_ids"])

    _require(audit.get("EXPORT_PASS") is True, "E0 export audit did not pass")
    _require(
        audit.get("trainable_artifact_forbidden_fields_absent") is True,
        "E0 artifact forbidden-field audit failed",
    )
    _require(
        set(keys).isdisjoint(
            {"Y", "labels", "corruption_mask", "clean_mask", "noise_mask"}
        ),
        "label or corruption mask reached the trainable artifact",
    )
    for view_id, (view, dimension) in enumerate(zip(views, VIEW_DIMS)):
        _require(
            view.shape == (SAMPLE_NUM, dimension)
            and view.dtype == np.dtype(np.float32)
            and np.isfinite(view).all(),
            "feature boundary mismatch for view " + str(view_id + 1),
        )
    canonical_ids = np.arange(SAMPLE_NUM, dtype=np.int64)
    _require(
        sample_ids.dtype == np.dtype(np.int64)
        and sample_ids.shape == (SAMPLE_NUM,)
        and np.array_equal(sample_ids, canonical_ids),
        "sample IDs must be exact canonical arange(1400)",
    )
    sample_hash = ndarray_sha256(sample_ids)
    _require(
        sample_hash == audit["sample_ids"]["sha256"],
        "sample ID logical hash mismatch against E0 audit",
    )
    expected_view_hashes = audit["noisy_artifact"]["view_content_sha256"]
    view_hashes = [ndarray_sha256(view) for view in views]
    _require(view_hashes == expected_view_hashes, "noisy view hash mismatch")
    return views, sample_ids, {
        "path": _display(feature_path),
        "file_sha256": file_sha256(feature_path),
        "keys": list(keys),
        "sample_ids_shape": list(sample_ids.shape),
        "sample_ids_dtype": str(sample_ids.dtype),
        "sample_ids_sha256": sample_hash,
        "sample_ids_exact_arange_pass": True,
        "view_content_sha256": view_hashes,
        "labels_present": False,
        "corruption_mask_present": False,
        "forbidden_fields_absent_pass": True,
        "d1_logical_mask_sha256_metadata": audit["d1_provenance"][
            "logical_mask_sha256"
        ],
    }


def load_frozen_reliability(
    expected_sample_ids,
    reliability_path=DEFAULT_RELIABILITY_PATH,
    d2_audit_path=DEFAULT_D2_AUDIT_PATH,
    feature_provenance=None,
):
    """Load only frozen D2 U and bind its row order to canonical E0 IDs.

    The historical D2 NPZ contains T/U/corruption_mask but no sample_ids.
    E1 does not rewrite it and does not load its mask.  D2 generated rows in
    datasets.py order; E0's audited canonical IDs and shared D1 condition bind
    that order explicitly here.
    """
    reliability_path = _resolve(reliability_path)
    d2_audit_path = _resolve(d2_audit_path)
    _require(reliability_path.is_file(), "frozen D2 reliability is missing")
    _require(d2_audit_path.is_file(), "D2 utility audit is missing")
    d2_audit = _read_json(d2_audit_path)
    with np.load(reliability_path, allow_pickle=False) as archive:
        archive_fields = tuple(archive.files)
        _require("U" in archive_fields, "D2 archive has no U field")
        # Deliberately access U only. corruption_mask never enters training.
        reliability = np.ascontiguousarray(archive["U"])

    expected_sample_ids = np.asarray(expected_sample_ids)
    _require(
        expected_sample_ids.shape == (SAMPLE_NUM,)
        and expected_sample_ids.dtype == np.dtype(np.int64)
        and np.array_equal(expected_sample_ids, np.arange(SAMPLE_NUM)),
        "E0 sample IDs do not define canonical D2 row order",
    )
    _require(
        reliability.shape == (SAMPLE_NUM, VIEW_NUM),
        "frozen reliability must have shape [1400,6]",
    )
    _require(
        np.issubdtype(reliability.dtype, np.floating)
        and np.isfinite(reliability).all()
        and float(reliability.min()) >= 0.0
        and float(reliability.max()) <= 1.0,
        "frozen reliability must be finite and within [0,1]",
    )
    reliability_hash = tensor_sha256(reliability)
    _require(
        reliability_hash
        == d2_audit.get("U_sha256")
        == EXPECTED_RELIABILITY_SHA256,
        "frozen D2 reliability logical SHA mismatch",
    )
    _require(
        d2_audit.get("N") == SAMPLE_NUM
        and d2_audit.get("V") == VIEW_NUM
        and d2_audit.get("K") == CLUSTER_NUM
        and d2_audit.get("condition") == "snr2p5_k3_seed20",
        "D2 condition boundary mismatch",
    )
    d2_mask_hash = d2_audit["source_provenance"][
        "source_corruption_mask_sha256"
    ]
    if feature_provenance is not None:
        _require(
            d2_mask_hash
            == feature_provenance["d1_logical_mask_sha256_metadata"],
            "D2 and E0 do not share the same frozen D1 row condition",
        )
    reliability.setflags(write=False)
    return reliability, {
        "path": _display(reliability_path),
        "file_sha256": file_sha256(reliability_path),
        "archive_fields": list(archive_fields),
        "loaded_fields": ["U"],
        "corruption_mask_loaded": False,
        "T_loaded": False,
        "R_shape": list(reliability.shape),
        "R_dtype": str(reliability.dtype),
        "R_logical_sha256": reliability_hash,
        "R_frozen_no_grad": True,
        "sample_ids_source": "E0 audited canonical row IDs; D2 row order",
        "sample_ids_sha256": ndarray_sha256(expected_sample_ids),
        "sample_ids_exact_coverage_pass": True,
        "shared_d1_condition_logical_mask_sha256": d2_mask_hash,
    }


def build_model_from_frozen_d1(
    checkpoint_dir, device, training_seed=DEFAULT_TRAINING_SEED
):
    checkpoint_dir = _resolve(checkpoint_dir)
    _require(checkpoint_dir.is_dir(), "frozen D1 checkpoint directory missing")
    config = get_default_config(DATASET)
    _require(
        float(config["training"]["lambda1"]) == LAMBDA1,
        "native Caltech lambda1 is no longer 0.01",
    )
    model = MvCAN(
        config=config,
        view_num=VIEW_NUM,
        view_size=list(VIEW_DIMS),
        n_clusters=CLUSTER_NUM,
        seed=validate_training_seed(training_seed),
        data_size=SAMPLE_NUM,
        semantic_config=None,
    )
    checkpoint_paths = []
    checkpoint_hashes = []
    for view_id, autoencoder in enumerate(model.autoencoders):
        path = checkpoint_dir / (DATASET + str(view_id + 1) + "V.pth")
        _require(path.is_file(), "missing D1 view checkpoint: " + str(path))
        autoencoder.load_state_dict(load_state_dict_cpu(path), strict=True)
        checkpoint_paths.append(_display(path))
        checkpoint_hashes.append(file_sha256(path))
    model.to_device(device)
    for autoencoder in model.autoencoders:
        autoencoder.train()
    return model, config, {
        "directory": _display(checkpoint_dir),
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_file_sha256": checkpoint_hashes,
        "initial_backbone_hash": hash_backbone(model.autoencoders),
    }


def build_fresh_optimizers(model):
    optimizers = [
        torch.optim.Adam(
            itertools.chain(autoencoder.parameters()), lr=LEARNING_RATE
        )
        for autoencoder in model.autoencoders
    ]
    _require(
        all(len(optimizer.state) == 0 for optimizer in optimizers),
        "continuation optimizers must begin with empty Adam state",
    )
    return optimizers


def native_mvcan_losses(x_views, reconstructions, q_views, p_all, matches):
    """Exact native per-view REC + 0.01*CLU objective; P_all is unchanged."""
    if not (
        len(x_views)
        == len(reconstructions)
        == len(q_views)
        == int(matches.shape[0])
    ):
        raise ValueError("native MVCAN view count mismatch")
    losses = []
    diagnostics = []
    for view_id in range(len(x_views)):
        # P_all: [B,K], native global target. M: [V,K,K], no grad.
        p_local = p_all @ matches[view_id].detach()
        reconstruction_loss = F.mse_loss(
            reconstructions[view_id], x_views[view_id]
        )
        clustering_loss = F.mse_loss(q_views[view_id], p_local)
        loss = reconstruction_loss + LAMBDA1 * clustering_loss
        losses.append(loss)
        diagnostics.append((reconstruction_loss, clustering_loss))
    return losses, diagnostics


@torch.no_grad()
def refresh_native_target(
    model,
    full_views,
    view_weights,
    device,
    training_seed=DEFAULT_TRAINING_SEED,
):
    """MVCAN-native fusion, Hungarian M, new_P, and target_distribution."""
    kmeans = KMeans(
        n_clusters=CLUSTER_NUM,
        n_init=KMEANS_N_INIT,
        random_state=validate_training_seed(training_seed),
    )
    y_views = None
    latent_fusion = None
    for _ in range(TARGET_WEIGHT_UPDATES):
        latent_views = []
        y_views = []
        for view_id in range(VIEW_NUM):
            x_view = full_views[view_id].to(device)
            latent = model.autoencoders[view_id].encoder(x_view)
            q_local = model.autoencoders[view_id].clustering(latent)
            # z: [N,D]. This is exactly MVCAN's per-refresh MinMax transform.
            scaled = MinMaxScaler().fit_transform(
                latent.detach().cpu().numpy()
            )
            latent_views.append(scaled * view_weights[view_id])
            y_views.append(q_local.detach().cpu().numpy().argmax(1))
        latent_fusion = np.hstack(latent_views)
        y_pred = kmeans.fit_predict(latent_fusion)
        for view_id in range(VIEW_NUM):
            nmi = round(
                normalized_mutual_info_score(y_pred, y_views[view_id]), 5
            )
            view_weights[view_id] = float(np.exp(nmi))

    match_arrays = []
    for view_id in range(VIEW_NUM):
        _, _, _, match = model.Match(y_views[view_id], y_pred)
        match_arrays.append(match)
    # M: [V,K,K], constant/no grad.
    matches = torch.from_numpy(np.stack(match_arrays)).float().to(device)
    matches.requires_grad_(False)
    # P_all keeps MVCAN's exact new_P -> target_distribution definition.
    p_all_numpy = model.target_distribution(
        model.new_P(latent_fusion, kmeans.cluster_centers_)
    )
    p_all = torch.from_numpy(p_all_numpy).float()
    return p_all, matches, np.asarray(y_pred, dtype=np.int64), view_weights


def _coverage_audit(visited_ids):
    ids = np.concatenate(visited_ids).astype(np.int64, copy=False)
    expected = np.arange(SAMPLE_NUM, dtype=np.int64)
    return {
        "sample_count": int(ids.size),
        "unique_sample_count": int(np.unique(ids).size),
        "all_1400_sample_ids_exactly_once_pass": bool(
            ids.size == SAMPLE_NUM
            and np.array_equal(np.sort(ids), expected)
            and np.unique(ids).size == SAMPLE_NUM
        ),
        "sample_order_sha256": tensor_sha256(ids),
    }


def _pair_gradient_audit(pair_loss, q_local, model):
    targets = [
        q_local,
        model.autoencoders[0]._cluster_layer,
        next(model.autoencoders[0]._encoder.parameters()),
    ]
    gradients = torch.autograd.grad(
        pair_loss, targets, retain_graph=True, allow_unused=True
    )
    names = ("q_local", "cluster_layer_view0", "encoder_view0")
    result = {}
    for name, gradient in zip(names, gradients):
        finite = gradient is not None and bool(torch.isfinite(gradient).all())
        nonzero = finite and bool(torch.count_nonzero(gradient).item() > 0)
        result[name + "_gradient_finite_pass"] = bool(finite)
        result[name + "_gradient_nonzero_pass"] = bool(nonzero)
    result["pair_branch_gradient_exists_pass"] = bool(all(result.values()))
    return result


def frozen_reliability_tensor(reliability):
    """Copy frozen NumPy R into independent writable Tensor storage."""
    writable_reliability = np.array(reliability, copy=True, order="C")
    tensor = torch.from_numpy(writable_reliability)
    tensor.requires_grad_(False)
    return tensor


def build_sample_order_generator(
    training_seed=DEFAULT_TRAINING_SEED,
):
    generator = torch.Generator()
    generator.manual_seed(validate_training_seed(training_seed))
    return generator


def train_continuation(
    arm,
    epochs,
    batch_size,
    model,
    optimizers,
    views,
    sample_ids,
    reliability,
    device,
    training_seed=DEFAULT_TRAINING_SEED,
):
    if arm not in ARMS:
        raise ValueError("unknown E1 arm")
    if int(epochs) not in (2, 100):
        raise ValueError("E1 permits only 2 or 100 continuation epochs")
    if int(batch_size) != 256:
        raise ValueError("E1 batch size is frozen at native Caltech value 256")

    training_seed = validate_training_seed(training_seed)
    generator = build_sample_order_generator(training_seed)
    view_weights = [1.0] * VIEW_NUM
    full_views = [torch.from_numpy(view) for view in views]
    id_tensor = torch.from_numpy(np.asarray(sample_ids, dtype=np.int64))
    active_reliability = reliability
    shuffle_audit = None
    if arm == "SHUFFLED_R_LWC":
        active_reliability, _, shuffle_audit = whole_row_reliability_shuffle(
            reliability, seed=training_seed
        )
    active_reliability_tensor = frozen_reliability_tensor(active_reliability)

    epoch_records = []
    native_target_refresh_count = 0
    pair_gradient_audit = None
    graph_stopgrad_pass = True
    reliability_frozen_pass = True
    base_native_identity_pass = True
    loss_finite_pass = True
    p_all = None
    matches = None
    for epoch in range(int(epochs)):
        if epoch % TARGET_REFRESH_INTERVAL == 0:
            p_all, matches, _, view_weights = refresh_native_target(
                model,
                full_views,
                view_weights,
                device,
                training_seed=training_seed,
            )
            native_target_refresh_count += 1
        _require(p_all is not None and matches is not None, "native target missing")
        dataset = torch.utils.data.TensorDataset(
            *full_views, p_all, id_tensor
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=True,
            drop_last=False,
            generator=generator,
            num_workers=0,
        )
        visited = []
        native_sum = 0.0
        pair_sum = 0.0
        total_sum = 0.0
        batch_count = 0
        for packed in loader:
            x_views = [packed[view_id].to(device) for view_id in range(VIEW_NUM)]
            p_batch = packed[VIEW_NUM].to(device)
            batch_ids = packed[VIEW_NUM + 1]
            visited.append(batch_ids.numpy().astype(np.int64, copy=False))
            # R_batch: [B,6], frozen/no grad, indexed only by sample IDs.
            reliability_batch = active_reliability_tensor[batch_ids].to(
                device=device, dtype=x_views[0].dtype
            ).detach()
            reliability_frozen_pass = bool(
                reliability_frozen_pass
                and not reliability_batch.requires_grad
                and reliability_batch.grad_fn is None
            )

            for optimizer in optimizers:
                optimizer.zero_grad()
            reconstructions = []
            z_views = []
            q_views = []
            for view_id in range(VIEW_NUM):
                x_hat, z, q = model.autoencoders[view_id](x_views[view_id])
                # z: [B,D_v_latent=10], q_local view: [B,K=7].
                reconstructions.append(x_hat)
                z_views.append(z)
                q_views.append(q)
            # q_local: [B,6,7].
            q_local = torch.stack(q_views, dim=1)
            native_losses, _ = native_mvcan_losses(
                x_views, reconstructions, q_views, p_batch, matches
            )
            native_total = torch.stack(native_losses).sum()
            pair_loss = None
            if arm == "BASE":
                total_loss = native_total
                base_native_identity_pass = bool(
                    base_native_identity_pass
                    and torch.equal(total_loss.detach(), native_total.detach())
                )
            else:
                # q_aligned/h_sem: [B,6,7]. q_aligned retains q gradients.
                q_aligned, h_sem = align_semantic_probabilities(
                    q_local, matches
                )
                _require(q_aligned.requires_grad, "q_aligned lost q gradient")
                # G_inter: [6,6,B,B], built only from h_sem.detach().
                graph_inter = robust_inter_affinity(h_sem.detach())
                graph_stopgrad_pass = bool(
                    graph_stopgrad_pass
                    and not graph_inter.requires_grad
                    and graph_inter.grad_fn is None
                )
                pair_loss, _ = pairwise_semantic_cooperation_loss(
                    h_sem,
                    graph_inter,
                    reliability_batch,
                    arm,
                )
                if pair_gradient_audit is None:
                    pair_gradient_audit = _pair_gradient_audit(
                        pair_loss, q_local, model
                    )
                # Reuse native lambda1=0.01; no additional coefficient exists.
                total_loss = native_total + LAMBDA1 * pair_loss

            finite = bool(
                torch.isfinite(native_total).item()
                and torch.isfinite(total_loss).item()
                and (pair_loss is None or torch.isfinite(pair_loss).item())
            )
            loss_finite_pass = bool(loss_finite_pass and finite)
            _require(finite, "non-finite E1 loss")
            total_loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            native_sum += float(native_total.detach().item())
            pair_sum += (
                0.0 if pair_loss is None else float(pair_loss.detach().item())
            )
            total_sum += float(total_loss.detach().item())
            batch_count += 1

        coverage = _coverage_audit(visited)
        _require(
            coverage["all_1400_sample_ids_exactly_once_pass"],
            "epoch did not cover every sample ID exactly once",
        )
        epoch_records.append({
            "epoch": epoch + 1,
            "batch_count": batch_count,
            "native_loss_mean": native_sum / batch_count,
            "pair_loss_mean": pair_sum / batch_count,
            "total_loss_mean": total_sum / batch_count,
            **coverage,
        })

    # Final predictor is generated without labels, then returned for persistence.
    _, final_matches, final_predictions, _ = refresh_native_target(
        model,
        full_views,
        view_weights,
        device,
        training_seed=training_seed,
    )
    return final_predictions, {
        "epoch_records": epoch_records,
        "native_target_refresh_count": native_target_refresh_count,
        "P_all_definition": "MVCAN target_distribution(new_P(latent_fusion, centers))",
        "P_all_rewritten": False,
        "final_M_shape": list(final_matches.shape),
        "M_no_grad_pass": bool(
            not final_matches.requires_grad and final_matches.grad_fn is None
        ),
        "base_native_loss_identity_pass": (
            bool(base_native_identity_pass) if arm == "BASE" else None
        ),
        "pair_gradient_audit": pair_gradient_audit,
        "graph_stopgrad_pass": bool(graph_stopgrad_pass),
        "reliability_frozen_no_grad_pass": bool(reliability_frozen_pass),
        "loss_finite_pass": bool(loss_finite_pass),
        "all_epochs_exact_sample_coverage_pass": bool(
            all(
                record["all_1400_sample_ids_exactly_once_pass"]
                for record in epoch_records
            )
        ),
        "sample_order_sha256_per_epoch": [
            record["sample_order_sha256"] for record in epoch_records
        ],
        "shuffle_audit": shuffle_audit,
    }


def save_final_models(model, output_dir):
    model_dir = output_dir / "models"
    model_dir.mkdir()
    paths = []
    hashes = []
    for view_id, autoencoder in enumerate(model.autoencoders):
        path = model_dir / (DATASET + str(view_id + 1) + "V.pth")
        torch.save(autoencoder.state_dict(), path)
        paths.append(_display(path))
        hashes.append(file_sha256(path))
    return {"paths": paths, "file_sha256": hashes}


def save_and_hash_predictions(predictions, output_dir):
    values = np.ascontiguousarray(predictions, dtype=np.int64)
    _require(values.shape == (SAMPLE_NUM,), "final predictions shape mismatch")
    path = output_dir / "final_predictions.npy"
    np.save(path, values, allow_pickle=False)
    reloaded = np.load(path, allow_pickle=False)
    _require(np.array_equal(reloaded, values), "saved prediction verification failed")
    return path, {
        "path": _display(path),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "logical_sha256": tensor_sha256(values),
        "file_sha256": file_sha256(path),
        "saved_and_reloaded_before_labels_pass": True,
    }


def load_labels_after_predictions(label_path, prediction_path):
    """The sole label boundary; called only after prediction persistence."""
    _require(Path(prediction_path).is_file(), "predictions must exist first")
    mat = sio.loadmat(_resolve(label_path), variable_names=["Y"])
    _require("Y" in mat, "Caltech label key Y is missing")
    labels = np.squeeze(np.asarray(mat["Y"])).astype(np.int64, copy=False)
    _require(labels.shape == (SAMPLE_NUM,), "Caltech label shape mismatch")
    labels = labels.copy()
    labels[labels == 95] = 5
    _require(
        np.array_equal(np.unique(labels), np.arange(CLUSTER_NUM)),
        "Caltech label class boundary mismatch",
    )
    return labels


def evaluate_predictions(labels, predictions):
    return {
        "ACC": float(clustering_accuracy(labels, predictions)),
        "NMI": float(normalized_mutual_info_score(labels, predictions)),
        "ARI": float(adjusted_rand_score(labels, predictions)),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="E1 controlled multi-seed continuation"
    )
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--epochs", required=True, type=int, choices=(2, 100))
    parser.add_argument(
        "--seed",
        type=int,
        choices=TRAINING_SEED_CHOICES,
        default=DEFAULT_TRAINING_SEED,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--reliability-path", default=str(DEFAULT_RELIABILITY_PATH))
    parser.add_argument("--d2-audit-path", default=str(DEFAULT_D2_AUDIT_PATH))
    parser.add_argument("--feature-path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument(
        "--feature-audit-path", default=str(DEFAULT_FEATURE_AUDIT_PATH)
    )
    parser.add_argument("--glgc-repository", default=str(DEFAULT_GLGC_REPOSITORY))
    parser.add_argument("--label-path", default=str(DEFAULT_LABEL_PATH))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    training_seed = validate_training_seed(args.seed)
    output_dir = (
        default_output_dir(args.arm, args.epochs, training_seed)
        if args.output_dir is None
        else _resolve(args.output_dir)
    )
    _require(not output_dir.exists(), "refusing to overwrite E1 output")
    device = torch.device(args.device)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "requested CUDA device unavailable")
        torch.cuda.set_device(device)

    set_global_seed(training_seed)
    glgc_provenance = verify_glgc_released_logic(args.glgc_repository)
    views, sample_ids, feature_provenance = load_frozen_feature_artifact(
        args.feature_path, args.feature_audit_path
    )
    reliability, reliability_provenance = load_frozen_reliability(
        sample_ids,
        args.reliability_path,
        args.d2_audit_path,
        feature_provenance,
    )
    model, config, checkpoint_provenance = build_model_from_frozen_d1(
        args.checkpoint_dir, device, training_seed=training_seed
    )
    d2_audit = _read_json(_resolve(args.d2_audit_path))
    _require(
        checkpoint_provenance["initial_backbone_hash"]
        == d2_audit["source_provenance"]["backbone_hash_before"],
        "initial model hash does not equal frozen D1/D2 provenance",
    )
    optimizers = build_fresh_optimizers(model)
    optimizer_audit = {
        "policy": "fresh independent Adam per MVCAN view from model-only D1 checkpoint",
        "optimizer_count": len(optimizers),
        "initial_state_empty_pass": bool(
            all(len(optimizer.state) == 0 for optimizer in optimizers)
        ),
        "learning_rate": LEARNING_RATE,
        "same_policy_all_arms": True,
    }

    output_dir.mkdir(parents=True)
    predictions, runtime_audit = train_continuation(
        arm=args.arm,
        epochs=args.epochs,
        batch_size=int(config["training"]["batch_size"]),
        model=model,
        optimizers=optimizers,
        views=views,
        sample_ids=sample_ids,
        reliability=reliability,
        device=device,
        training_seed=training_seed,
    )
    prediction_path, prediction_audit = save_and_hash_predictions(
        predictions, output_dir
    )
    # Ground-truth labels first enter the process after final predictions are
    # durable and content-hashed. They are used only for final ACC/NMI/ARI.
    labels = load_labels_after_predictions(args.label_path, prediction_path)
    metrics = evaluate_predictions(labels, predictions)
    model_outputs = save_final_models(model, output_dir)

    pair_arm = args.arm != "BASE"
    engineering_checks = {
        "initial_model_matches_frozen_d1_pass": True,
        "optimizer_initial_state_policy_pass": optimizer_audit[
            "initial_state_empty_pass"
        ],
        "base_native_loss_identity_pass": (
            runtime_audit["base_native_loss_identity_pass"]
            if args.arm == "BASE"
            else True
        ),
        "loss_finite_pass": runtime_audit["loss_finite_pass"],
        "all_1400_sample_ids_covered_each_epoch_pass": runtime_audit[
            "all_epochs_exact_sample_coverage_pass"
        ],
        "R_sample_alignment_pass": reliability_provenance[
            "sample_ids_exact_coverage_pass"
        ],
        "R_logical_sha_pass": reliability_provenance["R_logical_sha256"]
        == EXPECTED_RELIABILITY_SHA256,
        "no_labels_in_training_pass": True,
        "labels_loaded_after_prediction_hash_pass": prediction_audit[
            "saved_and_reloaded_before_labels_pass"
        ],
        "no_corruption_mask_loaded_pass": bool(
            not feature_provenance["corruption_mask_present"]
            and not reliability_provenance["corruption_mask_loaded"]
        ),
        "no_target_rewriting_pass": not runtime_audit["P_all_rewritten"],
        "M_no_grad_pass": runtime_audit["M_no_grad_pass"],
        "R_frozen_no_grad_pass": runtime_audit[
            "reliability_frozen_no_grad_pass"
        ],
        "graph_branch_no_grad_pass": runtime_audit["graph_stopgrad_pass"],
        "pair_branch_gradient_exists_pass": bool(
            not pair_arm
            or runtime_audit["pair_gradient_audit"][
                "pair_branch_gradient_exists_pass"
            ]
        ),
    }
    engineering_pass = bool(all(engineering_checks.values()))
    result = {
        "stage": STAGE,
        "arm": args.arm,
        "epochs": int(args.epochs),
        **training_seed_audit_fields(training_seed),
        "dataset": DATASET,
        "N": SAMPLE_NUM,
        "V": VIEW_NUM,
        "K": CLUSTER_NUM,
        "lambda1_native_and_pair": LAMBDA1,
        "pair_loss_coefficient": 0.0 if args.arm == "BASE" else LAMBDA1,
        "initial_model_hash": checkpoint_provenance["initial_backbone_hash"],
        "final_model_hash": hash_backbone(model.autoencoders),
        "checkpoint_provenance": checkpoint_provenance,
        "optimizer_audit": optimizer_audit,
        "feature_provenance": feature_provenance,
        "D2_R_provenance": reliability_provenance,
        "GLGC_released_logic_provenance": glgc_provenance,
        "runtime": runtime_audit,
        "prediction_audit": prediction_audit,
        "final_model_outputs": model_outputs,
        "label_protocol": {
            "training_label_count": 0,
            "labels_in_feature_artifact": False,
            "labels_loaded_only_after_final_predictions_saved_and_hashed": True,
            "labels_used_only_for_final_ACC_NMI_ARI": True,
        },
        "metrics": metrics,
        "engineering_checks": engineering_checks,
        "E1_ENGINEERING_PASS": engineering_pass,
    }
    _write_json(output_dir / "e1_audit.json", result)
    print("E1_ENGINEERING_PASS=" + str(engineering_pass))
    print("ARM=" + args.arm)
    print("ACC={:.10f}".format(metrics["ACC"]))
    print("NMI={:.10f}".format(metrics["NMI"]))
    print("ARI={:.10f}".format(metrics["ARI"]))
    print("Saved: " + _display(output_dir / "e1_audit.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
