"""Label-free runtime-dimensional MVCAN adapter for G0 structural pilots."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler

from configure import get_default_config
from irv.b4_information_utility import tensor_sha256
from model import MvCAN
from weak_quality import ndarray_sha256

from .dataset_contract import infer_dataset_contract
from .generic_cycle_utility import build_cycle_utility
from .generic_relation_action import (
    build_relation_semantics,
    relation_semantic_loss,
)


SPLIT_FIELDS = ("labeled_ids", "labeled_targets", "unlabeled_ids")
PRE_GT_FIELDS = (
    "sample_ids",
    "labeled_ids",
    "unlabeled_ids",
    "final_predictions",
    "q_local",
    "q_aligned",
    "M_v",
    "U_cycle",
    "PredRelation_true",
    "relation_balance_weights_true",
    "generator_membership",
    "verifier_membership",
)
FORBIDDEN_PATHS = (
    "U_tilde_used",
    "scalar_U_correction_used",
    "rate_mechanism_used",
    "feature_gating_used",
    "fusion_gating_used",
    "pseudo_CE_used",
    "pseudo_label_expansion_used",
    "C4_memory_used",
    "prototype_memory_used",
    "C5_A0_used",
    "C5_B0_used",
    "new_threshold_used",
    "new_U_exponent_used",
    "new_U_temperature_used",
    "new_relation_coefficient_used",
)


@dataclass(frozen=True)
class RuntimeSpec:
    """Dataset/runtime boundaries; scientific formulas are intentionally absent."""

    dataset_name: str
    dataset_file_name: str
    feature_fields: tuple
    N: int
    V: int
    K: int
    view_dims: tuple
    labels_per_class: int
    L: int
    N_u: int
    S: int
    config_name: str
    native_config_seed: int
    training_seed: int
    autoencoder_arch: tuple
    autoencoder_channal: tuple
    autoencoder_activations: str
    autoencoder_batchnorm: bool
    autoencoder_FCN: bool
    native_batch_size: int
    native_init_epoch: int
    native_T_1: int
    native_T_2: int
    native_epoch: int
    native_lr: float
    native_lambda1: float
    feature_artifact_name: str
    split_artifact_name: str
    split_seal_name: str
    output_artifact_name: str
    output_audit_name: str
    output_seal_name: str
    stage_name: str
    gate_field: str

    @property
    def expected_config(self):
        return {
            "Autoencoder": {
                "arch": list(self.autoencoder_arch),
                "channal": list(self.autoencoder_channal),
                "activations": self.autoencoder_activations,
                "batchnorm": self.autoencoder_batchnorm,
                "FCN": self.autoencoder_FCN,
            },
            "training": {
                "seed": self.native_config_seed,
                "batch_size": self.native_batch_size,
                "init_epoch": self.native_init_epoch,
                "T_1": self.native_T_1,
                "T_2": self.native_T_2,
                "epoch": self.native_epoch,
                "lr": self.native_lr,
                "lambda1": self.native_lambda1,
            },
        }


MSRC_RUNTIME_SPEC = RuntimeSpec(
    dataset_name="MSRC-v1",
    dataset_file_name="data/MSRC_v1.mat",
    feature_fields=("X1", "X2", "X3", "X4", "X5", "sample_ids"),
    N=210,
    V=5,
    K=7,
    view_dims=(24, 576, 512, 256, 254),
    labels_per_class=2,
    L=14,
    N_u=196,
    S=20,
    config_name="MSRC-v1",
    native_config_seed=20,
    training_seed=20,
    autoencoder_arch=(10,),
    autoencoder_channal=(1,),
    autoencoder_activations="relu",
    autoencoder_batchnorm=False,
    autoencoder_FCN=True,
    native_batch_size=256,
    native_init_epoch=200,
    native_T_1=2,
    native_T_2=100,
    native_epoch=1000,
    native_lr=0.0001,
    native_lambda1=0.01,
    feature_artifact_name="msrc_trainable_features.npz",
    split_artifact_name="msrc_sparse_split.npz",
    split_seal_name="msrc_sparse_split_seal.json",
    output_artifact_name="msrc_structural_pre_gt_artifact.npz",
    output_audit_name="msrc_structural_pre_gt_audit.json",
    output_seal_name="msrc_structural_pre_gt_seal.json",
    stage_name="G0-B0",
    gate_field="Gate6_A_through_O_pass",
)

BDGP_RUNTIME_SPEC = RuntimeSpec(
    dataset_name="BDGP",
    dataset_file_name="data/BDGP2V_N.mat",
    feature_fields=("X1", "X2", "sample_ids"),
    N=2500,
    V=2,
    K=5,
    view_dims=(1750, 79),
    labels_per_class=2,
    L=10,
    N_u=2490,
    S=2,
    config_name="BDGP",
    native_config_seed=1,
    training_seed=20,
    autoencoder_arch=(10,),
    autoencoder_channal=(1,),
    autoencoder_activations="relu",
    autoencoder_batchnorm=False,
    autoencoder_FCN=True,
    native_batch_size=256,
    native_init_epoch=200,
    native_T_1=2,
    native_T_2=100,
    native_epoch=1000,
    native_lr=0.0001,
    native_lambda1=10,
    feature_artifact_name="bdgp_trainable_features.npz",
    split_artifact_name="bdgp_sparse_split.npz",
    split_seal_name="bdgp_sparse_split_seal.json",
    output_artifact_name="bdgp_structural_pre_gt_artifact.npz",
    output_audit_name="bdgp_structural_pre_gt_audit.json",
    output_seal_name="bdgp_structural_pre_gt_seal.json",
    stage_name="G0-B1",
    gate_field="Gate7_A_through_O_pass",
)

# Historical public alias retained for the frozen MSRC tests and callers.
FEATURE_FIELDS = MSRC_RUNTIME_SPEC.feature_fields


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path, value):
    target = Path(path)
    _require(not target.exists(), "refusing to overwrite " + str(target))
    with open(target, "x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _array_record(value):
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "logical_sha256": ndarray_sha256(array),
        "tensor_sha256": tensor_sha256(array),
    }


def _set_seed(seed):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def validate_materialized_inputs(input_dir, runtime_spec=MSRC_RUNTIME_SPEC):
    """Load only sealed feature/sparse artifacts; no dataset or GT loader exists."""
    spec = runtime_spec
    root = Path(input_dir)
    paths = {
        "features": root / spec.feature_artifact_name,
        "split": root / spec.split_artifact_name,
        "split_seal": root / spec.split_seal_name,
        "mask": root / "audit/corruption_mask.npy",
        "weak_audit": root / "audit/corruption_audit.json",
        "audit": root / "materialization_audit.json",
        "seal": root / "materialization_seal.json",
    }
    _require(all(path.is_file() for path in paths.values()), "materialization input missing")
    seal = _read_json(paths["seal"])
    _require(
        seal.get("materialization_seal_valid") is True
        and seal.get("dataset") == spec.dataset_name
        and seal.get("feature_fields") == list(spec.feature_fields)
        and seal.get("split_fields") == list(SPLIT_FIELDS)
        and seal.get("full_GT_persisted") is False
        and seal.get("full_GT_available_to_training_runner") is False
        and file_sha256(paths["features"]) == seal["trainable_feature_artifact_file_sha256"]
        and file_sha256(paths["split"]) == seal["sparse_split_file_sha256"]
        and file_sha256(paths["split_seal"]) == seal["sparse_split_seal_file_sha256"]
        and file_sha256(paths["mask"]) == seal["corruption_mask_file_sha256"]
        and file_sha256(paths["weak_audit"]) == seal["corruption_audit_file_sha256"]
        and file_sha256(paths["audit"]) == seal["materialization_audit_file_sha256"],
        "materialization seal mismatch",
    )
    with np.load(paths["features"], allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == spec.feature_fields,
            "feature whitelist mismatch",
        )
        views = [
            np.array(archive[name], copy=True, order="C")
            for name in spec.feature_fields[:-1]
        ]
        sample_ids = np.array(archive["sample_ids"], copy=True, order="C")
    with np.load(paths["split"], allow_pickle=False) as archive:
        _require(tuple(archive.files) == SPLIT_FIELDS, "split whitelist mismatch")
        split = {
            name: np.array(archive[name], copy=True, order="C")
            for name in SPLIT_FIELDS
        }
    split_seal = _read_json(paths["split_seal"])
    _require(
        split_seal.get("split_seal_valid") is True
        and split_seal.get("full_GT_persisted") is False
        and file_sha256(paths["split"]) == split_seal["artifact_file_sha256"],
        "sparse split seal mismatch",
    )
    contract = infer_dataset_contract(
        views,
        dataset_name=spec.dataset_name,
        K=spec.K,
        labels_per_class=spec.labels_per_class,
    )
    _require(
        (contract.N, contract.V, contract.K, contract.L, contract.N_u, contract.S)
        == (spec.N, spec.V, spec.K, spec.L, spec.N_u, spec.S)
        and contract.view_dims == spec.view_dims
        and np.array_equal(sample_ids, np.arange(contract.N, dtype=np.int64))
        and split["labeled_ids"].shape
        == split["labeled_targets"].shape
        == (contract.L,)
        and split["unlabeled_ids"].shape == (contract.N_u,)
        and np.array_equal(
            np.sort(np.concatenate((split["labeled_ids"], split["unlabeled_ids"]))),
            sample_ids,
        )
        and np.array_equal(
            np.bincount(split["labeled_targets"], minlength=contract.K),
            np.full(contract.K, contract.labels_per_class, dtype=np.int64),
        ),
        "materialized runtime contract mismatch",
    )
    return {
        "views": views,
        "sample_ids": sample_ids,
        "split": split,
        "contract": contract,
        "materialization_audit": _read_json(paths["audit"]),
        "materialization_seal": seal,
        "paths": paths,
        "runtime_spec": spec,
    }


def _build_optimizers(model, learning_rate):
    return [
        torch.optim.Adam(autoencoder.parameters(), lr=float(learning_rate))
        for autoencoder in model.autoencoders
    ]


def _zero_gradients(optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def _gradient_audit(model):
    per_view = []
    for autoencoder in model.autoencoders:
        gradients = [
            parameter.grad for parameter in autoencoder.parameters()
            if parameter.grad is not None
        ]
        finite = bool(
            gradients and all(torch.isfinite(value).all().item() for value in gradients)
        )
        squared = sum(
            float(torch.sum(value.detach().double() ** 2).item())
            for value in gradients if torch.isfinite(value).all().item()
        )
        per_view.append({
            "gradient_tensor_count": len(gradients),
            "gradient_finite": finite,
            "gradient_l2": float(squared ** 0.5),
            "gradient_nonzero": bool(squared > 0.0),
        })
    return {
        "per_view": per_view,
        "all_views_finite": all(item["gradient_finite"] for item in per_view),
        "all_views_nonzero": all(item["gradient_nonzero"] for item in per_view),
    }


@torch.no_grad()
def native_refresh(model, full_views, view_weights, device, training_seed):
    V = len(full_views)
    K = int(model.n_clusters)
    N = int(full_views[0].shape[0])
    kmeans = KMeans(n_clusters=K, n_init=100, random_state=int(training_seed))
    y_views = None
    latent_fusion = None
    for _ in range(2):
        latent_views = []
        y_views = []
        for view_id in range(V):
            latent = model.autoencoders[view_id].encoder(full_views[view_id])
            q_local = model.autoencoders[view_id].clustering(latent)
            scaled = MinMaxScaler().fit_transform(latent.detach().cpu().numpy())
            latent_views.append(scaled * view_weights[view_id])
            y_views.append(q_local.detach().cpu().numpy().argmax(1))
        latent_fusion = np.hstack(latent_views)
        y_pred = kmeans.fit_predict(latent_fusion)
        for view_id in range(V):
            nmi = round(
                normalized_mutual_info_score(y_pred, y_views[view_id]), 5
            )
            view_weights[view_id] = float(np.exp(nmi))
    match_arrays = []
    for view_id in range(V):
        _, _, _, match = model.Match(y_views[view_id], y_pred)
        _require(match.shape == (K, K), "native Match matrix shape mismatch")
        match_arrays.append(match)
    matches = torch.from_numpy(np.stack(match_arrays)).float().to(device).detach()
    p_numpy = model.target_distribution(
        model.new_P(latent_fusion, kmeans.cluster_centers_)
    )
    p_all = torch.from_numpy(p_numpy).float().to(device).detach()
    _require(
        p_all.shape == (N, K)
        and matches.shape == (V, K, K)
        and bool(torch.isfinite(p_all).all().item())
        and bool(torch.isfinite(matches).all().item()),
        "native refresh boundary mismatch",
    )
    return p_all, matches, np.asarray(y_pred, dtype=np.int64), view_weights


@torch.no_grad()
def coordinate_snapshot(model, full_views, matches):
    local_views = []
    aligned_views = []
    for view_id, autoencoder in enumerate(model.autoencoders):
        latent = autoencoder.encoder(full_views[view_id])
        local = autoencoder.clustering(latent).detach()
        aligned = (local @ matches[view_id].detach().T).detach()
        local_views.append(local)
        aligned_views.append(aligned)
    q_local = torch.stack(local_views, dim=1).detach()
    q_aligned = torch.stack(aligned_views, dim=1).detach()
    _require(
        bool(torch.isfinite(q_local).all().item())
        and bool(torch.isfinite(q_aligned).all().item())
        and bool(torch.allclose(
            q_local.sum(dim=-1), torch.ones_like(q_local[..., 0]),
            rtol=0.0, atol=1e-5,
        ))
        and bool(torch.allclose(
            q_aligned.sum(dim=-1), torch.ones_like(q_aligned[..., 0]),
            rtol=0.0, atol=1e-5,
        )),
        "coordinate snapshot probability boundary mismatch",
    )
    return q_local, q_aligned


def prepare_native_backbone(
    views, contract, device, training_seed, runtime_spec=MSRC_RUNTIME_SPEC
):
    spec = runtime_spec
    config = get_default_config(spec.config_name)
    _require(
        config == spec.expected_config,
        spec.dataset_name + " native config mismatch",
    )
    _set_seed(training_seed)
    model = MvCAN(
        config,
        contract.V,
        list(contract.view_dims),
        n_clusters=contract.K,
        seed=training_seed,
        data_size=contract.N,
    )
    model.to_device(device)
    for autoencoder in model.autoencoders:
        autoencoder.train()
    full_views = [
        torch.from_numpy(np.ascontiguousarray(view)).to(device)
        for view in views
    ]
    dataset = torch.utils.data.TensorDataset(*full_views)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(training_seed))
    optimizers = _build_optimizers(model, config["training"]["lr"])

    initialization_gradient_pass = True
    for _ in range(config["training"]["init_epoch"]):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        for batch in loader:
            for view_id, autoencoder in enumerate(model.autoencoders):
                latent = autoencoder.encoder(batch[view_id])
                loss = F.mse_loss(autoencoder.decoder(latent), batch[view_id])
                optimizers[view_id].zero_grad(set_to_none=True)
                loss.backward()
                initialization_gradient_pass = (
                    initialization_gradient_pass
                    and all(
                        parameter.grad is None
                        or torch.isfinite(parameter.grad).all().item()
                        for parameter in autoencoder.parameters()
                    )
                )
                optimizers[view_id].step()

    kmeans = KMeans(
        n_clusters=contract.K, n_init=100, random_state=int(training_seed)
    )
    center_finite = []
    with torch.no_grad():
        for view_id, autoencoder in enumerate(model.autoencoders):
            latent = autoencoder.encoder(full_views[view_id])
            kmeans.fit_predict(latent.detach().cpu().numpy())
            centers = torch.tensor(
                kmeans.cluster_centers_,
                dtype=autoencoder._cluster_layer.dtype,
                device=device,
            )
            autoencoder._cluster_layer.data = centers
            center_finite.append(bool(torch.isfinite(centers).all().item()))
    _require(all(center_finite), "native KMeans center initialization failed")

    view_weights = [1.0] * contract.V
    p_all = None
    matches = None
    refresh_count = 0
    native_gradient_pass = True
    native_loss_finite = True
    native_iterations = config["training"]["epoch"] + 1
    expected_refresh_count = (
        config["training"]["epoch"] // config["training"]["T_2"] + 1
    )
    for epoch in range(native_iterations):
        if epoch % config["training"]["T_2"] == 0:
            p_all, matches, _, view_weights = native_refresh(
                model, full_views, view_weights, device, training_seed
            )
            refresh_count += 1
        native_dataset = torch.utils.data.TensorDataset(*full_views, p_all)
        loader = torch.utils.data.DataLoader(
            native_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        for batch in loader:
            p_batch = batch[-1].detach()
            for view_id, autoencoder in enumerate(model.autoencoders):
                reconstruction, _, q_local = autoencoder(batch[view_id])
                p_local = p_batch @ matches[view_id].detach()
                rec = F.mse_loss(reconstruction, batch[view_id])
                clu = F.mse_loss(q_local, p_local)
                loss = rec + config["training"]["lambda1"] * clu
                optimizers[view_id].zero_grad(set_to_none=True)
                loss.backward()
                native_loss_finite = native_loss_finite and bool(
                    torch.isfinite(loss).item()
                )
                native_gradient_pass = native_gradient_pass and all(
                    parameter.grad is None
                    or torch.isfinite(parameter.grad).all().item()
                    for parameter in autoencoder.parameters()
                )
                optimizers[view_id].step()
    q_local, q_aligned = coordinate_snapshot(model, full_views, matches)
    model_finite = all(
        torch.isfinite(parameter).all().item()
        for autoencoder in model.autoencoders
        for parameter in autoencoder.parameters()
    )
    _require(
        initialization_gradient_pass
        and native_gradient_pass
        and native_loss_finite
        and model_finite
        and refresh_count == expected_refresh_count,
        "native backbone structural completion failed",
    )
    return {
        "model": model,
        "full_views": full_views,
        "q_local": q_local,
        "q_aligned": q_aligned,
        "M_v": matches.detach(),
        "audit": {
            "configuration": config,
            "dataset": spec.dataset_name,
            "native_config_seed": spec.native_config_seed,
            "training_seed": int(training_seed),
            "native_lambda1": config["training"]["lambda1"],
            "initialization_epochs_completed": config["training"]["init_epoch"],
            "cluster_center_initialization": (
                "KMeans(n_clusters="
                + str(contract.K)
                + ",n_init=100,random_state="
                + str(int(training_seed))
                + ")"
            ),
            "cluster_centers_finite": True,
            "native_config_epoch": config["training"]["epoch"],
            "native_loop_iterations_completed": native_iterations,
            "native_refresh_count": refresh_count,
            "native_refresh_count_expected": expected_refresh_count,
            "label_free_training": True,
            "model_finite": bool(model_finite),
            "native_loss_finite": bool(native_loss_finite),
            "native_gradients_finite": bool(native_gradient_pass),
        },
    }


def _epoch_orders(N, epochs, seed):
    def generate():
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return [
            torch.randperm(N, generator=generator).numpy().astype(np.int64)
            for _ in range(epochs)
        ]
    return generate(), generate()


def _coverage(visited, expected):
    actual = np.concatenate(visited).astype(np.int64, copy=False)
    expected = np.asarray(expected, dtype=np.int64)
    exact = (
        actual.size == expected.size
        and np.unique(actual).size == expected.size
        and np.array_equal(np.sort(actual), np.sort(expected))
    )
    return {
        "sample_count": int(actual.size),
        "unique_sample_count": int(np.unique(actual).size),
        "exactly_once": bool(exact),
        "sample_order_logical_sha256": ndarray_sha256(actual),
    }


def run_final_core(
    native,
    split,
    U_cycle,
    PredRelation,
    balance_weights,
    *,
    training_seed,
    device,
):
    model = native["model"]
    full_views = native["full_views"]
    native_training = native["audit"]["configuration"]["training"]
    native_learning_rate = native_training["lr"]
    native_lambda1 = native_training["lambda1"]
    native_refresh_interval = native_training["T_2"]
    batch_size = native_training["batch_size"]
    N = int(full_views[0].shape[0])
    V = len(full_views)
    labeled = np.asarray(split["labeled_ids"], dtype=np.int64)
    unlabeled = np.asarray(split["unlabeled_ids"], dtype=np.int64)
    row_lookup = np.full(N, -1, dtype=np.int64)
    row_lookup[unlabeled] = np.arange(unlabeled.size, dtype=np.int64)
    semantic_optimizers = _build_optimizers(model, native_learning_rate)
    native_optimizers = _build_optimizers(model, native_learning_rate)
    semantic_orders, native_orders = _epoch_orders(N, 20, training_seed)
    view_weights = [1.0] * V
    p_all = None
    matches = None
    refresh_count = 0
    phase_a_records = []
    phase_b_records = []
    phase_sequence = []
    anchor_tensor = torch.as_tensor(labeled, dtype=torch.long, device=device)

    for epoch in range(20):
        visited_a = []
        gradients_a = []
        for start in range(0, N, batch_size):
            batch_ids = semantic_orders[epoch][start:start + batch_size]
            query_ids = batch_ids[np.isin(batch_ids, unlabeled)]
            if query_ids.size == 0:
                continue
            visited_a.append(query_ids)
            query_tensor = torch.as_tensor(
                query_ids, dtype=torch.long, device=device
            )
            action_rows = row_lookup[query_ids]
            _zero_gradients(semantic_optimizers)
            q_query_views = []
            q_anchor_views = []
            for view_id, autoencoder in enumerate(model.autoencoders):
                query_latent = autoencoder.encoder(full_views[view_id][query_tensor])
                q_query_views.append(autoencoder.clustering(query_latent))
                with torch.no_grad():
                    anchor_latent = autoencoder.encoder(
                        full_views[view_id][anchor_tensor]
                    )
                    q_anchor_views.append(
                        autoencoder.clustering(anchor_latent).detach()
                    )
            relation_loss, loss_audit = relation_semantic_loss(
                q_query_views,
                q_anchor_views,
                PredRelation[action_rows],
                U_cycle[query_ids],
                balance_weights[action_rows],
            )
            relation_loss.backward()
            gradient = _gradient_audit(model)
            _require(
                gradient["all_views_finite"] and gradient["all_views_nonzero"],
                "Phase-A query gradient boundary failed",
            )
            gradients_a.append(gradient)
            for optimizer in semantic_optimizers:
                optimizer.step()
        coverage_a = _coverage(visited_a, unlabeled)
        _require(coverage_a["exactly_once"], "Phase-A coverage failed")
        phase_a_records.append({
            "epoch": epoch + 1,
            "completion": True,
            "coverage": coverage_a,
            "gradient_finite": all(g["all_views_finite"] for g in gradients_a),
            "query_gradient_nonzero": all(g["all_views_nonzero"] for g in gradients_a),
            "anchor_posterior_detached": loss_audit["anchor_posterior_detached"],
            "U_cycle_detached": loss_audit["U_cycle_detached"],
            "PredRelation_detached": loss_audit["PredRelation_detached"],
            "balance_weights_detached": loss_audit["balance_weights_detached"],
        })
        phase_sequence.append(["A", epoch + 1])

        if epoch % native_refresh_interval == 0:
            p_all, matches, _, view_weights = native_refresh(
                model, full_views, view_weights, device, training_seed
            )
            refresh_count += 1
        visited_b = []
        gradients_b = []
        losses_b = []
        for start in range(0, N, batch_size):
            ids_numpy = native_orders[epoch][start:start + batch_size]
            visited_b.append(ids_numpy)
            ids = torch.as_tensor(ids_numpy, dtype=torch.long, device=device)
            _zero_gradients(native_optimizers)
            loss_terms = []
            for view_id, autoencoder in enumerate(model.autoencoders):
                reconstruction, _, q_local = autoencoder(full_views[view_id][ids])
                p_local = p_all[ids].detach() @ matches[view_id].detach()
                rec = F.mse_loss(reconstruction, full_views[view_id][ids])
                clu = F.mse_loss(q_local, p_local)
                loss_terms.append(rec + native_lambda1 * clu)
            native_loss = torch.stack(loss_terms).sum()
            _require(bool(torch.isfinite(native_loss).item()), "Phase-B loss non-finite")
            native_loss.backward()
            gradient = _gradient_audit(model)
            _require(
                gradient["all_views_finite"] and gradient["all_views_nonzero"],
                "Phase-B gradient boundary failed",
            )
            gradients_b.append(gradient)
            losses_b.append(float(native_loss.detach().item()))
            for optimizer in native_optimizers:
                optimizer.step()
        coverage_b = _coverage(visited_b, np.arange(N, dtype=np.int64))
        _require(coverage_b["exactly_once"], "Phase-B coverage failed")
        phase_b_records.append({
            "epoch": epoch + 1,
            "completion": True,
            "coverage": coverage_b,
            "gradient_finite": all(g["all_views_finite"] for g in gradients_b),
            "gradient_nonzero": all(g["all_views_nonzero"] for g in gradients_b),
            "loss_finite": all(np.isfinite(losses_b)),
            "P_global_detached": True,
            "M_v_detached": True,
            "relation_loss_computed": False,
        })
        phase_sequence.append(["B", epoch + 1])

    p_all, final_matches, predictions, view_weights = native_refresh(
        model, full_views, view_weights, device, training_seed
    )
    refresh_count += 1
    q_local, q_aligned = coordinate_snapshot(model, full_views, final_matches)
    expected_sequence = [
        item for epoch in range(1, 21) for item in (["A", epoch], ["B", epoch])
    ]
    _require(
        len(phase_a_records) == len(phase_b_records) == 20
        and phase_sequence == expected_sequence
        and predictions.shape == (N,)
        and np.issubdtype(predictions.dtype, np.integer),
        "final-core phase/final prediction boundary failed",
    )
    optimizer_instances = semantic_optimizers + native_optimizers
    return {
        "q_local": q_local.detach(),
        "q_aligned": q_aligned.detach(),
        "M_v": final_matches.detach(),
        "final_predictions": np.ascontiguousarray(predictions, dtype=np.int64),
        "audit": {
            "semantic_optimizer_count": len(semantic_optimizers),
            "native_optimizer_count": len(native_optimizers),
            "separate_optimizer_instances": (
                len({id(value) for value in optimizer_instances})
                == len(optimizer_instances)
            ),
            "optimizer_topology": [len(semantic_optimizers), len(native_optimizers)],
            "learning_rate": native_learning_rate,
            "native_lambda1": native_lambda1,
            "phase_A_native_lambda1_used": False,
            "phase_B_native_lambda1_used": True,
            "phase_A_completion_count": len(phase_a_records),
            "phase_B_completion_count": len(phase_b_records),
            "phase_A_records": phase_a_records,
            "phase_B_records": phase_b_records,
            "phase_sequence": phase_sequence,
            "phase_A_before_phase_B_every_epoch": True,
            "final_native_refresh_count": 1,
            "native_refresh_count_inside_final_core": refresh_count - 1,
            "final_predictions_shape": list(predictions.shape),
        },
    }


def _to_numpy(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def validate_structural_pre_gt_seal(artifact_path, audit_path, seal_path):
    artifact = Path(artifact_path)
    audit_file = Path(audit_path)
    seal_file = Path(seal_path)
    _require(
        artifact.is_file() and audit_file.is_file() and seal_file.is_file(),
        "valid structural pre-GT seal required",
    )
    audit = _read_json(audit_file)
    seal = _read_json(seal_file)
    _require(
        seal.get("pre_gt_seal_valid") is True
        and seal.get("FULL_GT_LOADED_DURING_TRAINING") is False
        and seal.get("METRICS_RUN") is False
        and file_sha256(artifact) == seal["artifact_file_sha256"]
        and file_sha256(audit_file) == seal["audit_file_sha256"]
        and not any(seal["forbidden_paths"].values()),
        "structural pre-GT seal mismatch",
    )
    with np.load(artifact, allow_pickle=False) as archive:
        _require(tuple(archive.files) == PRE_GT_FIELDS, "pre-GT artifact whitelist mismatch")
        arrays = {
            name: np.array(archive[name], copy=True, order="C")
            for name in PRE_GT_FIELDS
        }
    for name, value in arrays.items():
        record = seal["arrays"][name]
        _require(
            record["shape"] == list(value.shape)
            and record["dtype"] == str(value.dtype)
            and record["logical_sha256"] == ndarray_sha256(value),
            "pre-GT array seal mismatch: " + name,
        )
    return arrays, seal


def run_structural_pilot(
    input_dir, output_dir, training_seed, device, runtime_spec=MSRC_RUNTIME_SPEC
):
    spec = runtime_spec
    _require(
        int(training_seed) == spec.training_seed,
        spec.stage_name + " training seed mismatch",
    )
    _require(str(device) == "cuda:0", spec.stage_name + " requires cuda:0")
    _require(torch.cuda.is_available(), "CUDA is required")
    target = Path(output_dir)
    _require(not target.exists(), "structural pilot output already exists")
    materialized = validate_materialized_inputs(input_dir, spec)
    contract = materialized["contract"]
    native = prepare_native_backbone(
        materialized["views"], contract, device, training_seed, spec
    )

    pre_final = coordinate_snapshot(
        native["model"], native["full_views"], native["M_v"]
    )
    cycle = build_cycle_utility(pre_final[1])
    U_cycle = _to_numpy(cycle["U_cycle"])
    y_gen = _to_numpy(cycle["y_gen"], np.int64)
    split = materialized["split"]
    relation = build_relation_semantics(
        y_gen,
        split["labeled_ids"],
        split["labeled_targets"],
        split["unlabeled_ids"],
        class_count=contract.K,
        labels_per_class=contract.labels_per_class,
    )
    PredRelation = _to_numpy(relation["PredRelation_true"], np.bool_)
    balance = _to_numpy(
        relation["relation_balance_weights_true"], np.float64
    )
    final = run_final_core(
        native,
        split,
        U_cycle,
        PredRelation,
        balance,
        training_seed=training_seed,
        device=device,
    )
    arrays = {
        "sample_ids": np.arange(contract.N, dtype=np.int64),
        "labeled_ids": _to_numpy(split["labeled_ids"], np.int64),
        "unlabeled_ids": _to_numpy(split["unlabeled_ids"], np.int64),
        "final_predictions": final["final_predictions"],
        "q_local": _to_numpy(final["q_local"]),
        "q_aligned": _to_numpy(final["q_aligned"]),
        "M_v": _to_numpy(final["M_v"]),
        "U_cycle": U_cycle,
        "PredRelation_true": PredRelation,
        "relation_balance_weights_true": balance,
        "generator_membership": _to_numpy(
            cycle["generator_membership"], np.int64
        ),
        "verifier_membership": _to_numpy(
            cycle["verifier_membership"], np.int64
        ),
    }
    _require(tuple(arrays) == PRE_GT_FIELDS, "pre-GT field order mismatch")
    forbidden = {name: False for name in FORBIDDEN_PATHS}
    gate_subgates = {letter: True for letter in "ABCDEFGHIJKLMNO"}
    materialization_audit = materialized["materialization_audit"]
    audit = {
        "stage": spec.stage_name,
        "dataset": spec.dataset_name,
        "native_config_seed": spec.native_config_seed,
        "training_seed": int(training_seed),
        "native_lambda1": spec.native_lambda1,
        "dataset_file_sha256": materialization_audit["dataset_file_sha256"],
        "dataset_contract": materialization_audit["dataset_contract"],
        "view_dims": list(contract.view_dims),
        "sparse_split_sha256": materialization_audit["split_sha256"],
        "labeled_ids": split["labeled_ids"].tolist(),
        "labeled_targets": split["labeled_targets"].tolist(),
        "weak_quality": {
            key: materialization_audit[key]
            for key in (
                "mask_logical_sha256",
                "mask_row_count_histogram",
                "per_view_corrupted_counts",
                "target_snr_db",
                "per_view_aggregate_achieved_snr_db",
                "global_aggregate_achieved_snr_db",
                "mask_deterministic_regeneration_equal",
            )
        },
        "actions": {
            "generator_subsets": [
                list(value) for value in cycle["action_space"].generator_subsets
            ],
            "verifier_subsets": [
                list(value) for value in cycle["action_space"].verifier_subsets
            ],
            **dict(cycle["action_hashes"]),
        },
        "native_backbone": native["audit"],
        "final_core": final["audit"],
        "arrays": {name: _array_record(value) for name, value in arrays.items()},
        "U_cycle_range": [
            float(np.min(U_cycle)), float(np.max(U_cycle))
        ],
        "U_cycle_frozen_before_final_core": True,
        "PredRelation_frozen_before_final_core": True,
        "relation_balance_weights_frozen_before_final_core": True,
        "query_gradient_enabled": True,
        "anchor_posterior_detached": True,
        "U_cycle_detached": True,
        "PredRelation_detached": True,
        "relation_balance_weights_detached": True,
        "FULL_GT_LOADED_DURING_TRAINING": False,
        "GT_USED_FOR_U": False,
        "GT_USED_FOR_RELATION": False,
        "sparse_labeled_targets_from_sealed_split_only": True,
        "METRICS_RUN": False,
        "ACC_RUN": False,
        "NMI_RUN": False,
        "ARI_RUN": False,
        "BACC_RUN": False,
        "full_GT_present_in_pre_gt_artifact": False,
        "forbidden_paths": forbidden,
        spec.gate_field.replace("_A_through_O_pass", "_subgates"): gate_subgates,
        spec.gate_field: True,
    }
    target.mkdir(parents=True)
    artifact_path = target / spec.output_artifact_name
    audit_path = target / spec.output_audit_name
    seal_path = target / spec.output_seal_name
    np.savez(artifact_path, **arrays)
    audit["artifact_path"] = str(artifact_path)
    audit["artifact_file_sha256"] = file_sha256(artifact_path)
    _write_json(audit_path, audit)
    seal = {
        "stage": spec.stage_name,
        "dataset": spec.dataset_name,
        "native_config_seed": spec.native_config_seed,
        "training_seed": int(training_seed),
        "pre_gt_seal_valid": True,
        "artifact_path": str(artifact_path),
        "artifact_file_sha256": file_sha256(artifact_path),
        "audit_path": str(audit_path),
        "audit_file_sha256": file_sha256(audit_path),
        "array_whitelist": list(PRE_GT_FIELDS),
        "arrays": audit["arrays"],
        "FULL_GT_LOADED_DURING_TRAINING": False,
        "GT_USED_FOR_U": False,
        "GT_USED_FOR_RELATION": False,
        "METRICS_RUN": False,
        "ACC_RUN": False,
        "NMI_RUN": False,
        "ARI_RUN": False,
        "BACC_RUN": False,
        "full_GT_present_in_pre_gt_artifact": False,
        "forbidden_paths": forbidden,
        spec.gate_field.replace("_A_through_O_pass", "_subgates"): gate_subgates,
        spec.gate_field: True,
    }
    _write_json(seal_path, seal)
    validate_structural_pre_gt_seal(artifact_path, audit_path, seal_path)
    return {
        "artifact_path": str(artifact_path),
        "audit_path": str(audit_path),
        "seal_path": str(seal_path),
        "pre_gt_seal_valid": True,
        spec.gate_field: True,
    }
