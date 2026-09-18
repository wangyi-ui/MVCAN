"""R1-A2 exact parity against frozen legacy code on real frozen inputs."""

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

import datasets
import weak_quality as legacy_weak_quality
from configure import get_default_config
from experiments.generic_contract import generic_weak_quality
from model import Autoencoder as LegacyAutoencoder
from model import MvCAN
from release_core.backbone.clustering import (
    native_refresh_from_latents,
    target_distribution,
)
from release_core.backbone.multiview import MultiViewBackbone
from release_core.backbone.native_objective import native_objective
from release_core.config.native import get_native_config
from release_core.data import weak_quality as clean_weak_quality
from release_core.data.loaders import load_bdgp, load_caltech, load_msrc_v1
from release_core.data.sample_ids import canonical_sample_ids


ROOT = Path(__file__).resolve().parents[2]
BATCH_SIZE = 256
SNR_DB = 2.5
CORRUPTION_SEED = 20
REFRESH_SEED = 20

SPECS = {
    "Caltech-6V": {
        "path": ROOT / "data" / "Caltech.mat",
        "loader": load_caltech,
        "n": 1400,
        "v": 6,
        "k": 7,
        "dims": (48, 40, 254, 1984, 512, 928),
        "mask_hash": "e6250535db6e33b9263102cf335fcf9e8552a24cae4b70ec5a94887f4e58bf0b",
    },
    "MSRC-v1": {
        "path": ROOT / "data" / "MSRC_v1.mat",
        "loader": load_msrc_v1,
        "n": 210,
        "v": 5,
        "k": 7,
        "dims": (24, 576, 512, 256, 254),
        "mask_hash": "d64f5fbf95518e5a77e44a1fe18a15dee9924b8c2a341e930f86f20ce5ac9e9d",
    },
    "BDGP": {
        "path": ROOT / "data" / "BDGP2V_N.mat",
        "loader": load_bdgp,
        "n": 2500,
        "v": 2,
        "k": 5,
        "dims": (1750, 79),
        "mask_hash": "a3c882c29f2f5064d552e60bdc1a2d585764317e7ba78aa055cd1c3c22e5de07",
    },
}


def logical_ndarray_sha256(array):
    """Hash dtype string, shape tuple, and C-order raw bytes."""
    contiguous = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_real_data():
    result = {}
    for name, spec in SPECS.items():
        legacy_views, legacy_labels = datasets.load_data({"dataset": name})
        clean_views, clean_labels = spec["loader"](spec["path"])
        result[name] = {
            "legacy_views": legacy_views,
            "clean_views": clean_views,
            "legacy_labels": legacy_labels[0],
            "clean_labels": clean_labels[0],
        }
    return result


def _corrupt_real_data(real_data):
    result = {}
    for name, spec in SPECS.items():
        legacy_input = real_data[name]["legacy_views"]
        clean_input = real_data[name]["clean_views"]
        if spec["v"] % 2 == 0:
            legacy_views, legacy_audit = (
                legacy_weak_quality.apply_heterogeneous_gaussian_corruption(
                    legacy_input,
                    spec["v"] // 2,
                    SNR_DB,
                    CORRUPTION_SEED,
                )
            )
            legacy_path = "weak_quality.apply_heterogeneous_gaussian_corruption"
        else:
            legacy_views, legacy_audit = (
                generic_weak_quality.apply_half_gaussian_corruption(
                    legacy_input, SNR_DB, CORRUPTION_SEED
                )
            )
            legacy_path = (
                "experiments.generic_contract.generic_weak_quality."
                "apply_half_gaussian_corruption"
            )
        clean_views, clean_audit = clean_weak_quality.apply_half_gaussian_corruption(
            clean_input, SNR_DB, CORRUPTION_SEED
        )
        result[name] = {
            "legacy_views": legacy_views,
            "clean_views": clean_views,
            "legacy_audit": legacy_audit,
            "clean_audit": clean_audit,
            "legacy_path": legacy_path,
        }
    return result


def _legacy_refresh(latents, posteriors, weights, clusters, seed):
    """Frozen two-pass refresh, packaged for underlying-array comparison."""
    weights = np.asarray(weights, dtype=np.float64).copy()
    estimator = KMeans(n_clusters=clusters, n_init=100, random_state=seed)
    local_assignments = None
    latent_fusion = None
    predictions = None
    for _ in range(2):
        fused = []
        local_assignments = []
        for index in range(len(latents)):
            fused.append(MinMaxScaler().fit_transform(latents[index]) * weights[index])
            local_assignments.append(posteriors[index].argmax(1))
        latent_fusion = np.hstack(fused)
        predictions = estimator.fit_predict(latent_fusion)
        for index in range(len(latents)):
            internal_nmi = round(
                normalized_mutual_info_score(
                    predictions, local_assignments[index]
                ),
                5,
            )
            weights[index] = float(np.exp(internal_nmi))
    matrices = np.stack(
        [MvCAN.Match(None, value, predictions)[3] for value in local_assignments]
    )
    p_all = MvCAN.target_distribution(
        None, MvCAN.new_P(None, latent_fusion, estimator.cluster_centers_)
    )
    return (
        p_all,
        matrices,
        predictions.astype(np.int64),
        weights,
        estimator.cluster_centers_,
        np.stack(local_assignments),
    )


@pytest.fixture(scope="session", autouse=True)
def deterministic_cpu_policy():
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
    torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


@pytest.fixture(scope="session")
def real_data():
    return _load_real_data()


@pytest.fixture(scope="session")
def corrupted_data(real_data):
    # Labels are intentionally not present in this feature-only fixture.
    return _corrupt_real_data(real_data)


@pytest.fixture(scope="session")
def forward_results(corrupted_data):
    results = {}
    for name, spec in SPECS.items():
        config = get_native_config(name)
        seed = config["training"]["seed"]
        container = MultiViewBackbone(
            config, spec["v"], spec["dims"], n_clusters=spec["k"], seed=seed
        ).to_device(torch.device("cpu"))
        latents = []
        posteriors = []
        records = []
        models = []
        for view_index, values in enumerate(corrupted_data[name]["clean_views"]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            legacy = LegacyAutoencoder(
                [spec["dims"][view_index], 500, 500, 2000, 10],
                config["Autoencoder"]["activations"],
                config["Autoencoder"]["batchnorm"],
                spec["k"],
                config["Autoencoder"]["FCN"],
                config["Autoencoder"]["channal"],
            ).cpu()
            clean = container.autoencoders[view_index]
            legacy.eval()
            clean.eval()
            legacy_state = legacy.state_dict()
            clean_state = clean.state_dict()
            assert tuple(legacy_state) == tuple(clean_state)
            parameter_equal = True
            for key in legacy_state:
                parameter_equal = parameter_equal and (
                    legacy_state[key].shape == clean_state[key].shape
                    and legacy_state[key].dtype == clean_state[key].dtype
                    and torch.equal(legacy_state[key], clean_state[key])
                )
            parameter_count_equal = sum(
                value.numel() for value in legacy.parameters()
            ) == sum(value.numel() for value in clean.parameters())

            legacy_parts = [[], [], []]
            clean_parts = [[], [], []]
            contiguous_values = np.ascontiguousarray(values, dtype=np.float32)
            with torch.no_grad():
                for start in range(0, spec["n"], BATCH_SIZE):
                    batch = torch.from_numpy(
                        contiguous_values[start : start + BATCH_SIZE]
                    )
                    legacy_outputs = legacy(batch)
                    clean_outputs = container.forward_view(view_index, batch)
                    for output_index, (legacy_value, clean_value) in enumerate(
                        zip(legacy_outputs, clean_outputs)
                    ):
                        assert legacy_value.device.type == "cpu"
                        assert clean_value.device.type == "cpu"
                        assert legacy_value.dtype == clean_value.dtype == torch.float32
                        assert torch.equal(legacy_value, clean_value)
                        legacy_parts[output_index].append(legacy_value)
                        clean_parts[output_index].append(clean_value)
                legacy_outputs = [torch.cat(parts, dim=0) for parts in legacy_parts]
                clean_outputs = [torch.cat(parts, dim=0) for parts in clean_parts]
                legacy_p = MvCAN.target_distribution(None, legacy_outputs[2])
                clean_p = target_distribution(clean_outputs[2])
                assert torch.equal(legacy_p, clean_p)
                legacy_rec = F.mse_loss(legacy_outputs[0], torch.from_numpy(contiguous_values))
                legacy_clu = F.mse_loss(legacy_outputs[2], legacy_p)
                legacy_total = legacy_rec + config["training"]["lambda1"] * legacy_clu
                clean_total, clean_rec, clean_clu = native_objective(
                    clean_outputs[0],
                    torch.from_numpy(contiguous_values),
                    clean_outputs[2],
                    clean_p,
                    config["training"]["lambda1"],
                )
            assert torch.equal(legacy_rec, clean_rec)
            assert torch.equal(legacy_clu, clean_clu)
            assert torch.equal(legacy_total, clean_total)
            assert torch.isfinite(clean_p).all()
            latents.append(clean_outputs[1].numpy())
            posteriors.append(clean_outputs[2].numpy())
            records.append(
                {
                    "input_shape": tuple(contiguous_values.shape),
                    "reconstruction_shape": tuple(clean_outputs[0].shape),
                    "z_shape": tuple(clean_outputs[1].shape),
                    "q_shape": tuple(clean_outputs[2].shape),
                    "parameter_equal": parameter_equal,
                    "parameter_count_equal": parameter_count_equal,
                    "reconstruction_equal": torch.equal(
                        legacy_outputs[0], clean_outputs[0]
                    ),
                    "z_equal": torch.equal(legacy_outputs[1], clean_outputs[1]),
                    "q_equal": torch.equal(legacy_outputs[2], clean_outputs[2]),
                    "target_equal": torch.equal(legacy_p, clean_p),
                    "objective_equal": (
                        torch.equal(legacy_rec, clean_rec)
                        and torch.equal(legacy_clu, clean_clu)
                        and torch.equal(legacy_total, clean_total)
                    ),
                }
            )
            models.append(clean)
        results[name] = {
            "latents": latents,
            "posteriors": posteriors,
            "records": records,
            "model_ids": [id(model) for model in models],
            "cluster_ids": [id(model._cluster_layer) for model in models],
        }
    return results


@pytest.fixture(scope="session")
def refresh_results(forward_results):
    result = {}
    for name, spec in SPECS.items():
        latents = forward_results[name]["latents"]
        posteriors = forward_results[name]["posteriors"]
        weights = [1.0] * spec["v"]
        # KMeans reductions must use one CPU thread for bitwise repeatability
        # across the two independently executed old/new calls.
        with threadpool_limits(limits=1):
            legacy = _legacy_refresh(
                latents, posteriors, weights, spec["k"], REFRESH_SEED
            )
            clean = native_refresh_from_latents(
                latents, posteriors, weights, spec["k"], REFRESH_SEED
            )
        result[name] = {
            "legacy": legacy,
            "clean": clean,
            "clean_local_assignments": np.stack(
                [posterior.argmax(1) for posterior in posteriors]
            ),
        }
    return result


@pytest.mark.parametrize("name", list(SPECS))
def test_real_loader_exact(name, real_data):
    spec = SPECS[name]
    legacy_views = real_data[name]["legacy_views"]
    clean_views = real_data[name]["clean_views"]
    assert len(legacy_views) == len(clean_views) == spec["v"]
    assert tuple(view.shape[1] for view in clean_views) == spec["dims"]
    for legacy, clean in zip(legacy_views, clean_views):
        assert legacy.shape == clean.shape
        assert legacy.shape[0] == spec["n"]
        assert legacy.dtype == clean.dtype == np.float32
        assert np.array_equal(legacy, clean)
    assert np.array_equal(
        real_data[name]["legacy_labels"], real_data[name]["clean_labels"]
    )
    assert np.unique(real_data[name]["clean_labels"]).size == spec["k"]
    if name == "Caltech-6V":
        assert 95 not in real_data[name]["clean_labels"]
        assert 5 in real_data[name]["clean_labels"]
    if name in ("MSRC-v1", "BDGP"):
        assert all(view.flags.c_contiguous for view in clean_views)


@pytest.mark.parametrize("name", list(SPECS))
def test_dataset_native_configuration_exact(name):
    assert get_default_config(name) == get_native_config(name)


@pytest.mark.parametrize("name", list(SPECS))
def test_real_sample_id_and_row_order_exact(name, real_data, corrupted_data):
    spec = SPECS[name]
    sample_ids = canonical_sample_ids(spec["n"])
    assert sample_ids.dtype == np.int64
    assert np.array_equal(sample_ids, np.arange(spec["n"], dtype=np.int64))
    mask = corrupted_data[name]["clean_audit"]["mask"]
    assert mask.shape[0] == sample_ids.shape[0]
    for legacy, clean in zip(
        real_data[name]["legacy_views"], real_data[name]["clean_views"]
    ):
        assert np.array_equal(legacy[sample_ids], clean[sample_ids])


@pytest.mark.parametrize("name", list(SPECS))
def test_real_weak_quality_exact(name, real_data, corrupted_data):
    spec = SPECS[name]
    record = corrupted_data[name]
    legacy_audit = record["legacy_audit"]
    clean_audit = record["clean_audit"]
    legacy_mask = legacy_audit["mask"]
    clean_mask = clean_audit["mask"]
    assert np.array_equal(legacy_mask, clean_mask)
    assert clean_weak_quality.ndarray_sha256(clean_mask) == spec["mask_hash"]
    assert legacy_audit["mask_sha256"] == clean_audit["mask_sha256"]
    for key in (
        "corrupted_pair_count",
        "noise_seed",
        "per_view_aggregate_achieved_snr_db",
        "global_aggregate_achieved_snr_db",
        "snr_target_pass",
        "unchanged_clean_pairs_max_abs_error",
    ):
        assert legacy_audit[key] == clean_audit[key]
    for view_index, (legacy, clean, original) in enumerate(
        zip(
            record["legacy_views"],
            record["clean_views"],
            real_data[name]["clean_views"],
        )
    ):
        assert legacy.shape == clean.shape == original.shape
        assert legacy.dtype == clean.dtype == original.dtype
        assert np.array_equal(legacy, clean)
        unchanged = ~clean_mask[:, view_index]
        assert np.array_equal(clean[unchanged], original[unchanged])
    row_counts = clean_mask.sum(1)
    view_counts = clean_mask.sum(0)
    if name == "Caltech-6V":
        assert record["legacy_path"] == (
            "weak_quality.apply_heterogeneous_gaussian_corruption"
        )
        assert np.all(row_counts == 3)
        frozen_mask = ROOT / "outputs" / "d1_caltech6v" / "snr2p5_k3_seed20" / "audit" / "corruption_mask.npy"
        assert _file_sha256(frozen_mask) == (
            "d549ab7a8b4f7740b30e28d23638c86c1aaf9b1ce77fb6698f6e708fd2a59a50"
        )
        assert np.array_equal(clean_mask, np.load(frozen_mask, allow_pickle=False))
    elif name == "MSRC-v1":
        assert record["legacy_path"] == (
            "experiments.generic_contract.generic_weak_quality."
            "apply_half_gaussian_corruption"
        )
        assert np.count_nonzero(row_counts == 2) == 105
        assert np.count_nonzero(row_counts == 3) == 105
        assert int(clean_mask.sum()) == 525
        assert view_counts.tolist() == [105, 105, 105, 105, 105]
    else:
        assert np.all(row_counts == 1)
        assert view_counts.tolist() == [1250, 1250]


@pytest.mark.parametrize("name", list(SPECS))
def test_real_feature_logical_hashes_exact(name, real_data, corrupted_data):
    for legacy, clean in zip(
        real_data[name]["legacy_views"], real_data[name]["clean_views"]
    ):
        assert logical_ndarray_sha256(legacy) == logical_ndarray_sha256(clean)
    for legacy, clean in zip(
        corrupted_data[name]["legacy_views"],
        corrupted_data[name]["clean_views"],
    ):
        assert logical_ndarray_sha256(legacy) == logical_ndarray_sha256(clean)


@pytest.mark.parametrize("name", list(SPECS))
def test_real_autoencoder_parameters_exact(name, forward_results):
    assert all(
        record["parameter_equal"] and record["parameter_count_equal"]
        for record in forward_results[name]["records"]
    )


@pytest.mark.parametrize("name", list(SPECS))
def test_real_reconstruction_and_latent_exact(name, forward_results):
    spec = SPECS[name]
    for view_index, record in enumerate(forward_results[name]["records"]):
        assert record["input_shape"] == (spec["n"], spec["dims"][view_index])
        assert record["reconstruction_shape"] == record["input_shape"]
        assert record["z_shape"] == (spec["n"], 10)
        assert record["reconstruction_equal"]
        assert record["z_equal"]


@pytest.mark.parametrize("name", list(SPECS))
def test_real_q_local_exact(name, forward_results):
    spec = SPECS[name]
    for record in forward_results[name]["records"]:
        assert record["q_shape"] == (spec["n"], spec["k"])
        assert record["q_equal"]


@pytest.mark.parametrize("name", list(SPECS))
def test_real_multiview_container_exact(name, forward_results):
    record = forward_results[name]
    assert len(set(record["model_ids"])) == SPECS[name]["v"]
    assert len(set(record["cluster_ids"])) == SPECS[name]["v"]
    assert all(item["reconstruction_equal"] for item in record["records"])
    assert all(item["z_equal"] for item in record["records"])
    assert all(item["q_equal"] for item in record["records"])


@pytest.mark.parametrize("name", list(SPECS))
def test_real_native_objective_exact(name, forward_results):
    expected_lambda = {"Caltech-6V": 0.01, "MSRC-v1": 0.01, "BDGP": 10}[name]
    assert get_native_config(name)["training"]["lambda1"] == expected_lambda
    assert all(item["objective_equal"] for item in forward_results[name]["records"])


@pytest.mark.parametrize("name", list(SPECS))
def test_real_target_distribution_exact(name, forward_results):
    assert all(item["target_equal"] for item in forward_results[name]["records"])


@pytest.mark.parametrize("name", list(SPECS))
def test_real_native_refresh_exact(name, refresh_results):
    record = refresh_results[name]
    legacy = record["legacy"]
    clean = record["clean"]
    for legacy_value, clean_value in zip(legacy[:5], clean):
        assert np.array_equal(legacy_value, clean_value)
    assert np.array_equal(legacy[5], record["clean_local_assignments"])


def test_gt_metric_training_and_r2_r5_isolation():
    # The data-contract loader is the only helper that names labels. Every
    # numerical stage after it accepts feature-only records or tensors.
    assert "labels" not in inspect.getsource(_corrupt_real_data)
    assert tuple(inspect.signature(native_refresh_from_latents).parameters) == (
        "latent_views",
        "q_local_views",
        "view_weights",
        "n_clusters",
        "random_state",
    )
    assert tuple(inspect.signature(native_objective).parameters) == (
        "reconstruction",
        "inputs",
        "q_local",
        "p_local",
        "lambda1",
    )
    assert not {
        "U_cycle",
        "PredRelation",
        "relation_balance_weights",
        "pseudo_label",
        "semantic_memory",
    }.intersection(globals())
    assert BATCH_SIZE == 256
    assert REFRESH_SEED == 20
