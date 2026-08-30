"""Read-only protocol primitives for G2 utility-aligned semantic consensus."""

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score

from ClusteringTest import acc as clustering_accuracy
from experiments.b7_sparse_supervision import b7_sparse_anchor_protocol as b7_protocol
from irv.b4_information_utility import tensor_sha256
from weak_quality import ndarray_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STAGE = "G2-A0"
DATASET_NAME = "Caltech-6V"
SAMPLE_NUM = 1400
VIEW_NUM = 6
CLASS_NUM = 7
LATENT_DIM = 10
FUSION_DIM = VIEW_NUM * LATENT_DIM
TOP_K = 3
MODEL_SEED = 20
CORRUPTION_SEED = 20
CORRUPTION_K = 3
SNR_DB = 2.5
FUSION_UPDATES = 2
KMEANS_N_INIT = 100
CONTROL_SEED = 20
PROBABILITY_ATOL = 1e-6
EXPECTED_VIEW_DIMS = [48, 40, 254, 1984, 512, 928]
EXPECTED_U_SHA256 = b7_protocol.EXPECTED_U_SHA256
EXPECTED_UTILITY_FILE_SHA256 = b7_protocol.EXPECTED_UTILITY_FILE_SHA256
DEFAULT_DATA_PATH = REPOSITORY_ROOT / "data/Caltech.mat"
DEFAULT_BACKBONE_DIR = (
    REPOSITORY_ROOT / "outputs/d1_caltech6v/snr2p5_k3_seed20/models"
)
DEFAULT_D2_DIR = b7_protocol.DEFAULT_D2_DIR
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/g2_utility_semantic_consensus"
CONSENSUS_POLICIES = ("ALL", "U_TOP3", "SHUFFLED_U", "ORACLE")
RESULT_ARMS = CONSENSUS_POLICIES + ("NATIVE_MVCAN_GLOBAL",)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def resolve_path(path):
    value = Path(path)
    return value if value.is_absolute() else REPOSITORY_ROOT / value


def display_path(path):
    value = resolve_path(path).resolve()
    try:
        return str(value.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(value)


def read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def json_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_numpy(value, dtype=None):
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    return np.ascontiguousarray(array)


def load_caltech_views_only(data_path=DEFAULT_DATA_PATH):
    """Load only X1..X6; the ground-truth MATLAB variable is not requested."""
    source = resolve_path(data_path)
    _require(source.is_file(), "Caltech data file is missing")
    feature_names = ["X" + str(view_id + 1) for view_id in range(VIEW_NUM)]
    matrix = sio.loadmat(str(source), variable_names=feature_names)
    _require(
        all(name in matrix for name in feature_names),
        "Caltech data file is missing one or more view variables",
    )
    views = []
    for view_id, name in enumerate(feature_names):
        view = np.asarray(matrix[name], dtype=np.float32)
        if view_id == 2:
            view = preprocessing.MinMaxScaler().fit_transform(view)
        view = np.ascontiguousarray(view, dtype=np.float32)
        _require(
            view.shape == (SAMPLE_NUM, EXPECTED_VIEW_DIMS[view_id])
            and np.isfinite(view).all(),
            "Caltech view boundary mismatch",
        )
        views.append(view)
    return views


def load_caltech_labels_only(data_path=DEFAULT_DATA_PATH):
    """Load labels through the explicit post-prediction evaluation boundary."""
    source = resolve_path(data_path)
    _require(source.is_file(), "Caltech data file is missing")
    matrix = sio.loadmat(str(source), variable_names=["Y"])
    _require("Y" in matrix, "Caltech data file is missing Y")
    labels = np.squeeze(np.asarray(matrix["Y"])).astype(np.int64, copy=True)
    labels[labels == 95] = 5
    _require(
        labels.shape == (SAMPLE_NUM,)
        and np.array_equal(np.unique(labels), np.arange(CLASS_NUM)),
        "Caltech label boundary mismatch",
    )
    return np.ascontiguousarray(labels, dtype=np.int64)


def load_frozen_u_only(d2_dir=DEFAULT_D2_DIR):
    """Reuse the frozen D2 loader; no T, U, or Ridge computation is exposed."""
    return b7_protocol.load_frozen_u_only(d2_dir)


def load_oracle_corruption_only(d2_dir=DEFAULT_D2_DIR):
    """Isolated oracle-only artifact loader."""
    return b7_protocol.load_oracle_corruption_mask(d2_dir)


def extract_native_z_q(autoencoders, evaluation_views):
    """Reuse B7-A1's raw-z-to-q frozen extraction boundary."""
    return b7_protocol.extract_native_z_q(autoencoders, evaluation_views)


def freeze_autoencoders(autoencoders):
    for autoencoder in autoencoders:
        autoencoder.eval()
        for parameter in autoencoder.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None


def frozen_parameter_audit(autoencoders):
    parameters = [
        parameter
        for autoencoder in autoencoders
        for parameter in autoencoder.parameters()
    ]
    return {
        "parameter_count": len(parameters),
        "all_requires_grad_false": bool(
            parameters and all(not parameter.requires_grad for parameter in parameters)
        ),
        "all_grad_none": bool(
            parameters and all(parameter.grad is None for parameter in parameters)
        ),
    }


def validate_native_representation(raw_z_stack, q_stack, require_full=True):
    raw_z = _as_numpy(raw_z_stack)
    q = _as_numpy(q_stack)
    expected_n = SAMPLE_NUM if require_full else raw_z.shape[0]
    _require(
        raw_z.shape == (expected_n, VIEW_NUM, LATENT_DIM),
        "raw z must have shape [N,6,10]",
    )
    _require(
        q.shape == (expected_n, VIEW_NUM, CLASS_NUM),
        "q must have shape [N,6,7]",
    )
    _require(
        np.isfinite(raw_z).all()
        and np.isfinite(q).all()
        and np.allclose(
            q.sum(axis=2),
            1.0,
            rtol=0.0,
            atol=PROBABILITY_ATOL,
        ),
        "native representation finite/probability boundary mismatch",
    )
    return raw_z, q


def minmax_scale_raw_z(raw_z_stack):
    raw_z = _as_numpy(raw_z_stack)
    _require(
        raw_z.ndim == 3
        and raw_z.shape[1:] == (VIEW_NUM, LATENT_DIM)
        and np.isfinite(raw_z).all(),
        "raw z scaling boundary mismatch",
    )
    scaled_views = [
        preprocessing.MinMaxScaler().fit_transform(raw_z[:, view_id, :])
        for view_id in range(VIEW_NUM)
    ]
    scaled = np.stack(scaled_views, axis=1)
    _require(
        scaled.shape == raw_z.shape and np.isfinite(scaled).all(),
        "scaled z boundary mismatch",
    )
    return np.ascontiguousarray(scaled)


def _make_global_kmeans():
    return KMeans(
        n_clusters=CLASS_NUM,
        n_init=KMEANS_N_INIT,
        random_state=MODEL_SEED,
    )


def replay_native_global_reference(raw_z_stack, q_stack):
    """Replay the full-N MVCAN global fusion exactly once without labels."""
    raw_z, q = validate_native_representation(
        raw_z_stack,
        q_stack,
        require_full=True,
    )
    scaled_z = minmax_scale_raw_z(raw_z)
    view_assignments = np.ascontiguousarray(np.argmax(q, axis=2), dtype=np.int64)
    expected_cluster_ids = np.arange(CLASS_NUM, dtype=np.int64)
    for view_id in range(VIEW_NUM):
        _require(
            np.array_equal(np.unique(view_assignments[:, view_id]), expected_cluster_ids),
            "checkpoint-native q.argmax is missing a cluster in view "
            + str(view_id),
        )

    native_weights = np.ones(VIEW_NUM, dtype=np.float64)
    estimator = _make_global_kmeans()
    update_audit = []
    fused = None
    global_prediction = None
    global_centers = None
    fusion_weights_used = None
    for update_id in range(FUSION_UPDATES):
        fusion_weights_used = native_weights.copy()
        fused = np.hstack([
            scaled_z[:, view_id, :] * fusion_weights_used[view_id]
            for view_id in range(VIEW_NUM)
        ])
        global_prediction = np.asarray(
            estimator.fit_predict(fused), dtype=np.int64
        )
        global_centers = np.ascontiguousarray(estimator.cluster_centers_)
        native_weights = np.asarray([
            np.exp(
                np.round(
                    normalized_mutual_info_score(
                        global_prediction,
                        view_assignments[:, view_id],
                    ),
                    5,
                )
            )
            for view_id in range(VIEW_NUM)
        ], dtype=np.float64)
        update_audit.append({
            "update_id": int(update_id),
            "fusion_weights_used": fusion_weights_used.tolist(),
            "latent_fusion_sha256": tensor_sha256(fused),
            "global_prediction_sha256": tensor_sha256(global_prediction),
            "global_centers_sha256": tensor_sha256(global_centers),
            "native_nmi_weights_after": native_weights.tolist(),
        })

    _require(
        fused.shape == (SAMPLE_NUM, FUSION_DIM)
        and global_prediction.shape == (SAMPLE_NUM,)
        and global_centers.shape == (CLASS_NUM, FUSION_DIM)
        and native_weights.shape == (VIEW_NUM,)
        and np.isfinite(fused).all()
        and np.isfinite(global_centers).all()
        and np.isfinite(native_weights).all()
        and np.all(native_weights > 0.0),
        "native global reference boundary mismatch",
    )
    _require(
        np.array_equal(np.unique(global_prediction), expected_cluster_ids),
        "native global prediction is missing a cluster",
    )
    return {
        "scaled_z_stack": scaled_z,
        "view_assignments": view_assignments,
        "latent_fusion": np.ascontiguousarray(fused),
        "global_prediction": np.ascontiguousarray(global_prediction),
        "global_centers": global_centers,
        "fusion_weights_used_for_final_reference": np.ascontiguousarray(
            fusion_weights_used, dtype=np.float64
        ),
        "native_nmi_weights": np.ascontiguousarray(
            native_weights, dtype=np.float64
        ),
        "fusion_update_audit": update_audit,
    }


def validate_permutation_matrix(matrix, class_num=CLASS_NUM):
    values = np.asarray(matrix)
    class_num = int(class_num)
    valid = bool(
        values.shape == (class_num, class_num)
        and np.isfinite(values).all()
        and np.all((values == 0) | (values == 1))
        and np.array_equal(values.sum(axis=0), np.ones(class_num, dtype=np.int64))
        and np.array_equal(values.sum(axis=1), np.ones(class_num, dtype=np.int64))
    )
    _require(valid, "alignment matrix is not a complete permutation matrix")
    return True


def align_q_to_global(q_stack, alignment_matrix):
    q = _as_numpy(q_stack)
    matrices = _as_numpy(alignment_matrix)
    _require(
        q.ndim == 3
        and q.shape[1:] == (VIEW_NUM, CLASS_NUM)
        and matrices.shape == (VIEW_NUM, CLASS_NUM, CLASS_NUM),
        "q/alignment shape mismatch",
    )
    aligned = np.empty_like(q)
    for view_id in range(VIEW_NUM):
        validate_permutation_matrix(matrices[view_id])
        # M rows are global cluster IDs and columns are view-local IDs.
        aligned[:, view_id, :] = q[:, view_id, :] @ matrices[view_id].T
    _require(
        np.isfinite(aligned).all()
        and np.allclose(
            aligned.sum(axis=2),
            q.sum(axis=2),
            rtol=0.0,
            atol=PROBABILITY_ATOL,
        ),
        "q alignment did not preserve probability mass",
    )
    return np.ascontiguousarray(aligned)


def build_alignment(q_stack, global_prediction, match_function):
    """Map all view-local q columns into the fixed native-global coordinate."""
    q = _as_numpy(q_stack)
    global_ids = _as_numpy(global_prediction, dtype=np.int64)
    _require(
        q.ndim == 3
        and q.shape[1:] == (VIEW_NUM, CLASS_NUM)
        and global_ids.shape == (q.shape[0],),
        "alignment input boundary mismatch",
    )
    expected = np.arange(CLASS_NUM, dtype=np.int64)
    _require(
        np.array_equal(np.unique(global_ids), expected),
        "global reference is missing a cluster",
    )
    matrices = []
    view_assignments = np.argmax(q, axis=2).astype(np.int64, copy=False)
    for view_id in range(VIEW_NUM):
        local_ids = view_assignments[:, view_id]
        _require(
            np.array_equal(np.unique(local_ids), expected),
            "q.argmax is missing a cluster in view " + str(view_id),
        )
        _, _, _, matrix = match_function(local_ids, global_ids)
        matrix = np.ascontiguousarray(matrix, dtype=np.int64)
        validate_permutation_matrix(matrix)
        matrices.append(matrix)
    alignment_matrix = np.stack(matrices, axis=0)
    aligned_q = align_q_to_global(q, alignment_matrix)
    return {
        "alignment_matrix": alignment_matrix,
        "aligned_q": aligned_q,
        "view_assignments": np.ascontiguousarray(view_assignments),
        "alignment_sha256": tensor_sha256(alignment_matrix),
        "aligned_q_sha256": tensor_sha256(aligned_q),
        "q_mass_max_abs_error": float(
            np.max(np.abs(q.sum(axis=2) - 1.0))
        ),
        "aligned_q_mass_max_abs_error": float(
            np.max(np.abs(aligned_q.sum(axis=2) - 1.0))
        ),
    }


def build_normal_admissions(utility, control_seed=CONTROL_SEED):
    """Build ALL/U_TOP3/SHUFFLED_U without an oracle-data argument."""
    values = np.asarray(utility, dtype=np.float64)
    _require(
        values.ndim == 2
        and values.shape[0] > 1
        and values.shape[1] == VIEW_NUM
        and np.isfinite(values).all(),
        "frozen Utility must have shape [N,6]",
    )
    u_top3 = b7_protocol.frozen_top3_mask(values)
    permutation = b7_protocol.deterministic_permutation(
        values.shape[0], control_seed
    )
    shuffled_u = np.ascontiguousarray(u_top3[permutation], dtype=bool)
    all_admission = np.ones_like(u_top3, dtype=bool)
    changed_row_count = int(np.any(u_top3 != shuffled_u, axis=1).sum())
    _require(
        np.all(u_top3.sum(axis=1) == TOP_K)
        and np.all(shuffled_u.sum(axis=1) == TOP_K)
        and np.array_equal(u_top3.sum(axis=0), shuffled_u.sum(axis=0)),
        "SHUFFLED_U did not preserve exact admission counts",
    )
    _require(changed_row_count > 0, "SHUFFLED_U admission matrix is unchanged")
    return {
        "admissions": {
            "ALL": all_admission,
            "U_TOP3": u_top3,
            "SHUFFLED_U": shuffled_u,
        },
        "shuffled_u_row_permutation": permutation,
        "changed_row_count": changed_row_count,
    }


def build_oracle_admission(corruption_mask):
    values = np.asarray(corruption_mask, dtype=bool)
    _require(
        values.shape == (SAMPLE_NUM, VIEW_NUM)
        and np.all(values.sum(axis=1) == CORRUPTION_K),
        "oracle corruption boundary mismatch",
    )
    oracle = np.ascontiguousarray(~values, dtype=bool)
    _require(
        np.all(oracle.sum(axis=1) == TOP_K),
        "ORACLE must admit exactly three clean views per sample",
    )
    return oracle


def consensus_diagnostics(consensus_score):
    score = np.asarray(consensus_score, dtype=np.float64)
    _require(
        score.ndim == 2
        and score.shape[1] == CLASS_NUM
        and np.isfinite(score).all(),
        "consensus score diagnostic boundary mismatch",
    )
    safe_score = np.clip(score, np.finfo(np.float64).tiny, 1.0)
    entropy = -np.sum(score * np.log(safe_score), axis=1)
    ordered = np.sort(score, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    maximum = ordered[:, -1]
    return {
        "mean_entropy": float(entropy.mean()),
        "mean_top1_top2_margin": float(margin.mean()),
        "mean_max_confidence": float(maximum.mean()),
    }


def semantic_consensus(aligned_q, admission_mask, native_weights):
    q = _as_numpy(aligned_q)
    admission = np.asarray(admission_mask, dtype=bool)
    weights = np.asarray(native_weights, dtype=np.float64)
    _require(
        q.ndim == 3
        and q.shape[1:] == (VIEW_NUM, CLASS_NUM)
        and admission.shape == q.shape[:2]
        and weights.shape == (VIEW_NUM,)
        and np.isfinite(q).all()
        and np.isfinite(weights).all()
        and np.all(weights > 0.0),
        "semantic consensus input boundary mismatch",
    )
    weighted_mask = admission.astype(np.float64) * weights[None, :]
    denominator = weighted_mask.sum(axis=1)
    _require(np.all(denominator > 0.0), "semantic consensus denominator is zero")
    score = np.einsum("nv,nvk->nk", weighted_mask, q) / denominator[:, None]
    prediction = np.argmax(score, axis=1).astype(np.int64, copy=False)
    _require(
        score.shape == (q.shape[0], CLASS_NUM)
        and prediction.shape == (q.shape[0],)
        and np.isfinite(score).all()
        and np.allclose(
            score.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_ATOL
        ),
        "semantic consensus output boundary mismatch",
    )
    return {
        "weighted_mask": np.ascontiguousarray(weighted_mask),
        "denominator": np.ascontiguousarray(denominator),
        "consensus_score": np.ascontiguousarray(score),
        "prediction": np.ascontiguousarray(prediction),
        "diagnostics": consensus_diagnostics(score),
    }


def metrics_from_fixed_prediction(labels, prediction):
    targets = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(prediction, dtype=np.int64)
    _require(
        targets.ndim == 1
        and predicted.shape == targets.shape
        and targets.size > 0,
        "fixed prediction evaluation boundary mismatch",
    )
    return {
        "ACC": float(clustering_accuracy(targets, predicted)),
        "NMI": float(normalized_mutual_info_score(targets, predicted)),
        "ARI": float(adjusted_rand_score(targets, predicted)),
    }


def normal_api_oracle_isolation_pass():
    parameters = inspect.signature(build_normal_admissions).parameters
    return bool(
        set(parameters) == {"utility", "control_seed"}
        and all(
            token not in name.lower()
            for name in parameters
            for token in ("oracle", "corrupt", "clean", "mask")
        )
    )


def reference_hashes(reference, alignment):
    return {
        "global_prediction_sha256": tensor_sha256(
            reference["global_prediction"]
        ),
        "global_centers_sha256": tensor_sha256(reference["global_centers"]),
        "native_weights_sha256": tensor_sha256(
            reference["native_nmi_weights"]
        ),
        "final_latent_fusion_sha256": tensor_sha256(
            reference["latent_fusion"]
        ),
        "alignment_sha256": tensor_sha256(
            alignment["alignment_matrix"]
        ),
        "aligned_q_sha256": tensor_sha256(alignment["aligned_q"]),
    }


def select_evaluation_count(max_samples=None):
    if max_samples is None:
        return SAMPLE_NUM
    if isinstance(max_samples, bool):
        raise ValueError("max_samples must be an integer")
    count = int(max_samples)
    _require(0 < count <= SAMPLE_NUM, "max_samples must be in [1,1400]")
    return count


def fixed_output_hashes(reference, alignment, admissions, policy_outputs, sample_ids):
    hashes = reference_hashes(reference, alignment)
    hashes["evaluation_sample_ids_sha256"] = tensor_sha256(sample_ids)
    hashes["admission_sha256"] = {
        policy: ndarray_sha256(admissions[policy])
        for policy in CONSENSUS_POLICIES
    }
    hashes["score_sha256"] = {
        policy: tensor_sha256(policy_outputs[policy]["consensus_score"])
        for policy in CONSENSUS_POLICIES
    }
    hashes["prediction_sha256"] = {
        policy: tensor_sha256(policy_outputs[policy]["prediction"])
        for policy in CONSENSUS_POLICIES
    }
    hashes["prediction_sha256"]["NATIVE_MVCAN_GLOBAL"] = tensor_sha256(
        reference["global_prediction"][sample_ids]
    )
    hashes["fixed_output_seal_sha256"] = json_sha256(hashes)
    return hashes
