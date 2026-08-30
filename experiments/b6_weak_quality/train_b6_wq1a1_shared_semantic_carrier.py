"""B6-WQ1A-1 Utility-admitted shared semantic carrier prototype."""

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import datasets as dataset_module
import experiments.b6_weak_quality.audit_b6_wq1a0_admission_feasibility as wq1a0
import experiments.b6_weak_quality.evaluate_b6_wq0_utility_semantic_admission as b6
from experiments.b6_weak_quality.shared_semantic_isolation import (
    ViewProjectors,
)
from experiments.b6_weak_quality.shared_semantic_isolation import (
    admission_purity_diagnostic,
)
from experiments.b6_weak_quality.shared_semantic_isolation import (
    admitted_symmetric_infonce,
)
from experiments.b6_weak_quality.shared_semantic_isolation import (
    collapse_diagnostics,
)
from experiments.b6_weak_quality.shared_semantic_isolation import (
    shared_semantic_readout,
)
from irv.b3_audit import hash_backbone
from irv.b3_audit import hash_state_dict
from irv.b4_information_utility import tensor_sha256
from weak_quality import ndarray_sha256


STAGE = "B6-WQ1A-1"
DATASET_NAME = "MSRC-v1"
CONDITION = "snr2p5_k2"
SUPPORTED_HELPER_SEEDS = (20, 30, 50)
SCIENTIFIC_EXECUTION_SEED = 20
SAMPLE_NUM = 210
VIEW_NUM = 5
PRIVATE_DIM = 10
SEMANTIC_DIM = 10
HIDDEN_DIM = 32
CLUSTER_NUM = 7
TOP_K = 3
TEMPERATURE = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SMOKE_EPOCHS = 3
PILOT_EPOCHS = 100
SMOKE_SHUFFLE_REPEATS = 2
PILOT_SHUFFLE_REPEATS = 20
FORMAL_SHUFFLE_REPEATS = 200
DEFAULT_OUTPUT_ROOT = (
    "outputs/b6_weak_quality/wq1a1_shared_semantic_carrier"
)
PILOT_REFERENCE_SUMMARY = (
    "outputs/b6_weak_quality/wq1a1_shared_semantic_carrier/"
    "pilot_seed20/b6_wq1a1_seed20_summary.json"
)
EXPECTED_PILOT_REFERENCE_SHA256 = (
    "662cec723a3499b4e309677f9f6ff333"
    "8e3d9d565e6c46a02c237c86087043ba"
)
EXPECTED_PILOT_METRICS = {
    "canonical_base_metrics": {
        "acc": 0.7047619047619048,
        "nmi": 0.6367546962696464,
        "ari": 0.5365945661561063,
    },
    "uniform_shared_metrics": {
        "acc": 0.6761904761904762,
        "nmi": 0.6191121921966676,
        "ari": 0.5117593430454941,
    },
    "oracle_clean_shared_metrics": {
        "acc": 0.7571428571428571,
        "nmi": 0.689128923334074,
        "ari": 0.6093294689085902,
    },
    "correct_u_shared_metrics": {
        "acc": 0.6952380952380952,
        "nmi": 0.6332783458947051,
        "ari": 0.5474199462545538,
    },
}
EXPECTED_PILOT_SHUFFLED_ACC = {
    "mean": 0.6266666666666666,
    "p50": 0.6214285714285714,
    "p95": 0.6747619047619048,
}
EXPECTED_CANONICAL_MANIFEST_SHA256 = (
    "f7ba5bd25dfe4dfdbbb3db1f4956c601"
    "db33bf3fffce8f456ff2447c4ffa187e"
)
PRIMARY_ARMS = (
    "uniform_shared",
    "oracle_clean_shared",
    "correct_u_shared",
)
WQ1A0_SEED20_EXPECTED = {
    "admitted_clean_fraction": 0.9174603174603174,
    "all_clean_top3_fraction": 0.7523809523809524,
    "corrupted_views_admitted_mean": 0.24761904761904763,
    "oracle_set_jaccard_mean": 0.8761904761904762,
}


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


def load_validated_pilot_reference(seed):
    """Load the approved seed20 pilot without mutating any pilot artifact."""
    seed = int(seed)
    _require(
        seed == SCIENTIFIC_EXECUTION_SEED,
        "B6-WQ1A-1 formal supports only seed20",
    )
    path = _resolve(PILOT_REFERENCE_SUMMARY)
    _require(path.is_file(), "approved seed20 pilot summary missing")
    reference_hash = _file_sha256(path)
    _require(
        reference_hash == EXPECTED_PILOT_REFERENCE_SHA256,
        "approved seed20 pilot summary SHA mismatch",
    )
    summary = _read_json(path)
    _require(
        summary.get("stage") == STAGE
        and summary.get("mode") == "pilot"
        and summary.get("model_seed") == seed,
        "approved pilot identity mismatch",
    )
    for name, expected in EXPECTED_PILOT_METRICS.items():
        _require(
            summary.get(name) == expected,
            "approved pilot metric mismatch: " + name,
        )
    for name, expected in EXPECTED_PILOT_SHUFFLED_ACC.items():
        _require(
            summary["shuffled_acc_summary"].get(name) == expected,
            "approved pilot shuffled ACC mismatch: " + name,
        )
    for gate in (
        "B6_WQ1A1_CARRIER_VIABILITY_PASS",
        "B6_WQ1A1_UTILITY_DIRECTION_PASS",
        "B6_WQ1A1_PILOT_NULL_DIRECTION_PASS",
        "B6_WQ1A1_NO_COLLAPSE_PASS",
        "B6_WQ1A1_PILOT_PROGRESSION_ELIGIBLE",
    ):
        _require(
            summary.get(gate) is True,
            "approved pilot gate failed: " + gate,
        )
    return summary, path, reference_hash


def set_explicit_rng_seeds(model_seed):
    model_seed = int(model_seed)
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(model_seed)
        torch.cuda.manual_seed_all(model_seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return {
        "python_seed": model_seed,
        "numpy_seed": model_seed,
        "torch_seed": model_seed,
        "cuda_seed": model_seed,
        "shuffle_seed": wq1a0.SHUFFLE_SEED,
        "kmeans_seed": model_seed,
        "torch_deterministic_algorithms": True,
    }


def _msrc_data_path():
    for value in dataset_module.MSRC_V1_PATHS:
        path = _resolve(value)
        if path.is_file():
            return path
    raise FileNotFoundError("missing MSRC-v1 data file")


def load_feature_views_without_labels():
    """Read only the MAT fea variable; gt is not loaded in this phase."""
    data_path = _msrc_data_path()
    mat = sio.loadmat(
        data_path,
        variable_names=["fea"],
    )
    _require("fea" in mat and "gt" not in mat, "feature-only MAT read failed")
    raw_views = mat["fea"]
    _require(
        isinstance(raw_views, np.ndarray)
        and raw_views.dtype == object
        and raw_views.size == VIEW_NUM,
        "MSRC-v1 must contain five feature views",
    )
    views = []
    for raw_view in raw_views.ravel():
        view = np.asarray(raw_view)
        _require(view.ndim == 2, "feature view must be a matrix")
        if view.shape[0] == SAMPLE_NUM:
            oriented = view
        elif view.shape[1] == SAMPLE_NUM:
            oriented = view.T
        else:
            raise RuntimeError("feature view has no sample axis of length 210")
        oriented = np.ascontiguousarray(
            oriented.astype(np.float32, copy=False)
        )
        _require(np.isfinite(oriented).all(), "feature view non-finite")
        views.append(oriented)
    return views, data_path


def load_evaluation_labels(data_path):
    """Read only gt after every training arm has completed."""
    mat = sio.loadmat(
        data_path,
        variable_names=["gt"],
    )
    _require("gt" in mat and "fea" not in mat, "label-only MAT read failed")
    raw_labels = np.squeeze(np.asarray(mat["gt"]))
    _require(raw_labels.shape == (SAMPLE_NUM,), "label shape mismatch")
    _require(np.isfinite(raw_labels).all(), "labels non-finite")
    _, labels = np.unique(raw_labels, return_inverse=True)
    labels = labels.astype(np.int64, copy=False)
    _require(
        np.array_equal(np.unique(labels), np.arange(CLUSTER_NUM)),
        "labels must contain seven classes",
    )
    return labels


def _canonical_base_metrics(seed):
    manifest_hash = _file_sha256(b6.CANONICAL_MANIFEST_PATH)
    _require(
        manifest_hash == EXPECTED_CANONICAL_MANIFEST_SHA256,
        "canonical manifest SHA mismatch",
    )
    manifest, _ = b6.load_canonical_manifest()
    entry = manifest["canonical_entries"].get(str(int(seed)))
    _require(entry is not None, "canonical seed entry missing")
    return (
        dict(entry["expected_canonical_metrics"]),
        manifest_hash,
    )


def prepare_frozen_private_inputs(seed):
    """Build canonical frozen [210,5,10] z without loading labels."""
    seed = int(seed)
    _require(seed in SUPPORTED_HELPER_SEEDS, "unsupported helper seed")
    clean_views, data_path = load_feature_views_without_labels()
    evaluation_views, corruption_runtime = b6.apply_weak_quality_protocol(
        clean_views,
        mode="heterogeneous_gaussian",
        k=2,
        snr_db=2.5,
        corruption_seed=seed,
    )
    wq1a0_inputs = wq1a0.load_frozen_audit_inputs(seed)
    _require(
        np.array_equal(
            corruption_runtime["mask"],
            wq1a0_inputs["corruption_mask"],
        ),
        "runtime/stored corruption mask mismatch",
    )
    _require(
        corruption_runtime["mask_sha256"]
        == wq1a0_inputs["corruption_mask_sha256"],
        "runtime/stored corruption hash mismatch",
    )

    models, backbone_hash_before, checkpoint_paths = (
        b6.load_frozen_backbone(seed, evaluation_views)
    )
    fusion_inputs = b6.extract_fusion_inputs(
        models,
        evaluation_views,
        seed,
    )
    z_views = [
        value.detach()
        for value in fusion_inputs["normalized_z_views"]
    ]
    z_hash = b6.tensor_view_list_sha256(z_views)
    _require(
        z_hash == b6.SEED_PROVENANCE[seed]["z_hash"],
        "canonical normalized-z hash mismatch",
    )
    # z_stack: [N,V,Dz] -> frozen MVCAN private representation
    z_stack = torch.stack(z_views, dim=1).detach()
    _require(
        tuple(z_stack.shape)
        == (SAMPLE_NUM, VIEW_NUM, PRIVATE_DIM),
        "frozen z_stack shape mismatch",
    )
    _require(not z_stack.requires_grad, "z_stack must be detached")

    utility_array = np.asarray(wq1a0_inputs["utility"])
    utility_tensor = torch.from_numpy(utility_array).detach()
    _require(
        tuple(utility_tensor.shape) == (SAMPLE_NUM, VIEW_NUM),
        "Utility shape mismatch",
    )
    _require(
        not utility_tensor.requires_grad,
        "Utility must be detached",
    )
    correct_admission = wq1a0.top3_admission_mask(utility_array)
    formal_root = _resolve(wq1a0.DEFAULT_OUTPUT_DIR)
    formal_mask_path = (
        formal_root
        / ("seed" + str(seed))
        / "correct_admission_mask.npy"
    )
    formal_result_path = formal_root / (
        "seed" + str(seed) + "_admission_feasibility.json"
    )
    _require(
        formal_mask_path.is_file()
        and formal_result_path.is_file(),
        "formal WQ1A-0 artifact missing",
    )
    formal_mask = np.load(formal_mask_path, allow_pickle=False)
    formal_result = _read_json(formal_result_path)
    admission_repro = bool(
        np.array_equal(correct_admission, formal_mask)
        and ndarray_sha256(correct_admission)
        == formal_result["correct_admission_mask_sha256"]
        and wq1a0_inputs["utility_sha256"]
        == formal_result["utility_sha256"]
    )
    _require(admission_repro, "WQ1A-0 admission reproduction mismatch")

    oracle_admission = np.asarray(
        wq1a0_inputs["oracle_clean_mask"],
        dtype=bool,
    )
    uniform_admission = np.ones(
        (SAMPLE_NUM, VIEW_NUM),
        dtype=bool,
    )
    backbone_hash_after_extraction = hash_backbone(
        models.autoencoders
    )
    _require(
        backbone_hash_after_extraction == backbone_hash_before,
        "backbone changed during frozen extraction",
    )
    return {
        "model_seed": seed,
        "data_path": data_path,
        "models": models,
        "checkpoint_paths": checkpoint_paths,
        "z_stack": z_stack,
        "z_hash": z_hash,
        "utility_array": utility_array,
        "utility_tensor": utility_tensor,
        "utility_sha256": wq1a0_inputs["utility_sha256"],
        "utility_source": wq1a0_inputs["utility_source"],
        "correct_admission": correct_admission,
        "oracle_admission": oracle_admission,
        "uniform_admission": uniform_admission,
        "corruption_mask": wq1a0_inputs["corruption_mask"],
        "corruption_mask_sha256": (
            wq1a0_inputs["corruption_mask_sha256"]
        ),
        "formal_wq1a0_mask_path": formal_mask_path,
        "formal_wq1a0_result_path": formal_result_path,
        "B6_WQ1A1_WQ1A0_ADMISSION_REPRO_PASS": admission_repro,
        "backbone_hash_before": backbone_hash_before,
    }


def make_shuffled_admissions(utility, repeats):
    repeats = int(repeats)
    _require(
        repeats in (
            SMOKE_SHUFFLE_REPEATS,
            PILOT_SHUFFLE_REPEATS,
            FORMAL_SHUFFLE_REPEATS,
        ),
        "unsupported shuffled repeat count",
    )
    permutation_bank = wq1a0.generate_permutation_bank()
    masks = []
    utility_hashes = []
    for repeat_id in range(repeats):
        shuffled_utility = wq1a0.shuffle_utility_within_views(
            utility,
            permutation_bank[repeat_id],
        )
        for view_id in range(VIEW_NUM):
            _require(
                np.array_equal(
                    np.sort(shuffled_utility[:, view_id]),
                    np.sort(utility[:, view_id]),
                ),
                "shuffled Utility marginal changed",
            )
        masks.append(
            wq1a0.top3_admission_mask(shuffled_utility)
        )
        utility_hashes.append(tensor_sha256(shuffled_utility))
    return masks, utility_hashes


def _arm_metadata(
    arm_name,
    model_seed,
    epochs,
    admission_source,
    initial_hash,
    final_hash,
):
    return {
        "stage": STAGE,
        "arm": arm_name,
        "model_seed": int(model_seed),
        "epochs": int(epochs),
        "batch_size": SAMPLE_NUM,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "temperature": TEMPERATURE,
        "private_dim": PRIVATE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "semantic_dim": SEMANTIC_DIM,
        "admission_top_k": (
            None if arm_name == "uniform_shared" else TOP_K
        ),
        "admission_source": admission_source,
        "projector_architecture": (
            "per-view Linear(10,32)-GELU-Linear(32,10)"
        ),
        "projector_initial_state_sha256": initial_hash,
        "projector_final_state_sha256": final_hash,
        "oracle_diagnostic_only": bool(
            arm_name == "oracle_clean_shared"
        ),
        "labels_used_for_training": False,
        "labels_used_for_model_selection": False,
        "corruption_mask_available_to_training_function": False,
        "utility_values_available_to_training_function": False,
        "only_projector_parameters_optimized": True,
    }


def train_shared_arm(
    arm_name,
    z_stack,
    admission,
    initial_state,
    initial_state_hash,
    epochs,
    model_seed,
    output_dir,
    encoder_modules,
    detached_input_audit,
    admission_source,
):
    """Train only view projectors; no labels, Utility values, or mask oracle."""
    epochs = int(epochs)
    _require(
        epochs in (SMOKE_EPOCHS, PILOT_EPOCHS),
        "epochs must be the fixed smoke or pilot value",
    )
    _require(
        tuple(z_stack.shape)
        == (SAMPLE_NUM, VIEW_NUM, PRIVATE_DIM),
        "z_stack shape mismatch",
    )
    _require(
        not z_stack.requires_grad,
        "private z must remain detached",
    )
    admission_array = np.asarray(admission, dtype=bool)
    _require(
        admission_array.shape == (SAMPLE_NUM, VIEW_NUM),
        "admission shape mismatch",
    )
    admission_tensor = torch.from_numpy(
        admission_array
    ).to(dtype=torch.bool)
    _require(
        not admission_tensor.requires_grad,
        "admission must be non-differentiable",
    )

    projector = ViewProjectors()
    projector.load_state_dict(copy.deepcopy(initial_state), strict=True)
    arm_initial_hash = hash_state_dict(projector.state_dict())
    _require(
        arm_initial_hash == initial_state_hash,
        "projector initial state mismatch",
    )
    optimizer = torch.optim.Adam(
        projector.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    gradient_audit = None
    fixed_pair_sample_counts = None
    for epoch_id in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        # h_stack: [N,V,Ds]
        h_stack = projector(z_stack)
        semantic_loss, loss_audit = admitted_symmetric_infonce(
            h_stack,
            admission_tensor,
            TEMPERATURE,
        )
        pair_sample_counts = loss_audit["pair_sample_counts"]
        if fixed_pair_sample_counts is None:
            fixed_pair_sample_counts = dict(pair_sample_counts)
        else:
            _require(
                pair_sample_counts == fixed_pair_sample_counts,
                "pair sample counts changed across epochs",
            )
        semantic_loss.backward()
        if epoch_id == 0:
            projector_gradients = [
                parameter.grad
                for parameter in projector.parameters()
                if parameter.grad is not None
            ]
            encoder_gradients = [
                parameter.grad
                for encoder in encoder_modules
                for parameter in encoder.parameters()
                if parameter.grad is not None
            ]
            gradient_audit = {
                "private_z_requires_grad": bool(
                    z_stack.requires_grad
                ),
                "private_z_grad_exists": bool(
                    z_stack.grad is not None
                ),
                "utility_requires_grad": bool(
                    detached_input_audit[
                        "utility_requires_grad"
                    ]
                ),
                "utility_grad_exists": bool(
                    detached_input_audit[
                        "utility_grad_exists"
                    ]
                ),
                "admission_requires_grad": bool(
                    admission_tensor.requires_grad
                ),
                "encoder_grad_tensor_count": int(
                    len(encoder_gradients)
                ),
                "projector_grad_tensor_count": int(
                    len(projector_gradients)
                ),
                "projector_grad_all_finite": bool(
                    projector_gradients
                    and all(
                        torch.isfinite(gradient).all().item()
                        for gradient in projector_gradients
                    )
                ),
            }
            gradient_audit[
                "B6_WQ1A1_GRADIENT_PATH_PASS"
            ] = bool(
                not gradient_audit[
                    "private_z_requires_grad"
                ]
                and not gradient_audit[
                    "private_z_grad_exists"
                ]
                and not gradient_audit[
                    "utility_requires_grad"
                ]
                and not gradient_audit[
                    "utility_grad_exists"
                ]
                and not gradient_audit[
                    "admission_requires_grad"
                ]
                and gradient_audit[
                    "encoder_grad_tensor_count"
                ] == 0
                and gradient_audit[
                    "projector_grad_tensor_count"
                ] > 0
                and gradient_audit[
                    "projector_grad_all_finite"
                ]
            )
        optimizer.step()
        history.append({
            "epoch": int(epoch_id + 1),
            "semantic_loss": float(
                semantic_loss.detach().item()
            ),
            "weighted_pair_sample_count": int(
                loss_audit["weighted_sample_count"]
            ),
            "loss_finite": bool(
                torch.isfinite(semantic_loss).item()
            ),
        })

    projector.eval()
    with torch.no_grad():
        # h_stack: [N,V,Ds]
        final_h_stack = projector(z_stack)
        # semantic: [N,Ds]
        final_semantic = shared_semantic_readout(
            final_h_stack,
            admission_tensor,
        )
    semantic_array = np.ascontiguousarray(
        final_semantic.detach().cpu().numpy()
    )
    _require(
        semantic_array.shape == (SAMPLE_NUM, SEMANTIC_DIM),
        "final semantic shape mismatch",
    )
    prediction = b6._kmeans(
        model_seed,
        sample_num=SAMPLE_NUM,
    ).fit_predict(semantic_array)
    prediction = np.ascontiguousarray(
        np.asarray(prediction, dtype=np.int64)
    )
    _require(
        prediction.shape == (SAMPLE_NUM,),
        "KMeans prediction shape mismatch",
    )
    final_hash = hash_state_dict(projector.state_dict())
    semantic_hash = tensor_sha256(semantic_array)
    prediction_hash = tensor_sha256(prediction)
    collapse = collapse_diagnostics(
        final_semantic.detach(),
        prediction,
    )
    arm_dir = Path(output_dir)
    arm_dir.mkdir(parents=True, exist_ok=True)
    np.save(arm_dir / "final_semantic.npy", semantic_array)
    np.save(arm_dir / "prediction.npy", prediction)
    np.save(arm_dir / "admission_mask.npy", admission_array)
    metadata = _arm_metadata(
        arm_name,
        model_seed,
        epochs,
        admission_source,
        arm_initial_hash,
        final_hash,
    )
    metadata.update({
        "semantic_sha256": semantic_hash,
        "prediction_sha256": prediction_hash,
    })
    _write_json(arm_dir / "metadata.json", metadata)
    _write_json(
        arm_dir / "training_history.json",
        {
            "epochs": epochs,
            "rows": history,
        },
    )
    _write_json(
        arm_dir / "collapse_diagnostics.json",
        collapse,
    )
    _write_json(
        arm_dir / "gradient_audit.json",
        gradient_audit,
    )
    _write_json(
        arm_dir / "pair_sample_counts.json",
        fixed_pair_sample_counts,
    )
    return {
        "arm": arm_name,
        "arm_dir": arm_dir,
        "initial_state_sha256": arm_initial_hash,
        "final_state_sha256": final_hash,
        "semantic_sha256": semantic_hash,
        "prediction_sha256": prediction_hash,
        "history": history,
        "gradient_audit": gradient_audit,
        "pair_sample_counts": fixed_pair_sample_counts,
        "semantic": semantic_array,
        "prediction": prediction,
        "admission": admission_array,
        "collapse_diagnostics": collapse,
        "metadata": metadata,
    }


def _evaluate_and_save_metrics(record, labels):
    metrics = b6.metrics_from_predictions(
        labels,
        record["prediction"],
    )
    _write_json(record["arm_dir"] / "metrics.json", metrics)
    record["metrics"] = metrics
    return metrics


def _metric_delta(left, right):
    return {
        name: float(left[name] - right[name])
        for name in ("acc", "nmi", "ari")
    }


def _metric_distribution(rows, metric, screening_only=True):
    values = np.asarray(
        [row["metrics"][metric] for row in rows],
        dtype=np.float64,
    )
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "values": [float(value) for value in values.tolist()],
        "screening_only": bool(screening_only),
    }


def formal_acc_null_audit(shuffled_acc, correct_acc):
    values = np.asarray(shuffled_acc, dtype=np.float64)
    correct_acc = float(correct_acc)
    _require(
        values.shape == (FORMAL_SHUFFLE_REPEATS,),
        "formal shuffled ACC must contain exactly 200 values",
    )
    _require(
        np.isfinite(values).all() and np.isfinite(correct_acc),
        "formal ACC audit contains non-finite values",
    )
    p95 = float(np.percentile(values, 95))
    count_ge = int(np.sum(values >= correct_acc))
    empirical_p = float(
        (1 + count_ge) / (FORMAL_SHUFFLE_REPEATS + 1)
    )
    return {
        "formal_shuffled_acc_mean": float(np.mean(values)),
        "formal_shuffled_acc_std": float(np.std(values, ddof=0)),
        "formal_shuffled_acc_p05": float(np.percentile(values, 5)),
        "formal_shuffled_acc_p50": float(np.percentile(values, 50)),
        "formal_shuffled_acc_p95": p95,
        "count_shuffled_ge_correct": count_ge,
        "one_sided_empirical_p": empirical_p,
        "B6_WQ1A1_FORMAL_NULL_PASS": bool(
            correct_acc > p95 and empirical_p < 0.05
        ),
    }


def formal_seed20_gate(
    formal_null_pass,
    carrier_viability_pass,
    utility_direction_pass,
    no_collapse_pass,
    backbone_frozen_pass,
    admission_repro_pass,
    projector_init_match_pass,
    gradient_path_pass,
    no_label_leakage_pass,
):
    return bool(all((
        formal_null_pass,
        carrier_viability_pass,
        utility_direction_pass,
        no_collapse_pass,
        backbone_frozen_pass,
        admission_repro_pass,
        projector_init_match_pass,
        gradient_path_pass,
        no_label_leakage_pass,
    )))


def _record_summary(record):
    return {
        "metrics": record["metrics"],
        "projector_initial_state_sha256": (
            record["initial_state_sha256"]
        ),
        "projector_final_state_sha256": (
            record["final_state_sha256"]
        ),
        "semantic_sha256": record["semantic_sha256"],
        "prediction_sha256": record["prediction_sha256"],
        "collapse_diagnostics": (
            record["collapse_diagnostics"]
        ),
        "gradient_audit": record["gradient_audit"],
        "pair_sample_counts": record["pair_sample_counts"],
        "artifact_dir": _display(record["arm_dir"]),
    }


def _validate_correct_admission_purity(purity):
    return bool(
        purity["admitted_clean_fraction"]
        == WQ1A0_SEED20_EXPECTED["admitted_clean_fraction"]
        and purity["all_admitted_clean_fraction"]
        == WQ1A0_SEED20_EXPECTED[
            "all_clean_top3_fraction"
        ]
        and purity["corrupted_views_admitted_mean"]
        == WQ1A0_SEED20_EXPECTED[
            "corrupted_views_admitted_mean"
        ]
        and purity["oracle_set_jaccard_mean"]
        == WQ1A0_SEED20_EXPECTED[
            "oracle_set_jaccard_mean"
        ]
    )


def _mode_protocol(mode):
    if mode == "smoke":
        return SMOKE_EPOCHS, SMOKE_SHUFFLE_REPEATS
    if mode == "pilot":
        return PILOT_EPOCHS, PILOT_SHUFFLE_REPEATS
    if mode == "formal":
        return PILOT_EPOCHS, FORMAL_SHUFFLE_REPEATS
    raise RuntimeError("unsupported B6-WQ1A-1 mode")


def _default_mode_output(mode, seed):
    return (
        _resolve(DEFAULT_OUTPUT_ROOT)
        / (str(mode) + "_seed" + str(int(seed)))
    )


def run_formal_dry_validation(
    seed=SCIENTIFIC_EXECUTION_SEED,
    output_dir=None,
):
    """Validate formal plumbing without training, KMeans fitting, or labels."""
    seed = int(seed)
    _require(
        seed == SCIENTIFIC_EXECUTION_SEED,
        "B6-WQ1A-1 formal supports only seed20",
    )
    epochs, shuffled_repeats = _mode_protocol("formal")
    _require(
        epochs == PILOT_EPOCHS
        and shuffled_repeats == FORMAL_SHUFFLE_REPEATS,
        "formal protocol mismatch",
    )
    pilot_summary, pilot_path, pilot_hash_before = (
        load_validated_pilot_reference(seed)
    )
    canonical_base, manifest_hash_before = (
        _canonical_base_metrics(seed)
    )
    _require(
        canonical_base
        == EXPECTED_PILOT_METRICS["canonical_base_metrics"],
        "formal canonical base differs from approved pilot",
    )

    rng_audit = set_explicit_rng_seeds(seed)
    prepared = prepare_frozen_private_inputs(seed)
    shuffled_masks, shuffled_utility_hashes = (
        make_shuffled_admissions(
            prepared["utility_array"],
            shuffled_repeats,
        )
    )
    admission_count_pass = bool(
        len(shuffled_masks) == FORMAL_SHUFFLE_REPEATS
        and len(shuffled_utility_hashes)
        == FORMAL_SHUFFLE_REPEATS
        and all(
            mask.shape == (SAMPLE_NUM, VIEW_NUM)
            and np.all(mask.sum(axis=1) == TOP_K)
            for mask in shuffled_masks
        )
    )

    set_explicit_rng_seeds(seed)
    template = ViewProjectors()
    initial_state = copy.deepcopy(template.state_dict())
    initial_state_hash = hash_state_dict(initial_state)
    shuffled_initial_hashes = []
    for _ in range(FORMAL_SHUFFLE_REPEATS):
        projector = ViewProjectors()
        projector.load_state_dict(
            copy.deepcopy(initial_state),
            strict=True,
        )
        shuffled_initial_hashes.append(
            hash_state_dict(projector.state_dict())
        )
    projector_init_match_pass = bool(
        len(shuffled_initial_hashes)
        == FORMAL_SHUFFLE_REPEATS
        and all(
            value == initial_state_hash
            for value in shuffled_initial_hashes
        )
    )

    kmeans = b6._kmeans(seed, sample_num=SAMPLE_NUM)
    kmeans_protocol_pass = bool(
        kmeans.get_params()["n_init"] == 100
        and kmeans.get_params()["random_state"] == seed
    )
    backbone_hash_after = hash_backbone(
        prepared["models"].autoencoders
    )
    backbone_frozen_pass = bool(
        backbone_hash_after
        == prepared["backbone_hash_before"]
        and all(
            not parameter.requires_grad
            and parameter.grad is None
            for encoder in prepared["models"].autoencoders
            for parameter in encoder.parameters()
        )
    )
    utility_unchanged_pass = bool(
        tensor_sha256(prepared["utility_array"])
        == prepared["utility_sha256"]
        and not prepared["utility_tensor"].requires_grad
        and prepared["utility_tensor"].grad is None
    )
    manifest_hash_after = _file_sha256(
        b6.CANONICAL_MANIFEST_PATH
    )
    pilot_hash_after = _file_sha256(pilot_path)
    protected_artifacts_unchanged_pass = bool(
        manifest_hash_after == manifest_hash_before
        and pilot_hash_after == pilot_hash_before
    )
    dry_pass = bool(
        admission_count_pass
        and projector_init_match_pass
        and kmeans_protocol_pass
        and backbone_frozen_pass
        and utility_unchanged_pass
        and prepared[
            "B6_WQ1A1_WQ1A0_ADMISSION_REPRO_PASS"
        ]
        and protected_artifacts_unchanged_pass
    )
    result = {
        "stage": STAGE,
        "mode": "formal",
        "dry_validation": True,
        "model_seed": seed,
        "fixed_protocol": {
            "epochs": epochs,
            "shuffled_repeats": shuffled_repeats,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "temperature": TEMPERATURE,
            "semantic_dim": SEMANTIC_DIM,
            "hidden_dim": HIDDEN_DIM,
            "top_k": TOP_K,
            "full_batch_size": SAMPLE_NUM,
            "kmeans_n_init": int(kmeans.get_params()["n_init"]),
            "kmeans_random_state": int(
                kmeans.get_params()["random_state"]
            ),
        },
        "rng_audit": rng_audit,
        "approved_pilot_summary": _display(pilot_path),
        "approved_pilot_summary_sha256_before": (
            pilot_hash_before
        ),
        "approved_pilot_summary_sha256_after": (
            pilot_hash_after
        ),
        "approved_pilot_progression_eligible": bool(
            pilot_summary[
                "B6_WQ1A1_PILOT_PROGRESSION_ELIGIBLE"
            ]
        ),
        "canonical_manifest_sha256_before": (
            manifest_hash_before
        ),
        "canonical_manifest_sha256_after": (
            manifest_hash_after
        ),
        "canonical_base_metrics": canonical_base,
        "backbone_hash_before": (
            prepared["backbone_hash_before"]
        ),
        "backbone_hash_after": backbone_hash_after,
        "private_z_sha256": prepared["z_hash"],
        "utility_sha256": prepared["utility_sha256"],
        "shuffled_utility_sha256": shuffled_utility_hashes,
        "canonical_projector_initial_state_sha256": (
            initial_state_hash
        ),
        "shuffled_projector_initial_state_sha256": (
            shuffled_initial_hashes
        ),
        "training_performed": False,
        "optimizer_constructed": False,
        "backward_performed": False,
        "kmeans_fit_performed": False,
        "labels_loaded": False,
        "formal_scientific_run_performed": False,
        "multiseed_scientific_conclusion_performed": False,
        "B6_WQ1A1_FORMAL_ADMISSION_COUNT_PASS": (
            admission_count_pass
        ),
        "B6_WQ1A1_PROJECTOR_INIT_MATCH_PASS": (
            projector_init_match_pass
        ),
        "B6_WQ1A1_KMEANS_PROTOCOL_PASS": (
            kmeans_protocol_pass
        ),
        "B6_WQ1A1_BACKBONE_FROZEN_PASS": (
            backbone_frozen_pass
        ),
        "B6_WQ1A1_UTILITY_DETACHED_PASS": (
            utility_unchanged_pass
        ),
        "B6_WQ1A1_WQ1A0_ADMISSION_REPRO_PASS": (
            prepared[
                "B6_WQ1A1_WQ1A0_ADMISSION_REPRO_PASS"
            ]
        ),
        "B6_WQ1A1_PROTECTED_ARTIFACTS_UNCHANGED_PASS": (
            protected_artifacts_unchanged_pass
        ),
        "B6_WQ1A1_FORMAL_DRY_VALIDATION_PASS": dry_pass,
    }
    root = (
        _default_mode_output("formal", seed)
        if output_dir is None
        else _resolve(output_dir)
    )
    _write_json(root / "formal_dry_validation.json", result)
    return result


def run_experiment(mode, seed=SCIENTIFIC_EXECUTION_SEED):
    mode = str(mode)
    seed = int(seed)
    _require(
        seed == SCIENTIFIC_EXECUTION_SEED,
        "B6-WQ1A-1 scientific execution supports only seed20",
    )
    epochs, shuffled_repeats = _mode_protocol(mode)
    pilot_reference_summary = None
    pilot_reference_path = None
    pilot_reference_hash_before = None
    if mode == "formal":
        (
            pilot_reference_summary,
            pilot_reference_path,
            pilot_reference_hash_before,
        ) = load_validated_pilot_reference(seed)
    canonical_base, manifest_hash_before = (
        _canonical_base_metrics(seed)
    )
    rng_audit = set_explicit_rng_seeds(seed)
    prepared = prepare_frozen_private_inputs(seed)
    output_root = _default_mode_output(mode, seed)
    output_root.mkdir(parents=True, exist_ok=True)

    set_explicit_rng_seeds(seed)
    base_projector = ViewProjectors()
    initial_state = copy.deepcopy(
        base_projector.state_dict()
    )
    initial_state_hash = hash_state_dict(initial_state)
    torch.save(
        initial_state,
        output_root / "initial_state_dict.pt",
    )
    _write_json(
        output_root / "initial_state_audit.json",
        {
            "initial_state_sha256": initial_state_hash,
            **rng_audit,
        },
    )

    shuffled_masks, shuffled_utility_hashes = (
        make_shuffled_admissions(
            prepared["utility_array"],
            shuffled_repeats,
        )
    )
    detached_input_audit = {
        "utility_requires_grad": bool(
            prepared["utility_tensor"].requires_grad
        ),
        "utility_grad_exists": bool(
            prepared["utility_tensor"].grad is not None
        ),
    }
    arm_specs = [
        (
            "uniform_shared",
            prepared["uniform_admission"],
            "all-five binary admission",
            output_root / "uniform_shared",
        ),
        (
            "oracle_clean_shared",
            prepared["oracle_admission"],
            "evaluation-only clean-mask upper bound",
            output_root / "oracle_clean_shared",
        ),
        (
            "correct_u_shared",
            prepared["correct_admission"],
            "WQ1A-0 frozen Utility Top3 binary admission",
            output_root / "correct_u_shared",
        ),
    ]
    records = {}
    for arm_name, admission, source, arm_dir in arm_specs:
        records[arm_name] = train_shared_arm(
            arm_name,
            prepared["z_stack"],
            admission,
            initial_state,
            initial_state_hash,
            epochs,
            seed,
            arm_dir,
            prepared["models"].autoencoders,
            detached_input_audit,
            source,
        )

    shuffled_records = []
    for repeat_id, admission in enumerate(shuffled_masks):
        repeat_dir = (
            output_root
            / "shuffled_u"
            / ("repeat_" + str(repeat_id).zfill(3))
        )
        record = train_shared_arm(
            "shuffled_u_shared",
            prepared["z_stack"],
            admission,
            initial_state,
            initial_state_hash,
            epochs,
            seed,
            repeat_dir,
            prepared["models"].autoencoders,
            detached_input_audit,
            (
                "within-view shuffled frozen Utility Top3 "
                "binary admission"
            ),
        )
        record["repeat_id"] = int(repeat_id)
        record["shuffled_utility_sha256"] = (
            shuffled_utility_hashes[repeat_id]
        )
        shuffled_records.append(record)

    determinism_record = None
    if mode == "smoke":
        determinism_record = train_shared_arm(
            "correct_u_shared",
            prepared["z_stack"],
            prepared["correct_admission"],
            initial_state,
            initial_state_hash,
            epochs,
            seed,
            output_root / "correct_u_determinism_repeat",
            prepared["models"].autoencoders,
            detached_input_audit,
            "WQ1A-0 frozen Utility Top3 binary admission",
        )

    training_complete_before_label_load = bool(
        len(records) == len(PRIMARY_ARMS)
        and len(shuffled_records) == shuffled_repeats
        and (
            mode != "smoke"
            or determinism_record is not None
        )
    )
    _require(
        training_complete_before_label_load,
        "label load attempted before training completed",
    )
    labels = load_evaluation_labels(prepared["data_path"])
    for record in records.values():
        _evaluate_and_save_metrics(record, labels)
    for record in shuffled_records:
        _evaluate_and_save_metrics(record, labels)
    if determinism_record is not None:
        _evaluate_and_save_metrics(
            determinism_record,
            labels,
        )

    correct_record = records["correct_u_shared"]
    if determinism_record is None:
        smoke_determinism_pass = None
    else:
        smoke_determinism_pass = bool(
            correct_record["initial_state_sha256"]
            == determinism_record["initial_state_sha256"]
            and np.array_equal(
                correct_record["admission"],
                determinism_record["admission"],
            )
            and correct_record["history"]
            == determinism_record["history"]
            and correct_record["semantic_sha256"]
            == determinism_record["semantic_sha256"]
            and correct_record["prediction_sha256"]
            == determinism_record["prediction_sha256"]
            and correct_record["metrics"]
            == determinism_record["metrics"]
        )

    all_records = (
        list(records.values())
        + shuffled_records
        + (
            [determinism_record]
            if determinism_record is not None
            else []
        )
    )
    initial_match_pass = bool(
        all(
            record["initial_state_sha256"]
            == initial_state_hash
            for record in all_records
        )
    )
    gradient_path_pass = bool(
        all(
            record["gradient_audit"][
                "B6_WQ1A1_GRADIENT_PATH_PASS"
            ]
            for record in all_records
        )
    )
    backbone_hash_after = hash_backbone(
        prepared["models"].autoencoders
    )
    backbone_frozen_pass = bool(
        backbone_hash_after
        == prepared["backbone_hash_before"]
        and all(
            not parameter.requires_grad
            and parameter.grad is None
            for encoder in prepared["models"].autoencoders
            for parameter in encoder.parameters()
        )
    )
    utility_detached_pass = bool(
        not prepared["utility_tensor"].requires_grad
        and prepared["utility_tensor"].grad is None
    )
    admission_count_pass = bool(
        np.all(
            prepared["uniform_admission"].sum(axis=1)
            == VIEW_NUM
        )
        and np.all(
            prepared["oracle_admission"].sum(axis=1)
            == TOP_K
        )
        and np.all(
            prepared["correct_admission"].sum(axis=1)
            == TOP_K
        )
        and all(
            np.all(mask.sum(axis=1) == TOP_K)
            for mask in shuffled_masks
        )
    )
    oracle_mask_pass = bool(
        np.array_equal(
            prepared["oracle_admission"],
            np.logical_not(
                prepared["corruption_mask"]
            ),
        )
        and not np.any(
            np.logical_and(
                prepared["oracle_admission"],
                prepared["corruption_mask"],
            )
        )
    )
    purity = {
        "uniform_shared": admission_purity_diagnostic(
            prepared["uniform_admission"],
            prepared["oracle_admission"],
        ),
        "oracle_clean_shared": admission_purity_diagnostic(
            prepared["oracle_admission"],
            prepared["oracle_admission"],
        ),
        "correct_u_shared": admission_purity_diagnostic(
            prepared["correct_admission"],
            prepared["oracle_admission"],
        ),
        "shuffled_u": [
            {
                "repeat_id": int(repeat_id),
                **admission_purity_diagnostic(
                    mask,
                    prepared["oracle_admission"],
                ),
            }
            for repeat_id, mask in enumerate(shuffled_masks)
        ],
    }
    admission_a0_repro_pass = bool(
        prepared[
            "B6_WQ1A1_WQ1A0_ADMISSION_REPRO_PASS"
        ]
        and _validate_correct_admission_purity(
            purity["correct_u_shared"]
        )
    )
    all_finite_pass = bool(
        all(
            np.isfinite([
                row["semantic_loss"]
                for row in record["history"]
            ]).all()
            and record["collapse_diagnostics"][
                "semantic_all_finite"
            ]
            and all(
                np.isfinite(value)
                for value in record["metrics"].values()
            )
            for record in all_records
        )
    )
    primary_no_collapse_pass = bool(
        all(
            records[arm]["collapse_diagnostics"][
                "B6_WQ1A1_NO_COLLAPSE_PASS"
            ]
            for arm in PRIMARY_ARMS
        )
    )
    no_label_leakage_pass = bool(
        training_complete_before_label_load
        and all(
            not record["metadata"][
                "labels_used_for_training"
            ]
            and not record["metadata"][
                "labels_used_for_model_selection"
            ]
            for record in all_records
        )
    )

    uniform_metrics = records["uniform_shared"]["metrics"]
    oracle_metrics = records["oracle_clean_shared"]["metrics"]
    correct_metrics = records["correct_u_shared"]["metrics"]
    shuffled_acc_summary = _metric_distribution(
        shuffled_records,
        "acc",
        screening_only=mode != "formal",
    )
    shuffled_nmi_summary = _metric_distribution(
        shuffled_records,
        "nmi",
        screening_only=mode != "formal",
    )
    shuffled_ari_summary = _metric_distribution(
        shuffled_records,
        "ari",
        screening_only=mode != "formal",
    )
    formal_acc_audit = None
    formal_metric_arrays = None
    formal_primary_arm_repro_pass = None
    if mode == "formal":
        formal_metric_arrays = {
            metric: np.asarray(
                [
                    record["metrics"][metric]
                    for record in shuffled_records
                ],
                dtype=np.float64,
            )
            for metric in ("acc", "nmi", "ari")
        }
        for metric, values in formal_metric_arrays.items():
            _require(
                values.shape == (FORMAL_SHUFFLE_REPEATS,)
                and np.isfinite(values).all(),
                "formal shuffled " + metric + " array mismatch",
            )
            np.save(
                output_root / ("shuffled_" + metric + ".npy"),
                values,
            )
        formal_acc_audit = formal_acc_null_audit(
            formal_metric_arrays["acc"],
            correct_metrics["acc"],
        )
        formal_primary_arm_repro_pass = bool(
            canonical_base
            == EXPECTED_PILOT_METRICS[
                "canonical_base_metrics"
            ]
            and uniform_metrics
            == EXPECTED_PILOT_METRICS[
                "uniform_shared_metrics"
            ]
            and oracle_metrics
            == EXPECTED_PILOT_METRICS[
                "oracle_clean_shared_metrics"
            ]
            and correct_metrics
            == EXPECTED_PILOT_METRICS[
                "correct_u_shared_metrics"
            ]
        )

    if mode in ("pilot", "formal"):
        carrier_viability_pass = bool(
            oracle_metrics["acc"] > uniform_metrics["acc"]
        )
        utility_direction_pass = bool(
            correct_metrics["acc"] > uniform_metrics["acc"]
        )
        if mode == "pilot":
            pilot_null_direction_pass = bool(
                correct_metrics["acc"]
                > shuffled_acc_summary["p50"]
            )
            progression_eligible = bool(
                backbone_frozen_pass
                and admission_a0_repro_pass
                and initial_match_pass
                and gradient_path_pass
                and carrier_viability_pass
                and utility_direction_pass
                and pilot_null_direction_pass
                and primary_no_collapse_pass
            )
        else:
            pilot_null_direction_pass = bool(
                pilot_reference_summary[
                    "B6_WQ1A1_PILOT_NULL_DIRECTION_PASS"
                ]
            )
            progression_eligible = bool(
                pilot_reference_summary[
                    "B6_WQ1A1_PILOT_PROGRESSION_ELIGIBLE"
                ]
            )
    else:
        carrier_viability_pass = None
        utility_direction_pass = None
        pilot_null_direction_pass = None
        progression_eligible = None

    manifest_hash_after = _file_sha256(
        b6.CANONICAL_MANIFEST_PATH
    )
    _require(
        manifest_hash_after == manifest_hash_before,
        "canonical manifest changed",
    )
    pilot_reference_hash_after = None
    protected_pilot_unchanged_pass = None
    formal_null_pass = None
    formal_seed20_pass = None
    multiseed_entry_eligible = None
    if mode == "formal":
        pilot_reference_hash_after = _file_sha256(
            pilot_reference_path
        )
        protected_pilot_unchanged_pass = bool(
            pilot_reference_hash_after
            == pilot_reference_hash_before
        )
        _require(
            protected_pilot_unchanged_pass,
            "approved pilot artifact changed",
        )
        formal_null_pass = bool(
            formal_acc_audit[
                "B6_WQ1A1_FORMAL_NULL_PASS"
            ]
        )
        formal_seed20_pass = formal_seed20_gate(
            formal_null_pass,
            carrier_viability_pass,
            utility_direction_pass,
            primary_no_collapse_pass,
            backbone_frozen_pass,
            admission_a0_repro_pass,
            initial_match_pass,
            gradient_path_pass,
            no_label_leakage_pass,
        )
        multiseed_entry_eligible = formal_seed20_pass

    gates = {
        "B6_WQ1A1_BACKBONE_FROZEN_PASS": (
            backbone_frozen_pass
        ),
        "B6_WQ1A1_WQ1A0_ADMISSION_REPRO_PASS": (
            admission_a0_repro_pass
        ),
        "B6_WQ1A1_UTILITY_DETACHED_PASS": (
            utility_detached_pass
        ),
        "B6_WQ1A1_PROJECTOR_INIT_MATCH_PASS": (
            initial_match_pass
        ),
        "B6_WQ1A1_GRADIENT_PATH_PASS": (
            gradient_path_pass
        ),
        "B6_WQ1A1_NO_LABEL_LEAKAGE_PASS": (
            no_label_leakage_pass
        ),
        "B6_WQ1A1_ADMISSION_COUNT_PASS": (
            admission_count_pass
        ),
        "B6_WQ1A1_ORACLE_MASK_PASS": oracle_mask_pass,
        "B6_WQ1A1_ALL_FINITE_PASS": all_finite_pass,
        "B6_WQ1A1_NO_COLLAPSE_PASS": (
            primary_no_collapse_pass
        ),
        "B6_WQ1A1_SMOKE_DETERMINISM_PASS": (
            smoke_determinism_pass
        ),
        "B6_WQ1A1_CARRIER_VIABILITY_PASS": (
            carrier_viability_pass
        ),
        "B6_WQ1A1_UTILITY_DIRECTION_PASS": (
            utility_direction_pass
        ),
        "B6_WQ1A1_PILOT_NULL_DIRECTION_PASS": (
            pilot_null_direction_pass
        ),
        "B6_WQ1A1_PILOT_PROGRESSION_ELIGIBLE": (
            progression_eligible
        ),
        "B6_WQ1A1_FORMAL_NULL_PASS": formal_null_pass,
        "B6_WQ1A1_FORMAL_SEED20_PASS": formal_seed20_pass,
        "B6_WQ1A1_MULTISEED_ENTRY_ELIGIBLE": (
            multiseed_entry_eligible
        ),
        "B6_WQ1A1_FORMAL_PRIMARY_ARM_REPRO_PASS": (
            formal_primary_arm_repro_pass
        ),
        "B6_WQ1A1_PROTECTED_PILOT_UNCHANGED_PASS": (
            protected_pilot_unchanged_pass
        ),
    }
    smoke_complete = bool(
        mode == "smoke"
        and all(
            gates[name]
            for name in (
                "B6_WQ1A1_BACKBONE_FROZEN_PASS",
                "B6_WQ1A1_WQ1A0_ADMISSION_REPRO_PASS",
                "B6_WQ1A1_UTILITY_DETACHED_PASS",
                "B6_WQ1A1_PROJECTOR_INIT_MATCH_PASS",
                "B6_WQ1A1_GRADIENT_PATH_PASS",
                "B6_WQ1A1_NO_LABEL_LEAKAGE_PASS",
                "B6_WQ1A1_ADMISSION_COUNT_PASS",
                "B6_WQ1A1_ORACLE_MASK_PASS",
                "B6_WQ1A1_ALL_FINITE_PASS",
                "B6_WQ1A1_SMOKE_DETERMINISM_PASS",
            )
        )
    )
    gates["B6_WQ1A1_SMOKE_COMPLETE"] = (
        smoke_complete if mode == "smoke" else None
    )

    formal_summary_fields = {
        "formal_shuffled_acc_mean": None,
        "formal_shuffled_acc_std": None,
        "formal_shuffled_acc_p05": None,
        "formal_shuffled_acc_p50": None,
        "formal_shuffled_acc_p95": None,
        "count_shuffled_ge_correct": None,
        "one_sided_empirical_p": None,
    }
    if formal_acc_audit is not None:
        formal_summary_fields.update(formal_acc_audit)

    summary = {
        "stage": STAGE,
        "mode": mode,
        "dataset": DATASET_NAME,
        "condition": CONDITION,
        "model_seed": seed,
        "fixed_protocol": {
            "epochs": epochs,
            "shuffled_repeats": shuffled_repeats,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "temperature": TEMPERATURE,
            "semantic_dim": SEMANTIC_DIM,
            "hidden_dim": HIDDEN_DIM,
            "top_k": TOP_K,
            "full_batch_size": SAMPLE_NUM,
            "kmeans_n_init": 100,
            "kmeans_random_state": seed,
        },
        "rng_audit": rng_audit,
        "backbone_hash_before": (
            prepared["backbone_hash_before"]
        ),
        "backbone_hash_after": backbone_hash_after,
        "private_z_shape": list(
            prepared["z_stack"].shape
        ),
        "private_z_sha256": prepared["z_hash"],
        "private_z_requires_grad": bool(
            prepared["z_stack"].requires_grad
        ),
        "private_z_grad_exists": bool(
            prepared["z_stack"].grad is not None
        ),
        "utility_shape": list(
            prepared["utility_tensor"].shape
        ),
        "utility_sha256": prepared["utility_sha256"],
        "utility_source": _display(
            prepared["utility_source"]
        ),
        "utility_requires_grad": bool(
            prepared["utility_tensor"].requires_grad
        ),
        "utility_grad_exists": bool(
            prepared["utility_tensor"].grad is not None
        ),
        "initial_state_dict": _display(
            output_root / "initial_state_dict.pt"
        ),
        "initial_state_sha256": initial_state_hash,
        "formal_wq1a0_mask_path": _display(
            prepared["formal_wq1a0_mask_path"]
        ),
        "formal_wq1a0_result_path": _display(
            prepared["formal_wq1a0_result_path"]
        ),
        "canonical_manifest_sha256_before": (
            manifest_hash_before
        ),
        "canonical_manifest_sha256_after": (
            manifest_hash_after
        ),
        "approved_pilot_summary": (
            None
            if pilot_reference_path is None
            else _display(pilot_reference_path)
        ),
        "approved_pilot_summary_sha256_before": (
            pilot_reference_hash_before
        ),
        "approved_pilot_summary_sha256_after": (
            pilot_reference_hash_after
        ),
        "multiseed_scientific_conclusion_performed": False,

        "canonical_base_metrics": canonical_base,
        "uniform_shared_metrics": uniform_metrics,
        "oracle_clean_shared_metrics": oracle_metrics,
        "correct_u_shared_metrics": correct_metrics,
        "uniform_shared": _record_summary(
            records["uniform_shared"]
        ),
        "oracle_clean_shared": _record_summary(
            records["oracle_clean_shared"]
        ),
        "correct_u_shared": _record_summary(
            records["correct_u_shared"]
        ),
        "shuffled_u": [
            {
                "repeat_id": record["repeat_id"],
                "shuffled_utility_sha256": record[
                    "shuffled_utility_sha256"
                ],
                **_record_summary(record),
            }
            for record in shuffled_records
        ],
        "delta_oracle_vs_uniform": _metric_delta(
            oracle_metrics,
            uniform_metrics,
        ),
        "delta_correct_vs_uniform": _metric_delta(
            correct_metrics,
            uniform_metrics,
        ),
        "delta_correct_vs_oracle": _metric_delta(
            correct_metrics,
            oracle_metrics,
        ),
        "delta_uniform_vs_canonical_base": _metric_delta(
            uniform_metrics,
            canonical_base,
        ),
        "delta_oracle_vs_canonical_base": _metric_delta(
            oracle_metrics,
            canonical_base,
        ),
        "delta_correct_vs_canonical_base": _metric_delta(
            correct_metrics,
            canonical_base,
        ),
        "oracle_minus_correct_acc": float(
            oracle_metrics["acc"]
            - correct_metrics["acc"]
        ),
        "shuffled_acc_summary": shuffled_acc_summary,
        "shuffled_nmi_summary": shuffled_nmi_summary,
        "shuffled_ari_summary": shuffled_ari_summary,
        "formal_shuffled_metric_artifacts": (
            {
                metric: _display(
                    output_root / ("shuffled_" + metric + ".npy")
                )
                for metric in ("acc", "nmi", "ari")
            }
            if mode == "formal"
            else None
        ),
        "correct_u_acc_percentile_in_pilot_null": (
            None
            if mode == "formal"
            else float(
                100.0
                * np.mean(
                    np.asarray(
                        shuffled_acc_summary["values"]
                    )
                    < correct_metrics["acc"]
                )
            )
        ),
        "correct_u_acc_percentile_in_formal_null": (
            float(
                100.0
                * np.mean(
                    np.asarray(
                        shuffled_acc_summary["values"]
                    )
                    < correct_metrics["acc"]
                )
            )
            if mode == "formal"
            else None
        ),
        "pilot_null_screening_only": mode != "formal",
        "formal_null_primary_metric": (
            "acc" if mode == "formal" else None
        ),
        "admission_purity": purity,
        "collapse_diagnostics_summary": {
            arm: records[arm]["collapse_diagnostics"]
            for arm in PRIMARY_ARMS
        },
        "projector_initial_hashes": {
            record["arm_dir"].relative_to(
                output_root
            ).as_posix(): record["initial_state_sha256"]
            for record in all_records
        },
        "corruption_mask_used_by_uniform": False,
        "corruption_mask_used_by_correct_u": False,
        "corruption_mask_used_by_shuffled_u": False,
        "corruption_mask_used_by_oracle": True,
        "labels_loaded_before_training_complete": False,
        "labels_used_for_training": False,
        "labels_used_for_admission": False,
        "labels_used_for_model_selection": False,
        "labels_used_for_evaluation": True,
        **formal_summary_fields,
        **gates,
    }
    summary_name = (
        "b6_wq1a1_seed20_formal_summary.json"
        if mode == "formal"
        else "b6_wq1a1_seed20_summary.json"
    )
    _write_json(output_root / summary_name, summary)
    return summary


def _print_engineering_gates(summary):
    for key in (
        "B6_WQ1A1_BACKBONE_FROZEN_PASS",
        "B6_WQ1A1_WQ1A0_ADMISSION_REPRO_PASS",
        "B6_WQ1A1_UTILITY_DETACHED_PASS",
        "B6_WQ1A1_PROJECTOR_INIT_MATCH_PASS",
        "B6_WQ1A1_GRADIENT_PATH_PASS",
        "B6_WQ1A1_NO_LABEL_LEAKAGE_PASS",
        "B6_WQ1A1_ADMISSION_COUNT_PASS",
        "B6_WQ1A1_ORACLE_MASK_PASS",
        "B6_WQ1A1_ALL_FINITE_PASS",
        "B6_WQ1A1_SMOKE_DETERMINISM_PASS",
        "B6_WQ1A1_SMOKE_COMPLETE",
    ):
        value = summary[key]
        if isinstance(value, bool):
            value = str(value).lower()
        print(key + "=" + str(value))


def _print_pilot_gates(summary):
    for key in (
        "B6_WQ1A1_CARRIER_VIABILITY_PASS",
        "B6_WQ1A1_UTILITY_DIRECTION_PASS",
        "B6_WQ1A1_PILOT_NULL_DIRECTION_PASS",
        "B6_WQ1A1_PILOT_PROGRESSION_ELIGIBLE",
    ):
        value = summary[key]
        if isinstance(value, bool):
            value = str(value).lower()
        print(key + "=" + str(value))


def _print_formal_gates(summary):
    for key in (
        "B6_WQ1A1_CARRIER_VIABILITY_PASS",
        "B6_WQ1A1_UTILITY_DIRECTION_PASS",
        "B6_WQ1A1_NO_COLLAPSE_PASS",
        "B6_WQ1A1_BACKBONE_FROZEN_PASS",
        "B6_WQ1A1_WQ1A0_ADMISSION_REPRO_PASS",
        "B6_WQ1A1_PROJECTOR_INIT_MATCH_PASS",
        "B6_WQ1A1_GRADIENT_PATH_PASS",
        "B6_WQ1A1_NO_LABEL_LEAKAGE_PASS",
        "B6_WQ1A1_FORMAL_NULL_PASS",
        "B6_WQ1A1_FORMAL_SEED20_PASS",
        "B6_WQ1A1_MULTISEED_ENTRY_ELIGIBLE",
    ):
        value = summary[key]
        if isinstance(value, bool):
            value = str(value).lower()
        print(key + "=" + str(value))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot", "formal"),
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        _require(
            args.mode == "formal",
            "--dry-run is supported only with --mode formal",
        )
        summary = run_formal_dry_validation(seed=args.seed)
        value = summary[
            "B6_WQ1A1_FORMAL_DRY_VALIDATION_PASS"
        ]
        print(
            "B6_WQ1A1_FORMAL_DRY_VALIDATION_PASS="
            + str(value).lower()
        )
        return summary

    summary = run_experiment(
        mode=args.mode,
        seed=args.seed,
    )
    _print_engineering_gates(summary)
    if args.mode == "pilot":
        _print_pilot_gates(summary)
    elif args.mode == "formal":
        _print_formal_gates(summary)
    return summary


if __name__ == "__main__":
    main()
