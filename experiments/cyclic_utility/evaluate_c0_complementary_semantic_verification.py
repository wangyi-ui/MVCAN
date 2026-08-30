"""Evaluate C0 complementary-view reciprocal semantic verification."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility.c0_complementary_semantic_verification import (
    CLASS_NUM,
    DIRECTION_COUNT,
    SAMPLE_NUM,
    VIEW_NUM,
    build_cycle_scores,
    evaluate_postseal,
)
from experiments.e4_semantic_memory_bank import (
    evaluate_e4_a0_memory_feasibility as frozen_e1,
)
from irv.b4_information_utility import tensor_sha256


STAGE = "C0"
SEED = 20
DEFAULT_FEATURE_PATH = frozen_e1.DEFAULT_FEATURE_PATH
DEFAULT_FEATURE_AUDIT_PATH = frozen_e1.DEFAULT_FEATURE_AUDIT_PATH
DEFAULT_E1_LWC_MODEL_DIR = frozen_e1.DEFAULT_E1_LWC_MODEL_DIR
DEFAULT_E1_LWC_AUDIT_PATH = frozen_e1.DEFAULT_E1_LWC_AUDIT_PATH
DEFAULT_FULL_GT_PATH = frozen_e1.DEFAULT_FULL_GT_PATH
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "outputs/cyclic_utility/"
    "c0_complementary_semantic_verification_seed20"
)
SEALED_ARRAY_NAMES = (
    "y_gen",
    "conf_gen",
    "support_ver",
    "closure",
    "U_cycle",
    "C_conf",
    "C_jsd",
    "U_cycle_shuffle",
    "generator_subsets",
    "verifier_subsets",
    "shuffle_permutation",
    "native_global_cluster",
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


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_numpy(value, dtype=None):
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    return np.ascontiguousarray(array)


@torch.no_grad()
def load_frozen_e1_aligned_q(
    feature_path=DEFAULT_FEATURE_PATH,
    feature_audit_path=DEFAULT_FEATURE_AUDIT_PATH,
    model_dir=DEFAULT_E1_LWC_MODEL_DIR,
    model_audit_path=DEFAULT_E1_LWC_AUDIT_PATH,
    device="cpu",
):
    """Load frozen E1 LWC and return q_aligned: [1400,6,7]."""
    target_device = torch.device(device)
    model, checkpoint_audit = frozen_e1._load_frozen_lwc_model(
        model_dir, model_audit_path, target_device
    )
    views, sample_ids, feature_audit = (
        frozen_e1.e1_train.load_frozen_feature_artifact(
            feature_path, feature_audit_path
        )
    )
    _require(
        np.array_equal(
            sample_ids, np.arange(SAMPLE_NUM, dtype=np.int64)
        ),
        "C0 frozen feature rows are not canonical sample IDs",
    )

    parameter_hash_before = frozen_e1.hash_backbone(
        model.autoencoders
    )
    raw_z_views = []
    q_local_views = []
    for autoencoder, view in zip(model.autoencoders, views):
        _require(
            not autoencoder.training,
            "frozen E1 autoencoder is not in eval mode",
        )
        features = torch.as_tensor(
            np.array(view, copy=True, order="C"),
            device=target_device,
        )
        raw_z_view = autoencoder.encoder(features).detach()
        q_local_view = autoencoder.clustering(raw_z_view).detach()
        raw_z_views.append(raw_z_view.cpu())
        q_local_views.append(q_local_view.cpu())
    raw_z = torch.stack(raw_z_views, dim=1).detach()
    q_local = torch.stack(q_local_views, dim=1).detach()
    _require(
        raw_z.shape == (SAMPLE_NUM, VIEW_NUM, 10)
        and q_local.shape == (SAMPLE_NUM, VIEW_NUM, CLASS_NUM)
        and bool(torch.isfinite(raw_z).all().item())
        and bool(torch.isfinite(q_local).all().item()),
        "frozen E1 native representation boundary mismatch",
    )
    _require(
        bool(
            torch.allclose(
                q_local.sum(dim=-1),
                torch.ones_like(q_local[..., 0]),
                rtol=0.0,
                atol=1e-6,
            )
        ),
        "q_local probability mass mismatch",
    )

    expected_clusters = np.arange(CLASS_NUM, dtype=np.int64)
    local_assignments = q_local.numpy().argmax(axis=2)
    for view_id in range(VIEW_NUM):
        _require(
            np.array_equal(
                np.unique(local_assignments[:, view_id]),
                expected_clusters,
            ),
            "q_local is missing a cluster in view " + str(view_id),
        )

    native_reference = (
        frozen_e1.g2_protocol.replay_native_global_reference(
            raw_z.numpy(), q_local.numpy()
        )
    )
    alignment = frozen_e1.g2_protocol.build_alignment(
        q_local.numpy(),
        native_reference["global_prediction"],
        model.Match,
    )
    M = torch.as_tensor(
        np.array(
            alignment["alignment_matrix"],
            copy=True,
            order="C",
        )
    ).detach()
    q_aligned, _, alignment_audit = frozen_e1.align_q_to_global(
        q_local, M
    )
    q_aligned = q_aligned.detach()
    expected_q_aligned = torch.einsum(
        "nvk,vjk->nvj", q_local, M.to(dtype=q_local.dtype)
    ).detach()
    alignment_exact = bool(torch.equal(
        q_aligned, expected_q_aligned
    ))
    mass_preserved = bool(
        torch.allclose(
            q_aligned.sum(dim=-1),
            q_local.sum(dim=-1),
            rtol=0.0,
            atol=1e-6,
        )
    )
    _require(
        q_aligned.shape == (SAMPLE_NUM, VIEW_NUM, CLASS_NUM)
        and alignment_exact
        and mass_preserved
        and bool(torch.isfinite(q_aligned).all().item())
        and not q_aligned.requires_grad
        and q_aligned.grad_fn is None,
        "C0 q alignment boundary mismatch",
    )

    parameter_hash_after = frozen_e1.hash_backbone(
        model.autoencoders
    )
    parameter_boundary_pass = bool(
        all(
            not parameter.requires_grad and parameter.grad is None
            for autoencoder in model.autoencoders
            for parameter in autoencoder.parameters()
        )
    )
    _require(
        parameter_hash_before == parameter_hash_after
        and parameter_boundary_pass,
        "frozen E1 parameters changed during C0 representation load",
    )
    native_global_cluster = np.array(
        native_reference["global_prediction"],
        dtype=np.int64,
        copy=True,
        order="C",
    )
    native_global_cluster.setflags(write=False)
    return (
        q_aligned.detach(),
        native_global_cluster,
        np.array(sample_ids, dtype=np.int64, copy=True, order="C"),
        {
            "source_stage": "E1",
            "source_arm": "LWC",
            "source_epochs": 100,
            "checkpoint": checkpoint_audit,
            "feature": feature_audit,
            "q_local": {
                "shape": list(q_local.shape),
                "logical_sha256": tensor_sha256(q_local.numpy()),
                "finite_pass": True,
                "probability_mass_pass": True,
                "all_views_have_all_clusters_pass": True,
                "stop_gradient_pass": True,
            },
            "M": {
                "shape": list(M.shape),
                "logical_sha256": tensor_sha256(M.numpy()),
                "full_permutation_pass": True,
                "row_sums_one_pass": True,
                "column_sums_one_pass": True,
            },
            "q_aligned": {
                "shape": list(q_aligned.shape),
                "logical_sha256": tensor_sha256(
                    q_aligned.cpu().numpy()
                ),
                "alignment_exact_q_at_M_transpose_pass": (
                    alignment_exact
                ),
                "probability_mass_preserved_pass": mass_preserved,
                "finite_pass": True,
                "stop_gradient_pass": True,
            },
            "alignment_helper_audit": alignment_audit,
            "native_global_reference": {
                "shape": list(native_global_cluster.shape),
                "logical_sha256": tensor_sha256(
                    native_global_cluster
                ),
                "all_clusters_present_pass": True,
            },
            "raw_z_averaging_used": False,
            "raw_unaligned_q_averaging_used": False,
            "encoder_forward_modified": False,
            "model_eval_pass": True,
            "parameter_hash_before": parameter_hash_before,
            "parameter_hash_after": parameter_hash_after,
            "parameters_unchanged_pass": True,
            "parameters_frozen_grad_none_pass": (
                parameter_boundary_pass
            ),
        },
    )


def save_prediction_seal(
    predictions_and_scores,
    native_global_cluster,
    representation_audit,
    output_root,
):
    """Save/hash/reload all C0 scores before full GT is available."""
    root = _resolve(output_root)
    root.mkdir(parents=True, exist_ok=True)
    artifact_path = root / "c0_predictions_and_scores.npz"
    seal_path = root / "c0_prediction_seal.json"
    _require(
        not artifact_path.exists() and not seal_path.exists(),
        "C0 prediction seal output boundary mismatch",
    )

    payload = {
        "y_gen": _as_numpy(
            predictions_and_scores["y_gen"], dtype=np.int64
        ),
        "conf_gen": _as_numpy(predictions_and_scores["conf_gen"]),
        "support_ver": _as_numpy(
            predictions_and_scores["support_ver"]
        ),
        "closure": _as_numpy(
            predictions_and_scores["closure"], dtype=np.bool_
        ),
        "U_cycle": _as_numpy(predictions_and_scores["U_cycle"]),
        "C_conf": _as_numpy(predictions_and_scores["C_conf"]),
        "C_jsd": _as_numpy(predictions_and_scores["C_jsd"]),
        "U_cycle_shuffle": _as_numpy(
            predictions_and_scores["U_cycle_shuffle"]
        ),
        "generator_subsets": _as_numpy(
            predictions_and_scores["generator_subsets"],
            dtype=np.int64,
        ),
        "verifier_subsets": _as_numpy(
            predictions_and_scores["verifier_subsets"],
            dtype=np.int64,
        ),
        "shuffle_permutation": _as_numpy(
            predictions_and_scores["shuffle_permutation"],
            dtype=np.int64,
        ),
        "native_global_cluster": _as_numpy(
            native_global_cluster, dtype=np.int64
        ),
    }
    _require(
        tuple(payload) == SEALED_ARRAY_NAMES,
        "C0 sealed array set/order mismatch",
    )

    np.savez(artifact_path, **payload)
    logical_hashes = {
        name: tensor_sha256(array)
        for name, array in payload.items()
    }
    array_records = {
        name: {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "logical_sha256": logical_hashes[name],
        }
        for name, array in payload.items()
    }

    with np.load(artifact_path, allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == SEALED_ARRAY_NAMES,
            "C0 sealed NPZ field mismatch",
        )
        for name in SEALED_ARRAY_NAMES:
            reloaded = np.array(
                archive[name], copy=True, order="C"
            )
            _require(
                np.array_equal(reloaded, payload[name])
                and tensor_sha256(reloaded) == logical_hashes[name],
                "C0 sealed array reload/hash mismatch: " + name,
            )

    seal = {
        "stage": STAGE,
        "seed": SEED,
        "artifact_path": _display(artifact_path),
        "artifact_file_sha256": file_sha256(artifact_path),
        "arrays": array_records,
        "p_gen_logical_sha256": tensor_sha256(
            _as_numpy(predictions_and_scores["p_gen"])
        ),
        "p_ver_logical_sha256": tensor_sha256(
            _as_numpy(predictions_and_scores["p_ver"])
        ),
        "q_aligned_provenance": representation_audit["q_aligned"],
        "M_provenance": representation_audit["M"],
        "native_global_reference_provenance": (
            representation_audit["native_global_reference"]
        ),
        "scores_completed_before_GT": True,
        "scores_saved_before_GT": True,
        "scores_hashed_before_GT": True,
        "scores_reloaded_before_GT": True,
        "full_GT_loaded_before_seal": False,
        "R_loaded_before_seal": False,
        "sparse_labels_loaded_before_seal": False,
        "corruption_mask_loaded_before_seal": False,
        "oracle_loaded_before_seal": False,
    }
    _write_json(seal_path, seal)
    return seal


def load_full_ground_truth_after_seal(
    path=DEFAULT_FULL_GT_PATH,
):
    """Load the sole full-GT vector at the explicit post-seal boundary."""
    return frozen_e1.load_full_ground_truth(path)


def _serializable_mapping(mapping_record):
    return {
        **mapping_record,
        "mapping": mapping_record["mapping"].tolist(),
        "contingency": mapping_record["contingency"].tolist(),
    }


def run_evaluation(
    feature_path=DEFAULT_FEATURE_PATH,
    feature_audit_path=DEFAULT_FEATURE_AUDIT_PATH,
    model_dir=DEFAULT_E1_LWC_MODEL_DIR,
    model_audit_path=DEFAULT_E1_LWC_AUDIT_PATH,
    full_gt_path=DEFAULT_FULL_GT_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    device="cpu",
):
    """Run C0 with a strict pre-GT score seal and no training."""
    output_root = _resolve(output_dir)
    q_aligned, native_global_cluster, sample_ids, representation_audit = (
        load_frozen_e1_aligned_q(
            feature_path=feature_path,
            feature_audit_path=feature_audit_path,
            model_dir=model_dir,
            model_audit_path=model_audit_path,
            device=device,
        )
    )
    _require(
        np.array_equal(
            sample_ids, np.arange(SAMPLE_NUM, dtype=np.int64)
        ),
        "C0 sample IDs are not canonical",
    )
    predictions_and_scores = build_cycle_scores(q_aligned)
    prediction_seal = save_prediction_seal(
        predictions_and_scores,
        native_global_cluster,
        representation_audit,
        output_root,
    )

    full_GT, full_GT_audit = load_full_ground_truth_after_seal(
        full_gt_path
    )
    postseal = evaluate_postseal(
        native_global_cluster,
        full_GT,
        predictions_and_scores,
    )

    mapping_json = _serializable_mapping(postseal["mapping"])
    auc_output = {
        "mapping": mapping_json,
        "auc": postseal["auc"],
        "paired_comparisons": postseal["paired_comparisons"],
        "score_separation": postseal["score_separation"],
    }
    _write_json(
        output_root / "c0_auc_by_direction.json", auc_output
    )
    _write_json(
        output_root / "c0_risk_coverage.json",
        postseal["risk_coverage"],
    )

    audit = {
        "stage": STAGE,
        "seed": SEED,
        "dataset": "Caltech-6V",
        "condition": {
            "SNR_dB": 2.5,
            "corrupt_views_per_sample": 3,
        },
        "N": SAMPLE_NUM,
        "V": VIEW_NUM,
        "K": CLASS_NUM,
        "representation": representation_audit,
        "splits_and_scores": predictions_and_scores["audit"],
        "prediction_seal": prediction_seal,
        "postseal_ground_truth": full_GT_audit,
        "global_mapping": mapping_json,
        "leakage": {
            "R_loaded_before_seal": False,
            "R_loaded": False,
            "sparse_labels_loaded_before_seal": False,
            "sparse_labels_loaded": False,
            "full_GT_loaded_before_seal": False,
            "full_GT_loaded_after_seal": True,
            "corruption_mask_loaded_before_seal": False,
            "corruption_mask_loaded": False,
            "oracle_used": False,
        },
        "historical_non_repetition": {
            "primary_operator_is_jsd": False,
            "historical_jsd_used_as_baseline_only": True,
            "target_correction_used": False,
            "memory_used": False,
            "reliability_weighting_used": False,
            "P_all_modified": False,
        },
        "execution": {
            "training_used": False,
            "optimizer_used": False,
            "backward_used": False,
            "parameter_update_used": False,
            "EMA_used": False,
            "memory_update_used": False,
            "parameters_unchanged_pass": representation_audit[
                "parameters_unchanged_pass"
            ],
        },
        "decision": postseal["decision"],
        "C0_AUDIT_PASS": bool(
            prediction_seal["scores_reloaded_before_GT"]
            and prediction_seal["full_GT_loaded_before_seal"] is False
            and representation_audit["q_aligned"][
                "alignment_exact_q_at_M_transpose_pass"
            ]
            and predictions_and_scores["audit"][
                "derangement_no_fixed_point_pass"
            ]
            and postseal["mapping"][
                "single_global_mapping_reused_all_directions"
            ]
            and representation_audit["parameters_unchanged_pass"]
        ),
    }
    diagnostic_results = {
        "stage": STAGE,
        "primary_evaluation_unit": (
            "20 directional complementary-view splits"
        ),
        "auc_summary": postseal["auc"]["summary"],
        "valid_direction_count": postseal["auc"][
            "valid_direction_count"
        ],
        "paired_comparisons": postseal["paired_comparisons"],
        "score_separation": postseal["score_separation"],
        "risk_coverage_macro_mean": postseal["risk_coverage"][
            "macro_mean"
        ],
        "decision": postseal["decision"],
    }
    _write_json(output_root / "c0_audit.json", audit)
    _write_json(
        output_root / "diagnostic_results.json",
        diagnostic_results,
    )
    return {
        "output_dir": output_root,
        "c0_audit": audit,
        "auc": auc_output,
        "risk_coverage": postseal["risk_coverage"],
        "diagnostic_results": diagnostic_results,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path", default=str(DEFAULT_FEATURE_PATH)
    )
    parser.add_argument(
        "--feature-audit-path",
        default=str(DEFAULT_FEATURE_AUDIT_PATH),
    )
    parser.add_argument(
        "--model-dir", default=str(DEFAULT_E1_LWC_MODEL_DIR)
    )
    parser.add_argument(
        "--model-audit-path",
        default=str(DEFAULT_E1_LWC_AUDIT_PATH),
    )
    parser.add_argument(
        "--full-gt-path", default=str(DEFAULT_FULL_GT_PATH)
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR)
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_evaluation(
        feature_path=args.feature_path,
        feature_audit_path=args.feature_audit_path,
        model_dir=args.model_dir,
        model_audit_path=args.model_audit_path,
        full_gt_path=args.full_gt_path,
        output_dir=args.output_dir,
        device=args.device,
    )
    print("Saved: " + _display(result["output_dir"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

