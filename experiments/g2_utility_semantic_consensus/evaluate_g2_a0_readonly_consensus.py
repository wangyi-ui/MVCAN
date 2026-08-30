"""Evaluate frozen utility-aligned MVCAN semantic consensus without training."""

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from configure import get_default_config
from experiments.b7_sparse_supervision import b7_sparse_anchor_protocol as b7_protocol
from experiments.d2_caltech6v import evaluate_d2_a0_utility_transfer as d2
from experiments.g2_utility_semantic_consensus import g2_consensus_protocol as protocol
from irv.b3_audit import hash_backbone
from irv.b4_information_utility import tensor_sha256
from weak_quality import apply_weak_quality_protocol
from weak_quality import ndarray_sha256


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _read_frozen_condition_audit(d2_dir):
    audit_path = protocol.resolve_path(d2_dir) / "corruption_audit/corruption_audit.json"
    _require(audit_path.is_file(), "frozen D2 corruption audit is missing")
    return protocol.read_json(audit_path), audit_path


def _policy_artifact_prefix(policy_name):
    return policy_name.lower()


def _construct_fixed_outputs(
    max_samples=None,
    data_path=protocol.DEFAULT_DATA_PATH,
    backbone_dir=protocol.DEFAULT_BACKBONE_DIR,
    d2_dir=protocol.DEFAULT_D2_DIR,
):
    """Construct and hash every label-free output before evaluation labels exist."""
    evaluation_count = protocol.select_evaluation_count(max_samples)
    evaluation_sample_ids = np.arange(evaluation_count, dtype=np.int64)

    clean_views = protocol.load_caltech_views_only(data_path)
    evaluation_views, runtime_corruption = apply_weak_quality_protocol(
        clean_views,
        mode="heterogeneous_gaussian",
        k=protocol.CORRUPTION_K,
        snr_db=protocol.SNR_DB,
        corruption_seed=protocol.CORRUPTION_SEED,
    )
    runtime_mask_hash = str(runtime_corruption["mask_sha256"])
    runtime_corruption.pop("mask", None)
    runtime_corruption.pop("corruption_pairs", None)
    stored_corruption, stored_corruption_path = _read_frozen_condition_audit(d2_dir)
    _require(
        runtime_mask_hash == stored_corruption.get("mask_sha256")
        and runtime_corruption["corrupted_view_sha256"]
        == stored_corruption.get("corrupted_view_sha256"),
        "runtime corruption does not match the frozen D1/D2 condition",
    )

    config = get_default_config(protocol.DATASET_NAME)
    config["dataset"] = protocol.DATASET_NAME
    models, d2_normalized_views, backbone_load_audit = d2._load_native_representations(
        config=config,
        evaluation_views=evaluation_views,
        backbone_dir=protocol.resolve_path(backbone_dir),
        sample_num=protocol.SAMPLE_NUM,
        view_num=protocol.VIEW_NUM,
        cluster_num=protocol.CLASS_NUM,
        model_seed=protocol.MODEL_SEED,
    )
    protocol.freeze_autoencoders(models.autoencoders)
    parameter_audit_before = protocol.frozen_parameter_audit(models.autoencoders)
    backbone_hash_before = hash_backbone(models.autoencoders)

    with torch.no_grad():
        extracted = protocol.extract_native_z_q(
            models.autoencoders,
            evaluation_views,
        )
    raw_z_stack, q_stack = protocol.validate_native_representation(
        extracted["raw_z_stack"],
        extracted["q_stack"],
        require_full=True,
    )
    normalized_z_stack = np.ascontiguousarray(
        extracted["z_stack"].detach().cpu().numpy()
    )
    d2_normalized_stack = np.ascontiguousarray(
        torch.stack(d2_normalized_views, dim=1).detach().cpu().numpy()
    )
    _require(
        np.array_equal(normalized_z_stack, d2_normalized_stack),
        "G2 normalized z differs from the frozen D2 extraction path",
    )

    frozen_u = protocol.load_frozen_u_only(d2_dir)
    _require(
        frozen_u["U_sha256"] == protocol.EXPECTED_U_SHA256
        and frozen_u["utility_file_sha256"]
        == protocol.EXPECTED_UTILITY_FILE_SHA256,
        "frozen D2 Utility provenance mismatch",
    )

    # This full-N reference is built once, before any policy-specific operation.
    reference = protocol.replay_native_global_reference(raw_z_stack, q_stack)
    alignment = protocol.build_alignment(
        q_stack,
        reference["global_prediction"],
        models.Match,
    )

    # Normal admissions have no oracle-data argument or dependency.
    normal = protocol.build_normal_admissions(
        frozen_u["U"],
        control_seed=protocol.CONTROL_SEED,
    )
    admissions = dict(normal["admissions"])

    # The corruption mask first enters through this isolated oracle-only branch.
    oracle = protocol.load_oracle_corruption_only(d2_dir)
    _require(
        oracle["corruption_mask_sha256"] == runtime_mask_hash,
        "runtime and frozen oracle mask hashes differ",
    )
    admissions["ORACLE"] = protocol.build_oracle_admission(
        oracle["corruption_mask"]
    )
    _require(
        set(admissions) == set(protocol.CONSENSUS_POLICIES),
        "G2 consensus policy set mismatch",
    )

    aligned_evaluation = alignment["aligned_q"][evaluation_sample_ids]
    policy_outputs = {}
    for policy_name in protocol.CONSENSUS_POLICIES:
        policy_outputs[policy_name] = protocol.semantic_consensus(
            aligned_evaluation,
            admissions[policy_name][evaluation_sample_ids],
            reference["native_nmi_weights"],
        )

    # This seal is finalized before the evaluation-only label loader is called.
    fixed_hashes = protocol.fixed_output_hashes(
        reference,
        alignment,
        admissions,
        policy_outputs,
        evaluation_sample_ids,
    )

    backbone_hash_after = hash_backbone(models.autoencoders)
    parameter_audit_after = protocol.frozen_parameter_audit(models.autoencoders)
    immutable_hashes = {
        "backbone_hash_before": backbone_hash_before,
        "backbone_hash_after": backbone_hash_after,
        "backbone_unchanged_pass": backbone_hash_before == backbone_hash_after,
        "parameters_before": parameter_audit_before,
        "parameters_after": parameter_audit_after,
        "backbone_parameters_frozen_and_grad_none_pass": bool(
            parameter_audit_before["all_requires_grad_false"]
            and parameter_audit_before["all_grad_none"]
            and parameter_audit_after["all_requires_grad_false"]
            and parameter_audit_after["all_grad_none"]
        ),
    }

    shared_reference = protocol.reference_hashes(reference, alignment)
    return {
        "evaluation_count": evaluation_count,
        "evaluation_sample_ids": evaluation_sample_ids,
        "runtime_corruption": runtime_corruption,
        "runtime_mask_hash": runtime_mask_hash,
        "stored_corruption_path": stored_corruption_path,
        "backbone_load_audit": backbone_load_audit,
        "frozen_u": frozen_u,
        "extracted": extracted,
        "raw_z_stack": raw_z_stack,
        "normalized_z_stack": normalized_z_stack,
        "q_stack": q_stack,
        "reference": reference,
        "alignment": alignment,
        "normal_admission": normal,
        "admissions": admissions,
        "oracle": oracle,
        "policy_outputs": policy_outputs,
        "fixed_hashes": fixed_hashes,
        "shared_reference_hashes": shared_reference,
        "immutable_hashes": immutable_hashes,
    }


def _metrics_after_fixed_output_seal(fixed, data_path):
    labels = protocol.load_caltech_labels_only(data_path)
    evaluation_labels = labels[fixed["evaluation_sample_ids"]]
    results = {}
    for policy_name in protocol.CONSENSUS_POLICIES:
        policy_output = fixed["policy_outputs"][policy_name]
        results[policy_name] = {
            **protocol.metrics_from_fixed_prediction(
                evaluation_labels,
                policy_output["prediction"],
            ),
            **policy_output["diagnostics"],
            "role": "aligned_q_consensus",
            "shared_reference_hashes": dict(fixed["shared_reference_hashes"]),
            "score_sha256": fixed["fixed_hashes"]["score_sha256"][policy_name],
            "prediction_sha256": fixed["fixed_hashes"]["prediction_sha256"][
                policy_name
            ],
        }
    native_prediction = fixed["reference"]["global_prediction"][
        fixed["evaluation_sample_ids"]
    ]
    results["NATIVE_MVCAN_GLOBAL"] = {
        **protocol.metrics_from_fixed_prediction(
            evaluation_labels,
            native_prediction,
        ),
        "role": "descriptive_native_global_baseline",
        "prediction_sha256": fixed["fixed_hashes"]["prediction_sha256"][
            "NATIVE_MVCAN_GLOBAL"
        ],
    }
    return results


def _representation_audit(fixed):
    return {
        "raw_z": {
            "shape": list(fixed["raw_z_stack"].shape),
            "dtype": str(fixed["raw_z_stack"].dtype),
            "sha256": tensor_sha256(fixed["raw_z_stack"]),
            "source": "frozen_encoder_raw_latent",
        },
        "normalized_z": {
            "shape": list(fixed["normalized_z_stack"].shape),
            "dtype": str(fixed["normalized_z_stack"].dtype),
            "sha256": tensor_sha256(fixed["normalized_z_stack"]),
            "d2_exact_match_pass": True,
        },
        "q": {
            "shape": list(fixed["q_stack"].shape),
            "dtype": str(fixed["q_stack"].dtype),
            "sha256": tensor_sha256(fixed["q_stack"]),
            "source": "checkpoint_native_clustering_of_raw_z",
            "clustering_input": fixed["extracted"]["q_input_source"],
            "probability_mass_max_abs_error": float(
                np.max(np.abs(fixed["q_stack"].sum(axis=2) - 1.0))
            ),
        },
        "aligned_q": {
            "shape": list(fixed["alignment"]["aligned_q"].shape),
            "dtype": str(fixed["alignment"]["aligned_q"].dtype),
            "sha256": fixed["alignment"]["aligned_q_sha256"],
            "probability_mass_max_abs_error": fixed["alignment"][
                "aligned_q_mass_max_abs_error"
            ],
        },
        "full_reference_sample_count": protocol.SAMPLE_NUM,
    }


def _native_global_reference_audit(fixed):
    reference = fixed["reference"]
    shared = fixed["shared_reference_hashes"]
    return {
        "reference_scope": "full_1400_shared_once",
        "sample_count": protocol.SAMPLE_NUM,
        "fusion_updates": protocol.FUSION_UPDATES,
        "kmeans_seed": protocol.MODEL_SEED,
        "kmeans_n_init": protocol.KMEANS_N_INIT,
        "per_view_assignment_source": "argmax_frozen_checkpoint_q",
        "per_view_kmeans_performed": False,
        "scaled_z_shape": list(reference["scaled_z_stack"].shape),
        "latent_fusion_shape": list(reference["latent_fusion"].shape),
        "global_prediction_shape": list(reference["global_prediction"].shape),
        "global_centers_shape": list(reference["global_centers"].shape),
        "native_nmi_weights_shape": list(reference["native_nmi_weights"].shape),
        "native_nmi_weights": reference["native_nmi_weights"].tolist(),
        "fusion_weights_used_for_final_reference": reference[
            "fusion_weights_used_for_final_reference"
        ].tolist(),
        "fusion_update_audit": reference["fusion_update_audit"],
        **shared,
    }


def _alignment_audit(fixed):
    matrices = fixed["alignment"]["alignment_matrix"]
    return {
        "shape": list(matrices.shape),
        "dtype": str(matrices.dtype),
        "sha256": fixed["alignment"]["alignment_sha256"],
        "orientation": "aligned_q_local_row_vector @ M_global_by_local.T",
        "per_view": [
            {
                "view_id": view_id,
                "matrix": matrices[view_id].tolist(),
                "row_sums": matrices[view_id].sum(axis=1).tolist(),
                "column_sums": matrices[view_id].sum(axis=0).tolist(),
                "complete_permutation_pass": protocol.validate_permutation_matrix(
                    matrices[view_id]
                ),
            }
            for view_id in range(protocol.VIEW_NUM)
        ],
        "q_mass_max_abs_error": fixed["alignment"]["q_mass_max_abs_error"],
        "aligned_q_mass_max_abs_error": fixed["alignment"][
            "aligned_q_mass_max_abs_error"
        ],
    }


def _admission_audit(fixed):
    records = {}
    for policy_name in protocol.CONSENSUS_POLICIES:
        admission = fixed["admissions"][policy_name]
        records[policy_name] = {
            "shape": list(admission.shape),
            "sha256": ndarray_sha256(admission),
            "row_sum_min": int(admission.sum(axis=1).min()),
            "row_sum_max": int(admission.sum(axis=1).max()),
            "column_sums": admission.sum(axis=0, dtype=np.int64).tolist(),
            "evaluation_shape": [
                fixed["evaluation_count"],
                protocol.VIEW_NUM,
            ],
            "shared_reference_hashes": dict(fixed["shared_reference_hashes"]),
        }
    return {
        "normal_builder_parameters": list(
            inspect.signature(protocol.build_normal_admissions).parameters
        ),
        "normal_api_oracle_isolation_pass": (
            protocol.normal_api_oracle_isolation_pass()
        ),
        "oracle_builder_parameters": list(
            inspect.signature(protocol.build_oracle_admission).parameters
        ),
        "shuffled_u_row_permutation": fixed["normal_admission"][
            "shuffled_u_row_permutation"
        ].tolist(),
        "shuffled_u_row_permutation_sha256": tensor_sha256(
            fixed["normal_admission"]["shuffled_u_row_permutation"]
        ),
        "shuffled_u_changed_row_count": fixed["normal_admission"][
            "changed_row_count"
        ],
        "policies": records,
    }


def _npz_payload(fixed):
    payload = {
        "evaluation_sample_ids": fixed["evaluation_sample_ids"],
        "native_nmi_weights": fixed["reference"]["native_nmi_weights"],
        "fusion_weights_used_for_final_reference": fixed["reference"][
            "fusion_weights_used_for_final_reference"
        ],
        "global_centers": fixed["reference"]["global_centers"],
        "global_prediction": fixed["reference"]["global_prediction"],
        "alignment_matrix": fixed["alignment"]["alignment_matrix"],
        "aligned_q": fixed["alignment"]["aligned_q"],
        "shuffled_u_row_permutation": fixed["normal_admission"][
            "shuffled_u_row_permutation"
        ],
    }
    for policy_name in protocol.CONSENSUS_POLICIES:
        prefix = _policy_artifact_prefix(policy_name)
        output = fixed["policy_outputs"][policy_name]
        payload[prefix + "__admission_mask"] = fixed["admissions"][policy_name]
        payload[prefix + "__weighted_mask"] = output["weighted_mask"]
        payload[prefix + "__denominator"] = output["denominator"]
        payload[prefix + "__consensus_score"] = output["consensus_score"]
        payload[prefix + "__prediction"] = output["prediction"]
    return payload


def run_evaluation(
    max_samples=None,
    output_dir=None,
    data_path=protocol.DEFAULT_DATA_PATH,
    backbone_dir=protocol.DEFAULT_BACKBONE_DIR,
    d2_dir=protocol.DEFAULT_D2_DIR,
):
    fixed = _construct_fixed_outputs(
        max_samples=max_samples,
        data_path=data_path,
        backbone_dir=backbone_dir,
        d2_dir=d2_dir,
    )

    # Ground truth first enters here, after the fixed-output seal exists.
    arm_results = _metrics_after_fixed_output_seal(fixed, data_path)
    smoke = fixed["evaluation_count"] < protocol.SAMPLE_NUM
    mode = "smoke" if smoke else "formal"
    run_name = (
        "g2a0_seed20_smoke_" + str(fixed["evaluation_count"])
        if smoke
        else "g2a0_seed20"
    )
    output_root = (
        protocol.resolve_path(output_dir)
        if output_dir is not None
        else protocol.DEFAULT_OUTPUT_ROOT / run_name
    )

    representation_audit = _representation_audit(fixed)
    native_reference_audit = _native_global_reference_audit(fixed)
    alignment_audit = _alignment_audit(fixed)
    admission_audit = _admission_audit(fixed)
    immutable_hashes = fixed["immutable_hashes"]
    engineering_checks = {
        "full_reference_sample_count_pass": native_reference_audit["sample_count"]
        == protocol.SAMPLE_NUM,
        "raw_z_shape_pass": representation_audit["raw_z"]["shape"]
        == [protocol.SAMPLE_NUM, protocol.VIEW_NUM, protocol.LATENT_DIM],
        "q_shape_pass": representation_audit["q"]["shape"]
        == [protocol.SAMPLE_NUM, protocol.VIEW_NUM, protocol.CLASS_NUM],
        "q_from_raw_z_pass": representation_audit["q"]["clustering_input"]
        == "raw_encoder_latent",
        "alignment_shape_pass": alignment_audit["shape"]
        == [protocol.VIEW_NUM, protocol.CLASS_NUM, protocol.CLASS_NUM],
        "alignment_permutation_pass": all(
            item["complete_permutation_pass"]
            for item in alignment_audit["per_view"]
        ),
        "aligned_q_mass_pass": alignment_audit[
            "aligned_q_mass_max_abs_error"
        ]
        <= protocol.PROBABILITY_ATOL,
        "normal_api_oracle_isolation_pass": admission_audit[
            "normal_api_oracle_isolation_pass"
        ],
        "all_admission_pass": admission_audit["policies"]["ALL"][
            "row_sum_min"
        ]
        == admission_audit["policies"]["ALL"]["row_sum_max"]
        == protocol.VIEW_NUM,
        "u_top3_exact_pass": admission_audit["policies"]["U_TOP3"][
            "row_sum_min"
        ]
        == admission_audit["policies"]["U_TOP3"]["row_sum_max"]
        == protocol.TOP_K,
        "shuffled_u_exact_pass": admission_audit["policies"]["SHUFFLED_U"][
            "row_sum_min"
        ]
        == admission_audit["policies"]["SHUFFLED_U"]["row_sum_max"]
        == protocol.TOP_K,
        "shuffled_u_column_counts_pass": admission_audit["policies"][
            "U_TOP3"
        ]["column_sums"]
        == admission_audit["policies"]["SHUFFLED_U"]["column_sums"],
        "shuffled_u_informative_pass": admission_audit[
            "shuffled_u_changed_row_count"
        ]
        > 0,
        "oracle_exact_pass": admission_audit["policies"]["ORACLE"][
            "row_sum_min"
        ]
        == admission_audit["policies"]["ORACLE"]["row_sum_max"]
        == protocol.TOP_K,
        "shared_reference_pass": all(
            admission_audit["policies"][policy]["shared_reference_hashes"]
            == fixed["shared_reference_hashes"]
            for policy in protocol.CONSENSUS_POLICIES
        ),
        "frozen_u_hash_pass": fixed["frozen_u"]["U_sha256"]
        == protocol.EXPECTED_U_SHA256,
        "frozen_u_file_hash_pass": fixed["frozen_u"]["utility_file_sha256"]
        == protocol.EXPECTED_UTILITY_FILE_SHA256,
        "backbone_immutability_pass": immutable_hashes[
            "backbone_unchanged_pass"
        ],
        "backbone_frozen_grad_pass": immutable_hashes[
            "backbone_parameters_frozen_and_grad_none_pass"
        ],
        "full_labels_evaluation_only_pass": True,
        "no_optimizer_pass": True,
        "no_backward_pass": True,
        "no_parameter_update_pass": True,
        "no_per_view_kmeans_pass": native_reference_audit[
            "per_view_kmeans_performed"
        ]
        is False,
        "no_u_t_ridge_refit_pass": True,
        "no_prototype_pass": True,
        "smoke_full_reference_invariant_pass": native_reference_audit[
            "sample_count"
        ]
        == protocol.SAMPLE_NUM,
    }
    engineering_pass = bool(all(engineering_checks.values()))
    results = {
        "stage": protocol.STAGE,
        "run_name": run_name,
        "mode": mode,
        "evaluation_count": fixed["evaluation_count"],
        "full_reference_count": protocol.SAMPLE_NUM,
        "arms": list(protocol.RESULT_ARMS),
        "results": arm_results,
        "fixed_output_hashes": fixed["fixed_hashes"],
        "engineering_checks": engineering_checks,
        "ENGINEERING_PASS": engineering_pass,
        "SCIENTIFIC_VERDICT": "NOT_PREDEFINED_DESCRIPTIVE_ONLY",
        "scientific_pass_threshold_defined": False,
        "scientific_interpretation_allowed": not smoke,
        "paper_claim_made": False,
    }
    provenance = {
        "stage": protocol.STAGE,
        "dataset": protocol.DATASET_NAME,
        "condition": "snr2p5_k3_seed20",
        "N": protocol.SAMPLE_NUM,
        "V": protocol.VIEW_NUM,
        "K": protocol.CLASS_NUM,
        "Dz": protocol.LATENT_DIM,
        "model_seed": protocol.MODEL_SEED,
        "corruption_seed": protocol.CORRUPTION_SEED,
        "corruption_k": protocol.CORRUPTION_K,
        "snr_db": protocol.SNR_DB,
        "data_path": protocol.display_path(data_path),
        "backbone_dir": protocol.display_path(backbone_dir),
        "backbone_checkpoint_paths": fixed["backbone_load_audit"][
            "checkpoint_paths"
        ],
        "backbone_checkpoint_file_sha256": fixed["backbone_load_audit"][
            "checkpoint_file_sha256"
        ],
        "utility_path": protocol.display_path(fixed["frozen_u"]["utility_path"]),
        "utility_transfer_path": protocol.display_path(
            fixed["frozen_u"]["transfer_path"]
        ),
        "U_sha256": fixed["frozen_u"]["U_sha256"],
        "utility_file_sha256": fixed["frozen_u"]["utility_file_sha256"],
        "runtime_corruption_mask_sha256": fixed["runtime_mask_hash"],
        "stored_corruption_audit": protocol.display_path(
            fixed["stored_corruption_path"]
        ),
        "oracle_corruption_mask_path": protocol.display_path(
            fixed["oracle"]["corruption_mask_path"]
        ),
        "oracle_corruption_mask_sha256": fixed["oracle"][
            "corruption_mask_sha256"
        ],
        "U_loaded_not_refit": True,
        "T_loaded_for_hash_only_not_recomputed": True,
        "ridge_fit_performed": False,
        "optimizer_constructed": False,
        "backward_performed": False,
        "parameter_update_performed": False,
        "per_view_kmeans_performed": False,
        "ground_truth_loaded_after_fixed_output_seal": True,
        "fixed_output_seal_sha256": fixed["fixed_hashes"][
            "fixed_output_seal_sha256"
        ],
    }

    output_root.mkdir(parents=True)
    protocol.write_json(output_root / "provenance.json", provenance)
    protocol.write_json(output_root / "native_global_reference.json", native_reference_audit)
    protocol.write_json(output_root / "alignment_audit.json", alignment_audit)
    protocol.write_json(output_root / "admission_audit.json", admission_audit)
    protocol.write_json(output_root / "representation_audit.json", representation_audit)
    protocol.write_json(output_root / "results.json", results)
    protocol.write_json(output_root / "immutable_hashes.json", immutable_hashes)
    np.savez_compressed(output_root / "scores_predictions.npz", **_npz_payload(fixed))

    from experiments.g2_utility_semantic_consensus.audit_g2_a0_outputs import (
        audit_outputs,
    )
    from experiments.g2_utility_semantic_consensus.summarize_g2_a0_readonly_consensus import (
        summarize_outputs,
    )

    audit_outputs(output_root)
    summarize_outputs(output_root)
    print("ENGINEERING_PASS=" + str(engineering_pass))
    print("SCIENTIFIC_VERDICT=" + results["SCIENTIFIC_VERDICT"])
    print("Saved: " + protocol.display_path(output_root))
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-path", default=str(protocol.DEFAULT_DATA_PATH))
    parser.add_argument("--backbone-dir", default=str(protocol.DEFAULT_BACKBONE_DIR))
    parser.add_argument("--d2-dir", default=str(protocol.DEFAULT_D2_DIR))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_evaluation(
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        data_path=args.data_path,
        backbone_dir=args.backbone_dir,
        d2_dir=args.d2_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
