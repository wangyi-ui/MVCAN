"""Audit persisted G2-A0 artifacts without recomputation or model fitting."""

import argparse
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.g2_utility_semantic_consensus import g2_consensus_protocol as protocol
from irv.b4_information_utility import tensor_sha256
from weak_quality import ndarray_sha256


REQUIRED_ARTIFACTS = (
    "provenance.json",
    "native_global_reference.json",
    "alignment_audit.json",
    "admission_audit.json",
    "representation_audit.json",
    "scores_predictions.npz",
    "results.json",
    "immutable_hashes.json",
)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _prefix(policy_name):
    return policy_name.lower()


def _source_safety_audit():
    from experiments.g2_utility_semantic_consensus import (
        evaluate_g2_a0_readonly_consensus as evaluator,
    )

    evaluator_source = inspect.getsource(evaluator)
    protocol_source = inspect.getsource(protocol)
    construction_source = inspect.getsource(evaluator._construct_fixed_outputs)
    run_source = inspect.getsource(evaluator.run_evaluation)
    forbidden_calls = (
        "torch.optim",
        ".backward(",
        ".train(",
        "compute_transfer_scores(",
        "oof_ridge_predictability(",
        "compute_information_utility(",
    )
    no_update_calls = bool(
        all(
            token not in evaluator_source and token not in protocol_source
            for token in forbidden_calls
        )
    )
    label_api_absent_from_construction = (
        "load_caltech_labels_only" not in construction_source
    )
    seal_before_labels = bool(
        run_source.index("_construct_fixed_outputs(")
        < run_source.index("_metrics_after_fixed_output_seal(")
    )
    return {
        "no_update_or_refit_calls_pass": no_update_calls,
        "label_api_absent_from_fixed_construction_pass": (
            label_api_absent_from_construction
        ),
        "fixed_construction_precedes_label_metrics_pass": seal_before_labels,
    }


def audit_outputs(output_dir):
    root = Path(output_dir)
    missing = [name for name in REQUIRED_ARTIFACTS if not (root / name).is_file()]
    if missing:
        raise RuntimeError("missing G2-A0 artifacts: " + ", ".join(missing))

    provenance = _read_json(root / "provenance.json")
    native = _read_json(root / "native_global_reference.json")
    alignment = _read_json(root / "alignment_audit.json")
    admission = _read_json(root / "admission_audit.json")
    representation = _read_json(root / "representation_audit.json")
    results = _read_json(root / "results.json")
    immutable = _read_json(root / "immutable_hashes.json")
    with np.load(root / "scores_predictions.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    evaluation_count = int(results["evaluation_count"])
    evaluation_ids = arrays["evaluation_sample_ids"]
    expected_ids = np.arange(evaluation_count, dtype=np.int64)
    provenance_pass = bool(
        provenance["stage"] == protocol.STAGE
        and provenance["dataset"] == protocol.DATASET_NAME
        and provenance["N"] == protocol.SAMPLE_NUM
        and provenance["V"] == protocol.VIEW_NUM
        and provenance["K"] == protocol.CLASS_NUM
        and provenance["Dz"] == protocol.LATENT_DIM
        and provenance["U_sha256"] == protocol.EXPECTED_U_SHA256
        and provenance["utility_file_sha256"]
        == protocol.EXPECTED_UTILITY_FILE_SHA256
        and provenance["U_loaded_not_refit"] is True
        and provenance["T_loaded_for_hash_only_not_recomputed"] is True
        and provenance["ridge_fit_performed"] is False
        and provenance["optimizer_constructed"] is False
        and provenance["backward_performed"] is False
        and provenance["parameter_update_performed"] is False
        and provenance["per_view_kmeans_performed"] is False
        and provenance["ground_truth_loaded_after_fixed_output_seal"] is True
    )
    evaluation_boundary_pass = bool(
        0 < evaluation_count <= protocol.SAMPLE_NUM
        and evaluation_ids.dtype == np.dtype(np.int64)
        and evaluation_ids.shape == (evaluation_count,)
        and np.array_equal(evaluation_ids, expected_ids)
        and results["full_reference_count"] == protocol.SAMPLE_NUM
    )

    representation_pass = bool(
        representation["raw_z"]["shape"]
        == [protocol.SAMPLE_NUM, protocol.VIEW_NUM, protocol.LATENT_DIM]
        and representation["q"]["shape"]
        == [protocol.SAMPLE_NUM, protocol.VIEW_NUM, protocol.CLASS_NUM]
        and representation["q"]["clustering_input"] == "raw_encoder_latent"
        and representation["aligned_q"]["shape"]
        == [protocol.SAMPLE_NUM, protocol.VIEW_NUM, protocol.CLASS_NUM]
        and representation["q"]["probability_mass_max_abs_error"]
        <= protocol.PROBABILITY_ATOL
        and representation["aligned_q"]["probability_mass_max_abs_error"]
        <= protocol.PROBABILITY_ATOL
        and representation["full_reference_sample_count"] == protocol.SAMPLE_NUM
    )

    global_prediction = arrays["global_prediction"]
    global_centers = arrays["global_centers"]
    native_weights = arrays["native_nmi_weights"]
    fusion_weights = arrays["fusion_weights_used_for_final_reference"]
    native_reference_pass = bool(
        native["reference_scope"] == "full_1400_shared_once"
        and native["sample_count"] == protocol.SAMPLE_NUM
        and native["fusion_updates"] == protocol.FUSION_UPDATES
        and native["per_view_assignment_source"]
        == "argmax_frozen_checkpoint_q"
        and native["per_view_kmeans_performed"] is False
        and global_prediction.shape == (protocol.SAMPLE_NUM,)
        and global_prediction.dtype == np.dtype(np.int64)
        and global_centers.shape == (protocol.CLASS_NUM, protocol.FUSION_DIM)
        and native_weights.shape == (protocol.VIEW_NUM,)
        and fusion_weights.shape == (protocol.VIEW_NUM,)
        and np.isfinite(global_centers).all()
        and np.isfinite(native_weights).all()
        and np.all(native_weights > 0.0)
        and native["global_prediction_sha256"]
        == tensor_sha256(global_prediction)
        and native["global_centers_sha256"] == tensor_sha256(global_centers)
        and native["native_weights_sha256"] == tensor_sha256(native_weights)
        and native["native_nmi_weights"] == native_weights.tolist()
        and native["fusion_weights_used_for_final_reference"]
        == fusion_weights.tolist()
    )

    matrices = arrays["alignment_matrix"]
    aligned_q = arrays["aligned_q"]
    alignment_pass = bool(
        matrices.shape == (protocol.VIEW_NUM, protocol.CLASS_NUM, protocol.CLASS_NUM)
        and aligned_q.shape
        == (protocol.SAMPLE_NUM, protocol.VIEW_NUM, protocol.CLASS_NUM)
        and alignment["shape"]
        == [protocol.VIEW_NUM, protocol.CLASS_NUM, protocol.CLASS_NUM]
        and alignment["sha256"] == tensor_sha256(matrices)
        and native["alignment_sha256"] == tensor_sha256(matrices)
        and native["aligned_q_sha256"] == tensor_sha256(aligned_q)
        and np.isfinite(aligned_q).all()
        and np.allclose(
            aligned_q.sum(axis=2),
            1.0,
            rtol=0.0,
            atol=protocol.PROBABILITY_ATOL,
        )
    )
    for view_id in range(protocol.VIEW_NUM):
        alignment_pass = bool(
            alignment_pass
            and protocol.validate_permutation_matrix(matrices[view_id])
            and alignment["per_view"][view_id]["row_sums"]
            == [1] * protocol.CLASS_NUM
            and alignment["per_view"][view_id]["column_sums"]
            == [1] * protocol.CLASS_NUM
        )

    normal_api_pass = protocol.normal_api_oracle_isolation_pass()
    policy_arrays = {}
    admission_pass = bool(
        set(admission["policies"]) == set(protocol.CONSENSUS_POLICIES)
        and admission["normal_api_oracle_isolation_pass"] is True
        and normal_api_pass
    )
    for policy_name in protocol.CONSENSUS_POLICIES:
        mask = arrays[_prefix(policy_name) + "__admission_mask"]
        policy_arrays[policy_name] = mask
        expected_count = (
            protocol.VIEW_NUM if policy_name == "ALL" else protocol.TOP_K
        )
        record = admission["policies"][policy_name]
        admission_pass = bool(
            admission_pass
            and mask.dtype == np.dtype(bool)
            and mask.shape == (protocol.SAMPLE_NUM, protocol.VIEW_NUM)
            and np.all(mask.sum(axis=1) == expected_count)
            and record["sha256"] == ndarray_sha256(mask)
            and record["row_sum_min"] == expected_count
            and record["row_sum_max"] == expected_count
            and record["shared_reference_hashes"]
            == {
                key: native[key]
                for key in (
                    "global_prediction_sha256",
                    "global_centers_sha256",
                    "native_weights_sha256",
                    "final_latent_fusion_sha256",
                    "alignment_sha256",
                    "aligned_q_sha256",
                )
            }
        )
    permutation = arrays["shuffled_u_row_permutation"]
    shuffled_pass = bool(
        permutation.dtype == np.dtype(np.int64)
        and permutation.shape == (protocol.SAMPLE_NUM,)
        and np.array_equal(np.sort(permutation), np.arange(protocol.SAMPLE_NUM))
        and np.array_equal(
            policy_arrays["SHUFFLED_U"],
            policy_arrays["U_TOP3"][permutation],
        )
        and np.array_equal(
            policy_arrays["SHUFFLED_U"].sum(axis=0),
            policy_arrays["U_TOP3"].sum(axis=0),
        )
        and admission["shuffled_u_changed_row_count"] > 0
        and admission["shuffled_u_row_permutation_sha256"]
        == tensor_sha256(permutation)
    )

    fixed_hashes = results["fixed_output_hashes"]
    score_prediction_pass = True
    shared_reference_pass = True
    for policy_name in protocol.CONSENSUS_POLICIES:
        prefix = _prefix(policy_name)
        weighted = arrays[prefix + "__weighted_mask"]
        denominator = arrays[prefix + "__denominator"]
        score = arrays[prefix + "__consensus_score"]
        prediction = arrays[prefix + "__prediction"]
        score_prediction_pass = bool(
            score_prediction_pass
            and weighted.shape == (evaluation_count, protocol.VIEW_NUM)
            and denominator.shape == (evaluation_count,)
            and np.all(denominator > 0.0)
            and score.shape == (evaluation_count, protocol.CLASS_NUM)
            and prediction.shape == (evaluation_count,)
            and prediction.dtype == np.dtype(np.int64)
            and np.isfinite(score).all()
            and np.allclose(
                score.sum(axis=1),
                1.0,
                rtol=0.0,
                atol=protocol.PROBABILITY_ATOL,
            )
            and fixed_hashes["score_sha256"][policy_name]
            == tensor_sha256(score)
            and fixed_hashes["prediction_sha256"][policy_name]
            == tensor_sha256(prediction)
            and results["results"][policy_name]["score_sha256"]
            == tensor_sha256(score)
            and results["results"][policy_name]["prediction_sha256"]
            == tensor_sha256(prediction)
        )
        shared_reference_pass = bool(
            shared_reference_pass
            and results["results"][policy_name]["shared_reference_hashes"]
            == admission["policies"][policy_name]["shared_reference_hashes"]
        )
    native_prediction_hash_pass = bool(
        fixed_hashes["prediction_sha256"]["NATIVE_MVCAN_GLOBAL"]
        == tensor_sha256(global_prediction[evaluation_ids])
        == results["results"]["NATIVE_MVCAN_GLOBAL"]["prediction_sha256"]
    )
    seal_payload = dict(fixed_hashes)
    stored_seal = seal_payload.pop("fixed_output_seal_sha256")
    fixed_output_seal_pass = bool(
        stored_seal == protocol.json_sha256(seal_payload)
        and stored_seal == provenance["fixed_output_seal_sha256"]
    )

    metrics_pass = bool(set(results["results"]) == set(protocol.RESULT_ARMS))
    for arm_name in protocol.RESULT_ARMS:
        arm = results["results"][arm_name]
        metrics_pass = bool(
            metrics_pass
            and all(
                metric in arm and math.isfinite(float(arm[metric]))
                for metric in ("ACC", "NMI", "ARI")
            )
        )
        if arm_name in protocol.CONSENSUS_POLICIES:
            metrics_pass = bool(
                metrics_pass
                and all(
                    metric in arm and math.isfinite(float(arm[metric]))
                    for metric in (
                        "mean_entropy",
                        "mean_top1_top2_margin",
                        "mean_max_confidence",
                    )
                )
            )

    immutable_pass = bool(
        immutable["backbone_unchanged_pass"] is True
        and immutable["backbone_hash_before"] == immutable["backbone_hash_after"]
        and immutable["backbone_parameters_frozen_and_grad_none_pass"] is True
        and immutable["parameters_before"]["all_requires_grad_false"] is True
        and immutable["parameters_before"]["all_grad_none"] is True
        and immutable["parameters_after"]["all_requires_grad_false"] is True
        and immutable["parameters_after"]["all_grad_none"] is True
    )
    source_safety = _source_safety_audit()
    scientific_separation_pass = bool(
        results["SCIENTIFIC_VERDICT"] == "NOT_PREDEFINED_DESCRIPTIVE_ONLY"
        and results["scientific_pass_threshold_defined"] is False
        and results["paper_claim_made"] is False
        and results["results"]["NATIVE_MVCAN_GLOBAL"]["role"]
        == "descriptive_native_global_baseline"
    )
    checks = {
        "all_artifacts_present_pass": True,
        "provenance_pass": provenance_pass,
        "evaluation_boundary_pass": evaluation_boundary_pass,
        "representation_pass": representation_pass,
        "native_global_reference_pass": native_reference_pass,
        "alignment_permutation_and_mass_pass": alignment_pass,
        "admission_pass": admission_pass,
        "shuffled_u_preservation_and_informativeness_pass": shuffled_pass,
        "score_prediction_hash_pass": score_prediction_pass,
        "native_prediction_hash_pass": native_prediction_hash_pass,
        "shared_reference_pass": shared_reference_pass,
        "fixed_output_seal_pass": fixed_output_seal_pass,
        "metrics_finite_pass": metrics_pass,
        "immutability_pass": immutable_pass,
        "source_no_update_or_refit_pass": source_safety[
            "no_update_or_refit_calls_pass"
        ],
        "label_api_absent_from_fixed_construction_pass": source_safety[
            "label_api_absent_from_fixed_construction_pass"
        ],
        "fixed_construction_precedes_label_metrics_pass": source_safety[
            "fixed_construction_precedes_label_metrics_pass"
        ],
        "scientific_engineering_separation_pass": scientific_separation_pass,
        "full_reference_under_smoke_pass": native["sample_count"]
        == protocol.SAMPLE_NUM,
    }
    audit = {
        "stage": protocol.STAGE,
        "output_dir": str(root.resolve()),
        "mode": results["mode"],
        "evaluation_count": evaluation_count,
        "checks": checks,
        "source_safety": source_safety,
        "ENGINEERING_PASS": bool(all(checks.values())),
        "SCIENTIFIC_VERDICT": results["SCIENTIFIC_VERDICT"],
    }
    _write_json(root / "audit.json", audit)
    if not audit["ENGINEERING_PASS"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("G2-A0 audit failed: " + ", ".join(failed))
    return audit


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    audit = audit_outputs(args.output_dir)
    print("ENGINEERING_PASS=" + str(audit["ENGINEERING_PASS"]))
    print("SCIENTIFIC_VERDICT=" + audit["SCIENTIFIC_VERDICT"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
