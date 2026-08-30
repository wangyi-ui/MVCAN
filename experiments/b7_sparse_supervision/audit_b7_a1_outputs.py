"""Audit persisted B7-A1 artifacts without recomputation or model fitting."""

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

from experiments.b7_sparse_supervision import b7_sparse_anchor_protocol as protocol
from irv.b4_information_utility import tensor_sha256
from weak_quality import ndarray_sha256


def _read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _prefix(representation, policy_name):
    return representation + "__" + policy_name.lower()


def audit_outputs(output_dir):
    root = Path(output_dir)
    required = (
        "provenance.json",
        "representation_audit.json",
        "label_split_audit.json",
        "anchor_policy_audit.json",
        "prototype_coverage.json",
        "results.json",
        "predictions.npz",
        "prototype_counts.npz",
        "immutable_hashes.json",
    )
    all_artifacts_present_pass = bool(all((root / name).is_file() for name in required))
    if not all_artifacts_present_pass:
        missing = [name for name in required if not (root / name).is_file()]
        raise RuntimeError("missing B7-A1 artifacts: " + ", ".join(missing))

    provenance = _read_json(root / "provenance.json")
    representation = _read_json(root / "representation_audit.json")
    split = _read_json(root / "label_split_audit.json")
    anchors = _read_json(root / "anchor_policy_audit.json")
    coverage = _read_json(root / "prototype_coverage.json")
    results = _read_json(root / "results.json")
    immutable = _read_json(root / "immutable_hashes.json")

    provenance_pass = bool(
        provenance["stage"] == "B7-A1"
        and provenance["dataset"] == protocol.DATASET_NAME
        and provenance["N"] == protocol.SAMPLE_NUM
        and provenance["V"] == protocol.VIEW_NUM
        and provenance["K"] == protocol.CLASS_NUM
        and provenance["U_sha256"] == protocol.EXPECTED_U_SHA256
        and provenance["utility_file_sha256"]
        == protocol.EXPECTED_UTILITY_FILE_SHA256
        and provenance["label_split_sha256"]
        == protocol.EXPECTED_LABEL_SPLIT_SHA256
        and provenance["U_loaded_not_refit"] is True
        and provenance["T_loaded_for_provenance_not_recomputed"] is True
        and provenance["ridge_fit_performed"] is False
        and provenance["optimizer_constructed"] is False
        and provenance["backward_performed"] is False
        and provenance["parameter_update_performed"] is False
        and provenance["full_labels_entered_after_predictions_fixed"] is True
    )
    split_pass = bool(
        split["label_split_sha256"] == protocol.EXPECTED_LABEL_SPLIT_SHA256
        and split["labeled_count"] == protocol.LABELED_NUM
        and split["unlabeled_count"] == protocol.UNLABELED_NUM
        and split["intersection_count"] == 0
        and split["union_count"] == protocol.SAMPLE_NUM
        and split["sparse_class_histogram"] == [2] * protocol.CLASS_NUM
        and split["split_regenerated"] is False
    )
    representation_pass = bool(
        representation["primary_representations"] == ["z", "q"]
        and representation["z"]["shape"] == [protocol.SAMPLE_NUM, protocol.VIEW_NUM, 10]
        and representation["q"]["shape"]
        == [protocol.SAMPLE_NUM, protocol.VIEW_NUM, protocol.CLASS_NUM]
        and representation["q"]["representation_source"] == "native_q_raw"
        and representation["q"]["clustering_input"] == "raw_encoder_latent"
        and representation["q"]["metric_preprocessing"]
        == "l2_for_cosine_only"
        and representation["d2_normalized_z_exact_match_pass"] is True
    )
    if "h" in results["representations"]:
        representation_pass = bool(
            representation_pass
            and representation["h"] is not None
            and representation["h"]["shape"]
            == [protocol.SAMPLE_NUM, protocol.VIEW_NUM, 10]
        )

    normal_parameters = inspect.signature(
        protocol.build_normal_anchor_policies
    ).parameters
    normal_api_oracle_isolation_pass = bool(
        all(
            token not in name.lower()
            for name in normal_parameters
            for token in ("oracle", "corrupt", "clean", "mask")
        )
    )
    prototype_api_pass = protocol.prototype_builder_api_pass()

    query_num = int(results["query_count"])
    expected_query_num = (
        protocol.UNLABELED_NUM if results["mode"] == "formal" else query_num
    )
    query_count_pass = bool(
        0 < query_num <= protocol.UNLABELED_NUM and query_num == expected_query_num
    )
    with np.load(root / "predictions.npz", allow_pickle=False) as predictions:
        query_ids = predictions["query_sample_ids"]
        query_mask = predictions["query_mask"]
        query_mask_pass = bool(
            query_ids.dtype == np.dtype(np.int64)
            and query_ids.shape == (query_num,)
            and query_mask.dtype == np.dtype(bool)
            and query_mask.shape == (query_num, protocol.VIEW_NUM)
            and np.all(query_mask.sum(axis=1) == protocol.TOP_K)
            and ndarray_sha256(query_mask) == anchors["query_mask_sha256"]
        )
        prediction_arrays = {name: predictions[name] for name in predictions.files}
    with np.load(root / "prototype_counts.npz", allow_pickle=False) as prototypes:
        prototype_arrays = {name: prototypes[name] for name in prototypes.files}

    policy_records = anchors["policies"]
    policy_shapes_pass = bool(set(policy_records) == set(protocol.ANCHOR_POLICIES))
    for policy_name in protocol.ANCHOR_POLICIES:
        record = policy_records[policy_name]
        expected_sum = protocol.VIEW_NUM if policy_name == "ALL" else protocol.TOP_K
        policy_shapes_pass = bool(
            policy_shapes_pass
            and record["shape"] == [protocol.LABELED_NUM, protocol.VIEW_NUM]
            and record["row_sums"] == [expected_sum] * protocol.LABELED_NUM
        )
    shuffled_controls_pass = bool(
        policy_records["U_TOP3"]["column_sums"]
        == policy_records["SHUFFLED_U"]["column_sums"]
        and policy_records["U_TOP3"]["total_admissions"]
        == policy_records["SHUFFLED_U"]["total_admissions"]
        and policy_records["U_TOP3"]["sha256"]
        == policy_records["SHUFFLED_LABEL"]["sha256"]
        and anchors["original_sparse_class_histogram"]
        == anchors["shuffled_sparse_class_histogram"]
        == [2] * protocol.CLASS_NUM
        and sorted(anchors["shuffled_u_row_permutation"])
        == list(range(protocol.LABELED_NUM))
        and sorted(anchors["shuffled_label_permutation"])
        == list(range(protocol.LABELED_NUM))
    )
    matched_control_records = anchors["coverage_matched_controls"]
    matched_control_audit_pass = bool(
        set(matched_control_records)
        == {"U_CLASS_SHUFFLED", "ORACLE_CLASS_SHUFFLED"}
    )
    for control_name, source_name in (
        ("U_CLASS_SHUFFLED", "U_TOP3"),
        ("ORACLE_CLASS_SHUFFLED", "ORACLE"),
    ):
        record = matched_control_records[control_name]
        changed_flags = [
            bool(class_record["first_row_changed"])
            + bool(class_record["second_row_changed"])
            for class_record in record["classwise_swap_audit"]
        ]
        matched_control_audit_pass = bool(
            matched_control_audit_pass
            and record["source_policy"] == source_name
            and record["targets_unchanged"] is True
            and 0 < int(record["changed_row_count"]) <= protocol.LABELED_NUM
            and 0 <= int(record["identical_mask_pair_count"]) < protocol.CLASS_NUM
            and len(record["classwise_swap_audit"]) == protocol.CLASS_NUM
            and sum(changed_flags) == int(record["changed_row_count"])
            and policy_records[control_name]["row_sums"]
            == [protocol.TOP_K] * protocol.LABELED_NUM
        )


    tensor_boundary_pass = True
    availability_pass = True
    coverage_pass = True
    metrics_pass = True
    for representation_name in results["representations"]:
        for policy_name in protocol.ANCHOR_POLICIES:
            prefix = _prefix(representation_name, policy_name)
            count = prototype_arrays[prefix + "__prototype_count"]
            valid = prototype_arrays[prefix + "__prototype_valid_mask"]
            class_coverage = prototype_arrays[prefix + "__class_coverage"]
            view_coverage = prototype_arrays[prefix + "__view_coverage"]
            prediction = prediction_arrays[prefix + "__prediction"]
            scores = prediction_arrays[prefix + "__aggregated_scores"]
            available = prediction_arrays[
                prefix + "__query_class_score_available"
            ]
            tensor_boundary_pass = bool(
                tensor_boundary_pass
                and count.shape == (protocol.CLASS_NUM, protocol.VIEW_NUM)
                and np.issubdtype(count.dtype, np.integer)
                and valid.shape == (protocol.CLASS_NUM, protocol.VIEW_NUM)
                and valid.dtype == np.dtype(bool)
                and np.array_equal(valid, count > 0)
                and class_coverage.shape == (protocol.CLASS_NUM,)
                and view_coverage.shape == (protocol.VIEW_NUM,)
                and prediction.shape == (query_num,)
                and prediction.dtype == np.dtype(np.int64)
                and scores.shape == (query_num, protocol.CLASS_NUM)
                and available.shape == (query_num, protocol.CLASS_NUM)
                and available.dtype == np.dtype(bool)
            )
            availability_pass = bool(
                availability_pass
                and np.all(available.any(axis=1))
                and np.isfinite(scores[available]).all()
                and np.isnan(scores[~available]).all()
            )
            saved_coverage = coverage[representation_name][policy_name]
            coverage_pass = bool(
                coverage_pass
                and saved_coverage["prototype_count_shape"]
                == [protocol.CLASS_NUM, protocol.VIEW_NUM]
                and saved_coverage["prototype_valid_mask_shape"]
                == [protocol.CLASS_NUM, protocol.VIEW_NUM]
                and saved_coverage["query_class_score_available_shape"]
                == [query_num, protocol.CLASS_NUM]
                and saved_coverage["prototype_count_sha256"]
                == tensor_sha256(count)
                and saved_coverage["prototype_valid_mask_sha256"]
                == ndarray_sha256(valid)
                and saved_coverage["query_class_score_available_sha256"]
                == ndarray_sha256(available)
                and saved_coverage["class_coverage"] == class_coverage.tolist()
                and saved_coverage["view_coverage"] == view_coverage.tolist()
            )
            metrics = results["results"][representation_name][policy_name]
            metric_names = (
                "nearest_prototype_ACC",
                "margin_mean",
                "margin_median",
                "positive_margin_rate",
                "semantic_gap",
                "prototype_class_view_coverage_rate",
                "true_class_score_available_rate",
                "full_class_score_available_rate",
            )
            metrics_pass = bool(
                metrics_pass
                and all(math.isfinite(float(metrics[name])) for name in metric_names)
                and 0 <= int(metrics["margin_valid_count"]) <= query_num
                and metrics["samples_with_at_least_one_available_class"]
                == query_num
                and metrics["query_count"] == query_num
            )

    matched_prototype_coverage_pass = True
    for representation_name in results["representations"]:
        for source_name, control_name in (
            ("U_TOP3", "U_CLASS_SHUFFLED"),
            ("ORACLE", "ORACLE_CLASS_SHUFFLED"),
        ):
            source_prefix = _prefix(representation_name, source_name)
            control_prefix = _prefix(representation_name, control_name)
            source_coverage = coverage[representation_name][source_name]
            control_coverage = coverage[representation_name][control_name]
            matched_prototype_coverage_pass = bool(
                matched_prototype_coverage_pass
                and np.array_equal(
                    prototype_arrays[source_prefix + "__prototype_count"],
                    prototype_arrays[control_prefix + "__prototype_count"],
                )
                and np.array_equal(
                    prototype_arrays[source_prefix + "__prototype_valid_mask"],
                    prototype_arrays[control_prefix + "__prototype_valid_mask"],
                )
                and np.array_equal(
                    prototype_arrays[source_prefix + "__class_coverage"],
                    prototype_arrays[control_prefix + "__class_coverage"],
                )
                and np.array_equal(
                    prototype_arrays[source_prefix + "__view_coverage"],
                    prototype_arrays[control_prefix + "__view_coverage"],
                )
                and source_coverage["prototype_class_view_coverage_rate"]
                == control_coverage["prototype_class_view_coverage_rate"]
            )

    immutable_pass = bool(
        immutable["backbone_unchanged_pass"] is True
        and immutable["backbone_parameters_frozen_and_grad_none_pass"] is True
        and immutable["unsup_h_unchanged_pass"] is True
        and immutable["unsup_h_parameters_frozen_and_grad_none_pass"] is True
    )
    evaluator_source = (
        REPOSITORY_ROOT
        / "experiments/b7_sparse_supervision/evaluate_b7_a1_sparse_anchor_feasibility.py"
    ).read_text(encoding="utf-8")
    static_read_only_pass = bool(
        "torch.optim" not in evaluator_source
        and ".backward(" not in evaluator_source
        and ".train(" not in evaluator_source
        and "compute_transfer_scores(" not in evaluator_source
        and "oof_ridge_predictability(" not in evaluator_source
        and "compute_information_utility(" not in evaluator_source
    )
    no_scientific_threshold_pass = bool(
        results["scientific_pass_threshold_defined"] is False
    )
    propagated_engineering_pass = bool(
        results["ENGINEERING_PASS"] is True
        and all(results["engineering_checks"].values())
    )
    engineering_checks = {
        "all_artifacts_present_pass": all_artifacts_present_pass,
        "provenance_pass": provenance_pass,
        "split_pass": split_pass,
        "representation_pass": representation_pass,
        "normal_api_oracle_isolation_pass": normal_api_oracle_isolation_pass,
        "prototype_api_pass": prototype_api_pass,
        "query_count_pass": query_count_pass,
        "query_mask_pass": query_mask_pass,
        "policy_shapes_pass": policy_shapes_pass,
        "shuffled_controls_pass": shuffled_controls_pass,
        "matched_control_audit_pass": matched_control_audit_pass,
        "matched_prototype_coverage_pass": matched_prototype_coverage_pass,
        "tensor_boundary_pass": tensor_boundary_pass,
        "availability_pass": availability_pass,
        "coverage_pass": coverage_pass,
        "metrics_pass": metrics_pass,
        "immutable_pass": immutable_pass,
        "static_read_only_pass": static_read_only_pass,
        "no_scientific_threshold_pass": no_scientific_threshold_pass,
        "propagated_engineering_pass": propagated_engineering_pass,
    }
    result = {
        "stage": "B7-A1",
        "mode": results["mode"],
        "query_count": query_num,
        "representations": results["representations"],
        "engineering_checks": engineering_checks,
        "ENGINEERING_PASS": bool(all(engineering_checks.values())),
        "scientific_pass_threshold_defined": False,
    }
    _write_json(root / "audit.json", result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = audit_outputs(args.output_dir)
    print("ENGINEERING_PASS=" + str(result["ENGINEERING_PASS"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
