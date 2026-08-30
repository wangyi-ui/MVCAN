"""Read-only B7-A1 frozen sparse semantic-anchor feasibility evaluator."""

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from configure import get_default_config
from datasets import load_data
from experiments.b7_sparse_supervision import b7_sparse_anchor_protocol as protocol
from experiments.b7_sparse_supervision.train_b7_a0_sparse_supervision_admission import (
    B7SemanticCarrier,
)
import experiments.d2_caltech6v.evaluate_d2_a0_utility_transfer as d2
from irv.b3_audit import hash_backbone
from irv.b3_audit import hash_state_dict
from irv.b4_information_utility import tensor_sha256
from weak_quality import apply_weak_quality_protocol
from weak_quality import ndarray_sha256


STAGE = "B7-A1"
MODEL_SEED = 20
CORRUPTION_SEED = 20
CORRUPTION_K = 3
SNR_DB = 2.5
EXPECTED_VIEW_DIMS = [48, 40, 254, 1984, 512, 928]
DEFAULT_BACKBONE_DIR = (
    REPOSITORY_ROOT / "outputs/d1_caltech6v/snr2p5_k3_seed20/models"
)
DEFAULT_UNSUP_HEAD = (
    REPOSITORY_ROOT
    / "outputs/b7_sparse_supervision/b7a0_seed20/unsup/final_semantic_head.pt"
)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/b7_sparse_supervision"
REPRESENTATION_ORDER = ("z", "q", "h")


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


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def _freeze_modules(modules):
    for module in modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None


def _parameters_are_frozen(modules):
    return bool(
        all(
            not parameter.requires_grad and parameter.grad is None
            for module in modules
            for parameter in module.parameters()
        )
    )


def _load_unsup_h(z_stack, checkpoint_path):
    """Reconstruct formal B7-A0 UNSUP h without using initialized weights."""
    path = _resolve(checkpoint_path)
    _require(path.is_file(), "formal B7-A0 UNSUP semantic checkpoint is missing")
    state = d2._load_checkpoint(path)
    carrier = B7SemanticCarrier()
    carrier.load_state_dict(state, strict=True)
    carrier.eval()
    _freeze_modules([carrier])
    state_hash_before = hash_state_dict(carrier.state_dict())
    with torch.no_grad():
        # h_stack: [N,V,Dh]
        h_stack, logits = carrier(z_stack)
        h_stack = h_stack.detach()
        logits = logits.detach()
    _require(
        tuple(h_stack.shape) == (protocol.SAMPLE_NUM, protocol.VIEW_NUM, 10)
        and tuple(logits.shape)
        == (protocol.SAMPLE_NUM, protocol.VIEW_NUM, protocol.CLASS_NUM),
        "formal UNSUP h/logit shape mismatch",
    )
    return h_stack, carrier, {
        "checkpoint_path": _display(path),
        "checkpoint_file_sha256": protocol.file_sha256(path),
        "state_hash_before": state_hash_before,
        "shape": list(h_stack.shape),
        "dtype": str(h_stack.detach().cpu().numpy().dtype),
    }


def _artifact_key(representation, policy_name, suffix):
    return representation + "__" + policy_name.lower() + "__" + suffix


def _coverage_record(prototype_result, score_result, metrics):
    counts = prototype_result["prototype_count"].cpu().numpy()
    valid = prototype_result["prototype_valid_mask"].cpu().numpy()
    available = score_result["query_class_score_available"].cpu().numpy()
    return {
        "prototype_count": counts.tolist(),
        "prototype_count_shape": list(counts.shape),
        "prototype_count_sha256": tensor_sha256(counts),
        "prototype_valid_mask": valid.tolist(),
        "prototype_valid_mask_shape": list(valid.shape),
        "prototype_valid_mask_sha256": ndarray_sha256(valid),
        "class_coverage": prototype_result["class_coverage"].cpu().numpy().tolist(),
        "view_coverage": prototype_result["view_coverage"].cpu().numpy().tolist(),
        "prototype_class_view_coverage_rate": float(
            prototype_result["class_view_coverage_rate"]
        ),
        "query_class_score_available_shape": list(available.shape),
        "query_class_score_available_sha256": ndarray_sha256(available),
        "true_class_score_available_rate": metrics[
            "true_class_score_available_rate"
        ],
        "full_class_score_available_rate": metrics[
            "full_class_score_available_rate"
        ],
        "samples_with_at_least_one_available_class": metrics[
            "samples_with_at_least_one_available_class"
        ],
    }


def run_evaluation(
    representations=("z", "q"),
    max_unlabeled=None,
    output_dir=None,
    backbone_dir=DEFAULT_BACKBONE_DIR,
    d2_dir=protocol.DEFAULT_D2_DIR,
    split_dir=protocol.DEFAULT_LABEL_SPLIT_DIR,
    unsup_head=DEFAULT_UNSUP_HEAD,
):
    """Run the frozen diagnostic; no fitting or parameter update occurs."""
    requested = tuple(str(value).lower() for value in representations)
    _require(
        len(requested) == len(set(requested))
        and {"z", "q"}.issubset(set(requested))
        and set(requested).issubset(set(REPRESENTATION_ORDER)),
        "representations must be 'z q' with optional 'h'",
    )
    requested = tuple(value for value in REPRESENTATION_ORDER if value in requested)
    if max_unlabeled is not None:
        max_unlabeled = int(max_unlabeled)
        _require(
            0 < max_unlabeled <= protocol.UNLABELED_NUM,
            "max_unlabeled must be in [1,1386]",
        )
    smoke = max_unlabeled is not None
    if output_dir is None:
        run_name = "b7a1_seed20_smoke" if smoke else "b7a1_seed20"
        output_root = DEFAULT_OUTPUT_ROOT / run_name
    else:
        output_root = _resolve(output_dir)
        run_name = output_root.name
    _require(not output_root.exists(), "refusing to overwrite a B7-A1 output")

    torch.manual_seed(MODEL_SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    # Normal-path frozen artifacts do not expose the corruption mask.
    frozen_u = protocol.load_frozen_u_only(d2_dir)
    split = protocol.load_fixed_label_split(split_dir)
    labeled_ids = split["labeled_sample_ids"]
    all_unlabeled_ids = split["unlabeled_sample_ids"]
    query_ids = np.ascontiguousarray(
        all_unlabeled_ids[:max_unlabeled] if smoke else all_unlabeled_ids,
        dtype=np.int64,
    )
    query_num = int(query_ids.size)

    clean_views, initial_label_list = load_data({"dataset": protocol.DATASET_NAME})
    initial_targets = np.asarray(initial_label_list[0], dtype=np.int64)
    _require(
        len(clean_views) == protocol.VIEW_NUM
        and [int(view.shape[1]) for view in clean_views] == EXPECTED_VIEW_DIMS
        and initial_targets.shape == (protocol.SAMPLE_NUM,),
        "Caltech-6V dataset boundary mismatch",
    )
    # labeled_targets: [L].  This is the only target array crossing the
    # prototype-construction boundary.  The full array is discarded now.
    labeled_targets = np.ascontiguousarray(initial_targets[labeled_ids], dtype=np.int64)
    _require(
        np.array_equal(
            np.bincount(labeled_targets, minlength=protocol.CLASS_NUM),
            np.full(protocol.CLASS_NUM, 2, dtype=np.int64),
        ),
        "formal sparse labels are not exactly two per class",
    )
    del initial_targets
    del initial_label_list

    evaluation_views, runtime_corruption_audit = apply_weak_quality_protocol(
        clean_views,
        mode="heterogeneous_gaussian",
        k=CORRUPTION_K,
        snr_db=SNR_DB,
        corruption_seed=CORRUPTION_SEED,
    )
    runtime_mask_hash = str(runtime_corruption_audit["mask_sha256"])
    # The runtime mask is not retained or passed to any normal policy function.
    runtime_corruption_audit.pop("mask", None)
    runtime_corruption_audit.pop("corruption_pairs", None)

    config = get_default_config(protocol.DATASET_NAME)
    models, d2_normalized_views, backbone_load_audit = d2._load_native_representations(
        config=config,
        evaluation_views=evaluation_views,
        backbone_dir=_resolve(backbone_dir),
        sample_num=protocol.SAMPLE_NUM,
        view_num=protocol.VIEW_NUM,
        cluster_num=protocol.CLASS_NUM,
        model_seed=MODEL_SEED,
    )
    _freeze_modules(models.autoencoders)
    backbone_hash_before = hash_backbone(models.autoencoders)
    extracted = protocol.extract_native_z_q(models.autoencoders, evaluation_views)
    d2_z_stack = torch.stack(d2_normalized_views, dim=1).detach()
    _require(
        torch.equal(extracted["z_stack"], d2_z_stack),
        "native z helper differs from the frozen D2 normalized z path",
    )

    representation_tensors = {
        "z": extracted["z_stack"],
        "q": extracted["q_stack"],
    }
    carrier = None
    carrier_audit = None
    if "h" in requested:
        h_stack, carrier, carrier_audit = _load_unsup_h(
            extracted["z_stack"], unsup_head
        )
        representation_tensors["h"] = h_stack

    normal_policies = protocol.build_normal_anchor_policies(
        frozen_u["U"][labeled_ids], control_seed=protocol.CONTROL_SEED
    )
    anchor_masks = dict(normal_policies["anchor_masks"])
    u_class_shuffled, u_class_swap_audit = (
        protocol.build_classwise_swapped_anchor_mask(
            anchor_masks["U_TOP3"], labeled_targets, protocol.CLASS_NUM
        )
    )
    anchor_masks["U_CLASS_SHUFFLED"] = u_class_shuffled
    shuffled_targets, shuffled_label_permutation = (
        protocol.build_shuffled_labeled_targets(
            labeled_targets, control_seed=protocol.CONTROL_SEED
        )
    )
    # The oracle artifact enters only through the isolated loader/builder pair.
    oracle = protocol.load_oracle_corruption_mask(d2_dir)
    _require(
        oracle["corruption_mask_sha256"] == runtime_mask_hash,
        "runtime/frozen oracle corruption provenance mismatch",
    )
    anchor_masks["ORACLE"] = protocol.build_oracle_anchor_mask(
        oracle["corruption_mask"], labeled_ids
    )
    oracle_class_shuffled, oracle_class_swap_audit = (
        protocol.build_classwise_swapped_anchor_mask(
            anchor_masks["ORACLE"], labeled_targets, protocol.CLASS_NUM
        )
    )
    anchor_masks["ORACLE_CLASS_SHUFFLED"] = oracle_class_shuffled
    _require(set(anchor_masks) == set(protocol.ANCHOR_POLICIES), "policy set mismatch")

    # query_mask: [Nq,V], shared byte-exactly by every representation/policy.
    full_query_mask = protocol.frozen_top3_mask(frozen_u["U"])
    query_mask = np.ascontiguousarray(full_query_mask[query_ids], dtype=bool)
    _require(
        query_mask.shape == (query_num, protocol.VIEW_NUM)
        and np.all(query_mask.sum(axis=1) == protocol.TOP_K),
        "common frozen-U query mask boundary mismatch",
    )

    labeled_index = protocol._torch_boundary(labeled_ids, dtype=torch.long)
    query_index = protocol._torch_boundary(query_ids, dtype=torch.long)
    fixed_outputs = {}
    prototype_npz = {
        "labeled_sample_ids": np.asarray(labeled_ids, dtype=np.int64),
        "shuffled_u_row_permutation": normal_policies[
            "shuffled_u_row_permutation"
        ],
        "shuffled_label_permutation": shuffled_label_permutation,
    }
    prediction_npz = {
        "query_sample_ids": query_ids,
        "query_mask": query_mask,
    }
    # No full target vector exists in this scope.  Prototypes, semantic scores,
    # and predictions are fixed using only fourteen sparse targets.
    for representation in requested:
        values = representation_tensors[representation]
        # labeled_repr/query_repr: [L,V,D] / [Nq,V,D]
        labeled_repr = values[labeled_index].detach()
        query_repr = values[query_index].detach()
        for policy_name in protocol.ANCHOR_POLICIES:
            policy_targets = (
                shuffled_targets
                if policy_name == "SHUFFLED_LABEL"
                else labeled_targets
            )
            prototype_result = protocol.build_within_view_prototypes(
                labeled_repr,
                policy_targets,
                anchor_masks[policy_name],
                protocol.CLASS_NUM,
            )
            score_result = protocol.score_unlabeled_queries(
                query_repr,
                prototype_result["prototypes"],
                prototype_result["prototype_valid_mask"],
                query_mask,
            )
            key = representation + "/" + policy_name
            fixed_outputs[key] = {
                "prototype": prototype_result,
                "score": score_result,
            }
            artifact_prefix = _artifact_key(representation, policy_name, "")[:-2]
            prototype_npz[artifact_prefix + "__prototype_count"] = (
                prototype_result["prototype_count"].cpu().numpy()
            )
            prototype_npz[artifact_prefix + "__prototype_valid_mask"] = (
                prototype_result["prototype_valid_mask"].cpu().numpy()
            )
            prototype_npz[artifact_prefix + "__class_coverage"] = (
                prototype_result["class_coverage"].cpu().numpy()
            )
            prototype_npz[artifact_prefix + "__view_coverage"] = (
                prototype_result["view_coverage"].cpu().numpy()
            )
            prediction_npz[artifact_prefix + "__prediction"] = (
                score_result["predictions"].cpu().numpy().astype(np.int64)
            )
            prediction_npz[artifact_prefix + "__aggregated_scores"] = (
                score_result["aggregated_scores"].cpu().numpy()
            )
            prediction_npz[artifact_prefix + "__query_class_score_available"] = (
                score_result["query_class_score_available"].cpu().numpy()
            )

    # Full labels re-enter only now, after every prediction is fixed.
    _, evaluation_label_list = load_data({"dataset": protocol.DATASET_NAME})
    evaluation_targets = np.asarray(evaluation_label_list[0], dtype=np.int64)
    query_targets = np.ascontiguousarray(evaluation_targets[query_ids], dtype=np.int64)
    results_by_representation = {representation: {} for representation in requested}
    coverage_by_representation = {representation: {} for representation in requested}
    for representation in requested:
        for policy_name in protocol.ANCHOR_POLICIES:
            fixed = fixed_outputs[representation + "/" + policy_name]
            metrics = protocol.evaluate_fixed_predictions(
                query_targets,
                fixed["score"]["predictions"],
                fixed["score"]["aggregated_scores"],
                fixed["score"]["query_class_score_available"],
            )
            metrics["prototype_class_view_coverage_rate"] = float(
                fixed["prototype"]["class_view_coverage_rate"]
            )
            metrics["query_count"] = query_num
            results_by_representation[representation][policy_name] = metrics
            coverage_by_representation[representation][policy_name] = _coverage_record(
                fixed["prototype"], fixed["score"], metrics
            )

    backbone_hash_after = hash_backbone(models.autoencoders)
    backbone_frozen_pass = _parameters_are_frozen(models.autoencoders)
    carrier_hash_after = None
    carrier_frozen_pass = True
    if carrier is not None:
        carrier_hash_after = hash_state_dict(carrier.state_dict())
        carrier_frozen_pass = _parameters_are_frozen([carrier])
    immutable_hashes = {
        "backbone_hash_before": backbone_hash_before,
        "backbone_hash_after": backbone_hash_after,
        "backbone_unchanged_pass": backbone_hash_before == backbone_hash_after,
        "backbone_parameters_frozen_and_grad_none_pass": backbone_frozen_pass,
        "unsup_h_used": "h" in requested,
        "unsup_h_state_hash_before": (
            carrier_audit["state_hash_before"] if carrier_audit is not None else None
        ),
        "unsup_h_state_hash_after": carrier_hash_after,
        "unsup_h_unchanged_pass": bool(
            carrier_audit is None
            or carrier_audit["state_hash_before"] == carrier_hash_after
        ),
        "unsup_h_parameters_frozen_and_grad_none_pass": carrier_frozen_pass,
    }

    representation_audit = {
        "primary_representations": ["z", "q"],
        "auxiliary_representations": ["h"] if "h" in requested else [],
        "requested_representations": list(requested),
        "raw_z": {
            "shape": list(extracted["raw_z_stack"].shape),
            "dtype": str(extracted["raw_z_stack"].cpu().numpy().dtype),
            "sha256": tensor_sha256(extracted["raw_z_stack"]),
            "representation_source": "native_encoder_raw_latent",
        },
        "z": {
            "shape": list(extracted["z_stack"].shape),
            "dtype": str(extracted["z_stack"].cpu().numpy().dtype),
            "sha256": tensor_sha256(extracted["z_stack"]),
            "representation_source": "native_normalized_z",
            "metric_preprocessing": "already_l2_normalized",
        },
        "q": {
            "shape": list(extracted["q_stack"].shape),
            "dtype": str(extracted["q_stack"].cpu().numpy().dtype),
            "sha256": tensor_sha256(extracted["q_stack"]),
            "representation_source": "native_q_raw",
            "clustering_input": extracted["q_input_source"],
            "metric_preprocessing": "l2_for_cosine_only",
        },
        "h": carrier_audit,
        "d2_normalized_z_exact_match_pass": True,
    }
    label_split_audit = {
        "label_split_path": _display(split["label_split_path"]),
        "labeled_ids_path": _display(split["labeled_ids_path"]),
        "unlabeled_ids_path": _display(split["unlabeled_ids_path"]),
        "label_split_sha256": split["label_split_sha256"],
        "labeled_ids_sha256": split["labeled_ids_sha256"],
        "labeled_count": int(labeled_ids.size),
        "unlabeled_count": int(all_unlabeled_ids.size),
        "intersection_count": int(
            np.intersect1d(labeled_ids, all_unlabeled_ids).size
        ),
        "union_count": int(np.union1d(labeled_ids, all_unlabeled_ids).size),
        "sparse_class_histogram": np.bincount(
            labeled_targets, minlength=protocol.CLASS_NUM
        ).tolist(),
        "split_regenerated": False,
    }
    anchor_policy_audit = {
        "control_seed": protocol.CONTROL_SEED,
        "anchor_policy_is_only_varying_factor": True,
        "query_policy": "D2 frozen U_TOP3 on selected unlabeled IDs",
        "query_mask_shape": list(query_mask.shape),
        "query_mask_sha256": ndarray_sha256(query_mask),
        "query_row_sums": query_mask.sum(axis=1).astype(np.int64).tolist(),
        "normal_policy_builder_parameters": list(
            inspect.signature(protocol.build_normal_anchor_policies).parameters
        ),
        "oracle_policy_builder_parameters": list(
            inspect.signature(protocol.build_oracle_anchor_mask).parameters
        ),
        "prototype_builder_parameters": list(
            inspect.signature(protocol.build_within_view_prototypes).parameters
        ),
        "shuffled_u_row_permutation": normal_policies[
            "shuffled_u_row_permutation"
        ].tolist(),
        "shuffled_label_permutation": shuffled_label_permutation.tolist(),
        "original_sparse_class_histogram": np.bincount(
            labeled_targets, minlength=protocol.CLASS_NUM
        ).tolist(),
        "shuffled_sparse_class_histogram": np.bincount(
            shuffled_targets, minlength=protocol.CLASS_NUM
        ).tolist(),
        "coverage_matched_controls": {
            "U_CLASS_SHUFFLED": {
                "source_policy": "U_TOP3",
                **u_class_swap_audit,
            },
            "ORACLE_CLASS_SHUFFLED": {
                "source_policy": "ORACLE",
                **oracle_class_swap_audit,
            },
        },
        "policies": {},
    }
    for policy_name in protocol.ANCHOR_POLICIES:
        mask = np.asarray(anchor_masks[policy_name], dtype=bool)
        anchor_policy_audit["policies"][policy_name] = {
            "shape": list(mask.shape),
            "sha256": ndarray_sha256(mask),
            "row_sums": mask.sum(axis=1).astype(np.int64).tolist(),
            "column_sums": mask.sum(axis=0).astype(np.int64).tolist(),
            "total_admissions": int(mask.sum(dtype=np.int64)),
        }

    coverage_matched_control_pass = True
    for representation in requested:
        for source_policy, matched_policy in (
            ("U_TOP3", "U_CLASS_SHUFFLED"),
            ("ORACLE", "ORACLE_CLASS_SHUFFLED"),
        ):
            source_prototype = fixed_outputs[
                representation + "/" + source_policy
            ]["prototype"]
            matched_prototype = fixed_outputs[
                representation + "/" + matched_policy
            ]["prototype"]
            coverage_matched_control_pass = bool(
                coverage_matched_control_pass
                and torch.equal(
                    source_prototype["prototype_count"],
                    matched_prototype["prototype_count"],
                )
                and torch.equal(
                    source_prototype["prototype_valid_mask"],
                    matched_prototype["prototype_valid_mask"],
                )
                and source_prototype["class_view_coverage_rate"]
                == matched_prototype["class_view_coverage_rate"]
            )
    matched_control_informativeness_pass = bool(
        u_class_swap_audit["changed_row_count"] > 0
        and oracle_class_swap_audit["changed_row_count"] > 0
    )

    engineering_checks = {
        "frozen_u_hash_pass": frozen_u["U_sha256"] == protocol.EXPECTED_U_SHA256,
        "frozen_u_file_hash_pass": frozen_u["utility_file_sha256"]
        == protocol.EXPECTED_UTILITY_FILE_SHA256,
        "fixed_label_split_pass": split["label_split_sha256"]
        == protocol.EXPECTED_LABEL_SPLIT_SHA256,
        "label_partition_pass": label_split_audit["intersection_count"] == 0
        and label_split_audit["union_count"] == protocol.SAMPLE_NUM,
        "prototype_builder_api_pass": protocol.prototype_builder_api_pass(),
        "coverage_matched_control_pass": coverage_matched_control_pass,
        "matched_control_informativeness_pass": matched_control_informativeness_pass,
        "normal_policy_oracle_isolation_pass": all(
            token not in name.lower()
            for name in inspect.signature(
                protocol.build_normal_anchor_policies
            ).parameters
            for token in ("oracle", "corrupt", "clean", "mask")
        ),
        "query_mask_exact_top3_pass": bool(
            np.all(query_mask.sum(axis=1) == protocol.TOP_K)
        ),
        "all_predictions_available_pass": all(
            metrics["samples_with_at_least_one_available_class"] == query_num
            for policies in results_by_representation.values()
            for metrics in policies.values()
        ),
        "backbone_immutability_pass": immutable_hashes[
            "backbone_unchanged_pass"
        ],
        "backbone_frozen_grad_pass": backbone_frozen_pass,
        "unsup_h_immutability_pass": immutable_hashes["unsup_h_unchanged_pass"],
        "unsup_h_frozen_grad_pass": carrier_frozen_pass,
        "no_optimizer_pass": True,
        "no_backward_pass": True,
        "no_parameter_update_pass": True,
        "no_u_refit_pass": True,
        "no_t_recomputation_pass": True,
        "full_labels_evaluation_only_pass": True,
        "smoke_not_scientific_pass": True,
    }
    engineering_pass = bool(all(engineering_checks.values()))
    results = {
        "stage": STAGE,
        "run_name": run_name,
        "mode": "smoke" if smoke else "formal",
        "scientific_interpretation_allowed": not smoke,
        "query_count": query_num,
        "representations": list(requested),
        "anchor_policies": list(protocol.ANCHOR_POLICIES),
        "results": results_by_representation,
        "engineering_checks": engineering_checks,
        "ENGINEERING_PASS": engineering_pass,
        "scientific_pass_threshold_defined": False,
    }
    provenance = {
        "stage": STAGE,
        "dataset": protocol.DATASET_NAME,
        "N": protocol.SAMPLE_NUM,
        "V": protocol.VIEW_NUM,
        "K": protocol.CLASS_NUM,
        "model_seed": MODEL_SEED,
        "corruption_seed": CORRUPTION_SEED,
        "corruption_k": CORRUPTION_K,
        "snr_db": SNR_DB,
        "backbone_dir": _display(_resolve(backbone_dir)),
        "backbone_checkpoint_paths": backbone_load_audit["checkpoint_paths"],
        "backbone_checkpoint_file_sha256": backbone_load_audit[
            "checkpoint_file_sha256"
        ],
        "utility_path": _display(frozen_u["utility_path"]),
        "utility_transfer_path": _display(frozen_u["transfer_path"]),
        "T_sha256": frozen_u["T_sha256"],
        "U_sha256": frozen_u["U_sha256"],
        "utility_file_sha256": frozen_u["utility_file_sha256"],
        "runtime_corruption_mask_sha256": runtime_mask_hash,
        "oracle_corruption_mask_path": _display(oracle["corruption_mask_path"]),
        "oracle_corruption_mask_sha256": oracle["corruption_mask_sha256"],
        "label_split_sha256": split["label_split_sha256"],
        "U_loaded_not_refit": True,
        "T_loaded_for_provenance_not_recomputed": True,
        "ridge_fit_performed": False,
        "optimizer_constructed": False,
        "backward_performed": False,
        "parameter_update_performed": False,
        "full_labels_entered_after_predictions_fixed": True,
    }

    output_root.mkdir(parents=True)
    _write_json(output_root / "provenance.json", provenance)
    _write_json(output_root / "representation_audit.json", representation_audit)
    _write_json(output_root / "label_split_audit.json", label_split_audit)
    _write_json(output_root / "anchor_policy_audit.json", anchor_policy_audit)
    _write_json(
        output_root / "prototype_coverage.json", coverage_by_representation
    )
    _write_json(output_root / "results.json", results)
    _write_json(output_root / "immutable_hashes.json", immutable_hashes)
    np.savez_compressed(output_root / "predictions.npz", **prediction_npz)
    np.savez_compressed(output_root / "prototype_counts.npz", **prototype_npz)

    from experiments.b7_sparse_supervision.audit_b7_a1_outputs import audit_outputs
    from experiments.b7_sparse_supervision.summarize_b7_a1_sparse_anchor_feasibility import (
        summarize_outputs,
    )

    audit_outputs(output_root)
    summarize_outputs(output_root)
    print("ENGINEERING_PASS=" + str(engineering_pass))
    print("Saved: " + _display(output_root))
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATION_ORDER,
        default=["z", "q"],
    )
    parser.add_argument("--max-unlabeled", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--backbone-dir", default=str(DEFAULT_BACKBONE_DIR))
    parser.add_argument("--d2-dir", default=str(protocol.DEFAULT_D2_DIR))
    parser.add_argument("--split-dir", default=str(protocol.DEFAULT_LABEL_SPLIT_DIR))
    parser.add_argument("--unsup-head", default=str(DEFAULT_UNSUP_HEAD))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_evaluation(
        representations=args.representations,
        max_unlabeled=args.max_unlabeled,
        output_dir=args.output_dir,
        backbone_dir=args.backbone_dir,
        d2_dir=args.d2_dir,
        split_dir=args.split_dir,
        unsup_head=args.unsup_head,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
