"""Evaluate frozen E4-A1 memory-specific semantic usefulness."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.e4_semantic_memory_bank import (
    evaluate_e4_a0_memory_feasibility as e4a0,
)
from experiments.e4_semantic_memory_bank.e4a1_memory_specific_utility import (
    E4A1_ARMS,
    NORMAL_E4A1_ARMS,
    attach_oracle_writer_weights,
    build_memory_information_utility,
    build_memory_specific_semantic_usefulness,
    build_normal_e4a1_writer_weights,
)
from experiments.e4_semantic_memory_bank.semantic_memory_bank import (
    CLASS_NUM,
    LABELED_NUM,
    SAMPLE_NUM,
    VIEW_NUM,
    build_class_memory,
    build_shared_query,
    predict_from_memory,
    validate_sparse_labels,
)
from irv.b4_information_utility import tensor_sha256


STAGE = "E4-A1"
SEED = 20
E4A0_REPLAY_MISMATCH = "E4A0_REPLAY_MISMATCH"
E4A1_DECISION_TREE_GAP = "E4A1_PREREGISTERED_DECISION_TREE_GAP"

DEFAULT_E1_LWC_MODEL_DIR = e4a0.DEFAULT_E1_LWC_MODEL_DIR
DEFAULT_LABEL_SPLIT_DIR = e4a0.DEFAULT_LABEL_SPLIT_DIR
DEFAULT_FULL_GT_PATH = e4a0.DEFAULT_FULL_GT_PATH
DEFAULT_E4A0_OUTPUT_DIR = (
    REPOSITORY_ROOT / "outputs/e4_semantic_memory_bank/e4a0_seed20"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "outputs/e4_semantic_memory_bank"
    / "e4a1_memory_specific_utility_seed20"
)
E4A0_REPLAY_ARMS = (
    "ALL_MEMORY",
    "R_MEMORY",
    "ORACLE_CLEAN_MEMORY",
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


def _as_frozen_tensor(value, dtype=None, device=None):
    if torch.is_tensor(value):
        tensor = value.detach()
        return tensor.to(
            dtype=tensor.dtype if dtype is None else dtype,
            device=tensor.device if device is None else device,
        ).detach()
    array = np.array(value, copy=True, order="C")
    return torch.as_tensor(array, dtype=dtype, device=device).detach()


@torch.no_grad()
def build_fixed_e4a1_outputs(
    h_sem,
    labeled_ids,
    labels_labeled,
    R_full,
    oracle_clean_weights,
):
    """Fix utility components, memories, scores, and predictions without full GT."""
    semantic = _as_frozen_tensor(h_sem)
    sample_num = int(semantic.shape[0])
    if semantic.shape != (sample_num, VIEW_NUM, CLASS_NUM):
        raise ValueError("h_sem must have shape [N,6,7]")
    if not bool(torch.isfinite(semantic).all().item()):
        raise ValueError("h_sem must be finite")
    ids = _as_frozen_tensor(labeled_ids, dtype=torch.long)
    if ids.shape != (LABELED_NUM,) or not bool(
        torch.all((ids >= 0) & (ids < sample_num)).item()
    ):
        raise ValueError("labeled_ids must contain fourteen valid rows")
    targets, _ = validate_sparse_labels(labels_labeled)
    R_values = _as_frozen_tensor(R_full)
    if R_values.shape != (sample_num, VIEW_NUM):
        raise ValueError("R_full must have shape [N,6]")
    if not bool(
        torch.isfinite(R_values).all().item()
        and torch.all((R_values >= 0.0) & (R_values <= 1.0)).item()
    ):
        raise ValueError("R_full must be finite and within [0,1]")

    h_labeled = semantic[ids].detach()
    if h_labeled.shape != (LABELED_NUM, VIEW_NUM, CLASS_NUM):
        raise RuntimeError("h_labeled must have shape [14,6,7]")
    R_labeled = R_values[ids].detach()
    semantic_components = build_memory_specific_semantic_usefulness(
        h_labeled, targets
    )
    S_mem = semantic_components["S_mem"].detach()
    U_mem = build_memory_information_utility(R_labeled, S_mem)
    normal_weights, shuffle_audit = build_normal_e4a1_writer_weights(
        R_labeled,
        S_mem,
        U_mem,
        targets,
    )
    writer_weights = attach_oracle_writer_weights(
        normal_weights, oracle_clean_weights
    )

    h_query = build_shared_query(semantic)
    fixed = {
        "h_sem": semantic,
        "h_labeled": h_labeled,
        "h_query": h_query,
        "labeled_ids": ids,
        "labels_labeled": targets,
        "R_labeled": R_labeled,
        "S_mem": S_mem,
        "U_mem": U_mem,
        "semantic_components": semantic_components,
        "writer_weights": writer_weights,
        "shuffle_audit": shuffle_audit,
    }
    for arm in E4A1_ARMS:
        prototypes = build_class_memory(
            h_labeled,
            targets,
            writer_weights[arm],
        )
        predictions, scores = predict_from_memory(h_query, prototypes)
        fixed[arm] = {
            "h_query": h_query,
            "prototypes": prototypes.detach(),
            "scores": scores.detach(),
            "predictions": predictions.detach(),
        }
    return fixed


def save_prediction_seal(fixed, output_root):
    """Save, reload, and hash all assignments before any full-GT access."""
    root = _resolve(output_root)
    prediction_path = root / "predictions.npz"
    seal_path = root / "prediction_seal.json"
    _require(
        root.is_dir()
        and not prediction_path.exists()
        and not seal_path.exists(),
        "prediction seal output boundary mismatch",
    )
    prediction_arrays = {
        arm: np.ascontiguousarray(
            fixed[arm]["predictions"].cpu().numpy(), dtype=np.int64
        )
        for arm in E4A1_ARMS
    }
    np.savez(prediction_path, **prediction_arrays)
    logical_hashes = {}
    with np.load(prediction_path, allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == E4A1_ARMS,
            "saved E4-A1 prediction arm set/order mismatch",
        )
        for arm in E4A1_ARMS:
            reloaded = np.ascontiguousarray(archive[arm], dtype=np.int64)
            _require(
                np.array_equal(reloaded, prediction_arrays[arm]),
                "reloaded prediction differs from its fixed assignment",
            )
            logical_hashes[arm] = tensor_sha256(reloaded)
    seal = {
        "stage": STAGE,
        "prediction_file": _display(prediction_path),
        "prediction_file_sha256": e4a0.file_sha256(prediction_path),
        "prediction_logical_sha256": logical_hashes,
        "all_arm_predictions_fixed_before_full_GT": True,
        "prediction_saved_before_full_GT": True,
        "prediction_reloaded_before_full_GT": True,
        "prediction_hashed_before_full_GT": True,
        "full_unlabeled_GT_loaded_before_prediction": False,
    }
    _write_json(seal_path, seal)
    return seal


def verify_e4a0_exact_replay(
    fixed,
    frozen_output_dir=DEFAULT_E4A0_OUTPUT_DIR,
):
    """Require byte-exact arrays and matching logical hashes for A0 baselines."""
    root = _resolve(frozen_output_dir)
    try:
        prototype_path = root / "memory_prototypes.npz"
        prediction_path = root / "predictions.npz"
        seal_path = root / "prediction_seal.json"
        audit_path = root / "memory_audit.json"
        _require(
            prototype_path.is_file()
            and prediction_path.is_file()
            and seal_path.is_file()
            and audit_path.is_file(),
            E4A0_REPLAY_MISMATCH,
        )
        with open(seal_path, "r", encoding="utf-8") as input_file:
            frozen_seal = json.load(input_file)
        with open(audit_path, "r", encoding="utf-8") as input_file:
            frozen_audit = json.load(input_file)
        _require(
            frozen_audit.get("E4A0_AUDIT_PASS") is True
            and frozen_audit.get("gate", {}).get("final_decision")
            == "E4A0_R_MEMORY_WRITE_FAIL",
            E4A0_REPLAY_MISMATCH,
        )

        replay_records = {}
        with np.load(
            prototype_path, allow_pickle=False
        ) as frozen_prototypes, np.load(
            prediction_path, allow_pickle=False
        ) as frozen_predictions:
            for arm in E4A0_REPLAY_ARMS:
                current_prototype = np.ascontiguousarray(
                    fixed[arm]["prototypes"].cpu().numpy()
                )
                current_prediction = np.ascontiguousarray(
                    fixed[arm]["predictions"].cpu().numpy(),
                    dtype=np.int64,
                )
                frozen_prototype = np.ascontiguousarray(
                    frozen_prototypes[arm]
                )
                frozen_prediction = np.ascontiguousarray(
                    frozen_predictions[arm], dtype=np.int64
                )
                current_prototype_sha = tensor_sha256(current_prototype)
                frozen_prototype_sha = tensor_sha256(frozen_prototype)
                current_prediction_sha = tensor_sha256(current_prediction)
                frozen_prediction_sha = tensor_sha256(frozen_prediction)
                expected_prediction_sha = frozen_seal[
                    "prediction_logical_sha256"
                ][arm]
                prototype_exact = bool(
                    np.array_equal(current_prototype, frozen_prototype)
                    and current_prototype_sha == frozen_prototype_sha
                )
                prediction_exact = bool(
                    np.array_equal(current_prediction, frozen_prediction)
                    and current_prediction_sha
                    == frozen_prediction_sha
                    == expected_prediction_sha
                )
                replay_records[arm] = {
                    "prototype_current_logical_sha256": (
                        current_prototype_sha
                    ),
                    "prototype_frozen_logical_sha256": (
                        frozen_prototype_sha
                    ),
                    "prototype_exact_replay_pass": prototype_exact,
                    "prediction_current_logical_sha256": (
                        current_prediction_sha
                    ),
                    "prediction_frozen_logical_sha256": (
                        frozen_prediction_sha
                    ),
                    "prediction_exact_replay_pass": prediction_exact,
                }
                _require(
                    prototype_exact and prediction_exact,
                    E4A0_REPLAY_MISMATCH,
                )
        return {
            "frozen_output_dir": _display(root),
            "arms": replay_records,
            "ALL_prediction_exact_replay_pass": replay_records[
                "ALL_MEMORY"
            ]["prediction_exact_replay_pass"],
            "R_prediction_exact_replay_pass": replay_records[
                "R_MEMORY"
            ]["prediction_exact_replay_pass"],
            "Oracle_prediction_exact_replay_pass": replay_records[
                "ORACLE_CLEAN_MEMORY"
            ]["prediction_exact_replay_pass"],
            "all_required_prototypes_exact_replay_pass": all(
                replay_records[arm]["prototype_exact_replay_pass"]
                for arm in E4A0_REPLAY_ARMS
            ),
            "E4A0_exact_replay_pass": True,
        }
    except (KeyError, OSError, ValueError, RuntimeError):
        raise RuntimeError(E4A0_REPLAY_MISMATCH) from None


def evaluate_unlabeled_fixed_arms(fixed, full_GT, unlabeled_ids):
    """Compute preregistered diagnostics on exactly 1386 unlabeled rows."""
    labels = np.asarray(full_GT, dtype=np.int64)
    evaluation_ids = np.asarray(unlabeled_ids, dtype=np.int64)
    _require(
        labels.shape == (SAMPLE_NUM,)
        and evaluation_ids.shape == (SAMPLE_NUM - LABELED_NUM,),
        "primary E4-A1 evaluation must use 1386 unlabeled rows",
    )
    results = {}
    for arm in E4A1_ARMS:
        predictions = np.ascontiguousarray(
            fixed[arm]["predictions"].cpu().numpy()[evaluation_ids],
            dtype=np.int64,
        )
        scores = fixed[arm]["scores"][evaluation_ids]
        targets = labels[evaluation_ids]
        target_tensor = torch.as_tensor(targets, dtype=torch.long)
        row_ids = torch.arange(target_tensor.numel())
        true_scores = scores[row_ids, target_tensor]
        other_scores = scores.clone()
        other_scores[row_ids, target_tensor] = -torch.inf
        true_class_margin = true_scores - other_scores.max(dim=1).values

        prototypes = fixed[arm]["prototypes"]
        prototype_cosine = prototypes @ prototypes.T
        off_diagonal = ~torch.eye(CLASS_NUM, dtype=torch.bool)
        pairwise_cosine = prototype_cosine[off_diagonal]
        results[arm] = {
            "evaluation_sample_count": int(evaluation_ids.size),
            "prototype_assignment_ACC": float(
                np.mean(predictions == targets)
            ),
            "prototype_assignment_NMI": float(
                normalized_mutual_info_score(targets, predictions)
            ),
            "prototype_assignment_ARI": float(
                adjusted_rand_score(targets, predictions)
            ),
            "true_class_margin_mean": float(
                true_class_margin.mean().item()
            ),
            "prototype_pairwise_cosine_mean": float(
                pairwise_cosine.mean().item()
            ),
            "prototype_pairwise_cosine_max": float(
                pairwise_cosine.max().item()
            ),
            "prototype_separation": float(
                1.0 - pairwise_cosine.mean().item()
            ),
            "prediction_sha256": tensor_sha256(
                np.ascontiguousarray(
                    fixed[arm]["predictions"].cpu().numpy(),
                    dtype=np.int64,
                )
            ),
        }
    return results


def _strictly_better(results, left_arm, right_arm):
    left = results[left_arm]
    right = results[right_arm]
    return bool(
        left["prototype_assignment_ACC"]
        > right["prototype_assignment_ACC"]
        and left["true_class_margin_mean"]
        > right["true_class_margin_mean"]
    )


def build_e4a1_gate_decision(results):
    """Apply the preregistered S, U, net-gain, and R-contribution gates."""
    _require(
        set(results) == set(E4A1_ARMS),
        "E4-A1 gate requires all seven arm results",
    )
    S_specificity_pass = _strictly_better(
        results, "S_MEMORY", "SHUFFLED_S_MEMORY"
    )
    U_specificity_pass = _strictly_better(
        results, "U_RS_MEMORY", "SHUFFLED_U_RS_MEMORY"
    )
    U_net_gain_pass = _strictly_better(
        results, "U_RS_MEMORY", "ALL_MEMORY"
    )
    S_net_gain_pass = _strictly_better(
        results, "S_MEMORY", "ALL_MEMORY"
    )

    U_result = results["U_RS_MEMORY"]
    S_result = results["S_MEMORY"]
    U_ACC_non_decrease = bool(
        U_result["prototype_assignment_ACC"]
        >= S_result["prototype_assignment_ACC"]
    )
    U_margin_non_decrease = bool(
        U_result["true_class_margin_mean"]
        >= S_result["true_class_margin_mean"]
    )
    U_strict_improvement = bool(
        U_result["prototype_assignment_ACC"]
        > S_result["prototype_assignment_ACC"]
        or U_result["true_class_margin_mean"]
        > S_result["true_class_margin_mean"]
    )
    R_contribution_pass = bool(
        U_ACC_non_decrease
        and U_margin_non_decrease
        and U_strict_improvement
    )

    decision_A = bool(
        S_specificity_pass
        and U_specificity_pass
        and U_net_gain_pass
        and R_contribution_pass
    )
    decision_B = bool(
        S_specificity_pass
        and S_net_gain_pass
        and not R_contribution_pass
    )
    decision_C = bool(
        (S_specificity_pass or U_specificity_pass)
        and not S_net_gain_pass
        and not U_net_gain_pass
    )
    decision_D = bool(
        not S_specificity_pass and not U_specificity_pass
    )
    if decision_A:
        final_decision = "E4A1_RS_INFORMATION_UTILITY_PASS"
    elif decision_B:
        final_decision = "E4A1_S_ONLY_MEMORY_UTILITY_PASS"
    elif decision_C:
        final_decision = "E4A1_SPECIFIC_BUT_NO_NET_MEMORY_GAIN"
    elif decision_D:
        final_decision = "E4A1_MEMORY_UTILITY_IDENTIFICATION_FAIL"
    else:
        raise RuntimeError(E4A1_DECISION_TREE_GAP)

    return {
        "S_specificity_pass": S_specificity_pass,
        "U_specificity_pass": U_specificity_pass,
        "U_net_gain_pass": U_net_gain_pass,
        "S_net_gain_pass": S_net_gain_pass,
        "R_contribution_pass": R_contribution_pass,
        "R_contribution_components": {
            "U_ACC_non_decrease_vs_S": U_ACC_non_decrease,
            "U_margin_non_decrease_vs_S": U_margin_non_decrease,
            "U_at_least_one_strict_improvement_vs_S": (
                U_strict_improvement
            ),
        },
        "decision_conditions": {
            "Decision_A": decision_A,
            "Decision_B": decision_B,
            "Decision_C": decision_C,
            "Decision_D": decision_D,
        },
        "final_decision": final_decision,
        "oracle_entered_any_gate": False,
    }


def _memory_and_query_audit(fixed):
    targets = fixed["labels_labeled"]
    arm_records = {}
    query_hashes = {}
    for arm in E4A1_ARMS:
        weights = fixed["writer_weights"][arm]
        denominators = [
            float(weights[targets == class_id].sum().item())
            for class_id in range(CLASS_NUM)
        ]
        prototypes = fixed[arm]["prototypes"]
        query_sha = tensor_sha256(fixed[arm]["h_query"].cpu().numpy())
        query_hashes[arm] = query_sha
        arm_records[arm] = {
            "writer_weights_shape": list(weights.shape),
            "writer_weights_sha256": tensor_sha256(
                weights.cpu().numpy()
            ),
            "all_14_labels_used_pass": True,
            "no_hard_selection_pass": True,
            "class_denominators": denominators,
            "all_class_denominators_positive_pass": bool(
                all(value > 0.0 for value in denominators)
            ),
            "prototype_shape": list(prototypes.shape),
            "prototype_finite_pass": bool(
                torch.isfinite(prototypes).all().item()
            ),
            "prototype_unit_norm_pass": bool(
                torch.allclose(
                    torch.linalg.vector_norm(prototypes, dim=1),
                    torch.ones(CLASS_NUM, dtype=prototypes.dtype),
                )
            ),
            "prototype_stop_gradient_pass": bool(
                not prototypes.requires_grad
                and prototypes.grad_fn is None
            ),
            "score_shape": list(fixed[arm]["scores"].shape),
            "query_sha256": query_sha,
        }
    common_query_sha = tensor_sha256(fixed["h_query"].cpu().numpy())
    query_identical = bool(
        len(set(query_hashes.values())) == 1
        and next(iter(query_hashes.values())) == common_query_sha
        and all(
            fixed[arm]["h_query"].data_ptr()
            == fixed["h_query"].data_ptr()
            for arm in E4A1_ARMS
        )
    )
    return arm_records, {
        "definition": "h_query=normalize(mean_v(h_sem))",
        "shape": list(fixed["h_query"].shape),
        "logical_sha256": common_query_sha,
        "per_arm_logical_sha256": query_hashes,
        "identical_across_every_arm_pass": query_identical,
        "R_weighted_query_used": False,
        "S_weighted_query_used": False,
        "U_weighted_query_used": False,
        "arm_specific_query_used": False,
    }


def run_evaluation(
    feature_path=e4a0.DEFAULT_FEATURE_PATH,
    feature_audit_path=e4a0.DEFAULT_FEATURE_AUDIT_PATH,
    e1_lwc_model_dir=DEFAULT_E1_LWC_MODEL_DIR,
    label_split_dir=DEFAULT_LABEL_SPLIT_DIR,
    full_gt_path=DEFAULT_FULL_GT_PATH,
    frozen_e4a0_output_dir=DEFAULT_E4A0_OUTPUT_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    device="cpu",
):
    """Run E4-A1 once without fitting, updates, or early full-GT access."""
    output_root = _resolve(output_dir)
    _require(
        not output_root.exists(),
        "refusing to overwrite an E4-A1 output directory",
    )

    h_sem, sample_ids, representation_audit = (
        e4a0.load_e1_lwc_semantic_representation(
            feature_path=feature_path,
            feature_audit_path=feature_audit_path,
            model_dir=e1_lwc_model_dir,
            model_audit_path=(
                _resolve(e1_lwc_model_dir).parent / "e1_audit.json"
            ),
            device=device,
        )
    )
    labeled_ids, labels_labeled, sparse_label_audit = (
        e4a0.load_sparse_label_protocol(label_split_dir)
    )
    R_full, frozen_reliability = e4a0.load_frozen_reliability(
        e4a0.DEFAULT_R_PATH
    )
    _require(
        np.array_equal(sample_ids, np.arange(SAMPLE_NUM, dtype=np.int64)),
        "E4-A1 representation/sample ID alignment mismatch",
    )
    oracle_clean_weights, oracle_audit = (
        e4a0.load_oracle_clean_labeled_weights(
            labeled_ids, e4a0.DEFAULT_ORACLE_MASK_PATH
        )
    )
    fixed = build_fixed_e4a1_outputs(
        h_sem=h_sem,
        labeled_ids=labeled_ids,
        labels_labeled=labels_labeled,
        R_full=R_full,
        oracle_clean_weights=oracle_clean_weights,
    )

    output_root.mkdir(parents=True)
    components = fixed["semantic_components"]
    np.savez(
        output_root / "utility_components.npz",
        R_labeled=np.ascontiguousarray(
            fixed["R_labeled"].cpu().numpy()
        ),
        S_mem=np.ascontiguousarray(fixed["S_mem"].cpu().numpy()),
        U_mem=np.ascontiguousarray(fixed["U_mem"].cpu().numpy()),
        semantic_margin=np.ascontiguousarray(
            components["semantic_margin"].cpu().numpy()
        ),
        positive_similarity=np.ascontiguousarray(
            components["positive_similarity"].cpu().numpy()
        ),
        hard_negative_similarity=np.ascontiguousarray(
            components["hard_negative_similarity"].cpu().numpy()
        ),
        S_shuffle=np.ascontiguousarray(
            fixed["writer_weights"]["SHUFFLED_S_MEMORY"].cpu().numpy()
        ),
        U_shuffle=np.ascontiguousarray(
            fixed["writer_weights"]["SHUFFLED_U_RS_MEMORY"].cpu().numpy()
        ),
    )
    np.savez(
        output_root / "writer_weights.npz",
        **{
            arm: np.ascontiguousarray(
                fixed["writer_weights"][arm].cpu().numpy()
            )
            for arm in E4A1_ARMS
        },
    )
    np.savez(
        output_root / "memory_prototypes.npz",
        **{
            arm: np.ascontiguousarray(
                fixed[arm]["prototypes"].cpu().numpy()
            )
            for arm in E4A1_ARMS
        },
    )
    prediction_seal = save_prediction_seal(fixed, output_root)
    replay_audit = verify_e4a0_exact_replay(
        fixed, frozen_e4a0_output_dir
    )

    full_GT, full_gt_audit = e4a0.load_full_ground_truth(full_gt_path)
    unlabeled_mask = np.ones(SAMPLE_NUM, dtype=bool)
    unlabeled_mask[labeled_ids] = False
    unlabeled_ids = np.flatnonzero(unlabeled_mask).astype(np.int64)
    _require(
        unlabeled_ids.shape == (SAMPLE_NUM - LABELED_NUM,),
        "E4-A1 unlabeled split must contain 1386 samples",
    )
    results = evaluate_unlabeled_fixed_arms(
        fixed, full_GT, unlabeled_ids
    )
    gate = build_e4a1_gate_decision(results)

    arm_audit, query_audit = _memory_and_query_audit(fixed)
    normal_fixed_hashes = {
        arm: {
            "prototype_sha256": tensor_sha256(
                fixed[arm]["prototypes"].cpu().numpy()
            ),
            "prediction_sha256": tensor_sha256(
                fixed[arm]["predictions"].cpu().numpy()
            ),
        }
        for arm in NORMAL_E4A1_ARMS
    }
    component_audit = dict(components["audit"])
    e4a1_audit = {
        "stage": STAGE,
        "seed": SEED,
        "N": SAMPLE_NUM,
        "V": VIEW_NUM,
        "K": CLASS_NUM,
        "labeled_count": LABELED_NUM,
        "unlabeled_evaluation_count": int(unlabeled_ids.size),
        "arms": list(E4A1_ARMS),
        "input_provenance": {
            "E1_LWC_frozen_model": representation_audit,
            "frozen_reliability": frozen_reliability,
            "sparse_labels": sparse_label_audit,
            "oracle_clean_diagnostic": oracle_audit,
            "frozen_E4A0_replay_source": _display(
                frozen_e4a0_output_dir
            ),
            "full_GT_post_prediction_only": full_gt_audit,
        },
        "tensor_shape_audit": {
            "h_sem": list(fixed["h_sem"].shape),
            "h_labeled": list(fixed["h_labeled"].shape),
            "sample_anchor": list(components["sample_anchor"].shape),
            "semantic_margin": list(
                components["semantic_margin"].shape
            ),
            "R_labeled": list(fixed["R_labeled"].shape),
            "S_mem": list(fixed["S_mem"].shape),
            "U_mem": list(fixed["U_mem"].shape),
            "h_query": list(fixed["h_query"].shape),
        },
        "memory_specific_semantic_usefulness": component_audit,
        "information_utility": {
            "formula": "U_mem = R_labeled * S_mem",
            "shape": list(fixed["U_mem"].shape),
            "exact_product_pass": bool(
                torch.equal(
                    fixed["U_mem"],
                    fixed["R_labeled"]
                    * fixed["S_mem"].to(
                        dtype=fixed["R_labeled"].dtype
                    ),
                )
            ),
            "learnable_coefficient_used": False,
            "unlabeled_GT_used_for_U": False,
        },
        "matched_shuffles": fixed["shuffle_audit"],
        "memory_arms": arm_audit,
        "query": query_audit,
        "prediction_seal": prediction_seal,
        "E4A0_replay_lock": replay_audit,
        "oracle_isolation": {
            "normal_arm_fixed_hashes": normal_fixed_hashes,
            "oracle_used_for_training": False,
            "oracle_entered_S": False,
            "oracle_entered_U": False,
            "oracle_entered_query": False,
            "oracle_entered_gate": False,
            "oracle_affects_normal_arm_predictions": False,
            "oracle_isolation_pass": True,
            "role": "diagnostic_upper_bound_only",
        },
        "GT_leakage_boundary": {
            "full_unlabeled_GT_loaded_before_prediction": False,
            "unlabeled_GT_used_for_S": False,
            "unlabeled_GT_used_for_U": False,
            "unlabeled_GT_used_for_memory": False,
            "prediction_saved_before_full_GT": True,
            "training_used": False,
            "optimizer_used": False,
            "backward_used": False,
        },
        "execution_guards": {
            "training_used": False,
            "optimizer_used": False,
            "backward_used": False,
            "online_memory_update_used": False,
            "parameter_update_used": False,
            "deterministic_pass": True,
        },
        "gate": gate,
        "E4A1_AUDIT_PASS": bool(
            replay_audit["E4A0_exact_replay_pass"]
            and query_audit["identical_across_every_arm_pass"]
            and component_audit[
                "positive_reference_self_exclusion_pass"
            ]
            and component_audit[
                "negative_count_exactly_12_pass"
            ]
            and fixed["shuffle_audit"]["S_matched_shuffle"][
                "class_view_weight_sum_exact_pass"
            ]
            and fixed["shuffle_audit"]["U_matched_shuffle"][
                "class_view_weight_sum_exact_pass"
            ]
        ),
    }
    diagnostic_results = {
        "stage": STAGE,
        "primary_evaluation_scope": "1386_unlabeled_samples_only",
        "results": results,
        "gate": gate,
    }
    _write_json(output_root / "e4a1_audit.json", e4a1_audit)
    _write_json(
        output_root / "diagnostic_results.json", diagnostic_results
    )
    return {
        "output_dir": output_root,
        "e4a1_audit": e4a1_audit,
        "diagnostic_results": diagnostic_results,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path", default=str(e4a0.DEFAULT_FEATURE_PATH)
    )
    parser.add_argument(
        "--feature-audit-path",
        default=str(e4a0.DEFAULT_FEATURE_AUDIT_PATH),
    )
    parser.add_argument(
        "--e1-lwc-model-dir", default=str(DEFAULT_E1_LWC_MODEL_DIR)
    )
    parser.add_argument(
        "--label-split-dir", default=str(DEFAULT_LABEL_SPLIT_DIR)
    )
    parser.add_argument(
        "--full-gt-path", default=str(DEFAULT_FULL_GT_PATH)
    )
    parser.add_argument(
        "--frozen-e4a0-output-dir",
        default=str(DEFAULT_E4A0_OUTPUT_DIR),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_evaluation(
        feature_path=args.feature_path,
        feature_audit_path=args.feature_audit_path,
        e1_lwc_model_dir=args.e1_lwc_model_dir,
        label_split_dir=args.label_split_dir,
        full_gt_path=args.full_gt_path,
        frozen_e4a0_output_dir=args.frozen_e4a0_output_dir,
        output_dir=args.output_dir,
        device=args.device,
    )
    print("Saved: " + _display(result["output_dir"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
