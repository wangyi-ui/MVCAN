"""G3-A3 read-only utility-aware semantic target risk diagnostic."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score, roc_auc_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.g2_utility_semantic_consensus import g2_consensus_protocol as g2
from experiments.g3_selective_semantic_cooperation import (
    evaluate_g3_a1_readonly_target_interface as g3_a1,
)
from experiments.g3_selective_semantic_cooperation import (
    g3_target_correction_protocol as g3_protocol,
)
from irv.b4_information_utility import tensor_sha256


STAGE = "G3-A3"
SAMPLE_NUM = 1400
VIEW_NUM = 6
CLASS_NUM = 7
TOP_K = 3
EPS = 1e-12
PROBABILITY_ATOL = 1e-6
BOUND_ATOL = 1e-10
SOURCE_ARMS = {
    "U": "u_correction",
    "SHUFFLE": "shuffled_u_correction",
}
EXPECTED_SOURCE_FILE_SHA256 = {
    "U": "be6763f6f6b916a25a44cb0e1d796f370c04587d4f8af94e0963326fdfce7d0c",
    "SHUFFLE": "d09b5bf7e3104deeb10cc094d4e3771478792c75932fb74b22d7eee48853351a",
}
SEALED_ARRAY_NAMES = (
    "rho_U",
    "rho_shuffle",
    "C_high_U",
    "C_high_shuffle",
    "D_conflict_U",
    "D_conflict_shuffle",
    "E_coop_U",
    "E_coop_shuffle",
    "native_uncertainty",
    "P_all_prediction",
    "P_highU_U_prediction",
    "P_highU_shuffle_prediction",
    "sample_ids",
)
DEFAULT_G3_RUN_DIR = (
    REPOSITORY_ROOT
    / "outputs/g3_selective_semantic_cooperation/g3a0_seed20_100ep"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "outputs/g3_selective_semantic_cooperation/"
    "g3a3_seed20_semantic_target_risk"
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


def _readonly_array(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _validate_probability_rows(probabilities, name, expected_shape=None):
    values = np.asarray(probabilities)
    shape_pass = values.ndim >= 2 and values.shape[-1] == CLASS_NUM
    if expected_shape is not None:
        shape_pass = shape_pass and values.shape == expected_shape
    _require(
        shape_pass
        and np.isfinite(values).all()
        and float(values.min()) >= 0.0
        and np.allclose(
            values.sum(axis=-1), 1.0, rtol=0.0, atol=PROBABILITY_ATOL
        ),
        name + " probability boundary mismatch",
    )


def normalized_js_divergence(first, second):
    """Return JS(p,q)/log(2), using EPS only inside logarithms."""
    p = np.asarray(first, dtype=np.float64)
    q = np.asarray(second, dtype=np.float64)
    _require(p.shape == q.shape and p.ndim >= 1, "JSD shape mismatch")
    _require(
        np.isfinite(p).all()
        and np.isfinite(q).all()
        and float(p.min()) >= 0.0
        and float(q.min()) >= 0.0
        and np.allclose(p.sum(axis=-1), 1.0, rtol=0.0, atol=PROBABILITY_ATOL)
        and np.allclose(q.sum(axis=-1), 1.0, rtol=0.0, atol=PROBABILITY_ATOL),
        "JSD inputs are not probability rows",
    )
    midpoint = 0.5 * (p + q)
    log_midpoint = np.log(np.maximum(midpoint, EPS))
    p_term = np.zeros_like(p)
    q_term = np.zeros_like(q)
    p_positive = p > 0.0
    q_positive = q > 0.0
    p_term[p_positive] = p[p_positive] * (
        np.log(np.maximum(p[p_positive], EPS)) - log_midpoint[p_positive]
    )
    q_term[q_positive] = q[q_positive] * (
        np.log(np.maximum(q[q_positive], EPS)) - log_midpoint[q_positive]
    )
    result = 0.5 * (p_term.sum(axis=-1) + q_term.sum(axis=-1)) / math.log(2.0)
    _require(
        np.isfinite(result).all()
        and float(np.min(result)) >= -BOUND_ATOL
        and float(np.max(result)) <= 1.0 + BOUND_ATOL,
        "normalized JSD is outside [0,1] tolerance",
    )
    return np.ascontiguousarray(result, dtype=np.float64)


def high_u_internal_consistency(aligned_q, admission):
    """Compute 1 minus mean normalized JSD over the three admitted pairs."""
    q = np.asarray(aligned_q)
    mask = np.asarray(admission, dtype=bool)
    _validate_probability_rows(
        q, "aligned_q", (SAMPLE_NUM, VIEW_NUM, CLASS_NUM)
    )
    _require(mask.shape == (SAMPLE_NUM, VIEW_NUM), "admission shape mismatch")
    admitted_count = mask.sum(axis=1)
    _require(
        np.array_equal(admitted_count, np.full(SAMPLE_NUM, TOP_K)),
        "every admission row must contain exactly three views",
    )

    selected_js_sum = np.zeros(SAMPLE_NUM, dtype=np.float64)
    selected_pair_count = np.zeros(SAMPLE_NUM, dtype=np.int64)
    for left in range(VIEW_NUM):
        for right in range(left + 1, VIEW_NUM):
            selected = mask[:, left] & mask[:, right]
            pair_js = normalized_js_divergence(q[:, left], q[:, right])
            selected_js_sum += pair_js * selected
            selected_pair_count += selected.astype(np.int64)
    _require(
        np.array_equal(selected_pair_count, np.full(SAMPLE_NUM, 3)),
        "exactly three high-U view pairs are required per sample",
    )
    consistency = 1.0 - selected_js_sum / selected_pair_count
    _require(
        np.isfinite(consistency).all()
        and float(consistency.min()) >= -BOUND_ATOL
        and float(consistency.max()) <= 1.0 + BOUND_ATOL,
        "C_high is outside [0,1] tolerance",
    )
    return _readonly_array(consistency, dtype=np.float64)


def native_uncertainty(p_all):
    probabilities = np.asarray(p_all, dtype=np.float64)
    _validate_probability_rows(
        probabilities, "P_all", (SAMPLE_NUM, CLASS_NUM)
    )
    safe = np.maximum(probabilities, EPS)
    entropy = -np.sum(probabilities * np.log(safe), axis=1) / math.log(CLASS_NUM)
    _require(
        np.isfinite(entropy).all()
        and float(entropy.min()) >= -BOUND_ATOL
        and float(entropy.max()) <= 1.0 + BOUND_ATOL,
        "native uncertainty is outside [0,1] tolerance",
    )
    return _readonly_array(entropy, dtype=np.float64)


def action_risk_score(rho, consistency, conflict):
    selectivity = np.asarray(rho, dtype=np.float64)
    agreement = np.asarray(consistency, dtype=np.float64)
    divergence = np.asarray(conflict, dtype=np.float64)
    _require(
        selectivity.shape == agreement.shape == divergence.shape == (SAMPLE_NUM,),
        "action-risk input shape mismatch",
    )
    for name, values in (
        ("rho", selectivity),
        ("C_high", agreement),
        ("D_conflict", divergence),
    ):
        _require(
            np.isfinite(values).all()
            and float(values.min()) >= -BOUND_ATOL
            and float(values.max()) <= 1.0 + BOUND_ATOL,
            name + " is outside [0,1] tolerance",
        )
    risk = selectivity * agreement * divergence
    _require(
        np.isfinite(risk).all()
        and float(risk.min()) >= -BOUND_ATOL
        and float(risk.max()) <= 1.0 + BOUND_ATOL,
        "E_coop is outside [0,1] tolerance",
    )
    return _readonly_array(risk, dtype=np.float64)


def array_distribution(values):
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 1 and array.size > 0 and np.isfinite(array).all(),
             "distribution input boundary mismatch")
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p01": float(np.percentile(array, 1)),
        "p05": float(np.percentile(array, 5)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def _array_audit(values):
    array = np.asarray(values)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": tensor_sha256(array),
        "readonly": bool(not array.flags.writeable),
    }


def construct_fixed_scores(g3_run_dir=DEFAULT_G3_RUN_DIR):
    """Construct and hash every score/prediction without loading labels."""
    run_root = _resolve(g3_run_dir)
    sources = {
        arm: g3_a1.load_epoch0_archive(
            run_root / directory / "refresh_epoch_0000.npz"
        )
        for arm, directory in SOURCE_ARMS.items()
    }
    for arm, expected_hash in EXPECTED_SOURCE_FILE_SHA256.items():
        _require(
            sources[arm]["file_sha256"] == expected_hash,
            "frozen G3-A0 source file hash mismatch for " + arm,
        )

    u = sources["U"]["arrays"]
    shuffled = sources["SHUFFLE"]["arrays"]
    for key in (
        "P_all",
        "aligned_q",
        "sample_ids",
        "fusion_weights_used_for_P_all",
    ):
        _require(
            np.array_equal(u[key], shuffled[key]),
            "U/Shuffle shared frozen quantity differs: " + key,
        )
    _require(
        np.array_equal(u["sample_ids"], np.arange(SAMPLE_NUM, dtype=np.int64)),
        "frozen sample IDs are not arange(N)",
    )

    p_all = _readonly_array(u["P_all"], dtype=np.float32)
    aligned_q = _readonly_array(u["aligned_q"], dtype=np.float32)
    p_high_u = _readonly_array(u["P_highU"], dtype=np.float32)
    p_high_shuffle = _readonly_array(shuffled["P_highU"], dtype=np.float32)
    admission_u = _readonly_array(u["admission"], dtype=bool)
    admission_shuffle = _readonly_array(shuffled["admission"], dtype=bool)
    rho_u = _readonly_array(u["rho"], dtype=np.float64)
    rho_shuffle = _readonly_array(shuffled["rho"], dtype=np.float64)
    sample_ids = _readonly_array(u["sample_ids"], dtype=np.int64)
    fusion_weights = _readonly_array(
        u["fusion_weights_used_for_P_all"], dtype=np.float64
    )

    _validate_probability_rows(p_all, "P_all", (SAMPLE_NUM, CLASS_NUM))
    _validate_probability_rows(
        aligned_q, "aligned_q", (SAMPLE_NUM, VIEW_NUM, CLASS_NUM)
    )
    _validate_probability_rows(
        p_high_u, "P_highU_U", (SAMPLE_NUM, CLASS_NUM)
    )
    _validate_probability_rows(
        p_high_shuffle, "P_highU_shuffle", (SAMPLE_NUM, CLASS_NUM)
    )
    c_high_u = high_u_internal_consistency(aligned_q, admission_u)
    c_high_shuffle = high_u_internal_consistency(aligned_q, admission_shuffle)
    d_conflict_u = _readonly_array(
        normalized_js_divergence(p_high_u, p_all), dtype=np.float64
    )
    d_conflict_shuffle = _readonly_array(
        normalized_js_divergence(p_high_shuffle, p_all), dtype=np.float64
    )
    e_coop_u = action_risk_score(rho_u, c_high_u, d_conflict_u)
    e_coop_shuffle = action_risk_score(
        rho_shuffle, c_high_shuffle, d_conflict_shuffle
    )
    uncertainty = native_uncertainty(p_all)

    arrays = {
        "rho_U": rho_u,
        "rho_shuffle": rho_shuffle,
        "C_high_U": c_high_u,
        "C_high_shuffle": c_high_shuffle,
        "D_conflict_U": d_conflict_u,
        "D_conflict_shuffle": d_conflict_shuffle,
        "E_coop_U": e_coop_u,
        "E_coop_shuffle": e_coop_shuffle,
        "native_uncertainty": uncertainty,
        "P_all_prediction": _readonly_array(
            np.argmax(p_all, axis=1), dtype=np.int64
        ),
        "P_highU_U_prediction": _readonly_array(
            np.argmax(p_high_u, axis=1), dtype=np.int64
        ),
        "P_highU_shuffle_prediction": _readonly_array(
            np.argmax(p_high_shuffle, axis=1), dtype=np.int64
        ),
        "sample_ids": sample_ids,
    }
    _require(set(arrays) == set(SEALED_ARRAY_NAMES), "sealed array set mismatch")
    _require(
        all(not array.flags.writeable for array in arrays.values()),
        "one or more score arrays remain writable",
    )
    hashes = {name: _array_audit(array) for name, array in arrays.items()}
    distributions = {
        name: array_distribution(arrays[name])
        for name in (
            "rho_U",
            "rho_shuffle",
            "C_high_U",
            "C_high_shuffle",
            "D_conflict_U",
            "D_conflict_shuffle",
            "E_coop_U",
            "E_coop_shuffle",
            "native_uncertainty",
        )
    }
    source_audit = {
        arm: {
            "path": _display(source["path"]),
            "file_sha256": source["file_sha256"],
            "keys": source["keys"],
            "required_arrays": {
                key: source["schema"][key]
                for key in (
                    "P_all",
                    "P_highU",
                    "aligned_q",
                    "admission",
                    "rho",
                    "sample_ids",
                    "fusion_weights_used_for_P_all",
                )
            },
        }
        for arm, source in sources.items()
    }
    return {
        "arrays": arrays,
        "hashes": hashes,
        "distributions": distributions,
        "source_audit": source_audit,
        "fusion_weights": fusion_weights,
        "fusion_weights_audit": _array_audit(fusion_weights),
        "admission_exactly_three": {
            "U": bool(np.all(admission_u.sum(axis=1) == TOP_K)),
            "SHUFFLE": bool(np.all(admission_shuffle.sum(axis=1) == TOP_K)),
        },
        "shared_frozen_quantities_exact": True,
    }


def fit_global_cluster_mapping_once(y_true, y_all_cluster, class_num=CLASS_NUM):
    """Fit one P_all cluster-to-ground-truth Hungarian mapping."""
    labels = np.asarray(y_true, dtype=np.int64)
    clusters = np.asarray(y_all_cluster, dtype=np.int64)
    _require(
        labels.shape == clusters.shape
        and labels.ndim == 1
        and labels.size > 0
        and int(labels.min()) >= 0
        and int(labels.max()) < class_num
        and int(clusters.min()) >= 0
        and int(clusters.max()) < class_num,
        "Hungarian mapping input boundary mismatch",
    )
    counts = np.zeros((class_num, class_num), dtype=np.int64)
    np.add.at(counts, (clusters, labels), 1)
    rows, columns = linear_sum_assignment(counts.max() - counts)
    mapping = np.full(class_num, -1, dtype=np.int64)
    mapping[rows] = columns
    _require(np.all(mapping >= 0), "Hungarian mapping is incomplete")
    return {
        "mapping": _readonly_array(mapping, dtype=np.int64),
        "contingency": _readonly_array(counts, dtype=np.int64),
        "matched_count": int(counts[rows, columns].sum()),
        "mapping_fit_source": "P_all_prediction_only",
        "mapping_fit_count": 1,
    }


def rank_diagnostic(binary_outcome, score):
    outcome = np.asarray(binary_outcome, dtype=np.int64)
    values = np.asarray(score, dtype=np.float64)
    _require(
        outcome.shape == values.shape == (SAMPLE_NUM,)
        and np.array_equal(np.unique(outcome), np.array([0, 1]))
        and np.isfinite(values).all(),
        "rank diagnostic boundary mismatch",
    )
    correct = outcome == 0
    wrong = outcome == 1
    correct_mean = float(values[correct].mean())
    wrong_mean = float(values[wrong].mean())
    return {
        "ROC_AUC": float(roc_auc_score(outcome, values)),
        "PR_AUC": float(average_precision_score(outcome, values)),
        "P_all_correct_count": int(correct.sum()),
        "P_all_wrong_count": int(wrong.sum()),
        "mean_score_P_all_correct": correct_mean,
        "mean_score_P_all_wrong": wrong_mean,
        "wrong_minus_correct_mean_gap": float(wrong_mean - correct_mean),
    }


def error_enrichment(binary_error, score, sample_ids):
    errors = np.asarray(binary_error, dtype=np.int64)
    values = np.asarray(score, dtype=np.float64)
    identifiers = np.asarray(sample_ids, dtype=np.int64)
    _require(
        errors.shape == values.shape == identifiers.shape == (SAMPLE_NUM,)
        and set(np.unique(errors)).issubset({0, 1})
        and np.isfinite(values).all(),
        "enrichment input boundary mismatch",
    )
    global_rate = float(errors.mean())
    _require(global_rate > 0.0, "error enrichment is undefined with zero errors")
    order = np.lexsort((identifiers, -values))
    result = {"global_error_rate": global_rate, "ranking_tie_break": "sample_id_ascending"}
    for percentage in (10, 20, 30):
        count = int(math.ceil(SAMPLE_NUM * percentage / 100.0))
        selected = order[:count]
        error_count = int(errors[selected].sum())
        error_rate = float(errors[selected].mean())
        result["top_" + str(percentage) + "pct"] = {
            "sample_count": count,
            "error_count": error_count,
            "error_rate": error_rate,
            "enrichment_over_global": float(error_rate / global_rate),
        }
    return result


def action_risk_candidate_gate(primary, shuffled, rho_only):
    conditions = {
        "E_coop_U_ROC_AUC_gt_0p5": bool(primary["ROC_AUC"] > 0.5),
        "E_coop_U_ROC_AUC_gt_shuffle": bool(
            primary["ROC_AUC"] > shuffled["ROC_AUC"]
        ),
        "E_coop_U_ROC_AUC_gt_rho": bool(
            primary["ROC_AUC"] > rho_only["ROC_AUC"]
        ),
        "E_coop_wrong_mean_gt_correct_mean": bool(
            primary["mean_score_P_all_wrong"]
            > primary["mean_score_P_all_correct"]
        ),
    }
    return {
        "conditions": conditions,
        "magnitude_threshold": None,
        "native_uncertainty_is_hard_gate": False,
        "ACTION_RISK_CANDIDATE_PASS": bool(all(conditions.values())),
    }


def semantic_action_diagnostic(
    y_true,
    y_all_cluster,
    y_high_u_cluster,
    mapping,
    e_coop_u,
):
    labels = np.asarray(y_true, dtype=np.int64)
    all_cluster = np.asarray(y_all_cluster, dtype=np.int64)
    high_cluster = np.asarray(y_high_u_cluster, dtype=np.int64)
    cluster_to_class = np.asarray(mapping, dtype=np.int64)
    risk = np.asarray(e_coop_u, dtype=np.float64)
    _require(
        labels.shape == all_cluster.shape == high_cluster.shape == risk.shape
        == (SAMPLE_NUM,)
        and cluster_to_class.shape == (CLASS_NUM,),
        "semantic-action input boundary mismatch",
    )
    all_correct = cluster_to_class[all_cluster] == labels
    high_correct = cluster_to_class[high_cluster] == labels
    override = high_cluster != all_cluster
    beneficial = override & high_correct & ~all_correct
    harmful = override & ~high_correct & all_correct
    both_wrong = override & ~high_correct & ~all_correct
    both_correct = override & high_correct & all_correct
    _require(
        int(beneficial.sum() + harmful.sum() + both_wrong.sum() + both_correct.sum())
        == int(override.sum()),
        "semantic-action categories do not partition overrides",
    )
    decisive = beneficial | harmful
    beneficial_count = int(beneficial.sum())
    harmful_count = int(harmful.sum())
    if beneficial_count == 0 or harmful_count == 0:
        decisive_auc = {
            "status": "INSUFFICIENT_ONE_CLASS_EMPTY",
            "ROC_AUC": None,
            "positive_class": "BENEFICIAL_OVERRIDE",
        }
    else:
        decisive_auc = {
            "status": "AVAILABLE",
            "ROC_AUC": float(
                roc_auc_score(beneficial[decisive].astype(np.int64), risk[decisive])
            ),
            "positive_class": "BENEFICIAL_OVERRIDE",
        }
    return {
        "override_condition": "P_highU_U_prediction != P_all_prediction",
        "override_count": int(override.sum()),
        "BENEFICIAL_OVERRIDE": beneficial_count,
        "HARMFUL_OVERRIDE": harmful_count,
        "BOTH_WRONG": int(both_wrong.sum()),
        "BOTH_CORRECT": int(both_correct.sum()),
        "decisive_subset_count": int(decisive.sum()),
        "decisive_subset_E_coop_U": decisive_auc,
    }


def evaluate_after_score_seal(fixed, data_path):
    """Load labels only after every score and prediction has been sealed."""
    labels = g2.load_caltech_labels_only(data_path)
    arrays = fixed["arrays"]
    mapping_record = fit_global_cluster_mapping_once(
        labels, arrays["P_all_prediction"]
    )
    mapping = mapping_record["mapping"]
    mapped_all = mapping[arrays["P_all_prediction"]]
    p_all_error = _readonly_array(mapped_all != labels, dtype=np.int64)
    score_sources = {
        "E_COOP_U": arrays["E_coop_U"],
        "RHO_ONLY": arrays["rho_U"],
        "CONFLICT_ONLY": arrays["D_conflict_U"],
        "HIGHU_CONSISTENCY_ONLY": arrays["C_high_U"],
        "NATIVE_UNCERTAINTY": arrays["native_uncertainty"],
        "SHUFFLED_ACTION_RISK": arrays["E_coop_shuffle"],
    }
    diagnostics = {
        name: rank_diagnostic(p_all_error, score)
        for name, score in score_sources.items()
    }
    gate = action_risk_candidate_gate(
        diagnostics["E_COOP_U"],
        diagnostics["SHUFFLED_ACTION_RISK"],
        diagnostics["RHO_ONLY"],
    )
    action = semantic_action_diagnostic(
        labels,
        arrays["P_all_prediction"],
        arrays["P_highU_U_prediction"],
        mapping,
        arrays["E_coop_U"],
    )
    return {
        "P_all_error_definition": "mapped_once(P_all_argmax) != ground_truth",
        "P_all_error_sha256": tensor_sha256(p_all_error),
        "P_all_error_count": int(p_all_error.sum()),
        "P_all_error_rate": float(p_all_error.mean()),
        "hungarian_mapping": {
            "cluster_to_ground_truth_class": mapping.tolist(),
            "contingency": mapping_record["contingency"].tolist(),
            "matched_count": mapping_record["matched_count"],
            "mapping_fit_source": mapping_record["mapping_fit_source"],
            "mapping_fit_count": mapping_record["mapping_fit_count"],
            "reused_for_all_correctness_checks": True,
        },
        "rank_diagnostics": diagnostics,
        "error_enrichment": error_enrichment(
            p_all_error, arrays["E_coop_U"], arrays["sample_ids"]
        ),
        "semantic_action_diagnostic": action,
        "gate": gate,
    }


def _protocol_record(g3_run_dir):
    return {
        "stage": STAGE,
        "scope": "read_only_frozen_epoch0_semantic_target_risk",
        "source_G3_run_dir": _display(g3_run_dir),
        "source_arms": SOURCE_ARMS,
        "definitions": {
            "normalized_JSD": "JS(p,q) / log(2)",
            "C_high": "1 - mean_pairwise_normalized_JSD_over_exactly_3_admitted_views",
            "D_conflict": "normalized_JSD(P_highU, P_all)",
            "E_coop": "rho * C_high * D_conflict",
            "native_uncertainty": "entropy(P_all) / log(K)",
        },
        "fixed_constants": {
            "sample_num": SAMPLE_NUM,
            "view_num": VIEW_NUM,
            "class_num": CLASS_NUM,
            "top_k": TOP_K,
            "eps_for_log_only": EPS,
            "probability_atol": PROBABILITY_ATOL,
            "bound_atol": BOUND_ATOL,
        },
        "label_boundary": (
            "all score arrays and argmax predictions are saved and hashed before "
            "load_caltech_labels_only is called"
        ),
        "mapping_rule": (
            "fit one Hungarian P_all cluster-to-ground-truth mapping and reuse it "
            "for every correctness check"
        ),
        "gate_rule": (
            "AUC(E_U)>0.5 and AUC(E_U)>AUC(E_shuffle) and "
            "AUC(E_U)>AUC(rho_U) and mean(E_U|wrong)>mean(E_U|correct)"
        ),
        "scientific_magnitude_threshold": None,
        "top_percentiles_are_future_thresholds": False,
    }


def run_diagnostic(
    g3_run_dir=DEFAULT_G3_RUN_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    data_path=g3_protocol.DEFAULT_DATA_PATH,
):
    run_root = _resolve(g3_run_dir)
    output_root = _resolve(output_dir)
    data_source = _resolve(data_path)
    _require(run_root.is_dir(), "G3-A0 formal run directory is missing")
    _require(not output_root.exists(), "refusing to overwrite a G3-A3 output")

    fixed = construct_fixed_scores(run_root)
    output_root.mkdir(parents=True)
    _write_json(output_root / "protocol.json", _protocol_record(run_root))
    np.savez_compressed(output_root / "sealed_scores.npz", **fixed["arrays"])
    score_audit = {
        "stage": STAGE,
        "scores_and_predictions_sealed_before_labels": True,
        "sealed_arrays": fixed["hashes"],
        "label_free_distributions": fixed["distributions"],
        "source_audit": fixed["source_audit"],
        "fusion_weights": fixed["fusion_weights"].tolist(),
        "fusion_weights_audit": fixed["fusion_weights_audit"],
        "admission_exactly_three": fixed["admission_exactly_three"],
        "shared_frozen_quantities_exact": fixed[
            "shared_frozen_quantities_exact"
        ],
    }
    _write_json(output_root / "score_audit.json", score_audit)

    evaluation = evaluate_after_score_seal(fixed, data_source)
    engineering_checks = {
        "source_file_hashes_pass": True,
        "epoch0_schema_pass": True,
        "U_shuffle_share_P_all_aligned_q_sample_ids_weights_pass": bool(
            fixed["shared_frozen_quantities_exact"]
        ),
        "both_admission_rows_exactly_three_pass": bool(
            all(fixed["admission_exactly_three"].values())
        ),
        "all_scores_finite_and_bounded_pass": bool(
            all(
                np.isfinite(fixed["arrays"][name]).all()
                and float(fixed["arrays"][name].min()) >= -BOUND_ATOL
                and float(fixed["arrays"][name].max()) <= 1.0 + BOUND_ATOL
                for name in (
                    "rho_U",
                    "rho_shuffle",
                    "C_high_U",
                    "C_high_shuffle",
                    "D_conflict_U",
                    "D_conflict_shuffle",
                    "E_coop_U",
                    "E_coop_shuffle",
                    "native_uncertainty",
                )
            )
        ),
        "sealed_arrays_readonly_pass": bool(
            all(not value.flags.writeable for value in fixed["arrays"].values())
        ),
        "scores_and_predictions_sealed_before_labels_pass": True,
        "single_P_all_Hungarian_mapping_pass": bool(
            evaluation["hungarian_mapping"]["mapping_fit_count"] == 1
            and evaluation["hungarian_mapping"][
                "reused_for_all_correctness_checks"
            ]
        ),
        "rank_only_no_classifier_fit_pass": True,
        "read_only_no_model_or_D2_execution_pass": True,
    }
    result = {
        "stage": STAGE,
        "scope": "read_only_frozen_epoch0_semantic_target_risk",
        "source_audit": fixed["source_audit"],
        "sealed_array_hashes": fixed["hashes"],
        "score_distributions": fixed["distributions"],
        **evaluation,
        "scientific_magnitude_threshold": None,
        "engineering_checks": engineering_checks,
        "ENGINEERING_PASS": bool(all(engineering_checks.values())),
        "ACTION_RISK_CANDIDATE_PASS": evaluation["gate"][
            "ACTION_RISK_CANDIDATE_PASS"
        ],
    }
    summary = {
        "stage": STAGE,
        "ENGINEERING_PASS": result["ENGINEERING_PASS"],
        "C_high_distribution": {
            "U": fixed["distributions"]["C_high_U"],
            "SHUFFLE": fixed["distributions"]["C_high_shuffle"],
        },
        "D_conflict_distribution": {
            "U": fixed["distributions"]["D_conflict_U"],
            "SHUFFLE": fixed["distributions"]["D_conflict_shuffle"],
        },
        "E_coop_distribution": {
            "U": fixed["distributions"]["E_coop_U"],
            "SHUFFLE": fixed["distributions"]["E_coop_shuffle"],
        },
        "rank_diagnostics": evaluation["rank_diagnostics"],
        "error_enrichment": evaluation["error_enrichment"],
        "semantic_action_diagnostic": evaluation[
            "semantic_action_diagnostic"
        ],
        "gate": evaluation["gate"],
        "ACTION_RISK_CANDIDATE_PASS": result[
            "ACTION_RISK_CANDIDATE_PASS"
        ],
    }
    _write_json(output_root / "results.json", result)
    _write_json(output_root / "summary.json", summary)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="G3-A3 read-only semantic target risk diagnostic"
    )
    parser.add_argument("--g3-run-dir", type=Path, default=DEFAULT_G3_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-path", type=Path, default=g3_protocol.DEFAULT_DATA_PATH)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_diagnostic(
        g3_run_dir=args.g3_run_dir,
        output_dir=args.output_dir,
        data_path=args.data_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
