"""G3-A2 read-only confidence-preserving semantic target diagnostic."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


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


STAGE = "G3-A2"
SAMPLE_NUM = 1400
CLASS_NUM = 7
EPS = 1e-12
PROBABILITY_ATOL = 1e-6
ENTROPY_SOLVE_ATOL = 1e-12
ENTROPY_PASS_ATOL = 1e-10
IDENTITY_ATOL = 1e-7
DEGENERATE_LOGIT_SPREAD_ATOL = 1e-14
MAX_BRACKET_STEPS = 200
MAX_BISECTION_STEPS = 200
TARGET_NAMES = (
    "P_all",
    "P_highU_U",
    "P_highU_SHUFFLE",
    "P_cp_U",
    "P_cp_SHUFFLE",
)
SOURCE_ARMS = {
    "U": "u_correction",
    "SHUFFLE": "shuffled_u_correction",
}
EXPECTED_SOURCE_FILE_SHA256 = {
    "U": "be6763f6f6b916a25a44cb0e1d796f370c04587d4f8af94e0963326fdfce7d0c",
    "SHUFFLE": "d09b5bf7e3104deeb10cc094d4e3771478792c75932fb74b22d7eee48853351a",
}
DEFAULT_G3_RUN_DIR = (
    REPOSITORY_ROOT
    / "outputs/g3_selective_semantic_cooperation/g3a0_seed20_100ep"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "outputs/g3_selective_semantic_cooperation/"
    "g3a2_seed20_confidence_preserving_target"
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


def _validate_probability_rows(probabilities, name):
    values = np.asarray(probabilities)
    _require(
        values.ndim == 2
        and values.shape[1] == CLASS_NUM
        and np.isfinite(values).all()
        and float(values.min()) >= 0.0
        and np.allclose(
            values.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_ATOL
        ),
        name + " probability boundary mismatch",
    )


def entropy_per_sample(probabilities):
    values = np.asarray(probabilities, dtype=np.float64)
    _validate_probability_rows(values, "entropy input")
    safe = np.clip(values, np.finfo(np.float64).tiny, 1.0)
    return np.ascontiguousarray(-np.sum(values * np.log(safe), axis=1))


def log_space_semantic_correction(p_all, p_high_u, rho):
    """Build l_corr=(1-rho)log(P_all)+rho*log(P_highU)."""
    native = np.asarray(p_all, dtype=np.float64)
    high = np.asarray(p_high_u, dtype=np.float64)
    selectivity = np.asarray(rho, dtype=np.float64)
    _validate_probability_rows(native, "P_all")
    _validate_probability_rows(high, "P_highU")
    _require(
        native.shape == high.shape
        and selectivity.shape == (native.shape[0],)
        and np.isfinite(selectivity).all()
        and float(selectivity.min()) >= 0.0
        and float(selectivity.max()) <= 1.0,
        "log-correction input boundary mismatch",
    )
    l_all = np.log(np.clip(native, EPS, None))
    l_high = np.log(np.clip(high, EPS, None))
    semantic_residual = l_high - l_all
    corrected_logits = l_all + selectivity[:, None] * semantic_residual
    _require(
        corrected_logits.shape == native.shape
        and np.isfinite(corrected_logits).all(),
        "corrected logit boundary mismatch",
    )
    return {
        "l_all": np.ascontiguousarray(l_all),
        "l_high": np.ascontiguousarray(l_high),
        "semantic_residual": np.ascontiguousarray(semantic_residual),
        "l_corr": np.ascontiguousarray(corrected_logits),
    }


def _softmax_at_temperature(logits, tau):
    tau = float(tau)
    _require(math.isfinite(tau) and tau > 0.0, "tau must be finite and positive")
    centered = np.asarray(logits, dtype=np.float64)
    centered = centered - float(np.max(centered))
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        exponent = np.exp(centered / tau)
    mass = float(exponent.sum())
    _require(math.isfinite(mass) and mass > 0.0, "softmax mass is invalid")
    return exponent / mass


def _entropy_at_temperature(logits, tau):
    probability = _softmax_at_temperature(logits, tau)
    safe = np.clip(probability, np.finfo(np.float64).tiny, 1.0)
    return float(-np.sum(probability * np.log(safe)))


def solve_entropy_preserving_temperature(corrected_logits, p_all):
    """Deterministically solve one positive tau per sample by bisection."""
    logits = np.asarray(corrected_logits, dtype=np.float64)
    native = np.asarray(p_all, dtype=np.float64)
    _require(
        logits.shape == native.shape
        and logits.ndim == 2
        and logits.shape[1] == CLASS_NUM
        and np.isfinite(logits).all(),
        "entropy projection input boundary mismatch",
    )
    _validate_probability_rows(native, "projection P_all")
    target_entropy = entropy_per_sample(native)
    sample_num = int(logits.shape[0])
    tau = np.empty(sample_num, dtype=np.float64)
    projected = np.empty_like(logits, dtype=np.float64)
    iterations = np.zeros(sample_num, dtype=np.int64)
    uniform_degenerate_count = 0

    for sample_id in range(sample_num):
        row = logits[sample_id]
        entropy_target = float(target_entropy[sample_id])
        spread = float(np.max(row) - np.min(row))
        if spread <= DEGENERATE_LOGIT_SPREAD_ATOL:
            uniform_entropy = math.log(CLASS_NUM)
            if abs(entropy_target - uniform_entropy) > ENTROPY_PASS_ATOL:
                raise RuntimeError(
                    "degenerate corrected logits cannot match P_all entropy at sample "
                    + str(sample_id)
                )
            tau[sample_id] = 1.0
            projected[sample_id] = _softmax_at_temperature(row, 1.0)
            uniform_degenerate_count += 1
            continue

        entropy_one = _entropy_at_temperature(row, 1.0)
        if abs(entropy_one - entropy_target) <= ENTROPY_SOLVE_ATOL:
            tau[sample_id] = 1.0
            projected[sample_id] = _softmax_at_temperature(row, 1.0)
            continue

        if entropy_one > entropy_target:
            upper = 1.0
            lower = 0.5
            for bracket_step in range(MAX_BRACKET_STEPS):
                if _entropy_at_temperature(row, lower) <= entropy_target:
                    break
                lower *= 0.5
            else:
                raise RuntimeError(
                    "failed to bracket lower tau at sample " + str(sample_id)
                )
        else:
            lower = 1.0
            upper = 2.0
            for bracket_step in range(MAX_BRACKET_STEPS):
                if _entropy_at_temperature(row, upper) >= entropy_target:
                    break
                upper *= 2.0
            else:
                raise RuntimeError(
                    "failed to bracket upper tau at sample " + str(sample_id)
                )

        solution = None
        for step_id in range(MAX_BISECTION_STEPS):
            midpoint = 0.5 * (lower + upper)
            midpoint_entropy = _entropy_at_temperature(row, midpoint)
            iterations[sample_id] = step_id + 1
            if abs(midpoint_entropy - entropy_target) <= ENTROPY_SOLVE_ATOL:
                solution = midpoint
                break
            if midpoint_entropy < entropy_target:
                lower = midpoint
            else:
                upper = midpoint
        if solution is None:
            solution = 0.5 * (lower + upper)
        tau[sample_id] = solution
        projected[sample_id] = _softmax_at_temperature(row, solution)

    projected_entropy = entropy_per_sample(projected)
    absolute_error = np.abs(projected_entropy - target_entropy)
    _require(
        np.isfinite(tau).all()
        and np.all(tau > 0.0)
        and np.isfinite(projected).all(),
        "tau/projection validity failure",
    )
    _validate_probability_rows(projected, "P_cp")
    return {
        "tau": _readonly_array(tau),
        "P_cp": _readonly_array(projected),
        "target_entropy": _readonly_array(target_entropy),
        "projected_entropy": _readonly_array(projected_entropy),
        "absolute_entropy_error": _readonly_array(absolute_error),
        "iterations": _readonly_array(iterations),
        "uniform_degenerate_count": int(uniform_degenerate_count),
    }


def _tau_distribution(tau):
    values = np.asarray(tau, dtype=np.float64)
    _require(
        values.shape == (SAMPLE_NUM,)
        and np.isfinite(values).all()
        and np.all(values > 0.0),
        "tau distribution boundary mismatch",
    )
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _projection_audit(projection, p_all, rho):
    error = np.asarray(projection["absolute_entropy_error"], dtype=np.float64)
    projected = np.asarray(projection["P_cp"], dtype=np.float64)
    native = np.asarray(p_all, dtype=np.float64)
    selectivity = np.asarray(rho, dtype=np.float64)
    zero_mask = selectivity == 0.0
    if np.any(zero_mask):
        identity_absolute = np.abs(projected[zero_mask] - native[zero_mask])
        identity_mean = float(identity_absolute.mean())
        identity_max = float(identity_absolute.max())
    else:
        identity_mean = 0.0
        identity_max = 0.0
    probability_valid = bool(
        np.isfinite(projected).all()
        and float(projected.min()) >= 0.0
        and np.allclose(
            projected.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_ATOL
        )
    )
    entropy_pass = bool(float(error.max()) <= ENTROPY_PASS_ATOL)
    identity_pass = bool(not np.any(zero_mask) or identity_max <= IDENTITY_ATOL)
    tau_valid = bool(
        np.isfinite(projection["tau"]).all()
        and np.all(projection["tau"] > 0.0)
    )
    return {
        "mean_absolute_entropy_error": float(error.mean()),
        "max_absolute_entropy_error": float(error.max()),
        "p95_absolute_entropy_error": float(np.percentile(error, 95)),
        "entropy_preservation_atol": ENTROPY_PASS_ATOL,
        "entropy_preservation_pass": entropy_pass,
        "probability_validity_pass": probability_valid,
        "tau_validity_pass": tau_valid,
        "rho_zero_count": int(zero_mask.sum()),
        "rho_zero_identity_applicable": bool(np.any(zero_mask)),
        "rho_zero_identity_mean_absolute_error": identity_mean,
        "rho_zero_identity_max_absolute_error": identity_max,
        "rho_zero_identity_atol": IDENTITY_ATOL,
        "rho_zero_identity_pass": identity_pass,
        "uniform_degenerate_count": projection["uniform_degenerate_count"],
        "tau_distribution": _tau_distribution(projection["tau"]),
        "bisection_iterations_max": int(projection["iterations"].max()),
        "projection_valid": bool(
            entropy_pass and probability_valid and tau_valid and identity_pass
        ),
    }


def _build_cp_arm(p_all, p_high_u, rho):
    correction = log_space_semantic_correction(p_all, p_high_u, rho)
    projection = solve_entropy_preserving_temperature(
        correction["l_corr"], p_all
    )
    audit = _projection_audit(projection, p_all, rho)
    return {
        "correction": correction,
        "projection": projection,
        "audit": audit,
    }


def construct_fixed_outputs(g3_run_dir=DEFAULT_G3_RUN_DIR):
    """Construct and hash both CP arms without loading labels."""
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
    u_arrays = sources["U"]["arrays"]
    shuffled_arrays = sources["SHUFFLE"]["arrays"]
    _require(
        np.array_equal(u_arrays["P_all"], shuffled_arrays["P_all"])
        and np.array_equal(u_arrays["sample_ids"], shuffled_arrays["sample_ids"]),
        "U/Shuffle P_all or sample IDs differ",
    )
    _require(
        np.array_equal(
            u_arrays["sample_ids"], np.arange(SAMPLE_NUM, dtype=np.int64)
        ),
        "frozen sample IDs are not arange(N)",
    )

    p_all = _readonly_array(u_arrays["P_all"], dtype=np.float32)
    p_high_u = _readonly_array(u_arrays["P_highU"], dtype=np.float32)
    p_high_shuffle = _readonly_array(
        shuffled_arrays["P_highU"], dtype=np.float32
    )
    rho_u = _readonly_array(u_arrays["rho"], dtype=np.float64)
    rho_shuffle = _readonly_array(
        shuffled_arrays["rho"], dtype=np.float64
    )
    cp_u = _build_cp_arm(p_all, p_high_u, rho_u)
    cp_shuffle = _build_cp_arm(p_all, p_high_shuffle, rho_shuffle)
    targets = {
        "P_all": p_all,
        "P_highU_U": p_high_u,
        "P_highU_SHUFFLE": p_high_shuffle,
        "P_cp_U": cp_u["projection"]["P_cp"],
        "P_cp_SHUFFLE": cp_shuffle["projection"]["P_cp"],
    }
    for name, target in targets.items():
        _validate_probability_rows(target, name)
    predictions = {
        name: _readonly_array(np.argmax(target, axis=1), dtype=np.int64)
        for name, target in targets.items()
    }
    tau = {
        "tau_U": cp_u["projection"]["tau"],
        "tau_SHUFFLE": cp_shuffle["projection"]["tau"],
    }
    fixed_hashes = {
        name: {
            "target_sha256": tensor_sha256(targets[name]),
            "prediction_sha256": tensor_sha256(predictions[name]),
            "target_shape": list(targets[name].shape),
            "target_dtype": str(targets[name].dtype),
        }
        for name in TARGET_NAMES
    }
    fixed_hashes.update({
        name: {
            "sha256": tensor_sha256(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in tau.items()
    })
    diagnostics = {
        name: g3_a1.score_diagnostics(targets[name]) for name in TARGET_NAMES
    }
    high_disagreement = g3_a1.prediction_disagreement(
        predictions["P_highU_U"], predictions["P_highU_SHUFFLE"]
    )
    cp_disagreement = g3_a1.prediction_disagreement(
        predictions["P_cp_U"], predictions["P_cp_SHUFFLE"]
    )
    high_l1 = g3_a1.mean_row_l1(
        targets["P_highU_U"], targets["P_highU_SHUFFLE"]
    )
    cp_l1 = g3_a1.mean_row_l1(
        targets["P_cp_U"], targets["P_cp_SHUFFLE"]
    )
    _require(
        high_disagreement["rate"] > 0.0 and high_l1 > 0.0,
        "high-U reference has no U/Shuffle difference",
    )
    consistency = {
        "prediction_disagreement": {
            "P_highU_U_vs_P_highU_SHUFFLE": high_disagreement,
            "P_cp_U_vs_P_cp_SHUFFLE": cp_disagreement,
            "P_cp_U_vs_P_all": g3_a1.prediction_disagreement(
                predictions["P_cp_U"], predictions["P_all"]
            ),
            "P_cp_SHUFFLE_vs_P_all": g3_a1.prediction_disagreement(
                predictions["P_cp_SHUFFLE"], predictions["P_all"]
            ),
        },
        "mean_row_L1": {
            "P_highU_U_vs_P_highU_SHUFFLE": high_l1,
            "P_cp_U_vs_P_cp_SHUFFLE": cp_l1,
        },
        "argmax_change_from_P_all": {
            name: g3_a1.prediction_disagreement(
                predictions[name], predictions["P_all"]
            )["count"]
            for name in (
                "P_highU_U",
                "P_highU_SHUFFLE",
                "P_cp_U",
                "P_cp_SHUFFLE",
            )
        },
        "prediction_disagreement_retention": float(
            cp_disagreement["rate"] / high_disagreement["rate"]
        ),
        "L1_retention": float(cp_l1 / high_l1),
    }
    source_audit = {
        arm: {
            "path": _display(source["path"]),
            "file_sha256": source["file_sha256"],
            "keys": source["keys"],
            "P_all_sha256": source["schema"]["P_all"]["sha256"],
            "P_highU_sha256": source["schema"]["P_highU"]["sha256"],
            "rho_sha256": source["schema"]["rho"]["sha256"],
            "sample_ids_sha256": source["schema"]["sample_ids"]["sha256"],
        }
        for arm, source in sources.items()
    }
    return {
        "targets": targets,
        "predictions": predictions,
        "tau": tau,
        "rho": {"rho_U": rho_u, "rho_SHUFFLE": rho_shuffle},
        "sample_ids": _readonly_array(u_arrays["sample_ids"], dtype=np.int64),
        "fixed_hashes": fixed_hashes,
        "diagnostics": diagnostics,
        "projection_audit": {
            "CP_U": cp_u["audit"],
            "CP_SHUFFLED_U": cp_shuffle["audit"],
        },
        "consistency": consistency,
        "source_audit": source_audit,
    }


def _metric_delta(metrics, left, right):
    return {
        "delta_ACC": float(metrics[left]["ACC"] - metrics[right]["ACC"]),
        "delta_NMI": float(metrics[left]["NMI"] - metrics[right]["NMI"]),
        "delta_ARI": float(metrics[left]["ARI"] - metrics[right]["ARI"]),
    }


def _all_three_positive(delta):
    return bool(
        delta["delta_ACC"] > 0.0
        and delta["delta_NMI"] > 0.0
        and delta["delta_ARI"] > 0.0
    )


def diagnostic_verdict(projection_valid, high_delta, cp_delta):
    if not projection_valid:
        return "CASE_C_PROJECTION_INVALID"
    if _all_three_positive(cp_delta):
        return "CASE_A_CONFIDENCE_PRESERVING_SIGNAL_SURVIVES"
    _require(
        _all_three_positive(high_delta),
        "high-U specificity is absent; no preregistered G3-A2 verdict applies",
    )
    return "CASE_B_DIRECTION_STILL_ERASED"


def evaluate_after_fixed_output_seal(fixed, data_path):
    """Load labels only after CP targets, tau, predictions, and hashes are sealed."""
    labels = g2.load_caltech_labels_only(data_path)
    metrics = {
        name: g2.metrics_from_fixed_prediction(labels, fixed["predictions"][name])
        for name in TARGET_NAMES
    }
    deltas = {
        "CP_U_minus_CP_SHUFFLE": _metric_delta(
            metrics, "P_cp_U", "P_cp_SHUFFLE"
        ),
        "CP_U_minus_P_all": _metric_delta(metrics, "P_cp_U", "P_all"),
        "P_highU_U_minus_P_highU_SHUFFLE": _metric_delta(
            metrics, "P_highU_U", "P_highU_SHUFFLE"
        ),
    }
    projection_valid = bool(
        fixed["projection_audit"]["CP_U"]["projection_valid"]
        and fixed["projection_audit"]["CP_SHUFFLED_U"]["projection_valid"]
    )
    verdict = diagnostic_verdict(
        projection_valid,
        deltas["P_highU_U_minus_P_highU_SHUFFLE"],
        deltas["CP_U_minus_CP_SHUFFLE"],
    )
    return {"metrics": metrics, "deltas": deltas, "verdict": verdict}


def run_diagnostic(
    g3_run_dir=DEFAULT_G3_RUN_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    data_path=g3_protocol.DEFAULT_DATA_PATH,
):
    run_root = _resolve(g3_run_dir)
    output_root = _resolve(output_dir)
    data_source = _resolve(data_path)
    _require(run_root.is_dir(), "G3-A0 formal run directory is missing")
    _require(not output_root.exists(), "refusing to overwrite a G3-A2 output")

    fixed = construct_fixed_outputs(run_root)
    output_root.mkdir(parents=True)
    # Complete the data seal before the sole label-loading boundary below.
    np.savez_compressed(
        output_root / "sealed_confidence_preserving_targets.npz",
        sample_ids=fixed["sample_ids"],
        rho_U=fixed["rho"]["rho_U"],
        rho_SHUFFLE=fixed["rho"]["rho_SHUFFLE"],
        **fixed["targets"],
        **fixed["tau"],
        **{
            name + "__argmax": prediction
            for name, prediction in fixed["predictions"].items()
        },
    )
    fixed_audit = {
        "stage": STAGE,
        "scope": "read_only_confidence_preserving_epoch0_target_gate",
        "source_audit": fixed["source_audit"],
        "fixed_hashes": fixed["fixed_hashes"],
        "projection_audit": fixed["projection_audit"],
        "label_free_diagnostics": fixed["diagnostics"],
        "consistency": fixed["consistency"],
        "targets_tau_predictions_sealed_before_labels": True,
    }
    _write_json(output_root / "fixed_cp_target_audit.json", fixed_audit)

    evaluation = evaluate_after_fixed_output_seal(fixed, data_source)
    engineering_checks = {
        "frozen_source_file_hashes_pass": True,
        "shared_P_all_and_sample_ids_pass": True,
        "no_D2_U_recompute_pass": True,
        "no_model_execution_pass": True,
        "P_cp_probability_validity_pass": bool(
            fixed["projection_audit"]["CP_U"]["probability_validity_pass"]
            and fixed["projection_audit"]["CP_SHUFFLED_U"][
                "probability_validity_pass"
            ]
        ),
        "tau_validity_pass": bool(
            fixed["projection_audit"]["CP_U"]["tau_validity_pass"]
            and fixed["projection_audit"]["CP_SHUFFLED_U"][
                "tau_validity_pass"
            ]
        ),
        "entropy_preservation_pass": bool(
            fixed["projection_audit"]["CP_U"]["entropy_preservation_pass"]
            and fixed["projection_audit"]["CP_SHUFFLED_U"][
                "entropy_preservation_pass"
            ]
        ),
        "rho_zero_identity_pass": bool(
            fixed["projection_audit"]["CP_U"]["rho_zero_identity_pass"]
            and fixed["projection_audit"]["CP_SHUFFLED_U"][
                "rho_zero_identity_pass"
            ]
        ),
        "sealed_before_labels_pass": True,
        "no_optimizer_backward_training_pass": True,
    }
    result = {
        "stage": STAGE,
        "scope": "read_only_confidence_preserving_epoch0_target_gate",
        "fixed_constants": {
            "eps": EPS,
            "entropy_solve_atol": ENTROPY_SOLVE_ATOL,
            "entropy_preservation_atol": ENTROPY_PASS_ATOL,
            "identity_atol": IDENTITY_ATOL,
            "degenerate_logit_spread_atol": DEGENERATE_LOGIT_SPREAD_ATOL,
            "tau_is_per_sample_constraint_solution": True,
        },
        "source_audit": fixed["source_audit"],
        "fixed_hashes": fixed["fixed_hashes"],
        "projection_audit": fixed["projection_audit"],
        "label_free_diagnostics": fixed["diagnostics"],
        "semantic_information_retention": fixed["consistency"],
        "target_metrics": evaluation["metrics"],
        "target_metric_deltas": evaluation["deltas"],
        "verdict": evaluation["verdict"],
        "scientific_pass_threshold": None,
        "engineering_checks": engineering_checks,
        "ENGINEERING_PASS": bool(all(engineering_checks.values())),
    }
    _write_json(output_root / "confidence_preserving_target_diagnostic.json", result)
    _write_json(
        output_root / "audit.json",
        {
            "stage": STAGE,
            "ENGINEERING_PASS": result["ENGINEERING_PASS"],
            "engineering_checks": engineering_checks,
            "verdict": result["verdict"],
        },
    )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="G3-A2 read-only confidence-preserving target gate"
    )
    parser.add_argument(
        "--g3-run-dir", type=Path, default=DEFAULT_G3_RUN_DIR
    )
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
