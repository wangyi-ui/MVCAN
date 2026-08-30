"""G3-A1 read-only diagnostic of the frozen epoch-zero target interface."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.g2_utility_semantic_consensus import g2_consensus_protocol as g2
from experiments.g3_selective_semantic_cooperation import (
    g3_target_correction_protocol as g3_protocol,
)
from irv.b4_information_utility import tensor_sha256


STAGE = "G3-A1"
SAMPLE_NUM = 1400
CLASS_NUM = 7
PROBABILITY_ATOL = 1e-6
ARM_DIRECTORIES = {
    "BASE": "base",
    "U_CORRECTION": "u_correction",
    "SHUFFLED_U_CORRECTION": "shuffled_u_correction",
}
TARGET_NAMES = (
    "P_all",
    "P_highU_U",
    "P_highU_SHUFFLE",
    "P_util_U",
    "P_util_SHUFFLE",
)
EXPECTED_ARCHIVE_KEYS = (
    "epoch",
    "sample_ids",
    "P_all",
    "M",
    "q_detached",
    "aligned_q",
    "P_highU",
    "rho",
    "P_util",
    "admission",
    "fusion_weights_used_for_P_all",
    "next_fusion_weights",
)
REQUIRED_ARCHIVE_KEYS = tuple(
    key for key in EXPECTED_ARCHIVE_KEYS if key not in ("P_highU", "P_util")
)
EXPECTED_ARRAY_BOUNDARIES = {
    "epoch": ((), np.dtype(np.int64)),
    "sample_ids": ((SAMPLE_NUM,), np.dtype(np.int64)),
    "P_all": ((SAMPLE_NUM, CLASS_NUM), np.dtype(np.float32)),
    "M": ((6, CLASS_NUM, CLASS_NUM), np.dtype(np.int64)),
    "q_detached": ((SAMPLE_NUM, 6, CLASS_NUM), np.dtype(np.float32)),
    "aligned_q": ((SAMPLE_NUM, 6, CLASS_NUM), np.dtype(np.float32)),
    "P_highU": ((SAMPLE_NUM, CLASS_NUM), np.dtype(np.float32)),
    "rho": ((SAMPLE_NUM,), np.dtype(np.float64)),
    "P_util": ((SAMPLE_NUM, CLASS_NUM), np.dtype(np.float32)),
    "admission": ((SAMPLE_NUM, 6), np.dtype(bool)),
    "fusion_weights_used_for_P_all": ((6,), np.dtype(np.float64)),
    "next_fusion_weights": ((6,), np.dtype(np.float64)),
}
DEFAULT_G3_RUN_DIR = (
    REPOSITORY_ROOT
    / "outputs/g3_selective_semantic_cooperation/g3a0_seed20_100ep"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "outputs/g3_selective_semantic_cooperation/"
    "g3a1_seed20_epoch0_target_interface"
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


def _file_sha256(path):
    return g3_protocol.b7_protocol.file_sha256(path)


def _readonly_array(value):
    # np.ascontiguousarray promotes a scalar from shape [] to [1].  A copied
    # ndarray preserves the on-disk dimensionality while remaining C-order.
    array = np.array(value, copy=True, order="C")
    array.setflags(write=False)
    return array


def load_epoch0_archive(path):
    """Load and validate one frozen G3-A0 epoch-zero NPZ."""
    source = _resolve(path)
    _require(source.is_file(), "missing G3-A0 epoch-zero artifact: " + str(source))
    arrays = {}
    with np.load(source, allow_pickle=False) as archive:
        keys = tuple(archive.files)
        _require(
            set(REQUIRED_ARCHIVE_KEYS).issubset(set(keys))
            and set(keys).issubset(set(EXPECTED_ARCHIVE_KEYS)),
            "epoch-zero archive key mismatch",
        )
        for key in keys:
            arrays[key] = _readonly_array(archive[key])

    schema = {}
    for key in keys:
        array = arrays[key]
        expected_shape, expected_dtype = EXPECTED_ARRAY_BOUNDARIES[key]
        _require(
            array.shape == expected_shape and array.dtype == expected_dtype,
            "epoch-zero array boundary mismatch for " + key,
        )
        _require(
            array.dtype == np.dtype(bool)
            or np.issubdtype(array.dtype, np.integer)
            or np.isfinite(array).all(),
            "epoch-zero array contains NaN/Inf: " + key,
        )
        schema[key] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": tensor_sha256(array),
        }
    _require(int(arrays["epoch"]) == 0, "artifact is not epoch zero")
    _require(
        np.array_equal(
            arrays["sample_ids"], np.arange(SAMPLE_NUM, dtype=np.int64)
        ),
        "sample IDs are not arange(N)",
    )
    return {
        "path": source,
        "file_sha256": _file_sha256(source),
        "keys": list(keys),
        "schema": schema,
        "arrays": arrays,
    }


def _validate_probability_target(target, name):
    values = np.asarray(target)
    _require(
        values.shape == (SAMPLE_NUM, CLASS_NUM)
        and np.isfinite(values).all()
        and float(values.min()) >= 0.0
        and np.allclose(
            values.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_ATOL
        ),
        name + " is not a finite [1400,7] probability target",
    )


def reconstruct_high_u_from_saved_quantities(archive):
    """Fallback using only saved aligned_q/admission/current-P_all weights."""
    aligned_q = np.asarray(archive["aligned_q"], dtype=np.float32)
    admission = np.asarray(archive["admission"], dtype=bool)
    weights = np.asarray(
        archive["fusion_weights_used_for_P_all"], dtype=np.float32
    )
    weighted_admission = admission.astype(np.float32) * weights[None, :]
    denominator = weighted_admission.sum(axis=1, keepdims=True)
    _require(np.all(denominator > 0.0), "empty reconstructed P_highU row")
    result = (
        (aligned_q * weighted_admission[:, :, None]).sum(axis=1) / denominator
    ).astype(np.float32, copy=False)
    result = _readonly_array(result)
    _validate_probability_target(result, "reconstructed P_highU")
    return result


def reconstruct_p_util_from_saved_quantities(archive, p_high_u):
    """Fallback using only saved P_all, rho, and P_highU with the original formula."""
    p_all = np.asarray(archive["P_all"], dtype=np.float32)
    rho = np.asarray(archive["rho"], dtype=np.float32)[:, None]
    result = ((1.0 - rho) * p_all + rho * p_high_u).astype(
        np.float32, copy=False
    )
    result = _readonly_array(result)
    _validate_probability_target(result, "reconstructed P_util")
    return result


def _extract_or_reconstruct(archive, key):
    arrays = archive["arrays"]
    if key in arrays:
        return arrays[key], False
    if key == "P_highU":
        return reconstruct_high_u_from_saved_quantities(arrays), True
    if key == "P_util":
        high, _ = _extract_or_reconstruct(archive, "P_highU")
        return reconstruct_p_util_from_saved_quantities(arrays, high), True
    raise RuntimeError("cannot reconstruct missing epoch-zero key: " + key)


def score_diagnostics(target):
    """Compute label-free entropy, confidence, and top1-top2 margin."""
    values = np.asarray(target, dtype=np.float64)
    _validate_probability_target(values, "diagnostic target")
    safe = np.clip(values, np.finfo(np.float64).tiny, 1.0)
    entropy = -np.sum(values * np.log(safe), axis=1)
    ordered = np.sort(values, axis=1)
    return {
        "entropy_mean": float(entropy.mean()),
        "max_confidence_mean": float(ordered[:, -1].mean()),
        "top1_top2_margin_mean": float(
            (ordered[:, -1] - ordered[:, -2]).mean()
        ),
    }


def prediction_disagreement(first_prediction, second_prediction):
    first = np.asarray(first_prediction, dtype=np.int64)
    second = np.asarray(second_prediction, dtype=np.int64)
    _require(
        first.shape == second.shape == (SAMPLE_NUM,),
        "prediction comparison shape mismatch",
    )
    changed = first != second
    return {
        "count": int(changed.sum()),
        "rate": float(changed.mean()),
    }


def mean_row_l1(first_target, second_target):
    first = np.asarray(first_target, dtype=np.float64)
    second = np.asarray(second_target, dtype=np.float64)
    _require(
        first.shape == second.shape == (SAMPLE_NUM, CLASS_NUM),
        "L1 comparison shape mismatch",
    )
    return float(np.abs(first - second).sum(axis=1).mean())


def construct_fixed_target_outputs(g3_run_dir=DEFAULT_G3_RUN_DIR):
    """Load, validate, and hash targets/predictions without loading labels."""
    run_root = _resolve(g3_run_dir)
    archives = {
        arm: load_epoch0_archive(
            run_root / directory / "refresh_epoch_0000.npz"
        )
        for arm, directory in ARM_DIRECTORIES.items()
    }
    shared_keys = (
        "sample_ids",
        "P_all",
        "M",
        "q_detached",
        "aligned_q",
        "fusion_weights_used_for_P_all",
        "next_fusion_weights",
    )
    base_arrays = archives["BASE"]["arrays"]
    for arm in ("U_CORRECTION", "SHUFFLED_U_CORRECTION"):
        for key in shared_keys:
            _require(
                np.array_equal(base_arrays[key], archives[arm]["arrays"][key]),
                "native epoch-zero quantity differs across arms: " + key,
            )

    p_all, p_all_reconstructed = _extract_or_reconstruct(
        archives["BASE"], "P_all"
    )
    p_high_u, high_u_reconstructed = _extract_or_reconstruct(
        archives["U_CORRECTION"], "P_highU"
    )
    p_high_shuffle, high_shuffle_reconstructed = _extract_or_reconstruct(
        archives["SHUFFLED_U_CORRECTION"], "P_highU"
    )
    p_util_u, util_u_reconstructed = _extract_or_reconstruct(
        archives["U_CORRECTION"], "P_util"
    )
    p_util_shuffle, util_shuffle_reconstructed = _extract_or_reconstruct(
        archives["SHUFFLED_U_CORRECTION"], "P_util"
    )
    targets = {
        "P_all": _readonly_array(p_all),
        "P_highU_U": _readonly_array(p_high_u),
        "P_highU_SHUFFLE": _readonly_array(p_high_shuffle),
        "P_util_U": _readonly_array(p_util_u),
        "P_util_SHUFFLE": _readonly_array(p_util_shuffle),
    }
    for name, target in targets.items():
        _validate_probability_target(target, name)

    # Existing artifacts contain every target. These checks prove they retain
    # the exact preregistered convex formula; no reconstructed value replaces
    # an existing saved target.
    expected_u = reconstruct_p_util_from_saved_quantities(
        archives["U_CORRECTION"]["arrays"], targets["P_highU_U"]
    )
    expected_shuffle = reconstruct_p_util_from_saved_quantities(
        archives["SHUFFLED_U_CORRECTION"]["arrays"],
        targets["P_highU_SHUFFLE"],
    )
    _require(
        np.array_equal(targets["P_util_U"], expected_u)
        and np.array_equal(targets["P_util_SHUFFLE"], expected_shuffle),
        "saved P_util differs from the preregistered convex formula",
    )
    _require(
        np.array_equal(base_arrays["P_util"], targets["P_all"]),
        "BASE saved target is not exactly P_all",
    )

    predictions = {
        name: _readonly_array(np.argmax(target, axis=1).astype(np.int64))
        for name, target in targets.items()
    }
    fixed_hashes = {
        name: {
            "target_sha256": tensor_sha256(targets[name]),
            "prediction_sha256": tensor_sha256(predictions[name]),
            "target_shape": list(targets[name].shape),
            "target_dtype": str(targets[name].dtype),
            "prediction_shape": list(predictions[name].shape),
            "prediction_dtype": str(predictions[name].dtype),
        }
        for name in TARGET_NAMES
    }
    label_free_diagnostics = {
        name: score_diagnostics(targets[name]) for name in TARGET_NAMES
    }
    consistency = {
        "prediction_disagreement": {
            "P_highU_U_vs_P_highU_SHUFFLE": prediction_disagreement(
                predictions["P_highU_U"], predictions["P_highU_SHUFFLE"]
            ),
            "P_util_U_vs_P_util_SHUFFLE": prediction_disagreement(
                predictions["P_util_U"], predictions["P_util_SHUFFLE"]
            ),
            "P_util_U_vs_P_all": prediction_disagreement(
                predictions["P_util_U"], predictions["P_all"]
            ),
        },
        "mean_row_L1": {
            "P_highU_U_vs_P_highU_SHUFFLE": mean_row_l1(
                targets["P_highU_U"], targets["P_highU_SHUFFLE"]
            ),
            "P_util_U_vs_P_util_SHUFFLE": mean_row_l1(
                targets["P_util_U"], targets["P_util_SHUFFLE"]
            ),
            "P_util_U_vs_P_all": mean_row_l1(
                targets["P_util_U"], targets["P_all"]
            ),
        },
        "argmax_change_from_P_all": {
            name: prediction_disagreement(
                predictions[name], predictions["P_all"]
            )["count"]
            for name in (
                "P_highU_U",
                "P_highU_SHUFFLE",
                "P_util_U",
                "P_util_SHUFFLE",
            )
        },
    }
    artifact_schema = {
        arm: {
            "path": _display(archive["path"]),
            "file_sha256": archive["file_sha256"],
            "keys": archive["keys"],
            "arrays": archive["schema"],
        }
        for arm, archive in archives.items()
    }
    reconstruction_audit = {
        "P_all_reconstructed": p_all_reconstructed,
        "P_highU_U_reconstructed": high_u_reconstructed,
        "P_highU_SHUFFLE_reconstructed": high_shuffle_reconstructed,
        "P_util_U_reconstructed": util_u_reconstructed,
        "P_util_SHUFFLE_reconstructed": util_shuffle_reconstructed,
    }
    _require(
        not any(reconstruction_audit.values()),
        "formal epoch-zero artifacts unexpectedly required reconstruction",
    )
    return {
        "targets": targets,
        "predictions": predictions,
        "fixed_hashes": fixed_hashes,
        "label_free_diagnostics": label_free_diagnostics,
        "consistency": consistency,
        "artifact_schema": artifact_schema,
        "reconstruction_audit": reconstruction_audit,
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


def diagnostic_verdict(target_deltas, final_g3_delta):
    """Apply the three allowed cases using strict three-metric dominance."""
    high_survives = _all_three_positive(
        target_deltas["P_highU_U_minus_P_highU_SHUFFLE"]
    )
    util_survives = _all_three_positive(
        target_deltas["P_util_U_minus_P_util_SHUFFLE"]
    )
    final_loses = not _all_three_positive(final_g3_delta)
    if not high_survives:
        return "CASE_C_G3_EPOCH0_HIGHU_SIGNAL_NOT_REPRODUCED"
    if not util_survives:
        return "CASE_B_INTERPOLATION_ERASES_UTILITY_SIGNAL"
    _require(final_loses, "target and final signals survive; no allowed G3-A1 case")
    return "CASE_A_TARGET_SIGNAL_SURVIVES_BUT_TRAINING_LOSES_IT"


def evaluate_after_fixed_output_seal(fixed, data_path, comparison_summary_path):
    """Load labels only after targets and predictions have been fixed and hashed."""
    labels = g2.load_caltech_labels_only(data_path)
    metrics = {
        name: g2.metrics_from_fixed_prediction(labels, fixed["predictions"][name])
        for name in TARGET_NAMES
    }
    target_deltas = {
        "P_highU_U_minus_P_highU_SHUFFLE": _metric_delta(
            metrics, "P_highU_U", "P_highU_SHUFFLE"
        ),
        "P_util_U_minus_P_util_SHUFFLE": _metric_delta(
            metrics, "P_util_U", "P_util_SHUFFLE"
        ),
        "P_util_U_minus_P_all": _metric_delta(
            metrics, "P_util_U", "P_all"
        ),
    }
    with open(comparison_summary_path, "r", encoding="utf-8") as input_file:
        g3_comparison = json.load(input_file)
    final_g3_delta = g3_comparison["primary_gate_U_vs_SHUFFLED_U"]
    verdict = diagnostic_verdict(target_deltas, final_g3_delta)
    return {
        "target_metrics": metrics,
        "target_metric_deltas": target_deltas,
        "final_G3_U_minus_SHUFFLED": final_g3_delta,
        "verdict_rule": (
            "U>Shuffle means strict positive dominance in ACC,NMI,ARI; "
            "no magnitude threshold is introduced"
        ),
        "verdict": verdict,
    }


def _entropy_mismatch_audit(label_free_diagnostics):
    base = label_free_diagnostics["P_all"]
    entropy = {
        name: label_free_diagnostics[name]["entropy_mean"]
        for name in TARGET_NAMES
    }
    delta_from_p_all = {
        name: float(entropy[name] - entropy["P_all"])
        for name in TARGET_NAMES
    }
    desharpening = bool(
        all(
            label_free_diagnostics[name]["entropy_mean"]
            > base["entropy_mean"]
            and label_free_diagnostics[name]["max_confidence_mean"]
            < base["max_confidence_mean"]
            and label_free_diagnostics[name]["top1_top2_margin_mean"]
            < base["top1_top2_margin_mean"]
            for name in ("P_util_U", "P_util_SHUFFLE")
        )
    )
    return {
        "entropy_mean": entropy,
        "delta_entropy_relative_to_P_all": delta_from_p_all,
        "target_desharpening_observed": desharpening,
        "desharpening_sign_rule": (
            "both corrected targets have higher entropy, lower mean maximum "
            "confidence, and lower mean top1-top2 margin than P_all"
        ),
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
    _require(not output_root.exists(), "refusing to overwrite a G3-A1 output")

    fixed = construct_fixed_target_outputs(run_root)
    output_root.mkdir(parents=True)
    # These artifacts are fully written before the only label-loading function
    # is invoked below.
    np.savez_compressed(
        output_root / "sealed_targets_predictions.npz",
        **fixed["targets"],
        **{
            name + "__argmax": prediction
            for name, prediction in fixed["predictions"].items()
        },
    )
    fixed_record = {
        "stage": STAGE,
        "scope": "read_only_frozen_epoch0_target_interface",
        "source_G3_run_dir": _display(run_root),
        "fixed_hashes": fixed["fixed_hashes"],
        "label_free_diagnostics": fixed["label_free_diagnostics"],
        "consistency": fixed["consistency"],
        "reconstruction_audit": fixed["reconstruction_audit"],
        "targets_and_predictions_sealed_before_labels": True,
    }
    _write_json(output_root / "artifact_schema.json", fixed["artifact_schema"])
    _write_json(output_root / "fixed_target_audit.json", fixed_record)

    evaluation = evaluate_after_fixed_output_seal(
        fixed,
        data_source,
        run_root / "comparison_summary.json",
    )
    entropy_audit = _entropy_mismatch_audit(fixed["label_free_diagnostics"])
    engineering_checks = {
        "all_archives_epoch0_pass": True,
        "archive_schema_pass": True,
        "all_targets_shape_N7_pass": True,
        "all_targets_probability_pass": True,
        "common_native_quantities_across_arms_pass": True,
        "base_exact_identity_pass": True,
        "saved_convex_formula_exact_pass": True,
        "no_reconstruction_needed_pass": bool(
            not any(fixed["reconstruction_audit"].values())
        ),
        "targets_readonly_pass": bool(
            all(not value.flags.writeable for value in fixed["targets"].values())
        ),
        "predictions_readonly_pass": bool(
            all(
                not value.flags.writeable
                for value in fixed["predictions"].values()
            )
        ),
        "labels_after_fixed_output_seal_pass": True,
        "no_optimizer_backward_training_pass": True,
    }
    result = {
        "stage": STAGE,
        "scope": "read_only_frozen_epoch0_target_interface",
        "artifact_schema": fixed["artifact_schema"],
        "fixed_hashes": fixed["fixed_hashes"],
        "target_metrics": evaluation["target_metrics"],
        "label_free_diagnostics": fixed["label_free_diagnostics"],
        "target_metric_deltas": evaluation["target_metric_deltas"],
        "consistency": fixed["consistency"],
        "entropy_mismatch_audit": entropy_audit,
        "final_G3_U_minus_SHUFFLED": evaluation[
            "final_G3_U_minus_SHUFFLED"
        ],
        "verdict_rule": evaluation["verdict_rule"],
        "verdict": evaluation["verdict"],
        "scientific_pass_threshold": None,
        "engineering_checks": engineering_checks,
        "ENGINEERING_PASS": bool(all(engineering_checks.values())),
    }
    _write_json(output_root / "target_interface_diagnostic.json", result)
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
        description="G3-A1 read-only epoch-zero target interface diagnostic"
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
