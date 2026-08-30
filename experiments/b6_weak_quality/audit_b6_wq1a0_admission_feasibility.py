"""B6-WQ1A-0 read-only Information-Utility admission feasibility audit."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import experiments.b6_weak_quality.evaluate_b6_wq0_utility_semantic_admission as b6
from weak_quality import ndarray_sha256


STAGE = "B6-WQ1A-0"
DATASET_NAME = b6.DATASET_NAME
CONDITION = b6.CONDITION
SUPPORTED_SEEDS = b6.SUPPORTED_SEEDS
SAMPLE_NUM = b6.SAMPLE_NUM
VIEW_NUM = b6.VIEW_NUM
CORRUPTED_VIEWS_PER_SAMPLE = 2
ADMITTED_VIEWS_PER_SAMPLE = 3
NULL_REPEATS = b6.NULL_REPEATS
SHUFFLE_SEED = b6.SHUFFLE_SEED
DEFAULT_OUTPUT_DIR = (
    "outputs/b6_weak_quality/wq1a0_admission_feasibility"
)
HIGHER_IS_BETTER_METRICS = (
    "admitted_clean_fraction",
    "all_clean_top3_fraction",
    "oracle_set_jaccard_mean",
    "within_sample_pair_win_rate",
)
LOWER_IS_BETTER_METRICS = (
    "corrupted_views_admitted_mean",
)
METRIC_NAMES = (
    "admitted_clean_fraction",
    "all_clean_top3_fraction",
    "corrupted_views_admitted_mean",
    "oracle_set_jaccard_mean",
    "within_sample_pair_win_rate",
)
THEORETICAL_REFERENCES = {
    "random_theoretical_clean_fraction": 0.6,
    "random_theoretical_all_clean_top3": 0.1,
    "random_theoretical_corrupted_admitted_mean": 1.2,
    "random_pair_win_reference": 0.5,
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


def _validate_seed(seed):
    seed = int(seed)
    _require(seed in SUPPORTED_SEEDS, "unsupported B6-WQ1A-0 seed")
    return seed


def load_frozen_audit_inputs(seed):
    """Load frozen Utility and corruption provenance without labels or models."""
    seed = _validate_seed(seed)
    utility, utility_hash, utility_path, utility_record_path = (
        b6.load_source_utility(seed)
    )
    utility = np.asarray(utility)
    _require(
        utility.shape == (SAMPLE_NUM, VIEW_NUM),
        "Utility shape mismatch",
    )
    _require(np.isfinite(utility).all(), "Utility contains NaN or Inf")
    _require(
        float(utility.min()) >= 0.0 and float(utility.max()) <= 1.0,
        "Utility outside [0,1]",
    )

    condition_root = _resolve(
        b6.SEED_PROVENANCE[seed]["backbone_dir"]
    ).parent
    mask_path = condition_root / "audit/corruption_mask.npy"
    corruption_audit_path = condition_root / "audit/corruption_audit.json"
    _require(mask_path.is_file(), "frozen corruption mask missing")
    _require(
        corruption_audit_path.is_file(),
        "frozen corruption audit missing",
    )
    corruption_mask = np.load(mask_path, allow_pickle=False)
    _require(
        corruption_mask.dtype == np.dtype(bool),
        "corruption mask dtype must be bool",
    )
    _require(
        corruption_mask.shape == (SAMPLE_NUM, VIEW_NUM),
        "corruption mask shape mismatch",
    )
    _require(
        np.all(
            corruption_mask.sum(axis=1)
            == CORRUPTED_VIEWS_PER_SAMPLE
        ),
        "corruption mask must contain exactly two corrupt views per sample",
    )
    corruption_audit = _read_json(corruption_audit_path)
    mask_hash = ndarray_sha256(corruption_mask)
    _require(
        corruption_audit.get("dataset") == DATASET_NAME
        and corruption_audit.get("model_seed") == seed
        and corruption_audit.get("target_snr_db") == 2.5
        and corruption_audit.get("corruption_k")
        == CORRUPTED_VIEWS_PER_SAMPLE,
        "corruption audit condition mismatch",
    )
    _require(
        mask_hash == corruption_audit.get("mask_sha256"),
        "corruption mask hash mismatch",
    )

    manifest, manifest_path = b6.load_canonical_manifest()
    entry = manifest["canonical_entries"].get(str(seed))
    _require(entry is not None, "registered canonical entry missing")
    registered_provenance_path = _resolve(entry["source_provenance"])
    _require(
        registered_provenance_path.is_file(),
        "registered provenance missing",
    )
    registered_provenance = _read_json(registered_provenance_path)
    _require(
        registered_provenance.get("model_seed") == seed
        and registered_provenance.get("corruption_mask_sha256")
        == mask_hash,
        "registered corruption provenance mismatch",
    )

    oracle_clean_mask = np.logical_not(corruption_mask)
    _require(
        np.all(
            oracle_clean_mask.sum(axis=1)
            == ADMITTED_VIEWS_PER_SAMPLE
        ),
        "oracle mask must contain exactly three clean views per sample",
    )
    return {
        "model_seed": seed,
        "utility": utility,
        "utility_sha256": utility_hash,
        "utility_source": utility_path,
        "utility_provenance": utility_record_path,
        "corruption_mask": corruption_mask,
        "corruption_mask_sha256": mask_hash,
        "corruption_mask_source": mask_path,
        "corruption_audit_source": corruption_audit_path,
        "registered_provenance_source": registered_provenance_path,
        "canonical_manifest_source": manifest_path,
        "oracle_clean_mask": oracle_clean_mask,
    }


def top3_admission_mask(utility):
    """Select three largest Utility values with stable low-index tie breaking."""
    values = np.asarray(utility)
    _require(values.ndim == 2, "Utility must be two-dimensional")
    _require(
        values.shape[1] == VIEW_NUM,
        "Utility must contain exactly five views",
    )
    _require(np.isfinite(values).all(), "Utility contains NaN or Inf")
    order = np.argsort(-values, axis=1, kind="mergesort")
    selected = order[:, :ADMITTED_VIEWS_PER_SAMPLE]
    mask = np.zeros(values.shape, dtype=bool)
    mask[
        np.arange(values.shape[0], dtype=np.int64)[:, None],
        selected,
    ] = True
    _require(
        np.all(
            mask.sum(axis=1)
            == ADMITTED_VIEWS_PER_SAMPLE
        ),
        "Top3 admission row count mismatch",
    )
    return mask


def within_sample_clean_corrupt_pair_win_rate(
    utility,
    oracle_clean_mask,
):
    values = np.asarray(utility)
    oracle = np.asarray(oracle_clean_mask, dtype=bool)
    _require(values.shape == oracle.shape, "pair audit shape mismatch")
    _require(
        np.all(
            oracle.sum(axis=1)
            == ADMITTED_VIEWS_PER_SAMPLE
        ),
        "pair audit requires three clean views per sample",
    )
    clean_values = values[oracle].reshape(
        values.shape[0],
        ADMITTED_VIEWS_PER_SAMPLE,
    )
    corrupt_values = values[np.logical_not(oracle)].reshape(
        values.shape[0],
        CORRUPTED_VIEWS_PER_SAMPLE,
    )
    differences = (
        clean_values[:, :, None]
        - corrupt_values[:, None, :]
    )
    wins = (
        (differences > 0.0).astype(np.float64)
        + 0.5 * (differences == 0.0).astype(np.float64)
    )
    return float(np.mean(wins))


def admission_metrics(admission_mask, oracle_clean_mask, utility):
    admitted = np.asarray(admission_mask, dtype=bool)
    oracle = np.asarray(oracle_clean_mask, dtype=bool)
    values = np.asarray(utility)
    _require(
        admitted.shape == oracle.shape == values.shape,
        "admission audit shape mismatch",
    )
    _require(
        np.all(
            admitted.sum(axis=1)
            == ADMITTED_VIEWS_PER_SAMPLE
        ),
        "admission mask must contain exactly three views per sample",
    )
    _require(
        np.all(
            oracle.sum(axis=1)
            == ADMITTED_VIEWS_PER_SAMPLE
        ),
        "oracle mask must contain exactly three views per sample",
    )

    intersection = np.logical_and(admitted, oracle).sum(
        axis=1,
        dtype=np.int64,
    )
    union = np.logical_or(admitted, oracle).sum(
        axis=1,
        dtype=np.int64,
    )
    corrupted_admitted = np.logical_and(
        admitted,
        np.logical_not(oracle),
    ).sum(axis=1, dtype=np.int64)
    admitted_clean_fraction = float(
        np.sum(intersection, dtype=np.int64)
        / (
            admitted.shape[0]
            * ADMITTED_VIEWS_PER_SAMPLE
        )
    )
    corrupted_views_admitted_mean = float(
        np.mean(corrupted_admitted)
    )
    relation_value = float(
        1.0
        - corrupted_views_admitted_mean
        / ADMITTED_VIEWS_PER_SAMPLE
    )
    _require(
        np.isclose(
            admitted_clean_fraction,
            relation_value,
            rtol=0.0,
            atol=1e-15,
        ),
        "clean/corrupt admission relation mismatch",
    )
    return {
        "admitted_clean_fraction": admitted_clean_fraction,
        "all_clean_top3_fraction": float(
            np.mean(np.all(admitted == oracle, axis=1))
        ),
        "corrupted_views_admitted_mean": (
            corrupted_views_admitted_mean
        ),
        "oracle_set_jaccard_mean": float(
            np.mean(intersection / union)
        ),
        "within_sample_pair_win_rate": (
            within_sample_clean_corrupt_pair_win_rate(
                values,
                oracle,
            )
        ),
    }


def generate_permutation_bank():
    _require(NULL_REPEATS == 200, "null repeats must remain 200")
    _require(
        SHUFFLE_SEED == b6.SHUFFLE_SEED,
        "shuffle seed must match B6-WQ0",
    )
    return b6.generate_within_view_permutations(
        sample_num=SAMPLE_NUM,
        view_num=VIEW_NUM,
        repeats=NULL_REPEATS,
        seed=SHUFFLE_SEED,
    )


def shuffle_utility_within_views(utility, view_permutations):
    return b6.shuffle_utility_within_views(
        utility,
        view_permutations,
    )


def shuffled_null_metrics(utility, oracle_clean_mask):
    values = {
        name: np.empty(NULL_REPEATS, dtype=np.float64)
        for name in METRIC_NAMES
    }
    permutation_bank = generate_permutation_bank()
    for repeat_id in range(NULL_REPEATS):
        shuffled = shuffle_utility_within_views(
            utility,
            permutation_bank[repeat_id],
        )
        shuffled_mask = top3_admission_mask(shuffled)
        metrics = admission_metrics(
            shuffled_mask,
            oracle_clean_mask,
            shuffled,
        )
        for name in METRIC_NAMES:
            values[name][repeat_id] = metrics[name]
    return values


def summarize_null(values, correct_value, higher_is_better):
    array = np.asarray(values, dtype=np.float64)
    _require(
        array.shape == (NULL_REPEATS,),
        "null metric shape mismatch",
    )
    _require(np.isfinite(array).all(), "null metric non-finite")
    if higher_is_better:
        count_as_or_more_extreme = int(
            np.sum(array >= float(correct_value))
        )
        tail = "null_greater_than_or_equal_to_correct"
    else:
        count_as_or_more_extreme = int(
            np.sum(array <= float(correct_value))
        )
        tail = "null_less_than_or_equal_to_correct"
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "tail_definition": tail,
        "count_as_or_more_extreme": count_as_or_more_extreme,
        "one_sided_empirical_p": float(
            (1 + count_as_or_more_extreme)
            / (NULL_REPEATS + 1)
        ),
    }


def summarize_all_null_metrics(null_values, correct_metrics):
    result = {}
    for name in METRIC_NAMES:
        result[name] = summarize_null(
            null_values[name],
            correct_metrics[name],
            higher_is_better=(
                name in HIGHER_IS_BETTER_METRICS
            ),
        )
    return result


def per_view_diagnostics(utility, corruption_mask):
    values = np.asarray(utility)
    corrupt = np.asarray(corruption_mask, dtype=bool)
    rows = []
    for view_id in range(VIEW_NUM):
        clean_values = values[
            np.logical_not(corrupt[:, view_id]),
            view_id,
        ]
        corrupt_values = values[corrupt[:, view_id], view_id]
        clean_mean = float(np.mean(clean_values))
        corrupt_mean = float(np.mean(corrupt_values))
        rows.append({
            "view_id": int(view_id),
            "clean_count": int(clean_values.size),
            "corrupt_count": int(corrupt_values.size),
            "mean_utility_when_clean": clean_mean,
            "mean_utility_when_corrupt": corrupt_mean,
            "clean_corrupt_gap": float(
                clean_mean - corrupt_mean
            ),
            "diagnostic_only": True,
        })
    return rows


def seed_feasibility_gate(correct_metrics, null_summary):
    gate_a = bool(
        correct_metrics["admitted_clean_fraction"]
        > null_summary["admitted_clean_fraction"]["p95"]
        and null_summary[
            "admitted_clean_fraction"
        ]["one_sided_empirical_p"] < 0.05
    )
    gate_b = bool(
        correct_metrics["all_clean_top3_fraction"]
        > null_summary["all_clean_top3_fraction"]["p95"]
        and null_summary[
            "all_clean_top3_fraction"
        ]["one_sided_empirical_p"] < 0.05
    )
    gate_c = bool(
        correct_metrics["within_sample_pair_win_rate"]
        > null_summary["within_sample_pair_win_rate"]["p95"]
        and null_summary[
            "within_sample_pair_win_rate"
        ]["one_sided_empirical_p"] < 0.05
    )
    return {
        "gate_a_admitted_clean_fraction": gate_a,
        "gate_b_all_clean_top3_fraction": gate_b,
        "gate_c_within_sample_pair_win_rate": gate_c,
        "B6_WQ1A0_ADMISSION_FEASIBILITY_SEED_PASS": bool(
            gate_a and gate_b and gate_c
        ),
    }


def run_seed_audit(seed, output_dir=DEFAULT_OUTPUT_DIR):
    inputs = load_frozen_audit_inputs(seed)
    utility = inputs["utility"]
    oracle_clean_mask = inputs["oracle_clean_mask"]
    correct_mask = top3_admission_mask(utility)
    correct_metrics = admission_metrics(
        correct_mask,
        oracle_clean_mask,
        utility,
    )
    null_values = shuffled_null_metrics(
        utility,
        oracle_clean_mask,
    )
    null_summary = summarize_all_null_metrics(
        null_values,
        correct_metrics,
    )
    gate = seed_feasibility_gate(
        correct_metrics,
        null_summary,
    )

    output_root = _resolve(output_dir)
    artifact_dir = output_root / (
        "seed" + str(inputs["model_seed"])
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    correct_mask_path = artifact_dir / "correct_admission_mask.npy"
    oracle_mask_path = artifact_dir / "oracle_clean_mask.npy"
    null_path = artifact_dir / "shuffled_null_metrics.npz"
    np.save(correct_mask_path, correct_mask)
    np.save(oracle_mask_path, oracle_clean_mask)
    np.savez(null_path, **null_values)

    result = {
        "stage": STAGE,
        "dataset": DATASET_NAME,
        "condition": CONDITION,
        "snr_db": 2.5,
        "model_seed": inputs["model_seed"],
        "sample_count": SAMPLE_NUM,
        "view_count": VIEW_NUM,
        "corrupted_views_per_sample": (
            CORRUPTED_VIEWS_PER_SAMPLE
        ),
        "admitted_views_per_sample": (
            ADMITTED_VIEWS_PER_SAMPLE
        ),
        "utility_sha256": inputs["utility_sha256"],
        "utility_source": _display(inputs["utility_source"]),
        "utility_provenance": _display(
            inputs["utility_provenance"]
        ),
        "utility_shape": list(utility.shape),
        "utility_finite": bool(np.isfinite(utility).all()),
        "utility_min": float(utility.min()),
        "utility_max": float(utility.max()),
        "corruption_mask_sha256": (
            inputs["corruption_mask_sha256"]
        ),
        "corruption_mask_source": _display(
            inputs["corruption_mask_source"]
        ),
        "corruption_audit_source": _display(
            inputs["corruption_audit_source"]
        ),
        "registered_provenance_source": _display(
            inputs["registered_provenance_source"]
        ),
        "canonical_manifest_source": _display(
            inputs["canonical_manifest_source"]
        ),
        "correct_metrics": correct_metrics,
        "correct_admitted_clean_fraction": correct_metrics[
            "admitted_clean_fraction"
        ],
        "correct_all_clean_top3_fraction": correct_metrics[
            "all_clean_top3_fraction"
        ],
        "correct_corrupted_views_admitted_mean": correct_metrics[
            "corrupted_views_admitted_mean"
        ],
        "correct_oracle_set_jaccard_mean": correct_metrics[
            "oracle_set_jaccard_mean"
        ],
        "correct_within_sample_pair_win_rate": correct_metrics[
            "within_sample_pair_win_rate"
        ],
        **THEORETICAL_REFERENCES,
        "shuffled_null_repeats": NULL_REPEATS,
        "shuffled_null_seed": SHUFFLE_SEED,
        "shuffled_null_policy": (
            "independent sample permutation within each view; "
            "per-view Utility marginal preserved exactly"
        ),
        "shuffled_null_summary": null_summary,
        "primary_gate": gate,
        **gate,
        "per_view_diagnostics": per_view_diagnostics(
            utility,
            inputs["corruption_mask"],
        ),
        "correct_admission_mask_path": _display(
            correct_mask_path
        ),
        "oracle_clean_mask_path": _display(oracle_mask_path),
        "shuffled_null_metrics_path": _display(null_path),
        "correct_admission_mask_sha256": ndarray_sha256(
            correct_mask
        ),
        "oracle_clean_mask_sha256": ndarray_sha256(
            oracle_clean_mask
        ),
        "labels_loaded": False,
        "labels_used": False,
        "corruption_mask_used_for_evaluation": True,
        "corruption_mask_used_for_admission": False,
        "model_loaded": False,
        "optimizer_created": False,
        "backward_performed": False,
        "parameter_updates": False,
        "clustering_metrics_computed": False,
    }
    result_path = output_root / (
        "seed"
        + str(inputs["model_seed"])
        + "_admission_feasibility.json"
    )
    _write_json(result_path, result)
    return result


def multiseed_feasibility_gate(seed_results):
    _require(
        len(seed_results) == len(SUPPORTED_SEEDS),
        "multi-seed audit requires three seeds",
    )
    seed_pass_count = int(sum(
        bool(row[
            "B6_WQ1A0_ADMISSION_FEASIBILITY_SEED_PASS"
        ])
        for row in seed_results
    ))
    mean_correct_clean = float(np.mean([
        row["correct_metrics"]["admitted_clean_fraction"]
        for row in seed_results
    ]))
    mean_null_clean = float(np.mean([
        row["shuffled_null_summary"][
            "admitted_clean_fraction"
        ]["mean"]
        for row in seed_results
    ]))
    mean_correct_all_clean = float(np.mean([
        row["correct_metrics"]["all_clean_top3_fraction"]
        for row in seed_results
    ]))
    mean_null_all_clean = float(np.mean([
        row["shuffled_null_summary"][
            "all_clean_top3_fraction"
        ]["mean"]
        for row in seed_results
    ]))
    mean_correct_pair_win = float(np.mean([
        row["correct_metrics"]["within_sample_pair_win_rate"]
        for row in seed_results
    ]))
    conditions = {
        "at_least_two_of_three_seed_passes": bool(
            seed_pass_count >= 2
        ),
        "mean_correct_clean_fraction_exceeds_mean_null": bool(
            mean_correct_clean > mean_null_clean
        ),
        "mean_correct_all_clean_fraction_exceeds_mean_null": bool(
            mean_correct_all_clean > mean_null_all_clean
        ),
        "mean_correct_pair_win_rate_exceeds_random_reference": bool(
            mean_correct_pair_win
            > THEORETICAL_REFERENCES[
                "random_pair_win_reference"
            ]
        ),
    }
    return {
        "seed_pass_count": seed_pass_count,
        "mean_correct_admitted_clean_fraction": (
            mean_correct_clean
        ),
        "mean_shuffled_null_admitted_clean_fraction": (
            mean_null_clean
        ),
        "mean_correct_all_clean_top3_fraction": (
            mean_correct_all_clean
        ),
        "mean_shuffled_null_all_clean_top3_fraction": (
            mean_null_all_clean
        ),
        "mean_correct_within_sample_pair_win_rate": (
            mean_correct_pair_win
        ),
        "multiseed_gate_conditions": conditions,
        "B6_WQ1A0_ADMISSION_FEASIBILITY_MULTISEED_PASS": bool(
            all(conditions.values())
        ),
    }


def run_audit(
    seeds=SUPPORTED_SEEDS,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    selected = tuple(int(seed) for seed in seeds)
    _require(
        selected == SUPPORTED_SEEDS,
        "B6-WQ1A-0 requires seeds 20 30 50",
    )
    manifest_hash_before = _file_sha256(
        b6.CANONICAL_MANIFEST_PATH
    )
    results = [
        run_seed_audit(seed, output_dir)
        for seed in selected
    ]
    gate = multiseed_feasibility_gate(results)
    manifest_hash_after = _file_sha256(
        b6.CANONICAL_MANIFEST_PATH
    )
    _require(
        manifest_hash_after == manifest_hash_before,
        "B6 canonical manifest changed",
    )
    summary = {
        "stage": STAGE,
        "dataset": DATASET_NAME,
        "condition": CONDITION,
        "model_seeds": list(selected),
        "seed_results": results,
        **gate,
        "canonical_manifest_sha256_before": (
            manifest_hash_before
        ),
        "canonical_manifest_sha256_after": (
            manifest_hash_after
        ),
        "canonical_manifest_modified": False,
        "labels_loaded": False,
        "labels_used": False,
        "optimizer_created": False,
        "backward_performed": False,
        "parameter_updates": False,
        "clustering_metrics_computed": False,
    }
    output_path = _resolve(output_dir) / (
        "b6_wq1a0_multiseed_summary.json"
    )
    _write_json(output_path, summary)
    return summary


def _print_seed_summary(result):
    print("model_seed=" + str(result["model_seed"]))
    for key in (
        "correct_admitted_clean_fraction",
        "random_theoretical_clean_fraction",
        "correct_all_clean_top3_fraction",
        "random_theoretical_all_clean_top3",
        "correct_corrupted_views_admitted_mean",
        "random_theoretical_corrupted_admitted_mean",
        "correct_oracle_set_jaccard_mean",
        "correct_within_sample_pair_win_rate",
        "random_pair_win_reference",
    ):
        print(key + "=" + str(result[key]))
    for name in METRIC_NAMES:
        summary = result["shuffled_null_summary"][name]
        print(
            "shuffled_" + name + "_p95="
            + str(summary["p95"])
        )
        print(
            name + "_one_sided_empirical_p="
            + str(summary["one_sided_empirical_p"])
        )
    print(
        "B6_WQ1A0_ADMISSION_FEASIBILITY_SEED_PASS="
        + str(
            result[
                "B6_WQ1A0_ADMISSION_FEASIBILITY_SEED_PASS"
            ]
        ).lower()
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SUPPORTED_SEEDS),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = run_audit(
        seeds=args.seeds,
        output_dir=args.output_dir,
    )
    for result in summary["seed_results"]:
        _print_seed_summary(result)
    print(
        "B6_WQ1A0_ADMISSION_FEASIBILITY_MULTISEED_PASS="
        + str(
            summary[
                "B6_WQ1A0_ADMISSION_FEASIBILITY_MULTISEED_PASS"
            ]
        ).lower()
    )
    return summary


if __name__ == "__main__":
    main()
