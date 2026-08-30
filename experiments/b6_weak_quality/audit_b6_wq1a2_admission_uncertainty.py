"""B6-WQ1A-2 read-only admission uncertainty and purity diagnostic."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import experiments.b6_weak_quality.audit_b6_wq1a0_admission_feasibility as wq1a0
from irv.b4_information_utility import tensor_sha256
from weak_quality import ndarray_sha256


STAGE = "B6-WQ1A-2"
DATASET_NAME = wq1a0.DATASET_NAME
CONDITION = wq1a0.CONDITION
SUPPORTED_SEED = 20
SAMPLE_NUM = wq1a0.SAMPLE_NUM
VIEW_NUM = wq1a0.VIEW_NUM
TOP_K = wq1a0.ADMITTED_VIEWS_PER_SAMPLE
FORMAL_SHUFFLE_REPEATS = wq1a0.NULL_REPEATS
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20
PERMUTATION_REPEATS = 10000
PERMUTATION_SEED = 20
DEFAULT_OUTPUT_DIR = (
    "outputs/b6_weak_quality/wq1a2_admission_uncertainty/seed20"
)
FORMAL_ROOT = (
    "outputs/b6_weak_quality/"
    "wq1a1_shared_semantic_carrier/formal_seed20"
)
FORMAL_SUMMARY_NAME = "b6_wq1a1_seed20_formal_summary.json"
WQ1A0_ROOT = (
    "outputs/b6_weak_quality/wq1a0_admission_feasibility"
)
EXPECTED_FORMAL_FILE_SHA256 = {
    FORMAL_SUMMARY_NAME: (
        "bc989a0f53dfa02b69a5841384666e5b"
        "3ea479953822e0bc18965f42c048f8eb"
    ),
    "shuffled_acc.npy": (
        "6f7c87be4e29f16c5874b674c46381cb"
        "f3cba97d6474a4339b3352d6a1d5afbe"
    ),
    "shuffled_nmi.npy": (
        "04b7c394d109ef3595f2afa862159cd22"
        "359e7cce63f4e00adbb74d8411dddce"
    ),
    "shuffled_ari.npy": (
        "aca01a5c4641d89e9dc8e18f3077b390"
        "f879969112423e682260489db7c15dc7"
    ),
}
EXPECTED_PROTECTED_FILE_SHA256 = {
    "model.py": (
        "3f866536857f0a5df5d451dec93232c77"
        "f0805ff216db60a2dd894b0c1a357c9"
    ),
    "run.py": (
        "3a766742ca331471e7ece7f111921b7ff"
        "c44c6e883728e43ac7583741427fdc4"
    ),
    "ClusteringTest.py": (
        "5dd576ecd2f9fef161741608f5c94429"
        "e86c28fcb5d38cb443e22b8db7e092b4"
    ),
    "configure.py": (
        "20ea7aee71a66e3ef741056197afba731"
        "e96fa8ef4d95b09607fdc4dc601ec7a"
    ),
    "datasets.py": (
        "52ec339eac819b5a913d3c4c7b6dbdb1"
        "3dcd75a493f2da0efbf66303ba908d80"
    ),
    "weak_quality.py": (
        "2fd09439e9375392859db875dbd2dd032"
        "75ccfe383013b8717967b68eec9c83d"
    ),
    (
        "experiments/b6_weak_quality/"
        "evaluate_b6_wq0_utility_semantic_admission.py"
    ): (
        "10cc2c09d6bfd4b675853b29601af75a"
        "875cd7c2483f301f06a459eb62251bda"
    ),
    (
        "experiments/b6_weak_quality/"
        "audit_b6_wq1a0_admission_feasibility.py"
    ): (
        "5dc407e4c7e87e269fddf867308093a84"
        "417ab4e3d871d388bf96ea98f0989e0"
    ),
    (
        "experiments/b6_weak_quality/"
        "b6_wq0_frozen_baseline_manifest.json"
    ): (
        "f7ba5bd25dfe4dfdbbb3db1f4956c601"
        "db33bf3fffce8f456ff2447c4ffa187e"
    ),
    (
        "experiments/b6_weak_quality/"
        "shared_semantic_isolation.py"
    ): (
        "7beb1d95319f5939acf04738865072932"
        "446ab1678cc30c039f0139cb1ec4fe4"
    ),
    (
        "experiments/b6_weak_quality/"
        "train_b6_wq1a1_shared_semantic_carrier.py"
    ): (
        "2bf034c1795ed969e708f9a3412ca7db"
        "92f84ce05c65babbfe82878346c83fa8"
    ),
}
EXPECTED_WQ1A0_FILE_SHA256 = {
    "seed20_admission_feasibility.json": (
        "47a568a54d3a915d222df596a9b19ef4"
        "37df7485ccfe00b05a3e72ea6f9bf25d"
    ),
    "seed20/correct_admission_mask.npy": (
        "a2064b23b00dd90378871bc080293a1fc"
        "10d13eee51f00460066142e69c00443"
    ),
}
EXPECTED_FORMAL_RESULT = {
    "correct_acc": 0.6952380952380952,
    "shuffled_acc_mean": 0.6275714285714286,
    "shuffled_acc_std": 0.03793295792648069,
    "shuffled_acc_p50": 0.6238095238095238,
    "shuffled_acc_p95": 0.6952380952380952,
    "count_shuffled_ge_correct": 11,
    "one_sided_empirical_p": 0.05970149253731343,
}
EXPECTED_CORRECT_PURITY = {
    "admitted_clean_fraction": 0.9174603174603174,
    "all_clean_top3_fraction": 0.7523809523809524,
    "corrupted_views_admitted_mean": 0.24761904761904763,
    "oracle_set_jaccard_mean": 0.8761904761904762,
    "within_sample_pair_win_rate": 0.9392857142857143,
}
ADMISSION_DIAGNOSTIC_FIELDS = (
    "admitted_clean_fraction",
    "all_clean_top3_fraction",
    "oracle_set_jaccard_mean",
    "corrupted_views_admitted_mean",
    "within_sample_pair_win_rate",
)
PERFORMANCE_FIELDS = ("acc", "nmi", "ari")


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
        json.dump(
            value,
            output_file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        output_file.write("\n")


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_seed(seed):
    seed = int(seed)
    _require(seed == SUPPORTED_SEED, "B6-WQ1A-2 supports only seed20")
    return seed


def protected_file_hashes():
    hashes = {}
    for relative_path, expected_hash in (
        EXPECTED_PROTECTED_FILE_SHA256.items()
    ):
        actual_hash = _file_sha256(_resolve(relative_path))
        _require(
            actual_hash == expected_hash,
            "protected file SHA mismatch: " + relative_path,
        )
        hashes[relative_path] = actual_hash
    return hashes


def _formal_artifact_hashes():
    root = _resolve(FORMAL_ROOT)
    hashes = {}
    for filename, expected_hash in (
        EXPECTED_FORMAL_FILE_SHA256.items()
    ):
        path = root / filename
        _require(path.is_file(), "formal artifact missing: " + filename)
        actual_hash = _file_sha256(path)
        _require(
            actual_hash == expected_hash,
            "formal artifact SHA mismatch: " + filename,
        )
        hashes[filename] = actual_hash
    return hashes


def _wq1a0_artifact_hashes():
    root = _resolve(WQ1A0_ROOT)
    hashes = {}
    for relative_path, expected_hash in (
        EXPECTED_WQ1A0_FILE_SHA256.items()
    ):
        actual_hash = _file_sha256(root / relative_path)
        _require(
            actual_hash == expected_hash,
            "WQ1A-0 artifact SHA mismatch: " + relative_path,
        )
        hashes[relative_path] = actual_hash
    return hashes


def _validate_formal_summary(summary, arrays):
    _require(
        summary.get("stage") == "B6-WQ1A-1"
        and summary.get("mode") == "formal"
        and summary.get("model_seed") == SUPPORTED_SEED,
        "formal summary identity mismatch",
    )
    _require(
        summary["fixed_protocol"]["shuffled_repeats"]
        == FORMAL_SHUFFLE_REPEATS,
        "formal shuffled repeat count mismatch",
    )
    _require(
        summary["B6_WQ1A1_FORMAL_NULL_PASS"] is False
        and summary["B6_WQ1A1_FORMAL_SEED20_PASS"] is False
        and summary["B6_WQ1A1_MULTISEED_ENTRY_ELIGIBLE"] is False,
        "frozen WQ1A-1 formal conclusion changed",
    )
    checks = {
        "correct_acc": summary["correct_u_shared_metrics"]["acc"],
        "shuffled_acc_mean": summary["formal_shuffled_acc_mean"],
        "shuffled_acc_std": summary["formal_shuffled_acc_std"],
        "shuffled_acc_p50": summary["formal_shuffled_acc_p50"],
        "shuffled_acc_p95": summary["formal_shuffled_acc_p95"],
        "count_shuffled_ge_correct": (
            summary["count_shuffled_ge_correct"]
        ),
        "one_sided_empirical_p": (
            summary["one_sided_empirical_p"]
        ),
    }
    _require(
        checks == EXPECTED_FORMAL_RESULT,
        "frozen WQ1A-1 formal metrics changed",
    )
    shuffled_rows = summary.get("shuffled_u")
    _require(
        isinstance(shuffled_rows, list)
        and len(shuffled_rows) == FORMAL_SHUFFLE_REPEATS,
        "formal shuffled provenance count mismatch",
    )
    purity_rows = summary["admission_purity"]["shuffled_u"]
    _require(
        len(purity_rows) == FORMAL_SHUFFLE_REPEATS,
        "formal shuffled purity count mismatch",
    )
    for metric in PERFORMANCE_FIELDS:
        values = arrays[metric]
        summary_values = np.asarray(
            summary["shuffled_" + metric + "_summary"]["values"],
            dtype=np.float64,
        )
        record_values = np.asarray(
            [row["metrics"][metric] for row in shuffled_rows],
            dtype=np.float64,
        )
        _require(
            np.array_equal(values, summary_values)
            and np.array_equal(values, record_values),
            "formal shuffled " + metric + " alignment mismatch",
        )
    _require(
        float(np.mean(arrays["acc"]))
        == EXPECTED_FORMAL_RESULT["shuffled_acc_mean"]
        and float(np.std(arrays["acc"], ddof=0))
        == EXPECTED_FORMAL_RESULT["shuffled_acc_std"]
        and float(np.percentile(arrays["acc"], 50))
        == EXPECTED_FORMAL_RESULT["shuffled_acc_p50"]
        and float(np.percentile(arrays["acc"], 95))
        == EXPECTED_FORMAL_RESULT["shuffled_acc_p95"],
        "formal shuffled ACC array statistics changed",
    )


def load_read_only_inputs(seed=SUPPORTED_SEED):
    seed = _validate_seed(seed)
    protected_hashes = protected_file_hashes()
    formal_hashes = _formal_artifact_hashes()
    wq1a0_hashes = _wq1a0_artifact_hashes()
    formal_root = _resolve(FORMAL_ROOT)
    summary = _read_json(formal_root / FORMAL_SUMMARY_NAME)
    arrays = {}
    for metric in PERFORMANCE_FIELDS:
        values = np.load(
            formal_root / ("shuffled_" + metric + ".npy"),
            allow_pickle=False,
        )
        values = np.asarray(values, dtype=np.float64)
        _require(
            values.shape == (FORMAL_SHUFFLE_REPEATS,),
            "formal shuffled " + metric + " shape mismatch",
        )
        _require(
            np.isfinite(values).all(),
            "formal shuffled " + metric + " non-finite",
        )
        arrays[metric] = values
    _validate_formal_summary(summary, arrays)

    frozen = wq1a0.load_frozen_audit_inputs(seed)
    utility = np.asarray(frozen["utility"])
    corruption_mask = np.asarray(
        frozen["corruption_mask"],
        dtype=bool,
    )
    oracle_clean_mask = np.asarray(
        frozen["oracle_clean_mask"],
        dtype=bool,
    )
    _require(
        utility.shape == (SAMPLE_NUM, VIEW_NUM),
        "frozen Utility shape mismatch",
    )
    _require(
        corruption_mask.shape == (SAMPLE_NUM, VIEW_NUM)
        and np.all(corruption_mask.sum(axis=1) == 2),
        "frozen corruption mask mismatch",
    )
    _require(
        np.array_equal(oracle_clean_mask, ~corruption_mask)
        and np.all(oracle_clean_mask.sum(axis=1) == TOP_K),
        "oracle clean mask mismatch",
    )
    correct_mask = wq1a0.top3_admission_mask(utility)
    stored_mask = np.load(
        _resolve(WQ1A0_ROOT)
        / "seed20/correct_admission_mask.npy",
        allow_pickle=False,
    )
    wq1a0_summary = _read_json(
        _resolve(WQ1A0_ROOT)
        / "seed20_admission_feasibility.json"
    )
    _require(
        np.array_equal(correct_mask, stored_mask)
        and ndarray_sha256(correct_mask)
        == wq1a0_summary["correct_admission_mask_sha256"],
        "WQ1A-0 Correct-U admission mismatch",
    )
    _require(
        wq1a0_summary["correct_metrics"]
        == EXPECTED_CORRECT_PURITY,
        "WQ1A-0 Correct-U purity changed",
    )
    return {
        "seed": seed,
        "utility": utility,
        "corruption_mask": corruption_mask,
        "oracle_clean_mask": oracle_clean_mask,
        "correct_admission_mask": correct_mask,
        "formal_summary": summary,
        "shuffled_metrics": arrays,
        "wq1a0_summary": wq1a0_summary,
        "protected_hashes_before": protected_hashes,
        "formal_hashes_before": formal_hashes,
        "wq1a0_hashes_before": wq1a0_hashes,
        "utility_sha256": frozen["utility_sha256"],
        "corruption_mask_sha256": (
            frozen["corruption_mask_sha256"]
        ),
        "utility_source": frozen["utility_source"],
        "corruption_mask_source": frozen[
            "corruption_mask_source"
        ],
    }


def utility_ordering(utility):
    values = np.asarray(utility)
    _require(
        values.ndim == 2 and values.shape[1] == VIEW_NUM,
        "Utility ordering shape mismatch",
    )
    _require(np.isfinite(values).all(), "Utility ordering non-finite")
    order = np.argsort(-values, axis=1, kind="mergesort")
    sorted_values = np.take_along_axis(values, order, axis=1)
    _require(
        np.all(sorted_values[:, :-1] >= sorted_values[:, 1:]),
        "Utility ordering is not descending",
    )
    return {
        "descending_view_order": order,
        "top1_utility": sorted_values[:, 0],
        "top2_utility": sorted_values[:, 1],
        "top3_utility": sorted_values[:, 2],
        "top4_utility": sorted_values[:, 3],
        "top5_utility": sorted_values[:, 4],
        "margin_12": sorted_values[:, 0] - sorted_values[:, 1],
        "margin_23": sorted_values[:, 1] - sorted_values[:, 2],
        "margin_34": sorted_values[:, 2] - sorted_values[:, 3],
        "margin_45": sorted_values[:, 3] - sorted_values[:, 4],
    }


def build_sample_admission_diagnostics(utility, corruption_mask):
    values = np.asarray(utility)
    corrupt = np.asarray(corruption_mask, dtype=bool)
    _require(
        values.shape == corrupt.shape == (SAMPLE_NUM, VIEW_NUM),
        "sample admission diagnostic shape mismatch",
    )
    oracle = ~corrupt
    correct = wq1a0.top3_admission_mask(values)
    ordering = utility_ordering(values)
    all_clean = np.all(correct == oracle, axis=1).astype(np.int64)
    error = 1 - all_clean
    corrupt_admitted = np.logical_and(
        correct,
        corrupt,
    ).sum(axis=1, dtype=np.int64)
    _require(
        np.array_equal(error, (corrupt_admitted > 0).astype(np.int64)),
        "sample admission error derivation mismatch",
    )
    rows = []
    for sample_id in range(SAMPLE_NUM):
        row = {"sample_id": int(sample_id)}
        for name in (
            "top1_utility",
            "top2_utility",
            "top3_utility",
            "top4_utility",
            "top5_utility",
            "margin_12",
            "margin_23",
            "margin_34",
            "margin_45",
        ):
            row[name] = float(ordering[name][sample_id])
        row.update({
            "all_clean_top3": int(all_clean[sample_id]),
            "admission_error": int(error[sample_id]),
            "corrupt_admitted_count": int(
                corrupt_admitted[sample_id]
            ),
        })
        rows.append(row)
    return {
        "rows": rows,
        "ordering": ordering,
        "correct_admission_mask": correct,
        "oracle_clean_mask": oracle,
        "all_clean_top3": all_clean,
        "admission_error": error,
        "corrupt_admitted_count": corrupt_admitted,
    }


def margin_error_auc(margin_34, admission_error):
    margin = np.asarray(margin_34, dtype=np.float64)
    target = np.asarray(admission_error, dtype=np.int64)
    _require(
        margin.ndim == target.ndim == 1
        and margin.shape == target.shape,
        "margin AUC shape mismatch",
    )
    _require(
        np.isfinite(margin).all()
        and np.array_equal(np.unique(target), np.array([0, 1])),
        "margin AUC requires finite scores and both classes",
    )
    score = -margin
    ranks = rankdata(score, method="average")
    positive_count = int(np.sum(target == 1))
    negative_count = int(np.sum(target == 0))
    positive_rank_sum = float(np.sum(ranks[target == 1]))
    return float(
        (
            positive_rank_sum
            - positive_count * (positive_count + 1) / 2.0
        )
        / (positive_count * negative_count)
    )


def margin_group_summary(margin_34, admission_error):
    margin = np.asarray(margin_34, dtype=np.float64)
    error = np.asarray(admission_error, dtype=np.int64)
    all_clean_values = margin[error == 0]
    error_values = margin[error == 1]
    _require(
        all_clean_values.size > 0 and error_values.size > 0,
        "margin groups must both be non-empty",
    )
    return {
        "median_margin34_allclean": float(
            np.median(all_clean_values)
        ),
        "median_margin34_error": float(
            np.median(error_values)
        ),
        "mean_margin34_allclean": float(
            np.mean(all_clean_values)
        ),
        "mean_margin34_error": float(np.mean(error_values)),
    }


def margin_quartile_diagnostics(
    margin_34,
    all_clean_top3,
    admission_error,
    corrupt_admitted_count,
):
    margin = np.asarray(margin_34, dtype=np.float64)
    all_clean = np.asarray(all_clean_top3, dtype=np.int64)
    error = np.asarray(admission_error, dtype=np.int64)
    corrupt_count = np.asarray(
        corrupt_admitted_count,
        dtype=np.int64,
    )
    _require(
        margin.shape
        == all_clean.shape
        == error.shape
        == corrupt_count.shape
        == (SAMPLE_NUM,),
        "quartile diagnostic shape mismatch",
    )
    order = np.argsort(margin, kind="mergesort")
    partitions = np.array_split(order, 4)
    rows = []
    for quartile_id, indices in enumerate(partitions, start=1):
        quartile_margin = margin[indices]
        clean_fraction = (
            TOP_K - corrupt_count[indices]
        ) / float(TOP_K)
        rows.append({
            "quartile": "Q" + str(quartile_id),
            "sample_count": int(indices.size),
            "rank_start": int(sum(
                partition.size
                for partition in partitions[:quartile_id - 1]
            )),
            "rank_end_exclusive": int(sum(
                partition.size
                for partition in partitions[:quartile_id]
            )),
            "margin34_min": float(np.min(quartile_margin)),
            "margin34_max": float(np.max(quartile_margin)),
            "all_clean_top3_fraction": float(
                np.mean(all_clean[indices])
            ),
            "admission_error_fraction": float(
                np.mean(error[indices])
            ),
            "mean_corrupt_admitted_count": float(
                np.mean(corrupt_count[indices])
            ),
            "admitted_clean_fraction": float(
                np.mean(clean_fraction)
            ),
        })
    return rows


def bootstrap_margin_uncertainty(
    margin_34,
    admission_error,
    repeats=BOOTSTRAP_REPEATS,
    seed=BOOTSTRAP_SEED,
):
    repeats = int(repeats)
    seed = int(seed)
    _require(
        repeats == BOOTSTRAP_REPEATS,
        "bootstrap repeats must remain exactly 2000",
    )
    margin = np.asarray(margin_34, dtype=np.float64)
    error = np.asarray(admission_error, dtype=np.int64)
    _require(
        margin.shape == error.shape == (SAMPLE_NUM,),
        "bootstrap input shape mismatch",
    )
    rng = np.random.default_rng(seed)
    auc_values = np.empty(repeats, dtype=np.float64)
    median_difference = np.empty(repeats, dtype=np.float64)
    for repeat_id in range(repeats):
        indices = rng.integers(
            0,
            SAMPLE_NUM,
            size=SAMPLE_NUM,
        )
        sampled_margin = margin[indices]
        sampled_error = error[indices]
        _require(
            np.unique(sampled_error).size == 2,
            "degenerate bootstrap class sample",
        )
        auc_values[repeat_id] = margin_error_auc(
            sampled_margin,
            sampled_error,
        )
        median_difference[repeat_id] = (
            np.median(sampled_margin[sampled_error == 0])
            - np.median(sampled_margin[sampled_error == 1])
        )
    return {
        "bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
        "margin34_error_auc_bootstrap_mean": float(
            np.mean(auc_values)
        ),
        "margin34_error_auc_ci95": [
            float(np.percentile(auc_values, 2.5)),
            float(np.percentile(auc_values, 97.5)),
        ],
        (
            "median_margin34_allclean_minus_error_"
            "bootstrap_mean"
        ): float(np.mean(median_difference)),
        (
            "median_margin34_allclean_minus_error_ci95"
        ): [
            float(np.percentile(median_difference, 2.5)),
            float(np.percentile(median_difference, 97.5)),
        ],
        "auc_bootstrap_all_finite": bool(
            np.isfinite(auc_values).all()
        ),
        "median_difference_bootstrap_all_finite": bool(
            np.isfinite(median_difference).all()
        ),
    }


def regenerate_formal_shuffled_diagnostics(inputs):
    utility = np.asarray(inputs["utility"])
    oracle = np.asarray(inputs["oracle_clean_mask"], dtype=bool)
    summary = inputs["formal_summary"]
    arrays = inputs["shuffled_metrics"]
    permutation_bank = wq1a0.generate_permutation_bank()
    _require(
        len(permutation_bank) == FORMAL_SHUFFLE_REPEATS,
        "frozen shuffle bank count mismatch",
    )
    formal_records = summary["shuffled_u"]
    formal_purity = summary["admission_purity"]["shuffled_u"]
    rows = []
    utility_hash_match = True
    purity_match = True
    marginal_match = True
    for repeat_id in range(FORMAL_SHUFFLE_REPEATS):
        shuffled = wq1a0.shuffle_utility_within_views(
            utility,
            permutation_bank[repeat_id],
        )
        repeat_marginal_match = bool(all(
            np.array_equal(
                np.sort(shuffled[:, view_id]),
                np.sort(utility[:, view_id]),
            )
            for view_id in range(VIEW_NUM)
        ))
        marginal_match = marginal_match and repeat_marginal_match
        shuffled_hash = tensor_sha256(shuffled)
        utility_hash_match = bool(
            utility_hash_match
            and formal_records[repeat_id]["repeat_id"] == repeat_id
            and formal_records[repeat_id][
                "shuffled_utility_sha256"
            ] == shuffled_hash
        )
        admission = wq1a0.top3_admission_mask(shuffled)
        metrics = wq1a0.admission_metrics(
            admission,
            oracle,
            shuffled,
        )
        stored_purity = formal_purity[repeat_id]
        purity_match = bool(
            purity_match
            and stored_purity["repeat_id"] == repeat_id
            and metrics["admitted_clean_fraction"]
            == stored_purity["admitted_clean_fraction"]
            and metrics["all_clean_top3_fraction"]
            == stored_purity["all_admitted_clean_fraction"]
            and metrics["corrupted_views_admitted_mean"]
            == stored_purity["corrupted_views_admitted_mean"]
            and metrics["oracle_set_jaccard_mean"]
            == stored_purity["oracle_set_jaccard_mean"]
        )
        ordering = utility_ordering(shuffled)
        row = {
            "repeat_id": int(repeat_id),
            "shuffled_utility_sha256": shuffled_hash,
            "per_view_utility_distribution_preserved": (
                repeat_marginal_match
            ),
            "mean_margin34": float(
                np.mean(ordering["margin_34"])
            ),
        }
        row.update(metrics)
        for performance in PERFORMANCE_FIELDS:
            row[performance] = float(
                arrays[performance][repeat_id]
            )
        rows.append(row)
    shuffle_bank_repro_pass = bool(
        utility_hash_match
        and purity_match
        and marginal_match
        and len(rows) == FORMAL_SHUFFLE_REPEATS
    )
    return {
        "rows": rows,
        "utility_hash_alignment_pass": utility_hash_match,
        "formal_purity_alignment_pass": purity_match,
        "per_view_marginal_preservation_pass": marginal_match,
        "B6_WQ1A2_SHUFFLE_BANK_REPRO_PASS": (
            shuffle_bank_repro_pass
        ),
    }


def spearman_correlation(left, right):
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    _require(
        x.ndim == y.ndim == 1
        and x.shape == y.shape
        and x.size > 1,
        "Spearman input shape mismatch",
    )
    _require(
        np.isfinite(x).all() and np.isfinite(y).all(),
        "Spearman input non-finite",
    )
    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")
    x_centered = x_rank - np.mean(x_rank)
    y_centered = y_rank - np.mean(y_rank)
    denominator = float(
        np.sqrt(
            np.dot(x_centered, x_centered)
            * np.dot(y_centered, y_centered)
        )
    )
    _require(denominator > 0.0, "Spearman constant input")
    return float(np.dot(x_centered, y_centered) / denominator)


def empirical_two_sided_p(count_as_or_more_extreme, repeats):
    count = int(count_as_or_more_extreme)
    repeats = int(repeats)
    _require(
        repeats > 0 and 0 <= count <= repeats,
        "empirical permutation count mismatch",
    )
    return float((1 + count) / (repeats + 1))


def spearman_permutation_test(
    admission_diagnostic,
    performance,
    repeats=PERMUTATION_REPEATS,
    seed=PERMUTATION_SEED,
):
    repeats = int(repeats)
    seed = int(seed)
    _require(
        repeats == PERMUTATION_REPEATS,
        "permutation repeats must remain exactly 10000",
    )
    x = np.asarray(admission_diagnostic, dtype=np.float64)
    y = np.asarray(performance, dtype=np.float64)
    _require(
        x.shape == y.shape == (FORMAL_SHUFFLE_REPEATS,),
        "permutation input shape mismatch",
    )
    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")
    x_centered = x_rank - np.mean(x_rank)
    y_centered = y_rank - np.mean(y_rank)
    denominator = float(
        np.sqrt(
            np.dot(x_centered, x_centered)
            * np.dot(y_centered, y_centered)
        )
    )
    _require(denominator > 0.0, "permutation constant input")
    observed = float(
        np.dot(x_centered, y_centered) / denominator
    )
    rng = np.random.default_rng(seed)
    count_extreme = 0
    for _ in range(repeats):
        permuted = y_centered[rng.permutation(y_centered.size)]
        rho = float(np.dot(x_centered, permuted) / denominator)
        if abs(rho) >= abs(observed):
            count_extreme += 1
    return {
        "rho_observed": observed,
        "permutation_repeats": repeats,
        "permutation_seed": seed,
        "count_abs_rho_permuted_ge_observed": int(
            count_extreme
        ),
        "two_sided_empirical_p": empirical_two_sided_p(
            count_extreme,
            repeats,
        ),
    }


def correlation_diagnostics(shuffled_rows):
    _require(
        len(shuffled_rows) == FORMAL_SHUFFLE_REPEATS,
        "correlation requires exactly 200 shuffled rows",
    )
    result = {
        "primary_performance_metric": "acc",
        "correlations": {},
    }
    arrays = {
        name: np.asarray(
            [row[name] for row in shuffled_rows],
            dtype=np.float64,
        )
        for name in ADMISSION_DIAGNOSTIC_FIELDS
        + PERFORMANCE_FIELDS
    }
    for diagnostic in ADMISSION_DIAGNOSTIC_FIELDS:
        result["correlations"][diagnostic] = {}
        for performance in PERFORMANCE_FIELDS:
            result["correlations"][diagnostic][performance] = {
                "spearman_rho": spearman_correlation(
                    arrays[diagnostic],
                    arrays[performance],
                )
            }
    clean_test = spearman_permutation_test(
        arrays["admitted_clean_fraction"],
        arrays["acc"],
    )
    allclean_test = spearman_permutation_test(
        arrays["all_clean_top3_fraction"],
        arrays["acc"],
    )
    result.update({
        "rho_clean_fraction_acc": clean_test["rho_observed"],
        "p_clean_fraction_acc": (
            clean_test["two_sided_empirical_p"]
        ),
        "clean_fraction_acc_permutation": clean_test,
        "rho_allclean_fraction_acc": (
            allclean_test["rho_observed"]
        ),
        "p_allclean_fraction_acc": (
            allclean_test["two_sided_empirical_p"]
        ),
        "allclean_fraction_acc_permutation": allclean_test,
        "rho_jaccard_acc": result["correlations"][
            "oracle_set_jaccard_mean"
        ]["acc"]["spearman_rho"],
        "rho_corrupt_admitted_acc": result["correlations"][
            "corrupted_views_admitted_mean"
        ]["acc"]["spearman_rho"],
        "rho_pairwin_acc": result["correlations"][
            "within_sample_pair_win_rate"
        ]["acc"]["spearman_rho"],
    })
    return result


def strict_percentile(null_values, observed_value):
    values = np.asarray(null_values, dtype=np.float64)
    _require(
        values.shape == (FORMAL_SHUFFLE_REPEATS,)
        and np.isfinite(values).all(),
        "percentile null shape mismatch",
    )
    return float(100.0 * np.mean(values < float(observed_value)))


def fit_null_purity_trend(shuffled_purity, shuffled_acc):
    x = np.asarray(shuffled_purity, dtype=np.float64)
    y = np.asarray(shuffled_acc, dtype=np.float64)
    _require(
        x.shape == y.shape == (FORMAL_SHUFFLE_REPEATS,),
        "null regression requires exactly 200 shuffled points",
    )
    design = np.column_stack((
        np.ones(FORMAL_SHUFFLE_REPEATS, dtype=np.float64),
        x,
    ))
    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )
    intercept = float(coefficients[0])
    slope = float(coefficients[1])
    fitted = intercept + slope * x
    residual = y - fitted
    total = y - np.mean(y)
    total_sum_squares = float(np.dot(total, total))
    _require(total_sum_squares > 0.0, "null ACC is constant")
    r2 = float(
        1.0
        - np.dot(residual, residual) / total_sum_squares
    )
    return {
        "fit_population": "200 shuffled-U arms only",
        "correct_u_included_in_fit": False,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
    }


def _current_interpretation(margin_flag, purity_flag):
    if margin_flag and purity_flag:
        return (
            "Utility ranking is informative, margin identifies admission "
            "ambiguity, and higher admission purity is linked to higher "
            "shared-semantic ACC. A separately preregistered selective "
            "semantic-admission study is eligible."
        )
    if margin_flag:
        return (
            "Top3 ambiguity is detectable, but clean admission purity does "
            "not stably determine semantic ACC. Close the current shared-"
            "semantic purification action; do not add uncertainty gating."
        )
    return (
        "Top3/Top4 margin is not a reliable admission-uncertainty signal. "
        "Do not design a margin gate."
    )


def run_audit(seed=SUPPORTED_SEED, output_dir=DEFAULT_OUTPUT_DIR):
    inputs = load_read_only_inputs(seed)
    sample = build_sample_admission_diagnostics(
        inputs["utility"],
        inputs["corruption_mask"],
    )
    margin_34 = sample["ordering"]["margin_34"]
    admission_error = sample["admission_error"]
    margin_auc = margin_error_auc(
        margin_34,
        admission_error,
    )
    margin_groups = margin_group_summary(
        margin_34,
        admission_error,
    )
    quartiles = margin_quartile_diagnostics(
        margin_34,
        sample["all_clean_top3"],
        admission_error,
        sample["corrupt_admitted_count"],
    )
    bootstrap = bootstrap_margin_uncertainty(
        margin_34,
        admission_error,
    )

    correct_metrics = wq1a0.admission_metrics(
        sample["correct_admission_mask"],
        sample["oracle_clean_mask"],
        inputs["utility"],
    )
    _require(
        correct_metrics == EXPECTED_CORRECT_PURITY,
        "Correct-U purity exact reproduction failed",
    )
    shuffled = regenerate_formal_shuffled_diagnostics(inputs)
    _require(
        shuffled["B6_WQ1A2_SHUFFLE_BANK_REPRO_PASS"],
        "formal shuffle bank reproduction failed",
    )
    correlations = correlation_diagnostics(shuffled["rows"])

    shuffled_arrays = {
        name: np.asarray(
            [row[name] for row in shuffled["rows"]],
            dtype=np.float64,
        )
        for name in ADMISSION_DIAGNOSTIC_FIELDS
    }
    correct_percentiles = {
        "correct_clean_fraction_percentile": strict_percentile(
            shuffled_arrays["admitted_clean_fraction"],
            correct_metrics["admitted_clean_fraction"],
        ),
        "correct_allclean_percentile": strict_percentile(
            shuffled_arrays["all_clean_top3_fraction"],
            correct_metrics["all_clean_top3_fraction"],
        ),
        "correct_jaccard_percentile": strict_percentile(
            shuffled_arrays["oracle_set_jaccard_mean"],
            correct_metrics["oracle_set_jaccard_mean"],
        ),
        "correct_pairwin_percentile": strict_percentile(
            shuffled_arrays["within_sample_pair_win_rate"],
            correct_metrics["within_sample_pair_win_rate"],
        ),
        "correct_acc_percentile": strict_percentile(
            inputs["shuffled_metrics"]["acc"],
            EXPECTED_FORMAL_RESULT["correct_acc"],
        ),
    }
    _require(
        correct_percentiles["correct_acc_percentile"] == 94.5,
        "Correct-U formal ACC percentile changed",
    )
    regression = fit_null_purity_trend(
        shuffled_arrays["admitted_clean_fraction"],
        inputs["shuffled_metrics"]["acc"],
    )
    predicted_correct_acc = float(
        regression["intercept"]
        + regression["slope"]
        * correct_metrics["admitted_clean_fraction"]
    )
    correct_acc_residual = float(
        EXPECTED_FORMAL_RESULT["correct_acc"]
        - predicted_correct_acc
    )

    margin_informative = bool(
        margin_auc > 0.65
        and margin_groups["median_margin34_error"]
        < margin_groups["median_margin34_allclean"]
        and bootstrap["margin34_error_auc_ci95"][0] > 0.5
    )
    purity_performance_link = bool(
        correlations["rho_clean_fraction_acc"] > 0.0
        and correlations["p_clean_fraction_acc"] < 0.05
    )
    action_eligible = bool(
        margin_informative and purity_performance_link
    )

    protected_hashes_after = protected_file_hashes()
    formal_hashes_after = _formal_artifact_hashes()
    wq1a0_hashes_after = _wq1a0_artifact_hashes()
    protected_unchanged = bool(
        protected_hashes_after
        == inputs["protected_hashes_before"]
        and formal_hashes_after
        == inputs["formal_hashes_before"]
        and wq1a0_hashes_after
        == inputs["wq1a0_hashes_before"]
    )
    _require(
        protected_unchanged,
        "frozen scientific inputs changed during audit",
    )

    output_root = _resolve(output_dir)
    sample_path = output_root / "sample_admission_diagnostic.csv"
    quartile_path = output_root / "margin_quartile_summary.csv"
    shuffled_path = (
        output_root
        / "formal_shuffled_admission_diagnostics.csv"
    )
    correlation_path = output_root / "correlation_summary.json"
    bootstrap_path = output_root / "bootstrap_summary.json"
    metadata_path = output_root / "metadata.json"
    summary_path = output_root / "b6_wq1a2_seed20_summary.json"

    sample_fields = (
        "sample_id",
        "top1_utility",
        "top2_utility",
        "top3_utility",
        "top4_utility",
        "top5_utility",
        "margin_12",
        "margin_23",
        "margin_34",
        "margin_45",
        "all_clean_top3",
        "admission_error",
        "corrupt_admitted_count",
    )
    quartile_fields = (
        "quartile",
        "sample_count",
        "rank_start",
        "rank_end_exclusive",
        "margin34_min",
        "margin34_max",
        "all_clean_top3_fraction",
        "admission_error_fraction",
        "mean_corrupt_admitted_count",
        "admitted_clean_fraction",
    )
    shuffled_fields = (
        "repeat_id",
        "shuffled_utility_sha256",
        "per_view_utility_distribution_preserved",
        "admitted_clean_fraction",
        "all_clean_top3_fraction",
        "corrupted_views_admitted_mean",
        "oracle_set_jaccard_mean",
        "within_sample_pair_win_rate",
        "mean_margin34",
        "acc",
        "nmi",
        "ari",
    )
    _write_csv(sample_path, sample["rows"], sample_fields)
    _write_csv(quartile_path, quartiles, quartile_fields)
    _write_csv(shuffled_path, shuffled["rows"], shuffled_fields)
    _write_json(correlation_path, correlations)
    _write_json(bootstrap_path, bootstrap)

    metadata = {
        "stage": STAGE,
        "dataset": DATASET_NAME,
        "condition": CONDITION,
        "model_seed": int(seed),
        "diagnostic_only": True,
        "primary_uncertainty_variable": "margin_34",
        "margin_definition": "rank3 Utility minus rank4 Utility",
        "stable_sort_policy": (
            "descending numpy mergesort; low view index wins ties"
        ),
        "quartile_policy": (
            "stable ascending margin_34 rank split by numpy array_split "
            "into Q1..Q4; no post-hoc boundary changes"
        ),
        "sample_id_policy": "zero-based frozen sample order",
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "permutation_repeats": PERMUTATION_REPEATS,
        "permutation_seed": PERMUTATION_SEED,
        "shuffle_repeats": FORMAL_SHUFFLE_REPEATS,
        "shuffle_seed": wq1a0.SHUFFLE_SEED,
        "shuffle_bank_source": (
            "WQ1A-0 generate_permutation_bank()"
        ),
        "top3_helper_source": (
            "WQ1A-0 top3_admission_mask()"
        ),
        "labels_loaded": False,
        "clustering_labels_used": False,
        "model_loaded": False,
        "projector_training_performed": False,
        "optimizer_created": False,
        "backward_performed": False,
        "kmeans_fit_performed": False,
        "parameter_updates": False,
        "wq1a1_formal_conclusion_changed": False,
        "wq1a1_rescue_claimed": False,
        "multiseed_conclusion_claimed": False,
        "interpretation_rules": {
            "A_true_B_true": (
                "Eligible only for a new preregistered U + admission-"
                "confidence selective semantic-admission hypothesis."
            ),
            "A_true_B_false": (
                "Close the current shared-semantic purification action; "
                "do not continue uncertainty gating."
            ),
            "A_false": (
                "Do not use margin_34 to design a new admission gate."
            ),
        },
        "current_interpretation": _current_interpretation(
            margin_informative,
            purity_performance_link,
        ),
        "protected_file_sha256_before": (
            inputs["protected_hashes_before"]
        ),
        "protected_file_sha256_after": protected_hashes_after,
        "formal_artifact_sha256_before": (
            inputs["formal_hashes_before"]
        ),
        "formal_artifact_sha256_after": formal_hashes_after,
        "wq1a0_artifact_sha256_before": (
            inputs["wq1a0_hashes_before"]
        ),
        "wq1a0_artifact_sha256_after": wq1a0_hashes_after,
        "B6_WQ1A2_PROTECTED_INPUTS_UNCHANGED_PASS": (
            protected_unchanged
        ),
    }
    _write_json(metadata_path, metadata)

    summary = {
        "stage": STAGE,
        "dataset": DATASET_NAME,
        "condition": CONDITION,
        "model_seed": int(seed),
        "utility_shape": list(inputs["utility"].shape),
        "utility_sha256": inputs["utility_sha256"],
        "corruption_mask_sha256": (
            inputs["corruption_mask_sha256"]
        ),
        "formal_input_summary": _display(
            _resolve(FORMAL_ROOT) / FORMAL_SUMMARY_NAME
        ),
        "formal_input_artifact_sha256": formal_hashes_after,
        "frozen_wq1a1_formal_result": EXPECTED_FORMAL_RESULT,
        "frozen_wq1a1_formal_null_pass": False,
        "frozen_wq1a1_formal_seed20_pass": False,
        "frozen_wq1a1_multiseed_entry_eligible": False,
        "margin34_error_auc": margin_auc,
        **margin_groups,
        "margin34_error_auc_ci95": (
            bootstrap["margin34_error_auc_ci95"]
        ),
        "margin34_error_auc_bootstrap_mean": (
            bootstrap[
                "margin34_error_auc_bootstrap_mean"
            ]
        ),
        (
            "median_margin34_allclean_minus_error_ci95"
        ): bootstrap[
            "median_margin34_allclean_minus_error_ci95"
        ],
        "margin_quartiles": quartiles,
        "Q1_all_clean_fraction": (
            quartiles[0]["all_clean_top3_fraction"]
        ),
        "Q2_all_clean_fraction": (
            quartiles[1]["all_clean_top3_fraction"]
        ),
        "Q3_all_clean_fraction": (
            quartiles[2]["all_clean_top3_fraction"]
        ),
        "Q4_all_clean_fraction": (
            quartiles[3]["all_clean_top3_fraction"]
        ),
        "correct_admission_metrics": correct_metrics,
        **correct_percentiles,
        "rho_clean_fraction_acc": (
            correlations["rho_clean_fraction_acc"]
        ),
        "p_clean_fraction_acc": (
            correlations["p_clean_fraction_acc"]
        ),
        "rho_allclean_fraction_acc": (
            correlations["rho_allclean_fraction_acc"]
        ),
        "p_allclean_fraction_acc": (
            correlations["p_allclean_fraction_acc"]
        ),
        "rho_jaccard_acc": correlations["rho_jaccard_acc"],
        "rho_corrupt_admitted_acc": (
            correlations["rho_corrupt_admitted_acc"]
        ),
        "rho_pairwin_acc": correlations["rho_pairwin_acc"],
        "purity_performance_correlations": (
            correlations["correlations"]
        ),
        "null_purity_acc_regression": regression,
        "predicted_correct_acc_from_null_purity": (
            predicted_correct_acc
        ),
        "correct_acc_residual": correct_acc_residual,
        "sample_admission_diagnostic": _display(sample_path),
        "margin_quartile_summary": _display(quartile_path),
        "formal_shuffled_admission_diagnostics": (
            _display(shuffled_path)
        ),
        "correlation_summary": _display(correlation_path),
        "bootstrap_summary": _display(bootstrap_path),
        "metadata": _display(metadata_path),
        "B6_WQ1A2_SHUFFLE_BANK_REPRO_PASS": shuffled[
            "B6_WQ1A2_SHUFFLE_BANK_REPRO_PASS"
        ],
        "B6_WQ1A2_MARGIN_INFORMATIVE": margin_informative,
        "B6_WQ1A2_PURITY_PERFORMANCE_LINK": (
            purity_performance_link
        ),
        "B6_WQ1A2_UNCERTAINTY_ACTION_ELIGIBLE": (
            action_eligible
        ),
        "B6_WQ1A2_PROTECTED_INPUTS_UNCHANGED_PASS": (
            protected_unchanged
        ),
        "diagnostic_only": True,
        "wq1a1_rescue_claimed": False,
    }
    _write_json(summary_path, summary)
    return summary


def _print_summary(summary):
    for key in (
        "margin34_error_auc",
        "margin34_error_auc_ci95",
        "median_margin34_allclean",
        "median_margin34_error",
        "Q1_all_clean_fraction",
        "Q2_all_clean_fraction",
        "Q3_all_clean_fraction",
        "Q4_all_clean_fraction",
        "rho_clean_fraction_acc",
        "p_clean_fraction_acc",
        "rho_allclean_fraction_acc",
        "p_allclean_fraction_acc",
        "rho_jaccard_acc",
        "rho_corrupt_admitted_acc",
        "correct_clean_fraction_percentile",
        "correct_allclean_percentile",
        "correct_acc_percentile",
        "predicted_correct_acc_from_null_purity",
        "correct_acc_residual",
        "B6_WQ1A2_SHUFFLE_BANK_REPRO_PASS",
        "B6_WQ1A2_MARGIN_INFORMATIVE",
        "B6_WQ1A2_PURITY_PERFORMANCE_LINK",
        "B6_WQ1A2_UNCERTAINTY_ACTION_ELIGIBLE",
    ):
        value = summary[key]
        if isinstance(value, bool):
            value = str(value).lower()
        print(key + "=" + str(value))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = run_audit(
        seed=args.seed,
        output_dir=args.output_dir,
    )
    _print_summary(summary)
    return summary


if __name__ == "__main__":
    main()
