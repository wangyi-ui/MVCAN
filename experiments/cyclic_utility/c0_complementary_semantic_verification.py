"""C0 complementary-view reciprocal semantic verification primitives."""

import math
from itertools import combinations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score


SAMPLE_NUM = 1400
VIEW_NUM = 6
CLASS_NUM = 7
DIRECTION_COUNT = 20
SUBSET_SIZE = 3
PROBABILITY_ATOL = 1e-6
BOUND_ATOL = 1e-10
JSD_EPSILON = 1e-12
ALPHA = 0.05
RISK_COVERAGES = (0.2, 0.4, 0.6)
SCORE_NAMES = ("cycle", "confidence", "jsd", "shuffle")


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _frozen_tensor(value, *, dtype=None, device=None):
    if torch.is_tensor(value):
        tensor = value.detach()
        return tensor.to(
            dtype=tensor.dtype if dtype is None else dtype,
            device=tensor.device if device is None else device,
        ).detach()
    array = np.array(value, copy=True, order="C")
    return torch.as_tensor(array, dtype=dtype, device=device).detach()


def _as_numpy(value, dtype=None):
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    return np.ascontiguousarray(array)


def _readonly_array(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def validate_q_aligned(q_aligned):
    """Validate and freeze q_aligned: [N,V,K] = [1400,6,7]."""
    q_values = _frozen_tensor(q_aligned)
    _require(
        q_values.shape == (SAMPLE_NUM, VIEW_NUM, CLASS_NUM)
        and q_values.is_floating_point()
        and bool(torch.isfinite(q_values).all().item())
        and bool(torch.all(q_values >= 0.0).item())
        and bool(torch.all(q_values <= 1.0 + BOUND_ATOL).item()),
        "q_aligned tensor boundary mismatch",
    )
    row_mass = q_values.sum(dim=-1)
    _require(
        bool(
            torch.allclose(
                row_mass,
                torch.ones_like(row_mass),
                rtol=0.0,
                atol=PROBABILITY_ATOL,
            )
        ),
        "q_aligned probability mass mismatch",
    )
    assignments = q_values.argmax(dim=-1)
    expected = torch.arange(CLASS_NUM, device=q_values.device)
    for view_id in range(VIEW_NUM):
        present = torch.unique(assignments[:, view_id]).sort().values
        _require(
            torch.equal(present, expected),
            "q_aligned is missing a cluster in view " + str(view_id),
        )
    _require(
        not q_values.requires_grad and q_values.grad_fn is None,
        "q_aligned did not stop gradients",
    )
    return q_values.detach()


def enumerate_complementary_splits():
    """Enumerate all 20 directional 3-view generator/complement pairs."""
    generator = np.asarray(
        list(combinations(range(VIEW_NUM), SUBSET_SIZE)),
        dtype=np.int64,
    )
    verifier = np.asarray(
        [
            [view for view in range(VIEW_NUM) if view not in set(group)]
            for group in generator.tolist()
        ],
        dtype=np.int64,
    )
    validate_complementary_splits(generator, verifier)
    return (
        _readonly_array(generator, dtype=np.int64),
        _readonly_array(verifier, dtype=np.int64),
    )


def validate_complementary_splits(generator_subsets, verifier_subsets):
    """Validate directional G/H arrays: [S,3] and [S,3]."""
    generator = _as_numpy(generator_subsets, dtype=np.int64)
    verifier = _as_numpy(verifier_subsets, dtype=np.int64)
    expected_generator = np.asarray(
        list(combinations(range(VIEW_NUM), SUBSET_SIZE)),
        dtype=np.int64,
    )
    _require(
        generator.shape == verifier.shape == (
            DIRECTION_COUNT,
            SUBSET_SIZE,
        ),
        "complementary split shape mismatch",
    )
    all_views = set(range(VIEW_NUM))
    for direction_id in range(DIRECTION_COUNT):
        group = set(generator[direction_id].tolist())
        complement = set(verifier[direction_id].tolist())
        _require(
            len(group) == len(complement) == SUBSET_SIZE
            and group.isdisjoint(complement)
            and group | complement == all_views,
            "generator/verifier complement invariant failed",
        )
    _require(
        np.array_equal(generator, expected_generator)
        and np.unique(generator, axis=0).shape[0] == DIRECTION_COUNT,
        "generator subsets must contain every size-three subset once",
    )
    return {
        "direction_count": DIRECTION_COUNT,
        "generator_shape": list(generator.shape),
        "verifier_shape": list(verifier.shape),
        "generator_size_exact_pass": True,
        "verifier_size_exact_pass": True,
        "disjoint_pass": True,
        "full_six_view_union_pass": True,
        "all_generator_subsets_unique_pass": True,
        "all_size_three_generator_subsets_once_pass": True,
    }


@torch.no_grad()
def build_complementary_posteriors(
    q_aligned,
    generator_subsets=None,
    verifier_subsets=None,
):
    """Build p_gen/p_ver: [N,S,K] = [1400,20,7]."""
    q_values = validate_q_aligned(q_aligned)
    if generator_subsets is None or verifier_subsets is None:
        generator_subsets, verifier_subsets = (
            enumerate_complementary_splits()
        )
    split_audit = validate_complementary_splits(
        generator_subsets, verifier_subsets
    )
    generator_index = _frozen_tensor(
        generator_subsets,
        dtype=torch.long,
        device=q_values.device,
    )
    verifier_index = _frozen_tensor(
        verifier_subsets,
        dtype=torch.long,
        device=q_values.device,
    )

    # q_values[:, generator_index, :]: [N,S,3,K].
    p_gen = q_values[:, generator_index, :].mean(dim=2).detach()
    p_ver = q_values[:, verifier_index, :].mean(dim=2).detach()
    expected_shape = (SAMPLE_NUM, DIRECTION_COUNT, CLASS_NUM)
    _require(
        p_gen.shape == p_ver.shape == expected_shape
        and bool(torch.isfinite(p_gen).all().item())
        and bool(torch.isfinite(p_ver).all().item()),
        "complementary posterior shape/finite boundary mismatch",
    )
    for name, posterior in (("p_gen", p_gen), ("p_ver", p_ver)):
        _require(
            bool(
                torch.allclose(
                    posterior.sum(dim=-1),
                    torch.ones_like(posterior[..., 0]),
                    rtol=0.0,
                    atol=PROBABILITY_ATOL,
                )
            ),
            name + " probability mass mismatch",
        )
        _require(
            not posterior.requires_grad and posterior.grad_fn is None,
            name + " did not stop gradients",
        )
    return {
        "p_gen": p_gen,
        "p_ver": p_ver,
        "generator_subsets": _frozen_tensor(
            generator_subsets, dtype=torch.long
        ),
        "verifier_subsets": _frozen_tensor(
            verifier_subsets, dtype=torch.long
        ),
        "audit": {
            "splits": split_audit,
            "p_gen_shape": list(p_gen.shape),
            "p_ver_shape": list(p_ver.shape),
            "p_gen_finite_pass": True,
            "p_ver_finite_pass": True,
            "p_gen_probability_mass_pass": True,
            "p_ver_probability_mass_pass": True,
            "p_gen_stop_gradient_pass": True,
            "p_ver_stop_gradient_pass": True,
            "second_softmax_used": False,
        },
    }


def historical_jsd_consistency(p_gen, p_ver):
    """Return C_jsd = 1 - JSD(p_gen,p_ver)/log(2): [N,S]."""
    first = _frozen_tensor(p_gen, dtype=torch.float64)
    second = _frozen_tensor(
        p_ver, dtype=torch.float64, device=first.device
    )
    _require(
        first.shape == second.shape
        and first.shape == (SAMPLE_NUM, DIRECTION_COUNT, CLASS_NUM)
        and bool(torch.isfinite(first).all().item())
        and bool(torch.isfinite(second).all().item()),
        "JSD posterior boundary mismatch",
    )
    midpoint = 0.5 * (first + second)
    safe_first = torch.clamp(first, min=JSD_EPSILON)
    safe_second = torch.clamp(second, min=JSD_EPSILON)
    safe_midpoint = torch.clamp(midpoint, min=JSD_EPSILON)
    kl_first = torch.sum(
        first * (torch.log(safe_first) - torch.log(safe_midpoint)),
        dim=-1,
    )
    kl_second = torch.sum(
        second * (torch.log(safe_second) - torch.log(safe_midpoint)),
        dim=-1,
    )
    js_divergence = 0.5 * (kl_first + kl_second)
    consistency = 1.0 - js_divergence / math.log(2.0)
    _require(
        bool(torch.isfinite(consistency).all().item())
        and bool(torch.all(consistency >= -BOUND_ATOL).item())
        and bool(torch.all(consistency <= 1.0 + BOUND_ATOL).item()),
        "historical JSD consistency range mismatch",
    )
    return torch.clamp(consistency, min=0.0, max=1.0).detach()


@torch.no_grad()
def build_cycle_scores(q_aligned):
    """Build all pre-GT C0 predictions and scores from q_aligned only."""
    posteriors = build_complementary_posteriors(q_aligned)
    p_gen = posteriors["p_gen"]
    p_ver = posteriors["p_ver"]

    # y_gen/conf_gen/support_ver/closure: [N,S] = [1400,20].
    conf_gen, y_gen = torch.max(p_gen, dim=-1)
    y_ver = torch.argmax(p_ver, dim=-1)
    support_ver = torch.gather(
        p_ver, dim=-1, index=y_gen.unsqueeze(-1)
    ).squeeze(-1)
    closure = (y_ver == y_gen).detach()

    # U_cycle: [N,S] = closure * sqrt(conf_gen * support_ver).
    U_cycle = (
        closure.to(dtype=p_gen.dtype)
        * torch.sqrt(conf_gen * support_ver)
    ).detach()
    C_conf = conf_gen.clone().detach()
    C_jsd = historical_jsd_consistency(p_gen, p_ver).to(
        dtype=p_gen.dtype
    ).detach()

    shuffle_permutation = (
        torch.arange(SAMPLE_NUM, device=p_ver.device) + 1
    ) % SAMPLE_NUM
    p_ver_null = p_ver.index_select(0, shuffle_permutation).detach()
    y_ver_null = torch.argmax(p_ver_null, dim=-1)
    support_ver_null = torch.gather(
        p_ver_null, dim=-1, index=y_gen.unsqueeze(-1)
    ).squeeze(-1)
    closure_null = (y_ver_null == y_gen).detach()
    U_cycle_shuffle = (
        closure_null.to(dtype=p_gen.dtype)
        * torch.sqrt(conf_gen * support_ver_null)
    ).detach()

    score_shape = (SAMPLE_NUM, DIRECTION_COUNT)
    for name, value in (
        ("y_gen", y_gen),
        ("conf_gen", conf_gen),
        ("support_ver", support_ver),
        ("closure", closure),
        ("U_cycle", U_cycle),
        ("C_conf", C_conf),
        ("C_jsd", C_jsd),
        ("U_cycle_shuffle", U_cycle_shuffle),
    ):
        _require(
            value.shape == score_shape,
            name + " shape mismatch",
        )
    for name, value in (
        ("conf_gen", conf_gen),
        ("support_ver", support_ver),
        ("U_cycle", U_cycle),
        ("C_conf", C_conf),
        ("C_jsd", C_jsd),
        ("U_cycle_shuffle", U_cycle_shuffle),
    ):
        _require(
            bool(torch.isfinite(value).all().item())
            and bool(torch.all(value >= -BOUND_ATOL).item())
            and bool(torch.all(value <= 1.0 + BOUND_ATOL).item()),
            name + " finite/range mismatch",
        )
    _require(
        bool(torch.all(U_cycle[~closure] == 0.0).item()),
        "U_cycle must be zero when verifier disagrees",
    )
    inverse = torch.argsort(shuffle_permutation)
    verifier_multiset_preserved = bool(
        torch.equal(p_ver_null.index_select(0, inverse), p_ver)
    )
    _require(
        bool(torch.all(shuffle_permutation != torch.arange(
            SAMPLE_NUM, device=p_ver.device
        )).item())
        and verifier_multiset_preserved,
        "sample-correspondence shuffle invariant failed",
    )

    return {
        **posteriors,
        "y_gen": y_gen.detach(),
        "conf_gen": conf_gen.detach(),
        "y_ver": y_ver.detach(),
        "support_ver": support_ver.detach(),
        "closure": closure.detach(),
        "U_cycle": U_cycle.detach(),
        "C_conf": C_conf.detach(),
        "C_jsd": C_jsd.detach(),
        "shuffle_permutation": shuffle_permutation.detach(),
        "p_ver_null": p_ver_null.detach(),
        "y_ver_null": y_ver_null.detach(),
        "support_ver_null": support_ver_null.detach(),
        "closure_null": closure_null.detach(),
        "U_cycle_shuffle": U_cycle_shuffle.detach(),
        "audit": {
            **posteriors["audit"],
            "y_gen_shape": list(y_gen.shape),
            "conf_gen_shape": list(conf_gen.shape),
            "support_ver_shape": list(support_ver.shape),
            "closure_shape": list(closure.shape),
            "U_cycle_shape": list(U_cycle.shape),
            "C_conf_shape": list(C_conf.shape),
            "C_jsd_shape": list(C_jsd.shape),
            "U_cycle_shuffle_shape": list(U_cycle_shuffle.shape),
            "U_cycle_exact_formula_pass": True,
            "U_cycle_range_pass": True,
            "U_cycle_zero_on_disagreement_pass": True,
            "primary_operator_is_jsd": False,
            "historical_jsd_used_as_baseline_only": True,
            "confidence_used_as_baseline_only": True,
            "deterministic_no_rng": True,
            "derangement_no_fixed_point_pass": True,
            "generator_unchanged_pass": True,
            "verifier_row_multiset_preserved": (
                verifier_multiset_preserved
            ),
            "R_used": False,
            "labels_used": False,
        },
    }


def fit_global_cluster_mapping_once(
    native_global_cluster,
    ground_truth,
):
    """Fit one post-seal native-global-cluster to GT-class permutation."""
    clusters = _as_numpy(native_global_cluster, dtype=np.int64)
    labels = _as_numpy(ground_truth, dtype=np.int64)
    expected = np.arange(CLASS_NUM, dtype=np.int64)
    _require(
        clusters.shape == labels.shape == (SAMPLE_NUM,)
        and np.array_equal(np.unique(clusters), expected)
        and np.array_equal(np.unique(labels), expected),
        "global cluster mapping input boundary mismatch",
    )
    contingency = np.zeros((CLASS_NUM, CLASS_NUM), dtype=np.int64)
    np.add.at(contingency, (clusters, labels), 1)
    rows, columns = linear_sum_assignment(
        contingency.max() - contingency
    )
    mapping = np.full(CLASS_NUM, -1, dtype=np.int64)
    mapping[rows] = columns
    _require(
        np.array_equal(np.sort(mapping), expected),
        "global cluster mapping is not a full permutation",
    )
    return {
        "mapping": _readonly_array(mapping, dtype=np.int64),
        "contingency": _readonly_array(
            contingency, dtype=np.int64
        ),
        "matched_count": int(contingency[rows, columns].sum()),
        "mapping_fit_source": (
            "frozen_E1_LWC_native_global_reference_cluster"
        ),
        "mapping_fit_count": 1,
        "mapping_fit_after_score_seal": True,
        "single_global_mapping_reused_all_directions": True,
        "direction_specific_mapping_used": False,
        "sparse_labels_used_for_mapping": False,
    }


def apply_global_mapping(y_gen, mapping):
    """Apply one [K] mapping to all y_gen: [N,S] directions."""
    predictions = _as_numpy(y_gen, dtype=np.int64)
    permutation = _as_numpy(mapping, dtype=np.int64)
    _require(
        predictions.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and permutation.shape == (CLASS_NUM,)
        and int(predictions.min()) >= 0
        and int(predictions.max()) < CLASS_NUM,
        "global mapping application boundary mismatch",
    )
    return _readonly_array(permutation[predictions], dtype=np.int64)


def directional_roc_auc(correct, score):
    """Compute one ROC-AUC per directional split."""
    outcomes = _as_numpy(correct, dtype=np.int64)
    values = _as_numpy(score, dtype=np.float64)
    _require(
        outcomes.shape == values.shape == (
            SAMPLE_NUM,
            DIRECTION_COUNT,
        )
        and np.isfinite(values).all()
        and np.all((outcomes == 0) | (outcomes == 1)),
        "directional ROC-AUC input boundary mismatch",
    )
    auc_values = []
    invalid_directions = []
    for direction_id in range(DIRECTION_COUNT):
        if np.unique(outcomes[:, direction_id]).size != 2:
            auc_values.append(None)
            invalid_directions.append(direction_id)
        else:
            auc_values.append(
                float(
                    roc_auc_score(
                        outcomes[:, direction_id],
                        values[:, direction_id],
                    )
                )
            )
    return auc_values, invalid_directions


def _auc_summary(values):
    valid = np.asarray(
        [value for value in values if value is not None],
        dtype=np.float64,
    )
    _require(valid.size > 0, "no valid directional AUC values")
    return {
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
    }


def evaluate_directional_aucs(correct, score_arrays):
    """Evaluate Cycle/confidence/JSD/shuffle AUC on 20 directions."""
    _require(
        tuple(score_arrays) == SCORE_NAMES,
        "score array set/order mismatch",
    )
    auc_by_score = {}
    shared_invalid = None
    for score_name in SCORE_NAMES:
        values, invalid = directional_roc_auc(
            correct, score_arrays[score_name]
        )
        auc_by_score[score_name] = values
        if shared_invalid is None:
            shared_invalid = invalid
        else:
            _require(
                invalid == shared_invalid,
                "direction validity differs across score comparators",
            )
    valid_count = DIRECTION_COUNT - len(shared_invalid)
    return {
        "primary_evaluation_unit": (
            "20 directional complementary-view splits"
        ),
        "direction_count": DIRECTION_COUNT,
        "record_count_not_independent_experiments": (
            SAMPLE_NUM * DIRECTION_COUNT
        ),
        "valid_direction_count": valid_count,
        "invalid_direction_ids": shared_invalid,
        "AUC_cycle": auc_by_score["cycle"],
        "AUC_conf": auc_by_score["confidence"],
        "AUC_jsd": auc_by_score["jsd"],
        "AUC_shuffle": auc_by_score["shuffle"],
        "summary": {
            name: _auc_summary(auc_by_score[name])
            for name in SCORE_NAMES
        },
    }


def paired_wilcoxon_greater(first, second):
    """Run the fixed paired directional Wilcoxon greater test."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    _require(
        left.shape == right.shape
        and left.ndim == 1
        and left.size > 0
        and np.isfinite(left).all()
        and np.isfinite(right).all(),
        "paired Wilcoxon input boundary mismatch",
    )
    result = wilcoxon(left, right, alternative="greater")
    return float(result.pvalue)


def paired_auc_comparisons(auc_result):
    """Compare Cycle AUC against three fixed comparators by direction."""
    valid_ids = [
        direction_id
        for direction_id in range(DIRECTION_COUNT)
        if direction_id not in set(auc_result["invalid_direction_ids"])
    ]
    cycle = np.asarray(
        [auc_result["AUC_cycle"][index] for index in valid_ids],
        dtype=np.float64,
    )

    def compare(key):
        comparator = np.asarray(
            [auc_result[key][index] for index in valid_ids],
            dtype=np.float64,
        )
        return paired_wilcoxon_greater(cycle, comparator)

    return {
        "test": "scipy.stats.wilcoxon",
        "alternative": "greater",
        "paired_unit": "directional_split",
        "pair_count": len(valid_ids),
        "alpha": ALPHA,
        "p_cycle_vs_conf": compare("AUC_conf"),
        "p_cycle_vs_jsd": compare("AUC_jsd"),
        "p_cycle_vs_shuffle": compare("AUC_shuffle"),
    }


def score_separation(correct, U_cycle):
    """Compute pooled correct/wrong score separation as a diagnostic."""
    outcomes = _as_numpy(correct, dtype=np.bool_)
    values = _as_numpy(U_cycle, dtype=np.float64)
    _require(
        outcomes.shape == values.shape == (
            SAMPLE_NUM,
            DIRECTION_COUNT,
        )
        and np.any(outcomes)
        and np.any(~outcomes)
        and np.isfinite(values).all(),
        "score separation input boundary mismatch",
    )
    correct_values = values[outcomes]
    wrong_values = values[~outcomes]
    mean_correct = float(np.mean(correct_values))
    mean_wrong = float(np.mean(wrong_values))
    return {
        "pooled_records_secondary_diagnostic_only": True,
        "correct_count": int(correct_values.size),
        "wrong_count": int(wrong_values.size),
        "mean_U_correct": mean_correct,
        "mean_U_wrong": mean_wrong,
        "median_U_correct": float(np.median(correct_values)),
        "median_U_wrong": float(np.median(wrong_values)),
        "mean_U_correct_minus_wrong": mean_correct - mean_wrong,
    }


def risk_coverage_diagnostic(correct, score_arrays):
    """Compute fixed 20/40/60% precision by direction and macro mean."""
    outcomes = _as_numpy(correct, dtype=np.bool_)
    _require(
        outcomes.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and tuple(score_arrays) == ("cycle", "confidence", "jsd"),
        "risk-coverage input boundary mismatch",
    )
    result = {
        "coverages": list(RISK_COVERAGES),
        "primary_gate_used": False,
        "per_direction": {},
        "macro_mean": {},
    }
    for score_name in ("cycle", "confidence", "jsd"):
        values = _as_numpy(score_arrays[score_name], dtype=np.float64)
        _require(
            values.shape == outcomes.shape and np.isfinite(values).all(),
            "risk-coverage score boundary mismatch",
        )
        score_records = {}
        for coverage in RISK_COVERAGES:
            admitted_count = int(SAMPLE_NUM * coverage)
            precisions = []
            for direction_id in range(DIRECTION_COUNT):
                order = np.argsort(
                    -values[:, direction_id], kind="mergesort"
                )
                admitted = order[:admitted_count]
                precisions.append(
                    float(np.mean(outcomes[admitted, direction_id]))
                )
            coverage_key = str(int(coverage * 100))
            score_records[coverage_key] = {
                "coverage": coverage,
                "admitted_count_per_direction": admitted_count,
                "precision_by_direction": precisions,
                "macro_mean": float(np.mean(precisions)),
            }
            result["macro_mean"][
                "precision_" + score_name + "@" + coverage_key
            ] = float(np.mean(precisions))
        result["per_direction"][score_name] = score_records
    return result


def build_c0_decision(auc_result, paired_result, separation):
    """Apply the four fixed C0 gates and exhaustive decision tree."""
    summaries = auc_result["summary"]
    cycle_mean = summaries["cycle"]["mean"]
    signal_pass = bool(
        auc_result["valid_direction_count"] == DIRECTION_COUNT
        and cycle_mean > 0.5
        and summaries["cycle"]["median"] > 0.5
        and separation["mean_U_correct"]
        > separation["mean_U_wrong"]
    )
    specificity_pass = bool(
        cycle_mean > summaries["shuffle"]["mean"]
        and paired_result["p_cycle_vs_shuffle"] < ALPHA
    )
    beats_confidence_pass = bool(
        cycle_mean > summaries["confidence"]["mean"]
        and paired_result["p_cycle_vs_conf"] < ALPHA
    )
    beats_jsd_pass = bool(
        cycle_mean > summaries["jsd"]["mean"]
        and paired_result["p_cycle_vs_jsd"] < ALPHA
    )

    decision_pass = bool(
        signal_pass
        and specificity_pass
        and beats_confidence_pass
        and beats_jsd_pass
    )
    fail_A = bool(not signal_pass)
    fail_B = bool(signal_pass and not specificity_pass)
    fail_C = bool(
        signal_pass
        and specificity_pass
        and not (beats_confidence_pass and beats_jsd_pass)
    )
    conditions = {
        "PASS": decision_pass,
        "FAIL_A": fail_A,
        "FAIL_B": fail_B,
        "FAIL_C": fail_C,
    }
    _require(
        sum(conditions.values()) == 1,
        "C0_DECISION_TREE_LOGICALLY_IMPOSSIBLE_STATE",
    )
    if decision_pass:
        final_decision = "C0_COMPLEMENTARY_CYCLE_VALIDITY_PASS"
        explanation = (
            "Independent complementary-view semantic verification "
            "provides pseudo-label validity information beyond ordinary "
            "confidence, historical JSD consistency, and broken sample "
            "correspondence."
        )
    elif fail_A:
        final_decision = "C0_CYCLIC_SEMANTIC_SIGNAL_FAIL"
        explanation = (
            "Cycle verification cannot stably identify correct versus "
            "wrong pseudo hypotheses."
        )
    elif fail_B:
        final_decision = "C0_SAMPLE_CORRESPONDENCE_NOT_SPECIFIC"
        explanation = (
            "The score is not specific to correct sample-level "
            "cross-view correspondence."
        )
    else:
        final_decision = (
            "C0_CYCLE_REDUNDANT_WITH_EXISTING_CONFIDENCE"
        )
        explanation = (
            "The operator has predictive value but does not establish "
            "new information beyond confidence and distributional "
            "consistency."
        )
    return {
        "alpha": ALPHA,
        "C0_SIGNAL_PASS": signal_pass,
        "C0_CORRESPONDENCE_SPECIFICITY_PASS": specificity_pass,
        "C0_BEATS_CONFIDENCE_PASS": beats_confidence_pass,
        "C0_BEATS_JSD_PASS": beats_jsd_pass,
        "decision_conditions": conditions,
        "final_decision": final_decision,
        "explanation": explanation,
        "C1_pseudo_supervision_training_allowed": decision_pass,
        "training_entered": False,
    }


def evaluate_postseal(
    native_global_cluster,
    ground_truth,
    predictions_and_scores,
):
    """Run all GT-dependent C0 diagnostics after the score seal."""
    mapping_record = fit_global_cluster_mapping_once(
        native_global_cluster, ground_truth
    )
    y_gen_semantic = apply_global_mapping(
        predictions_and_scores["y_gen"],
        mapping_record["mapping"],
    )
    labels = _as_numpy(ground_truth, dtype=np.int64)
    correct = y_gen_semantic == labels[:, None]
    score_arrays = {
        "cycle": predictions_and_scores["U_cycle"],
        "confidence": predictions_and_scores["C_conf"],
        "jsd": predictions_and_scores["C_jsd"],
        "shuffle": predictions_and_scores["U_cycle_shuffle"],
    }
    auc_result = evaluate_directional_aucs(correct, score_arrays)
    paired_result = paired_auc_comparisons(auc_result)
    separation = score_separation(
        correct, predictions_and_scores["U_cycle"]
    )
    risk_coverage = risk_coverage_diagnostic(
        correct,
        {
            "cycle": predictions_and_scores["U_cycle"],
            "confidence": predictions_and_scores["C_conf"],
            "jsd": predictions_and_scores["C_jsd"],
        },
    )
    decision = build_c0_decision(
        auc_result, paired_result, separation
    )
    return {
        "mapping": mapping_record,
        "y_gen_semantic": y_gen_semantic,
        "correct": _readonly_array(correct, dtype=np.bool_),
        "auc": auc_result,
        "paired_comparisons": paired_result,
        "score_separation": separation,
        "risk_coverage": risk_coverage,
        "decision": decision,
    }

