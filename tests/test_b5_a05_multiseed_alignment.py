"""Tests for the pure B5-A0.5 multi-seed JSON summary."""

import inspect
import json

import experiments.b5_semantic_rate.summarize_b5_a05_multiseed_alignment as a05


def _row(
    seed,
    source_auc=0.56,
    source_rho=0.06,
    source_pass=True,
    pair_auc=0.49,
    pair_rho=-0.02,
    pair_pass=False,
    specificity_pass=True,
    specificity_delta=0.001,
):
    return {
        "seed": seed,
        "global_specificity_p": 0.01,
        "global_specificity_delta": specificity_delta,
        "global_specificity_pass": specificity_pass,
        "AUC_FE_source": source_auc,
        "AUC_FE_source_CI": [source_auc - 0.02, source_auc + 0.02],
        "Spearman_FE_source": source_rho,
        "Spearman_FE_source_CI": [source_rho - 0.02, source_rho + 0.02],
        "SOURCE_U_SIGNAL_PASS": source_pass,
        "AUC_FE_pair": pair_auc,
        "Spearman_FE_pair": pair_rho,
        "PAIRWISE_T_SIGNAL_PASS": pair_pass,
        "Delta_AUC_pair_minus_source": pair_auc - source_auc,
        "Delta_Spearman_pair_minus_source": pair_rho - source_rho,
    }


def _alignment_json(row):
    return {
        "stage": "B5-A0.4",
        "condition": "snr2p5_k2",
        "model_seed": row["seed"],
        "training_performed": False,
        "B5_A04_ALIGNMENT_AUDIT_COMPLETE": True,
        "B5_A04_SOURCE_U_SIGNAL_PASS": row["SOURCE_U_SIGNAL_PASS"],
        "B5_A04_PAIRWISE_T_SIGNAL_PASS": row["PAIRWISE_T_SIGNAL_PASS"],
        "global_specificity": {
            "permutation_p_value": row["global_specificity_p"],
            "delta_mean": row["global_specificity_delta"],
            "specificity_pass": row["global_specificity_pass"],
        },
        "fe_adjusted_metrics": {
            "source": {
                "auc": row["AUC_FE_source"],
                "spearman": row["Spearman_FE_source"],
            },
            "pair": {
                "auc": row["AUC_FE_pair"],
                "spearman": row["Spearman_FE_pair"],
            },
        },
        "bootstrap_cis": {
            "auc_fe_source": {
                "lower": row["AUC_FE_source_CI"][0],
                "upper": row["AUC_FE_source_CI"][1],
            },
            "spearman_fe_source": {
                "lower": row["Spearman_FE_source_CI"][0],
                "upper": row["Spearman_FE_source_CI"][1],
            },
        },
        "paired_differences": {
            "delta_auc": row["Delta_AUC_pair_minus_source"],
            "delta_spearman": row["Delta_Spearman_pair_minus_source"],
        },
    }


def test_two_of_three_gate_logic_and_all_three_point_conditions_pass():
    rows = [
        _row(20),
        _row(30),
        _row(50, source_pass=False, specificity_pass=False),
    ]
    gates = a05.compute_replication_gates(rows)
    assert gates["source_u_signal_pass_count"] == 2
    assert gates["global_specificity_pass_count"] == 2
    assert gates["pair_t_not_preferred_count"] == 3
    assert gates["B5_A05_SOURCE_U_REPLICATION_PASS"]
    assert gates["B5_A05_PAIRWISE_T_NOT_PREFERRED"]
    assert gates["B5_A05_CONDITIONAL_RATE_REPLICATION_PASS"]
    assert gates["B5_A1_ENTRY_ELIGIBLE"]


def test_all_three_source_point_estimate_conditions_are_mandatory():
    rows = [_row(20), _row(30), _row(50, source_auc=0.49)]
    gates = a05.compute_replication_gates(rows)
    assert gates["source_u_signal_pass_count"] == 3
    assert not gates["source_u_all_auc_point_estimates_gt_0p5"]
    assert not gates["B5_A05_SOURCE_U_REPLICATION_PASS"]

    rows = [_row(20), _row(30), _row(50, source_rho=-0.001)]
    gates = a05.compute_replication_gates(rows)
    assert not gates["source_u_all_spearman_point_estimates_gt_0"]
    assert not gates["B5_A05_SOURCE_U_REPLICATION_PASS"]


def test_entry_eligibility_is_exact_conjunction_of_three_gates():
    rows = [
        _row(20),
        _row(30, specificity_pass=False),
        _row(50, specificity_pass=False),
    ]
    gates = a05.compute_replication_gates(rows)
    assert gates["B5_A05_SOURCE_U_REPLICATION_PASS"]
    assert gates["B5_A05_PAIRWISE_T_NOT_PREFERRED"]
    assert not gates["B5_A05_CONDITIONAL_RATE_REPLICATION_PASS"]
    assert not gates["B5_A1_ENTRY_ELIGIBLE"]


def test_conditional_rate_requires_all_three_positive_deltas():
    rows = [_row(20), _row(30), _row(50, specificity_delta=-1e-6)]
    gates = a05.compute_replication_gates(rows)
    assert gates["global_specificity_pass_count"] == 3
    assert not gates["global_specificity_all_delta_mean_gt_0"]
    assert not gates["B5_A05_CONDITIONAL_RATE_REPLICATION_PASS"]


def test_pair_t_not_preferred_uses_two_of_three_pairwise_comparisons():
    rows = [
        _row(20),
        _row(30),
        _row(50, pair_auc=0.60, pair_rho=0.10),
    ]
    gates = a05.compute_replication_gates(rows)
    assert gates["pair_t_not_preferred_count"] == 2
    assert gates["B5_A05_PAIRWISE_T_NOT_PREFERRED"]


def test_summary_reads_exactly_three_json_inputs_without_recomputation(tmp_path):
    paths = {}
    for seed in (20, 30, 50):
        path = tmp_path / ("seed" + str(seed) + ".json")
        path.write_text(json.dumps(_alignment_json(_row(seed))))
        paths[seed] = path
    result = a05.summarize_alignment_jsons(paths)
    assert len(result["input_jsons"]) == 3
    assert [row["seed"] for row in result["rows"]] == [20, 30, 50]
    assert result["summary_reads_json_only"]
    assert not result["training_performed"]
    assert not result["recomputation_performed"]
    assert result["aggregate_statistics"]["seed_count"] == 3
    assert result["B5_A1_STARTED"] is False


def test_summary_has_no_training_or_sensitive_data_api():
    functions = (
        a05.load_alignment_json,
        a05.alignment_summary_row,
        a05.compute_replication_gates,
        a05.aggregate_statistics,
        a05.summarize_alignment_jsons,
    )
    for function in functions:
        names = [name.lower() for name in inspect.signature(function).parameters]
        assert all("label" not in name for name in names)
        assert all("mask" not in name for name in names)
    source = inspect.getsource(a05)
    assert "torch" not in source
    assert ".backward(" not in source
    assert ".fit(" not in source
    assert "np.load(" not in source
