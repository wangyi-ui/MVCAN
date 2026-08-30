"""Evaluate E4-CF0 counterfactual marginal memory-action utility."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.e4_semantic_memory_bank import (
    evaluate_e4_a0_memory_feasibility as e4a0,
)
from experiments.e4_semantic_memory_bank.e4a1_memory_specific_utility import (
    build_memory_information_utility,
    build_memory_specific_semantic_usefulness,
)
from experiments.e4_semantic_memory_bank.e4cf0_counterfactual_utility import (
    PROXY_NAMES,
    VIEW_NUM,
    build_counterfactual_marginal_utility,
    build_e4cf0_decision,
    compare_proxies_to_delta,
    summarize_delta,
)
from experiments.e4_semantic_memory_bank.semantic_memory_bank import (
    LABELED_NUM,
    SAMPLE_NUM,
)
from irv.b4_information_utility import tensor_sha256


STAGE = "E4-CF0"
SEED = 20
E4A1_PROXY_REPLAY_MISMATCH = "E4A1_PROXY_REPLAY_MISMATCH"
EXPECTED_PROXY_SHA256 = {
    "R": "dcece5096f0782a520fefc76facf4fe7f462511f2cbdf2061cb15252269c7d05",
    "S": "fde3a56a4e292b699975a625b1ef3b9ed99235ea14fb846c47765aab46765152",
    "U_RS": "2b4c51a6fd92c783cf6cd00d802ebbcc607af1bf24dc804de528706057dfb19e",
}
FROZEN_COMPONENT_KEYS = {
    "R": "R_labeled",
    "S": "S_mem",
    "U_RS": "U_mem",
}

DEFAULT_E1_LWC_MODEL_DIR = e4a0.DEFAULT_E1_LWC_MODEL_DIR
DEFAULT_LABEL_SPLIT_DIR = e4a0.DEFAULT_LABEL_SPLIT_DIR
DEFAULT_E4A1_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "outputs/e4_semantic_memory_bank"
    / "e4a1_memory_specific_utility_seed20"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "outputs/e4_semantic_memory_bank"
    / "e4cf0_counterfactual_utility_seed20"
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


def _as_numpy(value, dtype=None):
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    return np.ascontiguousarray(array)


def save_counterfactual_seal(
    counterfactual,
    labeled_ids,
    output_root,
):
    """Persist/reload/hash Delta and its writer mapping before post-hoc data."""
    root = _resolve(output_root)
    artifact_path = root / "counterfactual_utility.npz"
    seal_path = root / "counterfactual_seal.json"
    _require(
        root.is_dir()
        and not artifact_path.exists()
        and not seal_path.exists(),
        "counterfactual seal output boundary mismatch",
    )
    ids = np.asarray(labeled_ids, dtype=np.int64)
    _require(ids.shape == (LABELED_NUM,), "labeled_ids must have shape [14]")
    peer_rows = _as_numpy(
        counterfactual["heldout_peer_rows"], dtype=np.int64
    )
    writer_sample_ids = np.repeat(ids, VIEW_NUM)
    writer_views = np.tile(
        np.arange(VIEW_NUM, dtype=np.int64), LABELED_NUM
    )
    heldout_peer_ids = np.repeat(ids[peer_rows], VIEW_NUM)
    payload = {
        "Delta_mem": _as_numpy(counterfactual["Delta_mem"]),
        "Q_plus": _as_numpy(counterfactual["Q_plus"]),
        "Q_minus": _as_numpy(counterfactual["Q_minus"]),
        "writer_sample_ids": np.ascontiguousarray(
            writer_sample_ids, dtype=np.int64
        ),
        "writer_views": np.ascontiguousarray(
            writer_views, dtype=np.int64
        ),
        "heldout_peer_ids": np.ascontiguousarray(
            heldout_peer_ids, dtype=np.int64
        ),
    }
    np.savez(artifact_path, **payload)
    logical_hashes = {}
    with np.load(artifact_path, allow_pickle=False) as archive:
        _require(
            tuple(archive.files) == tuple(payload),
            "counterfactual artifact key/order mismatch",
        )
        for key, expected in payload.items():
            reloaded = np.ascontiguousarray(archive[key])
            _require(
                np.array_equal(reloaded, expected),
                "counterfactual artifact reload mismatch",
            )
            logical_hashes[key] = tensor_sha256(reloaded)
    seal = {
        "stage": STAGE,
        "counterfactual_file": _display(artifact_path),
        "counterfactual_file_sha256": e4a0.file_sha256(
            artifact_path
        ),
        "logical_sha256": logical_hashes,
        "Delta_mem_logical_sha256": logical_hashes["Delta_mem"],
        "Delta_completed_before_posthoc_data": True,
        "Delta_saved_before_posthoc_data": True,
        "Delta_reloaded_before_posthoc_data": True,
        "Delta_hashed_before_posthoc_data": True,
        "counterfactual_seal_written_before_posthoc_data": True,
        "full_GT_loaded_before_seal": False,
        "corruption_mask_loaded_before_seal": False,
    }
    _write_json(seal_path, seal)
    return seal


def verify_e4a1_proxy_replay(
    computed_proxy_scores,
    frozen_output_dir=DEFAULT_E4A1_OUTPUT_DIR,
):
    """Require exact arrays and logical hashes for frozen E4-A1 R/S/U_RS."""
    if tuple(computed_proxy_scores) != PROXY_NAMES:
        raise RuntimeError(E4A1_PROXY_REPLAY_MISMATCH)
    root = _resolve(frozen_output_dir)
    component_path = root / "utility_components.npz"
    audit_path = root / "e4a1_audit.json"
    try:
        _require(
            component_path.is_file() and audit_path.is_file(),
            E4A1_PROXY_REPLAY_MISMATCH,
        )
        with open(audit_path, "r", encoding="utf-8") as input_file:
            frozen_audit = json.load(input_file)
        _require(
            frozen_audit.get("E4A1_AUDIT_PASS") is True
            and frozen_audit.get("gate", {}).get("final_decision")
            == "E4A1_SPECIFIC_BUT_NO_NET_MEMORY_GAIN",
            E4A1_PROXY_REPLAY_MISMATCH,
        )
        records = {}
        frozen_scores = {}
        with np.load(component_path, allow_pickle=False) as archive:
            for proxy_name in PROXY_NAMES:
                key = FROZEN_COMPONENT_KEYS[proxy_name]
                current = _as_numpy(computed_proxy_scores[proxy_name])
                frozen = np.ascontiguousarray(archive[key])
                current_sha = tensor_sha256(current)
                frozen_sha = tensor_sha256(frozen)
                exact = bool(
                    np.array_equal(current, frozen)
                    and current_sha
                    == frozen_sha
                    == EXPECTED_PROXY_SHA256[proxy_name]
                )
                _require(exact, E4A1_PROXY_REPLAY_MISMATCH)
                frozen_scores[proxy_name] = torch.from_numpy(
                    np.array(frozen, copy=True, order="C")
                ).detach()
                records[proxy_name] = {
                    "frozen_component_key": key,
                    "shape": list(frozen.shape),
                    "dtype": str(frozen.dtype),
                    "current_logical_sha256": current_sha,
                    "frozen_logical_sha256": frozen_sha,
                    "exact_array_replay_pass": exact,
                    "exact_hash_replay_pass": exact,
                }
        return frozen_scores, {
            "frozen_output_dir": _display(root),
            "components": records,
            "R_exact_replay_pass": records["R"][
                "exact_array_replay_pass"
            ],
            "S_exact_replay_pass": records["S"][
                "exact_array_replay_pass"
            ],
            "U_RS_exact_replay_pass": records["U_RS"][
                "exact_array_replay_pass"
            ],
            "E4A1_proxy_exact_replay_pass": True,
        }
    except (KeyError, OSError, ValueError, RuntimeError):
        raise RuntimeError(E4A1_PROXY_REPLAY_MISMATCH) from None


def recompute_and_replay_proxies(
    h_labeled,
    labels_labeled,
    labeled_ids,
    frozen_output_dir=DEFAULT_E4A1_OUTPUT_DIR,
):
    """Recompute proxies after Delta seal, then lock them to frozen E4-A1."""
    R_full, R_audit = e4a0.load_frozen_reliability(
        e4a0.DEFAULT_R_PATH
    )
    ids = np.asarray(labeled_ids, dtype=np.int64)
    _require(ids.shape == (LABELED_NUM,), "labeled_ids must have shape [14]")
    R_labeled = torch.from_numpy(
        np.ascontiguousarray(R_full[ids])
    ).detach()
    S_components = build_memory_specific_semantic_usefulness(
        h_labeled, labels_labeled
    )
    S_mem = S_components["S_mem"].detach()
    U_RS = build_memory_information_utility(
        R_labeled, S_mem
    ).detach()
    computed = {
        "R": R_labeled,
        "S": S_mem,
        "U_RS": U_RS,
    }
    frozen_scores, replay_audit = verify_e4a1_proxy_replay(
        computed, frozen_output_dir
    )
    return frozen_scores, {
        "frozen_reliability": R_audit,
        "S_recomputation": S_components["audit"],
        "proxy_replay": replay_audit,
        "proxy_loaded_after_Delta_seal": True,
        "proxy_used_for_Delta": False,
    }


def corruption_posthoc_analysis(
    Delta_mem,
    labeled_ids,
):
    """Load clean/corrupt truth only after sealing Delta, for non-Gate analysis."""
    clean_weights, corruption_audit = (
        e4a0.load_oracle_clean_labeled_weights(
            labeled_ids, e4a0.DEFAULT_ORACLE_MASK_PATH
        )
    )
    clean_mask = np.asarray(clean_weights, dtype=bool)
    delta = _as_numpy(Delta_mem, dtype=np.float64)
    _require(
        clean_mask.shape == delta.shape == (LABELED_NUM, VIEW_NUM),
        "post-hoc corruption/Delta shape mismatch",
    )
    _require(
        clean_mask.any() and (~clean_mask).any(),
        "post-hoc clean/corrupt partition is empty",
    )
    return {
        "mean_Delta_clean_writers": float(delta[clean_mask].mean()),
        "mean_Delta_corrupt_writers": float(delta[~clean_mask].mean()),
        "clean_writer_count": int(clean_mask.sum()),
        "corrupt_writer_count": int((~clean_mask).sum()),
        "corruption_mask_loaded_after_Delta_seal": True,
        "corruption_used_for_Delta": False,
        "corruption_used_for_gate": False,
        "source": corruption_audit,
    }


def run_evaluation(
    feature_path=e4a0.DEFAULT_FEATURE_PATH,
    feature_audit_path=e4a0.DEFAULT_FEATURE_AUDIT_PATH,
    e1_lwc_model_dir=DEFAULT_E1_LWC_MODEL_DIR,
    label_split_dir=DEFAULT_LABEL_SPLIT_DIR,
    frozen_e4a1_output_dir=DEFAULT_E4A1_OUTPUT_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    device="cpu",
):
    """Run the read-only CF0 diagnostic without fitting or full-GT access."""
    output_root = _resolve(output_dir)
    _require(
        not output_root.exists(),
        "refusing to overwrite an E4-CF0 output directory",
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
    _require(
        np.array_equal(sample_ids, np.arange(SAMPLE_NUM, dtype=np.int64)),
        "CF0 representation/sample ID alignment mismatch",
    )
    labeled_index = torch.as_tensor(labeled_ids, dtype=torch.long)
    h_labeled = h_sem[labeled_index].detach()

    counterfactual = build_counterfactual_marginal_utility(
        h_labeled, labels_labeled
    )
    output_root.mkdir(parents=True)
    counterfactual_seal = save_counterfactual_seal(
        counterfactual,
        labeled_ids,
        output_root,
    )

    proxy_scores, proxy_replay_audit = (
        recompute_and_replay_proxies(
            h_labeled,
            labels_labeled,
            labeled_ids,
            frozen_e4a1_output_dir,
        )
    )
    delta_summary = summarize_delta(counterfactual["Delta_mem"])
    proxy_comparison = compare_proxies_to_delta(
        proxy_scores,
        counterfactual["Delta_mem"],
        labels_labeled,
    )
    decision = build_e4cf0_decision(
        delta_summary, proxy_comparison
    )

    posthoc_corruption = corruption_posthoc_analysis(
        counterfactual["Delta_mem"], labeled_ids
    )
    serializable_proxy_comparison = {
        key: value
        for key, value in proxy_comparison.items()
        if key != "shuffled_scores"
    }
    _write_json(
        output_root / "proxy_comparison.json",
        serializable_proxy_comparison,
    )

    audit = {
        "stage": STAGE,
        "seed": SEED,
        "N": SAMPLE_NUM,
        "V": VIEW_NUM,
        "K": 7,
        "labeled_count": LABELED_NUM,
        "writer_count": LABELED_NUM * VIEW_NUM,
        "input_provenance": {
            "E1_LWC_frozen_representation": representation_audit,
            "sparse_labels": sparse_label_audit,
            "frozen_E4A1_proxy_source": _display(
                frozen_e4a1_output_dir
            ),
        },
        "tensor_shape_audit": {
            "h_sem": list(h_sem.shape),
            "h_labeled": list(h_labeled.shape),
            "Delta_mem": list(counterfactual["Delta_mem"].shape),
            "Q_plus": list(counterfactual["Q_plus"].shape),
            "Q_minus": list(counterfactual["Q_minus"].shape),
            "writer_sample_ids": [LABELED_NUM * VIEW_NUM],
            "writer_views": [LABELED_NUM * VIEW_NUM],
            "heldout_peer_ids": [LABELED_NUM * VIEW_NUM],
        },
        "counterfactual_isolation": counterfactual["audit"],
        "counterfactual_seal": counterfactual_seal,
        "E4A1_proxy_replay": proxy_replay_audit,
        "delta_summary": delta_summary,
        "matched_pair_analysis": {
            "matched_pair_count": 42,
            "R_valid_pair_count": proxy_comparison["comparisons"][
                "R"
            ]["valid_pair_count"],
            "S_valid_pair_count": proxy_comparison["comparisons"][
                "S"
            ]["valid_pair_count"],
            "U_RS_valid_pair_count": proxy_comparison["comparisons"][
                "U_RS"
            ]["valid_pair_count"],
            "matched_shuffle_deterministic_pass": proxy_comparison[
                "matched_shuffles"
            ]["deterministic_no_RNG_pass"],
        },
        "posthoc_corruption_analysis": posthoc_corruption,
        "leakage_seal": {
            "unlabeled_GT_used_for_delta": False,
            "full_GT_loaded": False,
            "oracle_used_for_delta": False,
            "corruption_mask_used_for_delta": False,
            "R_used_for_delta": False,
            "S_used_for_delta": False,
            "U_used_for_delta": False,
            "Delta_saved_and_hashed_before_corruption_load": True,
            "training_used": False,
            "optimizer_used": False,
            "backward_used": False,
        },
        "execution_guards": {
            "training_used": False,
            "optimizer_used": False,
            "backward_used": False,
            "online_memory_update_used": False,
            "deterministic_pass": True,
        },
        "decision": decision,
        "E4CF0_AUDIT_PASS": bool(
            counterfactual["audit"][
                "heldout_sample_excluded_from_true_class_memory"
            ]
            and counterfactual["audit"][
                "removed_view_is_exactly_writer_view_pass"
            ]
            and counterfactual["audit"][
                "negative_prototypes_shared_plus_minus_pass"
            ]
            and proxy_replay_audit["proxy_replay"][
                "E4A1_proxy_exact_replay_pass"
            ]
            and proxy_comparison["matched_shuffles"][
                "deterministic_no_RNG_pass"
            ]
            and posthoc_corruption[
                "corruption_mask_loaded_after_Delta_seal"
            ]
        ),
    }
    diagnostic_results = {
        "stage": STAGE,
        "delta_summary": delta_summary,
        "proxy_primary_identifiability": {
            "R_pairwise_rank_accuracy": proxy_comparison[
                "R_pairwise_rank_accuracy"
            ],
            "S_pairwise_rank_accuracy": proxy_comparison[
                "S_pairwise_rank_accuracy"
            ],
            "U_pairwise_rank_accuracy": proxy_comparison[
                "U_pairwise_rank_accuracy"
            ],
            "matched_shuffled_U_pairwise_rank_accuracy": (
                proxy_comparison[
                    "matched_shuffled_U_pairwise_rank_accuracy"
                ]
            ),
        },
        "posthoc_corruption_analysis": posthoc_corruption,
        "decision": decision,
    }
    _write_json(output_root / "e4cf0_audit.json", audit)
    _write_json(
        output_root / "diagnostic_results.json",
        diagnostic_results,
    )
    return {
        "output_dir": output_root,
        "e4cf0_audit": audit,
        "proxy_comparison": serializable_proxy_comparison,
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
        "--frozen-e4a1-output-dir",
        default=str(DEFAULT_E4A1_OUTPUT_DIR),
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
        frozen_e4a1_output_dir=args.frozen_e4a1_output_dir,
        output_dir=args.output_dir,
        device=args.device,
    )
    print("Saved: " + _display(result["output_dir"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
