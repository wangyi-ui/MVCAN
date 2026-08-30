"""Run the frozen, read-only C1-P0 utility-to-action compression audit."""

import argparse
import gc
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import c1_frozen_pseudo_supervision as c1
from experiments.cyclic_utility import c1_p0_action_compression_audit as p0
from experiments.cyclic_utility import train_c1_frozen_pseudo_supervision as c1_train
from irv.b3_audit import hash_backbone


DEFAULT_C0_DIR = (
    REPOSITORY_ROOT
    / "outputs/cyclic_utility/c0_complementary_semantic_verification_seed20"
)
DEFAULT_C0_ARTIFACT_PATH = DEFAULT_C0_DIR / "c0_predictions_and_scores.npz"
DEFAULT_C0_SEAL_PATH = DEFAULT_C0_DIR / "c0_prediction_seal.json"
DEFAULT_C1A1_OUTPUT_DIR = (
    REPOSITORY_ROOT / "outputs/cyclic_utility/c1a1_frozen_action_seed20"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "outputs/cyclic_utility/c1p0_action_compression_seed20"
)
DEFAULT_FEATURE_PATH = c1_train.DEFAULT_FEATURE_PATH
DEFAULT_FEATURE_AUDIT_PATH = c1_train.DEFAULT_FEATURE_AUDIT_PATH
DEFAULT_E1_MODEL_DIR = c1_train.DEFAULT_E1_LWC_MODEL_DIR
DEFAULT_E1_AUDIT_PATH = c1_train.DEFAULT_E1_LWC_AUDIT_PATH

OUTPUT_FILES = (
    "c1p0_directional_structure.json",
    "c1p0_weight_comparison.json",
    "c1p0_target_comparison.json",
    "c1p0_strength_comparison.json",
    "c1p0_effective_action_comparison.json",
    "c1p0_pseudo_loss.json",
    "c1p0_gradient_comparison.json",
    "c1p0_native_gradient_scale.json",
    "c1p0_audit.json",
    "diagnostic_results.json",
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _display(path):
    value = Path(path).resolve()
    try:
        return str(value.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(value)


def _release_device_cache(device):
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def _load_initial_model(device):
    c1.set_deterministic_seed(c1.SEED)
    model, config, checkpoint = c1_train.load_trainable_e1_lwc_model(
        DEFAULT_E1_MODEL_DIR,
        DEFAULT_E1_AUDIT_PATH,
        torch.device(device),
    )
    initial_hash = hash_backbone(model.autoencoders)
    _require(
        initial_hash["aggregate"]
        == c1.EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256,
        "frozen E1 initialization aggregate mismatch",
    )
    _require(
        int(config["training"]["batch_size"]) == c1.BATCH_SIZE,
        "frozen C1 batch size mismatch",
    )
    return model, config, checkpoint


def _frozen_file_hashes():
    c1_files = [
        DEFAULT_C1A1_OUTPUT_DIR / "c1_summary.json",
        DEFAULT_C1A1_OUTPUT_DIR / "c1_decision.json",
        DEFAULT_C1A1_OUTPUT_DIR / "c1_audit.json",
    ]
    for arm in p0.PSEUDO_ARMS:
        c1_files.append(
            DEFAULT_C1A1_OUTPUT_DIR / arm / "pseudo_target_audit.json"
        )
    for path in c1_files:
        _require(path.is_file(), "frozen C1-A1 file is missing: " + str(path))
    return {_display(path): c1.file_sha256(path) for path in c1_files}


def run_audit(device="cuda:0", output_dir=DEFAULT_OUTPUT_DIR):
    """Execute diagnostics and write results only after every check succeeds."""
    output_path = Path(output_dir)
    _require(
        output_path.resolve() == DEFAULT_OUTPUT_DIR.resolve(),
        "formal C1-P0 output path is frozen",
    )
    _require(not output_path.exists(), "refusing to overwrite C1-P0 output")
    target_device = torch.device(device)
    if target_device.type == "cuda":
        _require(torch.cuda.is_available(), "requested CUDA device is unavailable")
        torch.cuda.set_device(target_device)

    c1.set_deterministic_seed(c1.SEED)
    c0_arrays, c0_provenance = c1.load_frozen_c0_artifact(
        DEFAULT_C0_ARTIFACT_PATH, DEFAULT_C0_SEAL_PATH
    )
    views, sample_ids, feature_provenance = (
        c1_train.e1_train.load_frozen_feature_artifact(
            DEFAULT_FEATURE_PATH, DEFAULT_FEATURE_AUDIT_PATH
        )
    )
    _require(
        np.array_equal(sample_ids, np.arange(c1.SAMPLE_NUM, dtype=np.int64)),
        "feature sample IDs are not canonical",
    )

    directional_record = p0.directional_hypothesis_structure(c0_arrays["y_gen"])
    weights, weight_record = p0.build_weight_diagnostics(c0_arrays)

    coordinate_model, _, coordinate_checkpoint = _load_initial_model(target_device)
    coordinate_mode_audit = p0.require_c1_model_mode_safe(coordinate_model)
    coordinate_hash_before = hash_backbone(coordinate_model.autoencoders)
    M0, M0_audit = c1.derive_frozen_M0(
        coordinate_model,
        views,
        c0_arrays["native_global_cluster"],
        target_device,
    )
    coordinate_hash_after = hash_backbone(coordinate_model.autoencoders)
    _require(
        coordinate_hash_before == coordinate_hash_after,
        "M0 reconstruction mutated the frozen model",
    )
    del coordinate_model
    _release_device_cache(target_device)

    targets, target_provenance = p0.reconstruct_frozen_targets(
        c0_arrays, M0, DEFAULT_C1A1_OUTPUT_DIR
    )
    _, target_record, strength_record, action_record = (
        p0.build_target_action_diagnostics(
            targets, target_provenance, M0_audit
        )
    )

    batches, sample_order_audit = p0.c1_epoch1_batches(sample_ids)
    public_sample_order = {
        key: value
        for key, value in sample_order_audit.items()
        if key != "sample_order"
    }

    pseudo_runtime = OrderedDict()
    cycle_model, _, cycle_checkpoint = _load_initial_model(target_device)
    cycle_gradient, cycle_runtime = p0.pseudo_loss_and_gradient(
        cycle_model, views, batches, targets["CYCLE"], target_device
    )
    pseudo_runtime["CYCLE"] = cycle_runtime
    parameter_names = cycle_runtime["parameter_names"]
    parameter_view_ids = cycle_runtime["parameter_view_ids"]
    del cycle_model
    _release_device_cache(target_device)

    native_model, _, native_checkpoint = _load_initial_model(target_device)
    native_gradient, native_runtime = p0.native_loss_and_gradient(
        native_model, views, batches, target_device
    )
    _require(
        native_runtime["parameter_names"] == parameter_names
        and native_runtime["parameter_view_ids"] == parameter_view_ids,
        "native/pseudo parameter order mismatch",
    )
    del native_model
    _release_device_cache(target_device)

    scale_comparisons = OrderedDict()
    checkpoint_records = {
        "M0": coordinate_checkpoint,
        "CYCLE": cycle_checkpoint,
        "NATIVE": native_checkpoint,
    }
    for arm in p0.COMPARATORS:
        arm_model, _, arm_checkpoint = _load_initial_model(target_device)
        arm_gradient, arm_runtime = p0.pseudo_loss_and_gradient(
            arm_model, views, batches, targets[arm], target_device
        )
        _require(
            arm_runtime["parameter_names"] == parameter_names
            and arm_runtime["parameter_view_ids"] == parameter_view_ids,
            "cross-arm parameter order mismatch: " + arm,
        )
        pseudo_runtime[arm] = arm_runtime
        checkpoint_records[arm] = arm_checkpoint
        scale_comparisons["CYCLE_vs_" + arm] = p0.gradient_scale_metrics(
            native_gradient,
            cycle_gradient,
            arm_gradient,
            parameter_view_ids,
        )
        del arm_model
        _release_device_cache(target_device)
        del arm_gradient

    pseudo_runtime = OrderedDict(
        (arm, pseudo_runtime[arm]) for arm in p0.PSEUDO_ARMS
    )
    pseudo_loss_values = {
        arm: pseudo_runtime[arm]["full_dataset_mean_pseudo_loss"]
        for arm in p0.PSEUDO_ARMS
    }
    pseudo_loss_record = {
        "stage": p0.STAGE,
        "layer": 4,
        "same_frozen_E1_checkpoint_all_arms": True,
        "same_deterministic_sample_order_all_arms": True,
        "sample_order_sha256": public_sample_order["sample_order_sha256"],
        "full_dataset_mean_pseudo_loss": pseudo_loss_values,
        "pairwise_absolute_loss_difference": {
            "CYCLE_vs_" + arm: abs(
                pseudo_loss_values["CYCLE"] - pseudo_loss_values[arm]
            )
            for arm in p0.COMPARATORS
        },
        "soft_CE_source": "C1 frozen helper soft_pseudo_cross_entropy",
        "model_parameter_update_used": False,
    }

    gradient_record = {
        "stage": p0.STAGE,
        "layer": 5,
        "autograd_grad_only": True,
        "sample_order": public_sample_order,
        "arm_gradient": {
            arm: {
                "norm": pseudo_runtime[arm]["gradient"]["norm"],
                "per_view_norm": pseudo_runtime[arm]["gradient"]["per_view_norm"],
                "logical_sha256": pseudo_runtime[arm]["gradient"]["logical_sha256"],
                "parameter_count": pseudo_runtime[arm]["gradient"]["parameter_count"],
                "model_hash_before_after_equal": pseudo_runtime[arm][
                    "model_hash_before_after_equal"
                ],
            }
            for arm in p0.PSEUDO_ARMS
        },
        "comparisons": {
            key: value["pseudo_pair"] for key, value in scale_comparisons.items()
        },
        "all_trainable_autoencoder_parameters_compared": True,
        "last_layer_only": False,
    }

    native_norm = native_runtime["gradient"]["norm"]
    pseudo_to_native = {
        arm: c1.LAMBDA_PSEUDO
        * pseudo_runtime[arm]["gradient"]["norm"]
        / (native_norm + p0.EPSILON)
        for arm in p0.PSEUDO_ARMS
    }
    native_scale_record = {
        "stage": p0.STAGE,
        "layer": 6,
        "native": {
            "objective": native_runtime["native_objective"],
            "full_dataset_mean_losses": native_runtime[
                "full_dataset_mean_losses"
            ],
            "gradient_norm": native_norm,
            "per_view_gradient_norm": native_runtime["gradient"][
                "per_view_norm"
            ],
            "gradient_logical_sha256": native_runtime["gradient"][
                "logical_sha256"
            ],
            "model_hash_before_after_equal": native_runtime[
                "model_hash_before_after_equal"
            ],
            "native_target_refresh_protocol": native_runtime[
                "native_target_refresh_protocol"
            ],
            "native_P_all_rewritten": False,
        },
        "lambda_pseudo": c1.LAMBDA_PSEUDO,
        "pseudo_to_native_gradient_ratio": pseudo_to_native,
        "comparisons": {
            key: {
                "pseudo_difference_to_native": value[
                    "pseudo_difference_to_native"
                ],
                "total_update": value["total_update"],
            }
            for key, value in scale_comparisons.items()
        },
        "g_total_analytic_sum_only": True,
        "parameter_update_used": False,
    }

    all_runtime = list(pseudo_runtime.values()) + [native_runtime]
    all_hashes_equal = all(
        runtime["model_hash_before_after_equal"] for runtime in all_runtime
    )
    all_initial_hashes = {
        runtime["model_hash_before"]["aggregate"] for runtime in all_runtime
    }
    all_checkpoint_hashes = {
        tuple(record["checkpoint_file_sha256"])
        for record in checkpoint_records.values()
    }
    target_hash_checks = {
        arm: target_provenance[arm]["all_available_hashes_equal"]
        for arm in p0.PSEUDO_ARMS
    }
    c1_frozen_hashes = _frozen_file_hashes()
    audit_record = {
        **p0.audit_boundary_record(),
        "seed": c1.SEED,
        "N": c1.SAMPLE_NUM,
        "V": c1.VIEW_NUM,
        "K": c1.CLASS_NUM,
        "S": c1.DIRECTION_COUNT,
        "batch_size": c1.BATCH_SIZE,
        "lambda_pseudo": c1.LAMBDA_PSEUDO,
        "model_hash_before_after_equal": all_hashes_equal,
        "same_initial_model_hash_all_arms": all_initial_hashes
        == {c1.EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256},
        "initial_model_aggregate_sha256": (
            c1.EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256
        ),
        "same_E1_checkpoint_hashes_all_arms": len(all_checkpoint_hashes) == 1,
        "E1_checkpoint_file_sha256": list(next(iter(all_checkpoint_hashes))),
        "E1_checkpoint_hashes_match_frozen": next(iter(all_checkpoint_hashes))
        == c1.EXPECTED_E1_CHECKPOINT_SHA256,
        "sample_ids_canonical": True,
        "sample_order_identical_all_arms": True,
        "sample_order_sha256": public_sample_order["sample_order_sha256"],
        "sample_order_replayed_from_C1_DataLoader": True,
        "M0_matches_C0_seal": M0_audit["matches_C0_seal_pass"],
        "M0_logical_sha256": M0_audit["logical_sha256"],
        "M0_model_hash_before_after_equal": coordinate_hash_before
        == coordinate_hash_after,
        "model_mode_audit": coordinate_mode_audit,
        "y_gen_hash_unchanged": c0_provenance["array_logical_sha256"]["y_gen"]
        == c1.EXPECTED_C0_ARRAY_SHA256["y_gen"],
        "U_cycle_hash_unchanged": c0_provenance["array_logical_sha256"][
            "U_cycle"
        ]
        == c1.EXPECTED_C0_ARRAY_SHA256["U_cycle"],
        "C_conf_hash_unchanged": c0_provenance["array_logical_sha256"][
            "C_conf"
        ]
        == c1.EXPECTED_C0_ARRAY_SHA256["C_conf"],
        "U_shuffle_hash_unchanged": c0_provenance["array_logical_sha256"][
            "U_cycle_shuffle"
        ]
        == c1.EXPECTED_C0_ARRAY_SHA256["U_cycle_shuffle"],
        "C0_artifact_file_sha256": c0_provenance["artifact_file_sha256"],
        "C0_seal_file_sha256": c0_provenance["seal_file_sha256"],
        "C1_reconstructed_target_hashes_match_frozen": all(
            target_hash_checks.values()
        ),
        "C1_reconstructed_target_hash_check_by_arm": target_hash_checks,
        "C1_frozen_file_sha256": c1_frozen_hashes,
        "feature_provenance": feature_provenance,
        "frozen_input_paths": {
            "C0_artifact": _display(DEFAULT_C0_ARTIFACT_PATH),
            "C0_seal": _display(DEFAULT_C0_SEAL_PATH),
            "C1_A1": _display(DEFAULT_C1A1_OUTPUT_DIR),
            "features": _display(DEFAULT_FEATURE_PATH),
            "E1_model_dir": _display(DEFAULT_E1_MODEL_DIR),
            "E1_audit": _display(DEFAULT_E1_AUDIT_PATH),
        },
        "A_used_for_training": False,
        "all_required_runtime_checks_pass": bool(
            all_hashes_equal
            and len(all_checkpoint_hashes) == 1
            and all(target_hash_checks.values())
            and M0_audit["matches_C0_seal_pass"]
        ),
    }

    diagnostic_results = {
        "stage": p0.STAGE,
        "question": (
            "Where is C0 utility specificity compressed in "
            "W -> T -> action -> gradient?"
        ),
        "read_only_failure_diagnostic": True,
        "new_method": False,
        "scientific_gate": None,
        "descriptive_label_selected": None,
        "human_review_required": True,
        "allowed_descriptive_labels": list(p0.ALLOWED_DESCRIPTIVE_LABELS),
        "continuous_diagnostic_sections": {
            "directional_structure": "c1p0_directional_structure.json",
            "weight_comparison": "c1p0_weight_comparison.json",
            "target_comparison": "c1p0_target_comparison.json",
            "strength_comparison": "c1p0_strength_comparison.json",
            "effective_action_comparison": (
                "c1p0_effective_action_comparison.json"
            ),
            "pseudo_loss": "c1p0_pseudo_loss.json",
            "pseudo_gradient": "c1p0_gradient_comparison.json",
            "native_and_total_gradient": "c1p0_native_gradient_scale.json",
        },
        "audit_file": "c1p0_audit.json",
    }

    records = OrderedDict(
        (
            ("c1p0_directional_structure.json", directional_record),
            ("c1p0_weight_comparison.json", weight_record),
            ("c1p0_target_comparison.json", target_record),
            ("c1p0_strength_comparison.json", strength_record),
            ("c1p0_effective_action_comparison.json", action_record),
            ("c1p0_pseudo_loss.json", pseudo_loss_record),
            ("c1p0_gradient_comparison.json", gradient_record),
            ("c1p0_native_gradient_scale.json", native_scale_record),
            ("c1p0_audit.json", audit_record),
            ("diagnostic_results.json", diagnostic_results),
        )
    )
    _require(tuple(records) == OUTPUT_FILES, "C1-P0 output schema mismatch")
    output_path.mkdir(parents=True, exist_ok=False)
    for name, record in records.items():
        c1.write_json(output_path / name, record)

    print("C1_P0_READ_ONLY_AUDIT_COMPLETE=True")
    print("DESCRIPTIVE_LABEL_SELECTED=None")
    print("Saved: " + _display(output_path))
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_audit(device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
