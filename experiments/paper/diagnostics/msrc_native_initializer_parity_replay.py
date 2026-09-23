"""P0-A3-R1 A6: replay legacy MSRC native initialization and localize parity."""

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

from experiments.paper.transfer_audit.input_artifacts import (
    file_sha256,
    load_materialized_inputs,
)
from release_core.data.weak_quality import (
    generate_half_corruption_mask,
    ndarray_sha256,
)
from release_core.semantics import build_relation_semantics
from release_core.utility import (
    build_directional_actions,
    compute_directional_cycle_utility,
)

from . import p0_a3_protocol as protocol
from .msrc_g0b0_parity_audit import (
    compare_arrays,
    reconstruct_current_msrc,
    validate_reconstruction_against_manifest,
    verify_current_true_u,
    verify_historical_msrc_files,
)


HISTORICAL_ROOT = Path("/root/autodl-tmp/CVPR24-MVCAN")
HISTORICAL_INPUT_DIR = HISTORICAL_ROOT / (
    "outputs/generic_contract/g0_b0_msrc_inputs_seed20_20260916"
)
HISTORICAL_ADAPTER = HISTORICAL_ROOT / (
    "experiments/generic_contract/generic_final_core_adapter.py"
)
HISTORICAL_SOURCE_FILES = {
    HISTORICAL_ADAPTER: "85a97cb72505ef7296c14442a322a6cb54c013697e51b181e0b7cf7962ffa3b6",
    HISTORICAL_ADAPTER.parent / "generic_cycle_utility.py": "2d98ac593fadb14910d9da937662cfd5518052e1cd3642ab30fd18cea42ede27",
    HISTORICAL_ADAPTER.parent / "generic_relation_action.py": "4b9445b9a6735239895b97667286323898d18752d944422367f64560d58e4d57",
    HISTORICAL_ADAPTER.parent / "action_space.py": "11612bb8ae58ca6440d9498c44b50c2a6ef04227e93227ba9c483e13e098cb9b",
    HISTORICAL_ADAPTER.parent / "dataset_contract.py": "1155fb08ff5e4504e8ba45c19bdf72f5f505f07866a205c8e6ee6a460cef4642",
}
HISTORICAL_INPUT_FILES = {
    HISTORICAL_INPUT_DIR / "audit/corruption_audit.json": "487d2945d09eb8b93180b2de8836fa063dfb911f75354d3cb736dccfa856b8fc",
    HISTORICAL_INPUT_DIR / "audit/corruption_mask.npy": "942e0155d8630ea2e03f0d3c665f468dc0e64ac262f874cb06e6cb5a0ec6edfa",
    HISTORICAL_INPUT_DIR / "materialization_audit.json": "c10760fb7e716255f566023d5b1ffa20ac853cfb427328992932301aec6674d5",
    HISTORICAL_INPUT_DIR / "materialization_seal.json": "520dccfdf5a54a6a74fa1cde0816d8cf131822a3941b3058aff1960a101d92c2",
    HISTORICAL_INPUT_DIR / "msrc_sparse_split.npz": "4f599679a368d75bf17f5fa7b142705b534f6f249b09c684c93aafd49bc8059a",
    HISTORICAL_INPUT_DIR / "msrc_sparse_split_seal.json": "cf6d8200f8c4730cfc49d838552e333374bb946a5c46ad471bfcf602c10a7a06",
    HISTORICAL_INPUT_DIR / "msrc_trainable_features.npz": "2a225229de5b4c4d87ac530860691c351a0e8e3c1a015f1ecd290673f663755d",
}
EXPECTED_HISTORICAL_ACTION_HASHES = {
    "U_cycle": "3552d88de9f894276d0d1ccae8f4c55147e98aac9a8ae146eaa1cf54dcc9dd10",
    "PredRelation_true": "cbd2d2de1dfd51abba9b8e80fcc1ee82f4e902df6532ca2af388e1266ff0d2b8",
    "relation_balance_weights_true": "2d69642382947465851d69fe9f151df440979ce626481f9ad4c060909f92ff9a",
}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _numpy(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    _require(isinstance(value, dict), "JSON root must be an object")
    return value


def _write_json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_frozen_legacy_files():
    observed = {}
    for path, expected in {**HISTORICAL_SOURCE_FILES, **HISTORICAL_INPUT_FILES}.items():
        _require(path.is_file(), "frozen legacy file missing: " + str(path))
        actual = file_sha256(path)
        _require(actual == expected, "frozen legacy SHA256 mismatch: " + str(path))
        observed[str(path)] = actual
    return observed


def _under_root(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def load_frozen_legacy_adapter():
    """Load the original historical module under an isolated package name."""
    verify_frozen_legacy_files()
    for name in ("configure", "model", "weak_quality", "irv"):
        existing = sys.modules.get(name)
        if existing is not None and getattr(existing, "__file__", None):
            _require(_under_root(existing.__file__, HISTORICAL_ROOT),
                     "legacy dependency already imported from another tree: " + name)
    root_text = str(HISTORICAL_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    package_name = "_p0_a3_r1_frozen_legacy_generic_contract"
    package = types.ModuleType(package_name)
    package.__path__ = [str(HISTORICAL_ADAPTER.parent)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    module_name = package_name + ".generic_final_core_adapter"
    spec = importlib.util.spec_from_file_location(module_name, HISTORICAL_ADAPTER)
    _require(spec is not None and spec.loader is not None,
             "cannot load frozen legacy adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _require(Path(module.__file__).resolve() == HISTORICAL_ADAPTER.resolve(),
             "legacy adapter origin mismatch")
    for name in ("configure", "model", "weak_quality", "irv.b4_information_utility"):
        loaded = sys.modules.get(name)
        _require(loaded is not None and getattr(loaded, "__file__", None)
                 and _under_root(loaded.__file__, HISTORICAL_ROOT),
                 "legacy dependency origin mismatch: " + name)
    return module


def compare_input_arrays(current, historical, current_mask, historical_mask):
    """Compare only scientific carriers; NPZ container bytes are not a gate."""
    records = {
        "sample_ids": compare_arrays(current["sample_ids"], historical["sample_ids"]),
        "corruption_mask": compare_arrays(current_mask, historical_mask),
    }
    for view_id, (left, right) in enumerate(zip(
        current["views"], historical["views"]
    ), start=1):
        records["view_%d" % view_id] = compare_arrays(left, right)
    for name in ("labeled_ids", "labeled_targets", "unlabeled_ids"):
        records[name] = compare_arrays(current["split"][name], historical["split"][name])
    exact_names = (
        "sample_ids", "corruption_mask", "labeled_ids", "labeled_targets",
        "unlabeled_ids",
    )
    exact = all(records[name]["array_equal"] for name in exact_names)
    views_close = all(
        records["view_%d" % view_id]["array_equal"]
        or records["view_%d" % view_id]["allclose_1e-8"]
        for view_id in range(1, 6)
    )
    return records, bool(exact and views_close)


def classify_a6(*, inputs_equal, historical_replay_exact, state_equal,
                r2_cross_equal, r3_cross_equal):
    if not inputs_equal:
        return "INPUT_MATERIALIZATION_DIVERGENCE"
    if not historical_replay_exact:
        return "HISTORICAL_REPLAY_NONDETERMINISM_OR_ENVIRONMENT_DIVERGENCE"
    if not r2_cross_equal:
        return "R2_MIGRATION_DIVERGENCE"
    if not r3_cross_equal:
        return "R3_MIGRATION_DIVERGENCE"
    if not state_equal:
        return "NATIVE_INITIALIZATION_MIGRATION_DIVERGENCE"
    return "FAIL_CLOSED_UNCLASSIFIED_FULL_PARITY_CONTRADICTION"


def _persist_report(output, record, legacy_state=None, current_state=None):
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    if legacy_state is not None:
        np.savez(target / "legacy_pre_r2_state.npz", **legacy_state)
    if current_state is not None:
        np.savez(target / "current_pre_r2_state.npz", **current_state)
    _write_json_exclusive(target / "msrc_native_initializer_parity_replay.json", record)
    return record


def run_replay(output_dir, device="cuda:0"):
    """Execute the explicitly authorized diagnostic replay; never load full GT."""
    _require(str(device) == "cuda:0", "A6 replay device is frozen at cuda:0")
    _require(not Path(output_dir).exists(), "refusing to overwrite A6 output")
    source_hashes = verify_frozen_legacy_files()
    verify_historical_msrc_files()
    legacy = load_frozen_legacy_adapter()
    historical = legacy.validate_materialized_inputs(
        HISTORICAL_INPUT_DIR, legacy.MSRC_RUNTIME_SPEC
    )
    _, current_views, current_sample_ids, current_split = load_materialized_inputs(
        protocol.MSRC_INPUT_DIR
    )
    current_mask, mask_audit = generate_half_corruption_mask(210, 5, 20)
    current_mask = np.ascontiguousarray(current_mask, dtype=np.bool_)
    current_feature_audit = _read_json(protocol.MSRC_INPUT_DIR / "feature_audit.json")
    _require(mask_audit["mask_sha256"] == current_feature_audit["mask_logical_sha256"],
             "current deterministic mask does not match its sealed audit")
    historical_mask = np.ascontiguousarray(np.load(
        HISTORICAL_INPUT_DIR / "audit/corruption_mask.npy", allow_pickle=False
    ), dtype=np.bool_)
    current_inputs = {
        "views": current_views,
        "sample_ids": current_sample_ids,
        "split": {
            "labeled_ids": current_split.labeled_ids,
            "labeled_targets": current_split.labeled_targets,
            "unlabeled_ids": current_split.unlabeled_ids,
        },
    }
    input_comparisons, inputs_equal = compare_input_arrays(
        current_inputs, historical, current_mask, historical_mask
    )
    base_record = {
        "schema": "paper-p0-a3-r1-msrc-native-initializer-parity-v1",
        "training_seed": 20,
        "device": device,
        "full_gt_loaded": False,
        "historical_original_prepare_native_backbone_called": False,
        "frozen_legacy_file_sha256": source_hashes,
        "input_comparisons": input_comparisons,
        "input_scientific_arrays_equal": inputs_equal,
    }
    if not inputs_equal:
        base_record["decision"] = classify_a6(
            inputs_equal=False, historical_replay_exact=False,
            state_equal=False, r2_cross_equal=False, r3_cross_equal=False,
        )
        return _persist_report(output_dir, base_record)

    native = legacy.prepare_native_backbone(
        historical["views"], historical["contract"], device, 20,
        legacy.MSRC_RUNTIME_SPEC,
    )
    legacy_state = {
        "q_local": _numpy(native["q_local"], np.float32),
        "q_aligned": _numpy(native["q_aligned"], np.float32),
        "M_v": _numpy(native["M_v"], np.float32),
    }
    legacy_cycle = legacy.build_cycle_utility(native["q_aligned"])
    legacy_u = _numpy(legacy_cycle["U_cycle"], np.float32)
    legacy_y = _numpy(legacy_cycle["y_gen"], np.int64)
    historical_split = historical["split"]
    legacy_relation = legacy.build_relation_semantics(
        legacy_y,
        historical_split["labeled_ids"],
        historical_split["labeled_targets"],
        historical_split["unlabeled_ids"],
        class_count=historical["contract"].K,
        labels_per_class=historical["contract"].labels_per_class,
    )
    legacy_pred = _numpy(legacy_relation["PredRelation_true"], np.bool_)
    legacy_balance = _numpy(
        legacy_relation["relation_balance_weights_true"], np.float64
    )
    replay_hashes = {
        "U_cycle": ndarray_sha256(legacy_u),
        "PredRelation_true": ndarray_sha256(legacy_pred),
        "relation_balance_weights_true": ndarray_sha256(legacy_balance),
    }
    replay_exact = replay_hashes == EXPECTED_HISTORICAL_ACTION_HASHES
    base_record.update({
        "historical_original_prepare_native_backbone_called": True,
        "historical_native_audit": native["audit"],
        "historical_replay_action_hashes": replay_hashes,
        "expected_historical_action_hashes": EXPECTED_HISTORICAL_ACTION_HASHES,
        "historical_replay_frozen_action_exact": replay_exact,
    })
    if not replay_exact:
        base_record["historical_replay_validation"] = (
            "HISTORICAL_NATIVE_REPLAY_NOT_EXACT"
        )
        base_record["decision"] = classify_a6(
            inputs_equal=True, historical_replay_exact=False,
            state_equal=False, r2_cross_equal=False, r3_cross_equal=False,
        )
        return _persist_report(output_dir, base_record, legacy_state=legacy_state)

    manifest = verify_current_true_u()
    current, release_split = reconstruct_current_msrc(device)
    current_identity = validate_reconstruction_against_manifest(current, manifest)
    current_state = {
        "q_local": current["q_local"],
        "q_aligned": current["q_aligned"],
        "M_v": current["M_v"],
    }
    state_comparisons = {
        name: compare_arrays(current_state[name], legacy_state[name])
        for name in ("q_local", "q_aligned", "M_v")
    }
    state_equal = all(value["array_equal"] for value in state_comparisons.values())

    release_actions = build_directional_actions(5)
    release_cycle = compute_directional_cycle_utility(
        torch.from_numpy(legacy_state["q_aligned"]).to(device), release_actions
    )
    r2_pairs = {
        "U_cycle": (legacy_u, _numpy(release_cycle["U_cycle"], np.float32)),
        "y_gen": (legacy_y, _numpy(release_cycle["y_gen"], np.int64)),
        "closure": (
            _numpy(legacy_cycle["closure"], np.bool_),
            _numpy(release_cycle["closure"], np.bool_),
        ),
        "generator_membership": (
            _numpy(legacy_cycle["generator_membership"], np.int64),
            _numpy(release_cycle["generator_membership"], np.int64),
        ),
        "verifier_membership": (
            _numpy(legacy_cycle["verifier_membership"], np.int64),
            _numpy(release_cycle["verifier_membership"], np.int64),
        ),
    }
    r2_comparisons = {
        name: compare_arrays(*pair) for name, pair in r2_pairs.items()
    }
    r2_equal = all(value["array_equal"] for value in r2_comparisons.values())

    release_relation = build_relation_semantics(legacy_y, release_split, release_actions)
    r3_pairs = {
        "mapping": (
            _numpy(legacy_relation["sparse_mapping"], np.int64),
            _numpy(release_relation.sparse_mapping, np.int64),
        ),
        "class_pred": (
            _numpy(legacy_relation["class_pred"], np.int64),
            _numpy(release_relation.class_pred, np.int64),
        ),
        "PredRelation_true": (
            legacy_pred, _numpy(release_relation.pred_relation, np.bool_),
        ),
        "relation_balance_weights_true": (
            legacy_balance,
            _numpy(release_relation.balance_weights, np.float64),
        ),
    }
    r3_comparisons = {
        name: compare_arrays(*pair) for name, pair in r3_pairs.items()
    }
    r3_equal = all(value["array_equal"] for value in r3_comparisons.values())
    base_record.update({
        "current_action_provenance_validation": current_identity,
        "pre_r2_state_comparisons": state_comparisons,
        "pre_r2_state_exact": state_equal,
        "same_legacy_state_historical_vs_release_r2": r2_comparisons,
        "r2_cross_call_exact": r2_equal,
        "same_y_gen_split_historical_vs_release_r3": r3_comparisons,
        "r3_cross_call_exact": r3_equal,
        "decision": classify_a6(
            inputs_equal=True, historical_replay_exact=True,
            state_equal=state_equal, r2_cross_equal=r2_equal,
            r3_cross_equal=r3_equal,
        ),
    })
    return _persist_report(
        output_dir, base_record, legacy_state=legacy_state,
        current_state=current_state,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    record = run_replay(args.output_dir, args.device)
    print(json.dumps({"decision": record["decision"], "output": args.output_dir},
                     sort_keys=True))


if __name__ == "__main__":
    main()
