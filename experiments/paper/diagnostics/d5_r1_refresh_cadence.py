"""Paper-local D5-R1: only R5 refresh_interval may differ from MSRC v1."""
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from release_core.runtime import SealedPredictionPaths, run_pre_gt
from experiments.paper.formal import p1_a1_action_materialization as actions
from experiments.paper.formal import p1_a1_native_preparation as preparation
from experiments.paper.formal import p1_a1_runner as formal_runner
from experiments.paper.formal import p1_a3_runtime_wiring as wiring
from experiments.paper.formal import p1_a4_postseal_evaluation as postseal
from experiments.paper.formal import run_formal_pipeline as formal_pipeline

FORMAL_ROOT = Path("outputs/paper/formal")
ROOT = Path("outputs/paper/diagnostics/d5_r1_refresh_cadence")
DATASET, SEEDS = "MSRC-v1", (20, 30, 50)
V1_REFRESH_INTERVAL, DIAGNOSTIC_REFRESH_INTERVAL = 100, 1

def _require(condition, message):
    if not condition:
        raise RuntimeError(message)

def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _write(path, record):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")

def _under(path, parent):
    return os.path.commonpath((str(Path(path).resolve()), str(Path(parent).resolve()))) == str(Path(parent).resolve())

def paths_for(seed):
    _require(seed in SEEDS, "D5_R1_SEED_NOT_AUTHORIZED")
    stem, root = "msrcv1", ROOT / "msrcv1" / ("seed" + str(seed))
    formal = FORMAL_ROOT / "main" / stem / ("seed" + str(seed))
    return {"root": root, "output": root / "OURS_TRUE_U_refresh1", "adapter": root / "_adapters" / "ours_true_u", "inputs": FORMAL_ROOT / "inputs" / stem, "initialization": FORMAL_ROOT / "preparation" / stem / ("seed" + str(seed)), "action": FORMAL_ROOT / "actions" / stem / ("seed" + str(seed)) / "true_action_state", "base": formal / "BASE", "ours_v1": formal / "OURS_TRUE_U", "v1_adapter": formal / "_adapters" / "OURS_TRUE_U"}

def _sealed(root):
    return SealedPredictionPaths(root / "pre_gt_bundle.npz", root / "pre_gt_audit.json", root / "pre_gt_seal.json")

def _baseline(root, arm):
    sealed = _sealed(root)
    postseal._validate(arm, sealed)
    metrics = root / "postseal_metrics.json"
    _require(metrics.is_file(), "D5_R1_BASELINE_METRICS_MISSING")
    return {"bundle_sha256": _sha(sealed.bundle), "audit_sha256": _sha(sealed.audit), "seal_sha256": _sha(sealed.seal), "metrics_sha256": _sha(metrics), "metrics": json.loads(metrics.read_text(encoding="utf-8"))["metrics"]}

def _verify_v1(seed):
    paths = paths_for(seed)
    formal_pipeline._verify_inputs(paths["inputs"], DATASET)
    initial = preparation.verify_initialization(paths["initialization"], dataset=DATASET, training_seed=seed)
    action = actions.verify_true_action(paths["action"], dataset=DATASET, training_seed=seed, initial_model_sha256=initial["initial_model_sha256"])
    for name in ("utility_audit.json", "semantic_audit.json"):
        record = json.loads((paths["v1_adapter"] / name).read_text(encoding="utf-8"))
        _require(record.get("source_action_sha256") == action["artifact_sha256"], "D5_R1_V1_ACTION_SOURCE_MISMATCH")
    baselines = {"BASE_v1": _baseline(paths["base"], "BASE"), "OURS_TRUE_U_v1_refresh100": _baseline(paths["ours_v1"], "OURS_TRUE_U")}
    return paths, initial, action, baselines

def diagnostic_runtime(seed, device):
    v1 = formal_runner.runtime_config(formal_runner.FormalRun(DATASET, seed, "OURS_TRUE_U", device))
    _require(v1.refresh_interval == V1_REFRESH_INTERVAL, "D5_R1_V1_REFRESH_INTERVAL_MISMATCH")
    diagnostic = replace(v1, refresh_interval=DIAGNOSTIC_REFRESH_INTERVAL)
    _require({key for key in asdict(v1) if getattr(v1, key) != getattr(diagnostic, key)} == {"refresh_interval"}, "D5_R1_NON_CADENCE_CONFIG_CHANGE")
    return v1, diagnostic

def refresh_epochs(runtime):
    return tuple(epoch for epoch in range(runtime.epochs) if epoch % runtime.refresh_interval == 0)

def prepare(seed, device):
    """Read/verify v1 only; this performs no training and accepts no GT."""
    candidate = paths_for(seed)
    _require(not _under(candidate["root"], FORMAL_ROOT), "D5_R1_OUTPUT_OVERLAPS_FORMAL_NAMESPACE")
    _require(not candidate["root"].exists(), "D5_R1_OUTPUT_ALREADY_EXISTS")
    paths, initial, action, baselines = _verify_v1(seed)
    v1, runtime = diagnostic_runtime(seed, device)
    _require(refresh_epochs(runtime) == tuple(range(runtime.epochs)), "D5_R1_REFRESH_EVERY_EPOCH_REQUIRED")
    adapter = wiring.materialize_ours_true_u_adapter(true_action=action, output_dir=paths["adapter"])
    for name in ("utility_audit", "semantic_audit"):
        record = json.loads(Path(adapter[name]).read_text(encoding="utf-8"))
        _require(record.get("source_action_sha256") == action["artifact_sha256"], "D5_R1_DIAGNOSTIC_ACTION_SOURCE_MISMATCH")
    provenance = wiring.arm_provenance(feature=paths["inputs"] / "features.npz", feature_audit=paths["inputs"] / "feature_audit.json", split=paths["inputs"] / "sparse_split.npz", split_audit=paths["inputs"] / "sparse_split_audit.json", utility=adapter["utility"], utility_audit=adapter["utility_audit"], semantic=adapter["semantic"], semantic_audit=adapter["semantic_audit"], initialization=initial, output=paths["output"])
    return {"paths": paths, "initialization": initial, "action": action, "baselines": baselines, "v1": v1, "runtime": runtime, "adapter": adapter, "provenance": provenance}

def _dynamics(audit):
    epochs = audit["r5_training_audit"]["epoch_audits"]
    return {"epochs": len(epochs), "refresh_count": audit["r5_training_audit"]["refresh_count"], "refresh_epochs": [item["epoch"] for item in epochs if item["target_refresh"]["executed"]], "phase_a_loss_trajectory": [item["phase_a"]["loss_sum"] for item in epochs], "phase_b_loss_trajectory": [item["phase_b"]["loss_sum"] for item in epochs]}

def run(seed, device):
    """The only D5-R1 trainable operation.  BASE and v1 OURS are never run."""
    plan = prepare(seed, device)
    sealed = run_pre_gt(plan["runtime"], plan["provenance"])
    postseal._validate("OURS_TRUE_U", sealed)
    audit = json.loads(sealed.audit.read_text(encoding="utf-8"))
    record = {"schema": "paper-d5-r1-refresh-cadence-v1", "dataset": DATASET, "training_seed": seed, "v1_refresh_interval": V1_REFRESH_INTERVAL, "diagnostic_refresh_interval": DIAGNOSTIC_REFRESH_INTERVAL, "initial_model_sha256": audit["initial_model_sha256"], "canonical_true_action_sha256": plan["action"]["artifact_sha256"], "utility_sha256": audit["input_identity"]["utility_sha256"], "semantic_sha256": audit["input_identity"]["semantic_sha256"], "frozen_v1_baselines": plan["baselines"], "dynamics": _dynamics(audit), "final_prediction_sha256": audit["prediction_logical_sha256"], "pre_gt_bundle_sha256": _sha(sealed.bundle), "pre_gt_audit_sha256": _sha(sealed.audit), "pre_gt_seal_sha256": _sha(sealed.seal), "u_or_relation_regenerated": False, "base_or_v1_ours_retrained": False, "gt_firewall": {"full_gt_loaded_before_seal": False, "metrics_computed_before_seal": False}}
    _write(plan["paths"]["root"] / "d5_r1_audit.json", record)
    return sealed, record

def evaluate_after_seal(seed, full_gt_path):
    """Explicit post-seal-only metric operation; it never trains."""
    paths = paths_for(seed)
    sealed = _sealed(paths["output"])
    postseal._validate("OURS_TRUE_U", sealed)
    return postseal.evaluate_formal_postseal(dataset=DATASET, training_seed=seed, arm="OURS_TRUE_U", full_gt_path=full_gt_path, sealed=sealed, output_path=paths["root"] / "postseal_metrics.json")
