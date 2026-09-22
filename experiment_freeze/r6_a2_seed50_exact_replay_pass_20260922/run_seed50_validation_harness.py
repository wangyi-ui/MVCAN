#!/usr/bin/env python3
"""R6-A2 seed50-only frozen replay validation harness."""

import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, "/root/autodl-tmp/UCRR-MVC-release")

import numpy as np

from release_core.runtime import (
    ProvenanceConfig,
    RuntimeConfig,
    SealedPredictionPaths,
    evaluate_postseal,
    run_pre_gt,
)
import release_core.runtime.entrypoint as runtime_entrypoint


RELEASE_ROOT = Path("/root/autodl-tmp/UCRR-MVC-release")
HISTORICAL_ROOT = Path("/root/autodl-tmp/CVPR24-MVCAN")
TEMP_ROOT = Path("/tmp/r6_a2_replay")
OUTPUT_ROOT = RELEASE_ROOT / "outputs/release_validation/r6_a2/seed50"
FREEZE_ROOT = (
    RELEASE_ROOT
    / "experiment_freeze/r6_a1_clean_runtime_entrypoint_pass_20260922"
)
BRIDGE_POLICY_FREEZE_ROOT = RELEASE_ROOT / (
    "experiment_freeze/r6_a2_r0_seed20_exact_replay_pass_20260922"
)
PREVIOUS_REPLAY_FREEZE_ROOT = RELEASE_ROOT / (
    "experiment_freeze/r6_a2_seed30_exact_replay_pass_20260922"
)

EXPECTED_BRANCH = "release/frozen-core-extraction"
EXPECTED_HEAD = "dd0fbe45c849e021733b6387e9db64152f14d9e1"
EXPECTED_TAG = "r6-a2-seed30-exact-replay-pass-20260922"
EXPECTED_FREEZE_MANIFEST_SHA = (
    "1f04294e2384041e5fcd64edd3201d5cad6051ec2b25b33490a6fdba9373d952"
)
EXPECTED_R6_SOURCE_MANIFEST_SHA = (
    "c8afab5754cee671f8ffc9c225c7546cffdeab53ba27b8c86849ac60c28c8d2a"
)
EXPECTED_BRIDGE_POLICY_FREEZE_MANIFEST_SHA = (
    "d50b0cce04e3b864c4b6fc701c1225f9b2410b33283f0ee38a4459f30d2662ca"
)
EXPECTED_PREVIOUS_REPLAY_FREEZE_MANIFEST_SHA = (
    "b28e9378283d452eaa75574a9bb42c1e0a8897e1800833604f2316953bd3a753"
)

HISTORICAL_HASH_PATH = HISTORICAL_ROOT / "irv/b3_audit.py"
EXPECTED_HISTORICAL_HASH_SOURCE_SHA = (
    "e098a32e7f1fd39770951f47383f1cc2fc88093f66d2e30cf10b875e75478d0b"
)
EXPECTED_R6_HASH_SOURCE_SHA = (
    "ac0fb8ff6358865e410911fc568003c5bd6fa7a3756c345d0a903b89cf2aedfb"
)
EXPECTED_HISTORICAL_INITIAL = (
    "41c98952c143c149d639451a16a5902fd25281013a6a937559f9835365ed1ca5"
)
EXPECTED_R6_INITIAL = (
    "d975025cdc5028f462a7362e04b12f0b383252da9b9a52a48785c9c54bfc6dc2"
)
EXPECTED_HISTORICAL_FINAL = (
    "0c44f569fc564ee4649e3c1882e2580e0a05dabc9850dce6826158ee17168062"
)
EXPECTED_PREDICTION = (
    "d2bfa6ae66d2541f07d8a630f81e69693115ce3fff15ede76140c3cdbd5a6f89"
)

FEATURE_ARTIFACT = (
    HISTORICAL_ROOT / "outputs/e0_glgc_adapter/caltech6v_snr2p5_k3_seed20.npz"
)
FEATURE_AUDIT = HISTORICAL_ROOT / "outputs/e0_glgc_adapter/export_audit.json"
ACTION_ARTIFACT = HISTORICAL_ROOT / (
    "outputs/cyclic_utility/c3_a0_utility_conditioned_action_granularity_seed50/"
    "c3_a0_action_pre_gt.npz"
)
ACTION_AUDIT = HISTORICAL_ROOT / (
    "outputs/cyclic_utility/c3_a0_utility_conditioned_action_granularity_seed50/"
    "c3_a0_action_seal.json"
)
CHECKPOINT_ROOT = HISTORICAL_ROOT / (
    "outputs/e1_pairwise_utility/lwc_100ep_seed50/models"
)
CHECKPOINTS = tuple(
    CHECKPOINT_ROOT / ("Caltech-6V" + str(index) + "V.pth")
    for index in range(1, 7)
)
CHECKPOINT_AUDIT = HISTORICAL_ROOT / (
    "outputs/e1_pairwise_utility/lwc_100ep_seed50/e1_audit.json"
)
FULL_GT = HISTORICAL_ROOT / "data/Caltech.mat"

REFERENCE_ROOT = HISTORICAL_ROOT / (
    "outputs/final_core/f0_a0_exact_replay_seed50"
)
REFERENCE_ARTIFACT = REFERENCE_ROOT / "final_core_pre_gt_artifact.npz"
REFERENCE_AUDIT = REFERENCE_ROOT / "final_core_pre_gt_audit.json"
REFERENCE_SEAL = REFERENCE_ROOT / "final_core_pre_gt_seal.json"
REFERENCE_METRICS = REFERENCE_ROOT / "final_core_metrics.json"

EXPECTED_FILES = {
    FEATURE_ARTIFACT: "44133379b756fa83744d7745dd477c409d77db3f078ccab6064643c85428d8e4",
    FEATURE_AUDIT: "c2b720b6acd84670e9645cedc5209970300ea09478d91005dca1a7a9e6c3399b",
    ACTION_ARTIFACT: "f6a3c3ec53fa2f327748f3a3e6b564fd839184be6c682375785d7436585d271b",
    ACTION_AUDIT: "2fab5003aee4d4dc134160e85dc4325a6aa4e5816e3bf65f1a4f7da4a39cf71d",
    CHECKPOINT_AUDIT: "20207f7efaa3eb702f730b2925feade5852242696e219f70a61863e27e658c7a",
    CHECKPOINTS[0]: "2a34b4fdd143edfd9014c13206ffbd7c889758b4a7df14fce8704227454a1fbb",
    CHECKPOINTS[1]: "960cdd4d83ae778da3401119f71fe5466487bfc93aad32f2981ef4765845200d",
    CHECKPOINTS[2]: "0ab7cb2d44dcb9cc7c06df10ffd0bc444f6f827621d340263d8fa9e222aaef94",
    CHECKPOINTS[3]: "fccbf9d5f7ddc7a74d67815c7761003cded1b567596b33d29a95ab3333e55815",
    CHECKPOINTS[4]: "575843a3efa98927a19b1ae0e2451d972d40b43f0195581c1e76314b95b9a767",
    CHECKPOINTS[5]: "fdcb04fc570e42e8302ce10374a8994a1928519b21dfa9c72c524bc1ff6b1bbe",
}
EXPECTED_REFERENCE_FILES = {
    REFERENCE_ARTIFACT: "b2a6aa3c46cc94c3382687d151b2666018e2743181098c761b2c7928ef9721cb",
    REFERENCE_AUDIT: "bd83c5ba606b9bb2f01cb2da1cb8b406a7e4f1f666f9aa9966e034076d64a9d4",
    REFERENCE_SEAL: "6082effc877ff3575a805bd3aa53c8d6483c87e55ac679b5defa1cc8d6923c65",
}
EXPECTED_METRICS = {
    "acc": 0.8607142857142858,
    "nmi": 0.7763024116674858,
    "ari": 0.7538086225874288,
}

SPLIT_ARTIFACT = TEMP_ROOT / "seed50_sparse_split_adapter.npz"
SPLIT_AUDIT = TEMP_ROOT / "seed50_sparse_split_adapter_audit.json"
BRIDGE_PROTOCOL = TEMP_ROOT / "hash_namespace_bridge_protocol_seed50.txt"
VALIDATION_RECORD = OUTPUT_ROOT / "r6_a2_seed50_validation.json"


class GateFailure(RuntimeError):
    pass


def require(condition, gate, expected, actual):
    if not condition:
        raise GateFailure(
            "GATE=" + str(gate)
            + "\nEXPECTED=" + str(expected)
            + "\nACTUAL=" + str(actual)
        )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    for component in (
        str(array.dtype).encode("ascii"),
        ",".join(str(int(size)) for size in array.shape).encode("ascii"),
        array.tobytes(order="C"),
    ):
        digest.update(struct.pack(">Q", len(component)))
        digest.update(component)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), "JSON_ROOT", "object", type(value).__name__)
    return value


def write_json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def git(*arguments):
    result = subprocess.run(
        ("git",) + tuple(arguments),
        cwd=RELEASE_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def verify_manifest(manifest, root):
    for raw_line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        expected, relative = raw_line.split(maxsplit=1)
        if relative.startswith("*"):
            relative = relative[1:]
        if relative.startswith("./"):
            relative = relative[2:]
        target = Path(root) / relative
        require(target.is_file(), "MANIFEST_FILE_PRESENT", target, "missing")
        actual = file_sha256(target)
        require(actual == expected, "MANIFEST_SHA256", expected, actual)


def verify_start_gate():
    require(git("branch", "--show-current") == EXPECTED_BRANCH,
            "GIT_BRANCH", EXPECTED_BRANCH, git("branch", "--show-current"))
    head = git("rev-parse", "HEAD")
    require(head == EXPECTED_HEAD, "GIT_HEAD", EXPECTED_HEAD, head)
    tag = git("rev-parse", EXPECTED_TAG)
    require(tag == EXPECTED_HEAD, "GIT_TAG", EXPECTED_HEAD, tag)
    require(git("status", "--short", "--untracked-files=all") == "",
            "GIT_STATUS", "clean", "dirty")
    require(git("diff", "--name-only") == "", "GIT_DIFF", "empty", "nonempty")
    require(git("diff", "--cached", "--name-only") == "",
            "GIT_CACHED_DIFF", "empty", "nonempty")
    require(not OUTPUT_ROOT.exists(), "OUTPUT_ABSENT", "absent", OUTPUT_ROOT)

    freeze_manifest = FREEZE_ROOT / "freeze_metadata_sha256.txt"
    require(file_sha256(freeze_manifest) == EXPECTED_FREEZE_MANIFEST_SHA,
            "R6_FREEZE_MANIFEST_SHA", EXPECTED_FREEZE_MANIFEST_SHA,
            file_sha256(freeze_manifest))
    verify_manifest(freeze_manifest, FREEZE_ROOT)
    source_manifest = FREEZE_ROOT / "r6_source_sha256.txt"
    require(file_sha256(source_manifest) == EXPECTED_R6_SOURCE_MANIFEST_SHA,
            "R6_SOURCE_MANIFEST_SHA", EXPECTED_R6_SOURCE_MANIFEST_SHA,
            file_sha256(source_manifest))
    verify_manifest(source_manifest, RELEASE_ROOT)

    bridge_policy_manifest = BRIDGE_POLICY_FREEZE_ROOT / "freeze_metadata_sha256.txt"
    require(
        file_sha256(bridge_policy_manifest)
        == EXPECTED_BRIDGE_POLICY_FREEZE_MANIFEST_SHA,
        "SEED20_BRIDGE_POLICY_FREEZE_MANIFEST_SHA",
        EXPECTED_BRIDGE_POLICY_FREEZE_MANIFEST_SHA,
        file_sha256(bridge_policy_manifest),
    )
    verify_manifest(bridge_policy_manifest, BRIDGE_POLICY_FREEZE_ROOT)

    previous_replay_manifest = PREVIOUS_REPLAY_FREEZE_ROOT / "freeze_metadata_sha256.txt"
    require(
        file_sha256(previous_replay_manifest)
        == EXPECTED_PREVIOUS_REPLAY_FREEZE_MANIFEST_SHA,
        "SEED30_REPLAY_FREEZE_MANIFEST_SHA",
        EXPECTED_PREVIOUS_REPLAY_FREEZE_MANIFEST_SHA,
        file_sha256(previous_replay_manifest),
    )
    verify_manifest(previous_replay_manifest, PREVIOUS_REPLAY_FREEZE_ROOT)

    require(file_sha256(HISTORICAL_HASH_PATH) == EXPECTED_HISTORICAL_HASH_SOURCE_SHA,
            "HISTORICAL_HASH_SOURCE_SHA", EXPECTED_HISTORICAL_HASH_SOURCE_SHA,
            file_sha256(HISTORICAL_HASH_PATH))
    r6_source = RELEASE_ROOT / "release_core/runtime/entrypoint.py"
    require(file_sha256(r6_source) == EXPECTED_R6_HASH_SOURCE_SHA,
            "R6_HASH_SOURCE_SHA", EXPECTED_R6_HASH_SOURCE_SHA,
            file_sha256(r6_source))
    for path, expected in {**EXPECTED_FILES, **EXPECTED_REFERENCE_FILES}.items():
        require(path.is_file(), "INPUT_PRESENT", path, "missing")
        actual = file_sha256(path)
        require(actual == expected, "INPUT_SHA256", expected, actual)
    require(FULL_GT.is_file(), "FULL_GT_PRESENT", FULL_GT, "missing")


def load_historical_hash_module():
    spec = importlib.util.spec_from_file_location(
        "r6_a2_frozen_b3_audit", HISTORICAL_HASH_PATH
    )
    require(spec is not None and spec.loader is not None,
            "HISTORICAL_HASH_IMPORT", "loadable", "unloadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def materialize_split_adapter():
    require(not SPLIT_ARTIFACT.exists(), "SPLIT_ADAPTER_ABSENT", "absent", "exists")
    require(not SPLIT_AUDIT.exists(), "SPLIT_AUDIT_ABSENT", "absent", "exists")
    names = ("sample_ids", "labeled_ids", "labeled_targets", "unlabeled_ids")
    with np.load(ACTION_ARTIFACT, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True, order="C") for name in names}
    np.savez(SPLIT_ARTIFACT, **arrays)
    with SPLIT_ARTIFACT.open("rb+") as stream:
        os.fsync(stream.fileno())
    adapter_sha = file_sha256(SPLIT_ARTIFACT)
    audit = {
        "schema": "r6-a2-r0-exact-copy-sparse-split-adapter-v1",
        "artifact_path": str(SPLIT_ARTIFACT),
        "artifact_sha256": adapter_sha,
        "parent_source_path": str(ACTION_ARTIFACT),
        "parent_source_sha256": EXPECTED_FILES[ACTION_ARTIFACT],
        "copied_keys": list(names),
        "scientific_recomputation": False,
        "label_reselection": False,
        "utility_transformation": False,
        "relation_transformation": False,
        "exact_array_copy": True,
    }
    write_json_exclusive(SPLIT_AUDIT, audit)
    with np.load(SPLIT_ARTIFACT, allow_pickle=False) as adapter, np.load(
        ACTION_ARTIFACT, allow_pickle=False
    ) as parent:
        require(tuple(adapter.files) == names, "SPLIT_ADAPTER_KEYS", names,
                tuple(adapter.files))
        for name in names:
            require(np.array_equal(adapter[name], parent[name]),
                    "SPLIT_ADAPTER_EXACT_" + name, True, False)
    return adapter_sha, file_sha256(SPLIT_AUDIT)


def freeze_bridge_protocol():
    require(not BRIDGE_PROTOCOL.exists(), "BRIDGE_PROTOCOL_ABSENT", "absent", "exists")
    text = """R6-A2-R0 MODEL HASH NAMESPACE BRIDGE PROTOCOL\n\nHistorical canonical hash implementation:\n/root/autodl-tmp/CVPR24-MVCAN/irv/b3_audit.py::hash_backbone(autoencoders)[\"aggregate\"]\nHistorical implementation SHA256:\ne098a32e7f1fd39770951f47383f1cc2fc88093f66d2e30cf10b875e75478d0b\n\nR6 audit-local hash implementation:\n/root/autodl-tmp/UCRR-MVC-release/release_core/runtime/entrypoint.py::_model_sha256\nR6 implementation source SHA256:\nac0fb8ff6358865e410911fc568003c5bd6fa7a3756c345d0a903b89cf2aedfb\n\nExpected historical initial hash:\n41c98952c143c149d639451a16a5902fd25281013a6a937559f9835365ed1ca5\nExpected R6 initial hash:\nd975025cdc5028f462a7362e04b12f0b383252da9b9a52a48785c9c54bfc6dc2\nExpected historical final post-B hash:\n0c44f569fc564ee4649e3c1882e2580e0a05dabc9850dce6826158ee17168062\nExpected prediction logical SHA256:\nd2bfa6ae66d2541f07d8a630f81e69693115ce3fff15ede76140c3cdbd5a6f89\n\nComparison policy:\n1. Historical canonical and R6 internal hashes are distinct namespaces.\n2. Initial model must equal both preregistered values in their own namespaces.\n3. Immediately after R5 returns and before final refresh, historical canonical hash must equal the preregistered R5-A2 final value.\n4. expected_final_model_sha256 is None because no pre-existing historical R6-namespace final state exists.\n5. The observed R6 final hash is run-local integrity evidence only and is not learned as an expected value.\n6. Before/after final refresh, historical hash must remain the preregistered final value and R6 hash must remain unchanged.\n7. Wrappers call originals exactly once, read hashes only, and return original outputs unchanged.\n"""
    with BRIDGE_PROTOCOL.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    digest = file_sha256(BRIDGE_PROTOCOL)
    print(digest + "  " + str(BRIDGE_PROTOCOL), flush=True)
    return digest


def install_instrumentation(historical_module, evidence):
    originals = {
        "load": runtime_entrypoint._load_initial_model,
        "train": runtime_entrypoint.run_alternating_training,
        "refresh": runtime_entrypoint._final_prediction_state,
    }

    def wrapped_load(*args, **kwargs):
        evidence["load_original_call_count"] += 1
        result = originals["load"](*args, **kwargs)
        model, r6_hash = result[0], result[1]
        historical_hash = historical_module.hash_backbone(
            model.autoencoders
        )["aggregate"]
        evidence["initial_historical_hash"] = historical_hash
        evidence["initial_r6_hash"] = r6_hash
        require(historical_hash == EXPECTED_HISTORICAL_INITIAL,
                "INITIAL_HISTORICAL_HASH", EXPECTED_HISTORICAL_INITIAL,
                historical_hash)
        require(r6_hash == EXPECTED_R6_INITIAL,
                "INITIAL_R6_HASH", EXPECTED_R6_INITIAL, r6_hash)
        return result

    def wrapped_train(*args, **kwargs):
        evidence["train_original_call_count"] += 1
        result = originals["train"](*args, **kwargs)
        trained_model = result[0]
        historical_hash = historical_module.hash_backbone(
            trained_model.autoencoders
        )["aggregate"]
        r6_hash = runtime_entrypoint._model_sha256(trained_model)
        evidence["final_post_b_historical_hash"] = historical_hash
        evidence["final_post_b_r6_hash"] = r6_hash
        require(historical_hash == EXPECTED_HISTORICAL_FINAL,
                "FINAL_POST_B_HISTORICAL_HASH", EXPECTED_HISTORICAL_FINAL,
                historical_hash)
        return result

    def wrapped_refresh(*args, **kwargs):
        evidence["refresh_original_call_count"] += 1
        model = args[0]
        historical_before = historical_module.hash_backbone(
            model.autoencoders
        )["aggregate"]
        r6_before = runtime_entrypoint._model_sha256(model)
        evidence["refresh_historical_before"] = historical_before
        evidence["refresh_r6_before"] = r6_before
        require(historical_before == EXPECTED_HISTORICAL_FINAL,
                "REFRESH_HISTORICAL_BEFORE", EXPECTED_HISTORICAL_FINAL,
                historical_before)
        result = originals["refresh"](*args, **kwargs)
        historical_after = historical_module.hash_backbone(
            model.autoencoders
        )["aggregate"]
        r6_after = runtime_entrypoint._model_sha256(model)
        evidence["refresh_historical_after"] = historical_after
        evidence["refresh_r6_after"] = r6_after
        require(historical_after == EXPECTED_HISTORICAL_FINAL,
                "REFRESH_HISTORICAL_AFTER", EXPECTED_HISTORICAL_FINAL,
                historical_after)
        require(historical_before == historical_after,
                "REFRESH_HISTORICAL_IMMUTABILITY", historical_before,
                historical_after)
        require(r6_before == r6_after,
                "REFRESH_R6_IMMUTABILITY", r6_before, r6_after)
        return result

    runtime_entrypoint._load_initial_model = wrapped_load
    runtime_entrypoint.run_alternating_training = wrapped_train
    runtime_entrypoint._final_prediction_state = wrapped_refresh
    return originals


def restore_instrumentation(originals):
    runtime_entrypoint._load_initial_model = originals["load"]
    runtime_entrypoint.run_alternating_training = originals["train"]
    runtime_entrypoint._final_prediction_state = originals["refresh"]


def verify_current_seal(sealed):
    require(isinstance(sealed, SealedPredictionPaths),
            "SEALED_PATHS_TYPE", "SealedPredictionPaths", type(sealed).__name__)
    for path in (sealed.bundle, sealed.audit, sealed.seal):
        require(path.is_file(), "CURRENT_SEALED_FILE_PRESENT", path, "missing")
    seal = read_json(sealed.seal)
    audit = read_json(sealed.audit)
    require(seal.get("schema") == "release-core-pre-gt-seal-v1",
            "CURRENT_SEAL_SCHEMA", "release-core-pre-gt-seal-v1",
            seal.get("schema"))
    require(file_sha256(sealed.bundle) == seal.get("bundle_sha256"),
            "CURRENT_BUNDLE_SHA", seal.get("bundle_sha256"),
            file_sha256(sealed.bundle))
    require(file_sha256(sealed.audit) == seal.get("audit_sha256"),
            "CURRENT_AUDIT_SHA", seal.get("audit_sha256"),
            file_sha256(sealed.audit))
    require(seal.get("full_gt_loaded_before_seal") is False,
            "FULL_GT_BEFORE_SEAL", False,
            seal.get("full_gt_loaded_before_seal"))
    require(seal.get("metrics_computed_before_seal") is False,
            "METRICS_BEFORE_SEAL", False,
            seal.get("metrics_computed_before_seal"))
    firewall = audit.get("gt_firewall", {})
    require(firewall.get("full_gt_loaded") is False,
            "AUDIT_FULL_GT_LOADED", False, firewall.get("full_gt_loaded"))
    require(firewall.get("metrics_computed") is False,
            "AUDIT_METRICS_COMPUTED", False, firewall.get("metrics_computed"))
    require(firewall.get("full_gt_present_in_bundle") is False,
            "AUDIT_FULL_GT_PRESENT_IN_BUNDLE", False,
            firewall.get("full_gt_present_in_bundle"))
    require(audit.get("initial_model_sha256") == EXPECTED_R6_INITIAL,
            "AUDIT_INITIAL_R6_HASH", EXPECTED_R6_INITIAL,
            audit.get("initial_model_sha256"))
    require(audit.get("prediction_logical_sha256") == EXPECTED_PREDICTION,
            "AUDIT_PREDICTION_SHA", EXPECTED_PREDICTION,
            audit.get("prediction_logical_sha256"))
    return audit, seal


def compare_scientific_payload(sealed):
    with np.load(sealed.bundle, allow_pickle=False) as current, np.load(
        REFERENCE_ARTIFACT, allow_pickle=False
    ) as reference:
        comparisons = {}
        for name in ("sample_ids", "final_predictions", "q_local", "q_aligned", "M_v"):
            current_value = np.asarray(current[name])
            reference_value = np.asarray(reference[name])
            equal = np.array_equal(current_value, reference_value)
            max_abs_diff = None
            if np.issubdtype(current_value.dtype, np.floating):
                max_abs_diff = float(np.max(np.abs(
                    current_value.astype(np.float64)
                    - reference_value.astype(np.float64)
                )))
            require(equal, name.upper() + "_EXACT", True,
                    {"equal": equal, "max_abs_diff": max_abs_diff})
            comparisons[name] = {
                "exact": equal,
                "max_abs_diff": max_abs_diff,
            }
        prediction_hash = payload_sha256(current["final_predictions"])
        require(prediction_hash == EXPECTED_PREDICTION,
                "PREDICTION_LOGICAL_SHA", EXPECTED_PREDICTION,
                prediction_hash)
        comparisons["prediction_logical_sha256"] = prediction_hash
        comparisons["current_final_r6_hash"] = str(current["final_model_sha256"].item())
    return comparisons


def main():
    verify_start_gate()
    adapter_sha, adapter_audit_sha = materialize_split_adapter()
    protocol_sha = freeze_bridge_protocol()
    historical_module = load_historical_hash_module()

    runtime = RuntimeConfig(
        dataset="Caltech-6V",
        training_seed=50,
        epochs=20,
        batch_size=256,
        learning_rate=1e-4,
        native_lambda1=0.01,
        refresh_interval=100,
        label_seed=20,
        labels_per_class=2,
        device="cuda:0",
    )
    expected_files = dict(EXPECTED_FILES)
    expected_files[SPLIT_ARTIFACT] = adapter_sha
    expected_files[SPLIT_AUDIT] = adapter_audit_sha
    provenance = ProvenanceConfig(
        feature_artifact=FEATURE_ARTIFACT,
        feature_audit=FEATURE_AUDIT,
        sparse_split_artifact=SPLIT_ARTIFACT,
        sparse_split_audit=SPLIT_AUDIT,
        utility_artifact=ACTION_ARTIFACT,
        utility_audit=ACTION_AUDIT,
        semantic_artifact=ACTION_ARTIFACT,
        semantic_audit=ACTION_AUDIT,
        checkpoint_paths=CHECKPOINTS,
        checkpoint_audit=CHECKPOINT_AUDIT,
        output_root=OUTPUT_ROOT,
        strict_replay=True,
        expected_file_sha256=tuple(expected_files.items()),
        expected_initial_model_sha256=EXPECTED_R6_INITIAL,
        expected_final_model_sha256=None,
        expected_prediction_sha256=EXPECTED_PREDICTION,
        historical_audit_sha256=EXPECTED_REFERENCE_FILES[REFERENCE_AUDIT],
        historical_seal_sha256=EXPECTED_REFERENCE_FILES[REFERENCE_SEAL],
    )

    evidence = {
        "load_original_call_count": 0,
        "train_original_call_count": 0,
        "refresh_original_call_count": 0,
        "r6_expected_final_model_sha256": None,
        "r6_expected_final_model_sha256_classification": (
            "unavailable: no pre-existing historical R6-namespace final state; "
            "replaced by mandatory historical canonical post-B exact gate"
        ),
    }
    originals = install_instrumentation(historical_module, evidence)
    try:
        sealed = run_pre_gt(runtime, provenance)
    finally:
        restore_instrumentation(originals)

    require(evidence["load_original_call_count"] == 1,
            "LOAD_ORIGINAL_CALL_COUNT", 1, evidence["load_original_call_count"])
    require(evidence["train_original_call_count"] == 1,
            "TRAIN_ORIGINAL_CALL_COUNT", 1, evidence["train_original_call_count"])
    require(evidence["refresh_original_call_count"] == 1,
            "REFRESH_ORIGINAL_CALL_COUNT", 1, evidence["refresh_original_call_count"])

    current_audit, current_seal = verify_current_seal(sealed)
    comparisons = compare_scientific_payload(sealed)

    # This is the first full-GT access by the harness, after complete seal validation.
    metrics = evaluate_postseal(sealed, FULL_GT)
    actual_metrics = {"acc": metrics.acc, "nmi": metrics.nmi, "ari": metrics.ari}
    metric_diffs = {}
    for name, expected in EXPECTED_METRICS.items():
        actual = actual_metrics[name]
        metric_diffs[name] = abs(actual - expected)
        require(actual == expected, "METRIC_" + name.upper(), expected, actual)

    final_status = git("status", "--short", "--untracked-files=all")
    final_diff = git("diff", "--name-only")
    final_cached = git("diff", "--cached", "--name-only")
    require(final_status == "", "FINAL_GIT_STATUS", "clean", final_status)
    require(final_diff == "", "FINAL_GIT_DIFF", "empty", final_diff)
    require(final_cached == "", "FINAL_GIT_CACHED_DIFF", "empty", final_cached)

    record = {
        "schema": "r6-a2-seed50-validation-v1",
        "result": "PASS",
        "runtime_config": {
            "dataset": "Caltech-6V",
            "training_seed": 50,
            "epochs": 20,
            "batch_size": 256,
            "learning_rate": 1e-4,
            "native_lambda1": 0.01,
            "refresh_interval": 100,
            "label_seed": 20,
            "labels_per_class": 2,
            "device": "cuda:0",
        },
        "hash_namespace_bridge_protocol": {
            "path": str(BRIDGE_PROTOCOL),
            "sha256": protocol_sha,
            "parent_seed20_bridge_freeze_manifest_sha256": (
                EXPECTED_BRIDGE_POLICY_FREEZE_MANIFEST_SHA
            ),
            "parent_seed30_replay_freeze_manifest_sha256": (
                EXPECTED_PREVIOUS_REPLAY_FREEZE_MANIFEST_SHA
            ),
        },
        "instrumentation": evidence,
        "scientific_payload": comparisons,
        "current_pre_gt": {
            "bundle": str(sealed.bundle),
            "bundle_sha256": file_sha256(sealed.bundle),
            "audit": str(sealed.audit),
            "audit_sha256": file_sha256(sealed.audit),
            "seal": str(sealed.seal),
            "seal_sha256": file_sha256(sealed.seal),
            "bundle_valid": True,
            "audit_valid": True,
            "seal_valid": True,
            "full_gt_loaded_before_seal": False,
            "r6_final_model_sha256": current_audit["final_model_sha256"],
        },
        "metrics": {
            name: {
                "expected": EXPECTED_METRICS[name],
                "actual": actual_metrics[name],
                "abs_diff": metric_diffs[name],
                "exact": True,
            }
            for name in ("acc", "nmi", "ari")
        },
        "historical_whole_file_sha_policy": {
            "audit": "reference only",
            "seal": "reference only",
        },
        "repository_unchanged": True,
        "seed20_rerun": False,
        "seed30_rerun": False,
        "seed50_run": True,
    }
    require(current_seal.get("prediction_logical_sha256") == EXPECTED_PREDICTION,
            "CURRENT_SEAL_PREDICTION_SHA", EXPECTED_PREDICTION,
            current_seal.get("prediction_logical_sha256"))
    write_json_exclusive(VALIDATION_RECORD, record)

    print("""
R6-A2 SEED50 END-TO-END FROZEN REPLAY:
PASS

INITIAL HISTORICAL HASH:
EXACT

INITIAL R6 HASH:
EXACT

FINAL POST-B HISTORICAL HASH:
EXACT

FINAL R6 HASH:
RUN-LOCAL INTEGRITY ONLY

FINAL REFRESH MODEL IMMUTABILITY:
EXACT IN BOTH HASH NAMESPACES

SAMPLE IDS:
EXACT

FINAL PREDICTIONS:
EXACT

Q_LOCAL:
EXACT / MAX_ABS_DIFF=0.0

Q_ALIGNED:
EXACT / MAX_ABS_DIFF=0.0

M_V:
EXACT / MAX_ABS_DIFF=0.0

PREDICTION LOGICAL SHA:
EXACT

CURRENT PRE-GT SEAL:
VALID

FULL GT BEFORE SEAL:
NO

ACC:
0.8607142857142858 / EXACT

NMI:
0.7763024116674858 / EXACT

ARI:
0.7538086225874288 / EXACT

REPOSITORY:
UNCHANGED

SEED50:
PASS / READY FOR INDEPENDENT REVIEW

DO NOT COMMIT.
DO NOT PUSH.
DO NOT TAG.

WAIT FOR INDEPENDENT REVIEW.""".strip())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("R6-A2 SEED50: FAIL-CLOSED", file=sys.stderr)
        print(str(error), file=sys.stderr)
        raise
