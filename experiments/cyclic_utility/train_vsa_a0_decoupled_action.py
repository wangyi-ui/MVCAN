"""Run one VSA arm with semantic action before native consolidation.

The runner supports the one-epoch VSA-E0 engineering identity and the later
twenty-epoch VSA-A0 scientific pilot.  It never combines the two phase losses
or shares optimizer state between phases.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for thread_environment in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[thread_environment] = "1"

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.cyclic_utility import (
    c1_frozen_pseudo_supervision as c1,
)
from experiments.cyclic_utility import (
    train_c1_frozen_pseudo_supervision as c1_train,
)
from experiments.cyclic_utility import (
    vsa_decoupled_action_protocol as vsa,
)
from experiments.e1_pairwise_utility import (
    train_e1_pairwise_utility as e1_train,
)
from irv.b3_audit import hash_backbone
from irv.b4_information_utility import tensor_sha256


DATASET = "Caltech-6V"
WEAK_QUALITY_CONDITION = e1_train.WEAK_QUALITY_CONDITION
RUN_KINDS = ("E0", "A0")
DEFAULT_C0_ARTIFACT_PATH = c1_train.DEFAULT_C0_ARTIFACT_PATH
DEFAULT_C0_SEAL_PATH = c1_train.DEFAULT_C0_SEAL_PATH
DEFAULT_FEATURE_PATH = c1_train.DEFAULT_FEATURE_PATH
DEFAULT_FEATURE_AUDIT_PATH = c1_train.DEFAULT_FEATURE_AUDIT_PATH
DEFAULT_E1_LWC_MODEL_DIR = c1_train.DEFAULT_E1_LWC_MODEL_DIR
DEFAULT_E1_LWC_AUDIT_PATH = c1_train.DEFAULT_E1_LWC_AUDIT_PATH
DEFAULT_FULL_GT_PATH = c1_train.DEFAULT_FULL_GT_PATH
DEFAULT_E0_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "outputs/cyclic_utility/vsa_e0_identity_seed20"
)
DEFAULT_A0_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "outputs/cyclic_utility/vsa_a0_decoupled_action_seed20"
)

FROZEN_C1_SOURCE_SHA256 = {
    "experiments/cyclic_utility/c1_frozen_pseudo_supervision.py": (
        "7a533bfbbc71241d8db96fad435b573c3491b0ade741f832f284846192a41374"
    ),
    "experiments/cyclic_utility/train_c1_frozen_pseudo_supervision.py": (
        "6f3e0649901f4cba1b9f3ed94079c7841d3ee9d20a8c76eaa359d319bcd7dd80"
    ),
    "experiments/cyclic_utility/summarize_c1_frozen_pseudo_supervision.py": (
        "0682dae0bdbb24a90d3585478e3ea30417f31d31c41bff14088d462fecbb672a"
    ),
    "experiments/cyclic_utility/c1_p0_action_compression_audit.py": (
        "5f5f3039b897b8555bfa3a87901caa47a5f598a9d5fe3ba00de7a524b86b7fb9"
    ),
    "experiments/cyclic_utility/evaluate_c1_p0_action_compression_audit.py": (
        "27db837a93045d0b9c5b55d77acecb72e17e9ea548f26bedf21c4253bccecea9"
    ),
}
FROZEN_PARAMETERIZED_E1_SOURCE_SHA256 = (
    "ad6dcb187ccb39d36f5f5839130b0345053b23d92df1c00d9db98d41f8b1ca73"
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


def verify_frozen_source_provenance(repository_root=REPOSITORY_ROOT):
    """Verify frozen upstream mechanics, including parameterized E1."""
    root = Path(repository_root)
    inherited_actual = {}
    inherited_expected = dict(c1_train.FROZEN_SOURCE_SHA256)
    inherited_expected[
        "experiments/e1_pairwise_utility/train_e1_pairwise_utility.py"
    ] = FROZEN_PARAMETERIZED_E1_SOURCE_SHA256
    for relative_path, expected_hash in inherited_expected.items():
        path = root / relative_path
        _require(path.is_file(), "frozen source is missing: " + relative_path)
        actual_hash = c1.file_sha256(path)
        _require(
            actual_hash == expected_hash,
            "frozen source provenance mismatch: " + relative_path,
        )
        inherited_actual[relative_path] = actual_hash
    inherited = {
        "expected_sha256": inherited_expected,
        "actual_sha256": inherited_actual,
        "all_frozen_sources_unchanged_pass": True,
    }
    actual = {}
    for relative_path, expected_hash in FROZEN_C1_SOURCE_SHA256.items():
        path = root / relative_path
        _require(path.is_file(), "frozen C1 source missing: " + relative_path)
        actual_hash = c1.file_sha256(path)
        _require(
            actual_hash == expected_hash,
            "frozen C1 source provenance mismatch: " + relative_path,
        )
        actual[relative_path] = actual_hash
    return {
        "inherited_C0_E1_E4_provenance": inherited,
        "expected_C1_sha256": dict(FROZEN_C1_SOURCE_SHA256),
        "actual_C1_sha256": actual,
        "parameterized_E1_source_sha256": (
            FROZEN_PARAMETERIZED_E1_SOURCE_SHA256
        ),
        "C1_helpers_reused_read_only_pass": True,
        "C1_P0_unchanged_pass": True,
        "all_frozen_sources_unchanged_pass": True,
    }


def _same_resolved_path(left, right):
    return _resolve(left).resolve() == _resolve(right).resolve()


def _same_resolved_path_lists(left, right):
    return tuple(_resolve(path).resolve() for path in left) == tuple(
        _resolve(path).resolve() for path in right
    )


def _audit_training_seed(audit):
    seed = audit.get("training_seed", audit.get("seed"))
    _require(seed is not None, "E1 audit training seed is missing")
    seed = vsa.validate_training_seed(seed)
    _require(
        int(audit.get("seed", seed)) == seed,
        "E1 audit seed fields disagree",
    )
    return seed


def validate_fixed_feature_realization(
    feature_path, feature_audit_path, feature_provenance
):
    """Keep the D1/E0 weak-quality realization fixed at seed20."""
    canonical_feature = _resolve(DEFAULT_FEATURE_PATH).resolve()
    canonical_audit = _resolve(DEFAULT_FEATURE_AUDIT_PATH).resolve()
    supplied_feature = _resolve(feature_path).resolve()
    supplied_audit = _resolve(feature_audit_path).resolve()
    _require(
        supplied_feature == canonical_feature
        and supplied_audit == canonical_audit,
        "VSA weak-quality feature realization must remain seed20",
    )
    actual_hash = c1.file_sha256(supplied_feature)
    _require(
        feature_provenance.get("file_sha256") == actual_hash
        and feature_provenance.get("corruption_mask_present") is False
        and feature_provenance.get("labels_present") is False,
        "VSA fixed feature provenance mismatch",
    )
    return {
        "weak_quality_condition_fixed": WEAK_QUALITY_CONDITION,
        "feature_realization_path": _display(supplied_feature),
        "feature_realization_file_sha256": actual_hash,
        "feature_audit_path": _display(supplied_audit),
        "feature_realization_unchanged_pass": True,
        "training_seed_changes_weak_quality_realization": False,
        "corruption_mask_loaded": False,
    }


def load_lineage_aware_e1_lwc_model(
    model_dir, model_audit_path, device, training_seed
):
    """Load E1_s and bind actual checkpoints to that same E1 audit."""
    requested_seed = vsa.validate_training_seed(training_seed)
    audit_path = _resolve(model_audit_path)
    checkpoint_dir = _resolve(model_dir)
    _require(audit_path.is_file(), "E1 LWC audit is missing")
    _require(checkpoint_dir.is_dir(), "E1 LWC model directory is missing")
    audit = c1.read_json(audit_path)
    audit_seed = _audit_training_seed(audit)
    _require(
        audit.get("stage") == "E1"
        and audit.get("arm") == "LWC"
        and audit.get("epochs") == 100
        and audit.get("N") == vsa.SAMPLE_NUM
        and audit.get("V") == vsa.VIEW_NUM
        and audit.get("K") == vsa.CLASS_NUM,
        "E1 audit is not a formal 100-epoch LWC run",
    )
    _require(
        audit_seed == requested_seed,
        "requested VSA seed and E1 audit training seed mismatch",
    )
    model, config, checkpoint_audit = (
        e1_train.build_model_from_frozen_d1(
            checkpoint_dir,
            device,
            training_seed=requested_seed,
        )
    )
    actual_model_hash = hash_backbone(model.autoencoders)
    expected_model_hash = audit.get("final_model_hash")
    expected_outputs = audit.get("final_model_outputs", {})
    expected_checkpoint_hashes = expected_outputs.get("file_sha256")
    expected_checkpoint_paths = expected_outputs.get("paths")
    _require(
        isinstance(expected_model_hash, dict)
        and actual_model_hash == expected_model_hash,
        "actual E1 model hash does not match its own audit",
    )
    _require(
        checkpoint_audit["checkpoint_file_sha256"]
        == expected_checkpoint_hashes
        and isinstance(expected_checkpoint_paths, list)
        and _same_resolved_path_lists(
            checkpoint_audit["checkpoint_paths"],
            expected_checkpoint_paths,
        )
        and all(
            _resolve(path).resolve().parent == checkpoint_dir.resolve()
            for path in expected_checkpoint_paths
        ),
        "actual E1 checkpoint files do not match their own audit",
    )
    audited_feature = audit.get("feature_provenance", {})
    _require(
        audited_feature.get("path") == _display(DEFAULT_FEATURE_PATH)
        and audited_feature.get("file_sha256")
        == c1.file_sha256(_resolve(DEFAULT_FEATURE_PATH))
        and audited_feature.get("corruption_mask_present") is False,
        "E1 audit does not use the fixed seed20 feature realization",
    )
    if requested_seed != vsa.SEED:
        semantics = audit.get("seed_semantics", {})
        _require(
            semantics.get("training_seed") == requested_seed
            and semantics.get("weak_quality_condition_fixed")
            == WEAK_QUALITY_CONDITION
            and semantics.get("weak_quality_realization_varied") is False,
            "E1 multi-seed audit weak-quality semantics mismatch",
        )
    seed20_compatibility = requested_seed != vsa.SEED or (
        actual_model_hash["aggregate"]
        == c1.EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256
        and tuple(checkpoint_audit["checkpoint_file_sha256"])
        == c1.EXPECTED_E1_CHECKPOINT_SHA256
    )
    _require(seed20_compatibility, "seed20 E1 historical provenance mismatch")
    return model, config, {
        **checkpoint_audit,
        "source_stage": "E1",
        "source_arm": "LWC",
        "source_epochs": 100,
        "source_seed": audit_seed,
        "source_training_seed": audit_seed,
        "source_audit_path": _display(audit_path),
        "source_audit_file_sha256": c1.file_sha256(audit_path),
        "actual_loaded_model_hash": actual_model_hash,
        "actual_loaded_aggregate_sha256": actual_model_hash["aggregate"],
        "expected_model_hash_from_own_audit": expected_model_hash,
        "expected_aggregate_sha256_from_own_audit": (
            expected_model_hash["aggregate"]
        ),
        "model_matches_own_audit_pass": True,
        "requested_seed_lineage_match_pass": True,
        "seed20_historical_hash_pass": seed20_compatibility,
        "weak_quality_condition_fixed": WEAK_QUALITY_CONDITION,
        "independently_reloaded_for_this_arm": True,
    }


def load_lineage_aware_c0_artifact(
    artifact_path,
    seal_path,
    training_seed,
    e1_provenance,
    feature_provenance,
):
    """Load C0_s against its own seal/audit and bind it to E1_s."""
    requested_seed = vsa.validate_training_seed(training_seed)
    artifact = _resolve(artifact_path)
    seal_file = _resolve(seal_path)
    audit_path = seal_file.with_name("c0_audit.json")
    _require(
        artifact.is_file() and seal_file.is_file() and audit_path.is_file(),
        "C0 artifact/seal/audit is missing",
    )
    artifact_sha = c1.file_sha256(artifact)
    seal_sha = c1.file_sha256(seal_file)
    seal = c1.read_json(seal_file)
    audit = c1.read_json(audit_path)
    _require(
        seal.get("stage") == "C0"
        and seal.get("artifact_file_sha256") == artifact_sha
        and _same_resolved_path(seal.get("artifact_path", ""), artifact)
        and seal.get("scores_completed_before_GT") is True
        and seal.get("scores_saved_before_GT") is True
        and seal.get("scores_hashed_before_GT") is True
        and seal.get("scores_reloaded_before_GT") is True
        and seal.get("full_GT_loaded_before_seal") is False
        and seal.get("sparse_labels_loaded_before_seal") is False
        and seal.get("R_loaded_before_seal") is False
        and seal.get("corruption_mask_loaded_before_seal") is False
        and seal.get("oracle_loaded_before_seal") is False,
        "C0 own pre-GT seal boundary mismatch",
    )
    _require(
        audit.get("stage") == "C0"
        and audit.get("C0_AUDIT_PASS") is True
        and audit.get("N") == vsa.SAMPLE_NUM
        and audit.get("V") == vsa.VIEW_NUM
        and audit.get("K") == vsa.CLASS_NUM
        and audit.get("prediction_seal") == seal,
        "C0 audit does not bind the provided own seal",
    )
    arrays = {}
    expected_shapes = {
        "y_gen": (vsa.SAMPLE_NUM, c1.DIRECTION_COUNT),
        "C_conf": (vsa.SAMPLE_NUM, c1.DIRECTION_COUNT),
        "U_cycle": (vsa.SAMPLE_NUM, c1.DIRECTION_COUNT),
        "U_cycle_shuffle": (vsa.SAMPLE_NUM, c1.DIRECTION_COUNT),
        "native_global_cluster": (vsa.SAMPLE_NUM,),
        "generator_subsets": (c1.DIRECTION_COUNT, 3),
        "verifier_subsets": (c1.DIRECTION_COUNT, 3),
    }
    with np.load(artifact, allow_pickle=False) as archive:
        archive_fields = tuple(archive.files)
        _require(
            set(c1.REQUIRED_C0_ARRAYS).issubset(archive_fields),
            "C0 artifact is missing a required pre-GT array",
        )
        for name in c1.REQUIRED_C0_ARRAYS:
            value = np.array(archive[name], copy=True, order="C")
            own_record = seal.get("arrays", {}).get(name, {})
            _require(
                tuple(value.shape) == expected_shapes[name]
                and own_record.get("shape") == list(value.shape)
                and own_record.get("dtype") == str(value.dtype)
                and own_record.get("logical_sha256")
                == tensor_sha256(value),
                "C0 own-seal array mismatch: " + name,
            )
            arrays[name] = c1.frozen_tensor(value)
    y_gen = arrays["y_gen"]
    native_global = arrays["native_global_cluster"]
    _require(
        y_gen.dtype == torch.long
        and bool(torch.all((y_gen >= 0) & (y_gen < vsa.CLASS_NUM)).item()),
        "C0 y_gen boundary mismatch",
    )
    for name in ("C_conf", "U_cycle", "U_cycle_shuffle"):
        values = arrays[name]
        _require(
            values.dtype == torch.float32
            and bool(torch.isfinite(values).all().item())
            and bool(torch.all((values >= 0.0) & (values <= 1.0)).item()),
            "C0 score boundary mismatch: " + name,
        )
    _require(
        native_global.dtype == torch.long
        and bool(torch.all(
            (native_global >= 0) & (native_global < vsa.CLASS_NUM)
        ).item())
        and torch.equal(
            torch.unique(native_global).sort().values,
            torch.arange(vsa.CLASS_NUM),
        ),
        "C0 native global coordinate boundary mismatch",
    )
    generator_subsets = arrays["generator_subsets"]
    verifier_subsets = arrays["verifier_subsets"]
    _require(
        generator_subsets.dtype == verifier_subsets.dtype == torch.long
        and bool(torch.all(
            (generator_subsets >= 0)
            & (generator_subsets < vsa.VIEW_NUM)
        ).item())
        and bool(torch.all(
            (verifier_subsets >= 0)
            & (verifier_subsets < vsa.VIEW_NUM)
        ).item())
        and all(
            sorted(
                generator_subsets[index].tolist()
                + verifier_subsets[index].tolist()
            ) == list(range(vsa.VIEW_NUM))
            for index in range(c1.DIRECTION_COUNT)
        ),
        "C0 generator/verifier subset boundary mismatch",
    )
    _require(
        all(
            not value.requires_grad and value.grad_fn is None
            for value in arrays.values()
        ),
        "C0 array did not remain frozen",
    )
    representation = audit.get("representation", {})
    checkpoint = representation.get("checkpoint", {})
    expected_e1_hash = e1_provenance[
        "expected_model_hash_from_own_audit"
    ]
    c0_parameter_before = representation.get("parameter_hash_before")
    c0_parameter_after = representation.get("parameter_hash_after")
    _require(
        representation.get("source_stage") == "E1"
        and representation.get("source_arm") == "LWC"
        and representation.get("source_epochs") == 100
        and checkpoint.get("arm") == "LWC"
        and checkpoint.get("epochs") == 100
        and _same_resolved_path(
            checkpoint.get("audit_path", ""),
            e1_provenance["source_audit_path"],
        )
        and checkpoint.get("checkpoint_file_sha256")
        == e1_provenance["checkpoint_file_sha256"]
        and _same_resolved_path_lists(
            checkpoint.get("checkpoint_paths", []),
            e1_provenance["checkpoint_paths"],
        )
        and c0_parameter_before == expected_e1_hash
        and c0_parameter_after == expected_e1_hash
        and representation.get("parameters_unchanged_pass") is True,
        "C0 and E1 model lineage mismatch",
    )
    c0_feature = representation.get("feature", {})
    _require(
        _same_resolved_path(
            c0_feature.get("path", ""), feature_provenance["path"]
        )
        and c0_feature.get("file_sha256")
        == feature_provenance["file_sha256"]
        and c0_feature.get("view_content_sha256")
        == feature_provenance["view_content_sha256"]
        and c0_feature.get("corruption_mask_present") is False,
        "C0 feature realization lineage mismatch",
    )
    m_record = seal.get("M_provenance", {})
    _require(
        m_record == representation.get("M")
        and m_record.get("shape")
        == [vsa.VIEW_NUM, vsa.CLASS_NUM, vsa.CLASS_NUM]
        and m_record.get("full_permutation_pass") is True
        and m_record.get("row_sums_one_pass") is True
        and m_record.get("column_sums_one_pass") is True,
        "C0 M provenance mismatch",
    )
    seed20_compatibility = requested_seed != vsa.SEED or (
        artifact_sha == c1.EXPECTED_C0_ARTIFACT_SHA256
        and seal_sha == c1.EXPECTED_C0_SEAL_SHA256
        and all(
            seal["arrays"][name]["logical_sha256"]
            == c1.EXPECTED_C0_ARRAY_SHA256[name]
            for name in c1.REQUIRED_C0_ARRAYS
        )
    )
    _require(seed20_compatibility, "seed20 C0 historical provenance mismatch")
    array_hashes = {
        name: seal["arrays"][name]["logical_sha256"]
        for name in c1.REQUIRED_C0_ARRAYS
    }
    return arrays, {
        "artifact_path": _display(artifact),
        "artifact_file_sha256": artifact_sha,
        "artifact_file_sha256_pass": True,
        "seal_path": _display(seal_file),
        "seal_file_sha256": seal_sha,
        "seal_file_sha256_pass": True,
        "audit_path": _display(audit_path),
        "audit_file_sha256": c1.file_sha256(audit_path),
        "own_seal_audit_validation_pass": True,
        "archive_fields": list(archive_fields),
        "loaded_fields": list(c1.REQUIRED_C0_ARRAYS),
        "array_logical_sha256": array_hashes,
        "M0_logical_sha256_from_C0_seal": m_record["logical_sha256"],
        "C0_source_model_hash": c0_parameter_before,
        "C0_source_E1_audit_path": checkpoint["audit_path"],
        "C0_source_E1_checkpoint_file_sha256": checkpoint[
            "checkpoint_file_sha256"
        ],
        "authoritative_training_seed_from_E1_audit": requested_seed,
        "legacy_C0_seed_field": seal.get("seed"),
        "legacy_C0_seed_field_used_as_authority": False,
        "C0_E1_lineage_match_pass": True,
        "requested_seed_lineage_match_pass": True,
        "seed20_historical_hashes_pass": seed20_compatibility,
        "loaded_pre_GT_arrays_only": True,
        "GT_derived_correctness_loaded": False,
        "R_loaded": False,
        "sparse_labels_loaded": False,
        "corruption_mask_loaded": False,
        "oracle_loaded": False,
    }


@torch.no_grad()
def derive_seed_specific_M0(
    model,
    views,
    sealed_global_ids,
    device,
    training_seed,
    c0_provenance,
):
    """Apply the frozen C1 alignment formula with seed-specific seals."""
    requested_seed = vsa.validate_training_seed(training_seed)
    target_device = torch.device(device)
    canonical_global_ids = c1.frozen_tensor(
        sealed_global_ids, dtype=torch.long, device="cpu"
    )
    canonical_global_sha = tensor_sha256(canonical_global_ids.numpy())
    _require(
        canonical_global_ids.shape == (vsa.SAMPLE_NUM,)
        and canonical_global_sha
        == c0_provenance["array_logical_sha256"][
            "native_global_cluster"
        ],
        "seed-specific C0 native coordinate provenance mismatch",
    )
    prior_modes = [
        autoencoder.training for autoencoder in model.autoencoders
    ]
    for autoencoder in model.autoencoders:
        autoencoder.eval()
    q_local_views = []
    for autoencoder, view in zip(model.autoencoders, views):
        features = c1.frozen_tensor(
            view, dtype=torch.float32, device=target_device
        )
        raw_z = autoencoder.encoder(features).detach()
        q_local_views.append(autoencoder.clustering(raw_z).detach().cpu())
    q_local_stack = torch.stack(q_local_views, dim=1).detach()
    _require(
        q_local_stack.shape
        == (vsa.SAMPLE_NUM, vsa.VIEW_NUM, vsa.CLASS_NUM),
        "seed-specific q_local shape mismatch for M0 alignment",
    )
    alignment = c1.g2_protocol.build_alignment(
        q_local_stack.numpy(),
        canonical_global_ids.numpy(),
        model.Match,
    )
    alignment_int64 = np.array(
        alignment["alignment_matrix"],
        dtype=np.int64,
        copy=True,
        order="C",
    )
    M0_int64 = c1.validate_M0(alignment_int64)
    alignment_sha = tensor_sha256(alignment_int64)
    _require(
        alignment_sha
        == c0_provenance["M0_logical_sha256_from_C0_seal"],
        "seed-specific M0 does not conserve its C0 alignment seal",
    )
    if requested_seed == vsa.SEED:
        _require(
            alignment_sha == c1.EXPECTED_M0_LOGICAL_SHA256,
            "seed20 historical M0 hash mismatch",
        )
    M0 = torch.as_tensor(
        np.array(alignment_int64, copy=True, order="C"),
        dtype=torch.float32,
    ).detach()
    M0 = c1.validate_M0(M0)
    exact_cast = torch.equal(M0, M0_int64.to(dtype=torch.float32))
    _require(exact_cast, "M0 is not an exact float32 0/1 cast")
    for autoencoder, was_training in zip(
        model.autoencoders, prior_modes
    ):
        autoencoder.train(was_training)
    return M0.detach(), {
        "training_seed": requested_seed,
        "shape": list(M0.shape),
        "logical_sha256": alignment_sha,
        "seed_specific_logical_sha256": alignment_sha,
        "matches_C0_seal_pass": True,
        "alignment_conservation_pass": True,
        "canonical_global_coordinate_source": (
            "C0_PRE_GT_sealed_native_global_cluster"
        ),
        "sealed_native_global_cluster_logical_sha256": (
            canonical_global_sha
        ),
        "alignment_int64_shape": list(alignment_int64.shape),
        "alignment_int64_logical_sha256": alignment_sha,
        "training_M0_dtype": str(M0.numpy().dtype),
        "training_M0_logical_sha256": tensor_sha256(M0.numpy()),
        "training_M0_is_exact_01_cast_of_int64_pass": exact_cast,
        "full_permutation_pass": True,
        "row_sums_one_pass": True,
        "column_sums_one_pass": True,
        "detached_pass": True,
        "derived_before_training": True,
        "frozen_throughout_VSA": True,
        "q_local_argmax_used_for_alignment": True,
        "Match_input_direction": "Match(local_ids, global_ids)",
        "GT_used_for_M0": False,
        "sparse_labels_used_for_M0": False,
        "corruption_mask_used_for_M0": False,
        "native_global_cluster_used_for_M0_coordinate_only": True,
        "local_to_global_direction_documented": "q_local @ M0.T",
        "global_to_local_direction": "T_global @ M0",
        "C0_scores_recomputed": False,
        "seed20_historical_hash_pass": (
            requested_seed != vsa.SEED
            or alignment_sha == c1.EXPECTED_M0_LOGICAL_SHA256
        ),
    }


def enable_strict_determinism(seed=vsa.SEED):
    """Enable bitwise CUDA replay required by the VSA-E0 identity gate."""
    c1.set_deterministic_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_TF32_allowed": (
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_TF32_allowed": torch.backends.cudnn.allow_tf32,
        "bitwise_E0_replay_control_enabled": True,
    }


def validate_run_contract(run_kind, arm, epochs):
    if run_kind not in RUN_KINDS:
        raise ValueError("VSA run_kind must be E0 or A0")
    if arm not in vsa.ALL_ARMS:
        raise ValueError("unknown VSA arm")
    if run_kind == "E0":
        _require(
            arm in ("BASE", "CYCLE_ZERO")
            and int(epochs) == vsa.ENGINEERING_EPOCHS,
            "VSA-E0 permits only BASE/CYCLE_ZERO for exactly one epoch",
        )
    else:
        _require(
            arm in vsa.FORMAL_ARMS
            and int(epochs) == vsa.FORMAL_EPOCHS,
            "VSA-A0 permits only five formal arms for exactly 20 epochs",
        )


def _batch_tensor_ids(batch_ids):
    return torch.as_tensor(
        np.asarray(batch_ids, dtype=np.int64), dtype=torch.long
    )


def _semantic_phase(
    model,
    semantic_optimizers,
    full_views,
    semantic_target,
    order,
    device,
):
    visited = []
    loss_sum = 0.0
    batch_records = []
    for batch_index, batch_ids_numpy in enumerate(vsa.ordered_batches(order)):
        batch_ids = _batch_tensor_ids(batch_ids_numpy)
        visited.append(np.asarray(batch_ids_numpy, dtype=np.int64))
        vsa.zero_optimizer_gradients(semantic_optimizers)
        _require(
            vsa.all_parameter_gradients_cleared(model),
            "semantic gradients were not zeroed before batch",
        )
        q_views = []
        for view_id in range(vsa.VIEW_NUM):
            x_view = full_views[view_id][batch_ids].to(device)
            latent = model.autoencoders[view_id].encoder(x_view)
            q_views.append(
                model.autoencoders[view_id].clustering(latent)
            )
        q_local = torch.stack(q_views, dim=1)
        target_batch, strength_batch = (
            vsa.semantic_batch_by_sample_ids(
                semantic_target, batch_ids
            )
        )
        _require(
            not target_batch.requires_grad
            and target_batch.grad_fn is None
            and not strength_batch.requires_grad
            and strength_batch.grad_fn is None,
            "semantic target batch is not detached",
        )
        semantic_loss = vsa.semantic_only_loss(
            q_local, target_batch, strength_batch
        )
        _require(
            bool(torch.isfinite(semantic_loss).item()),
            "non-finite VSA semantic loss",
        )
        semantic_loss.backward()
        gradient = vsa.parameter_gradient_audit(model)
        _require(
            gradient["gradient_finite_pass"],
            "non-finite VSA semantic gradient",
        )
        for semantic_optimizer in semantic_optimizers:
            semantic_optimizer.step()
        loss_value = float(semantic_loss.detach().item())
        loss_sum += loss_value
        batch_records.append({
            "batch": batch_index + 1,
            "batch_size": int(batch_ids.numel()),
            "semantic_loss": loss_value,
            "gradient": gradient,
        })
    vsa.zero_optimizer_gradients(semantic_optimizers)
    _require(
        vsa.all_parameter_gradients_cleared(model),
        "semantic gradients survived the semantic/native transition",
    )
    coverage = c1_train.coverage_record(visited)
    return {
        "semantic_phase_used": True,
        "batch_count": len(batch_records),
        "batch_boundaries": [
            record["batch_size"] for record in batch_records
        ],
        "semantic_loss_mean": loss_sum / len(batch_records),
        "semantic_loss_finite_pass": True,
        "semantic_gradient_finite_pass": all(
            record["gradient"]["gradient_finite_pass"]
            for record in batch_records
        ),
        "semantic_gradient_nonzero_any": any(
            record["gradient"]["gradient_nonzero"]
            for record in batch_records
        ),
        "pseudo_tensors_detached_pass": True,
        "gradients_zeroed_before_every_batch": True,
        "gradients_cleared_after_phase": True,
        "batch_records": batch_records,
        **coverage,
    }


def _native_phase(
    model,
    native_optimizers,
    full_views,
    native_p_all,
    native_matches,
    order,
    device,
):
    visited = []
    reconstruction_sum = 0.0
    clustering_sum = 0.0
    lwc_sum = 0.0
    native_sum = 0.0
    batch_records = []
    for batch_index, batch_ids_numpy in enumerate(vsa.ordered_batches(order)):
        batch_ids = _batch_tensor_ids(batch_ids_numpy)
        visited.append(np.asarray(batch_ids_numpy, dtype=np.int64))
        vsa.zero_optimizer_gradients(native_optimizers)
        _require(
            vsa.all_parameter_gradients_cleared(model),
            "native gradients were not zeroed before batch",
        )
        x_views = [
            full_views[view_id][batch_ids].to(device)
            for view_id in range(vsa.VIEW_NUM)
        ]
        p_batch = native_p_all[batch_ids].to(device)
        reconstructions = []
        q_views = []
        for view_id in range(vsa.VIEW_NUM):
            x_hat, _, q_local_view = model.autoencoders[view_id](
                x_views[view_id]
            )
            reconstructions.append(x_hat)
            q_views.append(q_local_view)
        q_local = torch.stack(q_views, dim=1)
        native_view_losses, native_diagnostics = (
            e1_train.native_mvcan_losses(
                x_views,
                reconstructions,
                q_views,
                p_batch,
                native_matches,
            )
        )
        reconstruction_loss = torch.stack(
            [record[0] for record in native_diagnostics]
        ).sum()
        clustering_loss = torch.stack(
            [record[1] for record in native_diagnostics]
        ).sum()
        rec_clu_loss = torch.stack(native_view_losses).sum()
        lwc_loss, lwc_diagnostics = c1_train.build_lwc_loss(
            q_local, native_matches
        )
        _require(
            all(record["r_mean"] is None for record in lwc_diagnostics),
            "native LWC unexpectedly consumed R",
        )
        native_loss = rec_clu_loss + e1_train.LAMBDA1 * lwc_loss
        _require(
            bool(torch.isfinite(native_loss).item()),
            "non-finite VSA native loss",
        )
        native_loss.backward()
        gradient = vsa.parameter_gradient_audit(model)
        _require(
            gradient["gradient_finite_pass"],
            "non-finite VSA native gradient",
        )
        for native_optimizer in native_optimizers:
            native_optimizer.step()
        values = {
            "reconstruction_loss_sum_views": float(
                reconstruction_loss.detach().item()
            ),
            "clustering_loss_sum_views": float(
                clustering_loss.detach().item()
            ),
            "lwc_loss": float(lwc_loss.detach().item()),
            "native_loss": float(native_loss.detach().item()),
        }
        reconstruction_sum += values["reconstruction_loss_sum_views"]
        clustering_sum += values["clustering_loss_sum_views"]
        lwc_sum += values["lwc_loss"]
        native_sum += values["native_loss"]
        batch_records.append({
            "batch": batch_index + 1,
            "batch_size": int(batch_ids.numel()),
            **values,
            "gradient": gradient,
        })
    vsa.zero_optimizer_gradients(native_optimizers)
    _require(
        vsa.all_parameter_gradients_cleared(model),
        "native gradients survived the end of the phase",
    )
    coverage = c1_train.coverage_record(visited)
    batch_count = len(batch_records)
    return {
        "native_phase_used": True,
        "batch_count": batch_count,
        "batch_boundaries": [
            record["batch_size"] for record in batch_records
        ],
        "reconstruction_loss_sum_views_mean": (
            reconstruction_sum / batch_count
        ),
        "clustering_loss_sum_views_mean": clustering_sum / batch_count,
        "lwc_loss_mean": lwc_sum / batch_count,
        "native_loss_mean": native_sum / batch_count,
        "native_loss_finite_pass": True,
        "native_gradient_finite_pass": all(
            record["gradient"]["gradient_finite_pass"]
            for record in batch_records
        ),
        "gradients_zeroed_before_every_batch": True,
        "gradients_cleared_after_phase": True,
        "batch_records": batch_records,
        **coverage,
    }


def train_vsa_arm(
    arm,
    epochs,
    model,
    semantic_optimizers,
    native_optimizers,
    views,
    sample_ids,
    semantic_target,
    orders,
    device,
    batch_size=vsa.BATCH_SIZE,
    training_seed=vsa.SEED,
):
    """Run Phase A completely, then Phase B completely, each epoch."""
    training_seed = vsa.validate_training_seed(training_seed)
    if arm not in vsa.ALL_ARMS:
        raise ValueError("unknown VSA arm")
    if int(batch_size) != vsa.BATCH_SIZE:
        raise ValueError("VSA batch size is frozen at 256")
    _require(
        (arm == "BASE" and semantic_target is None)
        or (arm != "BASE" and semantic_target is not None),
        "VSA arm/semantic-target boundary mismatch",
    )
    _require(
        len(orders["semantic_orders"])
        == len(orders["native_orders"])
        == int(epochs),
        "VSA precomputed order count mismatch",
    )
    full_views = [torch.from_numpy(view) for view in views]
    _require(
        np.array_equal(
            np.asarray(sample_ids, dtype=np.int64),
            np.arange(vsa.SAMPLE_NUM, dtype=np.int64),
        ),
        "VSA training sample IDs are not canonical",
    )
    view_weights = [1.0] * vsa.VIEW_NUM
    native_p_all = None
    native_matches = None
    native_target_refresh_count = 0
    semantic_epoch_records = []
    native_epoch_records = []
    semantic_hash_records = []
    native_hash_records = []
    phase_sequence = []

    for epoch in range(int(epochs)):
        semantic_hash_before = hash_backbone(model.autoencoders)
        if arm == "BASE":
            semantic_record = {
                "semantic_phase_used": False,
                "batch_count": 0,
                "batch_boundaries": [],
                "semantic_loss_mean": None,
                "pseudo_tensors_detached_pass": None,
                "gradients_zeroed_before_every_batch": None,
                "gradients_cleared_after_phase": True,
                "sample_order_sha256": orders[
                    "semantic_order_sha256_per_epoch"
                ][epoch],
                "precomputed_order_not_executed": True,
            }
            vsa.zero_optimizer_gradients(semantic_optimizers)
        else:
            semantic_record = _semantic_phase(
                model=model,
                semantic_optimizers=semantic_optimizers,
                full_views=full_views,
                semantic_target=semantic_target,
                order=orders["semantic_orders"][epoch],
                device=device,
            )
        semantic_hash_after = hash_backbone(model.autoencoders)
        if arm == vsa.ENGINEERING_ARM:
            _require(
                semantic_hash_before == semantic_hash_after,
                "CYCLE_ZERO changed model parameters in semantic phase",
            )
        semantic_record.update({
            "epoch": epoch + 1,
            "semantic_phase_model_hash_before": semantic_hash_before,
            "semantic_phase_model_hash_after": semantic_hash_after,
        })
        semantic_epoch_records.append(semantic_record)
        semantic_hash_records.append({
            "epoch": epoch + 1,
            "before": semantic_hash_before,
            "after": semantic_hash_after,
        })
        phase_sequence.append({
            "epoch": epoch + 1,
            "phase": "semantic",
            "executed": arm != "BASE",
        })

        if epoch % e1_train.TARGET_REFRESH_INTERVAL == 0:
            native_p_all, native_matches, _, view_weights = (
                e1_train.refresh_native_target(
                    model,
                    full_views,
                    view_weights,
                    device,
                    training_seed=training_seed,
                )
            )
            native_target_refresh_count += 1
        _require(
            native_p_all is not None and native_matches is not None,
            "native MVCAN target is missing",
        )
        native_hash_before = hash_backbone(model.autoencoders)
        _require(
            native_hash_before == semantic_hash_after,
            "native phase did not start from current post-semantic model",
        )
        native_record = _native_phase(
            model=model,
            native_optimizers=native_optimizers,
            full_views=full_views,
            native_p_all=native_p_all,
            native_matches=native_matches,
            order=orders["native_orders"][epoch],
            device=device,
        )
        native_hash_after = hash_backbone(model.autoencoders)
        native_record.update({
            "epoch": epoch + 1,
            "native_phase_model_hash_before": native_hash_before,
            "native_phase_model_hash_after": native_hash_after,
        })
        native_epoch_records.append(native_record)
        native_hash_records.append({
            "epoch": epoch + 1,
            "before": native_hash_before,
            "after": native_hash_after,
        })
        phase_sequence.append({
            "epoch": epoch + 1,
            "phase": "native",
            "executed": True,
        })

    _, final_matches, predictions, _ = e1_train.refresh_native_target(
        model,
        full_views,
        view_weights,
        device,
        training_seed=training_seed,
    )
    expected_sequence = [
        phase
        for _ in range(int(epochs))
        for phase in ("semantic", "native")
    ]
    _require(
        [record["phase"] for record in phase_sequence]
        == expected_sequence,
        "VSA phase ordering mismatch",
    )
    phase_transition_audit = {
        "phase_sequence": phase_sequence,
        "semantic_before_native_every_epoch": True,
        "current_post_semantic_params_used_by_native_phase": True,
        "semantic_phase_model_hash_per_epoch": semantic_hash_records,
        "native_phase_model_hash_per_epoch": native_hash_records,
        "no_cross_phase_gradient_accumulation": True,
        "single_backward_contains_native_and_pseudo": False,
        "semantic_native_gradient_mixed": False,
    }
    return predictions, {
        "semantic_epoch_records": semantic_epoch_records,
        "native_epoch_records": native_epoch_records,
        "phase_transition_audit": phase_transition_audit,
        "native_target_refresh_count": native_target_refresh_count,
        "native_P_all_definition": (
            "MVCAN target_distribution(new_P(latent_fusion, centers))"
        ),
        "native_P_all_rewritten": False,
        "native_Match_refreshed_independently": True,
        "auxiliary_M0_refreshed": False,
        "auxiliary_M0_used_by_native_Match": False,
        "final_native_Match_shape": list(final_matches.shape),
        "semantic_optimizer_used_only_in_semantic_phase": True,
        "native_optimizer_used_only_in_native_phase": True,
        "no_cross_phase_gradient_accumulation": True,
        "losses_finite_pass": True,
    }


def _default_epochs(run_kind):
    return (
        vsa.ENGINEERING_EPOCHS
        if run_kind == "E0"
        else vsa.FORMAL_EPOCHS
    )


def _default_device(run_kind):
    return "cpu" if run_kind == "E0" else "cuda:0"


def _default_output_root(run_kind):
    return (
        DEFAULT_E0_OUTPUT_ROOT
        if run_kind == "E0"
        else DEFAULT_A0_OUTPUT_ROOT
    )


def run_arm(
    arm,
    run_kind="E0",
    epochs=None,
    seed=vsa.SEED,
    output_dir=None,
    device=None,
    c0_artifact_path=DEFAULT_C0_ARTIFACT_PATH,
    c0_seal_path=DEFAULT_C0_SEAL_PATH,
    feature_path=DEFAULT_FEATURE_PATH,
    feature_audit_path=DEFAULT_FEATURE_AUDIT_PATH,
    model_dir=DEFAULT_E1_LWC_MODEL_DIR,
    model_audit_path=DEFAULT_E1_LWC_AUDIT_PATH,
    full_gt_path=DEFAULT_FULL_GT_PATH,
):
    active_epochs = _default_epochs(run_kind) if epochs is None else int(epochs)
    validate_run_contract(run_kind, arm, active_epochs)
    requested_seed = vsa.validate_training_seed(seed)
    if run_kind == "E0":
        _require(
            requested_seed == vsa.SEED,
            "VSA-E0 exact identity remains frozen at seed20",
        )
    output_root = (
        _default_output_root(run_kind) / arm
        if output_dir is None
        else _resolve(output_dir)
    )
    _require(not output_root.exists(), "refusing to overwrite VSA output")
    active_device = _default_device(run_kind) if device is None else device
    target_device = torch.device(active_device)
    if target_device.type == "cuda":
        _require(torch.cuda.is_available(), "requested CUDA device unavailable")
        torch.cuda.set_device(target_device)

    determinism_audit = enable_strict_determinism(requested_seed)
    source_provenance = verify_frozen_source_provenance()
    decoupling_source_audit = vsa.assert_no_additive_gradient_source((
        Path(__file__), Path(vsa.__file__)
    ))
    views, sample_ids, feature_provenance = (
        e1_train.load_frozen_feature_artifact(
            _resolve(feature_path), _resolve(feature_audit_path)
        )
    )
    fixed_feature_lineage = validate_fixed_feature_realization(
        feature_path, feature_audit_path, feature_provenance
    )
    _require(
        np.array_equal(
            sample_ids, np.arange(vsa.SAMPLE_NUM, dtype=np.int64)
        ),
        "VSA feature rows are not canonical sample IDs",
    )
    model, config, checkpoint_provenance = (
        load_lineage_aware_e1_lwc_model(
            model_dir,
            model_audit_path,
            target_device,
            requested_seed,
        )
    )
    initial_model_hash = hash_backbone(model.autoencoders)
    _require(
        initial_model_hash
        == checkpoint_provenance["expected_model_hash_from_own_audit"],
        "VSA initial model does not match same-seed E1 audit",
    )
    c0_arrays, c0_provenance = load_lineage_aware_c0_artifact(
        c0_artifact_path,
        c0_seal_path,
        requested_seed,
        checkpoint_provenance,
        feature_provenance,
    )
    stateful_layer_audit = vsa.audit_model_stateful_layers(model)
    M0, M0_audit = derive_seed_specific_M0(
        model,
        views,
        c0_arrays["native_global_cluster"],
        target_device,
        requested_seed,
        c0_provenance,
    )
    _require(
        hash_backbone(model.autoencoders) == initial_model_hash,
        "VSA M0 derivation changed the initial model",
    )
    semantic_target, semantic_target_audit = (
        vsa.build_frozen_semantic_target(arm, c0_arrays, M0)
    )
    semantic_target_audit.update({
        "semantic_objective": (
            "none" if arm == "BASE" else vsa.SEMANTIC_OBJECTIVE
        ),
        "legacy_C1_target_builder_lambda_metadata": (
            semantic_target_audit.pop("lambda_pseudo", 0.0)
        ),
        "lambda_pseudo_training_used": False,
        "exact_C1_target_builder_reused": True,
        "E_unchanged": True,
        "m_unchanged": True,
        "T_unchanged": True,
        "a_unchanged": True,
        "T_local_unchanged": True,
        "pseudo_tensors_detached": arm != "BASE",
        "C0_arrays_frozen": True,
        "M0_frozen": True,
        "M0_auxiliary_only": True,
        "native_Match_refreshed_independently": True,
        "C0_artifact_provenance": c0_provenance,
        "M0_audit": M0_audit,
        "dynamic_cycle_used": False,
        "top_k_used": False,
        "threshold_used": False,
    })
    semantic_optimizers, native_optimizers, optimizer_audit = (
        vsa.build_decoupled_optimizers(model, arm)
    )
    _require(
        int(config["training"]["batch_size"]) == vsa.BATCH_SIZE,
        "native batch size mismatch",
    )
    orders = vsa.precompute_epoch_orders(
        active_epochs, sample_ids, seed=requested_seed
    )

    output_root.mkdir(parents=True)
    predictions, runtime = train_vsa_arm(
        arm=arm,
        epochs=active_epochs,
        model=model,
        semantic_optimizers=semantic_optimizers,
        native_optimizers=native_optimizers,
        views=views,
        sample_ids=sample_ids,
        semantic_target=semantic_target,
        orders=orders,
        device=target_device,
        batch_size=vsa.BATCH_SIZE,
        training_seed=requested_seed,
    )
    prediction_path, prediction_audit = (
        c1_train.save_predictions_before_GT(
            predictions, sample_ids, output_root
        )
    )
    labels = e1_train.load_labels_after_predictions(
        _resolve(full_gt_path), prediction_path
    )
    metrics = e1_train.evaluate_predictions(labels, predictions)
    final_model_hash = hash_backbone(model.autoencoders)
    stage = "VSA-E0" if run_kind == "E0" else "VSA-A0"
    semantic_loss_history = {
        "stage": stage,
        "arm": arm,
        "epochs": active_epochs,
        "semantic_phase_used": arm != "BASE",
        "objective": (
            "none" if arm == "BASE" else vsa.SEMANTIC_OBJECTIVE
        ),
        "epoch_records": runtime["semantic_epoch_records"],
    }
    native_loss_history = {
        "stage": stage,
        "arm": arm,
        "epochs": active_epochs,
        "objective": vsa.NATIVE_OBJECTIVE,
        "epoch_records": runtime["native_epoch_records"],
    }
    phase_transition_audit = {
        "stage": stage,
        "arm": arm,
        "epochs": active_epochs,
        **runtime["phase_transition_audit"],
    }
    metrics_record = {
        "stage": stage,
        "arm": arm,
        "epochs": active_epochs,
        "seed": requested_seed,
        "metrics": metrics,
    }
    train_audit = {
        "stage": stage,
        "arm": arm,
        "run_kind": run_kind,
        "formal_scientific_arm": (
            run_kind == "A0" and arm in vsa.FORMAL_ARMS
        ),
        "engineering_only_arm": arm == vsa.ENGINEERING_ARM,
        "epochs": active_epochs,
        "seed": requested_seed,
        "training_seed": requested_seed,
        "weak_quality_condition_fixed": WEAK_QUALITY_CONDITION,
        "E1_audit_path": checkpoint_provenance["source_audit_path"],
        "E1_training_seed": checkpoint_provenance[
            "source_training_seed"
        ],
        "E1_actual_loaded_aggregate_hash": checkpoint_provenance[
            "actual_loaded_aggregate_sha256"
        ],
        "E1_expected_from_own_audit_aggregate_hash": (
            checkpoint_provenance[
                "expected_aggregate_sha256_from_own_audit"
            ]
        ),
        "C0_artifact_path": c0_provenance["artifact_path"],
        "C0_seal_path": c0_provenance["seal_path"],
        "C0_artifact_file_hash": c0_provenance[
            "artifact_file_sha256"
        ],
        "C0_source_model_provenance_hash": c0_provenance[
            "C0_source_model_hash"
        ],
        "C0_E1_lineage_match_pass": c0_provenance[
            "C0_E1_lineage_match_pass"
        ],
        "requested_seed_lineage_match_pass": bool(
            checkpoint_provenance["requested_seed_lineage_match_pass"]
            and c0_provenance["requested_seed_lineage_match_pass"]
        ),
        "M0_seed_specific_logical_hash": M0_audit[
            "seed_specific_logical_sha256"
        ],
        "feature_realization_unchanged": fixed_feature_lineage[
            "feature_realization_unchanged_pass"
        ],
        "fixed_feature_lineage": fixed_feature_lineage,
        "R_loaded": False,
        "sparse_labels_loaded": False,
        "corruption_mask_loaded_into_training": False,
        "GT_loaded_before_prediction_seal": False,
        "N": vsa.SAMPLE_NUM,
        "V": vsa.VIEW_NUM,
        "K": vsa.CLASS_NUM,
        "batch_size": vsa.BATCH_SIZE,
        "initial_model_hash": initial_model_hash,
        "initial_model_aggregate_sha256": initial_model_hash["aggregate"],
        "final_model_hash": final_model_hash,
        "final_model_aggregate_sha256": final_model_hash["aggregate"],
        "checkpoint_provenance": checkpoint_provenance,
        "determinism_audit": determinism_audit,
        "source_provenance": source_provenance,
        "feature_provenance": feature_provenance,
        "C0_artifact_provenance": c0_provenance,
        "M0_audit": M0_audit,
        "stateful_layer_audit": stateful_layer_audit,
        "semantic_optimizer_config": optimizer_audit[
            "semantic_optimizer_config"
        ],
        "native_optimizer_config": optimizer_audit[
            "native_optimizer_config"
        ],
        "optimizer_decoupling_audit": optimizer_audit,
        "semantic_order_sha256_per_epoch": orders[
            "semantic_order_sha256_per_epoch"
        ],
        "native_order_sha256_per_epoch": orders[
            "native_order_sha256_per_epoch"
        ],
        "orders_precomputed_before_training": True,
        "canonical_sample_ids": True,
        "sample_ID_target_indexing": arm != "BASE",
        "semantic_phase_used": arm != "BASE",
        "semantic_objective": (
            "none" if arm == "BASE" else vsa.SEMANTIC_OBJECTIVE
        ),
        "native_objective": vsa.NATIVE_OBJECTIVE,
        "additive_loss_used": False,
        "additive_native_plus_pseudo_used": False,
        "single_backward_contains_native_and_pseudo": False,
        "semantic_native_gradient_mixed": False,
        "lambda_pseudo_training_used": False,
        "semantic_optimizer_used_only_in_semantic_phase": True,
        "native_optimizer_used_only_in_native_phase": True,
        "no_cross_phase_gradient_accumulation": True,
        "pseudo_tensors_detached": arm == "BASE" or bool(
            semantic_target_audit["all_pseudo_tensors_detached_pass"]
        ),
        "C0_arrays_frozen": True,
        "M0_frozen": True,
        "M0_auxiliary_only": True,
        "native_Match_refreshed_independently": True,
        "native_P_all_rewritten": False,
        "prediction_audit": prediction_audit,
        "prediction_logical_sha256": prediction_audit[
            "prediction_logical_sha256"
        ],
        "final_prediction_sealed": True,
        "semantic_loss_history": semantic_loss_history,
        "native_loss_history": native_loss_history,
        "phase_transition_audit": phase_transition_audit,
        "metrics": metrics,
        "runtime": runtime,
        "decoupling_source_audit": decoupling_source_audit,
        "leakage": {
            "full_GT_loaded_during_training": False,
            "full_GT_loaded_after_prediction_seal": True,
            "sparse_labels_loaded": False,
            "B7_labels_loaded": False,
            "one_percent_labels_loaded": False,
            "R_loaded": False,
            "corruption_mask_loaded": False,
            "oracle_used": False,
        },
        "forbidden_mechanisms": {
            "Memory_used": False,
            "P_corr_used": False,
            "P_util_used": False,
            "dynamic_cycle_used": False,
            "top_k_used": False,
            "threshold_used": False,
            "gradient_normalization_used": False,
            "PCGrad_used": False,
            "GradNorm_used": False,
            "lambda_sweep_used": False,
        },
        "CYCLE_ZERO_semantic_lr_zero": (
            arm == vsa.ENGINEERING_ARM
            and optimizer_audit["semantic_optimizer_learning_rate"] == 0.0
        ),
        "CYCLE_ZERO_semantic_model_unchanged": (
            None
            if arm != vsa.ENGINEERING_ARM
            else all(
                record["before"] == record["after"]
                for record in phase_transition_audit[
                    "semantic_phase_model_hash_per_epoch"
                ]
            )
        ),
        "VSA_ENGINEERING_PASS": True,
    }
    c1.write_json(output_root / "metrics.json", metrics_record)
    c1.write_json(output_root / "train_audit.json", train_audit)
    c1.write_json(
        output_root / "semantic_target_audit.json",
        semantic_target_audit,
    )
    c1.write_json(
        output_root / "semantic_loss_history.json",
        semantic_loss_history,
    )
    c1.write_json(
        output_root / "native_loss_history.json", native_loss_history
    )
    c1.write_json(
        output_root / "phase_transition_audit.json",
        phase_transition_audit,
    )
    print("VSA_ENGINEERING_PASS=True")
    print("STAGE=" + stage)
    print("ARM=" + arm)
    print("ACC={:.10f}".format(metrics["ACC"]))
    print("NMI={:.10f}".format(metrics["NMI"]))
    print("ARI={:.10f}".format(metrics["ARI"]))
    print("Saved: " + _display(output_root))
    return {
        "output_dir": output_root,
        "metrics": metrics_record,
        "train_audit": train_audit,
        "semantic_target_audit": semantic_target_audit,
        "semantic_loss_history": semantic_loss_history,
        "native_loss_history": native_loss_history,
        "phase_transition_audit": phase_transition_audit,
    }


def _load_e0_record(root, arm):
    arm_dir = Path(root) / arm
    required = (
        "metrics.json",
        "train_audit.json",
        "semantic_loss_history.json",
        "native_loss_history.json",
        "phase_transition_audit.json",
        "final_predictions.npz",
    )
    _require(arm_dir.is_dir(), "missing VSA-E0 arm: " + arm)
    for filename in required:
        _require(
            (arm_dir / filename).is_file(),
            "missing VSA-E0 file: " + arm + "/" + filename,
        )
    audit = c1.read_json(arm_dir / "train_audit.json")
    with np.load(arm_dir / "final_predictions.npz", allow_pickle=False) as archive:
        predictions = np.asarray(archive["predictions"], dtype=np.int64)
        sample_ids = np.asarray(archive["sample_ids"], dtype=np.int64)
    _require(
        predictions.shape == sample_ids.shape == (vsa.SAMPLE_NUM,)
        and np.array_equal(
            sample_ids, np.arange(vsa.SAMPLE_NUM, dtype=np.int64)
        )
        and tensor_sha256(predictions)
        == audit["prediction_logical_sha256"],
        "VSA-E0 prediction seal mismatch: " + arm,
    )
    return audit


def verify_saved_e0_identity(input_dir=DEFAULT_E0_OUTPUT_ROOT):
    """Compare completed BASE/CYCLE_ZERO runs without writing outputs."""
    root = _resolve(input_dir)
    base_record = _load_e0_record(root, "BASE")
    cycle_zero_record = _load_e0_record(root, "CYCLE_ZERO")
    return vsa.validate_vsa_e0_identity(
        base_record, cycle_zero_record
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=vsa.ALL_ARMS)
    parser.add_argument("--run-kind", choices=RUN_KINDS, default="E0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        choices=vsa.TRAINING_SEED_CHOICES,
        default=vsa.SEED,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verify-e0", action="store_true")
    parser.add_argument(
        "--e0-input-dir", default=str(DEFAULT_E0_OUTPUT_ROOT)
    )
    parser.add_argument(
        "--c0-artifact-path", default=str(DEFAULT_C0_ARTIFACT_PATH)
    )
    parser.add_argument(
        "--c0-seal-path", default=str(DEFAULT_C0_SEAL_PATH)
    )
    parser.add_argument("--feature-path", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument(
        "--feature-audit-path", default=str(DEFAULT_FEATURE_AUDIT_PATH)
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_E1_LWC_MODEL_DIR))
    parser.add_argument(
        "--model-audit-path", default=str(DEFAULT_E1_LWC_AUDIT_PATH)
    )
    parser.add_argument("--full-gt-path", default=str(DEFAULT_FULL_GT_PATH))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.verify_e0:
        _require(args.arm is None, "--verify-e0 does not accept --arm")
        result = verify_saved_e0_identity(args.e0_input_dir)
        print("DECISION=" + result["final_decision"])
        return 0 if result["VSA_E0_IDENTITY_PASS"] else 1
    _require(args.arm is not None, "--arm is required for a VSA run")
    run_arm(
        arm=args.arm,
        run_kind=args.run_kind,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
        c0_artifact_path=args.c0_artifact_path,
        c0_seal_path=args.c0_seal_path,
        feature_path=args.feature_path,
        feature_audit_path=args.feature_audit_path,
        model_dir=args.model_dir,
        model_audit_path=args.model_audit_path,
        full_gt_path=args.full_gt_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
