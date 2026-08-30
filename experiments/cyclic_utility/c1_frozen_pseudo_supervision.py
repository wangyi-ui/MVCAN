"""C1 frozen cycle-verified pseudo-supervision primitives."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.g2_utility_semantic_consensus import (
    g2_consensus_protocol as g2_protocol,
)
from irv.b3_audit import hash_backbone
from irv.b4_information_utility import tensor_sha256


SAMPLE_NUM = 1400
VIEW_NUM = 6
CLASS_NUM = 7
DIRECTION_COUNT = 20
SEED = 20
DEFAULT_EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
LAMBDA_PSEUDO = 0.01
LOG_EPSILON = 1e-8
MASS_EPSILON = 1e-12
MASS_IDENTITY_RTOL = 1e-6
MASS_IDENTITY_ATOL = 1e-7
FORMAL_ARMS = (
    "BASE",
    "UNIFORM",
    "CONF",
    "CYCLE",
    "SHUFFLED_CYCLE",
)
PSEUDO_ARMS = FORMAL_ARMS[1:]
ENGINEERING_ARM = "CYCLE_ZERO"
ALL_ARMS = FORMAL_ARMS + (ENGINEERING_ARM,)
METRIC_NAMES = ("ACC", "NMI", "ARI")

EXPECTED_C0_ARTIFACT_SHA256 = (
    "92c3e8f8ff87b06bda2491ad463727560319d8ee0f3a5dda1a0201c9339f746b"
)
EXPECTED_C0_SEAL_SHA256 = (
    "93e41c2eb2541628889bfa6dcf9a5aa66d52b01be8883e8806c28902d92bdcab"
)
EXPECTED_M0_LOGICAL_SHA256 = (
    "aaa7be3a9b4516278e508330299f881b6e7c5855a80cb43101c715a455f5183d"
)
EXPECTED_C0_ARRAY_SHA256 = {
    "y_gen": (
        "95d4a98ef44a8edbe2051f04008b866bf499248d800245e2d0e46eefa238861e"
    ),
    "C_conf": (
        "01c403599a3956ff1b531320a91207c79c7e8ccf3e60c42326df5b3e615712f0"
    ),
    "U_cycle": (
        "b23dde6bd4f524e5e47748a73df3224f77e93d166d94f5a6c60db8865afefbfe"
    ),
    "U_cycle_shuffle": (
        "2d9e985263606297189c1636d24a3e02ea61ab2e9967c399394e61349c03217f"
    ),
    "native_global_cluster": (
        "95b0dc0294272786164088565b78e9514586c833f91dd87375909dc45d68ee88"
    ),
    "generator_subsets": (
        "1da78797e059f952bf6ae3749fc94ca2dd5bebf83d76b1e8eeb6a03583fe36da"
    ),
    "verifier_subsets": (
        "aa726d58832d5eb17ff952049d6be1b3c6eb7295b6fcba6b656c3c1ead80baca"
    ),
}
EXPECTED_E1_CHECKPOINT_SHA256 = (
    "b914af6e3740d1808d44ed2592b012dd67bb90356b745c2c117d63743256d77c",
    "7c5ce3be3fc2a62b0731b58651e29af2da6e2dfd2eb51f53d2e574e019e3c576",
    "69d65bfeb51617a3a560aa040e7be8c2ad9e2ef90368ca052914c49fb04a28ac",
    "cd629a4b604e814595d1f9c9e720a94cdcf805d871863b457b95b4f97866a33c",
    "6268ceeda5c9cf1dedc0de1550888da0d61e25ca1f786cd6c181639257df692d",
    "8a9117c23a45b24e4e6641746e4f67c864c0a0638edbe410f4cf7866c79a1701",
)
EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256 = (
    "a9b1e523c90fc450ba59fa206b78d1c02c5dd410c452766d3942cc37fc1e9d62"
)
PSEUDO_SUPERVISION_C0_ARRAYS = (
    "y_gen",
    "C_conf",
    "U_cycle",
    "U_cycle_shuffle",
    "generator_subsets",
    "verifier_subsets",
)
COORDINATE_PROVENANCE_C0_ARRAYS = ("native_global_cluster",)
REQUIRED_C0_ARRAYS = (
    PSEUDO_SUPERVISION_C0_ARRAYS + COORDINATE_PROVENANCE_C0_ARRAYS
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


def frozen_tensor(value, *, dtype=None, device=None):
    if torch.is_tensor(value):
        tensor = value.detach()
        return tensor.to(
            dtype=tensor.dtype if dtype is None else dtype,
            device=tensor.device if device is None else device,
        ).detach()
    array = np.array(value, copy=True, order="C")
    return torch.as_tensor(
        array, dtype=dtype, device=device
    ).detach()


def as_numpy(value, dtype=None):
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    return np.ascontiguousarray(array)


def set_deterministic_seed(seed=SEED):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_frozen_c0_artifact(artifact_path, seal_path):
    """Load frozen pre-GT C1 arrays plus the coordinate-only C0 anchor."""
    artifact = Path(artifact_path)
    seal_file = Path(seal_path)
    _require(
        artifact.is_file() and seal_file.is_file(),
        "frozen C0 artifact/seal is missing",
    )
    artifact_sha = file_sha256(artifact)
    seal_sha = file_sha256(seal_file)
    _require(
        artifact_sha == EXPECTED_C0_ARTIFACT_SHA256
        and seal_sha == EXPECTED_C0_SEAL_SHA256,
        "frozen C0 file provenance mismatch",
    )
    seal = read_json(seal_file)
    _require(
        seal.get("stage") == "C0"
        and seal.get("artifact_file_sha256") == artifact_sha
        and seal.get("scores_completed_before_GT") is True
        and seal.get("scores_saved_before_GT") is True
        and seal.get("scores_hashed_before_GT") is True
        and seal.get("scores_reloaded_before_GT") is True
        and seal.get("full_GT_loaded_before_seal") is False
        and seal.get("native_global_reference_provenance", {}).get(
            "logical_sha256"
        )
        == EXPECTED_C0_ARRAY_SHA256["native_global_cluster"],
        "C0 pre-GT seal boundary mismatch",
    )
    arrays = {}
    with np.load(artifact, allow_pickle=False) as archive:
        archive_fields = tuple(archive.files)
        _require(
            set(REQUIRED_C0_ARRAYS).issubset(archive_fields),
            "C0 artifact is missing a required pre-GT array",
        )
        for name in REQUIRED_C0_ARRAYS:
            value = np.array(archive[name], copy=True, order="C")
            expected_hash = EXPECTED_C0_ARRAY_SHA256[name]
            _require(
                tensor_sha256(value)
                == seal["arrays"][name]["logical_sha256"]
                == expected_hash,
                "C0 logical array provenance mismatch: " + name,
            )
            arrays[name] = frozen_tensor(value)

    expected_shapes = {
        "y_gen": (SAMPLE_NUM, DIRECTION_COUNT),
        "C_conf": (SAMPLE_NUM, DIRECTION_COUNT),
        "U_cycle": (SAMPLE_NUM, DIRECTION_COUNT),
        "U_cycle_shuffle": (SAMPLE_NUM, DIRECTION_COUNT),
        "native_global_cluster": (SAMPLE_NUM,),
        "generator_subsets": (DIRECTION_COUNT, 3),
        "verifier_subsets": (DIRECTION_COUNT, 3),
    }
    for name, expected_shape in expected_shapes.items():
        _require(
            tuple(arrays[name].shape) == expected_shape,
            "C0 array shape mismatch: " + name,
        )
    y_gen = arrays["y_gen"]
    _require(
        y_gen.dtype == torch.long
        and bool(torch.all((y_gen >= 0) & (y_gen < CLASS_NUM)).item()),
        "C0 y_gen boundary mismatch",
    )
    sealed_global_ids = arrays["native_global_cluster"]
    _require(
        sealed_global_ids.dtype == torch.long
        and bool(torch.all(
            (sealed_global_ids >= 0)
            & (sealed_global_ids < CLASS_NUM)
        ).item())
        and torch.equal(
            torch.unique(sealed_global_ids).sort().values,
            torch.arange(CLASS_NUM),
        ),
        "C0 sealed native global coordinate boundary mismatch",
    )
    for name in ("C_conf", "U_cycle", "U_cycle_shuffle"):
        values = arrays[name]
        _require(
            values.is_floating_point()
            and bool(torch.isfinite(values).all().item())
            and bool(torch.all((values >= 0.0) & (values <= 1.0)).item()),
            "C0 score boundary mismatch: " + name,
        )
    for value in arrays.values():
        _require(
            not value.requires_grad and value.grad_fn is None,
            "C0 array did not remain frozen",
        )
    return arrays, {
        "artifact_path": str(artifact),
        "artifact_file_sha256": artifact_sha,
        "artifact_file_sha256_pass": True,
        "seal_path": str(seal_file),
        "seal_file_sha256": seal_sha,
        "seal_file_sha256_pass": True,
        "archive_fields": list(archive_fields),
        "loaded_fields": list(REQUIRED_C0_ARRAYS),
        "pseudo_supervision_fields": list(
            PSEUDO_SUPERVISION_C0_ARRAYS
        ),
        "coordinate_provenance_fields": list(
            COORDINATE_PROVENANCE_C0_ARRAYS
        ),
        "loaded_pre_GT_arrays_only": True,
        "GT_derived_correctness_loaded": False,
        "C0_scores_recomputed": False,
        "native_global_cluster_shape": list(
            sealed_global_ids.shape
        ),
        "native_global_cluster_dtype": str(
            sealed_global_ids.numpy().dtype
        ),
        "native_global_cluster_logical_sha256": (
            EXPECTED_C0_ARRAY_SHA256["native_global_cluster"]
        ),
        "native_global_cluster_hash_pass": True,
        "native_global_cluster_row_order": (
            "C0 original dataset row i; bound to arange(1400) "
            "sample_ids by the C1 runner"
        ),
        "native_global_cluster_used_for_weighting": False,
        "native_global_cluster_used_for_pseudo_label": False,
        "native_global_cluster_used_for_M0_coordinate_only": True,
        "array_logical_sha256": dict(EXPECTED_C0_ARRAY_SHA256),
    }


def validate_M0(M0):
    """Validate frozen M0: [V,K,K] = [6,7,7] permutations."""
    matrices = frozen_tensor(M0)
    expected = torch.ones(
        VIEW_NUM,
        CLASS_NUM,
        dtype=matrices.dtype,
        device=matrices.device,
    )
    _require(
        matrices.shape == (VIEW_NUM, CLASS_NUM, CLASS_NUM)
        and bool(torch.isfinite(matrices).all().item())
        and bool(torch.all(
            (matrices == 0.0) | (matrices == 1.0)
        ).item())
        and torch.equal(matrices.sum(dim=1), expected)
        and torch.equal(matrices.sum(dim=2), expected),
        "M0 must contain six full 7x7 permutations",
    )
    _require(
        not matrices.requires_grad and matrices.grad_fn is None,
        "M0 did not remain frozen",
    )
    return matrices.detach()


@torch.no_grad()
def derive_frozen_M0(model, views, sealed_global_ids, device):
    """Rebuild M0 in the frozen C0 global coordinate system."""
    target_device = torch.device(device)
    canonical_global_ids = frozen_tensor(
        sealed_global_ids, dtype=torch.long, device="cpu"
    )
    canonical_global_sha = tensor_sha256(
        canonical_global_ids.numpy()
    )
    _require(
        canonical_global_ids.shape == (SAMPLE_NUM,)
        and canonical_global_ids.dtype == torch.long
        and canonical_global_sha
        == EXPECTED_C0_ARRAY_SHA256["native_global_cluster"],
        "sealed native global coordinate provenance mismatch",
    )

    prior_modes = [
        autoencoder.training for autoencoder in model.autoencoders
    ]
    for autoencoder in model.autoencoders:
        autoencoder.eval()
    q_local_views = []
    for autoencoder, view in zip(model.autoencoders, views):
        features = frozen_tensor(
            view, dtype=torch.float32, device=target_device
        )
        raw_z = autoencoder.encoder(features).detach()
        q_local = autoencoder.clustering(raw_z).detach()
        q_local_views.append(q_local.cpu())
    q_local_stack = torch.stack(q_local_views, dim=1).detach()
    _require(
        q_local_stack.shape
        == (SAMPLE_NUM, VIEW_NUM, CLASS_NUM),
        "frozen q_local shape mismatch for M0 alignment",
    )

    alignment = g2_protocol.build_alignment(
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
    M0_int64 = validate_M0(alignment_int64)
    alignment_int64_sha = tensor_sha256(alignment_int64)
    _require(
        alignment_int64_sha == EXPECTED_M0_LOGICAL_SHA256,
        "frozen M0 does not match C0 sealed provenance",
    )

    M0 = torch.as_tensor(
        np.array(alignment_int64, copy=True, order="C"),
        dtype=torch.float32,
    ).detach()
    M0 = validate_M0(M0)
    exact_cast = torch.equal(
        M0, M0_int64.to(dtype=torch.float32)
    )
    _require(
        exact_cast,
        "training M0 is not an exact float32 cast of sealed int64 M0",
    )
    training_M0_sha = tensor_sha256(M0.numpy())
    for autoencoder, was_training in zip(
        model.autoencoders, prior_modes
    ):
        autoencoder.train(was_training)
    return M0.detach(), {
        "shape": list(M0.shape),
        "logical_sha256": alignment_int64_sha,
        "matches_C0_seal_pass": True,
        "canonical_global_coordinate_source": (
            "C0_PRE_GT_sealed_native_global_cluster"
        ),
        "sealed_native_global_cluster_logical_sha256": (
            canonical_global_sha
        ),
        "sealed_native_global_cluster_hash_pass": True,
        "alignment_int64_shape": list(alignment_int64.shape),
        "alignment_int64_logical_sha256": alignment_int64_sha,
        "alignment_int64_matches_C0_seal_pass": True,
        "training_M0_dtype": str(M0.numpy().dtype),
        "training_M0_logical_sha256": training_M0_sha,
        "training_M0_is_exact_01_cast_of_int64_pass": exact_cast,
        "full_permutation_pass": True,
        "row_sums_one_pass": True,
        "column_sums_one_pass": True,
        "detached_pass": True,
        "derived_before_training": True,
        "frozen_throughout_C1": True,
        "q_local_argmax_used_for_alignment": True,
        "Match_input_direction": "Match(local_ids, global_ids)",
        "C1_replayed_KMeans_used_as_coordinate_anchor": False,
        "GT_used_for_M0": False,
        "sparse_labels_used_for_M0": False,
        "corruption_mask_used_for_M0": False,
        "native_global_cluster_used_for_weighting": False,
        "native_global_cluster_used_for_pseudo_label": False,
        "native_global_cluster_used_for_M0_coordinate_only": True,
        "local_to_global_direction_documented": "q_local @ M0.T",
        "global_to_local_direction": "T_global @ M0",
        "C0_scores_recomputed": False,
    }


def verify_e1_initialization(checkpoint_audit):
    """Verify the common frozen E1-LWC initialization provenance."""
    file_hashes = tuple(checkpoint_audit["checkpoint_file_sha256"])
    aggregate = checkpoint_audit["initial_backbone_hash"]["aggregate"]
    _require(
        file_hashes == EXPECTED_E1_CHECKPOINT_SHA256
        and aggregate == EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256,
        "E1 LWC initialization provenance mismatch",
    )
    return {
        "checkpoint_file_sha256": list(file_hashes),
        "initial_model_aggregate_sha256": aggregate,
        "frozen_E1_LWC_100ep_seed20_pass": True,
    }


def build_arm_weights(arm, c0_arrays):
    """Select frozen weights: [N,S] = [1400,20]."""
    if arm not in ALL_ARMS:
        raise ValueError("unknown C1 arm")
    if arm == "BASE":
        return None, {
            "weight_source": None,
            "pseudo_path_constructed": False,
        }
    weight_sources = {
        "UNIFORM": None,
        "CONF": "C_conf",
        "CYCLE": "U_cycle",
        "SHUFFLED_CYCLE": "U_cycle_shuffle",
        "CYCLE_ZERO": "U_cycle",
    }
    source_name = weight_sources[arm]
    if arm == "UNIFORM":
        weights = torch.ones(
            SAMPLE_NUM,
            DIRECTION_COUNT,
            dtype=torch.float32,
        )
    else:
        weights = frozen_tensor(
            c0_arrays[source_name], dtype=torch.float32
        )
    _require(
        weights.shape == (SAMPLE_NUM, DIRECTION_COUNT)
        and bool(torch.isfinite(weights).all().item())
        and bool(torch.all(weights >= 0.0).item()),
        "C1 arm weight boundary mismatch",
    )
    return weights.detach(), {
        "weight_source": (
            "exact_ones" if arm == "UNIFORM" else source_name
        ),
        "weights_shape": list(weights.shape),
        "weights_logical_sha256": tensor_sha256(weights.numpy()),
        "pseudo_path_constructed": True,
        "detached_pass": True,
    }


def validate_evidence_mass_identity(
    mass_from_evidence,
    mass_from_weights,
):
    """Audit the theoretical mass identity at float32 precision."""
    evidence_mass = frozen_tensor(
        mass_from_evidence, dtype=torch.float32
    )
    directional_mass = frozen_tensor(
        mass_from_weights,
        dtype=torch.float32,
        device=evidence_mass.device,
    )
    _require(
        evidence_mass.shape == directional_mass.shape
        == (SAMPLE_NUM,),
        "evidence mass identity shape mismatch",
    )
    max_abs_diff = float(
        torch.max(torch.abs(evidence_mass - directional_mass)).item()
    )
    bitwise_exact = torch.equal(evidence_mass, directional_mass)
    numerical_pass = bool(torch.allclose(
        evidence_mass,
        directional_mass,
        rtol=MASS_IDENTITY_RTOL,
        atol=MASS_IDENTITY_ATOL,
    ))
    audit = {
        "mass_identity_theoretical": True,
        "mass_identity_bitwise_exact": bitwise_exact,
        "mass_identity_numerical_pass": numerical_pass,
        "mass_identity_rtol": MASS_IDENTITY_RTOL,
        "mass_identity_atol": MASS_IDENTITY_ATOL,
        "mass_identity_max_abs_diff": max_abs_diff,
    }
    _require(
        numerical_pass,
        "evidence mass is not mean directional weight",
    )
    return audit


def build_sample_semantic_evidence(y_gen, weights):
    """Build E/T/m/a from frozen y_gen and weights without sample loops."""
    # y_gen/weights: [N,S] = [1400,20].
    hypotheses = frozen_tensor(y_gen, dtype=torch.long)
    action_weights = frozen_tensor(
        weights, dtype=torch.float32, device=hypotheses.device
    )
    _require(
        hypotheses.shape == action_weights.shape
        == (SAMPLE_NUM, DIRECTION_COUNT)
        and bool(torch.all(
            (hypotheses >= 0) & (hypotheses < CLASS_NUM)
        ).item())
        and bool(torch.isfinite(action_weights).all().item())
        and bool(torch.all(action_weights >= 0.0).item()),
        "frozen hypothesis/weight boundary mismatch",
    )

    # one_hot: [N,S,K] = [1400,20,7].
    one_hot = F.one_hot(
        hypotheses, num_classes=CLASS_NUM
    ).to(dtype=action_weights.dtype)
    # evidence: [N,K] = mean_s w[i,s] * one_hot(y_gen[i,s]).
    evidence = torch.mean(
        action_weights.unsqueeze(-1) * one_hot,
        dim=1,
    ).detach()
    # mass/sample_strength: [N] = [1400].
    mass = evidence.sum(dim=-1).detach()
    directional_mean_mass = action_weights.mean(dim=1)
    mass_identity_audit = validate_evidence_mass_identity(
        mass, directional_mean_mass
    )
    positive_mass = mass > 0.0
    target_global = torch.zeros_like(evidence)
    target_global[positive_mass] = (
        evidence[positive_mass] / mass[positive_mass, None]
    )
    target_global = target_global.detach()
    mean_mass = mass.mean()
    _require(
        bool(torch.isfinite(mean_mass).item())
        and bool((mean_mass > 0.0).item()),
        "arm mean semantic mass must be positive",
    )
    sample_strength = (mass / (mean_mass + MASS_EPSILON)).detach()
    _require(
        torch.allclose(
            sample_strength.mean(),
            torch.ones_like(sample_strength.mean()),
            rtol=0.0,
            atol=1e-6,
        ),
        "mean sample action strength must equal one",
    )
    _require(
        bool(torch.isfinite(evidence).all().item())
        and bool(torch.isfinite(target_global).all().item())
        and bool(torch.isfinite(sample_strength).all().item())
        and bool(torch.all(evidence >= 0.0).item())
        and bool(torch.all(target_global >= 0.0).item())
        and bool(torch.all(
            target_global[~positive_mass] == 0.0
        ).item()),
        "sample semantic evidence boundary mismatch",
    )
    if bool(positive_mass.any().item()):
        _require(
            torch.allclose(
                target_global[positive_mass].sum(dim=-1),
                torch.ones_like(mass[positive_mass]),
                rtol=0.0,
                atol=1e-6,
            ),
            "positive-mass target rows do not sum to one",
        )
    for tensor in (
        hypotheses,
        action_weights,
        evidence,
        mass,
        target_global,
        sample_strength,
    ):
        _require(
            not tensor.requires_grad and tensor.grad_fn is None,
            "a frozen pseudo tensor retained gradients",
        )
    return {
        "y_gen": hypotheses.detach(),
        "weights": action_weights.detach(),
        "evidence": evidence.detach(),
        "mass": mass.detach(),
        "target_global": target_global.detach(),
        "sample_strength": sample_strength.detach(),
        "positive_mass": positive_mass.detach(),
        "mass_identity_audit": mass_identity_audit,
    }


def map_global_target_to_local(target_global, M0):
    """Map T_global @ M0 to target_local: [N,V,K]."""
    # target_global: [N,K] = [1400,7].
    target = frozen_tensor(target_global, dtype=torch.float32)
    # M0: [V,K,K] = [6,7,7], rows global and columns local.
    matrices = validate_M0(M0).to(dtype=target.dtype)
    _require(
        target.shape == (SAMPLE_NUM, CLASS_NUM),
        "global target shape mismatch",
    )
    # target_local[i,v,:] = target_global[i,:] @ M0[v].
    target_local = torch.einsum(
        "nk,vkj->nvj", target, matrices
    ).detach()
    _require(
        target_local.shape == (
            SAMPLE_NUM,
            VIEW_NUM,
            CLASS_NUM,
        )
        and bool(torch.isfinite(target_local).all().item())
        and bool(torch.all(target_local >= 0.0).item())
        and torch.allclose(
            target_local.sum(dim=-1),
            target.sum(dim=-1)[:, None].expand(-1, VIEW_NUM),
            rtol=0.0,
            atol=1e-6,
        ),
        "global-to-local frozen target mapping mismatch",
    )
    _require(
        not target_local.requires_grad
        and target_local.grad_fn is None,
        "local pseudo target retained gradients",
    )
    return target_local.detach()


def build_frozen_pseudo_target(arm, c0_arrays, M0):
    """Precompute all frozen per-sample C1 target tensors."""
    if arm == "BASE":
        return None, {
            "arm": arm,
            "pseudo_supervision_used": False,
            "lambda_pseudo": 0.0,
        }
    weights, weight_audit = build_arm_weights(arm, c0_arrays)
    evidence = build_sample_semantic_evidence(
        c0_arrays["y_gen"], weights
    )
    target_local = map_global_target_to_local(
        evidence["target_global"], M0
    )
    hashes = {
        name: tensor_sha256(as_numpy(value))
        for name, value in (
            ("y_gen", evidence["y_gen"]),
            ("weights", evidence["weights"]),
            ("evidence", evidence["evidence"]),
            ("mass", evidence["mass"]),
            ("target_global", evidence["target_global"]),
            ("sample_strength", evidence["sample_strength"]),
            ("M0", validate_M0(M0)),
            ("target_local", target_local),
        )
    }
    positive = evidence["positive_mass"]
    audit = {
        "arm": arm,
        "pseudo_supervision_used": True,
        "lambda_pseudo": effective_lambda_pseudo(arm),
        "weight": weight_audit,
        "shapes": {
            "evidence": list(evidence["evidence"].shape),
            "mass": list(evidence["mass"].shape),
            "target_global": list(
                evidence["target_global"].shape
            ),
            "sample_strength": list(
                evidence["sample_strength"].shape
            ),
            "M0": list(validate_M0(M0).shape),
            "target_local": list(target_local.shape),
        },
        "finite_pass": True,
        "nonnegative_pass": True,
        "positive_mass_target_rows_sum_one_pass": True,
        "zero_mass_target_rows_zero_pass": True,
        "zero_mass_row_count": int((~positive).sum().item()),
        "mean_mass": float(evidence["mass"].mean().item()),
        "min_mass": float(evidence["mass"].min().item()),
        "max_mass": float(evidence["mass"].max().item()),
        "mean_sample_strength": float(
            evidence["sample_strength"].mean().item()
        ),
        "mean_sample_strength_one_pass": True,
        **evidence["mass_identity_audit"],
        "all_pseudo_tensors_detached_pass": True,
        "logical_sha256": hashes,
        "same_frozen_y_gen_sha256": hashes["y_gen"],
        "global_to_local_direction": "T_global @ M0",
        "local_to_global_direction": "q_local @ M0.T",
    }
    return {
        **evidence,
        "target_local": target_local.detach(),
    }, audit


def soft_pseudo_cross_entropy(
    q_local_batch,
    target_local_batch,
    sample_strength_batch,
):
    """Return mean_{i,v} a_i * CE_soft[i,v]."""
    # q_local_batch/target_local_batch: [B,V,K].
    q_local = q_local_batch
    target_local = frozen_tensor(
        target_local_batch,
        dtype=q_local.dtype,
        device=q_local.device,
    )
    # sample_strength_batch: [B].
    sample_strength = frozen_tensor(
        sample_strength_batch,
        dtype=q_local.dtype,
        device=q_local.device,
    )
    _require(
        q_local.ndim == 3
        and q_local.shape[1:] == (VIEW_NUM, CLASS_NUM)
        and target_local.shape == q_local.shape
        and sample_strength.shape == (q_local.shape[0],)
        and bool(torch.isfinite(q_local).all().item())
        and bool(torch.isfinite(target_local).all().item())
        and bool(torch.isfinite(sample_strength).all().item())
        and bool(torch.all(sample_strength >= 0.0).item()),
        "soft pseudo loss input boundary mismatch",
    )
    CE_soft = -torch.sum(
        target_local * torch.log(
            torch.clamp(q_local, min=LOG_EPSILON)
        ),
        dim=-1,
    )
    loss = torch.mean(
        sample_strength[:, None] * CE_soft
    )
    _require(
        bool(torch.isfinite(loss).item()),
        "soft pseudo loss is non-finite",
    )
    return loss


def effective_lambda_pseudo(arm):
    if arm == "BASE" or arm == ENGINEERING_ARM:
        return 0.0
    if arm not in PSEUDO_ARMS:
        raise ValueError("unknown C1 arm")
    return LAMBDA_PSEUDO


def combine_native_and_pseudo_loss(
    arm,
    native_loss,
    pseudo_loss=None,
):
    """Keep BASE/CYCLE_ZERO exactly native; add fixed C1 loss otherwise."""
    coefficient = effective_lambda_pseudo(arm)
    if coefficient == 0.0:
        return native_loss
    _require(
        pseudo_loss is not None,
        "pseudo arm is missing auxiliary loss",
    )
    return native_loss + LAMBDA_PSEUDO * pseudo_loss


def pseudo_batch_by_sample_ids(pseudo_target, sample_ids_batch):
    """Index frozen target/strength only by explicit original row IDs."""
    ids = frozen_tensor(sample_ids_batch, dtype=torch.long)
    _require(
        ids.ndim == 1
        and bool(torch.all((ids >= 0) & (ids < SAMPLE_NUM)).item()),
        "sample_ids_batch boundary mismatch",
    )
    target_local_batch = pseudo_target["target_local"][ids].detach()
    sample_strength_batch = (
        pseudo_target["sample_strength"][ids].detach()
    )
    return target_local_batch, sample_strength_batch


def optimizer_configuration_audit(optimizers):
    groups = [
        {
            "class": optimizer.__class__.__name__,
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "initial_state_empty": len(optimizer.state) == 0,
        }
        for optimizer in optimizers
    ]
    _require(
        len(groups) == VIEW_NUM
        and all(
            group["class"] == "Adam"
            and group["learning_rate"] == LEARNING_RATE
            and group["initial_state_empty"]
            for group in groups
        ),
        "C1 optimizer fairness boundary mismatch",
    )
    return {
        "policy": (
            "fresh independent Adam per MVCAN view from "
            "model-only E1-LWC checkpoint"
        ),
        "optimizer_count": len(groups),
        "learning_rate": LEARNING_RATE,
        "initial_state_empty_pass": True,
        "same_policy_all_arms": True,
        "per_optimizer": groups,
    }


def prediction_seal_payload(predictions, sample_ids):
    values = as_numpy(predictions, dtype=np.int64)
    ids = as_numpy(sample_ids, dtype=np.int64)
    _require(
        values.shape == ids.shape == (SAMPLE_NUM,)
        and np.array_equal(ids, np.arange(SAMPLE_NUM, dtype=np.int64)),
        "C1 prediction seal boundary mismatch",
    )
    return {
        "predictions": values,
        "sample_ids": ids,
    }


def metric_delta_record(candidate, comparator):
    deltas = {
        metric: float(candidate[metric] - comparator[metric])
        for metric in METRIC_NAMES
    }
    strict_improvement_count = sum(
        deltas[metric] > 0.0 for metric in METRIC_NAMES
    )
    delta_sum = float(sum(deltas.values()))
    return {
        "delta": deltas,
        "strict_improvement_count": strict_improvement_count,
        "delta_sum": delta_sum,
        "pass": bool(
            strict_improvement_count >= 2 and delta_sum > 0.0
        ),
    }


def build_c1_pilot_decision(metrics_by_arm):
    """Apply the four preregistered C1 pilot gates exhaustively."""
    _require(
        tuple(metrics_by_arm) == FORMAL_ARMS,
        "C1 formal metric arm set/order mismatch",
    )
    cycle = metrics_by_arm["CYCLE"]
    comparisons = {
        "vs_BASE": metric_delta_record(
            cycle, metrics_by_arm["BASE"]
        ),
        "vs_SHUFFLED_CYCLE": metric_delta_record(
            cycle, metrics_by_arm["SHUFFLED_CYCLE"]
        ),
        "vs_CONF": metric_delta_record(
            cycle, metrics_by_arm["CONF"]
        ),
        "vs_UNIFORM": metric_delta_record(
            cycle, metrics_by_arm["UNIFORM"]
        ),
    }
    net_gain = comparisons["vs_BASE"]["pass"]
    specificity = comparisons["vs_SHUFFLED_CYCLE"]["pass"]
    beats_confidence = comparisons["vs_CONF"]["pass"]
    beats_uniform = comparisons["vs_UNIFORM"]["pass"]

    decision_pass = bool(
        net_gain
        and specificity
        and beats_confidence
        and beats_uniform
    )
    fail_A = bool(not net_gain)
    fail_B = bool(net_gain and not specificity)
    fail_C = bool(
        net_gain
        and specificity
        and not (beats_confidence and beats_uniform)
    )
    conditions = {
        "PASS": decision_pass,
        "FAIL_A": fail_A,
        "FAIL_B": fail_B,
        "FAIL_C": fail_C,
    }
    _require(
        sum(conditions.values()) == 1,
        "C1_DECISION_TREE_LOGICALLY_IMPOSSIBLE_STATE",
    )
    if decision_pass:
        final_decision = "C1_FROZEN_CYCLE_ACTION_PILOT_PASS"
        explanation = (
            "Frozen cycle-verified semantic utility produces "
            "non-redundant and correspondence-specific training benefit."
        )
    elif fail_A:
        final_decision = "C1_CYCLE_ACTION_NO_NET_GAIN"
        explanation = (
            "Frozen Cycle supervision does not produce net training gain."
        )
    elif fail_B:
        final_decision = (
            "C1_ACTION_NOT_CORRESPONDENCE_SPECIFIC"
        )
        explanation = (
            "Cycle action gain is not specific to correct sample "
            "correspondence."
        )
    else:
        final_decision = (
            "C1_CYCLE_ACTION_REDUNDANT_WITH_PSEUDO_BASELINES"
        )
        explanation = (
            "Cycle action does not simultaneously beat confidence "
            "and uniform pseudo-supervision baselines."
        )
    return {
        "C1_NET_GAIN_PASS": net_gain,
        "C1_CORRESPONDENCE_ACTION_SPECIFICITY_PASS": specificity,
        "C1_BEATS_CONFIDENCE_ACTION_PASS": beats_confidence,
        "C1_BEATS_UNIFORM_ACTION_PASS": beats_uniform,
        "comparisons": comparisons,
        "decision_conditions": conditions,
        "final_decision": final_decision,
        "explanation": explanation,
        "dynamic_loop_allowed": False,
        "longer_horizon_multiseed_validation_allowed": decision_pass,
    }


def validate_cycle_zero_equivalence(base_record, cycle_zero_record):
    """Validate the one-epoch BASE/CYCLE_ZERO engineering identity."""
    _require(
        base_record["epochs"] == cycle_zero_record["epochs"] == 1,
        "CYCLE_ZERO control must use one epoch",
    )
    checks = {
        "parameter_aggregate_hash_exact": (
            base_record["final_model_aggregate_sha256"]
            == cycle_zero_record["final_model_aggregate_sha256"]
        ),
        "predictions_exact": (
            base_record["prediction_logical_sha256"]
            == cycle_zero_record["prediction_logical_sha256"]
        ),
        "loss_trajectory_exact": (
            base_record["loss_history"]
            == cycle_zero_record["loss_history"]
        ),
        "metrics_exact": (
            base_record["metrics"] == cycle_zero_record["metrics"]
        ),
    }
    return {
        **checks,
        "CYCLE_ZERO_vs_BASE_1ep_equivalence_pass": bool(
            all(checks.values())
        ),
        "scientific_result_arm": False,
    }


def common_initialization_fairness(train_audits):
    """Require identical initialization/optimizer/seed/order for five arms."""
    _require(
        tuple(train_audits) == FORMAL_ARMS,
        "formal train audit arm set/order mismatch",
    )
    initial_hashes = {
        train_audits[arm]["initial_model_aggregate_sha256"]
        for arm in FORMAL_ARMS
    }
    optimizer_configs = {
        json.dumps(
            train_audits[arm]["optimizer_configuration"],
            sort_keys=True,
        )
        for arm in FORMAL_ARMS
    }
    seeds = {train_audits[arm]["seed"] for arm in FORMAL_ARMS}
    order_hashes = {
        tuple(train_audits[arm]["sample_order_sha256_per_epoch"])
        for arm in FORMAL_ARMS
    }
    checks = {
        "identical_initial_model_hash_all_arms": (
            initial_hashes
            == {EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256}
        ),
        "identical_optimizer_configuration_all_arms": (
            len(optimizer_configs) == 1
        ),
        "identical_seed_all_arms": seeds == {SEED},
        "identical_data_order_all_arms": len(order_hashes) == 1,
    }
    return {
        **checks,
        "initial_model_aggregate_sha256": next(iter(initial_hashes)),
        "same_y_gen_all_pseudo_arms": len({
            train_audits[arm]["pseudo_y_gen_logical_sha256"]
            for arm in PSEUDO_ARMS
        }) == 1,
        "fairness_pass": bool(all(checks.values())),
    }

