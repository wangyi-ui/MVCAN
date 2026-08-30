"""Protocol tests for C1 frozen pseudo-supervision (no training runs)."""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.cyclic_utility import (
    c1_frozen_pseudo_supervision as c1,
)
from experiments.cyclic_utility import (
    summarize_c1_frozen_pseudo_supervision as summarize,
)
from experiments.cyclic_utility import (
    train_c1_frozen_pseudo_supervision as train,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
C0_ARTIFACT = train.DEFAULT_C0_ARTIFACT_PATH
C0_SEAL = train.DEFAULT_C0_SEAL_PATH


@pytest.fixture(scope="session")
def synthetic_c0_arrays():
    rows = torch.arange(c1.SAMPLE_NUM)[:, None]
    directions = torch.arange(c1.DIRECTION_COUNT)[None, :]
    y_gen = ((rows + directions) % c1.CLASS_NUM).long()
    binary = ((rows + directions) % 3 != 0).float()
    alternating = ((rows + directions) % 2 == 0).float()
    return {
        "y_gen": y_gen.detach(),
        "C_conf": torch.full(
            (c1.SAMPLE_NUM, c1.DIRECTION_COUNT), 0.75
        ).detach(),
        "U_cycle": (0.5 * binary).detach(),
        "U_cycle_shuffle": (
            0.25 * torch.roll(alternating, shifts=1, dims=0)
        ).detach(),
        "generator_subsets": torch.zeros(
            c1.DIRECTION_COUNT, 3, dtype=torch.long
        ),
        "verifier_subsets": torch.zeros(
            c1.DIRECTION_COUNT, 3, dtype=torch.long
        ),
    }


@pytest.fixture(scope="session")
def permutation_M0():
    return torch.stack(
        [
            torch.roll(
                torch.eye(c1.CLASS_NUM), shifts=view_id, dims=1
            )
            for view_id in range(c1.VIEW_NUM)
        ],
        dim=0,
    ).detach()


@pytest.fixture(scope="session")
def evidence(synthetic_c0_arrays):
    weights = synthetic_c0_arrays["U_cycle"]
    return c1.build_sample_semantic_evidence(
        synthetic_c0_arrays["y_gen"], weights
    )


@pytest.fixture(scope="session")
def cycle_target(synthetic_c0_arrays, permutation_M0):
    return c1.build_frozen_pseudo_target(
        "CYCLE", synthetic_c0_arrays, permutation_M0
    )[0]


@pytest.fixture(scope="session")
def loaded_c0():
    return c1.load_frozen_c0_artifact(C0_ARTIFACT, C0_SEAL)


@pytest.fixture(scope="session")
def derived_sealed_M0(loaded_c0):
    c1.set_deterministic_seed(c1.SEED)
    views, sample_ids, _ = (
        train.e1_train.load_frozen_feature_artifact(
            train.DEFAULT_FEATURE_PATH,
            train.DEFAULT_FEATURE_AUDIT_PATH,
        )
    )
    assert np.array_equal(
        sample_ids, np.arange(c1.SAMPLE_NUM, dtype=np.int64)
    )
    model, _, _ = train.load_trainable_e1_lwc_model(
        train.DEFAULT_E1_LWC_MODEL_DIR,
        train.DEFAULT_E1_LWC_AUDIT_PATH,
        torch.device("cpu"),
    )
    return c1.derive_frozen_M0(
        model,
        views,
        loaded_c0[0]["native_global_cluster"],
        torch.device("cpu"),
    )


@pytest.mark.parametrize(
    "relative_path,expected",
    tuple(train.FROZEN_SOURCE_SHA256.items()),
)
def test_frozen_source_sha256(relative_path, expected):
    assert c1.file_sha256(REPOSITORY_ROOT / relative_path) == expected


@pytest.mark.parametrize(
    "view_id,expected",
    tuple(enumerate(c1.EXPECTED_E1_CHECKPOINT_SHA256, start=1)),
)
def test_e1_checkpoint_sha256(view_id, expected):
    path = (
        train.DEFAULT_E1_LWC_MODEL_DIR
        / ("Caltech-6V" + str(view_id) + "V.pth")
    )
    assert c1.file_sha256(path) == expected


@pytest.mark.parametrize(
    "path,expected",
    (
        (C0_ARTIFACT, c1.EXPECTED_C0_ARTIFACT_SHA256),
        (C0_SEAL, c1.EXPECTED_C0_SEAL_SHA256),
    ),
)
def test_c0_file_provenance(path, expected):
    assert c1.file_sha256(path) == expected


@pytest.mark.parametrize(
    "name,shape",
    (
        ("y_gen", (1400, 20)),
        ("C_conf", (1400, 20)),
        ("U_cycle", (1400, 20)),
        ("U_cycle_shuffle", (1400, 20)),
        ("native_global_cluster", (1400,)),
        ("generator_subsets", (20, 3)),
        ("verifier_subsets", (20, 3)),
    ),
)
def test_loaded_c0_shapes(loaded_c0, name, shape):
    assert tuple(loaded_c0[0][name].shape) == shape


@pytest.mark.parametrize("name", c1.REQUIRED_C0_ARRAYS)
def test_loaded_c0_arrays_are_detached(loaded_c0, name):
    value = loaded_c0[0][name]
    assert not value.requires_grad
    assert value.grad_fn is None


def test_c0_loader_reads_pre_gt_fields_only(loaded_c0):
    audit = loaded_c0[1]
    assert audit["loaded_pre_GT_arrays_only"]
    assert not audit["GT_derived_correctness_loaded"]
    assert not audit["C0_scores_recomputed"]


def test_sealed_native_global_cluster_provenance(loaded_c0):
    coordinate = loaded_c0[0]["native_global_cluster"]
    audit = loaded_c0[1]
    assert coordinate.shape == (1400,)
    assert coordinate.dtype == torch.int64
    assert c1.tensor_sha256(coordinate.numpy()) == (
        "95b0dc0294272786164088565b78e9514586c833f91dd87375909dc45d68ee88"
    )
    assert audit["native_global_cluster_hash_pass"]
    assert audit["native_global_cluster_dtype"] == "int64"


def test_native_global_cluster_is_coordinate_only(loaded_c0):
    audit = loaded_c0[1]
    assert not audit["native_global_cluster_used_for_weighting"]
    assert not audit["native_global_cluster_used_for_pseudo_label"]
    assert audit["native_global_cluster_used_for_M0_coordinate_only"]
    assert audit["coordinate_provenance_fields"] == [
        "native_global_cluster"
    ]
    assert "native_global_cluster" not in (
        audit["pseudo_supervision_fields"]
    )


def test_native_global_cluster_has_no_GT_source(loaded_c0):
    audit = loaded_c0[1]
    assert audit["loaded_pre_GT_arrays_only"]
    assert not audit["GT_derived_correctness_loaded"]


def test_derive_uses_sealed_global_ids_without_replayed_kmeans():
    source = inspect.getsource(c1.derive_frozen_M0)
    assert "replay_native_global_reference" not in source
    assert "canonical_global_ids.numpy()" in source
    assert "g2_protocol.build_alignment" in source
    assert "model.Match" in source


def test_int64_provenance_guard_precedes_float_cast():
    source = inspect.getsource(c1.derive_frozen_M0)
    hash_position = source.index(
        "alignment_int64_sha = tensor_sha256(alignment_int64)"
    )
    guard_position = source.index(
        "alignment_int64_sha == EXPECTED_M0_LOGICAL_SHA256"
    )
    float_position = source.index("M0 = torch.as_tensor")
    assert hash_position < guard_position < float_position


def test_runner_passes_coordinate_anchor_to_derive():
    source = inspect.getsource(train.run_arm)
    assert 'c0_arrays["native_global_cluster"]' in source
    assert source.index('c0_arrays["native_global_cluster"]') < (
        source.index("train_c1_arm")
    )


def test_coordinate_anchor_never_enters_weight_builder():
    source = inspect.getsource(c1.build_arm_weights)
    assert "native_global_cluster" not in source


def test_rebuilt_int64_M_exact_sealed_hash(derived_sealed_M0):
    M0, audit = derived_sealed_M0
    assert audit["alignment_int64_shape"] == [6, 7, 7]
    assert audit["alignment_int64_logical_sha256"] == (
        c1.EXPECTED_M0_LOGICAL_SHA256
    )
    assert audit["alignment_int64_matches_C0_seal_pass"]
    assert c1.tensor_sha256(M0.long().numpy()) == (
        "aaa7be3a9b4516278e508330299f881b6e7c5855a80cb43101c715a455f5183d"
    )


def test_training_M0_is_exact_float32_cast(derived_sealed_M0):
    M0, audit = derived_sealed_M0
    assert M0.dtype == torch.float32
    assert audit["training_M0_dtype"] == "float32"
    assert audit["training_M0_logical_sha256"] == (
        "b75e0e23e7302919a52eea42e9cb8b0d6dd12ac84c7937dbec0569078c49ee4a"
    )
    assert audit["training_M0_is_exact_01_cast_of_int64_pass"]
    assert torch.equal(M0, M0.long().float())


def test_rebuilt_M0_full_permutation(derived_sealed_M0):
    M0, audit = derived_sealed_M0
    assert torch.all((M0 == 0.0) | (M0 == 1.0))
    assert torch.equal(M0.sum(dim=1), torch.ones(6, 7))
    assert torch.equal(M0.sum(dim=2), torch.ones(6, 7))
    assert audit["full_permutation_pass"]
    assert audit["row_sums_one_pass"]
    assert audit["column_sums_one_pass"]


def test_M0_coordinate_audit_is_complete(derived_sealed_M0):
    _, audit = derived_sealed_M0
    assert audit["canonical_global_coordinate_source"] == (
        "C0_PRE_GT_sealed_native_global_cluster"
    )
    assert audit["sealed_native_global_cluster_hash_pass"]
    assert audit["q_local_argmax_used_for_alignment"]
    assert audit["Match_input_direction"] == (
        "Match(local_ids, global_ids)"
    )
    assert not audit["C1_replayed_KMeans_used_as_coordinate_anchor"]
    assert not audit["GT_used_for_M0"]
    assert not audit["sparse_labels_used_for_M0"]
    assert not audit["corruption_mask_used_for_M0"]
    assert audit["native_global_cluster_used_for_M0_coordinate_only"]


@pytest.mark.parametrize(
    "name",
    ("y_gen", "U_cycle", "C_conf", "U_cycle_shuffle"),
)
def test_scientific_C0_array_hashes_unchanged(loaded_c0, name):
    assert c1.tensor_sha256(loaded_c0[0][name].numpy()) == (
        c1.EXPECTED_C0_ARRAY_SHA256[name]
    )


def test_frozen_source_verifier_passes():
    audit = train.verify_frozen_source_provenance()
    assert audit["all_frozen_sources_unchanged_pass"]
    assert audit["C0_source_unchanged_pass"]
    assert audit["E1_source_unchanged_pass"]
    assert audit["E4_source_unchanged_pass"]


def test_formal_arms_are_frozen():
    assert c1.FORMAL_ARMS == (
        "BASE",
        "UNIFORM",
        "CONF",
        "CYCLE",
        "SHUFFLED_CYCLE",
    )


def test_engineering_arm_is_not_formal():
    assert c1.ENGINEERING_ARM == "CYCLE_ZERO"
    assert c1.ENGINEERING_ARM not in c1.FORMAL_ARMS
    assert c1.ALL_ARMS == c1.FORMAL_ARMS + ("CYCLE_ZERO",)


@pytest.mark.parametrize("arm", c1.ALL_ARMS)
def test_cli_accepts_every_registered_arm(arm):
    args = train.parse_args(["--arm", arm])
    assert args.arm == arm


def test_cli_frozen_defaults():
    args = train.parse_args(["--arm", "BASE"])
    assert args.epochs == 20
    assert args.seed == 20
    assert args.output_dir is None
    assert args.device == "cuda:0"


@pytest.mark.parametrize(
    "arm,expected_source",
    (
        ("UNIFORM", "exact_ones"),
        ("CONF", "C_conf"),
        ("CYCLE", "U_cycle"),
        ("SHUFFLED_CYCLE", "U_cycle_shuffle"),
        ("CYCLE_ZERO", "U_cycle"),
    ),
)
def test_arm_weight_sources(
    synthetic_c0_arrays, arm, expected_source
):
    weights, audit = c1.build_arm_weights(arm, synthetic_c0_arrays)
    assert tuple(weights.shape) == (1400, 20)
    assert audit["weight_source"] == expected_source
    assert not weights.requires_grad


def test_uniform_weights_are_exact_ones(synthetic_c0_arrays):
    weights, _ = c1.build_arm_weights(
        "UNIFORM", synthetic_c0_arrays
    )
    assert torch.equal(weights, torch.ones_like(weights))


def test_base_has_no_weight_or_pseudo_path(synthetic_c0_arrays):
    weights, audit = c1.build_arm_weights(
        "BASE", synthetic_c0_arrays
    )
    assert weights is None
    assert not audit["pseudo_path_constructed"]


def test_all_pseudo_arms_share_exact_y_gen(synthetic_c0_arrays):
    hashes = set()
    for arm in c1.PSEUDO_ARMS:
        weights, _ = c1.build_arm_weights(arm, synthetic_c0_arrays)
        result = c1.build_sample_semantic_evidence(
            synthetic_c0_arrays["y_gen"], weights
        )
        hashes.add(c1.tensor_sha256(result["y_gen"].numpy()))
    assert len(hashes) == 1


def test_evidence_shapes(evidence):
    assert tuple(evidence["evidence"].shape) == (1400, 7)
    assert tuple(evidence["mass"].shape) == (1400,)
    assert tuple(evidence["target_global"].shape) == (1400, 7)
    assert tuple(evidence["sample_strength"].shape) == (1400,)


def test_evidence_exact_formula(synthetic_c0_arrays, evidence):
    one_hot = F.one_hot(
        synthetic_c0_arrays["y_gen"],
        num_classes=c1.CLASS_NUM,
    ).float()
    expected = torch.mean(
        synthetic_c0_arrays["U_cycle"].unsqueeze(-1) * one_hot,
        dim=1,
    )
    assert torch.equal(evidence["evidence"], expected)


def test_mass_equals_directional_weight_mean(
    synthetic_c0_arrays, evidence
):
    expected = synthetic_c0_arrays["U_cycle"].mean(dim=1)
    assert torch.allclose(
        evidence["mass"], expected, rtol=0.0, atol=1e-7
    )


def test_float32_mass_identity_uses_preregistered_numerical_tolerance():
    generator = torch.Generator().manual_seed(20260828)
    weights = (
        0.0137
        + 0.9231
        * torch.rand(
            c1.SAMPLE_NUM,
            c1.DIRECTION_COUNT,
            generator=generator,
            dtype=torch.float32,
        )
    )
    rows = torch.arange(c1.SAMPLE_NUM)[:, None]
    directions = torch.arange(c1.DIRECTION_COUNT)[None, :]
    y_gen = ((3 * rows + 5 * directions) % c1.CLASS_NUM).long()

    result = c1.build_sample_semantic_evidence(y_gen, weights)
    mass_from_evidence = result["evidence"].sum(dim=-1)
    mass_from_weights = weights.mean(dim=1)
    audit = result["mass_identity_audit"]

    assert not torch.equal(mass_from_evidence, mass_from_weights)
    assert audit["mass_identity_theoretical"]
    assert not audit["mass_identity_bitwise_exact"]
    assert audit["mass_identity_numerical_pass"]
    assert audit["mass_identity_rtol"] == 1e-6
    assert audit["mass_identity_atol"] == 1e-7
    assert audit["mass_identity_max_abs_diff"] > 0.0


def test_mass_identity_still_hard_fails_above_tolerance():
    mass_from_evidence = torch.full(
        (c1.SAMPLE_NUM,), 0.25, dtype=torch.float32
    )
    mismatched_directional_mass = mass_from_evidence.clone()
    mismatched_directional_mass[37] += 1e-3

    with pytest.raises(
        RuntimeError,
        match="evidence mass is not mean directional weight",
    ):
        c1.validate_evidence_mass_identity(
            mass_from_evidence, mismatched_directional_mass
        )

def test_pseudo_target_audit_records_mass_identity_fields(
    synthetic_c0_arrays, permutation_M0
):
    _, audit = c1.build_frozen_pseudo_target(
        "CYCLE", synthetic_c0_arrays, permutation_M0
    )

    assert audit["mass_identity_theoretical"]
    assert isinstance(
        audit["mass_identity_bitwise_exact"], bool
    )
    assert audit["mass_identity_numerical_pass"]
    assert audit["mass_identity_rtol"] == c1.MASS_IDENTITY_RTOL
    assert audit["mass_identity_atol"] == c1.MASS_IDENTITY_ATOL
    assert audit["mass_identity_max_abs_diff"] >= 0.0


def test_mass_identity_tolerances_are_fixed_float32_values():
    assert c1.MASS_IDENTITY_RTOL == 1e-6
    assert c1.MASS_IDENTITY_ATOL == 1e-7
    assert c1.MASS_IDENTITY_RTOL != 0.0
    assert c1.MASS_IDENTITY_ATOL != 0.0



def test_positive_mass_targets_normalize(evidence):
    positive = evidence["positive_mass"]
    assert torch.allclose(
        evidence["target_global"][positive].sum(dim=-1),
        torch.ones_like(evidence["mass"][positive]),
        rtol=0.0,
        atol=1e-6,
    )


def test_zero_mass_rows_stay_zero(synthetic_c0_arrays):
    weights = torch.ones(c1.SAMPLE_NUM, c1.DIRECTION_COUNT)
    weights[0] = 0.0
    result = c1.build_sample_semantic_evidence(
        synthetic_c0_arrays["y_gen"], weights
    )
    assert result["mass"][0].item() == 0.0
    assert torch.count_nonzero(result["target_global"][0]).item() == 0
    assert result["sample_strength"][0].item() == 0.0


def test_sample_strength_exact_frozen_formula(evidence):
    mass = evidence["mass"]
    expected = mass / (mass.mean() + c1.MASS_EPSILON)
    assert torch.equal(evidence["sample_strength"], expected)


def test_sample_strength_mean_approximately_one(evidence):
    assert evidence["sample_strength"].mean().item() == pytest.approx(
        1.0, abs=1e-6
    )


@pytest.mark.parametrize(
    "name",
    (
        "y_gen",
        "weights",
        "evidence",
        "mass",
        "target_global",
        "sample_strength",
    ),
)
def test_semantic_evidence_tensors_are_frozen(evidence, name):
    value = evidence[name]
    assert not value.requires_grad
    assert value.grad_fn is None


def test_M0_shape_and_full_permutations(permutation_M0):
    value = c1.validate_M0(permutation_M0)
    assert tuple(value.shape) == (6, 7, 7)
    assert torch.equal(value.sum(dim=1), torch.ones(6, 7))
    assert torch.equal(value.sum(dim=2), torch.ones(6, 7))


def test_M0_rejects_non_permutation(permutation_M0):
    invalid = permutation_M0.clone()
    invalid[0, 0, 0] = 0.5
    with pytest.raises(RuntimeError, match="full 7x7 permutations"):
        c1.validate_M0(invalid)


def test_global_to_local_direction_is_T_at_M(permutation_M0):
    target = torch.zeros(c1.SAMPLE_NUM, c1.CLASS_NUM)
    target[:, 0] = 1.0
    actual = c1.map_global_target_to_local(
        target, permutation_M0
    )
    expected = torch.stack(
        [target @ permutation_M0[v] for v in range(c1.VIEW_NUM)],
        dim=1,
    )
    assert torch.equal(actual, expected)


def test_local_target_shape_and_detach(cycle_target):
    target = cycle_target["target_local"]
    assert tuple(target.shape) == (1400, 6, 7)
    assert not target.requires_grad
    assert target.grad_fn is None


def test_target_mapping_preserves_mass(cycle_target):
    global_mass = cycle_target["target_global"].sum(dim=-1)
    local_mass = cycle_target["target_local"].sum(dim=-1)
    assert torch.allclose(
        local_mass,
        global_mass[:, None].expand(-1, c1.VIEW_NUM),
        rtol=0.0,
        atol=1e-6,
    )


def test_soft_cross_entropy_exact_formula():
    torch.manual_seed(3)
    logits = torch.randn(4, 6, 7)
    q_local = torch.softmax(logits, dim=-1).requires_grad_(True)
    target = torch.softmax(torch.randn(4, 6, 7), dim=-1)
    strength = torch.tensor([0.2, 0.7, 1.3, 1.8])
    actual = c1.soft_pseudo_cross_entropy(
        q_local, target, strength
    )
    expected = torch.mean(
        strength[:, None]
        * (
            -torch.sum(
                target
                * torch.log(torch.clamp(q_local, min=1e-8)),
                dim=-1,
            )
        )
    )
    assert torch.equal(actual, expected)


def test_sample_strength_enters_soft_loss():
    q_local = torch.full((2, 6, 7), 1.0 / 7.0)
    target = torch.zeros_like(q_local)
    target[0, :, 0] = 1.0
    target[1, :, 1] = 1.0
    zero_second = c1.soft_pseudo_cross_entropy(
        q_local, target, torch.tensor([2.0, 0.0])
    )
    zero_first = c1.soft_pseudo_cross_entropy(
        q_local, target, torch.tensor([0.0, 2.0])
    )
    assert zero_second.item() == pytest.approx(zero_first.item())


def test_soft_loss_gradients_only_reach_current_q():
    q_local = torch.full(
        (2, 6, 7), 1.0 / 7.0, requires_grad=True
    )
    target = torch.full((2, 6, 7), 1.0 / 7.0)
    strength = torch.ones(2)
    loss = c1.soft_pseudo_cross_entropy(
        q_local, target, strength
    )
    loss.backward()
    assert q_local.grad is not None
    assert torch.isfinite(q_local.grad).all()
    assert target.grad is None
    assert strength.grad is None


def test_sample_id_lookup_uses_original_rows(cycle_target):
    ids = torch.tensor([1399, 0, 37], dtype=torch.long)
    target, strength = c1.pseudo_batch_by_sample_ids(
        cycle_target, ids
    )
    assert torch.equal(target, cycle_target["target_local"][ids])
    assert torch.equal(
        strength, cycle_target["sample_strength"][ids]
    )


def test_sample_id_lookup_rejects_out_of_range(cycle_target):
    with pytest.raises(RuntimeError, match="sample_ids_batch"):
        c1.pseudo_batch_by_sample_ids(
            cycle_target, torch.tensor([c1.SAMPLE_NUM])
        )


@pytest.mark.parametrize(
    "arm,expected",
    (
        ("BASE", 0.0),
        ("UNIFORM", 0.01),
        ("CONF", 0.01),
        ("CYCLE", 0.01),
        ("SHUFFLED_CYCLE", 0.01),
        ("CYCLE_ZERO", 0.0),
    ),
)
def test_effective_lambda_is_frozen(arm, expected):
    assert c1.effective_lambda_pseudo(arm) == expected


@pytest.mark.parametrize("arm", ("BASE", "CYCLE_ZERO"))
def test_zero_lambda_arms_return_native_loss_identity(arm):
    native = torch.tensor(2.0, requires_grad=True)
    pseudo = torch.tensor(9.0, requires_grad=True)
    total = c1.combine_native_and_pseudo_loss(
        arm, native, pseudo
    )
    assert total is native


@pytest.mark.parametrize(
    "arm", ("UNIFORM", "CONF", "CYCLE", "SHUFFLED_CYCLE")
)
def test_pseudo_arms_add_exact_point_zero_one(arm):
    native = torch.tensor(2.0)
    pseudo = torch.tensor(3.0)
    total = c1.combine_native_and_pseudo_loss(
        arm, native, pseudo
    )
    assert total.item() == pytest.approx(2.03)


def test_prediction_seal_requires_canonical_ids():
    predictions = np.arange(c1.SAMPLE_NUM) % c1.CLASS_NUM
    payload = c1.prediction_seal_payload(
        predictions, np.arange(c1.SAMPLE_NUM)
    )
    assert np.array_equal(payload["predictions"], predictions)
    with pytest.raises(RuntimeError, match="prediction seal"):
        c1.prediction_seal_payload(
            predictions, np.arange(c1.SAMPLE_NUM)[::-1]
        )


@pytest.mark.parametrize(
    "candidate,comparator,expected",
    (
        (
            {"ACC": 2.0, "NMI": 2.0, "ARI": 0.0},
            {"ACC": 1.0, "NMI": 1.0, "ARI": 1.0},
            True,
        ),
        (
            {"ACC": 2.0, "NMI": 2.0, "ARI": -1.0},
            {"ACC": 1.0, "NMI": 1.0, "ARI": 1.0},
            False,
        ),
        (
            {"ACC": 2.0, "NMI": 1.0, "ARI": 1.0},
            {"ACC": 1.0, "NMI": 1.0, "ARI": 1.0},
            False,
        ),
    ),
)
def test_metric_gate_exact(candidate, comparator, expected):
    result = c1.metric_delta_record(candidate, comparator)
    assert result["pass"] is expected


def metrics_by_arm(
    base=0.1,
    uniform=0.2,
    conf=0.3,
    cycle=0.8,
    shuffled=0.4,
):
    value = lambda score: {
        "ACC": score,
        "NMI": score,
        "ARI": score,
    }
    return {
        "BASE": value(base),
        "UNIFORM": value(uniform),
        "CONF": value(conf),
        "CYCLE": value(cycle),
        "SHUFFLED_CYCLE": value(shuffled),
    }


@pytest.mark.parametrize(
    "metrics,expected",
    (
        (
            metrics_by_arm(),
            "C1_FROZEN_CYCLE_ACTION_PILOT_PASS",
        ),
        (
            metrics_by_arm(base=0.9),
            "C1_CYCLE_ACTION_NO_NET_GAIN",
        ),
        (
            metrics_by_arm(shuffled=0.9),
            "C1_ACTION_NOT_CORRESPONDENCE_SPECIFIC",
        ),
        (
            metrics_by_arm(conf=0.9),
            "C1_CYCLE_ACTION_REDUNDANT_WITH_PSEUDO_BASELINES",
        ),
    ),
)
def test_decision_tree_is_exhaustive_and_prioritized(metrics, expected):
    result = c1.build_c1_pilot_decision(metrics)
    assert result["final_decision"] == expected
    assert sum(result["decision_conditions"].values()) == 1


def test_four_gate_names_are_exact():
    result = c1.build_c1_pilot_decision(metrics_by_arm())
    assert result["C1_NET_GAIN_PASS"]
    assert result["C1_CORRESPONDENCE_ACTION_SPECIFICITY_PASS"]
    assert result["C1_BEATS_CONFIDENCE_ACTION_PASS"]
    assert result["C1_BEATS_UNIFORM_ACTION_PASS"]


def test_cycle_zero_equivalence_checks_all_outputs():
    common = {
        "epochs": 1,
        "final_model_aggregate_sha256": "model",
        "prediction_logical_sha256": "prediction",
        "loss_history": [{"loss": 1.0}],
        "metrics": {"ACC": 1.0, "NMI": 1.0, "ARI": 1.0},
    }
    result = c1.validate_cycle_zero_equivalence(
        common, dict(common)
    )
    assert result["CYCLE_ZERO_vs_BASE_1ep_equivalence_pass"]
    assert not result["scientific_result_arm"]


def test_common_initialization_fairness_contract():
    audits = {}
    optimizer = {
        "policy": "fresh",
        "learning_rate": c1.LEARNING_RATE,
    }
    for arm in c1.FORMAL_ARMS:
        audits[arm] = {
            "initial_model_aggregate_sha256": (
                c1.EXPECTED_E1_INITIAL_MODEL_AGGREGATE_SHA256
            ),
            "optimizer_configuration": optimizer,
            "seed": c1.SEED,
            "sample_order_sha256_per_epoch": ["same"],
            "pseudo_y_gen_logical_sha256": (
                None
                if arm == "BASE"
                else c1.EXPECTED_C0_ARRAY_SHA256["y_gen"]
            ),
        }
    result = c1.common_initialization_fairness(audits)
    assert result["fairness_pass"]
    assert result["same_y_gen_all_pseudo_arms"]


def test_runner_precomputes_pseudo_target_before_training():
    source = inspect.getsource(train.run_arm)
    assert source.index("build_frozen_pseudo_target") < source.index(
        "train_c1_arm"
    )


def test_training_uses_explicit_sample_ids_not_batch_index():
    source = inspect.getsource(train.train_c1_arm)
    assert "sample_ids_batch" in source
    assert "pseudo_batch_by_sample_ids" in source
    tree = ast.parse(source)
    assert all(
        node.id != "batch_idx"
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    )


def test_native_graph_and_lwc_calls_are_preserved():
    source = inspect.getsource(train.train_c1_arm)
    assert "refresh_native_target" in source
    assert "native_mvcan_losses" in source
    assert "build_lwc_loss" in source
    assert "native_p_all" in source
    assert "native_matches" in source


def test_no_R_loader_is_called_by_c1_runner():
    tree = ast.parse(inspect.getsource(train))
    called_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Load)
    }
    assert "load_frozen_reliability" not in called_attributes
    assert "load_sparse_label_protocol" not in called_attributes
    assert "load_oracle_clean_labeled_weights" not in called_attributes


def test_full_gt_loader_occurs_after_prediction_seal():
    source = inspect.getsource(train.run_arm)
    assert source.index("save_predictions_before_GT") < source.index(
        "load_labels_after_predictions"
    )


def test_common_acc_nmi_ari_evaluator_is_reused():
    source = inspect.getsource(train.run_arm)
    assert "e1_train.evaluate_predictions" in source
    assert c1.METRIC_NAMES == ("ACC", "NMI", "ARI")


def test_forbidden_pseudo_operators_are_not_in_target_builder():
    tree = ast.parse(
        inspect.getsource(c1.build_frozen_pseudo_target)
    )
    call_names = {
        (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
        ).lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert call_names.isdisjoint(
        {"topk", "threshold", "temperature", "entropy", "ema"}
    )


def test_cycle_zero_constructs_cycle_path_with_zero_lambda():
    target_source = inspect.getsource(c1.build_arm_weights)
    runner_source = inspect.getsource(train.train_c1_arm)
    assert '"CYCLE_ZERO": "U_cycle"' in target_source
    assert 'if arm != "BASE"' in runner_source
    assert c1.effective_lambda_pseudo("CYCLE_ZERO") == 0.0


def test_required_per_arm_output_schema_is_exact():
    assert summarize.REQUIRED_ARM_FILES == (
        "metrics.json",
        "train_audit.json",
        "pseudo_target_audit.json",
        "final_predictions.npz",
        "loss_history.json",
    )


def test_required_root_output_schema_is_exact():
    assert summarize.SUMMARY_FILES == (
        "c1_summary.json",
        "c1_decision.json",
        "c1_audit.json",
    )


def test_summarizer_accepts_json_sorted_metric_key_order():
    source = inspect.getsource(summarize.load_arm_bundle)
    assert "set(metrics) == set(c1.METRIC_NAMES)" in source
    assert tuple(sorted(c1.METRIC_NAMES)) == ("ACC", "ARI", "NMI")


def test_summarizer_excludes_cycle_zero_from_scientific_arms():
    source = inspect.getsource(summarize.summarize)
    assert "c1.FORMAL_ARMS" in source
    assert "engineering_arm_excluded" in source


def test_base_pseudo_target_is_absent(permutation_M0, synthetic_c0_arrays):
    target, audit = c1.build_frozen_pseudo_target(
        "BASE", synthetic_c0_arrays, permutation_M0
    )
    assert target is None
    assert not audit["pseudo_supervision_used"]
    assert audit["lambda_pseudo"] == 0.0


def test_cycle_pseudo_audit_records_all_frozen_shapes(
    permutation_M0, synthetic_c0_arrays
):
    _, audit = c1.build_frozen_pseudo_target(
        "CYCLE", synthetic_c0_arrays, permutation_M0
    )
    assert audit["shapes"] == {
        "evidence": [1400, 7],
        "mass": [1400],
        "target_global": [1400, 7],
        "sample_strength": [1400],
        "M0": [6, 7, 7],
        "target_local": [1400, 6, 7],
    }
    assert audit["all_pseudo_tensors_detached_pass"]
    assert audit["global_to_local_direction"] == "T_global @ M0"
    assert audit["local_to_global_direction"] == "q_local @ M0.T"

