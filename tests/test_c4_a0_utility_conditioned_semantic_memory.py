import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from experiments.cyclic_utility import c0_complementary_semantic_verification as c0
from experiments.cyclic_utility import c2_a0_sparse_label_utility_protocol as c2
from experiments.cyclic_utility import (
    c4_a0_utility_conditioned_semantic_memory_protocol as c4,
)
from experiments.cyclic_utility import (
    evaluate_c4_a0_utility_conditioned_semantic_memory as evaluate,
)
from experiments.cyclic_utility import (
    run_c4_a0_utility_conditioned_semantic_memory as runner,
)
from irv.b4_information_utility import tensor_sha256
from weak_quality import ndarray_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def split_ids():
    labeled = c2.fixed_labeled_ids()
    unlabeled = np.setdiff1d(np.arange(c4.N, dtype=np.int64), labeled)
    return labeled, c2.fixed_labeled_targets(), unlabeled


@pytest.fixture(scope="module")
def memory_split(split_ids):
    return c4.fixed_memory_split(split_ids[2])


@pytest.fixture(scope="module")
def semantic_q(split_ids):
    labeled, targets, _ = split_ids
    classes = np.arange(c4.N, dtype=np.int64) % c4.K
    classes[labeled] = targets
    q = torch.nn.functional.one_hot(
        torch.from_numpy(classes), num_classes=c4.K
    ).to(torch.float64)
    return q[:, None, :].repeat(1, c4.V, 1).detach()


@pytest.fixture(scope="module")
def identity_matches():
    return torch.eye(c4.K, dtype=torch.float64).repeat(c4.V, 1, 1)


@pytest.fixture(scope="module")
def eligibility(semantic_q, memory_split, split_ids):
    return c4.freeze_epoch1_writer_eligibility(
        semantic_q, memory_split["W"], split_ids[0], split_ids[1]
    )


@pytest.fixture(scope="module")
def utility(split_ids):
    values = np.arange(c4.UNLABELED_COUNT * c4.S, dtype=np.float64)
    return values.reshape(c4.UNLABELED_COUNT, c4.S)


def test_01_exact_20_direction_order():
    generator, verifier = c4.frozen_direction_definitions()
    expected = np.asarray([
        [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 2, 3],
        [0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5], [0, 4, 5],
        [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5],
        [1, 4, 5], [2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 5],
    ], dtype=np.int64)
    assert np.array_equal(generator, expected)
    assert np.array_equal(np.sort(np.concatenate((generator, verifier), axis=1)),
                          np.tile(np.arange(6), (20, 1)))


def test_02_generator_verifier_hash_match():
    generator, verifier = c4.frozen_direction_definitions()
    assert tensor_sha256(generator) == c4.EXPECTED_GENERATOR_SUBSETS_SHA256
    assert tensor_sha256(verifier) == c4.EXPECTED_VERIFIER_SUBSETS_SHA256


def test_03_q_alignment_uses_transpose(identity_matches):
    q = torch.zeros((c4.N, c4.V, c4.K), dtype=torch.float64)
    q[..., 1] = 1.0
    matches = identity_matches.clone()
    matches[0] = torch.eye(c4.K, dtype=torch.float64)[[1, 2, 0, 3, 4, 5, 6]]
    aligned = c4.align_q_readonly(q, matches)
    expected = q[:, 0, :] @ matches[0].T
    assert torch.equal(aligned[:, 0, :], expected)
    assert not torch.equal(aligned[:, 0, :], q[:, 0, :] @ matches[0])


def test_04_g_direction_matches_c0_p_gen(semantic_q):
    actual = c4.build_directional_payload(semantic_q)
    expected = c0.build_complementary_posteriors(semantic_q)["p_gen"]
    assert torch.allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_05_holdout_count(memory_split):
    assert memory_split["H"].shape == (277,)


def test_06_writer_pool_count(memory_split):
    assert memory_split["W"].shape == (1109,)


def test_07_exact_holdout_writer_hashes(memory_split):
    assert ndarray_sha256(memory_split["H"]) == c4.EXPECTED_HOLDOUT_SHA256
    assert ndarray_sha256(memory_split["W"]) == c4.EXPECTED_WRITER_POOL_SHA256


def test_08_holdout_writer_disjoint_exhaustive(memory_split, split_ids):
    assert np.intersect1d(memory_split["H"], memory_split["W"]).size == 0
    assert np.array_equal(
        np.sort(np.concatenate((memory_split["H"], memory_split["W"]))), split_ids[2]
    )


def test_09_labeled_ids_excluded_from_H_W(memory_split, split_ids):
    assert np.intersect1d(split_ids[0], memory_split["H"]).size == 0
    assert np.intersect1d(split_ids[0], memory_split["W"]).size == 0


def test_10_fixed_budget_is_15():
    assert int(np.floor(c4.WRITE_FRACTION * c4.WRITER_POOL_COUNT / c4.K)) == 15
    assert c4.WRITERS_PER_CLASS == 15


def test_11_semantic_eligibility_independent_of_U(eligibility):
    assert eligibility["U_cycle_used"] is False
    assert "U_cycle" not in inspect.signature(c4.freeze_epoch1_writer_eligibility).parameters


def test_12_small_class_pool_fails_closed(memory_split, split_ids):
    """Writer-pool gate must still fail after anchor consistency passes.

    Construct:
    - every sparse labeled anchor is perfectly consistent with its own class;
    - every unlabeled writer points to semantic class 0.

    Therefore the anchor-consistency gate passes for all 14 anchors,
    but classes 1..6 have writer pools smaller than WRITERS_PER_CLASS.
    The expected failure must come from the original writer-pool gate.
    """
    q = torch.zeros(
        (c4.N, c4.V, c4.K),
        dtype=torch.float64,
    )

    # All non-labeled rows point to class 0.
    q[:, :, 0] = 1.0

    labeled_ids = split_ids[0]
    labeled_targets = split_ids[1]

    # Make the two sparse anchors of every class perfectly consistent:
    #
    # q_l = one_hot(y_l) on every view.
    #
    # Then for every labeled anchor:
    # own-class LOO relation = 1
    # competing-class relation = 0
    # anchor-consistency gap = +1
    for sample_id, target in zip(
        labeled_ids,
        labeled_targets,
    ):
        sid = int(sample_id)
        cls = int(target)

        q[sid, :, :] = 0.0
        q[sid, :, cls] = 1.0

    with pytest.raises(
        RuntimeError,
        match="PRE_GT_PROTOCOL_FAIL_CLOSED",
    ):
        c4.freeze_epoch1_writer_eligibility(
            q,
            memory_split["W"],
            labeled_ids,
            labeled_targets,
        )


def test_12b_anchor_consistency_gate_fails_closed(
    memory_split,
    split_ids,
):
    """Uninformative sparse anchors must fail before writer admission.

    Uniform semantic posteriors give zero LOO margin for every anchor:

        own_score == best_other_score

    Since C4-A0-v2 admits propagation anchors only when:

        LOO own - best_other > 0

    this state must fail closed at the anchor-consistency gate.
    """
    q = torch.full(
        (c4.N, c4.V, c4.K),
        1.0 / c4.K,
        dtype=torch.float64,
    )

    with pytest.raises(
        RuntimeError,
        match="C4_ANCHOR_CONSISTENCY_FAIL_CLOSED",
    ):
        c4.freeze_epoch1_writer_eligibility(
            q,
            memory_split["W"],
            split_ids[0],
            split_ids[1],
        )

def test_13_writer_tie_break_is_sample_id_ascending(
    eligibility, memory_split, semantic_q,
):
    writer_index = torch.from_numpy(np.array(memory_split["W"], copy=True))
    writer_classes = torch.argmax(semantic_q[writer_index, 0], dim=1).numpy()
    for class_id in range(c4.K):
        expected = np.sort(memory_split["W"][writer_classes == class_id])[:15]
        assert np.array_equal(eligibility["E_c"][class_id], expected)


def test_14_E_c_is_frozen_after_epoch1(eligibility):
    assert eligibility["eligibility_epoch"] == 1
    assert eligibility["eligibility_frozen_after_epoch1"] is True
    assert not eligibility["E_c"].flags.writeable


def test_15_same_E_c_is_shared_by_all_write_arms():
    source = inspect.getsource(runner.run_pre_gt)
    assert source.count('eligibility["E_c"]') >= 3
    assert source.count("freeze_epoch1_writer_eligibility(") == 1


def test_16_A0_Z0_M0_exact_definition(semantic_q, split_ids):
    state = c4.initialize_label_memory(semantic_q, split_ids[0], split_ids[1])
    assert torch.equal(state["A"], 12.0 * torch.eye(c4.K, dtype=torch.float64))
    assert torch.equal(state["Z"], torch.full((c4.K,), 12.0, dtype=torch.float64))
    assert torch.equal(state["M"], torch.eye(c4.K, dtype=torch.float64))


def test_17_all_arms_bitwise_identical_independent_initialization(semantic_q, split_ids):
    initial = c4.initialize_label_memory(semantic_q, split_ids[0], split_ids[1])
    arms = c4.clone_initial_memory_arms(initial)
    for arm in c4.MEMORY_ARMS[1:]:
        for name in ("A", "Z", "M"):
            assert torch.equal(arms["LABEL_ONLY"][name], arms[arm][name])
            assert arms["LABEL_ONLY"][name].data_ptr() != arms[arm][name].data_ptr()


def test_18_label_only_never_writes(semantic_q, split_ids, eligibility, utility):
    initial = c4.initialize_label_memory(semantic_q, split_ids[0], split_ids[1])
    arms = c4.clone_initial_memory_arms(initial)
    controls = c4.build_c4_writer_utilities(utility, split_ids[2], eligibility["E_c"])
    payload = c4.build_directional_payload(semantic_q)
    updated = c4.apply_memory_event(arms, payload, eligibility["E_c"], controls)
    for name in ("A", "Z", "M"):
        assert torch.equal(updated["LABEL_ONLY"][name], initial[name])


def test_19_U_lookup_is_by_sample_ID_not_global_position(split_ids, utility):
    chosen = np.asarray([68, 83, 1000], dtype=np.int64)
    actual = c4.lookup_utility_by_sample_ids(utility, split_ids[2], chosen)
    expected_rows = [np.flatnonzero(split_ids[2] == value)[0] for value in chosen]
    assert np.array_equal(actual, utility[expected_rows])
    assert not np.array_equal(actual, utility[chosen])


def test_20_c4_shuffle_has_no_fixed_point(eligibility, utility, split_ids):
    controls = c4.build_c4_writer_utilities(utility, split_ids[2], eligibility["E_c"])
    assert np.all(controls["permutation_ids"] != eligibility["E_c"])


def test_21_shuffle_preserves_utility_row_multiset(eligibility, utility, split_ids):
    controls = c4.build_c4_writer_utilities(utility, split_ids[2], eligibility["E_c"])
    for class_id in range(c4.K):
        true_rows = sorted(map(tuple, controls["true"][class_id]))
        shuffle_rows = sorted(map(tuple, controls["shuffle"][class_id]))
        assert true_rows == shuffle_rows


def test_22_shuffle_preserves_each_direction_multiset(eligibility, utility, split_ids):
    controls = c4.build_c4_writer_utilities(utility, split_ids[2], eligibility["E_c"])
    assert np.array_equal(np.sort(controls["true"], axis=1),
                          np.sort(controls["shuffle"], axis=1))


def test_23_true_shuffle_total_mass_equal(eligibility, utility, split_ids):
    controls = c4.build_c4_writer_utilities(utility, split_ids[2], eligibility["E_c"])
    assert np.array_equal(
        np.sort(controls["true"], axis=None), np.sort(controls["shuffle"], axis=None)
    )


def test_24_true_U_keeps_sample_direction_tensor(eligibility, utility, split_ids):
    controls = c4.build_c4_writer_utilities(utility, split_ids[2], eligibility["E_c"])
    expected = c4.lookup_utility_by_sample_ids(utility, split_ids[2], eligibility["E_c"])
    assert controls["true"].shape == (7, 15, 20)
    assert np.array_equal(controls["true"], expected)


def test_25_no_20_to_6_mapping_exists(eligibility, utility, split_ids):
    controls = c4.build_c4_writer_utilities(utility, split_ids[2], eligibility["E_c"])
    assert controls["true"].shape[-1] == c4.S
    assert c4.S != c4.V


def test_26_no_scalar_U_i_generated(eligibility, utility, split_ids):
    controls = c4.build_c4_writer_utilities(utility, split_ids[2], eligibility["E_c"])
    assert all(value.ndim == 3 for key, value in controls.items()
               if key in ("true", "shuffle"))


def test_27_DeltaA_DeltaZ_exact_formula():
    payload = torch.arange(7 * 15 * 20 * 7, dtype=torch.float64).reshape(7, 15, 20, 7)
    weights = torch.arange(7 * 15 * 20, dtype=torch.float64).reshape(7, 15, 20)
    delta_A, delta_Z = c4.directional_write_delta(payload, weights)
    assert torch.equal(delta_A, (weights[..., None] * payload).sum((1, 2)) / 20.0)
    assert torch.equal(delta_Z, weights.sum((1, 2)) / 20.0)


def test_28_persistent_A_Z_accumulates_20_events(semantic_q, split_ids, eligibility, utility):
    initial = c4.initialize_label_memory(semantic_q, split_ids[0], split_ids[1])
    arms = c4.clone_initial_memory_arms(initial)
    controls = c4.build_c4_writer_utilities(utility, split_ids[2], eligibility["E_c"])
    payload = c4.build_directional_payload(semantic_q)
    one = c4.apply_memory_event(arms, payload, eligibility["E_c"], controls)
    delta_A = one["TRUE_U_WRITE"]["A"] - initial["A"]
    delta_Z = one["TRUE_U_WRITE"]["Z"] - initial["Z"]
    for _ in range(20):
        arms = c4.apply_memory_event(arms, payload, eligibility["E_c"], controls)
    assert torch.equal(arms["TRUE_U_WRITE"]["A"], initial["A"] + 20 * delta_A)
    assert torch.equal(arms["TRUE_U_WRITE"]["Z"], initial["Z"] + 20 * delta_Z)


def test_29_q_U_g_are_detached(semantic_q, identity_matches, eligibility, utility, split_ids):
    q = semantic_q.clone().requires_grad_(True)
    aligned = c4.align_q_readonly(q, identity_matches)
    payload = c4.build_directional_payload(aligned)
    controls = c4.build_c4_writer_utilities(utility, split_ids[2], eligibility["E_c"])
    weight = c4.memory_write_weights("TRUE_U_WRITE", controls, dtype=torch.float64, device="cpu")
    assert not aligned.requires_grad and not payload.requires_grad and not weight.requires_grad


def test_30_readout_z_is_uniform_six_view_mean(semantic_q, memory_split):
    z = c4.holdout_representation(semantic_q, memory_split["H"])
    assert torch.equal(z, semantic_q[torch.from_numpy(np.array(memory_split["H"], copy=True))].mean(dim=1))


def test_31_readout_API_contains_no_U():
    assert "U" not in inspect.signature(c4.holdout_representation).parameters
    assert "U" not in inspect.signature(c4.memory_readout).parameters


def test_32_same_z_used_across_four_arms(semantic_q, memory_split, split_ids):
    state = c4.initialize_label_memory(semantic_q, split_ids[0], split_ids[1])
    arms = c4.clone_initial_memory_arms(state)
    z = c4.holdout_representation(semantic_q, memory_split["H"])
    output = c4.memory_readout(z, arms)
    assert all(torch.equal(output["LABEL_ONLY"]["scores"], output[arm]["scores"])
               for arm in c4.MEMORY_ARMS)


def test_33_final_readout_snapshot_is_post_write20_pre_PhaseB():
    source = inspect.getsource(runner.run_pre_gt)
    readout = source.index("epoch20_z =")
    phase_b = source.index("c3_train.native_consolidation_phase(", readout)
    assert readout < phase_b
    assert '"epoch20_post_write_pre_PhaseB"' in source


def test_34_final_native_refresh_not_used_for_C4_readout():
    source = inspect.getsource(runner.run_pre_gt)
    readout = source.index("c4_readout =")
    final_refresh = source.rindex("e1_train.refresh_native_target(")
    assert readout < final_refresh
    assert '"final_native_refresh_used_for_C4": False' in source


def test_35_semantic_identity_safety():
    identity = torch.eye(c4.K, dtype=torch.float64)
    similarity = c4.semantic_identity_safety(identity, identity)
    assert torch.equal(torch.argmax(similarity, dim=1), torch.arange(c4.K))


@pytest.mark.parametrize("bad", ["zero", "nan", "duplicate"])
def test_36_nonfinite_zero_norm_collapse_fail_closed(bad):
    initial = torch.eye(c4.K, dtype=torch.float64)
    final = initial.clone()
    if bad == "zero":
        final[0] = 0
    elif bad == "nan":
        final[0, 0] = float("nan")
    else:
        final[1] = final[0]
    with pytest.raises(RuntimeError):
        c4.semantic_identity_safety(final, initial)


def test_37_pre_GT_whitelist_excludes_GT():
    assert not any("gt" in name.lower() for name in c4.PRE_GT_ARRAY_WHITELIST)
    invalid = {name: np.empty(0) for name in c4.PRE_GT_ARRAY_WHITELIST}
    invalid["full_GT"] = np.empty(0)
    with pytest.raises(RuntimeError):
        c4.validate_pre_gt_array_whitelist(invalid)


def test_38_evaluator_requires_sealed_artifact_before_GT(tmp_path, monkeypatch):
    called = {"GT": False}
    def forbidden(*args, **kwargs):
        called["GT"] = True
        raise AssertionError("GT boundary reached")
    monkeypatch.setattr(evaluate.e1_train, "load_labels_after_predictions", forbidden)
    with pytest.raises(RuntimeError, match="sealed C4-A0 pre-GT artifact"):
        evaluate.evaluate(20, tmp_path / "missing.npz", tmp_path / "missing.json",
                          tmp_path / "missing-seal.json", tmp_path / "GT.mat",
                          tmp_path / "metrics.json")
    assert called["GT"] is False


def test_39_carrier_parity_failure_blocks_evaluation_path():
    model = type("Carrier", (), {"autoencoders": [nn.Linear(2, 2)]})()
    reference = {
        "final_model_hash": {"aggregate": "wrong", "per_view": []},
        "predictions": np.zeros(c4.N, dtype=np.int64),
        "sample_ids": np.arange(c4.N, dtype=np.int64),
    }
    with pytest.raises(RuntimeError, match="C4_CARRIER_PARITY_FAIL_CLOSED"):
        runner.verify_carrier_parity(
            model, np.zeros(c4.N, dtype=np.int64), np.arange(c4.N), reference
        )


def test_40_seed_specific_E1_carrier_provenance_enforced():
    for seed in c4.SEEDS:
        model_dir, audit_path = runner.seed_specific_e1_carrier_paths(seed)
        assert model_dir.name == "models"
        assert model_dir.parent.name == "lwc_100ep_seed" + str(seed)
        assert audit_path == model_dir.parent / "e1_audit.json"
        valid = {
            "source_seed": seed, "source_training_seed": seed,
            "requested_seed_lineage_match_pass": True,
            "model_matches_own_audit_pass": True,
        }
        assert runner.validate_carrier_lineage(seed, valid, {"seed": seed})[
            "all_seed_lineage_equal_pass"
        ]
    with pytest.raises(RuntimeError):
        runner.validate_carrier_lineage(30, {
            "source_seed": 20, "source_training_seed": 20,
            "requested_seed_lineage_match_pass": True,
            "model_matches_own_audit_pass": True,
        }, {"seed": 30})


def test_41_protocol_has_no_optimizer_step_or_backward():
    source = inspect.getsource(c4)
    assert ".step(" not in source
    assert ".backward(" not in source


def test_42_protocol_has_no_trainable_C4_parameters():
    source = inspect.getsource(c4)
    assert "nn.Parameter" not in source
    assert "torch.optim" not in source


def test_43_frozen_C3_and_C0_sources_are_unchanged():
    expected = {
        "experiments/cyclic_utility/c0_complementary_semantic_verification.py":
            "d52fbf0816557ba57a11fc490ead1a26598b35e68bac7808b7077c448bc7a4a9",
        "experiments/cyclic_utility/c3_a0_utility_conditioned_action_granularity_protocol.py":
            "8182370acd4cda507bfa425a3f40e05779945e49359d0b6c3c80b88472cb1115",
        "experiments/cyclic_utility/c3_b0_relation_action_protocol.py":
            "9ae55b93c2803d8dea594fed1cf3f26e32e8963779a87ee5027cdf9ae3340b1d",
        "experiments/cyclic_utility/train_c3_b0_relation_action_pilot.py":
            "0a657cb6e5faeaf21478c6bb7bd626bbd8385343408185d3730eb4b366aa8d50",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest() == digest


def test_44_snapshot_state_comparison_detects_change():
    base = {
        "model_hash": {"aggregate": "a"}, "optimizer_hash": "b",
        "buffers_hash": "c", "cpu_rng": torch.tensor([1], dtype=torch.uint8),
        "cuda_rng": [], "native_P_hash": "d", "native_M_hash": "e",
        "view_weights_hash": "f", "training_flags": ((0, "", True),),
    }
    assert runner.assert_side_effect_state_equal(base, dict(base))["CPU_RNG_unchanged"]
    changed = dict(base)
    changed["native_M_hash"] = "changed"
    with pytest.raises(RuntimeError, match="C4_SIDE_EFFECT_FAIL_CLOSED"):
        runner.assert_side_effect_state_equal(base, changed)


def test_45_holdout_hash_payload_is_exact(memory_split):
    first = int(memory_split["H"][0])
    payload = ("C4A0_HOLDOUT_V1:" + str(first)).encode("utf-8")
    ranked_first = min(
        (hashlib.sha256(("C4A0_HOLDOUT_V1:" + str(i)).encode("utf-8")).hexdigest(), i)
        for i in np.concatenate((memory_split["H"], memory_split["W"]))
    )
    assert (hashlib.sha256(payload).hexdigest(), first) == ranked_first


def test_46_shuffle_moves_entire_direction_vector(eligibility, utility, split_ids):
    controls = c4.build_c4_writer_utilities(utility, split_ids[2], eligibility["E_c"])
    for class_id in range(c4.K):
        for position, source_id in enumerate(controls["permutation_ids"][class_id]):
            expected = c4.lookup_utility_by_sample_ids(
                utility, split_ids[2], np.asarray([source_id])
            )[0]
            assert np.array_equal(controls["shuffle"][class_id, position], expected)


def test_47_label_only_skips_all_twenty_events(semantic_q, split_ids, eligibility, utility):
    initial = c4.initialize_label_memory(semantic_q, split_ids[0], split_ids[1])
    arms = c4.clone_initial_memory_arms(initial)
    controls = c4.build_c4_writer_utilities(utility, split_ids[2], eligibility["E_c"])
    payload = c4.build_directional_payload(semantic_q)
    for _ in range(c4.TRAIN_EPOCHS):
        arms = c4.apply_memory_event(arms, payload, eligibility["E_c"], controls)
    assert torch.equal(arms["LABEL_ONLY"]["A"], initial["A"])
    assert torch.equal(arms["LABEL_ONLY"]["Z"], initial["Z"])
    assert torch.equal(arms["LABEL_ONLY"]["M"], initial["M"])
