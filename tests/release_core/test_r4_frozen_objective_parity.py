import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.generic_contract import generic_relation_action as authoritative
from release_core.action import (
    build_action_weight,
    relation_bce,
    utility_conditioned_relation_loss,
)
from release_core.semantics import compute_view_relations


ARCHIVE = Path("/root/autodl-tmp/CVPR24-MVCAN")
CALTECH_ACTION = (
    ARCHIVE / "outputs/cyclic_utility"
    / "c3_a0_utility_conditioned_action_granularity_seed20"
    / "c3_a0_action_pre_gt.npz"
)
CALTECH_TRAJECTORY = (
    ARCHIVE / "outputs/final_core/f0_a1_formal_seed20_true_u"
    / "f0_a1_trajectory_pre_gt.npz"
)
CALTECH_AUDIT = CALTECH_TRAJECTORY.with_name("f0_a1_trajectory_audit.json")
MSRC = (
    ARCHIVE / "outputs/generic_contract"
    / "g0_b0_msrc_structural_pilot_seed20_20260916"
    / "msrc_structural_pre_gt_artifact.npz"
)
BDGP = (
    ARCHIVE / "outputs/generic_contract"
    / "g0_b1_bdgp_structural_pilot_seed20_20260917"
    / "bdgp_structural_pre_gt_artifact.npz"
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ndarray_sha256(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(size) for size in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load(path, names):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in names}


def _aligned_views(arrays):
    sample_ids = arrays["sample_ids"]
    row_lookup = {int(sample_id): row for row, sample_id in enumerate(sample_ids)}
    query_rows = np.asarray(
        [row_lookup[int(value)] for value in arrays["unlabeled_ids"]],
        dtype=np.int64,
    )
    anchor_rows = np.asarray(
        [row_lookup[int(value)] for value in arrays["labeled_ids"]],
        dtype=np.int64,
    )
    q = arrays["q_local"]
    queries = [torch.from_numpy(q[query_rows, view, :]) for view in range(q.shape[1])]
    anchors = [torch.from_numpy(q[anchor_rows, view, :]) for view in range(q.shape[1])]
    return queries, anchors, arrays["U_cycle"][query_rows]


def _assert_exact_objective(queries, anchors, target, cycle, balance):
    clean_weight = build_action_weight(cycle, balance, like=queries[0])
    expected_cycle = torch.as_tensor(cycle, dtype=queries[0].dtype)
    expected_balance = torch.as_tensor(balance, dtype=queries[0].dtype)
    expected_weight = (expected_cycle[:, None, :] * expected_balance).detach()
    assert torch.equal(clean_weight, expected_weight)
    clean_denominator = clean_weight.sum()
    expected_denominator = expected_weight.sum()
    assert torch.equal(clean_denominator, expected_denominator)

    expected_views = []
    for query, anchor in zip(queries, anchors):
        clean_probability = compute_view_relations(query, anchor)
        expected_probability = authoritative.relation_probability(query, anchor)
        assert torch.equal(clean_probability, expected_probability)
        clean_bce = relation_bce(clean_probability, target)
        epsilon = torch.finfo(query.dtype).eps
        probability = expected_probability.clamp(
            min=epsilon, max=1.0 - epsilon
        ).unsqueeze(-1)
        converted = torch.as_tensor(target, dtype=query.dtype)
        expected_bce = -converted * torch.log(probability) - (
            1.0 - converted
        ) * torch.log(1.0 - probability)
        assert torch.equal(clean_bce, expected_bce)
        clean_numerator = torch.sum(clean_weight * clean_bce)
        expected_numerator = torch.sum(expected_weight * expected_bce)
        assert torch.equal(clean_numerator, expected_numerator)
        expected_views.append(expected_numerator / expected_denominator)

    clean_loss, clean_audit = utility_conditioned_relation_loss(
        queries, anchors, target, cycle, balance
    )
    frozen_loss, frozen_audit = authoritative.relation_semantic_loss(
        queries, anchors, target, cycle, balance
    )
    assert torch.equal(clean_loss, frozen_loss)
    assert clean_audit.view_losses == frozen_audit["view_losses"]
    assert clean_audit.denominator == frozen_audit["denominator"]
    assert torch.equal(clean_loss, torch.stack(expected_views).mean())


def test_caltech_all_41_frozen_snapshots_exact_objective_and_intermediate_parity():
    assert _file_sha256(CALTECH_ACTION) == (
        "b4f9ff0241b28ec9f6a8ec6c55c0f9da9c4631ffde2b032c0d3550340fdece71"
    )
    assert _file_sha256(CALTECH_TRAJECTORY) == (
        "0be0b96af1164fe577ee9b02e7bb76fac017cf7742c0b3559281d0fb6a67b576"
    )
    action = _load(
        CALTECH_ACTION,
        ("sample_ids", "unlabeled_ids", "labeled_ids", "U_cycle",
         "PredRelation_true", "relation_balance_weights_true"),
    )
    trajectory = _load(
        CALTECH_TRAJECTORY,
        ("sample_ids", "unlabeled_ids", "labeled_ids", "snapshot_epoch",
         "snapshot_phase", "q_local_snapshots", "arm"),
    )
    expected_hashes = {
        "sample_ids": "887cb8dff9918e38f0b0c7f467d3b9aec75dcef9bf821b2307fd076ca5c42237",
        "unlabeled_ids": "3df6b38608ea4a59dc03cbd1f658a79fadd6cd4402bc12d091b477d96095bfde",
        "labeled_ids": "8bb6d96a7593bb7916915a66fd25a5671df01d923c811ebdaaa5cc5be1f381b3",
        "U_cycle": "fb09bc2a69ef255868b27a0c036c4717c575b4403cbcaeef2f93d8702b6ffeeb",
        "PredRelation_true": "a61460618339930b577c0b53e9fe6f15f4e40650e3522fdab34fa9a43d7bfca5",
        "relation_balance_weights_true": "1d7f1dcc65f49b37fdd6d0d8947456cdf828dff83936e10ca507cf1093d300aa",
    }
    for name, expected in expected_hashes.items():
        assert _ndarray_sha256(action[name]) == expected
    assert _ndarray_sha256(trajectory["q_local_snapshots"]) == (
        "fdf3858d6f46aac31c8fa711623069bc744774b5c404ef0ed231f6ce63938c1a"
    )
    assert _ndarray_sha256(trajectory["snapshot_epoch"]) == (
        "e7e95e69a3e212e31dbc5ee42e3d7058f380c8c29ab552bd0c4916229cbdf81c"
    )
    assert _ndarray_sha256(trajectory["snapshot_phase"]) == (
        "3d7aff5c20f8cb64b31eb3746dc797dcfbd418c3711b1a088b7fe8d51b790eac"
    )
    assert _ndarray_sha256(trajectory["arm"]) == (
        "507d2843a3632368e8373cd70cc9cac9464d2342d54627c3cb139374437cc2fd"
    )
    assert trajectory["q_local_snapshots"].shape == (41, 1400, 6, 7)
    assert np.array_equal(action["sample_ids"], trajectory["sample_ids"])
    assert np.array_equal(action["unlabeled_ids"], trajectory["unlabeled_ids"])
    assert np.array_equal(action["labeled_ids"], trajectory["labeled_ids"])

    query_ids = action["unlabeled_ids"]
    anchor_ids = action["labeled_ids"]
    cycle = action["U_cycle"][query_ids]
    for snapshot in trajectory["q_local_snapshots"]:
        queries = [torch.from_numpy(snapshot[query_ids, view, :]) for view in range(6)]
        anchors = [torch.from_numpy(snapshot[anchor_ids, view, :]) for view in range(6)]
        _assert_exact_objective(
            queries, anchors, action["PredRelation_true"], cycle,
            action["relation_balance_weights_true"],
        )


def test_f0_persisted_diagnostic_identity_is_immutable():
    assert _file_sha256(CALTECH_AUDIT) == (
        "d1f89803fa95c915cad74782b9ff40c3b64202b47eb7cb2c4dc8614e383ead46"
    )
    audit = json.loads(CALTECH_AUDIT.read_text(encoding="utf-8"))
    expected = {
        "E_pre": "37595a598a6f15609707c436a4e3700ddcffde5885348377a9b3e995e8657bb6",
        "E_post": "f634b25d215c1747e139a8c55fd6b85bc1c75590f1da43faa0915be493077e0d",
        "G": "27eccd09585c480a3d939ddf0551a5e6278b6c149004e9ffdec36b2537bb2d3c",
    }
    for name, digest in expected.items():
        assert _ndarray_sha256(
            np.asarray(audit["pre_gt_diagnostics"][name], dtype=np.float64)
        ) == digest


def _assert_structural_reference(path, file_hash, logical_hashes):
    assert _file_sha256(path) == file_hash
    arrays = _load(
        path,
        ("sample_ids", "unlabeled_ids", "labeled_ids", "q_local", "U_cycle",
         "PredRelation_true", "relation_balance_weights_true"),
    )
    for name, digest in logical_hashes.items():
        assert _ndarray_sha256(arrays[name]) == digest
    queries, anchors, cycle = _aligned_views(arrays)
    _assert_exact_objective(
        queries, anchors, arrays["PredRelation_true"], cycle,
        arrays["relation_balance_weights_true"],
    )


def test_msrc_partial_intermediate_structural_exact_formula_parity():
    _assert_structural_reference(
        MSRC,
        "b04f107f6279839d534e55500fbb0f9aba94d7686944ab20f255337a511af234",
        {
            "q_local": "59af5246af0785a26365746aaed4295b22d338eea4bc902eca86c79213432125",
            "U_cycle": "3552d88de9f894276d0d1ccae8f4c55147e98aac9a8ae146eaa1cf54dcc9dd10",
            "PredRelation_true": "cbd2d2de1dfd51abba9b8e80fcc1ee82f4e902df6532ca2af388e1266ff0d2b8",
            "relation_balance_weights_true": "2d69642382947465851d69fe9f151df440979ce626481f9ad4c060909f92ff9a",
        },
    )


def test_bdgp_partial_intermediate_structural_exact_formula_parity():
    _assert_structural_reference(
        BDGP,
        "6e929633526c037e5480ccfac8af496caac02e237489b012988641679f973029",
        {
            "q_local": "8db5104c9a1e51209570353db63f3a6b4f032c5f559ac07ce883f0445af626f5",
            "U_cycle": "b563bc9ecebfc138ce8b0adca24729e5f9f614c1c0ffc3762f8505d7a7f8f2f5",
            "PredRelation_true": "b4df46922ca857bea6d6c8db57bd00d71ec7a5fb3af3b66d1e7119d705bd45d4",
            "relation_balance_weights_true": "5e3d6ccf9c93822d96104a3974438a41ac9f236277c804a0ef59f885b9d75adb",
        },
    )
