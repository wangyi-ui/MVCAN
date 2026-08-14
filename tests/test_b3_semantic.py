import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

from configure import get_default_config
from irv.b3_audit import hash_backbone, hash_semantic_heads, hash_state_dict
from irv.semantic_loss import uniform_cross_view_infonce
from irv.semantic_head import DetachedSemanticHeadBank
from model import MvCAN


def _semantic_config(seed=1020):
    return {
        "mode": "detached",
        "semantic_dim": 10,
        "semantic_seed": seed,
    }


def _uniform_config(seed=1020):
    return {
        "mode": "uniform",
        "semantic_dim": 10,
        "semantic_seed": seed,
        "semantic_lr": 1e-4,
        "semantic_temperature": 0.2,
    }


def _deterministic_z_views(requires_grad=False):
    # base: [batch_size=8, latent_dim=10]
    base = torch.arange(80, dtype=torch.float32).reshape(8, 10) / 80.0
    z_views = []
    for view_idx in range(5):
        # z_v: [batch_size=8, latent_dim=10]
        z_v = torch.roll(base, shifts=view_idx, dims=1) + view_idx * 0.01
        z_views.append(z_v.clone().requires_grad_(requires_grad))
    return z_views


def _audit_fixture(semantic_enabled):
    backbone_per_view = ["backbone-" + str(index) for index in range(5)]
    audit = {
        "stage": "B3-A0",
        "dataset": "MSRC-v1",
        "model_seed": 20,
        "corruption_protocol_version": "b2-fallback-v1",
        "corruption_mode": "none",
        "corruption_seed": None,
        "corruption_k": None,
        "target_snr_db": None,
        "corruption_mask_sha256": None,
        "semantic_mode": "detached" if semantic_enabled else "off",
        "semantic_enabled": semantic_enabled,
        "latent_dim": 10,
        "semantic_dim": 10 if semantic_enabled else None,
        "semantic_seed": 1020 if semantic_enabled else None,
        "view_num": 5,
        "native_metrics": {
            "acc": 0.5333333333333333,
            "nmi": 0.3901793186908467,
            "ari": 0.26623285214890685,
        },
        "backbone_hash": {
            "aggregate": "backbone-aggregate",
            "per_view": backbone_per_view,
        },
        "semantic_hash": None,
        "runtime": {
            "semantic_forward_calls": 0,
            "semantic_shape_pass": True,
            "semantic_finite_pass": True,
            "semantic_input_detached_pass": True,
            "semantic_norm_finite_pass": True,
            "semantic_norm_mean": None,
            "semantic_norm_min": None,
            "semantic_norm_max": None,
        },
    }
    if semantic_enabled:
        audit["semantic_hash"] = {
            "aggregate": "semantic-aggregate",
            "per_view": ["semantic-" + str(index) for index in range(5)],
        }
        audit["runtime"].update({
            "semantic_forward_calls": 3,
            "semantic_norm_mean": 1.0,
            "semantic_norm_min": 0.9999999,
            "semantic_norm_max": 1.0000001,
        })
    return audit


def test_semantic_head_shapes_for_five_views():
    head_bank = DetachedSemanticHeadBank(5, 10, 10, 1020)
    # z_views[v]: [batch_size=8, latent_dim=10]
    z_views = [torch.randn(8, 10) for _ in range(5)]

    # s_views[v]: [batch_size=8, semantic_dim=10]
    s_views = head_bank.forward_views(z_views)

    assert len(s_views) == 5
    assert all(s_v.shape == (8, 10) for s_v in s_views)


def test_detach_blocks_input_gradient_but_head_gradient_is_reachable():
    torch.manual_seed(7)
    head_bank = DetachedSemanticHeadBank(1, 10, 10, 1020)
    # z: [batch_size=8, latent_dim=10]
    z = torch.randn(8, 10, requires_grad=True)
    # target: [batch_size=8, semantic_dim=10]
    target = F.normalize(z.detach(), p=2, dim=1)

    # s: [batch_size=8, semantic_dim=10]
    s = head_bank.forward_one(z, 0)
    loss = 1.0 - (s * target).sum(dim=1).mean()
    loss.backward()

    assert z.grad is None
    for parameter in head_bank.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0.0


def test_semantic_parameters_are_disjoint_from_backbone_and_native_optimizer():
    config = get_default_config("MSRC-v1")
    model = MvCAN(
        config,
        view_num=1,
        view_size=[4],
        n_clusters=2,
        seed=20,
        data_size=8,
        semantic_config=_semantic_config(),
    )
    semantic_ids = {id(parameter) for parameter in model.semantic_heads.parameters()}
    backbone_ids = {
        id(parameter)
        for autoencoder in model.autoencoders
        for parameter in autoencoder.parameters()
    }
    optimizer = torch.optim.Adam(model.autoencoders[0].parameters(), lr=1e-4)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert semantic_ids
    assert semantic_ids.isdisjoint(backbone_ids)
    assert optimizer_ids == {id(p) for p in model.autoencoders[0].parameters()}
    assert optimizer_ids.isdisjoint(semantic_ids)


def test_semantic_initialization_is_rng_isolated_and_seeded():
    torch.manual_seed(1234)
    state_before = torch.get_rng_state().clone()
    first = DetachedSemanticHeadBank(5, 10, 10, 1020)
    state_after = torch.get_rng_state().clone()
    second = DetachedSemanticHeadBank(5, 10, 10, 1020)
    different = DetachedSemanticHeadBank(5, 10, 10, 1021)

    assert torch.equal(state_before, state_after)
    assert hash_semantic_heads(first) == hash_semantic_heads(second)
    assert hash_semantic_heads(first) != hash_semantic_heads(different)


def test_semantic_off_does_not_instantiate_heads_or_change_native_shape():
    config = get_default_config("MSRC-v1")
    model = MvCAN(
        config,
        view_num=2,
        view_size=[4, 6],
        n_clusters=2,
        seed=20,
        data_size=8,
        semantic_config=None,
    )
    backbone_ids = {
        id(parameter)
        for autoencoder in model.autoencoders
        for parameter in autoencoder.parameters()
    }

    assert model.semantic_mode == "off"
    assert model.semantic_heads is None
    assert model.semantic_dim is None
    assert model.semantic_seed is None
    assert model._latent_dim == 10
    assert len(model.autoencoders) == 2
    assert backbone_ids
    assert model.semantic_runtime_audit["semantic_forward_calls"] == 0


def test_semantic_outputs_and_norms_are_finite_and_normalized():
    head_bank = DetachedSemanticHeadBank(5, 10, 10, 1020)
    # z_views[v]: [batch_size=8, latent_dim=10]
    z_views = [torch.randn(8, 10) for _ in range(5)]
    # s_views[v]: [batch_size=8, semantic_dim=10]
    s_views = head_bank.forward_views(z_views)

    for s_v in s_views:
        # row_norms: [batch_size=8]
        row_norms = torch.linalg.vector_norm(s_v, ord=2, dim=1)
        assert torch.isfinite(s_v).all()
        assert torch.isfinite(row_norms).all()
        assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-6)


def test_state_dict_tensor_hash_is_exact_and_sensitive():
    original = OrderedDict([
        ("weight", torch.arange(12, dtype=torch.float32).reshape(3, 4)),
        ("counter", torch.tensor(2, dtype=torch.int64)),
    ])
    same = OrderedDict((key, value.clone()) for key, value in original.items())
    changed = OrderedDict((key, value.clone()) for key, value in original.items())
    changed["weight"][0, 0] += 1.0

    assert hash_state_dict(original) == hash_state_dict(same)
    assert hash_state_dict(original) != hash_state_dict(changed)


def test_compare_audit_script_passes_exact_pair_and_fails_hash_change(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    script = (
        repository_root
        / "experiments"
        / "b3_semantic"
        / "compare_b3_a0_identity.py"
    )
    off_path = tmp_path / "off.json"
    detached_path = tmp_path / "detached.json"
    off_path.write_text(json.dumps(_audit_fixture(False)))
    detached = _audit_fixture(True)
    detached_path.write_text(json.dumps(detached))
    command = [
        sys.executable,
        str(script),
        "--off_audit",
        str(off_path),
        "--detached_audit",
        str(detached_path),
    ]

    passed = subprocess.run(command, cwd=str(repository_root), capture_output=True)
    assert passed.returncode == 0, passed.stdout.decode() + passed.stderr.decode()
    assert b"B3_A0_EXACT_PROTECTION_PASS=true" in passed.stdout

    detached["backbone_hash"]["per_view"][0] = "tampered"
    detached_path.write_text(json.dumps(detached))
    failed = subprocess.run(command, cwd=str(repository_root), capture_output=True)
    assert failed.returncode != 0
    assert b"backbone_per_view_hash_exact: FAIL" in failed.stdout
    assert b"B3_A0_EXACT_PROTECTION_PASS=false" in failed.stdout



def _a1_audit_fixture(semantic_mode):
    semantic_enabled = semantic_mode == "uniform"
    backbone_per_view = ["a1-backbone-" + str(index) for index in range(5)]
    audit = {
        "stage": "B3-A1",
        "dataset": "MSRC-v1",
        "model_seed": 20,
        "corruption_protocol_version": "b2-fallback-v1",
        "corruption_mode": "none",
        "corruption_seed": None,
        "corruption_k": None,
        "target_snr_db": None,
        "corruption_mask_sha256": None,
        "semantic_mode": semantic_mode,
        "semantic_enabled": semantic_enabled,
        "latent_dim": 10,
        "semantic_dim": 10 if semantic_enabled else None,
        "semantic_seed": 1020 if semantic_enabled else None,
        "semantic_lr": 1e-4 if semantic_enabled else None,
        "semantic_temperature": 0.2 if semantic_enabled else None,
        "view_num": 5,
        "native_metrics": {
            "acc": 0.5333333333333333,
            "nmi": 0.3901793186908467,
            "ari": 0.26623285214890685,
        },
        "backbone_hash": {
            "aggregate": "a1-backbone-aggregate",
            "per_view": backbone_per_view,
        },
        "semantic_hash_initial": None,
        "semantic_hash_final": None,
        "runtime": {
            "semantic_forward_calls": 0,
            "semantic_optimizer_steps": 0,
            "semantic_pair_count": 0,
            "semantic_loss_first": None,
            "semantic_loss_last": None,
            "semantic_loss_mean": None,
            "semantic_loss_min": None,
            "semantic_loss_max": None,
            "semantic_loss_finite_pass": True,
            "semantic_grad_l2_last": None,
            "semantic_grad_l2_mean": None,
            "semantic_grad_nonzero_pass": True,
            "semantic_grad_finite_pass": True,
            "semantic_all_heads_grad_pass": True,
            "semantic_head_hash_initial": None,
            "semantic_head_hash_final": None,
            "semantic_head_updated_pass": False,
            "semantic_shape_pass": True,
            "semantic_finite_pass": True,
            "semantic_input_detached_pass": True,
            "semantic_norm_finite_pass": True,
        },
    }
    if semantic_enabled:
        initial_per_view = [
            "a1-semantic-initial-" + str(index) for index in range(5)
        ]
        final_per_view = [
            "a1-semantic-final-" + str(index) for index in range(5)
        ]
        audit["semantic_hash_initial"] = {
            "aggregate": "a1-semantic-initial",
            "per_view": initial_per_view,
        }
        audit["semantic_hash_final"] = {
            "aggregate": "a1-semantic-final",
            "per_view": final_per_view,
        }
        audit["runtime"].update({
            "semantic_forward_calls": 3,
            "semantic_optimizer_steps": 3,
            "semantic_pair_count": 10,
            "semantic_loss_first": 2.4,
            "semantic_loss_last": 2.3,
            "semantic_loss_mean": 2.35,
            "semantic_loss_min": 2.3,
            "semantic_loss_max": 2.4,
            "semantic_grad_l2_last": 0.8,
            "semantic_grad_l2_mean": 0.9,
            "semantic_head_hash_initial": "a1-semantic-initial",
            "semantic_head_hash_final": "a1-semantic-final",
            "semantic_head_updated_pass": True,
        })
    return audit


def test_uniform_infonce_pair_count_for_five_views():
    semantic_views = [
        F.normalize(z_v, p=2, dim=1) for z_v in _deterministic_z_views()
    ]
    _, diagnostics = uniform_cross_view_infonce(
        semantic_views,
        temperature=0.2,
    )

    assert diagnostics["pair_count"] == 10


def test_uniform_infonce_returns_finite_scalar_loss():
    semantic_views = [
        F.normalize(z_v, p=2, dim=1) for z_v in _deterministic_z_views()
    ]
    loss, diagnostics = uniform_cross_view_infonce(
        semantic_views,
        temperature=0.2,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert diagnostics["loss_finite"] is True
    assert diagnostics["batch_size"] == 8
    assert diagnostics["semantic_dim"] == 10


def test_uniform_infonce_rewards_symmetric_sample_correspondence():
    # aligned: [batch_size=8, semantic_dim=10]
    aligned = torch.eye(8, 10, dtype=torch.float32)
    permutation = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0])
    # permuted: [batch_size=8, semantic_dim=10]
    permuted = aligned[permutation]

    aligned_loss, _ = uniform_cross_view_infonce(
        [aligned, aligned.clone()],
        temperature=0.2,
    )
    permuted_loss, _ = uniform_cross_view_infonce(
        [aligned, permuted],
        temperature=0.2,
    )

    assert aligned_loss.item() < permuted_loss.item()


def test_uniform_infonce_gives_all_five_semantic_heads_gradient():
    head_bank = DetachedSemanticHeadBank(5, 10, 10, 1020)
    # z_views[v]: [batch_size=8, latent_dim=10]
    z_views = _deterministic_z_views()
    # semantic_views[v]: [batch_size=8, semantic_dim=10]
    semantic_views = head_bank.forward_views(z_views)
    semantic_loss, _ = uniform_cross_view_infonce(
        semantic_views,
        temperature=0.2,
    )
    semantic_loss.backward()

    for head in head_bank.heads:
        head_grad_l1 = 0.0
        for parameter in head.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            head_grad_l1 += parameter.grad.abs().sum().item()
        assert head_grad_l1 > 0.0


def test_uniform_infonce_has_no_gradient_into_all_five_latent_inputs():
    head_bank = DetachedSemanticHeadBank(5, 10, 10, 1020)
    # z_views[v]: [batch_size=8, latent_dim=10]
    z_views = _deterministic_z_views(requires_grad=True)
    # semantic_views[v]: [batch_size=8, semantic_dim=10]
    semantic_views = head_bank.forward_views(z_views)
    semantic_loss, _ = uniform_cross_view_infonce(
        semantic_views,
        temperature=0.2,
    )
    semantic_loss.backward()

    assert all(z_v.grad is None for z_v in z_views)


def test_uniform_semantic_optimizer_parameters_are_native_disjoint():
    config = get_default_config("MSRC-v1")
    model = MvCAN(
        config,
        view_num=2,
        view_size=[4, 6],
        n_clusters=2,
        seed=20,
        data_size=8,
        semantic_config=_uniform_config(),
    )
    semantic_optimizer = torch.optim.Adam(
        model.semantic_heads.parameters(),
        lr=1e-4,
    )
    semantic_optimizer_ids = {
        id(parameter)
        for group in semantic_optimizer.param_groups
        for parameter in group["params"]
    }
    backbone_ids = {
        id(parameter)
        for autoencoder in model.autoencoders
        for parameter in autoencoder.parameters()
    }

    assert semantic_optimizer_ids
    assert semantic_optimizer_ids.isdisjoint(backbone_ids)


def test_one_uniform_optimizer_step_changes_only_semantic_hash():
    config = get_default_config("MSRC-v1")
    model = MvCAN(
        config,
        view_num=5,
        view_size=[4, 5, 6, 7, 8],
        n_clusters=2,
        seed=20,
        data_size=8,
        semantic_config=_uniform_config(),
    )
    semantic_optimizer = torch.optim.Adam(
        model.semantic_heads.parameters(),
        lr=1e-4,
    )
    backbone_before = hash_backbone(model.autoencoders)
    semantic_before = hash_semantic_heads(model.semantic_heads)
    # z_views[v]: [batch_size=8, latent_dim=10]
    z_views = _deterministic_z_views()
    # semantic_views[v]: [batch_size=8, semantic_dim=10]
    semantic_views = model.semantic_heads.forward_views(z_views)
    semantic_loss, _ = uniform_cross_view_infonce(
        semantic_views,
        temperature=0.2,
    )

    semantic_optimizer.zero_grad()
    semantic_loss.backward()
    semantic_optimizer.step()

    assert hash_backbone(model.autoencoders) == backbone_before
    assert hash_semantic_heads(model.semantic_heads) != semantic_before


def test_detached_mode_probe_does_not_update_semantic_head():
    config = get_default_config("MSRC-v1")
    model = MvCAN(
        config,
        view_num=5,
        view_size=[4, 5, 6, 7, 8],
        n_clusters=2,
        seed=20,
        data_size=8,
        semantic_config=_semantic_config(),
    )
    semantic_optimizer = None
    semantic_before = hash_semantic_heads(model.semantic_heads)
    # z_views[v]: [batch_size=8, latent_dim=10]
    z_views = _deterministic_z_views()
    with torch.no_grad():
        model.semantic_heads.forward_views(z_views)

    assert semantic_optimizer is None
    assert model.semantic_mode == "detached"
    assert hash_semantic_heads(model.semantic_heads) == semantic_before


def test_a1_comparator_passes_and_rejects_backbone_or_update_tampering(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    script = (
        repository_root
        / "experiments"
        / "b3_semantic"
        / "compare_b3_a1_protection.py"
    )
    off_path = tmp_path / "a1_off.json"
    uniform_path = tmp_path / "a1_uniform.json"
    off_path.write_text(json.dumps(_a1_audit_fixture("off")))
    uniform = _a1_audit_fixture("uniform")
    uniform_path.write_text(json.dumps(uniform))
    command = [
        sys.executable,
        str(script),
        "--off_audit",
        str(off_path),
        "--uniform_audit",
        str(uniform_path),
    ]

    passed = subprocess.run(command, cwd=str(repository_root), capture_output=True)
    assert passed.returncode == 0, passed.stdout.decode() + passed.stderr.decode()
    assert b"B3_A1_BACKBONE_PROTECTION_PASS=true" in passed.stdout
    assert b"B3_A1_SEMANTIC_UPDATE_PASS=true" in passed.stdout

    uniform["backbone_hash"]["per_view"][0] = "tampered"
    uniform_path.write_text(json.dumps(uniform))
    backbone_failed = subprocess.run(
        command,
        cwd=str(repository_root),
        capture_output=True,
    )
    assert backbone_failed.returncode != 0
    assert b"backbone_per_view_hash_exact: FAIL" in backbone_failed.stdout

    uniform = _a1_audit_fixture("uniform")
    uniform["runtime"]["semantic_head_updated_pass"] = False
    uniform_path.write_text(json.dumps(uniform))
    update_failed = subprocess.run(
        command,
        cwd=str(repository_root),
        capture_output=True,
    )
    assert update_failed.returncode != 0
    assert b"semantic_head_updated: FAIL" in update_failed.stdout
    assert b"B3_A1_SEMANTIC_UPDATE_PASS=false" in update_failed.stdout
