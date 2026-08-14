import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

from configure import get_default_config
from irv.b3_audit import hash_semantic_heads, hash_state_dict
from irv.semantic_head import DetachedSemanticHeadBank
from model import MvCAN


def _semantic_config(seed=1020):
    return {
        "mode": "detached",
        "semantic_dim": 10,
        "semantic_seed": seed,
    }


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
