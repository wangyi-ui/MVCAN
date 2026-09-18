from pathlib import Path

import torch
import torch.nn.functional as F

from release_core.backbone.native_objective import native_objective


def test_native_objective_exact_for_frozen_lambdas():
    reconstruction = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 13
    inputs = torch.flip(reconstruction, dims=(0,))
    q_local = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=torch.float32)
    p_local = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]], dtype=torch.float32)
    for lambda1 in (0.01, 10):
        old_rec = F.mse_loss(reconstruction, inputs)
        old_clu = F.mse_loss(q_local, p_local)
        old_total = old_rec + lambda1 * old_clu
        new_total, new_rec, new_clu = native_objective(
            reconstruction, inputs, q_local, p_local, lambda1
        )
        assert torch.equal(old_rec, new_rec)
        assert torch.equal(old_clu, new_clu)
        assert torch.equal(old_total, new_total)


def test_objective_source_has_no_extra_scientific_terms():
    source = Path(native_objective.__code__.co_filename).read_text(encoding="utf-8").lower()
    for forbidden in ("relation", "semantic", "utility"):
        assert forbidden not in source

