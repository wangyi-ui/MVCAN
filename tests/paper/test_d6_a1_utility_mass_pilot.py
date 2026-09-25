import pytest
from release_core.training import alternating
from release_core.action.relation import utility_conditioned_relation_loss
from experiments.paper.diagnostics import d6_a1_utility_mass_pilot as p
def test_patch_restores_on_success_and_exception():
 records=[]
 with p.patched_loss(records): assert alternating.utility_conditioned_relation_loss is not utility_conditioned_relation_loss
 assert alternating.utility_conditioned_relation_loss is utility_conditioned_relation_loss
 with pytest.raises(RuntimeError):
  with p.patched_loss(records): raise RuntimeError("x")
 assert alternating.utility_conditioned_relation_loss is utility_conditioned_relation_loss
def test_paths_and_runtime_invariants(tmp_path,monkeypatch):
 monkeypatch.setattr(p,"ROOT",tmp_path); root,_=p._paths(20); assert "formal" not in str(root)
 with pytest.raises(RuntimeError): p._paths(1)
