import torch
import pytest
from experiments.paper.candidates.utility_mass_preserving_relation import utility_mass_preserving_relation_loss
from release_core.action.relation import utility_conditioned_relation_loss

def _case(u):
 torch.manual_seed(4); q=[torch.softmax(torch.randn(3,4),-1).requires_grad_() for _ in range(2)]; a=[torch.softmax(torch.randn(2,4),-1).requires_grad_() for _ in range(2)]; rel=torch.randint(0,2,(3,2,5),dtype=torch.bool); bal=torch.rand(3,2,5)+.1; return q,a,rel,torch.as_tensor(u,dtype=torch.float32),bal
def test_ones_equals_frozen_and_candidate_uses_frozen_primitives():
 q,a,r,u,b=_case(torch.ones(3,5)); old,_=utility_conditioned_relation_loss(q,a,r,u,b); new,audit=utility_mass_preserving_relation_loss(q,a,r,u,b)
 assert torch.allclose(new,old,atol=1e-6) and audit.utility_mass == pytest.approx(1.0) and audit.no_new_semantic_lambda
def test_scale_invariance_and_candidate_scale():
 q,a,r,u,b=_case(torch.ones(3,5)); old,_=utility_conditioned_relation_loss(q,a,r,u,b); new,_=utility_mass_preserving_relation_loss(q,a,r,u,b)
 for c in (.1,.5):
  oldc,_=utility_conditioned_relation_loss(q,a,r,u*c,b); newc,_=utility_mass_preserving_relation_loss(q,a,r,u*c,b); assert torch.allclose(oldc,old,atol=1e-6) and torch.allclose(newc,new*c,atol=1e-6)
def test_arbitrary_identity_gradient_anchor_and_detached_inputs():
 q,a,r,u,b=_case(torch.rand(3,5)+.1); old,_=utility_conditioned_relation_loss(q,a,r,u,b); new,audit=utility_mass_preserving_relation_loss(q,a,r,u,b); rho=(u[:,None,:]*b).sum()/b.sum(); assert torch.allclose(new,rho*old,atol=1e-6)
 go=torch.autograd.grad(old,q,retain_graph=True); gn=torch.autograd.grad(new,q); assert all(torch.allclose(x*rho,y,atol=1e-6) for x,y in zip(go,gn)); assert all(x.grad is None for x in a) and audit.anchor_stop_gradient_enforced
def test_zero_u_is_zero_with_finite_zero_query_gradient():
 q,a,r,u,b=_case(torch.zeros(3,5)); loss,_=utility_mass_preserving_relation_loss(q,a,r,u,b); grad=torch.autograd.grad(loss,q); assert loss.item()==0 and all(torch.isfinite(x).all() and torch.count_nonzero(x)==0 for x in grad)
def test_shape_validation():
 q,a,r,u,b=_case(torch.ones(3,5));
 with pytest.raises(ValueError): utility_mass_preserving_relation_loss(q[:1],a,r,u,b)
