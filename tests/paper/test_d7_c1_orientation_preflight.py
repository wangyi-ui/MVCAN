import numpy as np
import pytest
from experiments.paper.candidates import utility_weighted_sparse_orientation as c
from experiments.paper.diagnostics import d7_c1_orientation_preflight as d
def test_weighted_vote_and_zero_labeled_contract():
 y=np.array([[0,1],[1,1]]);u=np.array([[1.,3.],[2.,2.]]);v=c.build_utility_weighted_vote_state(y,u,2);assert v.shape==(2,2)and np.allclose(v[0],[.25,.75])and np.allclose(v.sum(1),1)
def test_candidate_relation_and_subsets():
 from release_core.semantics import SparseLabelSplit
 sp=SparseLabelSplit(np.arange(5),np.array([0,1,2,3]),np.array([0,0,1,1]),np.array([4]),2,2,20,'x');y=np.array([[0,0],[0,0],[1,1],[1,1],[0,1]]);u=np.ones((5,2));r=c.build_utility_weighted_relation_semantics(y,u,sp);assert r['pred_relation'].shape==(1,4,2)and len(d._subs(sp.labeled_ids,sp.labeled_targets,2))==4
def test_invalid_fails_closed():
 with pytest.raises(RuntimeError):c.build_utility_weighted_vote_state(np.array([[0]]),np.array([[1.]]),0)
