import ast
from pathlib import Path
import numpy as np
import pytest
from experiments.paper.diagnostics import d7_b0_relation_ambiguity as d
def test_consensus_entropy_and_zero_mass():
 r=d.consensus(np.array([[0,0],[0,1],[1,1]]),np.array([[1.,1.],[1.,1.],[0.,0.]]));assert r['class_consensus']['mean']==.75 and r['zero_utility_sample_fraction']==1/3
def test_contradiction_balanced_raw_and_edges():
 t=np.array([[[True,True]],[[False,False]]]);u=np.ones((2,2));assert d.contradiction(t,u)['D_global']==0
 t=np.array([[[True,False]]]);r=d.contradiction(t,np.ones((1,2)));assert r['D_global']==.5 and r['pair_median']==.5
def test_mapping_stability_and_margin():
 vote=np.eye(2)[[0,0,1,1]];ids=np.arange(4);targets=np.array([0,0,1,1]);full=d._loo_mapping(vote,ids,targets,2,0);s=d.mapping_stability(vote,ids,targets,full,2);assert s['mapping_entry_stability']==1
def test_fail_closed_and_no_training_evaluation_import():
 with pytest.raises(RuntimeError):d.run_one('bad',20)
 mods=[x.module or '' for x in ast.walk(ast.parse(Path(d.__file__).read_text())) if isinstance(x,ast.ImportFrom)];assert all('training' not in x and 'evaluation' not in x for x in mods)
