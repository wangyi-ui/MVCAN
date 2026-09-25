import ast
from pathlib import Path
import numpy as np
import pytest
from experiments.paper.diagnostics import d7_c0_sparse_orientation_robustness as d
def test_exhaustive_balanced_subsets_and_fit():
 ids=np.arange(4);targets=np.array([0,0,1,1]);subs=d.balanced_subsets(ids,targets,2);assert len(subs)==4 and all(len(x)==2 for x in subs)
 vote=np.eye(2)[[0,0,1,1]];m=d.fit_subset_mapping(vote,subs[0],ids,targets,2);assert np.array_equal(m,np.array([0,1]))
def test_flip_formulas_and_identical_zero():
 full=np.array([[0,1]]);u=np.ones((1,2));b=np.ones((1,1,2));fc,fr,_=d.flip_masses(full,full,np.array([0]),u,b);assert fc==0 and fr==0
 swapped=np.array([[1,0]]);fc,fr,_=d.flip_masses(full,swapped,np.array([0]),u,b);assert fc==1 and fr==1
def test_invalid_and_no_training_evaluation_import():
 with pytest.raises(RuntimeError):d.run_one('bad',20)
 mods=[x.module or ''for x in ast.walk(ast.parse(Path(d.__file__).read_text()))if isinstance(x,ast.ImportFrom)];assert all('training'not in x and'evaluation'not in x for x in mods)
def test_namespace_disjoint():assert d.ROOT.name!='d7_b0_relation_ambiguity'
