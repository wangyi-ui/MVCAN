import ast
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.paper.diagnostics import d7_a0_granularity_adequacy as d
def test_prediction_averaging_and_fixed_granularities():
 q=np.array([[[1.,0.],[3.,0.],[5.,0.]]]); p=d.generator_predictions(q,((0,1),(2,))); assert p.shape==(1,2,2); assert np.array_equal(p,np.array([[[2.,0.],[5.,0.]]]))
 assert d.granularities(7)==(4,7,14) and d.granularities(5)==(3,5,10)
def test_anchor_states_and_mass_partition():
 labels=np.array([0,0,1,2,2,3]); ids=np.array([0,1,2,3,4]); targets=np.array([1,1,2,3,4]); states=d.anchor_states(labels,ids,targets); assert list(states)==[d.STRONG,d.STRONG,d.SINGLETON,d.CONFLICT,d.CONFLICT,d.ORPHAN]
 s=np.stack([states,states],axis=1); m=d.mass_summary(s,np.ones((6,2)),np.arange(6),((0,),(1,))); assert abs(sum(m["global"].values())-1)<1e-12 and m["global"]["strong"]>0
def test_ward_counts_dominance_and_ambiguous_tie():
 p=np.arange(24,dtype=float).reshape(8,3); assert set(d.ward_partitions(p,(2,3)).keys())=={2,3}
 native={"strong":.2,"conflict":.1}; coarse={"strong":.3,"conflict":.1}; fine={"strong":.3,"conflict":.1}; assert d.dominates(coarse,native) and d.preferred({4:{"global":coarse},7:{"global":native},14:{"global":fine}},4,7,14)=="ambiguous"
def test_invalid_request_fails_closed_and_no_evaluation_import():
 with pytest.raises(RuntimeError): d.run_one("bad",20)
 assert "evaluation" not in open(d.__file__).read().replace("no evaluation","")

def test_frozen_y_gen_parity_evidence_and_disjoint_output_namespace():
    root=Path("outputs/paper/diagnostics/d7_a0_granularity_adequacy")
    assert root != Path("outputs/paper/diagnostics/d6_b0_update_compatibility_recovery1")
    rows=json.loads((root/"d7_a0_summary.json").read_text())["rows"]
    assert len(rows)==9 and all(row["y_gen_argmax_exact"] for row in rows)

def test_no_training_or_evaluation_import_dependency():
    tree=ast.parse(Path(d.__file__).read_text())
    modules=[node.module or "" for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    assert all("evaluation" not in name and "training" not in name for name in modules)

def test_invalid_mass_shape_fails_closed():
    with pytest.raises(RuntimeError,match="MASS_SHAPE"):
        d.mass_summary(np.zeros((2,2),dtype=np.int8),np.ones((3,2)),np.array([0]),((0,),(1,)))
