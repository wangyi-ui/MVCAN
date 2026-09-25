import ast
from pathlib import Path
import pytest
import torch
from experiments.paper.diagnostics import d6_b0_update_compatibility as d
from experiments.paper.diagnostics import run_d6_b0_update_compatibility as r

def test_geometry_sign_and_parameter_family():
    p=[("x",torch.tensor([0.])),("y",torch.tensor([0.]))]; a=[("x",torch.tensor([1.])),("y",torch.tensor([0.]))]; b=[("x",torch.tensor([.5])),("y",torch.tensor([0.]))]
    g=d._geom(p,a,b); assert g["cosine"]<0 and 0<g["retention"]<1

def test_request_fail_closed():
    with pytest.raises(RuntimeError, match="REQUEST"):
        d.run_one("bad",20,"cpu")

def test_recovery_root_is_disjoint_and_all_completed_cells_verify():
    assert r.RECOVERY_ROOT != d.ROOT
    rows=[d.verify_complete(dataset, seed, r.RECOVERY_ROOT) for dataset in d.DATASETS for seed in d.SEEDS]
    assert len(rows)==9 and all(row["exact_prediction"] and row["no_gt"] for row in rows)

def test_missing_geometry_is_never_complete_or_overwritten(tmp_path):
    cell=d.cell_path("MSRC-v1",20,tmp_path); cell.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        d.verify_complete("MSRC-v1",20,tmp_path)
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        d.run_one("MSRC-v1",20,"cpu",output_root=tmp_path,resume=True)

def test_family_excludes_decoder_and_instrument_restores_after_exception():
    class AE:
        _cluster_layer=torch.nn.Parameter(torch.ones(1))
        _encoder=torch.nn.Linear(2,1)
        _decoder=torch.nn.Linear(1,2)
    class Model: autoencoders=[AE()]
    names=[name for name,_ in d._family(Model(),"combined")]
    assert all("decoder" not in name for name in names)
    old=(d.alternating.run_relation_refinement_phase,d.alternating.run_native_consolidation_phase)
    with pytest.raises(RuntimeError):
        with d.instrument([]): raise RuntimeError("intentional")
    assert old==(d.alternating.run_relation_refinement_phase,d.alternating.run_native_consolidation_phase)

def test_frozen_runtime_and_no_gt_import_contract():
    tree=ast.parse(Path(d.__file__).read_text())
    names=[node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert all(name is None or "evaluation" not in name for name in names)
    for dataset in d.DATASETS:
        for seed in d.SEEDS:
            assert d.runner.runtime_config(d.runner.FormalRun(dataset,seed,"OURS_TRUE_U","cuda:0")).refresh_interval==100

def test_first_attempt_manifest_preserves_every_old_artifact():
    manifest=Path("experiment_freeze/d6_b0_first_attempt_incomplete_20260925/PARTIAL_MANIFEST.txt")
    rows=[line.split(maxsplit=1) for line in manifest.read_text().splitlines() if line]
    assert rows and all(Path(path).is_file() and d._sha(path)==digest for digest,path in rows)

def test_gate2_requires_strict_conflict_excess():
    rows=[]
    for dataset in d.DATASETS:
        for seed in d.SEEDS:
            median=-2.0 if dataset=="MSRC-v1" else -1.0
            rows.append({"dataset":dataset,"seed":seed,"exact_prediction":True,"no_gt":True,"restored":True,"epochs":20,"families":{"combined":{"median":median,"conflict_epoch_fraction":1.0,"retention":{"median":0.5}}}})
    gates=r._gate_rows(rows)
    assert gates["gate2_msrc_opposition"] is False
