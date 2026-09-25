"""Frozen-v1 D6-B0 update geometry replay; no GT/evaluation imports."""
import hashlib,json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import numpy as np
from release_core.runtime import run_pre_gt
from release_core.training import alternating
from experiments.paper.formal import p1_a1_action_materialization as actions
from experiments.paper.formal import p1_a1_native_preparation as prep
from experiments.paper.formal import p1_a1_runner as runner
from experiments.paper.formal import p1_a3_runtime_wiring as wiring
from experiments.paper.formal import run_formal_pipeline as pipeline
ROOT=Path("outputs/paper/diagnostics/d6_b0_update_compatibility"); DATASETS=("Caltech-6V","MSRC-v1","BDGP"); SEEDS=(20,30,50)
def _req(x,m):
 if not x: raise RuntimeError(m)
def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _family(model,kind):
 out=[]
 for v,a in enumerate(model.autoencoders):
  if kind in ("encoder","combined"):
   out += [("v%d.encoder.%s"%(v,n),p.detach().clone()) for n,p in a._encoder.named_parameters()]
  if kind in ("cluster","combined"): out += [("v%d.cluster"%v,a._cluster_layer.detach().clone())]
 return out
def _geom(pre,a,b):
 da=sum((((x-y)**2).sum() for (_,x),(_,y) in zip(a,pre))); db=sum((((x-y)**2).sum() for (_,x),(_,y) in zip(b,a))); dot=sum((((x-y)*(z-x)).sum() for (_,x),(_,y),(_,z) in zip(a,pre,b))); na=float(da.sqrt()); nb=float(db.sqrt()); d=float(dot); e=1e-30; return {"cosine":d/(na*nb+e),"retention":1+d/(na*na+e),"norm_ratio":nb/(na+e)}
@contextmanager
def instrument(records):
 oa,ob=alternating.run_relation_refinement_phase,alternating.run_native_consolidation_phase; state={}
 def A(model,*args,**kw):
  state["pre"]={k:_family(model,k) for k in ("encoder","cluster","combined")}; r=oa(model,*args,**kw); state["post"]={k:_family(model,k) for k in state["pre"]}; return r
 def B(model,*args,**kw):
  r=ob(model,*args,**kw); post={k:_family(model,k) for k in state["pre"]}; records.append({k:_geom(state["pre"][k],state["post"][k],post[k]) for k in state["pre"]}); return r
 alternating.run_relation_refinement_phase,alternating.run_native_consolidation_phase=A,B
 try: yield
 finally: alternating.run_relation_refinement_phase,alternating.run_native_consolidation_phase=oa,ob
def _q(v):
 a=np.asarray(v); return {"min":float(a.min()),"p25":float(np.percentile(a,25)),"median":float(np.median(a)),"mean":float(a.mean()),"p75":float(np.percentile(a,75)),"max":float(a.max()),"conflict_epoch_fraction":float(np.mean(a<0))}
def cell_path(dataset,seed,output_root=ROOT):
 return Path(output_root)/dataset.lower().replace("-","")/("seed"+str(seed))
def _expected_prediction(dataset,seed):
 p=runner.paths_for(runner.FormalRun(dataset,seed,"OURS_TRUE_U","cuda:0")); return json.loads((p["output"] / "pre_gt_seal.json").read_text())["prediction_logical_sha256"]
def verify_complete(dataset,seed,output_root=ROOT):
 out=cell_path(dataset,seed,output_root); geometry=out/"geometry.json"; seal=out/"pre_gt"/"pre_gt_seal.json"; audit=out/"pre_gt"/"pre_gt_audit.json"; bundle=out/"pre_gt"/"pre_gt_bundle.npz"
 _req(all(x.is_file() for x in (geometry,seal,audit,bundle)),"D6_B0_RECOVERY_CELL_INCOMPLETE")
 row=json.loads(geometry.read_text()); s=json.loads(seal.read_text()); a=json.loads(audit.read_text()); expected=_expected_prediction(dataset,seed)
 _req(_sha(bundle)==s.get("bundle_sha256"),"D6_B0_RECOVERY_BUNDLE_HASH_MISMATCH"); _req(_sha(audit)==s.get("audit_sha256"),"D6_B0_RECOVERY_AUDIT_HASH_MISMATCH")
 _req(s.get("prediction_logical_sha256")==expected and a.get("prediction_logical_sha256")==expected,"D6_B0_RECOVERY_PREDICTION_PARITY_MISMATCH")
 _req(row.get("dataset")==dataset and row.get("seed")==seed and row.get("exact_prediction") is True and row.get("epochs")==20,"D6_B0_RECOVERY_GEOMETRY_INVALID")
 _req(row.get("no_gt") is True and a.get("gt_firewall",{}).get("full_gt_loaded") is False,"D6_B0_RECOVERY_GT_FIREWALL_INVALID")
 _req(bool(a.get("clean_source_sha256")) and a.get("determinism",{}).get("seed")==seed,"D6_B0_RECOVERY_SOURCE_IDENTITY_MISSING")
 _req(all(Path(path).is_file() and _sha(path)==digest for path,digest in a["clean_source_sha256"].items()),"D6_B0_RECOVERY_RELEASE_SOURCE_CHANGED")
 return row
def run_one(dataset,seed,device,output_root=ROOT,resume=False):
 _req(dataset in DATASETS and seed in SEEDS,"D6_B0_REQUEST_INVALID"); out=cell_path(dataset,seed,output_root)
 if out.exists():
  if resume: return verify_complete(dataset,seed,output_root)
  _req(False,"D6_B0_OUTPUT_EXISTS")
 run=runner.FormalRun(dataset,seed,"OURS_TRUE_U",device); p=runner.paths_for(run); pipeline._verify_inputs(p["features"].parent,dataset); init=prep.verify_initialization(p["initialization"],dataset=dataset,training_seed=seed); act=actions.verify_true_action(p["action"],dataset=dataset,training_seed=seed,initial_model_sha256=init["initial_model_sha256"]); adapter=wiring.materialize_ours_true_u_adapter(true_action=act,output_dir=out/"_adapter"); prov=wiring.arm_provenance(feature=p["features"],feature_audit=p["feature_audit"],split=p["split"],split_audit=p["split_audit"],utility=adapter["utility"],utility_audit=adapter["utility_audit"],semantic=adapter["semantic"],semantic_audit=adapter["semantic_audit"],initialization=init,output=out/"pre_gt"); expected=_expected_prediction(dataset,seed); prov=replace(prov,expected_prediction_sha256=expected); runtime=runner.runtime_config(run); _req(runtime.refresh_interval==100,"D6_B0_REFRESH_CHANGED"); src=_sha(alternating.__file__); rec=[]
 with instrument(rec): sealed=run_pre_gt(runtime,prov)
 _req(_sha(alternating.__file__)==src,"D6_B0_RELEASE_CHANGED"); audit=json.loads(sealed.audit.read_text()); row={"dataset":dataset,"seed":seed,"exact_prediction":audit["prediction_logical_sha256"]==expected,"epochs":len(rec),"families":{k:{**_q([x[k]["cosine"] for x in rec]),"retention":_q([x[k]["retention"] for x in rec]),"norm_ratio":_q([x[k]["norm_ratio"] for x in rec])} for k in ("encoder","cluster","combined")},"no_gt":True,"restored":alternating.run_relation_refinement_phase is not None}; (out/"geometry.json").write_text(json.dumps(row,indent=2)+"\n"); return row
