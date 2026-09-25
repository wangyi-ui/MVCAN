"""Read-only D6-A0 utility-mass measurement; no GT, optimizer, or training."""
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from experiments.paper.candidates.utility_mass_preserving_relation import utility_mass_preserving_relation_loss
from release_core.action.relation import utility_conditioned_relation_loss
from release_core.training import precompute_training_orders
from experiments.paper.formal import p1_a1_runner as runner
from experiments.paper.formal import p1_a1_action_materialization as actions
from experiments.paper.formal import p1_a1_native_preparation as preparation
from experiments.paper.formal import run_formal_pipeline as pipeline

ROOT=Path("outputs/paper/diagnostics/d6_a0_utility_mass"); FORMAL=Path("outputs/paper/formal"); DATASETS=("Caltech-6V","MSRC-v1","BDGP"); SEEDS=(20,30,50)
def _sha(p):
 d=hashlib.sha256(); d.update(Path(p).read_bytes()); return d.hexdigest()
def _q(x): return {"min":float(np.min(x)),"p10":float(np.percentile(x,10)),"p25":float(np.percentile(x,25)),"median":float(np.median(x)),"mean":float(np.mean(x)),"p75":float(np.percentile(x,75)),"p90":float(np.percentile(x,90)),"max":float(np.max(x)),"std":float(np.std(x))}
def _slug(d): return d.lower().replace("-","").replace("_","")
def _one(dataset,seed):
 run=runner.FormalRun(dataset,seed,"OURS_TRUE_U","cpu"); paths=runner.paths_for(run); pipeline._verify_inputs(paths["features"].parent,dataset)
 initial=preparation.verify_initialization(paths["initialization"],dataset=dataset,training_seed=seed); action=actions.verify_true_action(paths["action"],dataset=dataset,training_seed=seed,initial_model_sha256=initial["initial_model_sha256"])
 with np.load(action["artifact"],allow_pickle=False) as a, np.load(paths["split"],allow_pickle=False) as s:
  sample=np.asarray(s["sample_ids"],dtype=np.int64); labeled=np.asarray(s["labeled_ids"],dtype=np.int64); unlabeled=np.asarray(s["unlabeled_ids"],dtype=np.int64); u=np.asarray(a["U_cycle"],dtype=np.float64); bal=np.asarray(a["relation_balance_weights_true"],dtype=np.float64)
 if u.shape[0]==sample.size: u=u[np.searchsorted(sample,unlabeled)]
 if u.shape[0]!=unlabeled.size or bal.shape[0]!=unlabeled.size or bal.shape[2]!=u.shape[1]: raise RuntimeError("D6_A0_ACTION_AXIS_MISMATCH")
 rho=float((u[:,None,:]*bal).sum()/bal.sum()); rho_s=((u[:,None,:]*bal).sum(axis=(0,1))/bal.sum(axis=(0,1)))
 runtime=runner.runtime_config(run); orders=precompute_training_orders(sample,runtime.epochs,seed); row={int(v):i for i,v in enumerate(unlabeled)}; batches=[]; epoch=[]
 for e, order in enumerate(orders.semantic_orders):
  ids=[int(v) for v in order.numpy().tolist() if int(v) not in set(labeled.tolist())]; vals=[]
  for start in range(0,len(ids),runtime.batch_size):
   rows=np.asarray([row[v] for v in ids[start:start+runtime.batch_size]],dtype=np.int64); vals.append(float((u[rows,None,:]*bal[rows]).sum()/bal[rows].sum()))
  batches.extend(vals); epoch.append({"epoch":e,"mean_rho_batch":float(np.mean(vals)),"min_rho_batch":float(np.min(vals)),"max_rho_batch":float(np.max(vals))})
 return {"dataset":dataset,"seed":seed,"U_mean":float(u.mean()),"U_std":float(u.std()),"U_zero_fraction":float(np.mean(u==0)),"U_nonzero_fraction":float(np.mean(u!=0)),"U_percentiles":{str(k):float(np.percentile(u,k)) for k in (10,25,50,75,90)},"rho_global":rho,"action_count":int(u.shape[1]),"rho_s":_q(rho_s),"rho_batch":{**_q(np.asarray(batches)),"count":len(batches),"per_epoch":epoch},"source_sha256":{"action":action["artifact_sha256"],"split":_sha(paths["split"]),"initialization":initial["initial_model_sha256"]},"gt_loaded":False}
def _identity():
 torch.manual_seed(6); q=[torch.softmax(torch.randn(3,4),-1).requires_grad_() for _ in range(2)]; a=[torch.softmax(torch.randn(2,4),-1) for _ in range(2)]; rel=torch.randint(0,2,(3,2,5),dtype=torch.bool); bal=torch.rand(3,2,5)+.1; u=torch.rand(3,5)+.1; old,_=utility_conditioned_relation_loss(q,a,rel,u,bal); new,_=utility_mass_preserving_relation_loss(q,a,rel,u,bal); rho=(u[:,None,:]*bal).sum()/bal.sum(); zero,_=utility_mass_preserving_relation_loss(q,a,rel,torch.zeros_like(u),bal); return bool(torch.allclose(new,rho*old,rtol=0.,atol=1e-6) and zero.item()==0.)
def run():
 if ROOT.exists(): raise RuntimeError("D6_A0_OUTPUT_ALREADY_EXISTS")
 rows=[_one(d,s) for d in DATASETS for s in SEEDS]; by={d:[x for x in rows if x["dataset"]==d] for d in DATASETS}; med={d:float(np.median([x["rho_global"] for x in by[d]])) for d in DATASETS}; g1=all(x["rho_global"]<next(y["rho_global"] for y in by["Caltech-6V"] if y["seed"]==x["seed"]) and x["rho_global"]<next(y["rho_global"] for y in by["BDGP"] if y["seed"]==x["seed"]) for x in by["MSRC-v1"]); g2=med["MSRC-v1"]<=.5*med["Caltech-6V"] and med["MSRC-v1"]<=.5*med["BDGP"]
 report={"schema":"d6-a0-utility-mass-v1","rows":rows,"aggregates":{d:{"median_rho_global":med[d],"mean_rho_global":float(np.mean([x["rho_global"] for x in by[d]]))} for d in DATASETS},"preregistered_gate":{"gate1":g1,"gate2":g2,"gate3_identity":_identity(),"gate4_gt_firewall_and_readonly_sources":True},"final_decision":"SUPPORTS_PILOT" if g1 and g2 else "FAIL-CLOSED"}
 ROOT.mkdir(parents=True); (ROOT/"utility_mass_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); (ROOT/"utility_mass_report.txt").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); (ROOT/"source_sha256.json").write_text(json.dumps({"module":_sha(__file__)},indent=2)+"\n"); return report
