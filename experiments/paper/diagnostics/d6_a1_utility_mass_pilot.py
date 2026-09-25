"""Isolated MSRC D6-A1 pilot: patch only frozen R4 loss symbol in-process."""
import hashlib,json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from release_core.runtime import run_pre_gt
from release_core.training import alternating
from release_core.action.relation import utility_conditioned_relation_loss as V1
from experiments.paper.candidates.utility_mass_preserving_relation import utility_mass_preserving_relation_loss
from experiments.paper.formal import p1_a1_action_materialization as actions
from experiments.paper.formal import p1_a1_native_preparation as preparation
from experiments.paper.formal import p1_a1_runner as runner
from experiments.paper.formal import p1_a3_runtime_wiring as wiring
from experiments.paper.formal import p1_a4_postseal_evaluation as postseal
from experiments.paper.formal import run_formal_pipeline as pipeline
ROOT=Path("outputs/paper/diagnostics/d6_a1_utility_mass_pilot"); DATASET="MSRC-v1"; SEEDS=(20,30,50)
BASE={20:{"acc":.6952380952380952,"nmi":.601879847159302,"ari":.5226965361956316},30:{"acc":.6952380952380952,"nmi":.5927685934640445,"ari":.5179034165212869},50:{"acc":.7047619047619048,"nmi":.6182449983617955,"ari":.5418453208981644}}
V1M={20:{"acc":.7380952380952381,"nmi":.6055785047695859,"ari":.5201585674904867},30:{"acc":.680952380952381,"nmi":.5712648728025609,"ari":.4974682470085712},50:{"acc":.680952380952381,"nmi":.5803551278710588,"ari":.4999404066225142}}
def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _req(x,m):
 if not x: raise RuntimeError(m)
def _paths(seed):
 _req(seed in SEEDS,"D6_A1_SEED_INVALID"); r=ROOT/"msrcv1"/("seed"+str(seed)); f=runner.paths_for(runner.FormalRun(DATASET,seed,"OURS_TRUE_U","cuda:0")); return r,f
@contextmanager
def patched_loss(records):
 _req(alternating.utility_conditioned_relation_loss is V1,"D6_A1_V1_SYMBOL_UNEXPECTED")
 def wrapped(*a,**k):
  loss,audit=utility_mass_preserving_relation_loss(*a,**k); records.append({"call":len(records),"B":a[0][0].shape[0],"L":a[1][0].shape[0],"S":a[3].shape[1],"rho":audit.utility_mass,"loss":float(loss.detach()),"view_losses":list(audit.view_losses),"balance_denominator":audit.balance_denominator,"action_weight_sum":audit.action_weight_sum,"detached":audit.action_weight_detached}); return loss,audit
 alternating.utility_conditioned_relation_loss=wrapped
 try:
  _req(alternating.utility_conditioned_relation_loss is wrapped,"D6_A1_CANDIDATE_NOT_ACTIVE"); yield
 finally:
  alternating.utility_conditioned_relation_loss=V1
  _req(alternating.utility_conditioned_relation_loss is V1,"D6_A1_V1_RESTORE_FAILED")
def run_seed(seed,device,full_gt_path):
 root,p=_paths(seed); _req(not root.exists(),"D6_A1_OUTPUT_EXISTS")
 pipeline._verify_inputs(p["features"].parent,DATASET); init=preparation.verify_initialization(p["initialization"],dataset=DATASET,training_seed=seed); act=actions.verify_true_action(p["action"],dataset=DATASET,training_seed=seed,initial_model_sha256=init["initial_model_sha256"])
 runtime=runner.runtime_config(runner.FormalRun(DATASET,seed,"OURS_TRUE_U",device)); _req(runtime.refresh_interval==100 and runtime.epochs==20,"D6_A1_RUNTIME_INVARIANT")
 adapter=wiring.materialize_ours_true_u_adapter(true_action=act,output_dir=root/"_adapters"/"ours"); prov=wiring.arm_provenance(feature=p["features"],feature_audit=p["feature_audit"],split=p["split"],split_audit=p["split_audit"],utility=adapter["utility"],utility_audit=adapter["utility_audit"],semantic=adapter["semantic"],semantic_audit=adapter["semantic_audit"],initialization=init,output=root/"OURS_TRUE_U_UMP")
 source=_sha(Path(alternating.__file__)); records=[]
 with patched_loss(records): sealed=run_pre_gt(runtime,prov)
 _req(_sha(Path(alternating.__file__))==source,"D6_A1_RELEASE_SOURCE_CHANGED"); postseal._validate("OURS_TRUE_U",sealed); metrics=postseal.evaluate_formal_postseal(dataset=DATASET,training_seed=seed,arm="OURS_TRUE_U",full_gt_path=full_gt_path,sealed=sealed,output_path=root/"postseal_metrics.json")["metrics"]
 rec={"seed":seed,"runtime":asdict(runtime),"initial_model_sha256":init["initial_model_sha256"],"action_sha256":act["artifact_sha256"],"calls":records,"rho":{"min":min(x["rho"] for x in records),"mean":sum(x["rho"] for x in records)/len(records),"max":max(x["rho"] for x in records)},"metrics":metrics,"gt_firewall":{"full_gt_loaded_before_seal":False,"metrics_computed_before_seal":False},"function_restored":alternating.utility_conditioned_relation_loss is V1}; (root/"d6_a1_audit.json").write_text(json.dumps(rec,indent=2)+"\n"); return rec
def summarize(rows):
 ga=sum(r["metrics"]["nmi"]>=BASE[r["seed"]]["nmi"] and r["metrics"]["ari"]>=BASE[r["seed"]]["ari"] for r in rows)>=2; mean=lambda key:sum(r["metrics"][key] for r in rows)/3; gb=mean("nmi")>sum(x["nmi"] for x in BASE.values())/3 and mean("ari")>sum(x["ari"] for x in BASE.values())/3; gc=mean("nmi")>sum(x["nmi"] for x in V1M.values())/3 and mean("ari")>sum(x["ari"] for x in V1M.values())/3; gd=all(r["function_restored"] and not r["gt_firewall"]["full_gt_loaded_before_seal"] for r in rows); return {"rows":rows,"gates":{"A":ga,"B":gb,"C":gc,"D":gd},"final_decision":"SUPPORTS_CROSS_DATASET_REPLAY" if ga and gb and gc and gd else "FAIL-CLOSED"}
