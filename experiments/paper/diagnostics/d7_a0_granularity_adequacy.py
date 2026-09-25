"""Read-only D7-A0 utility-conditioned granularity adequacy diagnostic."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from release_core.utility import build_directional_actions
from release_core.data.weak_quality import ndarray_sha256
from experiments.paper.formal import p1_a1_action_materialization as actions
from experiments.paper.formal import p1_a1_native_preparation as preparation
from experiments.paper.formal import p1_a1_runner as runner
from experiments.paper.formal import run_formal_pipeline as pipeline

ROOT=Path("outputs/paper/diagnostics/d7_a0_granularity_adequacy")
DATASETS=("Caltech-6V","MSRC-v1","BDGP"); SEEDS=(20,30,50)
STRONG,SINGLETON,CONFLICT,ORPHAN=range(4)
STATE_NAMES=("strong","singleton","conflict","orphan")
def _req(x,m):
 if not x: raise RuntimeError(m)
def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def granularities(k): return (int(np.ceil(k/2.0)),k,2*k)
def generator_predictions(q_aligned, generators):
 q=np.asarray(q_aligned); _req(q.ndim==3,"D7_A0_Q_SHAPE")
 return np.stack([q[:,tuple(group),:].mean(axis=1) for group in generators],axis=1)
def ward_partitions(p_gen, counts):
 z=linkage(np.asarray(p_gen),method="ward")
 out={}
 for count in counts:
  labels=cut_tree(z,n_clusters=[count]).reshape(-1).astype(np.int64)
  _req(np.unique(labels).size==count,"D7_A0_CLUSTER_COUNT")
  out[int(count)]=labels
 return out
def anchor_states(labels,labeled_ids,labeled_targets):
 labels=np.asarray(labels); state=np.full(labels.shape,ORPHAN,dtype=np.int8)
 for cluster in np.unique(labels):
  targets=np.asarray(labeled_targets)[np.asarray(labels)[np.asarray(labeled_ids)]==cluster]
  mask=labels==cluster
  if targets.size==0: state[mask]=ORPHAN
  elif targets.size==1: state[mask]=SINGLETON
  elif np.unique(targets).size==1: state[mask]=STRONG
  else: state[mask]=CONFLICT
 return state
def mass_summary(states,u_cycle,unlabeled_ids,generators):
 u=np.asarray(u_cycle,dtype=np.float64); ids=np.asarray(unlabeled_ids,dtype=np.int64); _req(u.ndim==2 and states.shape==(u.shape[0],u.shape[1]),"D7_A0_MASS_SHAPE")
 denominator=float(u[ids].sum()); _req(denominator>0,"D7_A0_ZERO_UTILITY")
 def mass(sample_ids, action_ids):
  values=u[np.ix_(sample_ids,action_ids)]; selected=states[np.ix_(sample_ids,action_ids)]
  return {STATE_NAMES[t]:float(values[selected==t].sum()/values.sum()) if values.sum()>0 else 0.0 for t in range(4)}
 global_mass=mass(ids,np.arange(u.shape[1])); _req(abs(sum(global_mass.values())-1.0)<=1e-12,"D7_A0_MASS_NOT_UNIT")
 per_action=[mass(ids,np.array([s])) for s in range(u.shape[1])]
 cards={}
 for card in sorted({len(g) for g in generators}): cards[str(card)]=mass(ids,np.array([s for s,g in enumerate(generators) if len(g)==card]))
 return {"global":global_mass,"per_action":per_action,"generator_cardinality":cards}
def dominates(candidate,native): return candidate["strong"]>native["strong"] and candidate["conflict"]<=native["conflict"]
def preferred(masses,coarse,native,fine):
 candidates=[name for name,count in (("coarse",coarse),("fine",fine)) if dominates(masses[count]["global"],masses[native]["global"])]
 if not candidates: return "native"
 if len(candidates)==1: return candidates[0]
 a,b=(masses[coarse]["global"],masses[fine]["global"])
 if a["strong"]!=b["strong"]: return "coarse" if a["strong"]>b["strong"] else "fine"
 if a["conflict"]!=b["conflict"]: return "coarse" if a["conflict"]<b["conflict"] else "fine"
 return "ambiguous"
def _cell(dataset,seed,output_root=ROOT): return Path(output_root)/runner.slug(dataset)/("seed"+str(seed))
def run_one(dataset,seed,output_root=ROOT):
 _req(dataset in DATASETS and seed in SEEDS,"D7_A0_REQUEST_INVALID"); out=_cell(dataset,seed,output_root); _req(not out.exists(),"D7_A0_OUTPUT_EXISTS")
 run=runner.FormalRun(dataset,seed,"OURS_TRUE_U","cpu"); paths=runner.paths_for(run); pipeline._verify_inputs(paths["features"].parent,dataset)
 init=preparation.verify_initialization(paths["initialization"],dataset=dataset,training_seed=seed); action=actions.verify_true_action(paths["action"],dataset=dataset,training_seed=seed,initial_model_sha256=init["initial_model_sha256"])
 carrier=paths["initialization"] / "carrier_state.npz"; _req(carrier.is_file(),"D7_A0_CARRIER_MISSING")
 with np.load(carrier,allow_pickle=False) as z: q=np.ascontiguousarray(z["q_aligned"])
 with np.load(action["artifact"],allow_pickle=False) as z:
  u=np.ascontiguousarray(z["U_cycle"]); y=np.ascontiguousarray(z["y_gen"]); labeled_ids=np.ascontiguousarray(z["labeled_ids"]); unlabeled_ids=np.ascontiguousarray(z["unlabeled_ids"])
 audit=action["audit"]; init_audit=json.loads(init["audit"].read_text()); _req(ndarray_sha256(q)==audit["q_aligned_logical_sha256"]==init_audit["post_final_q_aligned_logical_sha256"],"D7_A0_Q_PROVENANCE")
 _req(ndarray_sha256(u)==audit["U_cycle_logical_sha256"],"D7_A0_U_PROVENANCE")
 _req(np.array_equal(np.arange(q.shape[0]),np.sort(np.concatenate((labeled_ids,unlabeled_ids)))),"D7_A0_SPLIT_INVALID")
 split=np.load(paths["split"],allow_pickle=False); targets=np.ascontiguousarray(split["labeled_targets"]); _req(np.array_equal(labeled_ids,split["labeled_ids"]),"D7_A0_SPLIT_PROVENANCE")
 k=q.shape[2]; actions_space=build_directional_actions(q.shape[1]); _req(y.shape==(q.shape[0],actions_space.S),"D7_A0_Y_SHAPE")
 p=generator_predictions(q,actions_space.generators); _req(np.array_equal(p.argmax(axis=2),y),"D7_A0_Y_GEN_PARITY")
 grams=granularities(k); states={g:np.empty((q.shape[0],actions_space.S),dtype=np.int8) for g in grams}
 for s in range(actions_space.S):
  for g,part in ward_partitions(p[:,s,:],grams).items(): states[g][:,s]=anchor_states(part,labeled_ids,targets)
 masses={g:mass_summary(states[g],u,unlabeled_ids,actions_space.generators) for g in grams}
 pref=preferred(masses,*grams); row={"dataset":dataset,"seed":seed,"class_count":k,"granularities":{"coarse":grams[0],"native":grams[1],"fine":grams[2]},"y_gen_argmax_exact":True,"masses":{str(g):masses[g] for g in grams},"dominates_native":{"coarse":dominates(masses[grams[0]]["global"],masses[grams[1]]["global"]),"fine":dominates(masses[grams[2]]["global"],masses[grams[1]]["global"])},"preferred":pref,"no_gt":True,"no_training":True}
 out.mkdir(parents=True); (out/"d7_a0_cell.json").write_text(json.dumps(row,indent=2,sort_keys=True)+"\n"); return row
def gates(rows):
 _req(len(rows)==9,"D7_A0_INCOMPLETE")
 gate1=all(r["y_gen_argmax_exact"] and r["no_gt"] and r["no_training"] for r in rows)
 msrc=[r for r in rows if r["dataset"]=="MSRC-v1"]; dirs=[r["preferred"] for r in msrc]; gate2=sum(x in ("coarse","fine") for x in dirs)>=2; gate3=(dirs.count("coarse")>=2 or dirs.count("fine")>=2)
 modal={d:max((sum(r["preferred"]==p for r in rows if r["dataset"]==d),p) for p in ("coarse","native","fine","ambiguous"))[1] for d in DATASETS}; gate4=len(set(modal.values()))>=2
 return {"gate1_integrity":gate1,"gate2_msrc_mismatch":gate2,"gate3_msrc_stability":gate3,"gate4_cross_dataset_heterogeneity":gate4,"modal_preference":modal,"final":"SUPPORTS_GRANULARITY_PILOT" if all((gate1,gate2,gate3,gate4)) else "FAIL-CLOSED"}
