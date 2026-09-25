"""Read-only D7-B0 utility-conditioned semantic relation ambiguity diagnostic."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from release_core.data.weak_quality import ndarray_sha256
from release_core.semantics import SparseLabelSplit,build_vote_semantic_state,fit_sparse_mapping,build_relation_semantics
from release_core.utility import build_directional_actions
from experiments.paper.formal import p1_a1_action_materialization as actions
from experiments.paper.formal import p1_a1_native_preparation as preparation
from experiments.paper.formal import p1_a1_runner as runner
from experiments.paper.formal import run_formal_pipeline as pipeline
ROOT=Path("outputs/paper/diagnostics/d7_b0_relation_ambiguity"); DATASETS=("Caltech-6V","MSRC-v1","BDGP"); SEEDS=(20,30,50)
def _req(x,m):
 if not x: raise RuntimeError(m)
def _cell(d,s,root=ROOT): return Path(root)/runner.slug(d)/("seed"+str(s))
def _split(paths,dataset,k):
 with np.load(paths["split"],allow_pickle=False) as z: x={n:np.ascontiguousarray(z[n]) for n in z.files}
 return SparseLabelSplit(x["sample_ids"],x["labeled_ids"],x["labeled_targets"],x["unlabeled_ids"],k,2,20,dataset)
def _weighted_quantile(values,weights,q):
 order=np.argsort(values,kind="mergesort"); v=np.asarray(values)[order]; w=np.asarray(weights,dtype=float)[order]; return float(v[np.searchsorted(np.cumsum(w),q*w.sum(),side="left")])
def consensus(class_pred,u):
 mass=u.sum(axis=1); valid=mass>0; k=int(class_pred.max())+1; probs=np.zeros((class_pred.shape[0],k),float)
 for c in range(k): probs[:,c]=(u*(class_pred==c)).sum(axis=1)
 probs[valid]/=mass[valid,None]; cc=probs.max(axis=1); ent=np.zeros_like(cc); nz=probs>0; ent[valid]=-(np.where(nz[valid],probs[valid]*np.log(np.where(nz[valid],probs[valid],1)),0).sum(axis=1)/np.log(k))
 def stats(v): return {"mean":float(np.average(v[valid],weights=mass[valid])),"median":_weighted_quantile(v[valid],mass[valid],.5),"p25":_weighted_quantile(v[valid],mass[valid],.25),"p75":_weighted_quantile(v[valid],mass[valid],.75)}
 return {"class_consensus":stats(cc),"normalized_entropy":stats(ent),"zero_utility_sample_fraction":float((~valid).mean())}
def contradiction(target,u,balance=None):
 w=u[:,None,:] if balance is None else u[:,None,:]*balance; same=(w*target).sum(axis=2); diff=(w*(~target)).sum(axis=2); pair=same+diff; valid=pair>0; _req(np.any(valid),"D7_B0_ZERO_PAIR_MASS"); c=np.minimum(same[valid],diff[valid])/pair[valid]
 return {"D_global":float(np.minimum(same,diff).sum()/pair.sum()),"pair_median":float(np.median(c)),"pair_p25":float(np.quantile(c,.25)),"pair_p75":float(np.quantile(c,.75)),"pair_max":float(c.max()),"same_dominant_pair_fraction":float((same[valid]>diff[valid]).mean()),"different_dominant_pair_fraction":float((diff[valid]>same[valid]).mean())}
def _loo_mapping(vote,labeled_ids,targets,k,drop):
 keep=np.arange(len(labeled_ids))!=drop; ids=np.asarray(labeled_ids)[keep]; y=np.asarray(targets)[keep]; cont=np.zeros((k,k),float)
 for c in range(k): cont[:,c]=vote[ids[y==c]].sum(axis=0)
 r,col=linear_sum_assignment(-cont); mapping=np.empty(k,dtype=np.int64);mapping[r]=col;return mapping
def mapping_stability(vote,labeled_ids,targets,full,k):
 maps=np.stack([_loo_mapping(vote,labeled_ids,targets,k,j) for j in range(len(labeled_ids))]); return {"mapping_exact_stability":float(np.all(maps==full,axis=1).mean()),"mapping_entry_stability":float((maps==full).mean())}
def margins(cont,mapping):
 out=[]
 for row,assigned in enumerate(mapping):
  runner=np.max(np.delete(cont[row],assigned)); out.append((cont[row,assigned]-runner)/max(float(cont[row].sum()),np.finfo(float).eps))
 return {"min":float(np.min(out)),"median":float(np.median(out)),"mean":float(np.mean(out)),"max":float(np.max(out))}
def run_one(dataset,seed,output_root=ROOT):
 _req(dataset in DATASETS and seed in SEEDS,"D7_B0_REQUEST_INVALID"); out=_cell(dataset,seed,output_root);_req(not out.exists(),"D7_B0_OUTPUT_EXISTS")
 run=runner.FormalRun(dataset,seed,"OURS_TRUE_U","cpu");paths=runner.paths_for(run);pipeline._verify_inputs(paths["features"].parent,dataset);init=preparation.verify_initialization(paths["initialization"],dataset=dataset,training_seed=seed);act=actions.verify_true_action(paths["action"],dataset=dataset,training_seed=seed,initial_model_sha256=init["initial_model_sha256"])
 with np.load(act["artifact"],allow_pickle=False) as z: u=np.ascontiguousarray(z["U_cycle"]);y=np.ascontiguousarray(z["y_gen"]);stored_t=np.ascontiguousarray(z["PredRelation_true"]);stored_b=np.ascontiguousarray(z["relation_balance_weights_true"]);stored_l=np.ascontiguousarray(z["labeled_ids"]);stored_ul=np.ascontiguousarray(z["unlabeled_ids"])
 audit=act["audit"];_req(ndarray_sha256(u)==audit["U_cycle_logical_sha256"] and ndarray_sha256(y)==audit["y_gen_logical_sha256"],"D7_B0_ACTION_PROVENANCE")
 k=int(y.max())+1; split=_split(paths,dataset,k);_req(np.array_equal(split.labeled_ids,stored_l) and np.array_equal(split.unlabeled_ids,stored_ul),"D7_B0_SPLIT_PARITY")
 a=build_directional_actions(y.shape[1] if False else np.load(paths["initialization"]/'carrier_state.npz',allow_pickle=False)["q_aligned"].shape[1]); vote=build_vote_semantic_state(y,k); fitted=fit_sparse_mapping(vote,split); sem=build_relation_semantics(y,split,a)
 _req(np.array_equal(sem.pred_relation,stored_t) and np.array_equal(sem.balance_weights,stored_b),"D7_B0_R3_PARITY")
 rows=np.asarray([np.where(split.sample_ids==i)[0][0] for i in split.unlabeled_ids]); us=u[rows]; con=consensus(sem.class_pred,us); bal=contradiction(sem.pred_relation,us,sem.balance_weights);raw=contradiction(sem.pred_relation,us)
 row={"dataset":dataset,"seed":seed,"r3_exact":True,"u_y_provenance":True,"consensus":con,"D_balanced":bal,"D_raw":raw,"mapping":{**mapping_stability(vote,split.labeled_ids,split.labeled_targets,fitted.mapping,k),"margin":margins(fitted.soft_contingency,fitted.mapping)},"no_gt":True,"no_training":True};out.mkdir(parents=True);(out/'d7_b0_cell.json').write_text(json.dumps(row,indent=2,sort_keys=True)+"\n");return row
def gates(rows):
 _req(len(rows)==9,"D7_B0_INCOMPLETE");by={(r['dataset'],r['seed']):r for r in rows};g1=all(r['r3_exact'] and r['u_y_provenance'] and r['no_gt'] and r['no_training'] for r in rows);g2=[];g3=[];g4=[]
 for s in SEEDS:
  m,c,b=(by[('MSRC-v1',s)],by[('Caltech-6V',s)],by[('BDGP',s)]);g2.append(m['D_balanced']['D_global']>c['D_balanced']['D_global'] and m['D_balanced']['D_global']>b['D_balanced']['D_global']);g3.append(m['consensus']['class_consensus']['mean']<c['consensus']['class_consensus']['mean'] and m['consensus']['class_consensus']['mean']<b['consensus']['class_consensus']['mean'] and m['consensus']['normalized_entropy']['mean']>c['consensus']['normalized_entropy']['mean'] and m['consensus']['normalized_entropy']['mean']>b['consensus']['normalized_entropy']['mean']);g4.append(m['mapping']['mapping_exact_stability']<c['mapping']['mapping_exact_stability'] and m['mapping']['mapping_exact_stability']<b['mapping']['mapping_exact_stability'] or m['mapping']['margin']['median']<c['mapping']['margin']['median'] and m['mapping']['margin']['median']<b['mapping']['margin']['median'])
 return {'gate1_integrity':g1,'gate2_msrc_relation_contradiction':sum(g2)>=2,'gate3_msrc_semantic_consensus':sum(g3)>=2,'gate4_weak_quantity_orientation':sum(g4)>=2,'gate2_seed_passes':g2,'gate3_seed_passes':g3,'gate4_seed_passes':g4,'final':'SUPPORTS_RELATION_AMBIGUITY_PILOT' if g1 and sum(g2)>=2 and sum(g3)>=2 else 'FAIL-CLOSED'}
