"""Final read-only D7-C0 sparse semantic-orientation robustness diagnostic."""
import ast,hashlib,itertools,json
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
ROOT=Path('outputs/paper/diagnostics/d7_c0_sparse_orientation_robustness');DATASETS=('Caltech-6V','MSRC-v1','BDGP');SEEDS=(20,30,50)
def _req(x,m):
 if not x:raise RuntimeError(m)
def _cell(d,s,root=ROOT):return Path(root)/runner.slug(d)/('seed'+str(s))
def _split(paths,d,k):
 with np.load(paths['split'],allow_pickle=False)as z:x={n:np.ascontiguousarray(z[n])for n in z.files}
 return SparseLabelSplit(x['sample_ids'],x['labeled_ids'],x['labeled_targets'],x['unlabeled_ids'],k,2,20,d)
def _stats(x):
 x=np.asarray(x,float);return {'min':float(x.min()),'p25':float(np.quantile(x,.25)),'median':float(np.median(x)),'mean':float(x.mean()),'p75':float(np.quantile(x,.75)),'max':float(x.max())}
def balanced_subsets(ids,targets,k):
 groups=[np.asarray(ids)[np.asarray(targets)==c]for c in range(k)];_req(all(len(g)==2 for g in groups),'D7_C0_NOT_TWO_PER_CLASS');return tuple(np.asarray(p,dtype=np.int64)for p in itertools.product(*groups))
def fit_subset_mapping(vote,selected,full_ids,full_targets,k):
 target_by_id={int(i):int(t)for i,t in zip(full_ids,full_targets)};y=np.asarray([target_by_id[int(i)]for i in selected]);cont=np.zeros((k,k),float)
 for c in range(k):cont[:,c]=vote[np.asarray(selected)[y==c]].sum(axis=0)
 rows,cols=linear_sum_assignment(-cont);m=np.empty(k,dtype=np.int64);m[rows]=cols;return m
def flip_masses(full_pred,sub_pred,target_full,u,balance):
 fc=(u*(sub_pred!=full_pred)).sum()/u.sum();tf=sub_pred[:,None,:]==target_full[None,:,None];t0=full_pred[:,None,:]==target_full[None,:,None];w=u[:,None,:]*balance;fr=(w*(tf!=t0)).sum()/w.sum();per=[]
 for s in range(u.shape[1]):per.append(float((w[:,:,s]*(tf[:,:,s]!=t0[:,:,s])).sum()/w[:,:,s].sum()))
 return float(fc),float(fr),per
def margins(cont,mapping):
 z=[]
 for r,a in enumerate(mapping):z.append((cont[r,a]-np.max(np.delete(cont[r],a)))/max(float(cont[r].sum()),np.finfo(float).eps))
 return _stats(z)
def run_one(dataset,seed,output_root=ROOT):
 _req(dataset in DATASETS and seed in SEEDS,'D7_C0_REQUEST_INVALID');out=_cell(dataset,seed,output_root);_req(not out.exists(),'D7_C0_OUTPUT_EXISTS');run=runner.FormalRun(dataset,seed,'OURS_TRUE_U','cpu');paths=runner.paths_for(run);pipeline._verify_inputs(paths['features'].parent,dataset);init=preparation.verify_initialization(paths['initialization'],dataset=dataset,training_seed=seed);act=actions.verify_true_action(paths['action'],dataset=dataset,training_seed=seed,initial_model_sha256=init['initial_model_sha256'])
 with np.load(act['artifact'],allow_pickle=False)as z:u=np.ascontiguousarray(z['U_cycle']);y=np.ascontiguousarray(z['y_gen']);stored_t=np.ascontiguousarray(z['PredRelation_true']);stored_b=np.ascontiguousarray(z['relation_balance_weights_true']);stored_l=np.ascontiguousarray(z['labeled_ids']);stored_ul=np.ascontiguousarray(z['unlabeled_ids'])
 audit=act['audit'];_req(ndarray_sha256(u)==audit['U_cycle_logical_sha256'] and ndarray_sha256(y)==audit['y_gen_logical_sha256'],'D7_C0_ACTION_PROVENANCE');k=int(y.max())+1;split=_split(paths,dataset,k);_req(np.array_equal(split.labeled_ids,stored_l)and np.array_equal(split.unlabeled_ids,stored_ul),'D7_C0_SPLIT_PARITY')
 with np.load(paths['initialization']/'carrier_state.npz',allow_pickle=False)as z:v=int(z['q_aligned'].shape[1])
 sem=build_relation_semantics(y,split,build_directional_actions(v));_req(np.array_equal(sem.pred_relation,stored_t)and np.array_equal(sem.balance_weights,stored_b),'D7_C0_R3_PARITY');vote=build_vote_semantic_state(y,k);fitted=fit_sparse_mapping(vote,split);_req(np.array_equal(fitted.mapping,sem.sparse_mapping),'D7_C0_MAPPING_PARITY');lookup={int(i):j for j,i in enumerate(split.sample_ids)};rows=np.asarray([lookup[int(i)]for i in split.unlabeled_ids]);us=u[rows];subsets=balanced_subsets(split.labeled_ids,split.labeled_targets,k);maps=[];fcs=[];frs=[];pa=[]
 for selected in subsets:
  m=fit_subset_mapping(vote,selected,split.labeled_ids,split.labeled_targets,k);pred=m[y[rows]];fc,fr,per=flip_masses(sem.class_pred,pred,split.labeled_targets,us,sem.balance_weights);maps.append(m);fcs.append(fc);frs.append(fr);pa.append(per)
 maps=np.stack(maps);cards=build_directional_actions(v).generators;card_summary={}
 for card in sorted({len(g)for g in cards}):card_summary[str(card)]=_stats(np.asarray(pa)[:,[i for i,g in enumerate(cards)if len(g)==card]].mean(axis=1))
 row={'dataset':dataset,'seed':seed,'subset_count':len(subsets),'expected_subset_count':2**k,'r3_exact':True,'u_y_provenance':True,'mapping':{'exact_stability_fraction':float(np.all(maps==sem.sparse_mapping,axis=1).mean()),'entry_agreement':_stats((maps==sem.sparse_mapping).mean(axis=1)),'hamming':_stats((maps!=sem.sparse_mapping).sum(axis=1))},'F_class':_stats(fcs),'F_relation':_stats(frs),'F_relation_per_action':[_stats(np.asarray(pa)[:,s])for s in range(len(cards))],'F_relation_generator_cardinality':card_summary,'full_mapping_margin':margins(fitted.soft_contingency,fitted.mapping),'no_gt':True,'no_training':True};_req(row['subset_count']==row['expected_subset_count'],'D7_C0_SUBSET_COUNT');out.mkdir(parents=True);(out/'d7_c0_cell.json').write_text(json.dumps(row,indent=2,sort_keys=True)+'\n');return row
def gates(rows):
 _req(len(rows)==9,'D7_C0_INCOMPLETE');by={(r['dataset'],r['seed']):r for r in rows};g1=all(r['r3_exact']and r['u_y_provenance']and r['subset_count']==r['expected_subset_count']and r['no_gt']and r['no_training']for r in rows);g2=[];g3=[];g4=[]
 for s in SEEDS:
  m,c,b=by[('MSRC-v1',s)],by[('Caltech-6V',s)],by[('BDGP',s)];g2.append(m['mapping']['exact_stability_fraction']<c['mapping']['exact_stability_fraction']and m['mapping']['exact_stability_fraction']<b['mapping']['exact_stability_fraction']);g3.append(m['F_class']['median']>c['F_class']['median']and m['F_class']['median']>b['F_class']['median']);g4.append(m['F_relation']['median']>c['F_relation']['median']and m['F_relation']['median']>b['F_relation']['median'])
 return {'gate1_integrity':g1,'gate2_msrc_mapping_fragility':sum(g2)>=2,'gate3_msrc_utility_class_flip':sum(g3)>=2,'gate4_msrc_r4_target_flip':sum(g4)>=2,'gate2_seed_passes':g2,'gate3_seed_passes':g3,'gate4_seed_passes':g4,'final':'SUPPORTS_SPARSE_ORIENTATION_ROBUSTNESS_PILOT'if g1 and sum(g2)>=2 and sum(g3)>=2 and sum(g4)>=2 else 'FAIL-CLOSED'}
