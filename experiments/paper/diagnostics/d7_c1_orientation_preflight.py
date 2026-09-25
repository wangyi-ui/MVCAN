"""Read-only D7-C1 candidate orientation preflight."""
import itertools,json
from pathlib import Path
import numpy as np
from release_core.data.weak_quality import ndarray_sha256
from release_core.semantics import SparseLabelSplit,build_relation_semantics
from release_core.utility import build_directional_actions
from experiments.paper.candidates import utility_weighted_sparse_orientation as candidate
from experiments.paper.diagnostics import d7_c0_sparse_orientation_robustness as c0
from experiments.paper.formal import p1_a1_action_materialization as actions
from experiments.paper.formal import p1_a1_native_preparation as preparation
from experiments.paper.formal import p1_a1_runner as runner
from experiments.paper.formal import run_formal_pipeline as pipeline
ROOT=Path('outputs/paper/diagnostics/d7_c1_orientation_preflight');DATASETS=('Caltech-6V','MSRC-v1','BDGP');SEEDS=(20,30,50)
def _req(x,m):
 if not x:raise RuntimeError(m)
def _stats(x):
 x=np.asarray(x,float);return {'min':float(x.min()),'p25':float(np.quantile(x,.25)),'median':float(np.median(x)),'mean':float(x.mean()),'p75':float(np.quantile(x,.75)),'max':float(x.max())}
def _split(paths,d,k):
 with np.load(paths['split'],allow_pickle=False)as z:x={n:np.ascontiguousarray(z[n])for n in z.files}
 return SparseLabelSplit(x['sample_ids'],x['labeled_ids'],x['labeled_targets'],x['unlabeled_ids'],k,2,20,d)
def _subs(ids,targets,k):
 g=[np.asarray(ids)[np.asarray(targets)==c]for c in range(k)];_req(all(len(x)==2 for x in g),'D7_C1_SUBSET_INVALID');return tuple(np.asarray(x,dtype=np.int64)for x in itertools.product(*g))
def _subset_map(vote,selected,split):
 lookup={int(x):i for i,x in enumerate(split.sample_ids)};rows=np.asarray([lookup[int(x)]for x in selected]);t={int(i):int(v)for i,v in zip(split.labeled_ids,split.labeled_targets)};targets=np.asarray([t[int(x)]for x in selected]);return candidate._mapping(vote,rows,targets,split.class_count)[0]
def _flip(full,sub,targets,u,balance):
 fc=float((u*(sub!=full)).sum()/u.sum());a=sub[:,None,:]==targets[None,:,None];b=full[:,None,:]==targets[None,:,None];w=u[:,None,:]*balance;return fc,float((w*(a!=b)).sum()/w.sum())
def run_one(dataset,seed,output_root=ROOT):
 out=Path(output_root)/runner.slug(dataset)/('seed'+str(seed));_req(dataset in DATASETS and seed in SEEDS,'D7_C1_REQUEST_INVALID');_req(not out.exists(),'D7_C1_OUTPUT_EXISTS');run=runner.FormalRun(dataset,seed,'OURS_TRUE_U','cpu');p=runner.paths_for(run);pipeline._verify_inputs(p['features'].parent,dataset);init=preparation.verify_initialization(p['initialization'],dataset=dataset,training_seed=seed);act=actions.verify_true_action(p['action'],dataset=dataset,training_seed=seed,initial_model_sha256=init['initial_model_sha256'])
 with np.load(act['artifact'],allow_pickle=False)as z:u=np.ascontiguousarray(z['U_cycle']);y=np.ascontiguousarray(z['y_gen']);old_t=np.ascontiguousarray(z['PredRelation_true']);old_b=np.ascontiguousarray(z['relation_balance_weights_true'])
 _req(ndarray_sha256(u)==act['audit']['U_cycle_logical_sha256']and ndarray_sha256(y)==act['audit']['y_gen_logical_sha256'],'D7_C1_PROVENANCE');k=int(y.max())+1;split=_split(p,dataset,k)
 with np.load(p['initialization']/'carrier_state.npz',allow_pickle=False)as z:v=int(z['q_aligned'].shape[1])
 old=build_relation_semantics(y,split,build_directional_actions(v));_req(np.array_equal(old.pred_relation,old_t)and np.array_equal(old.balance_weights,old_b),'D7_C1_V1_PARITY');full=candidate.build_utility_weighted_relation_semantics(y,u,split,build_directional_actions(v));subs=_subs(split.labeled_ids,split.labeled_targets,k);maps=[];fcs=[];frs=[]
 for chosen in subs:
  m=_subset_map(full['vote_state'],chosen,split);pred=m[y[full['unlabeled_rows']]];fc,fr=_flip(full['class_pred'],pred,split.labeled_targets,u[full['unlabeled_rows']],full['balance_weights']);maps.append(m);fcs.append(fc);frs.append(fr)
 maps=np.stack(maps);oldrow=next(x for x in json.loads((c0.ROOT/'d7_c0_summary.json').read_text())['rows']if x['dataset']==dataset and x['seed']==seed);row={'dataset':dataset,'seed':seed,'labeled_utility_mass':_stats(u[full['labeled_rows']].sum(1)),'mapping_U_full':full['mapping'].tolist(),'full_mapping_equal_to_v1':bool(np.array_equal(full['mapping'],old.sparse_mapping)),'full_pred_relation_equal_to_v1':bool(np.array_equal(full['pred_relation'],old.pred_relation)),'full_relation_flip_vs_v1':float(((u[full['unlabeled_rows'],None,:]*old.balance_weights)*(full['pred_relation']!=old.pred_relation)).sum()/(u[full['unlabeled_rows'],None,:]*old.balance_weights).sum()),'subset_count':len(subs),'expected_subset_count':2**k,'candidate_exact_mapping_stability':float(np.all(maps==full['mapping'],axis=1).mean()),'candidate_mapping_entry_agreement':_stats((maps==full['mapping']).mean(1)),'candidate_F_class':_stats(fcs),'candidate_F_relation':_stats(frs),'frozen_d7_c0_reference':{'exact_stability':oldrow['mapping']['exact_stability_fraction'],'F_class_median':oldrow['F_class']['median'],'F_relation_median':oldrow['F_relation']['median']},'no_gt':True,'no_training':True};_req(row['labeled_utility_mass']['min']>0 and row['subset_count']==row['expected_subset_count'],'D7_C1_PREFLIGHT_INTEGRITY');out.mkdir(parents=True);(out/'d7_c1_preflight_cell.json').write_text(json.dumps(row,indent=2,sort_keys=True)+'\n');return row
def gates(rows):
 _req(len(rows)==9,'D7_C1_INCOMPLETE');by={(x['dataset'],x['seed']):x for x in rows};g1=all(x['labeled_utility_mass']['min']>0 and x['subset_count']==x['expected_subset_count']and x['no_gt']and x['no_training']for x in rows);others=[x for x in rows if x['dataset']in('Caltech-6V','BDGP')];g2=all(x['full_mapping_equal_to_v1']and x['full_pred_relation_equal_to_v1']and x['candidate_exact_mapping_stability']==1 and x['candidate_F_class']['median']==0 and x['candidate_F_relation']['median']==0 for x in others);m=[by[('MSRC-v1',s)]for s in SEEDS];g3=sum(x['candidate_F_class']['median']<x['frozen_d7_c0_reference']['F_class_median']and x['candidate_F_relation']['median']<x['frozen_d7_c0_reference']['F_relation_median']for x in m)>=2;g3=g3 and np.mean([x['candidate_exact_mapping_stability']for x in m])>np.mean([x['frozen_d7_c0_reference']['exact_stability']for x in m]);g4=sum(not x['full_mapping_equal_to_v1']for x in m)>=2;return {'S0_1_integrity':g1,'S0_2_caltech_bdgp_preservation':g2,'S0_3_msrc_robustness_improvement':bool(g3),'S0_4_msrc_actionability':g4,'stage0':'SUPPORTS_CAUSAL_PILOT'if all((g1,g2,g3,g4))else'FAIL-CLOSED'}
