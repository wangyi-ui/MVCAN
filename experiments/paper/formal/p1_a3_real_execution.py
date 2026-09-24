"""Persisted P1-A3R1 formal materialization over frozen release primitives."""
import json, os, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from release_core.backbone import MultiViewBackbone
from release_core.backbone.clustering import initialize_kmeans_centers, native_refresh_from_latents
from release_core.backbone.native_objective import native_objective
from release_core.config import get_native_config
from release_core.data.weak_quality import ndarray_sha256
from release_core.semantics import SparseLabelSplit, build_relation_semantics
from release_core.utility import build_directional_actions, compute_directional_cycle_utility
import release_core.runtime.entrypoint as entry
from . import p1_a0_formal_protocol as p0
from . import p1_a2_execution_contract as p2

def sha(path):
 d=hashlib.sha256();
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): d.update(b)
 return d.hexdigest()
def write(path,x):
 with Path(path).open('x',encoding='utf8') as f: json.dump(x,f,sort_keys=True,indent=2);f.write('\n')
def item(dataset): return next(x for x in p0.FORMAL_DATASETS if x.name==dataset)

def _refresh(model, views, weights, seed, device):
 lat=[]; q=[]
 with torch.no_grad():
  for ae,v in zip(model.autoencoders,views):
   z=ae.encoder(v); qq=ae.clustering(z);lat.append(z.detach().cpu().numpy());q.append(qq.detach().cpu().numpy())
 p,m,_,w,_=native_refresh_from_latents(lat,q,weights,model.n_clusters,seed)
 return torch.from_numpy(np.asarray(p)).float().to(device),torch.from_numpy(np.asarray(m)).float().to(device),tuple(float(x) for x in w)

def build_initialization(*,dataset,training_seed,feature_path,output_dir,device):
 p2.validate_execution_contract(); out=Path(output_dir); spec=item(dataset)
 if out.exists(): raise RuntimeError('FORMAL_INITIALIZATION_OUTPUT_ALREADY_EXISTS')
 config=get_native_config(dataset)
 if config['training']['seed']!=spec.native_config_seed: raise RuntimeError('FORMAL_NATIVE_CONFIG_SEED_MISMATCH')
 with np.load(feature_path,allow_pickle=False) as a:
  views_np=tuple(np.ascontiguousarray(a['view_'+str(i+1)],dtype=np.float32) for i in range(spec.n_views)); ids=np.ascontiguousarray(a['sample_ids'],dtype=np.int64)
 if tuple(x.shape for x in views_np)!=tuple((spec.n_samples,d) for d in spec.view_dims): raise RuntimeError('FORMAL_INPUT_CONTRACT_MISMATCH')
 entry._configure_determinism(training_seed,device); dev=torch.device(device)
 model=MultiViewBackbone(config,spec.n_views,spec.view_dims,spec.n_clusters,seed=spec.native_config_seed).to_device(dev)
 for ae in model.autoencoders: ae.train()
 views=tuple(torch.from_numpy(x).to(dev) for x in views_np); gen=torch.Generator(device='cpu');gen.manual_seed(training_seed)
 opts=tuple(torch.optim.Adam(ae.parameters(),lr=config['training']['lr']) for ae in model.autoencoders); ds=torch.utils.data.TensorDataset(*views)
 for _ in range(config['training']['init_epoch']):
  for b in torch.utils.data.DataLoader(ds,batch_size=config['training']['batch_size'],shuffle=True,generator=gen):
   for i,ae in enumerate(model.autoencoders):
    z=ae.encoder(b[i]);loss=F.mse_loss(ae.decoder(z),b[i]);opts[i].zero_grad(set_to_none=True);loss.backward();opts[i].step()
 with torch.no_grad():
  for ae,v in zip(model.autoencoders,views):
   _,c=initialize_kmeans_centers(ae.encoder(v).detach().cpu().numpy(),spec.n_clusters,training_seed);ae._cluster_layer.data=torch.as_tensor(c,dtype=ae._cluster_layer.dtype,device=dev)
 weights=tuple(1.0 for _ in views); retained=None; refresh=[]
 for epoch in range(config['training']['epoch']+1):
  if epoch%config['training']['T_2']==0:
   p,m,weights=_refresh(model,views,weights,training_seed,dev);refresh.append(epoch)
   if epoch==config['training']['epoch']: retained=(p.detach().cpu(),m.detach().cpu(),weights)
  for b in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*views,p),batch_size=config['training']['batch_size'],shuffle=True,generator=gen):
   for i,ae in enumerate(model.autoencoders):
    r,_,q=ae(b[i]);loss,_,_=native_objective(r,b[i],q,(b[-1]@m[i]).detach(),config['training']['lambda1']);opts[i].zero_grad(set_to_none=True);loss.backward();opts[i].step()
 with torch.no_grad(): q=np.stack([ae.clustering(ae.encoder(v)).detach().cpu().numpy() for ae,v in zip(model.autoencoders,views)],axis=1).astype(np.float32)
 p,m,w=retained; mm=np.asarray(m); aligned=np.stack([q[:,i]@mm[i].T for i in range(spec.n_views)],axis=1).astype(np.float32)
 out.mkdir(parents=True); cps=[]
 for i,state in enumerate(model.state_dicts()):
  cp=out/('view_'+str(i+1)+'_checkpoint.pth');torch.save(state,cp);cps.append(cp)
 carrier=out/'carrier_state.npz';np.savez(carrier,sample_ids=ids,M_v=mm,P_global=np.asarray(p),view_weights=np.asarray(w),q_local=q,q_aligned=aligned)
 h=entry._model_sha256(model); hashes=[sha(x) for x in cps]
 audit={'dataset':dataset,'training_seed':training_seed,'native_config_seed':spec.native_config_seed,'generator_create_count':1,'generator_seed':training_seed,'generator_reset_count':0,'shared_across_ae_native':True,'ae_epochs_completed':config['training']['init_epoch'],'native_iterations_completed':config['training']['epoch']+1,'refresh_epochs':refresh,'last_refresh_epoch':1000,'retained_M_v_logical_sha256':ndarray_sha256(mm),'post_final_q_local_logical_sha256':ndarray_sha256(q),'post_final_q_aligned_logical_sha256':ndarray_sha256(aligned),'checkpoint_sha256':hashes,'combined_model_sha256':h,'full_gt_loaded':False,'sparse_labels_used':False,'U_cycle_used':False,'relation_used':False,'metrics_computed':False}
 write(out/'initialization_audit.json',audit);manifest={**audit,'checkpoint_paths':[str(x) for x in cps]};write(out/'initialization_manifest.json',manifest);write(out/'initialization_seal.json',{'seal_valid':True,'checkpoint_sha256':hashes,'combined_model_sha256':h,'initialization_audit_sha256':sha(out/'initialization_audit.json'),'initialization_manifest_sha256':sha(out/'initialization_manifest.json')})
 return {'root':out,'audit':out/'initialization_audit.json','manifest':out/'initialization_manifest.json','seal':out/'initialization_seal.json','checkpoint_paths':tuple(cps),'checkpoint_sha256':tuple(hashes),'initial_model_sha256':h}

def build_action(*,dataset,training_seed,carrier_state_path,split_path,output_dir,initial_model_sha256=None):
 out=Path(output_dir)
 if initial_model_sha256 is None:
  manifest=Path(carrier_state_path).parent / "initialization_manifest.json"
  if not manifest.is_file(): raise RuntimeError("FORMAL_INITIALIZATION_INCOMPLETE")
  initial_model_sha256=json.loads(manifest.read_text(encoding="utf8"))["combined_model_sha256"]
 if out.exists(): raise RuntimeError('FORMAL_ACTION_OUTPUT_ALREADY_EXISTS')
 with np.load(carrier_state_path,allow_pickle=False) as c, np.load(split_path,allow_pickle=False) as s:
  qa=np.ascontiguousarray(c['q_aligned']);ids=np.ascontiguousarray(s['sample_ids'],dtype=np.int64);sp=SparseLabelSplit(ids,s['labeled_ids'],s['labeled_targets'],s['unlabeled_ids'],qa.shape[-1],p0.SPARSE_LABEL_PROTOCOL["labels_per_class"],p0.SPARSE_LABEL_PROTOCOL["label_seed"],dataset);mv=np.ascontiguousarray(c['M_v'])
 act=build_directional_actions(qa.shape[1]);cycle=compute_directional_cycle_utility(torch.from_numpy(qa),act);sem=build_relation_semantics(cycle['y_gen'].cpu().numpy(),sp,act)
 out.mkdir(parents=True);npz=out/'true_action_state.npz';np.savez(npz,sample_ids=ids,labeled_ids=sp.labeled_ids,unlabeled_ids=sp.unlabeled_ids,U_cycle=cycle['U_cycle'].cpu().numpy(),y_gen=cycle['y_gen'].cpu().numpy(),PredRelation_true=sem.pred_relation,relation_balance_weights_true=sem.balance_weights)
 audit={"dataset":dataset,"training_seed":training_seed,"initial_model_sha256":initial_model_sha256,"M_v_logical_sha256":ndarray_sha256(mv),"q_aligned_logical_sha256":ndarray_sha256(qa),"U_cycle_logical_sha256":ndarray_sha256(cycle["U_cycle"].cpu().numpy()),"y_gen_logical_sha256":ndarray_sha256(cycle["y_gen"].cpu().numpy()),"PredRelation_logical_sha256":ndarray_sha256(sem.pred_relation),"balance_logical_sha256":ndarray_sha256(sem.balance_weights),"detached_frozen":True,"full_gt_loaded":False}
 write(out / "action_audit.json",audit)
 write(out / "action_seal.json",{"seal_valid":True,"artifact_sha256":sha(npz),"action_audit_sha256":sha(out / "action_audit.json")})
 return {"root":out,"artifact":npz,"audit":out / "action_audit.json","seal":out / "action_seal.json"}
