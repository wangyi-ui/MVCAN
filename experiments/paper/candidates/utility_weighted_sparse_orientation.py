"""Pure paper-local D7-C1 utility-weighted sparse-orientation candidate."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from release_core.semantics import validate_sparse_label_split,build_relation_balance_weights
from release_core.utility.action_space import ActionSpace
def _req(x,m):
 if not x:raise RuntimeError(m)
def build_utility_weighted_vote_state(y_gen,u_cycle,class_count):
 y=np.asarray(y_gen);u=np.asarray(u_cycle,dtype=np.float64);_req(y.ndim==2 and u.shape==y.shape and np.issubdtype(y.dtype,np.integer),'D7_C1_VOTE_SHAPE');_req(np.all((y>=0)&(y<class_count))and np.isfinite(u).all()and np.all(u>=0),'D7_C1_VOTE_VALUES');mass=u.sum(1);out=np.zeros((y.shape[0],class_count),float)
 for k in range(class_count):out[:,k]=(u*(y==k)).sum(1)
 good=mass>0;out[good]/=mass[good,None];return out
def _mapping(vote,ids,targets,k):
 cont=np.zeros((k,k),float)
 for c in range(k):cont[:,c]=vote[np.asarray(ids)[np.asarray(targets)==c]].sum(0)
 r,col=linear_sum_assignment(-cont);m=np.empty(k,dtype=np.int64);m[r]=col;return m,cont
def fit_utility_weighted_sparse_mapping(y_gen,u_cycle,split):
 split=validate_sparse_label_split(split);vote=build_utility_weighted_vote_state(y_gen,u_cycle,split.class_count);lookup={int(x):i for i,x in enumerate(split.sample_ids)};rows=np.asarray([lookup[int(x)]for x in split.labeled_ids]);_req(np.all(vote[rows].sum(1)>0),'D7_C1_ZERO_LABELED_UTILITY');return vote,_mapping(vote,rows,split.labeled_targets,split.class_count)
def build_utility_weighted_relation_semantics(y_gen,u_cycle,split,actions=None):
 split=validate_sparse_label_split(split);y=np.asarray(y_gen);_req(actions is None or isinstance(actions,ActionSpace)and actions.S==y.shape[1],'D7_C1_ACTIONS');vote,(mapping,cont)=fit_utility_weighted_sparse_mapping(y,u_cycle,split);lookup={int(x):i for i,x in enumerate(split.sample_ids)};rows=np.asarray([lookup[int(x)]for x in split.unlabeled_ids]);pred=mapping[y[rows]];rel=np.ascontiguousarray(pred[:,None,:]==split.labeled_targets[None,:,None],dtype=bool);balance=build_relation_balance_weights(rel);return {'vote_state':vote,'mapping':mapping,'contingency':cont,'class_pred':pred,'pred_relation':rel,'balance_weights':balance,'labeled_rows':np.asarray([lookup[int(x)]for x in split.labeled_ids]),'unlabeled_rows':rows}
