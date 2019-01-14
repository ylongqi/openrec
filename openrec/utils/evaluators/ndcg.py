import numpy as np
from math import log
from openrec.utils.evaluators import Evaluator

class NDCG(Evaluator):

    def __init__(self, ndcg_at, name='NDCG'):
        
        self._ndcg_at = np.array(ndcg_at)

        super(NDCG, self).__init__(etype='rank', name=name)
    
    def compute(self, pos_mask, pred, excl_mask=None):
        
        num_users = pred.shape[0]
        if excl_mask is not None:
            pred[excl_mask] = -np.inf
        ndcg = []
        
        for user_i in range(num_users):
            pos_pred = pred[user_i][pos_mask[user_i]]
            rank_above = np.sum(pred[user_i] > pos_pred.reshape((-1, 1)), axis=1) 
            
            user_ndcg = []
            for k in self._ndcg_at:
                r = rank_above[rank_above < k]
                user_ndcg.append(np.sum(1.0 / np.log2(r+2)))
            ndcg.append(user_ndcg)
            
        return np.array(ndcg).reshape((num_users, -1))

