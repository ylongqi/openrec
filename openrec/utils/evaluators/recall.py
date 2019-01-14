from openrec.utils.evaluators import Evaluator
import numpy as np

class Recall(Evaluator):

    def __init__(self, recall_at, name='Recall'):
        
        self._recall_at = np.array(recall_at)

        super(Recall, self).__init__(etype='rank', name=name)
    
    def compute(self, pos_mask, pred, excl_mask=None):
        
        num_users = pred.shape[0]
        if excl_mask is not None:
            pred[excl_mask] = -np.inf
        recall = []
        
        idx = np.argpartition(-pred, self._recall_at, axis=1)
        for k in self._recall_at:
            pred_binary = np.zeros_like(pred, dtype=bool)
            pred_binary[np.arange(num_users)[:, np.newaxis], idx[:, :k]] = True
            
            tmp = (np.logical_and(pos_mask, pred_binary).sum(axis=1)).astype(np.float32)
            recall.append((tmp / np.minimum(k, pos_mask.sum(axis=1))).reshape((num_users, -1)))
            
        return np.concatenate(recall, axis=1)
