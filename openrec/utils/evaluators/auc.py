import numpy as np
from openrec.utils.evaluators import Evaluator
from sklearn.metrics import roc_auc_score

class AUC(Evaluator):

    def __init__(self, name='AUC'):
        
        super(AUC, self).__init__(etype='rank', name=name)
    
    def compute(self, pos_mask, pred, excl_mask=None):
        
        num_users = pred.shape[0]
        auc = []
        
        for user_i in range(num_users):
            if excl_mask is not None:
                eval_mask = np.logical_not(np.logical_or(pos_mask[user_i], excl_mask[user_i]))
            else:
                eval_mask = np.logical_not(pos_mask[user_i])
            eval_pred = pred[user_i][eval_mask]
            eval_num = np.sum(eval_mask)
            
            pos_pred = pred[user_i][pos_mask[user_i]]
            user_auc = np.sum(eval_pred <= pos_pred.reshape((-1, 1))) \
                        / (len(pos_pred) * eval_num)
            
            auc.append(user_auc)
            
        return np.array(auc).reshape((num_users, -1))