import numpy as np
import math
from openrec.utils.samplers import Sampler

def EvaluationSampler(batch_size, dataset, excl_datasets=[], seed=100):
    
    def batch(batch_size=batch_size, dataset=dataset, excl_datasets=excl_datasets):
        
        eval_users = dataset.warm_users()
        
        for st_idx in range(0, len(eval_users), batch_size):
            current_batch_size = min(batch_size, len(eval_users) - st_idx)
            input_npy = np.zeros(current_batch_size, dtype=[('user_id', np.int32)])
            pos_mask_npy = np.zeros((current_batch_size, dataset.total_items()), 
                                    dtype=np.bool)
            
            if dataset.contain_negatives():
                # By default to exclude all.
                excl_mask_npy = np.ones((current_batch_size, dataset.total_items()),
                                       dtype=np.bool)
            else:
                # By default to exclude none.
                excl_mask_npy = np.zeros((current_batch_size, dataset.total_items()),
                                        dtype=np.bool)
                
            for idx in range(current_batch_size):
                user_id = eval_users[st_idx + idx]
                input_npy[idx]['user_id'] = user_id
                
                positive_items = dataset.get_positive_items(user_id)
                pos_mask_npy[idx][positive_items] = True
                
                if dataset.contain_negatives():
                    excl_mask_npy[idx][positive_items] = False
                    negative_items = dataset.get_negative_items(user_id)
                    excl_mask_npy[idx][negative_items] = False
                    
                excl_positive_items = []
                for excl_d in excl_datasets:
                    excl_positive_items += excl_d.get_positive_items(user_id)
                excl_mask_npy[idx][excl_positive_items] = True
            
            yield {'progress': st_idx+current_batch_size, 
                   'pos_mask': pos_mask_npy,
                   'excl_mask': excl_mask_npy,
                   'input': input_npy}
    
    s = Sampler(generate_batch=batch, num_process=1, seed=seed, name=dataset.name)
    
    return s
    