from __future__ import print_function
import numpy as np
import random
from openrec.utils.samplers import Sampler

def VBPRPairwiseSampler(dataset, batch_size, item_vfeature, num_process=5, seed=100):
    
    random.seed(seed)
    def batch(dataset, batch_size=batch_size, 
              item_vfeature=item_vfeature, seed=seed):
        
        _, dim_v = item_vfeature.shape
        while True:
            
            input_npy = np.zeros(batch_size, dtype=[('user_id', np.int32),
                                                    ('p_item_id', np.int32),
                                                    ('n_item_id', np.int32), 
                                                    ('p_item_vfeature', np.float32, (dim_v)), 
                                                    ('n_item_vfeature', np.float32, (dim_v))])
            
            for ind in range(batch_size):
                entry = dataset.next_random_record()
                user_id = entry['user_id']
                p_item_id = entry['item_id']
                n_item_id = dataset.sample_negative_items(user_id)[0]
                input_npy[ind] = (user_id, p_item_id, n_item_id, 
                                item_vfeature[p_item_id], item_vfeature[n_item_id])
            yield input_npy
    
    s = Sampler(dataset=dataset, generate_batch=batch, num_process=num_process)
    
    return s