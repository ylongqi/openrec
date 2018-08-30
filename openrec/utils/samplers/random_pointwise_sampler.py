import numpy as np
import random
from openrec.utils.samplers import Sampler

def RandomPointwiseSampler(dataset, batch_size, num_process=5, seed=100):
    
    random.seed(seed)
    def batch(dataset, batch_size=batch_size):
        
        while True:
            input_npy = np.zeros(batch_size, dtype=[('user_id', np.int32),
                                                        ('item_id', np.int32),
                                                        ('label', np.float32)])
            
            for ind in range(batch_size):
                user_id = random.randint(0, dataset.total_users()-1)
                item_id = random.randint(0, dataset.total_items()-1)
                label = 1.0 if dataset.is_positive(user_id, item_id) else 0.0
                input_npy[ind] = (user_id, item_id, label)
            yield input_npy
    
    s = Sampler(dataset=dataset, generate_batch=batch, num_process=num_process)
    
    return s