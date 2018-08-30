import numpy as np
import random
from openrec.utils.samplers import Sampler


def StratifiedPointwiseSampler(dataset, batch_size, pos_ratio=0.5, num_process=5, seed=100):
    
    random.seed(seed)
    def batch(dataset, batch_size=batch_size, pos_ratio=pos_ratio, seed=seed):
        
        num_pos = int(batch_size * pos_ratio)
        while True:
            input_npy = np.zeros(batch_size, dtype=[('user_id', np.int32),
                                                        ('item_id', np.int32),
                                                        ('label', np.float32)])
            
            for ind in range(num_pos):
                entry = dataset.next_random_record()
                input_npy[ind] = (entry['user_id'], entry['item_id'], 1.0)

            for ind in range(batch_size - num_pos):
                user_id = random.randint(0, dataset.total_users()-1)
                item_id = random.randint(0, dataset.total_items()-1)
                while dataset.is_positive(user_id, item_id):
                    user_id = random.randint(0, dataset.total_users()-1)
                    item_id = random.randint(0, dataset.total_items()-1)
                input_npy[ind + num_pos] = (user_id, item_id, 0.0)
            
            yield input_npy
    
    s = Sampler(dataset=dataset, generate_batch=batch, num_process=num_process)
    
    return s
    