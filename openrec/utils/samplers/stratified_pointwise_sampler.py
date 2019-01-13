import numpy as np
import random
from openrec.utils.samplers import Sampler

def StratifiedPointwiseSampler(batch_size, dataset, pos_ratio=0.5, num_process=5, seed=100):
    
    def batch(dataset=dataset, batch_size=batch_size, pos_ratio=pos_ratio):
        
        num_pos = int(batch_size * pos_ratio)
        while True:
            _input = {'user_id': np.zeros(batch_size, dtype=np.int32), 
                      'item_id': np.zeros(batch_size, dtype=np.int32), 
                      'label': np.zeros(batch_size, dtype=np.float32)}
            
            for ind in range(num_pos):
                entry = dataset.next_random_record()
                _input['user_id'][ind] = entry['user_id']
                _input['item_id'][ind] = entry['item_id']
                _input['label'][ind] = 1.0

            for ind in range(batch_size - num_pos):
                user_id = random.randint(0, dataset.total_users()-1)
                item_id = random.randint(0, dataset.total_items()-1)
                while dataset.is_positive(user_id, item_id):
                    user_id = random.randint(0, dataset.total_users()-1)
                    item_id = random.randint(0, dataset.total_items()-1)
                _input['user_id'][ind + num_pos] = user_id
                _input['item_id'][ind + num_pos] = item_id
                _input['label'][ind + num_pos] = 0.0
            
            yield {'input': _input}
    
    s = Sampler(generate_batch=batch, num_process=num_process, seed=seed)
    
    return s
    