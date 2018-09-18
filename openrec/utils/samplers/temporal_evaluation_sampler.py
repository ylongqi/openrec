import numpy as np
import random
from openrec.utils.samplers import Sampler

def TemporalEvaluationSampler(dataset, max_seq_len, seed=100):
    
    random.seed(seed)
    def batch(dataset, max_seq_len=max_seq_len):
        
        while True:
            for user_id in dataset.warm_users():
                input_npy = np.zeros(1, dtype=[('seq_item_id', (np.int32,  max_seq_len)),
                                                ('seq_len', np.int32)])
                
                item_list = dataset.get_positive_items(user_id, sort=True)
                if len(item_list) <= 1:
                    continue
                train_items = item_list[-max_seq_len-1:-1]
                pad_train_items = np.zeros(max_seq_len, np.int32)
                pad_train_items[:len(train_items)] = train_items
                input_npy[0] = (pad_train_items, len(train_items))
                yield [train_items[-1]], input_npy
                yield [], []
            yield None, None
            
    s = Sampler(dataset=dataset, generate_batch=batch, num_process=1)
    
    return s