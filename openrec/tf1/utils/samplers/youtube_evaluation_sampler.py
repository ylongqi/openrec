import numpy as np
import random
from openrec.tf1.utils.samplers import Sampler

def YouTubeEvaluationSampler(dataset, max_seq_len, user_feature, seed=100, sort=True):
    
    random.seed(seed)
    def batch(dataset, user_feature=user_feature, max_seq_len=max_seq_len):
        
        while True:
            for user_id in dataset.warm_users():
                input_npy = np.zeros(1, dtype=[('seq_item_id', (np.int32,  max_seq_len)),
                                               ('seq_len', np.int32),
                                               ('user_gender', np.int32),
                                               ('user_geo', np.int32)])
                
                item_list = dataset.get_positive_items(user_id, sort=sort)
                if len(item_list) <= 1:
                    continue
                train_items = item_list[-max_seq_len-1:-1]
                pad_train_items = np.zeros(max_seq_len, np.int32)
                pad_train_items[:len(train_items)] = train_items
                input_npy[0] = (pad_train_items, 
                                len(train_items), 
                                user_feature[user_id]['user_gender'],
                                user_feature[user_id]['user_geo'])
                yield [train_items[-1]], input_npy
                yield [], []
            yield None, None
            
    s = Sampler(dataset=dataset, generate_batch=batch, num_process=1)
    
    return s
