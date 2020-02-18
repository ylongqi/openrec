import numpy as np
import random
from openrec.tf1.utils.samplers import Sampler

def YouTubeSampler(dataset, batch_size, max_seq_len, user_feature, num_process=5, seed=100, sort=True):
    
    random.seed(seed)
    def batch(dataset, user_feature=user_feature, max_seq_len=max_seq_len, batch_size=batch_size):
        
        while True:
            input_npy = np.zeros(batch_size, dtype=[('seq_item_id', (np.int32,  max_seq_len)),
                                                   ('seq_len', np.int32),
                                                   ('label', np.int32),
                                                   ('user_gender', np.int32),
                                                   ('user_geo', np.int32)])
            
            for ind in range(batch_size):
                user_id = random.randint(0, dataset.total_users()-1)
                item_list = dataset.get_positive_items(user_id, sort=sort)
                while len(item_list) <= 1:
                    user_id = random.randint(0, dataset.total_users()-1)
                    item_list = dataset.get_positive_items(user_id, sort=sort)
                predict_pos = random.randint(1, len(item_list) - 1)
                train_items = item_list[max(0, predict_pos-max_seq_len):predict_pos]
                pad_train_items = np.zeros(max_seq_len, np.int32)
                pad_train_items[:len(train_items)] = train_items
                input_npy[ind] = (pad_train_items, 
                                  len(train_items), 
                                  item_list[predict_pos],
                                  user_feature[user_id]['user_gender'], 
                                  user_feature[user_id]['user_geo'])
            yield input_npy
    
    s = Sampler(dataset=dataset, generate_batch=batch, num_process=num_process)
    
    return s
