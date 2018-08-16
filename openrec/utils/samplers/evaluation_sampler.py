import numpy as np
import random
from openrec.utils.samplers import Sampler

def EvaluationSampler(dataset, seed=100):
    
    random.seed(seed)
    def batch(dataset):
        while True:
            for user_id in dataset.warm_users():
                positive_items = dataset.get_positive_items(user_id)
                negative_items = dataset.get_negative_items(user_id)
                
                input_npy = np.zeros(len(positive_items)+len(negative_items), 
                                     dtype=[('user_id', np.int32),
                                            ('item_id', np.int32)])
                
                for ind, item_id in enumerate(positive_items+negative_items):
                    input_npy[ind] = (user_id, item_id)
                
                yield range(len(positive_items)), input_npy
            yield None, None
    
    s = Sampler(dataset=dataset, generate_batch=batch, num_process=1)
    return s
    