import numpy as np
import random
from openrec.utils.samplers import Sampler
import math

def VBPREvaluationSampler(batch_size, dataset, item_vfeature, seed=100):
    
    random.seed(seed)
    def batch(dataset, batch_size=batch_size, item_vfeature=item_vfeature):
        _, dim_v = item_vfeature.shape
        while True:
            for user_id in dataset.warm_users():
                positive_items = dataset.get_positive_items(user_id)
                negative_items = dataset.get_negative_items(user_id)
                all_items = positive_items + negative_items
                
                for batch_ind in range(int(math.ceil(float(len(all_items)) / batch_size))):
                    current_batch_size = min(len(all_items)-batch_ind*batch_size, batch_size)
                    input_npy = np.zeros(current_batch_size, dtype=[('user_id', np.int32),
                                                            ('item_id', np.int32),
                                                            ('item_vfeature', np.float32, (dim_v))])
                    for inst_ind in range(current_batch_size):
                        item_id = all_items[batch_ind*batch_size+inst_ind]
                        input_npy[inst_ind] = (user_id, item_id, item_vfeature[item_id])
                    num_positives = len(positive_items) - batch_ind*batch_size
                    if num_positives > 0:
                        yield range(num_positives), input_npy
                    else:
                        yield [], input_npy
                
                yield [], []
            yield None, None
    
    s = Sampler(dataset=dataset, generate_batch=batch, num_process=1)
    return s
    