import numpy as np
import random
from openrec.utils.samplers import Sampler

def RandomPointwiseSampler(dataset, batch_size, num_process=5, seed=100):
    
    random.seed(seed)
    def batch(dataset, batch_size=batch_size, seed=seed):
        
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
    
    
# class _PointwiseSampler(Process):

#     def __init__(self, dataset, batch_size, pos_ratio, q, chronological=False):
        
#         self._dataset = dataset
#         self._batch_size = batch_size
#         self._num_pos = int(batch_size * pos_ratio)

#         self._user_list = self._dataset.get_unique_user_list()
#         self._q = q
#         self._state = 0
#         self._chronological = chronological

#         if not chronological:
#             self._dataset.shuffle()
#         super(_PointwiseSampler, self).__init__()


#     def run(self):
#         while True:
#             input_npy = np.zeros(self._batch_size, dtype=[('user_id', np.int32),
#                                                         ('item_id', np.int32),
#                                                         ('labels', np.float32)])
            
#             if self._state + self._num_pos >= len(self._dataset.data):
#                 if not self._chronological:
#                     self._state = 0
#                     self._dataset.shuffle()
#                 else:
#                     break

#             for ind in range(self._num_pos):
#                 entry = self._dataset.data[self._state + ind]
#                 input_npy[ind] = (entry['user_id'], entry['item_id'], 1.0)

#             for ind in range(self._batch_size - self._num_pos):
#                 user_ind = min(int(random.random() * len(self._user_list)), len(self._user_list) - 1)
#                 user_id = self._user_list[user_ind]
#                 neg_id = min(int(random.random() * self._dataset.max_item()), self._dataset.max_item() - 1)

#                 while neg_id in self._dataset.get_interactions_by_user_gb_item(user_id):
#                     neg_id = min(int(random.random() * self._dataset.max_item()), self._dataset.max_item() - 1)
#                 input_npy[ind + self._num_pos] = (user_id, neg_id, 0.0)
                
#             self._state += self._num_pos
#             self._q.put(input_npy, block=True)


# class PointwiseSampler(Sampler):

#     def __init__(self, dataset, batch_size, pos_ratio=0.5, num_process=5, chronological=False, seed=0):
        
#         self._pos_ratio = pos_ratio
#         self._chronological = chronological
        
#         if chronological:
#             num_process = 1
#         random.seed(seed)
#         super(PointwiseSampler, self).__init__(dataset=dataset, batch_size=batch_size, num_process=num_process)
        
#     def _get_runner(self):
        
#         return _PointwiseSampler(dataset=self._dataset,
#                                pos_ratio=self._pos_ratio,
#                                batch_size=self._batch_size,
#                                q=self._q, chronological=self._chronological)
