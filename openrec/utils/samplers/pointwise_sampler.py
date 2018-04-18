import numpy as np
import random
from multiprocessing import Process
from openrec.utils.samplers import Sampler

class _PointwiseSampler(Process):

    def __init__(self, dataset, batch_size, pos_ratio, q, chronological=False):
        
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_pos = int(batch_size * pos_ratio)

        self._user_list = self._dataset.get_unique_user_list()
        self._q = q
        self._state = 0
        self._chronological = chronological

        if not chronological:
            self._dataset.shuffle()
        super(_PointwiseSampler, self).__init__()


    def run(self):
        while True:
            
            input_npy = np.zeros(self._batch_size, dtype=[('user_id_input', np.int32),
                                                        ('item_id_input', np.int32),
                                                        ('labels', np.float32)])

            if self._state + self._num_pos >= len(self._dataset.data):
                if not self._chronological:
                    self._state = 0
                    self._dataset.shuffle()
                else:
                    break

            for ind in range(self._num_pos):
                entry = self._dataset.data[self._state + ind]
                input_npy[ind] = (entry['user_id'], entry['item_id'], 1.0)

            for ind in range(self._batch_size - self._num_pos):
                user_ind = int(random.random() * (len(self._user_list) - 1))
                user_id = self._user_list[user_ind]
                neg_id = int(random.random() * (self._dataset.max_item() - 1))

                while neg_id in self._dataset.get_interactions_by_user_gb_item(user_id):
                    neg_id = int(random.random() * (self._dataset.max_item() - 1))
                input_npy[ind + self._num_pos] = (user_id, neg_id, 0.0)
                
            self._state += self._num_pos
            self._q.put(input_npy, block=True)


class PointwiseSampler(Sampler):

    def __init__(self, dataset, batch_size, pos_ratio=0.5, num_process=5, chronological=False, seed=0):
        
        self._pos_ratio = pos_ratio
        self._chronological = chronological
        
        if chronological:
            num_process = 1
        random.seed(seed)
        super(PointwiseSampler, self).__init__(dataset=dataset, batch_size=batch_size, num_process=num_process)
        
    def _get_runner(self):
        
        return _PointwiseSampler(dataset=self._dataset,
                               pos_ratio=self._pos_ratio,
                               batch_size=self._batch_size,
                               q=self._q, chronological=self._chronological)
