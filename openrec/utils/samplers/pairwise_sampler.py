from __future__ import print_function
import numpy as np
import random
from multiprocessing import Process
from openrec.utils.samplers import Sampler

class _PairwiseSampler(Process):

    def __init__(self, dataset, batch_size, q, chronological):
        self._dataset = dataset
        self._batch_size = batch_size
        self._q = q
        self._state = 0
        self._chronological = chronological

        if not chronological:
            self._dataset.shuffle()

        super(_PairwiseSampler, self).__init__()

    def run(self):
        while True:
            
            input_npy = np.zeros(self._batch_size, dtype=[('user_id_input', np.int32),
                                                        ('p_item_id_input', np.int32),
                                                        ('n_item_id_input', np.int32)])

            if self._state + self._batch_size >= len(self._dataset.data):
                if not self._chronological:
                    self._state = 0
                    self._dataset.shuffle()
                else:
                    break

            for sample_itr, entry in enumerate(self._dataset.data[self._state:(self._state + self._batch_size)]):
                neg_id = int(random.random() * (self._dataset.max_item() - 1))
                while neg_id in self._dataset.get_interactions_by_user_gb_item(entry['user_id']):
                    neg_id = int(random.random() * (self._dataset.max_item() - 1))
                input_npy[sample_itr] = (entry['user_id'], entry['item_id'], neg_id)

            self._state += self._batch_size
            self._q.put(input_npy, block=True)


class PairwiseSampler(Sampler):

    def __init__(self, dataset, batch_size, chronological=False, num_process=5, seed=0):
        
        self._chronological = chronological
        if chronological:
            num_process = 1
        random.seed(seed) 
        super(PairwiseSampler, self).__init__(dataset=dataset, batch_size=batch_size, num_process=num_process)

    def _get_runner(self):
        
        return _PairwiseSampler(dataset=self._dataset,
                               batch_size=self._batch_size,
                               q=self._q, chronological=self._chronological)
