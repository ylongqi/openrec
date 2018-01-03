from __future__ import print_function
import numpy as np
import random
from multiprocessing import Process, Queue

class _PointwiseSampler(Process):

    def __init__(self, dataset, batch_size, pos_ratio, q):
        
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_pos = int(batch_size * pos_ratio)

        self._user_list = self._dataset.get_unique_user_list()
        self._q = q
        self._state = 0
        super(_PointwiseSampler, self).__init__()


    def run(self):
        while True:
            
            input_npy = np.zeros(self._batch_size, dtype=[('user_id_input', np.int32),
                                                        ('item_id_input', np.int32),
                                                        ('labels', np.float32)])

            if self._state + self._num_pos >= len(self._dataset.data):
                self._state = 0
                self._dataset.shuffle()

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


class PointwiseSampler(object):

    def __init__(self, dataset, batch_size, pos_ratio=0.5, num_process=5, chronological=False):

        self._q = Queue(maxsize=5)
        self._runner_list = []

        if chronological:
            num_process = 1

        for i in range(num_process):
            runner = _PointwiseSampler(dataset=dataset,
                                   pos_ratio=pos_ratio,
                                   batch_size=batch_size,
                                   q=self._q)
            runner.daemon = True
            runner.start()
            self._runner_list.append(runner)

    def next_batch(self):

        return self._q.get(block=True)
