from __future__ import print_function
import numpy as np
import random
from multiprocessing import Process, Queue

class _PairwiseSampler(Process):

    def __init__(self, dataset, batch_size, q):
        self._dataset = dataset
        self._dataset.shuffle()
        self._batch_size = batch_size
        self._q = q
        self._state = 0
        super(_PairwiseSampler, self).__init__()

    def run(self):
        while True:
            # inputType = self._train_data.data.dtype.descr
            # if not 'interaction' in self._train_data.data.dtype.names:
            #     inputType.append(('interaction','<f2'))
            # input_npy = np.zeros(self._BATCH_SIZE*2, dtype=inputType)
            input_npy = np.zeros(self._batch_size, dtype=[('user_id_input', np.int32),
                                                        ('p_item_id_input', np.int32),
                                                        ('n_item_id_input', np.int32)])

            if self._state + self._batch_size >= len(self._dataset.data):
                self._state = 0
                self._dataset.shuffle()

            for sample_itr, entry in enumerate(self._dataset.data[self._state:(self._state + self._batch_size)]):
                neg_id = int(random.random() * (self._dataset.max_item - 1))
                while neg_id in self._dataset.gy_user_item[entry['user_id']]:
                    neg_id = int(random.random() * (self._dataset.max_item - 1))
                input_npy[sample_itr] = (entry['user_id'], entry['item_id'], neg_id)
                # if not 'interaction' in self._train_data.data.dtype.names:
                #     input_npy[sample_itr] = append_fields(interaction,'interaction',[1])
                # else:
                #     input_npy[sample_itr] = interaction
                #     input_npy[sample_itr]['interaction'] = 1

                # sample_itr += 1

                # input_npy[sample_itr] = np.zeros(1,dtype=inputType)
                # input_npy[sample_itr]['user_id'] = interaction['user_id']
                # input_npy[sample_itr]['item_id'] = neg_id
                # input_npy[sample_itr]['interaction'] = 0

            self._state += self._batch_size
            self._q.put(input_npy, block=True)


class PairwiseSampler(object):

    def __init__(self, dataset, batch_size, chronological=False, num_process=5):

        # assert train_data is not None, 'train_data cannot be None'
        # assert batch_size is not None, 'batch_size cannot be None'

        self._q = Queue(maxsize=5)
        self._runner_list = []

        if chronological:
            num_process = 1
            # dataset.sort(order=['date'])

        for i in range(num_process):
            runner = _PairwiseSampler(dataset=dataset,
                                      batch_size=batch_size,
                                      q=self._q)
            runner.daemon = True
            runner.start()
            self._runner_list.append(runner)

    def next_batch(self):
        # model.push_batch(self._q.get(block=True))
        return self._q.get(block=True)
