import numpy as np
from multiprocessing import Process
from openrec.utils.samplers import Sampler

class _ExplicitSampler(Process):

    def __init__(self, dataset, batch_size, q, shuffle=True, loop=True):
        
        self._dataset = dataset
        self._batch_size = batch_size
        
        self._q = q
        self._shuffle = shuffle
        self._loop = loop
        self._state = 0
        
        super(_ExplicitSampler, self).__init__()


    def run(self):
        
        if self._shuffle:
            self._dataset.shuffle()
            
        while True:
            
            input_npy = np.zeros(self._batch_size, dtype=[('user_id_input', np.int32),
                                                        ('item_id_input', np.int32),
                                                        ('labels', np.float32)])

            if self._state + self._batch_size >= len(self._dataset.data):
                break

            for ind in range(self._batch_size):
                entry = self._dataset.data[self._state + ind]
                input_npy[ind] = (entry['user_id'], entry['item_id'], entry['label'])
                
            self._state += self._batch_size
            self._q.put(input_npy, block=True)

class ExplicitSampler(Sampler):

    def __init__(self, dataset, batch_size, num_process=5, chronological=False):

        self._chronological = chronological

        if self._chronological:
            num_process = 1
        
        super(ExplicitSampler, self).__init__(dataset=dataset, batch_size=batch_size, num_process=num_process)

    def _get_runner(self):
        
        return _ExplicitSampler(dataset=self._dataset,
                                batch_size=self._batch_size,
                               q=self._q,
                               shuffle=not self._chronological,
                               loop=not self._chronological)
    
