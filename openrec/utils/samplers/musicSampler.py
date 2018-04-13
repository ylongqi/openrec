from __future__ import print_function
import numpy as np
import random
from multiprocessing import Process
from openrec.utils.samplers import Sampler

class _GeneralSampler(Process):

    def __init__(self, dataset, batch_size, q, genre_f):
        self._dataset = dataset
        self._dataset.shuffle()
        self._batch_size = batch_size
        self._q = q
        self._state = 0
        self._genre_f = genre_f
        super(_GeneralSampler, self).__init__()

    def run(self):
        while True:
            
            input_npy = np.zeros(self._batch_size, dtype=[('song_id', np.int32),
                                                        ('artist', np.int32),
                                                        ('genre_input', np.int32),
                                                        ('language', np.int32),
                                                        ('lyricist', np.int32),
                                                        ('composer', np.int32),
                                                        ('source', np.int32)])

            if self._state + self._batch_size >= len(self._dataset.data):
                self._state = 0
                self._dataset.shuffle()

            for sample_itr, entry in enumerate(self._dataset.data[self._state:(self._state + self._batch_size)]):
                input_npy[sample_itr] = (entry['song_id'], entry['artist'], entry['genre_input'], entry['language'], entry['lyricist'], entry['composer'], entry['source'])
            self._state += self._batch_size
            self._q.put(input_npy, block=True)


class GeneralSampler(Sampler):

    def __init__(self, dataset, batch_size, genre_f, chronological=False, num_process=5):
        
        self._chronological = chronological
        if chronological:
            num_process = 1
        
        super(GeneralSampler, self).__init__(dataset=dataset, batch_size=batch_size, num_process=num_process)

    def _get_runner(self):
        
        return _GeneralSampler(dataset=self._dataset,
                               batch_size=self._batch_size,
                               q=self._q,
                               genre_f=self._genre_f)
