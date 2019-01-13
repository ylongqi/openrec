from multiprocessing import Process, Queue
import random

class _Sampler(Process):

    def __init__(self, q, generate_batch, seed):

        self._q = q
        self._generate_batch = generate_batch
        self._seed = seed
        super(_Sampler, self).__init__()

    def run(self):
        
        random.seed(self._seed)
        while True:
            for batch_data in self._generate_batch():
                self._q.put(batch_data, block=True)
            self._q.put(None, block=True)
                
class Sampler(object):
    
    def __init__(self, generate_batch=None, num_process=5, seed=100, name='Sampler'):
        
        assert generate_batch is not None, "Batch generation function is not specified"
        self._q = None
        self._runner_list = []
        self._start = False
        self._num_process = num_process
        self._generate_batch = generate_batch
        self._seed = seed
        self.name = name
        
    def next_batch(self):
        
        if not self._start:
            self.reset()
        
        return self._q.get(block=True)
        
    def reset(self):
        
        while len(self._runner_list) > 0:
            runner = self._runner_list.pop()
            runner.terminate()
            del runner
        
        if self._q is not None:
            del self._q
        self._q = Queue(maxsize=self._num_process)
            
        for i in range(self._num_process):
            runner = self._start_a_new_runner(seed=self._seed+i)
            
        self._start = True
            
    def _start_a_new_runner(self, seed):
        
        runner = _Sampler(self._q, self._generate_batch, seed)
        runner.daemon = True
        runner.start()
        self._runner_list.append(runner)