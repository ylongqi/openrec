from multiprocessing import Process, Queue

class _Sampler(Process):

    def __init__(self, dataset, q, generate_batch):

        self._q = q
        self._generate_batch = generate_batch
        self._dataset = dataset
        super(_Sampler, self).__init__()

    def run(self):
        while True:
            for batch_data in self._generate_batch(self._dataset):
                self._q.put(batch_data, block=True)
            self._q.put(None, block=True)
                
class Sampler(object):
    
    def __init__(self, dataset=None, generate_batch=None, num_process=5):
        
        assert generate_batch is not None, "Batch generation function is not specified"
        assert dataset is not None, "Dataset is not specified"
        self._q = None
        self._dataset = dataset
        self._runner_list = []
        self._start = False
        self._num_process = num_process
        self._generate_batch = generate_batch
        self.name = self._dataset.name
        
    def next_batch(self):
        
        if not self._start:
            self.reset()
        # self._auto_restart_dead_runners()
        
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
            runner = self._start_a_new_runner()
            
        self._start = True
    
    def _auto_restart_dead_runners(self):
        
        dead_runners = []
        for runner in self._runner_list:
            if runner.is_finished():
                dead_runners.append(runner)
        
        for runner in dead_runners:
            runner.terminate()
            self._runner_list.remove(runner)
            runner = self._start_a_new_runner()
            print('restart...')
            
    def _start_a_new_runner(self):
        
        runner = _Sampler(self._dataset, self._q, self._generate_batch)
        runner.daemon = True
        runner.start()
        self._runner_list.append(runner)