from multiprocessing import Process, Queue

class Sampler(object):
    
    def __init__(self, dataset, batch_size, num_process=5):
        
        self._dataset = dataset
        self._batch_size = batch_size
        self._q = Queue(maxsize=5)
        self._runner_list = []

        for i in range(num_process):
            runner = self._get_runner()
            runner.daemon = True
            runner.start()
            self._runner_list.append(runner)
    
    def _get_runner(self):
        
        return None

    def next_batch(self):

        return self._q.get(block=True)
