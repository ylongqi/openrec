from multiprocessing import Process, Queue

class _Sampler(Process):

    def __init__(self, dataset, q, generate_batch):

        self._q = q
        self._generate_batch = generate_batch
        self._dataset = dataset
        super(_Sampler, self).__init__()

    def run(self):
        for input_npy in self._generate_batch(self._dataset):
            self._q.put(input_npy, block=True)
                
class Sampler(object):
    
    def __init__(self, dataset=None, generate_batch=None, num_process=5):
        
        assert generate_batch is not None, "Batch generation function is not specified"
        assert dataset is not None, "Dataset is not specified"
        self._q = Queue(maxsize=num_process)
        self._dataset = dataset
        self._runner_list = []
        self.name = self._dataset.name

        for i in range(num_process):
            runner = _Sampler(dataset, self._q, generate_batch)
            runner.daemon = True
            runner.start()
            self._runner_list.append(runner)

    def next_batch(self):

        return self._q.get(block=True)