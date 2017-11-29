import numpy as np
from openrec.utils.evaluators import Evaluator

class Recall(Evaluator):

    def __init__(self, recall_at, name='Recall'):
        
        self._recall_at = np.array(recall_at)

        super(Recall, self).__init__(etype='rank', name=name)

    def compute(self, rank_above, negative_num):

        del negative_num
        results = np.zeros(len(self._recall_at))
        for rank in rank_above:
            results += (rank <= self._recall_at).astype(np.float32)

        return results / len(rank_above)
