import numpy as np
from math import log
from openrec.utils.evaluators import Evaluator

class NDCG(Evaluator):

    def __init__(self, ndcg_at, name='NDCG'):
        
        self._ndcg_at = np.array(ndcg_at)

        super(NDCG, self).__init__(etype='rank', name=name)

    def compute(self, rank_above, negative_num):

        del negative_num
        denominator = 0.0
        for i in range(len(rank_above)):
            denominator += 1.0 / log(i+2, 2)
        
        results = np.zeros(len(self._ndcg_at))
        for r in rank_above:
            tmp = 1.0 / log(r+2, 2)
            results[r < self._ndcg_at] += tmp
        
        return results / denominator

