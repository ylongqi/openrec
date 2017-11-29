import numpy as np
from openrec.utils.evaluators import Evaluator

class AUC(Evaluator):

    def __init__(self, name='AUC'):
        
        super(AUC, self).__init__(etype='rank', name=name)

    def compute(self, rank_above, negative_num):

        return np.mean((negative_num - rank_above) / negative_num)