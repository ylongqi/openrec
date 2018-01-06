import numpy as np
from openrec.utils.evaluators import Evaluator

class MSE(Evaluator):
    
    def __init__(self, name='MSE'):

        super(MSE, self).__init__(etype='regression', name=name)

    def compute(self, predictions, labels):

        return np.square(predictions - labels)
