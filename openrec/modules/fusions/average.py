import tensorflow as tf
from openrec.modules.fusions import Fusion

class Average(Fusion):

    """
    The Average module outputs the element-wise average of the outputs from multiple modules.

    Parameters
    ----------
    module_list: list
        The list of modules.
    weight: float 
        A value elementwise multiplied to module outputs.
    scope: str, optional
        Scope for module variables.
    reuse: bool, optional
        Whether or not to reuse module variables.
    """
    
    def __init__(self, module_list, weight=1.0, scope=None, reuse=False):
        
        self._module_list = module_list
        self._weight = weight

        super(Average, self).__init__(scope=scope, reuse=reuse)

    def _build_shared_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):

            outputs = sum([module.get_outputs() for module in self._module_list], [])
            self._outputs.append(self._weight * tf.add_n(outputs) /  len(self._module_list))
            self._loss = 0.0
