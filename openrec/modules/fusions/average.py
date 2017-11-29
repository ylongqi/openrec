import tensorflow as tf
from openrec.modules.fusions import Fusion

class Average(Fusion):

    def __init__(self, module_list, weight=1.0, scope=None, reuse=False):
        
        self._module_list = module_list
        self._weight = weight

        super(Average, self).__init__(scope=scope, reuse=reuse)

    def _build_shared_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):

            self._loss = sum([module.get_loss() for module in self._module_list])
            outputs = sum([module.get_outputs() for module in self._module_list], [])
            self._outputs.append(self._weight * tf.add_n(outputs) /  len(self._module_list))
