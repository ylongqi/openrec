import tensorflow as tf
from openrec.modules.fusions import Fusion

class Concat(Fusion):

    def __init__(self, module_list, scope=None, reuse=False):

        self._module_list = module_list

        super(Concat, self).__init__(l2_reg=None, scope=scope, reuse=reuse)

    def _build_shared_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):

            self._loss = sum([cell.get_loss() for cell in self._module_list])
            outputs = sum([cell.get_outputs() for cell in self._module_list], [])
            self._outputs.append(tf.concat(values=outputs, axis=1))
