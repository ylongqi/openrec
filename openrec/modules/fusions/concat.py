import tensorflow as tf
from openrec.modules.fusions import Fusion

class Concat(Fusion):

    """
    The Concat module outputs the concatenation of the outputs from multiple modules.

    Parameters
    ----------
    module_list: list
        The list of modules.
    scope: str, optional
        Scope for module variables.
    reuse: bool, optional
        Whether or not to reuse module variables.
    """
    
    def __init__(self, module_list, scope=None, reuse=False):

        self._module_list = module_list

        super(Concat, self).__init__(l2_reg=None, scope=scope, reuse=reuse)

    def _build_shared_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):

            outputs = sum([cell.get_outputs() for cell in self._module_list], [])
            self._outputs.append(tf.concat(values=outputs, axis=1))
            self._loss = 0.0
