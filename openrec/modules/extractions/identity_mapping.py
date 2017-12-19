import tensorflow as tf
from openrec.modules.extractions import Extraction

class IdentityMapping(Extraction):

    """
    The IdentityMapping module executes an identity function.

    Parameters
    ----------
    value: Tensorflow tensor
        Input tensor
    scope: str, optional
        Scope for module variables.
    reuse: bool, optional
        Whether or not to reuse module variables.
    """

    def __init__(self, value, scope=None, reuse=False):

        assert value is not None, 'value cannot be None'
        self._value = value
        super(IdentityMapping, self).__init__(l2_reg=None, scope=scope, reuse=False)

    def _build_shared_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            self._outputs.append(self._value)
