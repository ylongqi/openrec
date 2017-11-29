import tensorflow as tf
from openrec.modules.extractions import Extraction

class IdentityMapping(Extraction):

    def __init__(self, value, scope=None, reuse=False):

        assert value is not None, 'value cannot be None'
        self._value = value
        super(IdentityMapping, self).__init__(l2_reg=None, scope=None, reuse=False)

    def _build_shared_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            self._outputs.append(self._value)
