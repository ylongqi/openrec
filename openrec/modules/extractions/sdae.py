from __future__ import print_function
import tensorflow as tf
from termcolor import colored
from openrec.modules.extractions import Extraction
from openrec.modules.extractions import MultiLayerFC

class SDAE(Extraction):

    def __init__(self, in_tensor, dims, dropout_rate=None, l2_reg_mlp=None, l2_reg_n=None, scope=None, reuse=False):

        assert dims is not None, 'dims cannot be None'
        assert in_tensor is not None, 'in_tensor cannot be None'

        self._in_tensor = in_tensor
        self._dims = dims
        self._dropout_rate = dropout_rate
        self._l2_reg_mlp = l2_reg_mlp
        self._l2_reg_n = l2_reg_n

        super(SDAE, self).__init__(l2_reg=None, scope=scope, reuse=reuse)

    def _build_shared_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):

            _encoder = MultiLayerFC(l2_reg=self._l2_reg_mlp, in_tensor=self._in_tensor, dims=self._dims[1:], scope='encoder',
                            dropout_in=self._dropout_rate, dropout_mid=self._dropout_rate, reuse=self._reuse)
            _decoder = MultiLayerFC(l2_reg=self._l2_reg_mlp, in_tensor=_encoder.get_outputs()[0], dims=self._dims[::-1][1:],
                            scope='decoder', relu_in=True, dropout_in=self._dropout_rate, relu_mid=True,
                            dropout_mid=self._dropout_rate, relu_out=True, dropout_out=self._dropout_rate, reuse=self._reuse)

            self._outputs += _encoder.get_outputs()
            self._loss = _encoder.get_loss() + _decoder.get_loss()
            if self._l2_reg_n is not None:
                self._loss += self._l2_reg_n * tf.nn.l2_loss(_decoder.get_outputs()[0] - self._in_tensor)
