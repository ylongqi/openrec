from __future__ import print_function
import tensorflow as tf
from openrec.modules.interactions import Interaction
from openrec.modules.extractions import MultiLayerFC

class PointwiseMLPCE(Interaction):

    def __init__(self, user, item, item_bias, dims, l2_reg=None, dropout=None, labels=None,
                 train=None, scope=None, reuse=False):
        
        assert dims is not None, 'dims cannot be None'
        assert dims[-1] == 1, 'last value of dims should be 1'

        self._user = user
        self._item = item
        self._item_bias = item_bias

        if train:
            assert labels is not None, 'labels cannot be None'
            self._labels = labels

        self._dims = dims
        self._dropout = dropout

        super(PointwiseMLP, self).__init__(l2_reg=l2_reg, train=train, scope=scope, reuse=reuse)

    def _build_shared_graph(self):
        
        with tf.variable_scope(self._scope, reuse=self._reuse):
            in_tensor = tf.concat([self._user, self._item], axis=1)
            self._reg = MultiLayerFC(
                in_tensor=in_tensor,
                dims=self._dims,
                bias_in=True,
                bias_mid=True,
                bias_out=False,
                l2_reg=self._l2_reg,
                scope='mlp_reg',
                reuse=self._reuse)

            self._embed = self._reg.get_outputs()[0]

    def _build_training_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            labels_float = tf.reshape(tf.to_float(self._labels), (-1, 1))
            self._loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_float, logits=self._reg.get_outputs()[0] + self._item_bias))

    def _build_serving_graph(self):
        
        with tf.variable_scope(self._scope, reuse=self._reuse):
            self._outputs.append(self._reg.get_outputs()[0] + self._item_bias)
