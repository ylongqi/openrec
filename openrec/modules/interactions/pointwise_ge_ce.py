from __future__ import print_function
import tensorflow as tf
from termcolor import colored
from openrec.modules.extractions import MultiLayerFC
from openrec.modules.interactions import Interaction 

# Generalized Matrix Factorization
# Pointwise Generalized Cross-Entropy

class PointwiseGeCE(Interaction):

    def __init__(self, user, item, item_bias, l2_reg=None, labels=None,
            train=None, scope=None, reuse=False):

        assert train is not None, 'train cannot be None'
        assert user is not None, 'user cannot be None'
        assert item is not None, 'item cannot be None'
        assert item_bias is not None, 'item_bias cannot be None'

        self._user = user
        self._item = item
        self._item_bias = item_bias

        if train:
            assert labels is not None, 'labels cannot be None'
            self._labels = labels

        super(PointwiseGeCE, self).__init__(train=train, l2_reg=l2_reg, scope=scope, reuse=reuse)

    def _build_training_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            pointwise_product = tf.multiply(self._user, self._item)
            gdp = MultiLayerFC(
                in_tensor=pointwise_product,
                dims=[1],
                bias_in=False,
                bias_mid=False,
                bias_out=False,
                l2_reg=self._l2_reg,
                scope='gmf_reg',
                reuse=self._reuse)

            labels_float = tf.reshape(tf.to_float(self._labels), (-1, 1))
            self._loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_float, logits=gdp.get_outputs()[0] + self._item_bias))

    def _build_serving_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            user_rep = tf.reshape(tf.tile(self._user, [1, tf.shape(self._item)[0]]), (-1, tf.shape(self._user)[1]))
            item_rep = tf.tile(self._item, (tf.shape(self._user)[0], 1))
            pointwise_product = tf.multiply(user_rep, item_rep)
            gdp = MultiLayerFC(
                        in_tensor=pointwise_product,
                        dims=[1],
                        bias_in=False,
                        bias_mid=False,
                        bias_out=False,
                        l2_reg=self._l2_reg,
                        scope='gmf_reg',
                        reuse=self._reuse)
            self._outputs.append(tf.reshape(gdp.get_outputs()[0], (tf.shape(self._user)[0], tf.shape(self._item)[0])) 
                        + self._item_bias)

    
