from __future__ import print_function
import tensorflow as tf
from termcolor import colored
from openrec.modules.interactions import Interaction

class PointwiseMSE(Interaction):

    def __init__(self, user, item, item_bias, labels=None, train=None, a=1.0, b=1.0, 
                scale=None, offset=None, scope=None, reuse=False):

        assert train is not None, 'train cannot be None'
        assert user is not None, 'user cannot be None'
        assert item is not None, 'item cannot be None'
        assert item_bias is not None, 'item_bias cannot be None'

        self._user = user
        self._item = item
        self._item_bias = item_bias
        self._scale = scale
        self._offset = offset

        if train:
            assert labels is not None, 'labels cannot be None'
            self._labels = tf.reshape(tf.to_float(labels), (-1,))
            self._a = a
            self._b = b

        super(PointwiseMSE, self).__init__(train=train, scope=scope, reuse=reuse)

    def _build_training_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):

            labels_weight = (self._a - self._b) * self._labels + self._b
            dot_user_item = tf.reduce_sum(tf.multiply(self._user, self._item),
                                          axis=1, keep_dims=False, name="dot_user_item")

            prediction = tf.sigmoid(dot_user_item + self._item_bias)
            self._loss = tf.nn.l2_loss(labels_weight * (self._labels - prediction))

    def _build_serving_graph(self):
        
        with tf.variable_scope(self._scope, reuse=self._reuse):
            self._outputs.append(tf.matmul(self._user, self._item, transpose_b=True) + tf.reshape(self._item_bias, [-1]))
