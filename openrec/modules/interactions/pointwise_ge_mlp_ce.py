from __future__ import print_function
import tensorflow as tf
from termcolor import colored
from openrec.modules.interactions import Interaction
from openrec.modules.interactions import PointwiseGeCE
from openrec.modules.interactions import PointwiseMLPCE

class PointwiseGeMLPCE(Interaction):

    def __init__(self, user_mlp, user_ge, item_mlp, item_ge, item_bias, dims, dropout=None, alpha=0.5, labels=None,
                l2_reg=None, train=None, scope=None, reuse=False):

        self._user_mlp = user_mlp
        self._user_ge = user_ge
        self._item_mlp = item_mlp
        self._item_ge = item_ge
        self._item_bias = item_bias
        self._dims = dims
        self._alpha = alpha
        self._train = train
        self._labels = labels
        self._dropout = dropout

        super(PointwiseGeMLPCE, self).__init__(train=train, l2_reg=l2_reg, scope=scope, reuse=reuse)

    def _build_shared_graph(self):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            self._ge = PointwiseGeCE(user=self._user_ge, item=self._item_ge, item_bias=self._item_bias, labels=self._labels, l2_reg=self._l2_reg,
                        train=self._train, scope='ge', reuse=self._reuse)
            self._mlp = PointwiseMLPCE(user=self._user_mlp, item=self._item_mlp, item_bias=self._item_bias, labels=self._labels, l2_reg=self._l2_reg,
                        dims=self._dims, train=self._train, dropout=self._dropout, scope='mlp', reuse=self._reuse)
            self._logits = self._alpha * self._ge.get_embed() + (1 - self._alpha) * self._mlp.get_embed() + self._item_bias

    def _build_training_graph(self):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            labels_float = tf.reshape(tf.to_float(self._labels), (-1, 1))
            self._loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_float, logits=self._logits))

    def _build_serving_graph(self):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            self._outputs.append(self._logits)

    
