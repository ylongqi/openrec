from __future__ import print_function
import tensorflow as tf
from openrec.legacy.modules.interactions import Interaction
from openrec.legacy.modules.extractions import MultiLayerFC

class PointwiseMLPCE(Interaction):

    """
    The PointwiseMLPCE module minimizes the cross entropy classification loss with outputs of a Multi-Layer Perceptron (MLP) as logits. \
    The inputs to the MLP are the concatenation between user and item representations [ncf]_.

    Parameters
    ----------
    user: Tensorflow tensor
        Representations for users involved in the interactions. Shape: **[number of interactions, dimensionality of \
        user representations]**.
    item: Tensorflow tensor
        Representations for items involved in the interactions. Shape: **[number of interactions, dimensionality of \
        item representations]**.
    dims: Numpy array.
        Specify the size of the MLP (openrec.legacy.modules.extractions.MultiLayerFC).
    item_bias: Tensorflow tensor, optional
        Biases for items involved in the interactions. Shape: **[number of interactions, 1]**.
    extra: Tensorflow tensor, optional
        Representations for context involved in the interactions. Shape: **[number of interaction, dimensionality of \
        context representations]**
    l2_reg: float, optional
        Weight for L2 regularization, i.e., weight decay.
    labels: Tensorflow tensor, required for training.
        Groundtruth labels for the interactions. Shape **[number of interactions, ]**.
    dropout: float, optional
        Dropout rate for MLP (intermediate layers only).
    train: bool, optional
        An indicator for training or servining phase.
    batch_serving: bool, optional
        An indicator for batch serving / pointwise serving.
    scope: str, optional
        Scope for module variables.
    reuse: bool, optional
        Whether or not to reuse module variables.
    """

    def __init__(self, user, item, dims, item_bias=None, extra=None, l2_reg=None, labels=None,
                 dropout=None, train=None, batch_serving=True, scope=None, reuse=False):
        
        assert dims is not None, 'dims cannot be None'
        assert dims[-1] == 1, 'last value of dims should be 1'

        self._user = user
        self._item = item
        self._item_bias = item_bias
        self._extra = extra
        self._dropout = dropout
        self._batch_serving = batch_serving

        if train:
            assert labels is not None, 'labels cannot be None'
            self._labels = labels

        self._dims = dims

        super(PointwiseMLPCE, self).__init__(l2_reg=l2_reg, train=train, scope=scope, reuse=reuse)

    def _build_training_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            
            if self._extra is not None:
                in_tensor = tf.concat([self._user, self._item, self._extra], axis=1)
            else:
                in_tensor = tf.concat([self._user, self._item], axis=1)
            
            reg = MultiLayerFC(
                in_tensor=in_tensor,
                dims=self._dims,
                bias_in=True,
                bias_mid=True,
                bias_out=False,
                dropout_mid=self._dropout,
                l2_reg=self._l2_reg,
                scope='mlp_reg',
                reuse=self._reuse)
            
            logits = reg.get_outputs()[0]
            if self._item_bias is not None:
                logits += self._item_bias
            
            labels_float = tf.reshape(tf.to_float(self._labels), (-1, 1))
            self._loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_float, logits=logits))
            self._outputs.append(logits)

    def _build_serving_graph(self):
        
        with tf.variable_scope(self._scope, reuse=self._reuse):
            if self._batch_serving:
                user_rep = tf.reshape(tf.tile(self._user, [1, tf.shape(self._item)[0]]), (-1, tf.shape(self._user)[1]))
                item_rep = tf.tile(self._item, (tf.shape(self._user)[0], 1))
                if self._extra is not None:
                    extra_rep = tf.tile(self._extra, (tf.shape(self._user)[0], 1))
                    in_tensor = tf.concat([user_rep, item_rep, extra_rep], axis=1)
                else:
                    in_tensor = tf.concat([user_rep, item_rep], axis=1)
                reg = MultiLayerFC(in_tensor=in_tensor,
                                   dims=self._dims,
                                   bias_in=True,
                                   bias_mid=True,
                                   bias_out=False,
                                   l2_reg=self._l2_reg,
                                   scope='mlp_reg',
                                   reuse=self._reuse)
                if self._item_bias is not None:
                    item_bias_rep = tf.tile(self._item_bias, (tf.shape(self._user)[0], 1))
                    self._outputs.append(tf.reshape(reg.get_outputs()[0] + item_bias_rep, (tf.shape(self._user)[0], tf.shape(self._item)[0])))
                else:
                    self._outputs.append(tf.reshape(reg.get_outputs()[0], (tf.shape(self._user)[0], tf.shape(self._item)[0])))
            else:
                if self._extra is not None:
                    in_tensor = tf.concat([self._user, self._item, self._extra], axis=1)
                else:
                    in_tensor = tf.concat([self._user, self._item], axis=1)
                reg = MultiLayerFC(in_tensor=in_tensor,
                                   dims=self._dims,
                                   bias_in=True,
                                   bias_mid=True,
                                   bias_out=False,
                                   l2_reg=self._l2_reg,
                                   scope='mlp_reg',
                                   reuse=self._reuse)
                logits = reg.get_outputs()[0]
                if self._item_bias is not None:
                    logits += self._item_bias
                self._outputs.append(tf.sigmoid(logits))

