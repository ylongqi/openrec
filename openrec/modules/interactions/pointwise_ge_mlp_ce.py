from __future__ import print_function
import tensorflow as tf
from termcolor import colored
from openrec.modules.interactions import Interaction
from openrec.modules.interactions import PointwiseGeCE
from openrec.modules.interactions import PointwiseMLPCE

class PointwiseGeMLPCE(Interaction):

    """
    The PointwiseGeMLPCE module minimizes the cross entropy classification loss. The logits are calculated as follows [ncf]_ \
    (Bias term is not included).
    
    .. math::
        \\alpha h^T(u_i^{ge} \odot v_j^{ge}) + (1 - \\alpha)MLP([u_i^{mlp}, v_j^{mlp}])

    Parameters
    ----------
    user_mlp: Tensorflow tensor
        :math:`u^{mlp}` for users involved in the interactions. Shape: [number of interactions, dimensionality of \
        :math:`u^{mlp}`].
    user_ge: Tensorflow tensor
        :math:`u^{ge}` for users involved in the interactions. Shape: [number of interactions, dimensionality of \
        :math:`u^{ge}`].
    item_mlp: Tensorflow tensor
        :math:`v^{mlp}` for items involved in the interactions. Shape: [number of interactions, dimensionality of \
        :math:`v^{mlp}`].
    item_ge: Tensorflow tensor
        :math:`v^{ge}` for items involved in the interactions. Shape: [number of interactions, dimensionality of \
        :math:`v^{ge}`].
    item_bias: Tensorflow tensor
        Biases for items involved in the interactions. Shape: [number of interactions, 1].
    dims: Numpy array.
        Specify the size of the MLP (openrec.modules.extractions.MultiLayerFC).
    labels: Tensorflow tensor, required for training.
        Groundtruth labels for the interactions. Shape [number of interactions, ].
    dropout: float, optional.
        Dropout rate for MLP (intermediate layers only).
    alpha: float, optional.
        Value of :math:`\\alpha`. Default to 0.5.
    l2_reg: float, optional
        Weight for L2 regularization, i.e., weight decay.
    train: bool, optionl
        An indicator for training or servining phase.
    scope: str, optional
        Scope for module variables.
    reuse: bool, optional
        Whether or not to reuse module variables.
    """

    def __init__(self, user_mlp, user_ge, item_mlp, item_ge, item_bias, dims, labels=None, dropout=None, alpha=0.5, 
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
            self._ge = PointwiseGeCE(user=self._user_ge, item=self._item_ge, item_bias=0, labels=self._labels, l2_reg=self._l2_reg,
                        train=self._train, scope='ge', reuse=self._reuse)
            self._mlp = PointwiseMLPCE(user=self._user_mlp, item=self._item_mlp, item_bias=0, labels=self._labels, l2_reg=self._l2_reg,
                        dims=self._dims, train=self._train, dropout=self._dropout, scope='mlp', reuse=self._reuse)

    def _build_training_graph(self):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            logits = self._alpha * self._ge.get_outputs()[0] + (1 - self._alpha) * self._mlp.get_outputs()[0] + self._item_bias
            labels_float = tf.reshape(tf.to_float(self._labels), (-1, 1))
            self._loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_float, logits=logits))
            self._outputs.append(logits)

    def _build_serving_graph(self):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            item_bias_rep = tf.tile(self._item_bias, (tf.shape(self._user)[0], 1))
            logits = self._alpha * self._ge.get_outputs()[0] + (1 - self._alpha) * self._mlp.get_outputs()[0] \
                        + tf.reshape(item_bias_rep, (tf.shape(self._user)[0], tf.shape(self._item)[0]))
            self._outputs.append(logits)

    
