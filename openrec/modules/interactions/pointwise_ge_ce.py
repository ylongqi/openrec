from __future__ import print_function
import tensorflow as tf
from termcolor import colored
from openrec.modules.extractions import MultiLayerFC
from openrec.modules.interactions import Interaction 

class PointwiseGeCE(Interaction):

    """
    The PointwiseGeCE module minimizes the cross entropy classification loss with generalized dot product as logits. The generalized\
    dot-product [ncf]_ between user representation :math:`u_i` and item representation :math:`v_j` is defined as: 
    
    .. math::
        h^T(u_i \odot v_j)

    where :math:`\odot` denotes element-wise dot product of two vectors, and :math:`h` denotes learnable model parameters.

    Parameters
    ----------
    user: Tensorflow tensor
        Representations for users involved in the interactions. Shape: **[number of interactions, dimensionality of \
        user representations]**.
    item: Tensorflow tensor
        Representations for items involved in the interactions. Shape: **[number of interactions, dimensionality of \
        item representations]**.
    item_bias: Tensorflow tensor
        Biases for items involved in the interactions. Shape: **[number of interactions, 1]**.
    labels: Tensorflow tensor, required for training.
        Groundtruth labels for the interactions. Shape **[number of interactions, ]**.
    l2_reg: float, optional
        Weight for L2 regularization, i.e., weight decay.
    train: bool, optionl
        An indicator for training or servining phase.
    scope: str, optional
        Scope for module variables.
    reuse: bool, optional
        Whether or not to reuse module variables.
    
    References
    ----------
    .. [ncf] He, X., Liao, L., Zhang, H., Nie, L., Hu, X. and Chua, T.S., 2017, April. Neural collaborative filtering. \
        In Proceedings of the 26th International Conference on World Wide Web (pp. 173-182). International World Wide Web \
        Conferences Steering Committee.
    """

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

            logits = gdp.get_outputs()[0] + self._item_bias
            labels_float = tf.reshape(tf.to_float(self._labels), (-1, 1))
            self._loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_float, logits=logits))
            self._outputs.append(logits)

    def _build_serving_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            user_rep = tf.reshape(tf.tile(self._user, [1, tf.shape(self._item)[0]]), (-1, tf.shape(self._user)[1]))
            item_rep = tf.tile(self._item, (tf.shape(self._user)[0], 1))
            item_bias_rep = tf.tile(self._item_bias, (tf.shape(self._user)[0], 1))
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
            self._outputs.append(tf.reshape(gdp.get_outputs()[0] + item_bias_rep, (tf.shape(self._user)[0], tf.shape(self._item)[0])))

    
