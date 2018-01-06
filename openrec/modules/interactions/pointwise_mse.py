from __future__ import print_function
import tensorflow as tf
from termcolor import colored
from openrec.modules.interactions import Interaction

class PointwiseMSE(Interaction):

    """
    The PointwiseMSE module minimizes the pointwise mean-squre-error [ctm]_ as follows (regularization terms \
    are not included):
    
    .. math::
        \min \sum_{ij}c_{ij}(r_{ij} - u_i^T v_j)^2

    where :math:`u_i` and :math:`v_j` are representations for user :math:`i` and item :math:`j` respectively; \
    :math:`c_{ij}=a` if :math:`r_{ij}=1`, otherwise :math:`c_{ij}=b`.

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
    labels: Tensorflow tensor, required for training
        Groundtruth labels for the interactions. Shape **[number of interactions, ]**.
    a: float, optional
        The value of :math:`c_{ij}` if :math:`r_{ij}=1`.
    b: float, optional
        The value of :math:`c_{ij}` if :math:`r_{ij}=0`.
    sigmoid: bool, optional
        Normalize the dot products, i.e., sigmoid(:math:`u_i^T v_j`).
    train: bool, optionl
        An indicator for training or servining phase.
    batch_serving: bool, optional
        If True, the model calculates scores for all users against all items, and returns scores with shape [len(user), len(item)]. Otherwise, it returns scores for specified user item pairs (require :code:`len(user)==len(item)`).
    scope: str, optional
        Scope for module variables.
    reuse: bool, optional
        Whether or not to reuse module variables.
    
    References
    ----------
    .. [ctm] Wang, C. and Blei, D.M., 2011, August. Collaborative topic modeling for recommending scientific articles. \
        In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 448-456). ACM.
    """

    def __init__(self, user, item, item_bias, labels=None, a=1.0, b=1.0, 
                sigmoid=False, train=True, batch_serving=True, scope=None, reuse=False):

        assert train is not None, 'train cannot be None'
        assert user is not None, 'user cannot be None'
        assert item is not None, 'item cannot be None'
        assert item_bias is not None, 'item_bias cannot be None'

        self._user = user
        self._item = item
        self._item_bias = item_bias
        self._sigmoid = sigmoid
        self._batch_serving = batch_serving

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
            
            if self._sigmoid:
                prediction = tf.sigmoid(dot_user_item + tf.reshape(self._item_bias, [-1]))
            else:
                prediction = dot_user_item + tf.reshape(self._item_bias, [-1])
                
            self._loss = tf.nn.l2_loss(labels_weight * (self._labels - prediction))

    def _build_serving_graph(self):
        
        with tf.variable_scope(self._scope, reuse=self._reuse):
            
            if self._batch_serving:
                prediction = tf.matmul(self._user, self._item, transpose_b=True) + tf.reshape(self._item_bias, [-1])
            else:
                dot_user_item = tf.reduce_sum(tf.multiply(self._user, self._item),
                                          axis=1, keep_dims=False, name="dot_user_item")
                prediction = dot_user_item + tf.reshape(self._item_bias, [-1])
            
            if self._sigmoid:
                prediction = tf.sigmoid(prediction)
                
            self._outputs.append(prediction)
