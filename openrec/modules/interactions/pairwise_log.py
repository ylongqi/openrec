import tensorflow as tf
from openrec.modules.interactions import Interaction

class PairwiseLog(Interaction):

    """
    The PairwiseLog module minimizes the pairwise logarithm loss [bpr]_ as follows (regularization and bias terms \
    are not included):
    
    .. math::
        \min \sum_{(i, p, n)} -ln\sigma (u_i^T v_p - u_i^T v_n)

    where :math:`u_i` denotes the representation for user :math:`i`; :math:`v_p` and :math:`v_n` denote representations for \
    *positive item* :math:`p` and *negative item* :math:`n`, respectively.

    Parameters
    ----------
    user: Tensorflow tensor
        Representations for users involved in the interactions. Shape: **[number of interactions, dimensionality of \
        user representations]**.
    item: Tensorflow tensor, required for testing
        Representations for items involved in the interactions. Shape: **[number of interactions, dimensionality of \
        item representations]**.
    item_bias: Tensorflow tensor, required for testing
        Biases for items involved in the interactions. Shape: **[number of interactions, 1]**.
    p_item: Tensorflow tensor, required for training
        Representations for positive items involved in the interactions. Shape: **[number of interactions, dimensionality of \
        item representations]**.
    p_item_bias: Tensorflow tensor, required for training
        Biases for positive items involved in the interactions. Shape: **[number of interactions, 1]**.
    n_item: Tensorflow tensor, required for training
        Representations for negative items involved in the interactions. Shape: **[number of interactions, dimensionality of \
        item representations]**.
    n_item_bias: Tensorflow tensor, required for training
        Biases for negative items involved in the interactions. Shape: **[number of interactions, 1]**.
    train: bool, optionl
        An indicator for training or serving phase.
    scope: str, optional
        Scope for module variables.
    reuse: bool, optional
        Whether or not to reuse module variables.
    
    References
    ----------
    .. [bpr] Rendle, S., Freudenthaler, C., Gantner, Z. and Schmidt-Thieme, L., 2009, June. BPR: Bayesian personalized ranking \
      from implicit feedback. In Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence (pp. 452-461).\
       AUAI Press.
    """

    def __init__(self, user, item=None, item_bias=None, p_item=None, p_item_bias=None, 
                n_item=None, n_item_bias=None, train=None, scope=None, reuse=False):

        assert train is not None, 'train cannot be None'
        assert user is not None, 'user cannot be None'
        self._user = user

        if train:

            assert p_item is not None, 'p_item cannot be None'
            assert n_item is not None, 'n_item cannot be None'
            assert p_item_bias is not None, 'p_item_bias cannot be None'
            assert n_item_bias is not None, 'n_item_bias cannot be None'

            self._p_item = p_item
            self._n_item = n_item
            self._p_item_bias = p_item_bias
            self._n_item_bias = n_item_bias
        else:
            assert item is not None, 'item cannot be None'
            assert item_bias is not None, 'item_bias cannot be None'

            self._item = item
            self._item_bias = item_bias

        super(PairwiseLog, self).__init__(train=train, scope=scope, reuse=reuse)

    def _build_training_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            dot_user_pos = tf.reduce_sum(tf.multiply(self._user, self._p_item),
                                         reduction_indices=1,
                                         keep_dims=True,
                                         name="dot_user_pos")
            dot_user_neg = tf.reduce_sum(tf.multiply(self._user, self._n_item),
                                         reduction_indices=1,
                                         keep_dims=True,
                                         name="dot_user_neg")
            self._loss = - tf.reduce_sum(tf.log(tf.sigmoid(tf.maximum(dot_user_pos + self._p_item_bias -
                                                                      dot_user_neg - self._n_item_bias,
                                                                      -30.0))))

    def _build_serving_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            self._outputs.append(tf.matmul(self._user, self._item, transpose_b=True) + tf.reshape(self._item_bias, [-1]))

