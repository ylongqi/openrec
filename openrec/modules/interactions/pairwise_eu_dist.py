import tensorflow as tf
from openrec.modules.interactions import PairwiseLog


class PairwiseEuDist(PairwiseLog):

    """
    The PairwiseEuDist module minimizes the weighted pairwise euclidean distance-based hinge loss [cml]_ as follows (regularization and bias terms \
    are not included):
    
    .. math::
        \min \sum_{(i, p, n)} w_{ip} [m + \lVert c(u_i)-c(v_p) \lVert^2 - \lVert c(u_i)-c(v_n) \lVert^2]_+

    where :math:`c(x) = \\frac{x}{\max(\lVert x \lVert, 1.0)}`; :math:`u_i` denotes the representation for user :math:`i`; :math:`v_p` and :math:`v_n` denote representations for \
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
    weights: Tensorflow tensor, optional
        Weights :math:`w`. Shape: **[number of interactions, 1]**.
    margin: float, optional
        Margin :math:`m`. Default to 1.0.
    train: bool, optionl
        An indicator for training or serving phase.
    scope: str, optional
        Scope for module variables.
    reuse: bool, optional
        Whether or not to reuse module variables.
    
    References
    ----------
    .. [cml] Hsieh, C.K., Yang, L., Cui, Y., Lin, T.Y., Belongie, S. and Estrin, D., 2017, April. Collaborative metric learning. \
        In Proceedings of the 26th International Conference on World Wide Web (pp. 193-201). International World Wide Web Conferences \
        Steering Committee.
    """

    def __init__(self, user, item=None, item_bias=None, p_item=None,
    p_item_bias=None,  n_item=None, n_item_bias=None, weights=1.0, margin=1.0, train=None, 
    scope=None, reuse=False):

        self._weights = weights
        self._margin = margin

        super(PairwiseEuDist, self).__init__(user=user, item=item, item_bias=item_bias, p_item=p_item,
        n_item=n_item, p_item_bias=p_item_bias, n_item_bias=n_item_bias, train=train, scope=scope, reuse=reuse)

    def _build_training_graph(self):
        
        with tf.variable_scope(self._scope, reuse=self._reuse):
            self._user = self._censor_norm(self._user)
            self._p_item = self._censor_norm(self._p_item)
            self._n_item = self._censor_norm(self._n_item)

            l2_user_pos = tf.reduce_sum(tf.square(tf.subtract(self._user, self._p_item)),
                                        reduction_indices=1,
                                        keep_dims=True, name="l2_user_pos")
            l2_user_neg = tf.reduce_sum(tf.square(tf.subtract(self._user, self._n_item)),
                                        reduction_indices=1,
                                        keep_dims=True, name="l2_user_neg")
            pos_score = (-l2_user_pos) + self._p_item_bias
            neg_score = (-l2_user_neg) + self._n_item_bias
            diff = pos_score - neg_score
            self._loss = tf.reduce_sum(self._weights * tf.maximum(self._margin - diff, 0))

    def _build_serving_graph(self):
        
        with tf.variable_scope(self._scope, reuse=self._reuse):
            self._user = self._censor_norm(self._user)
            self._item = self._censor_norm(self._item)
            item_norms = tf.reduce_sum(tf.square(self._item), axis=1)
            self._outputs.append(2 * tf.matmul(self._user, self._item, transpose_b=True) + \
                            tf.reshape(self._item_bias, [-1]) - tf.reshape(item_norms, [-1]))

    def _censor_norm(self, in_tensor):
        norm = tf.sqrt(tf.reduce_sum(tf.square(in_tensor), axis=1, keep_dims=True))
        return in_tensor / tf.maximum(norm, 1.0)
