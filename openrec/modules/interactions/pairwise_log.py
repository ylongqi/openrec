import tensorflow as tf
from openrec.modules.interactions import Interaction

class PairwiseLog(Interaction):

    def __init__(self, user, train=None, item=None, item_bias=None, p_item=None, n_item=None,
                p_item_bias=None, n_item_bias=None, scope=None, reuse=False):

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

