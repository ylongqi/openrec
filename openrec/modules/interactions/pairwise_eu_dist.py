import tensorflow as tf
from openrec.modules.interactions import PairwiseLog


class PairwiseEuDist(PairwiseLog):

    def __init__(self, user, item=None, item_bias=None, p_item=None, n_item=None,
    p_item_bias=None, n_item_bias=None, train=None, scope=None, reuse=False):

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
            self._loss = tf.reduce_sum(tf.maximum(1-diff, 0))

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
