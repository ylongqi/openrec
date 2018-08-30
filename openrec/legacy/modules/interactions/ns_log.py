import tensorflow as tf
from openrec.legacy.modules.interactions import Interaction

class NsLog(Interaction):

    def __init__(self, user, max_item, item=None, item_bias=None, p_item=None, p_item_bias=None, neg_num=5,
                n_item=None, n_item_bias=None, train=None, scope=None, reuse=False):

        assert train is not None, 'train cannot be None'
        assert user is not None, 'user cannot be None'
        self._user = user
        self._neg_num = neg_num
        self._max_item = max_item

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

        super(NsLog, self).__init__(train=train, scope=scope, reuse=reuse)

    def _build_training_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            tmp_user = tf.tile(tf.expand_dims(self._user, 1), [1, self._neg_num, 1])
            dot_user_pos = tf.tile(tf.reduce_sum(tf.multiply(self._user, self._p_item),
                                         reduction_indices=1,
                                         keep_dims=True,
                                         name="dot_user_pos"),[1,self._neg_num])
            dot_user_neg = tf.reduce_sum(tf.multiply(tmp_user, self._n_item),
                                         reduction_indices=2,
                                         name="dot_user_neg")
            
            pos_score = dot_user_pos + tf.tile(self._p_item_bias, [1, self._neg_num])
            neg_score = dot_user_neg + tf.reduce_sum(self._n_item_bias, reduction_indices=2)
            diff = pos_score - neg_score
            weights = tf.count_nonzero(tf.less(diff, 0.0), axis=1)
            weights = tf.log(tf.floor(self._max_item * tf.to_float(weights) / self._neg_num) + 1.0)
            self._loss = - tf.reduce_sum(tf.log(tf.sigmoid(tf.maximum(weights * tf.reduce_min(diff, axis = 1),
                                                                      -30.0))))

    def _build_serving_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            self._outputs.append(tf.matmul(self._user, self._item, transpose_b=True) + tf.reshape(self._item_bias, [-1]))

