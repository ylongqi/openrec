import tensorflow as tf
from openrec.legacy.modules.interactions import Interaction

class NSEuDist(Interaction):

    def __init__(self, user, max_item, item=None, item_bias=None, p_item=None, neg_num=5,
    p_item_bias=None,  n_item=None, n_item_bias=None, margin=1.0, train=None, 
    scope=None, reuse=False):

        self._max_item = max_item
        self._margin = margin
        self._neg_num = neg_num

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

        super(NSEuDist, self).__init__(train=train, scope=scope, reuse=reuse)


    def _build_training_graph(self):
        
        with tf.variable_scope(self._scope, reuse=self._reuse):
            tmp_user = tf.tile(tf.expand_dims(self._user, 1), [1, self._neg_num, 1])

            l2_user_pos = tf.tile(tf.reduce_sum(tf.square(tf.subtract(self._user, self._p_item)),
                                                reduction_indices=1,
                                                keep_dims=True, name="l2_user_pos"), [1, self._neg_num])
            l2_user_neg = tf.reduce_sum(tf.square(tf.subtract(tmp_user, self._n_item)),
                                        reduction_indices=2, 
                                        name="l2_user_neg")
            pos_score = (-l2_user_pos) + tf.tile(self._p_item_bias, [1, self._neg_num])
            neg_score = (-l2_user_neg) + tf.reduce_sum(self._n_item_bias, reduction_indices=2)
            scores = tf.maximum(self._margin - pos_score + neg_score, 0)
            weights = tf.count_nonzero(scores, axis=1)
            weights = tf.log(tf.floor(self._max_item * tf.to_float(weights) / self._neg_num) + 1.0)
            self._loss = tf.reduce_sum(weights * tf.reduce_max(scores, axis=1))
            # self._loss = tf.reduce_sum(tf.tile(tf.reshape(weights, [-1, 1]), [1, self._neg_num]) * scores)

    def _build_serving_graph(self):
        
        with tf.variable_scope(self._scope, reuse=self._reuse):
            item_norms = tf.reduce_sum(tf.square(self._item), axis=1)
            self._outputs.append(2 * tf.matmul(self._user, self._item, transpose_b=True) + \
                            tf.reshape(self._item_bias, [-1]) - tf.reshape(item_norms, [-1]))

