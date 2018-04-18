import tensorflow as tf
from openrec.modules.interactions import Interaction

class NSEuDist(Interaction):

    def __init__(self, user, item=None, item_bias=None, p_item=None, neg_num=5,
    p_item_bias=None,  n_item=None, n_item_bias=None, weights=1.0, margin=1.0, train=None, 
    scope=None, reuse=False):

        self._weights = weights
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
            self._user = self._censor_norm(self._user)
            self._p_item = self._censor_norm(self._p_item)
            self._n_item = self._censor_norm(self._n_item)
            tmp_user = tf.tile(tf.expand_dims(self._user, 1), [1, self._neg_num, 1])

            l2_user_pos = tf.tile(tf.reduce_sum(tf.square(tf.subtract(self._user, self._p_item)),
                                                reduction_indices=1,
                                                keep_dims=True, name="l2_user_pos"), [1, self._neg_num])
            l2_user_neg = tf.reduce_sum(tf.square(tf.subtract(tmp_user, self._n_item)),
                                        reduction_indices=2, 
                                        name="l2_user_neg")
            pos_score = l2_user_pos + tf.tile(self._p_item_bias, [1, self._neg_num])         # shape=(2000, self._neg_num)
            neg_score = l2_user_neg + tf.reduce_sum(self._n_item_bias, reduction_indices=2)  # shape=(2000, self._neg_num)
            self._loss = tf.reduce_sum(self._weights * tf.maximum(self._margin + pos_score - neg_score, 0))

    def _build_serving_graph(self):
        
        with tf.variable_scope(self._scope, reuse=self._reuse):
            self._user = self._censor_norm(self._user)
            self._item = self._censor_norm(self._item)
            item_norms = tf.reduce_sum(tf.square(self._item), axis=1)
            self._outputs.append(2 * tf.matmul(self._user, self._item, transpose_b=True) + \
                            tf.reshape(self._item_bias, [-1]) - tf.reshape(item_norms, [-1]))

    def _censor_norm(self, in_tensor):
        norm = tf.sqrt(tf.reduce_sum(tf.square(in_tensor), axis=-1, keep_dims=True))
        return in_tensor / tf.maximum(norm, 1.0)
