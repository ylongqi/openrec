from __future__ import print_function
import tensorflow as tf
from termcolor import colored
from openrec.modules.extractions import Extraction


class LatentFactor(Extraction):

    def __init__(self, shape, init, ids=None, l2_reg=None, scope=None, reuse=False):

        assert init is not None, 'init cannot be None'
        assert shape is not None, 'shape cannot be None'

        if init == 'normal':
            self._initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)
        elif init == 'zero':
            self._initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
        self._shape = shape
        self._ids = ids

        super(LatentFactor, self).__init__(l2_reg=l2_reg, scope=scope, reuse=reuse)

    def _build_shared_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            
            embedding = tf.get_variable('embedding', shape=self._shape, trainable=True,
                                      initializer=self._initializer)

            if self._ids is not None:
                self._outputs.append(tf.nn.embedding_lookup(embedding, self._ids))

                if self._l2_reg is not None:
                    self._loss = self._l2_reg * tf.nn.l2_loss(self._outputs[0])
                    
            else:
                self._outputs.append(embedding)

    def censor_l2_norm_op(self, censor_id_list=None, max_norm=1):

        with tf.variable_scope(self._scope, reuse=True):
            embedding = tf.get_variable('embedding', shape=self._shape, trainable=True,
                                        initializer=self._initializer)
            embedding_gather = tf.gather(embedding, indices=censor_id_list)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_gather), axis=1, keep_dims=True))
        return tf.scatter_update(embedding, indices=censor_id_list, updates=embedding_gather / tf.maximum(norm, max_norm))

