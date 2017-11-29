from __future__ import print_function
import tensorflow as tf
from termcolor import colored
from openrec.modules.extractions import Extraction


class MultiLayerFC(Extraction):

    def __init__(self, in_tensor, dims,
                relu_in=False, relu_mid=True, phase=None, relu_out=False,
                dropout_in=None, dropout_mid=None, dropout_out=None, batch_norm=False,
                l2_reg=None, bias_in=True, bias_mid=True, bias_out=True,
                scope=None, reuse=False, id=0):

        assert dims is not None, 'dims cannot be None'
        assert in_tensor is not None, 'in_tensor cannot be None'
        self._in_tensor = in_tensor
        self._dims = dims
        self._relu_in = relu_in
        self._relu_mid = relu_mid
        self._relu_out = relu_out
        self._dropout_in = dropout_in
        self._dropout_mid = dropout_mid
        self._dropout_out = dropout_out
        self._batch_norm = batch_norm
        self._bias_in = bias_in
        self._bias_mid = bias_mid
        self._bias_out = bias_out
        self._phase = phase
        self._id = id


        super(MultiLayerFC, self).__init__(l2_reg=l2_reg, scope=scope, reuse=reuse)

    def _build_shared_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse) as var_scope:
            _in = self._in_tensor

            if self._relu_in:
                _in = tf.nn.relu(_in)

            if self._dropout_in is not None:
                _in = tf.nn.dropout(_in, 1 - self._dropout_in) # Tensorflow uses keep_prob
                
            for index, _out_dim in enumerate(self._dims):
                self._mat = tf.get_variable('FC_' + str(self._id) + '_' + str(index), shape=[_in.shape[1], _out_dim], trainable=True,
                                initializer=tf.contrib.layers.xavier_initializer())
                if index == 0:
                    add_bias = self._bias_in
                elif index == len(self._dims) - 1:
                    add_bias = self._bias_out
                else:
                    add_bias = self._bias_mid

                if add_bias:
                    _bias = tf.get_variable('bias_' + str(self._id) + '_' + str(index), shape=[_out_dim], trainable=True,
                                    initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
                    _out = tf.matmul(_in, self._mat) + _bias
                else:
                    _out = tf.matmul(_in, self._mat)

                if index < len(self._dims) - 1:
                    if self._relu_mid:
                        _out = tf.nn.relu(_out)
                    if self._dropout_mid is not None:
                        _out = tf.nn.dropout(_out, 1 - self._dropout_mid)
                elif index == len(self._dims) - 1:
                    if self._relu_out:
                        _out = tf.nn.relu(_out)
                    if self._dropout_out is not None:
                        _out = tf.nn.dropout(_out, 1 - self._dropout_out)
                if self._batch_norm:
                    _out = tf.cond(self._phase, lambda: tf.contrib.layers.batch_norm(_out, fused=True, decay=0.95, center=True, scale=True, is_training=True, reuse=False,scope="bn_"+str(index)), lambda: tf.contrib.layers.batch_norm(_out, fused=True, decay=0.95, center=True, scale=True, is_training=False, reuse=True,scope="bn_"+str(index)))
                _in = _out
            self._outputs.append(_out)

            if self._l2_reg is not None:
                for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=var_scope.name):
                    self._loss += self._l2_reg * tf.nn.l2_loss(var)
