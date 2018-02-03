from __future__ import print_function
import tensorflow as tf
from termcolor import colored
from openrec.modules.extractions import Extraction


class MultiLayerFC(Extraction):

    """
    The MultiLayerFC module implements multi-layer perceptrons with ReLU as non-linear activation \
    functions. Each layer is often referred as a *fully-connected layer*.

    Parameters
    ----------
    in_tensor: Tensorflow tensor
        An input tensor with shape **[*, feature dimensionality]**
    dims: list
        Specify the *feature size* of each layer's outputs. For example, setting **dims=[512, 258, 128]** to create \
        three fully-connected layers with output shape **[*, 512]**, **[*, 256]**, and **[*, 128]**, respectively.
    relu_in: bool, optional
        Whether or not to add ReLU to the input tensor.
    relu_mid: bool, optional
        Whether or not to add ReLU to the outputs of intermediate layers.
    relu_out: bool, optional
        Whether or not to add ReLU to the final output tensor.
    dropout_in: float, optional
        Dropout rate for the input tensor. If *None*, no dropout is used for the input tensor.
    dropout_mid: float, optional
        Dropout rate for the outputs of intermediate layers. If *None*, no dropout is used for the intermediate outputs.
    dropout_out: float, optional
        Dropout rate for the outputs of the final layer. If *None*, no dropout is used for the final outputs.
    bias_in: bool, optional
        Whether or not to add bias to the input tensor.
    bias_mid: bool, optional
        Whether or not to add bias to the outputs of intermediate layers.
    bias_out: bool, optional
        Whether or not to add bias to the final output tensor.
    batch_norm: bool, optional
        Whether or not to add batch normalization [1]_ to each layer's outputs.
    train: bool, optionl
        An indicator for training or servining phase.
    l2_reg: float, optional
        Weight for L2 regularization, i.e., weight decay.
    scope: str, optional
        Scope for module variables.
    reuse: bool, optional
        Whether or not to reuse module variables.
    
    References
    ----------
    .. [1] Ioffe, S. and Szegedy, C., 2015, June. Batch normalization: Accelerating deep network training by reducing internal \
        covariate shift. In International Conference on Machine Learning (pp. 448-456).
    """

    def __init__(self, in_tensor, dims,
                relu_in=False, relu_mid=True, relu_out=False,
                dropout_in=None, dropout_mid=None, dropout_out=None, 
                bias_in=True, bias_mid=True, bias_out=True, batch_norm=False,
                train=True, l2_reg=None, scope=None, reuse=False):

        assert dims is not None, 'dims cannot be None'
        assert in_tensor is not None, 'in_tensor cannot be None'
        self._in_tensor = in_tensor
        self._dims = dims
        self._relu_in = relu_in
        self._relu_mid = relu_mid
        self._relu_out = relu_out

        if train:
            self._dropout_in = dropout_in
            self._dropout_mid = dropout_mid
            self._dropout_out = dropout_out
        else:
            self._dropout_in = None
            self._dropout_mid = None
            self._dropout_out = None
        
        self._batch_norm = batch_norm
        self._bias_in = bias_in
        self._bias_mid = bias_mid
        self._bias_out = bias_out

        super(MultiLayerFC, self).__init__(train=train, l2_reg=l2_reg, scope=scope, reuse=reuse)

    def _build_shared_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse) as var_scope:
            _in = self._in_tensor

            if self._relu_in:
                _in = tf.nn.relu(_in)

            if self._dropout_in is not None:
                _in = tf.nn.dropout(_in, 1 - self._dropout_in) # Tensorflow uses keep_prob
                
            for index, _out_dim in enumerate(self._dims):
                self._mat = tf.get_variable('FC_' + '_' + str(index), shape=[_in.shape[1], _out_dim], trainable=True,
                                initializer=tf.contrib.layers.xavier_initializer())
                if index == 0:
                    add_bias = self._bias_in
                elif index == len(self._dims) - 1:
                    add_bias = self._bias_out
                else:
                    add_bias = self._bias_mid

                if add_bias:
                    _bias = tf.get_variable('bias_' + '_' + str(index), shape=[_out_dim], trainable=True,
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
                    _out = tf.contrib.layers.batch_norm(_out, fused=True, decay=0.95, 
                                        center=True, scale=True, is_training=self._train, 
                                        reuse=False, scope="bn_"+str(index), updates_collections=None)
                _in = _out
            self._outputs.append(_out)

            if self._l2_reg is not None:
                for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=var_scope.name):
                    self._loss += self._l2_reg * tf.nn.l2_loss(var)
