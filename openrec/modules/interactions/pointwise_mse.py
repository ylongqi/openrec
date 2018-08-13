from __future__ import print_function
import tensorflow as tf
from termcolor import colored

def PointwiseMSE(user_vec, item_vec, item_bias, subgraph, 
                 labels=None, a=1.0, b=1.0, sigmoid=False,
                train=True, scope='PointwiseMSE'):
    
    if train:
        labels_weight = (a - b) * labels + b
        dot_user_item = tf.reduce_sum(tf.multiply(user_vec, item_vec),
                                      axis=1, keepdims=False, name="dot_user_item")

        if sigmoid:
            predictions = tf.sigmoid(dot_user_item + tf.reshape(item_bias, [-1]))
        else:
            predictions = dot_user_item + tf.reshape(item_bias, [-1])

        subgraph.super.register_loss(tf.reduce_sum(labels_weight * tf.square(labels - predictions)))
        subgraph.super.register_output(predictions)
        
    else:
        predictions = tf.matmul(user_vec, item_vec, transpose_b=True) + tf.reshape(item_bias, [-1])
        subgraph.super.register_output(predictions)