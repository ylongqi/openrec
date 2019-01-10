from __future__ import print_function
import tensorflow as tf
from termcolor import colored

def PointwiseMSE(user_vec, item_vec, item_bias, subgraph, 
                 label=None, a=1.0, b=1.0, sigmoid=False,
                train=True, serve_mode='pairwise', scope='PointwiseMSE'):
    
    if train or serve_mode == 'pairwise':
        dot_user_item = tf.reduce_sum(tf.multiply(user_vec, item_vec),
                                  axis=1, keepdims=False, name="dot_user_item")
    else:
        dot_user_item = tf.matmul(user_vec, item_vec, transpose_b=True)
        
    if sigmoid:
        prediction = tf.sigmoid(dot_user_item + tf.reshape(item_bias, [-1]))
    else:
        prediction = dot_user_item + tf.reshape(item_bias, [-1])

    if train:
        label_weight = (a - b) * label + b
        subgraph.register_global_loss(tf.reduce_sum(label_weight * tf.square(label - prediction)))
    subgraph.register_global_output(prediction)