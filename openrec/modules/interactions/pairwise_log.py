import tensorflow as tf

def PairwiseLog(user_vec, subgraph, item_vec=None, item_bias=None, p_item_vec=None, 
                p_item_bias=None, n_item_vec=None, n_item_bias=None,
                train=True, scope='PointwiseMSE'):
    
    if train:
        dot_user_pos = tf.reduce_sum(tf.multiply(user_vec, p_item_vec),
                                         reduction_indices=1,
                                         keepdims=True,
                                         name="dot_user_pos")
        dot_user_neg = tf.reduce_sum(tf.multiply(user_vec, n_item_vec),
                                     reduction_indices=1,
                                     keepdims=True,
                                     name="dot_user_neg")
        loss = - tf.reduce_sum(tf.log(tf.sigmoid(tf.maximum(dot_user_pos + p_item_bias -
                                                            dot_user_neg - n_item_bias,
                                                                  -30.0))))
        subgraph.register_global_loss(loss)
    else:
        predictions = tf.reduce_sum(tf.multiply(user_vec, item_vec),
                                    reduction_indices=1,
                                    keepdims=False) + tf.reshape(item_bias, [-1])
        subgraph.register_global_output(predictions)