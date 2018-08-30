import tensorflow as tf

def PairwiseEuDist(user_vec, subgraph, item_vec=None, item_bias=None, p_item_vec=None, 
                p_item_bias=None, n_item_vec=None, n_item_bias=None,
                train=True, weights=1.0, margin=1.0, scope='PointwiseMSE'):
    
    if train:
        l2_user_pos = tf.reduce_sum(tf.square(tf.subtract(user_vec, p_item_vec)),
                                        reduction_indices=1,
                                        keepdims=True, name="l2_user_pos")
        l2_user_neg = tf.reduce_sum(tf.square(tf.subtract(user_vec, n_item_vec)),
                                    reduction_indices=1,
                                    keepdims=True, name="l2_user_neg")
        pos_score = (-l2_user_pos) + p_item_bias
        neg_score = (-l2_user_neg) + n_item_bias
        diff = pos_score - neg_score
        loss = tf.reduce_sum(weights * tf.maximum(margin - diff, 0))
        subgraph.register_global_loss(loss)
    else:
        predictions = -tf.reduce_sum(tf.square(tf.subtract(user_vec, item_vec)),
                                        reduction_indices=1,
                                        keepdims=True, name="l2_user_pos") + item_bias
        subgraph.register_global_output(predictions)