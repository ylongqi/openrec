import tensorflow as tf
from tensorflow.keras.layers import Layer

class PairwiseLogLoss(Layer):
    
    def __call__(self, user_vec, p_item_vec, n_item_vec, p_item_bias=None, n_item_bias=None):
        
        outputs = super(PairwiseLogLoss, self).__call__((user_vec, 
                                                p_item_vec, 
                                                n_item_vec,
                                                p_item_bias, 
                                                n_item_bias))
        return outputs
    
    def call(self, inputs):
        
        user_vec, p_item_vec, n_item_vec, p_item_bias, n_item_bias = inputs
        
        dot_user_pos = tf.math.reduce_sum(user_vec*p_item_vec,
                                         axis=1,
                                         keepdims=True)
        dot_user_neg = tf.math.reduce_sum(user_vec*n_item_vec,
                                         axis=1,
                                         keepdims=True)
        
        if p_item_bias is not None:
            dot_user_pos += p_item_bias
            
        if n_item_bias is not None:
            dot_user_neg += n_item_bias
            
        loss = -tf.math.reduce_mean(tf.math.log_sigmoid(tf.math.maximum(dot_user_pos-dot_user_neg, -30.0)))
        
        return loss