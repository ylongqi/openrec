from tensorflow.keras.layers import Embedding
import tensorflow as tf

class LatentFactor(Embedding):
    
    def __init__(self, num_instances, dim, zero_init=False, name=None):
        
        if zero_init:
            initializer = 'zeros'
        else:
            initializer = 'uniform'
        super(LatentFactor, self).__init__(input_dim=num_instances, 
                                           output_dim=dim, 
                                           embeddings_initializer=initializer,
                                           name=name)
    
    def censor(self, censor_id):
        
        unique_censor_id, _ = tf.unique(censor_id)
        embedding_gather = tf.gather(self.variables[0], indices=unique_censor_id)
        norm = tf.norm(embedding_gather, axis=1, keepdims=True)
        return self.variables[0].scatter_nd_update(indices=tf.expand_dims(unique_censor_id, 1), 
                                                   updates=embedding_gather / tf.math.maximum(norm, 0.1))