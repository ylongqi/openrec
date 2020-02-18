from tensorflow.keras.layers import Layer
import tensorflow as tf

class SecondOrderFeatureInteraction(Layer):
    
    def __init__(self, self_interaction=False):
        
        self._self_interaction = self_interaction
        
        super(SecondOrderFeatureInteraction, self).__init__()
    
    def call(self, inputs):
        
        '''
        inputs: list of features with shape [batch_size, num_features, feature_dim] or [batch_size, feature_dim]
        '''
        
        batch_size = tf.shape(inputs[0])[0]
        
        def _expand_tensor(tensor):
            if tf.shape(tensor) == 3:
                return tensor
            else:
                return tf.expand_dims(tensor, 1)
            
        expand_inputs = list(map(_expand_tensor, inputs))
        concat_features = tf.concat(expand_inputs, axis=1)
        dot_products = tf.linalg.LinearOperatorLowerTriangular(tf.matmul(concat_features, concat_features, transpose_b=True)).to_dense()
        ones = tf.ones_like(dot_products)
        mask = tf.linalg.band_part(ones, 0, -1)
        
        if not self._self_interaction:
            mask = mask - tf.linalg.band_part(ones, 0, 0)
            out_dim = int(len(inputs) * (len(inputs)-1) / 2)
        else:
            out_dim = int(len(inputs) * (len(inputs)+1) / 2)
        
        flat_interactions = tf.reshape(tf.boolean_mask(dot_products, mask), (batch_size, out_dim))
            
        return flat_interactions