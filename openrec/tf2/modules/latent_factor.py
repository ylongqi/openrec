from tensorflow.keras.layers import Embedding

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