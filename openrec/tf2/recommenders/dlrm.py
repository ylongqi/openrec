import sys
import tensorflow as tf
from tensorflow.keras import Model
from openrec.tf2.modules import LatentFactor, SecondOrderFeatureInteraction, MLP

class DLRM(Model):
    
    def __init__(
        self, 
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op='dot',
        arch_interaction_itself=False,
        sigmoid_bot=False,
        sigmoid_top=True,
        loss_func='mse',
        loss_threshold=0.0):
        
        '''
        m_spa: the dimensionality of sparse feature embeddings
        ln_emb: the size of sparse feature embeddings (num_instances)
        ln_bot: the size of the bottom MLP
        ln_top: the size of the top MLP
        '''
        
        super(DLRM, self).__init__()
        
        self._loss_threshold = loss_threshold
        self._loss_func = loss_func
        self._latent_factors = [LatentFactor(num_instances=num, 
                                             dim=m_spa) for num in ln_emb]
        self._mlp_bot = MLP(units_list=ln_bot, 
                            out_activation='sigmoid' if sigmoid_bot else 'relu')
        self._mlp_top = MLP(units_list=ln_top, 
                            out_activation='sigmoid' if sigmoid_top else 'relu')
        
        self._dot_interaction = None
        if arch_interaction_op == 'dot':
            self._dot_interaction = SecondOrderFeatureInteraction(
                                        self_interaction=arch_interaction_itself
                                    )
        
        elif self._arch_interaction_op != 'cat':
            sys.exit(
                "ERROR: arch_interaction_op="
                + self._arch_interaction_op
                + " is not supported"
            )
        
        if loss_func == 'mse':
            self._loss = tf.keras.losses.MeanSquaredError()
        elif loss_func == 'bce':
            self._loss = tf.keras.losses.BinaryCrossentropy()
        else:
            sys.exit(
                "ERROR: loss_func="
                + loss_func
                + " is not supported"
            )
        
    def call(self, dense_features, sparse_features, label):
        
        '''
        dense_features shape: [batch_size, num of dense features]
        sparse_features shape: [batch_size, num_of_sparse_features]
        label shape: [batch_size]
        '''
        
        prediction = self.inference(dense_features, sparse_features)
        loss = self._loss(y_true=label, 
                          y_pred=prediction)
        return loss
        
    def inference(self, dense_features, sparse_features):
    
        '''
        dense_features shape: [batch_size, num of dense features]
        sparse_features shape: [num_of_sparse_features, batch_size]
        '''
        
        sparse_emb_vecs = list(map(lambda pair: pair[1](pair[0]), 
                                      zip(tf.unstack(sparse_features, axis=1), 
                                          self._latent_factors)))
        
        dense_emb_vec = self._mlp_bot(dense_features)
        
        if self._dot_interaction is not None:
            prediction = self._mlp_top(tf.concat([dense_emb_vec, 
                                              self._dot_interaction(sparse_emb_vecs + [dense_emb_vec])],
                                             axis=1))
        else:
            prediction = self._mlp_top(tf.concat(sparse_emb_vecs + [dense_emb_vec], 
                                             axis=1))
        
        if 0.0 < self._loss_threshold and self._loss_threshold < 1.0:
            prediction = tf.clip_by_value(prediction, self._loss_threshold, 1.0 - self._loss_threshold)
        
        return tf.reshape(prediction, [-1])
