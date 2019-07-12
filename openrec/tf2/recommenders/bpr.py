import tensorflow as tf
from tensorflow.keras import Model
from openrec.tf2.modules import LatentFactor, PairwiseLogLoss

class BPR(Model):
    
    def __init__(self, dim_user_embed, dim_item_embed, total_users, total_items):
        
        super(BPR, self).__init__()
        self.user_latent_factor = LatentFactor(num_instances=total_users, 
                                                dim=dim_user_embed, 
                                                name='user_latent_factor')
        self.item_latent_factor = LatentFactor(num_instances=total_items, 
                                                dim=dim_item_embed, 
                                                name='item_latent_factor')
        self.item_bias = LatentFactor(num_instances=total_items, 
                                       dim=1, 
                                       name='item_bias')
        self.pairwise_log_loss = PairwiseLogLoss()
    
    def call(self, user_id, p_item_id, n_item_id):
        
        loss = self.pairwise_log_loss(user_vec=self.user_latent_factor(user_id),
                                      p_item_vec=self.item_latent_factor(p_item_id),
                                      p_item_bias=self.item_bias(p_item_id),
                                      n_item_vec=self.item_latent_factor(n_item_id),
                                      n_item_bias=self.item_bias(n_item_id))
        return loss
    
    def inference(self, user_id):
        
        user_vec = self.user_latent_factor(user_id)
        return tf.linalg.matmul(user_vec, self.item_latent_factor.variables[0], transpose_b=True) + \
            tf.reshape(self.item_bias.variables[0], [-1])