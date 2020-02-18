import tensorflow as tf
from tensorflow.keras import Model
from openrec.tf2.modules import LatentFactor, PointwiseMSELoss

class WRMF(Model):
    
    def __init__(self, dim_user_embed, dim_item_embed, total_users, total_items, a=1.0, b=1.0):
        
        super(WRMF, self).__init__()
        self.user_latent_factor = LatentFactor(num_instances=total_users, 
                                                dim=dim_user_embed, 
                                                name='user_latent_factor')
        self.item_latent_factor = LatentFactor(num_instances=total_items, 
                                                dim=dim_item_embed, 
                                                name='item_latent_factor')
        self.item_bias = LatentFactor(num_instances=total_items, 
                                       dim=1, 
                                       name='item_bias')
        self.pointwise_mse_loss = PointwiseMSELoss(a=a, b=b)
    
    def call(self, user_id, item_id, label):

        user_vec = self.user_latent_factor(user_id)
        item_vec = self.item_latent_factor(item_id)
        item_bias = self.item_bias(item_id)
        
        loss = self.pointwise_mse_loss(user_vec=user_vec,
                                      item_vec=item_vec,
                                      item_bias=item_bias,
                                      label=label)
        
        l2_loss = tf.nn.l2_loss(user_vec) + tf.nn.l2_loss(item_vec)

        return loss, l2_loss
    
    def inference(self, user_id):

        user_vec = self.user_latent_factor(user_id)
        return tf.linalg.matmul(user_vec, self.item_latent_factor.variables[0], transpose_b=True) + \
            tf.reshape(self.item_bias.variables[0], [-1])