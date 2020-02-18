import tensorflow as tf
from tensorflow.keras import Model
from openrec.tf2.modules import LatentFactor

class UCML(Model):
    
    def __init__(self, dim_user_embed, dim_item_embed, total_users, total_items, margin=0.5):
        
        super(UCML, self).__init__()
        self.user_latent_factor = LatentFactor(num_instances=total_users, 
                                                dim=dim_user_embed, 
                                                name='user_latent_factor')
        self.item_latent_factor = LatentFactor(num_instances=total_items, 
                                                dim=dim_item_embed, 
                                                name='item_latent_factor')
        self.item_bias = LatentFactor(num_instances=total_items, 
                                       dim=1, 
                                       name='item_bias')
        self.margin = margin
    
    def call(self, user_id, p_item_id, n_item_id):
        
        user_vec = self.user_latent_factor(user_id)
        p_item_vec = self.item_latent_factor(p_item_id)
        p_item_bias = self.item_bias(p_item_id)
        n_item_vec = self.item_latent_factor(n_item_id)
        n_item_bias = self.item_bias(n_item_id)
        
        l2_user_pos = tf.math.reduce_sum(tf.math.square(user_vec - p_item_vec),
                                        axis=-1,
                                        keepdims=True)
        l2_user_neg = tf.math.reduce_sum(tf.math.square(user_vec - n_item_vec),
                                        axis=-1,
                                        keepdims=True)
        pos_score = (-l2_user_pos) + p_item_bias
        neg_score = (-l2_user_neg) + n_item_bias
        diff = pos_score - neg_score
        
        loss = tf.reduce_sum(tf.maximum(self.margin - diff, 0))
        l2_loss = tf.nn.l2_loss(user_vec) + tf.nn.l2_loss(p_item_vec) + tf.nn.l2_loss(n_item_vec)

        return loss, l2_loss
    
    def censor_vec(self, user_id, p_item_id, n_item_id):
        
        return self.user_latent_factor.censor(user_id), \
                self.item_latent_factor.censor(p_item_id), \
                self.item_latent_factor.censor(n_item_id)
    
    def inference(self, user_id):
        
        user_vec = self.user_latent_factor(user_id)
        return -tf.math.reduce_sum(tf.math.square(tf.expand_dims(user_vec, axis=1) - self.item_latent_factor.variables[0]), axis=-1, keepdims=False) + tf.reshape(self.item_bias.variables[0], [-1])