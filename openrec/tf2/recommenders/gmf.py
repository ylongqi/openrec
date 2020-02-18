import tensorflow as tf
from tensorflow.keras import Model
from openrec.tf2.modules import LatentFactor, MLP

class GMF(Model):
    
    def __init__(self, dim_user_embed, dim_item_embed, total_users, total_items):
        
        super(GMF, self).__init__()
        self.user_latent_factor = LatentFactor(num_instances=total_users, 
                                                dim=dim_user_embed, 
                                                name='user_latent_factor')
        self.item_latent_factor = LatentFactor(num_instances=total_items, 
                                                dim=dim_item_embed, 
                                                name='item_latent_factor')
        self.item_bias = LatentFactor(num_instances=total_items, 
                                       dim=1, 
                                       name='item_bias')
        self.mlp = MLP(units_list=[1], use_bias=False)
        self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def call(self, user_id, item_id, label):

        user_vec = self.user_latent_factor(user_id)
        item_vec = self.item_latent_factor(item_id)
        item_bias = self.item_bias(item_id)
        
        logit = tf.reshape(self.mlp(user_vec * item_vec) + item_bias, [-1])
        loss = self._bce(y_true=label, y_pred=logit)
        
        l2_loss = tf.nn.l2_loss(user_vec) + tf.nn.l2_loss(item_vec) \
                + sum([tf.nn.l2_loss(v) for v in self.mlp.trainable_variables])

        return loss, l2_loss
    
    def inference(self, user_id):

        user_vec = self.user_latent_factor(user_id)
        logit = tf.squeeze(self.mlp(tf.expand_dims(user_vec, 1) * self.item_latent_factor.variables[0]), axis=-1) \
                        + tf.reshape(self.item_bias.variables[0], [-1])
        return logit