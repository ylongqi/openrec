import tensorflow as tf
import numpy as np
from tqdm import tqdm
from openrec.modules.extractions import Extraction
from openrec.modules.extractions import MultiLayerFC

class TemporalLatentFactor(Extraction):
    
    def __init__(self, shape, mlp_dims, ids, init='normal', mlp_pretrain=True, l2_reg=None, train=True, scope=None, reuse=False):
        
        if type(init) == str:
            if init == 'normal':
                self._initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)
            elif init == 'zero':
                self._initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
        else:
            self._initializer = tf.constant_initializer(value=init, dtype=tf.float32)
        
        self._shape = shape
        self._ids = ids
        self._mlp_dims = mlp_dims
        self._mlp_pretrain = mlp_pretrain

        super(TemporalLatentFactor, self).__init__(train=train, l2_reg=l2_reg, scope=scope, reuse=reuse)
    
    def _build_shared_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            
            self._embedding = tf.get_variable('embedding', dtype=tf.float32, shape=self._shape, trainable=False,
                                      initializer=self._initializer)
            
            self._flag = tf.get_variable('flag', dtype=tf.bool, shape=[self._shape[0]], trainable=False,
                                    initializer=tf.constant_initializer(value=False, dtype=tf.bool))
            unique_ids, _ = tf.unique(self._ids)

            with tf.control_dependencies([tf.scatter_update(self._flag, unique_ids, 
                                                            tf.ones_like(unique_ids, dtype=tf.bool))]):
                trans_embedding = MultiLayerFC(in_tensor=tf.nn.embedding_lookup(self._embedding, self._ids), 
                                               dims=self._mlp_dims, 
                                               batch_norm=True, 
                                               scope=self._scope+'/MLP', 
                                               train=self._train,
                                               reuse=self._reuse,
                                               l2_reg=self._l2_reg,
                                               relu_out=True)

            self._outputs += trans_embedding.get_outputs()
            self._loss += trans_embedding.get_loss()
            
            update_ids = tf.reshape(tf.where(self._flag), [-1])
            update_embedding = MultiLayerFC(in_tensor=tf.nn.embedding_lookup(self._embedding, update_ids), 
                                           dims=self._mlp_dims, 
                                           batch_norm=True, 
                                           scope=self._scope+'/MLP', 
                                           train=False,
                                           reuse=True,
                                           l2_reg=self._l2_reg,
                                           relu_out=True)
            self._update_node = tf.scatter_update(self._embedding, update_ids, update_embedding.get_outputs()[0])
            self._clear_flag = tf.scatter_update(self._flag, update_ids, tf.zeros_like(update_ids, dtype=tf.bool))
            
    def _build_training_graph(self):
        
        with tf.variable_scope(self._scope, reuse=self._reuse):
            
            if self._mlp_pretrain:
                self._pretrain_input = tf.placeholder(tf.float32, shape=(32, self._shape[1]), name='pretrain_input')
                trans_embedding = MultiLayerFC(in_tensor=self._pretrain_input, 
                                               dims=self._mlp_dims, 
                                               batch_norm=True, 
                                               scope=self._scope+'/MLP', 
                                               train=True,
                                               reuse=True,
                                               l2_reg=self._l2_reg,
                                               relu_out=True)
                identity_loss = tf.nn.l2_loss(trans_embedding.get_outputs()[0] - self._pretrain_input)
                self._pretrain_ops = tf.train.AdamOptimizer(learning_rate=0.001).minimize(identity_loss)
    
    def pretrain_mlp_as_identity(self, sess):
        
        for i in tqdm(range(20000)):
            input_npy = np.random.uniform(-0.5, 0.5, (32, self._shape[1]))
            sess.run(self._pretrain_ops, feed_dict={self._pretrain_input: input_npy})
            
    def forward_update_embeddings(self, sess):
        
        """ Retrieve update node.
        """
        sess.run(self._update_node)
        sess.run(self._clear_flag)
