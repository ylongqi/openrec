from openrec.recommenders import Recommender
from openrec.modules.extractions import LatentFactor
from openrec.modules.interactions import PointwiseMSE
import tensorflow as tf

def PMF(batch_size, dim_embed, total_users, total_items, a=1.0, b=1.0, l2_reg=None,
    init_model_dir=None, save_model_dir='Recommender/', training=True, serving=False):

    rec = Recommender(init_model_dir=init_model_dir, save_model_dir=save_model_dir, 
                    training=training, serving=serving)
    
    T = rec.TrainingGraph
    S = rec.ServingGraph
    
    @T.InputGraph(outs=['user_id', 'item_id', 'label'])
    def training_input_graph(subgraph):
        subgraph['user_id'] = tf.placeholder(tf.int32, shape=[batch_size], name='user_id')
        subgraph['item_id'] = tf.placeholder(tf.int32, shape=[batch_size], name='item_id')
        subgraph['label'] = tf.placeholder(tf.float32, shape=[batch_size], name='label')
        subgraph.register_global_input_mapping({'user_id': subgraph['user_id'],
                            'item_id': subgraph['item_id'],
                            'label': subgraph['label']})
    
    @S.InputGraph(outs=['user_id', 'item_id'])
    def serving_input_graph(subgraph):
        subgraph['user_id'] = tf.placeholder(tf.int32, shape=[None], name='user_id')
        subgraph['item_id'] = tf.placeholder(tf.int32, shape=[None], name='item_id')
        subgraph.register_global_input_mapping({'user_id': subgraph['user_id'],
                                'item_id': subgraph['item_id']})

    @T.UserGraph(ins=['user_id'], outs=['user_vec'])
    @S.UserGraph(ins=['user_id'], outs=['user_vec'])
    def user_graph(subgraph):
        _, subgraph['user_vec'] = LatentFactor(l2_reg=l2_reg, 
                                               init='normal', 
                                               id_=subgraph['user_id'],
                                               shape=[total_users, dim_embed], 
                                               subgraph=subgraph, 
                                               scope='user')
    
    @T.ItemGraph(ins=['item_id'], outs=['item_vec', 'item_bias'])
    @S.ItemGraph(ins=['item_id'], outs=['item_vec', 'item_bias'])
    def item_graph(subgraph):
        _, subgraph['item_vec'] = LatentFactor(l2_reg=l2_reg, init='normal', id_=subgraph['item_id'],
                    shape=[total_items, dim_embed], subgraph=subgraph, scope='item')
        _, subgraph['item_bias'] = LatentFactor(l2_reg=l2_reg, init='zero', id_=subgraph['item_id'],
                    shape=[total_items, 1], subgraph=subgraph, scope='item_bias')
    
    @T.InteractionGraph(ins=['user_vec', 'item_vec', 'item_bias', 'label'])
    def interaction_graph(subgraph):
        PointwiseMSE(user_vec=subgraph['user_vec'], 
                     item_vec=subgraph['item_vec'],
                     item_bias=subgraph['item_bias'], 
                     label=subgraph['label'], 
                    a=a, b=b, sigmoid=False,
                    train=True, subgraph=subgraph, scope='PointwiseMSE')

    @S.InteractionGraph(ins=['user_vec', 'item_vec', 'item_bias'])
    def serving_interaction_graph(subgraph):
        PointwiseMSE(user_vec=subgraph['user_vec'], 
                     item_vec=subgraph['item_vec'],
                     item_bias=subgraph['item_bias'], 
                     a=a, b=b, sigmoid=False,
                    train=False, subgraph=subgraph, scope='PointwiseMSE')
    
    @T.OptimizerGraph()
    def optimizer_graph(subgraph):
        losses = tf.add_n(subgraph.get_global_losses())
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        subgraph.register_global_train_op(optimizer.minimize(losses))
    
    # Connecting Training Graph
    T.UserGraph['user_id'] = T.InputGraph['user_id']
    T.ItemGraph['item_id'] = T.InputGraph['item_id']
    T.InteractionGraph['label'] = T.InputGraph['label']
    T.InteractionGraph['user_vec'] = T.UserGraph['user_vec']
    T.InteractionGraph['item_vec'] = T.ItemGraph['item_vec']
    T.InteractionGraph['item_bias'] = T.ItemGraph['item_bias']
    
    # Connecting Serving Graph
    S.UserGraph['user_id'] = S.InputGraph['user_id']
    S.ItemGraph['item_id'] = S.InputGraph['item_id']
    S.InteractionGraph['user_vec'] = S.UserGraph['user_vec']
    S.InteractionGraph['item_vec'] = S.ItemGraph['item_vec']
    S.InteractionGraph['item_bias'] = S.ItemGraph['item_bias']
    
    return rec