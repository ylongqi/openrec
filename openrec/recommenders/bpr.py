from openrec.recommenders import Recommender
from openrec.modules.extractions import LatentFactor
from openrec.modules.interactions import PairwiseLog
import tensorflow as tf

def BPR(batch_size, dim_embed, total_users, total_items, l2_reg=None,
    init_model_dir=None, save_model_dir='Recommender/', training=True, serving=False):
    
    rec = Recommender(init_model_dir=init_model_dir, save_model_dir=save_model_dir, 
                    training=training, serving=serving)
    
    T = rec.TrainingGraph
    S = rec.ServingGraph
    
    @T.InputGraph(outs=['user_id', 'p_item_id', 'n_item_id'])
    def training_input_graph(subgraph):
        subgraph['user_id'] = tf.placeholder(tf.int32, shape=[batch_size], name='user_id')
        subgraph['p_item_id'] = tf.placeholder(tf.int32, shape=[batch_size], name='p_item_id')
        subgraph['n_item_id'] = tf.placeholder(tf.int32, shape=[batch_size], name='n_item_id')
        
        subgraph.register_global_input_mapping({'user_id': subgraph['user_id'],
                            'p_item_id': subgraph['p_item_id'],
                            'n_item_id': subgraph['n_item_id']})

    @S.InputGraph(outs=['user_id', 'item_id'])
    def serving_input_graph(subgraph):
        subgraph['user_id'] = tf.placeholder(tf.int32, shape=[None], name='user_id')
        subgraph['item_id'] = tf.placeholder(tf.int32, shape=[None], name='item_id')
        
        subgraph.register_global_input_mapping({'user_id': subgraph['user_id'],
                                'item_id': subgraph['item_id']})

    @T.UserGraph(outs=['user_vec'], ins=['user_id'])
    @S.UserGraph(outs=['user_vec'], ins=['user_id'])
    def user_graph(subgraph):
        _, subgraph['user_vec'] = LatentFactor(l2_reg=l2_reg, 
                                   init='normal', 
                                   id_=subgraph['user_id'],
                                   shape=[total_users, dim_embed], 
                                   scope='user')

    @T.ItemGraph(outs=['p_item_vec', 'p_item_bias', 'n_item_vec', 'n_item_bias'],
                ins=['p_item_id', 'n_item_id'])
    def training_item_graph(subgraph):
        _, subgraph['p_item_vec'] = LatentFactor(l2_reg=l2_reg, init='normal', id_=subgraph['p_item_id'],
                    shape=[total_items, dim_embed], subgraph=subgraph, scope='item')
        _, subgraph['p_item_bias'] = LatentFactor(l2_reg=l2_reg, init='zero', id_=subgraph['p_item_id'],
                    shape=[total_items, 1], subgraph=subgraph, scope='item_bias')
        _, subgraph['n_item_vec'] = LatentFactor(l2_reg=l2_reg, init='normal', id_=subgraph['n_item_id'],
                    shape=[total_items, dim_embed], subgraph=subgraph, scope='item')
        _, subgraph['n_item_bias'] = LatentFactor(l2_reg=l2_reg, init='zero', id_=subgraph['n_item_id'],
                    shape=[total_items, 1], subgraph=subgraph, scope='item_bias')
        
    @S.ItemGraph(outs=['item_vec', 'item_bias'], ins=['item_id'])
    def serving_item_graph(subgraph):
        _, subgraph['item_vec'] = LatentFactor(l2_reg=l2_reg, init='normal', id_=subgraph['item_id'],
                    shape=[total_items, dim_embed], subgraph=subgraph, scope='item')
        _, subgraph['item_bias'] = LatentFactor(l2_reg=l2_reg, init='zero', id_=subgraph['item_id'],
                    shape=[total_items, 1], subgraph=subgraph, scope='item_bias')

    @T.InteractionGraph(ins=['user_vec', 'p_item_vec', 'p_item_bias', 'n_item_vec', 'n_item_bias'])
    def interaction_graph(subgraph):
        PairwiseLog(user_vec=subgraph['user_vec'], 
                    p_item_vec=subgraph['p_item_vec'], 
                    n_item_vec=subgraph['n_item_vec'], 
                    p_item_bias=subgraph['p_item_bias'], 
                    n_item_bias=subgraph['n_item_bias'], 
                    subgraph=subgraph, train=True, scope='PairwiseLog')

    @T.OptimizerGraph()
    def optimizer_graph(subgraph):
        losses = tf.add_n(subgraph.get_global_losses())
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        subgraph.super.register_train_op(optimizer.minimize(losses))

    @S.InteractionGraph(ins=['user_vec', 'item_vec', 'item_bias'])
    def serving_interaction_graph(subgraph):
        PairwiseLog(user_vec=subgraph['user_vec'], 
                    item_vec=subgraph['item_vec'],
                    item_bias=subgraph['item_bias'], 
                    train=False, subgraph=subgraph, scope='PairwiseLog')
    
    # Connecting Training Graph
    T.UserGraph['user_id'] = T.InputGraph['user_id']
    T.ItemGraph['p_item_id'] = T.InputGraph['p_item_id']
    T.ItemGraph['n_item_id'] = T.InputGraph['n_item_id']
    T.InteractionGraph['user_vec'] = T.UserGraph['user_vec']
    T.InteractionGraph['p_item_vec'] = T.ItemGraph['p_item_vec']
    T.InteractionGraph['n_item_vec'] = T.ItemGraph['n_item_vec']
    T.InteractionGraph['p_item_bias'] = T.ItemGraph['p_item_bias']
    T.InteractionGraph['n_item_bias'] = T.ItemGraph['n_item_bias']
    
    # Connecting Serving Graph
    S.UserGraph['user_id'] = S.InputGraph['user_id']
    S.ItemGraph['item_id'] = S.InputGraph['item_id']
    S.InteractionGraph['user_vec'] = S.UserGraph['user_vec']
    S.InteractionGraph['item_vec'] = S.ItemGraph['item_vec']
    S.InteractionGraph['item_bias'] = S.ItemGraph['item_bias']
    
    return rec