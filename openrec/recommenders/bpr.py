from openrec.recommenders import Recommender
from openrec.modules.extractions import LatentFactor
from openrec.modules.interactions import PairwiseLog
import tensorflow as tf

def BPR(batch_size, dim_user_embed, dim_item_embed, total_users, total_items, l2_reg=None,
    init_model_dir=None, save_model_dir='Recommender/', train=True, serve=False):
    
    rec = Recommender(init_model_dir=init_model_dir, save_model_dir=save_model_dir, 
                    train=train, serve=serve)
    
    t = rec.traingraph
    s = rec.servegraph
    
    @t.inputgraph(outs=['user_id', 'p_item_id', 'n_item_id'])
    def train_input_graph(subgraph):
        subgraph['user_id'] = tf.placeholder(tf.int32, shape=[batch_size], name='user_id')
        subgraph['p_item_id'] = tf.placeholder(tf.int32, shape=[batch_size], name='p_item_id')
        subgraph['n_item_id'] = tf.placeholder(tf.int32, shape=[batch_size], name='n_item_id')
        
        subgraph.register_global_input_mapping({'user_id': subgraph['user_id'],
                            'p_item_id': subgraph['p_item_id'],
                            'n_item_id': subgraph['n_item_id']})

    @s.inputgraph(outs=['user_id', 'item_id'])
    def serve_input_graph(subgraph):
        subgraph['user_id'] = tf.placeholder(tf.int32, shape=[None], name='user_id')
        subgraph['item_id'] = tf.placeholder(tf.int32, shape=[None], name='item_id')
        
        subgraph.register_global_input_mapping({'user_id': subgraph['user_id'],
                                'item_id': subgraph['item_id']})

    @t.usergraph(outs=['user_vec'], ins=['user_id'])
    @s.usergraph(outs=['user_vec'], ins=['user_id'])
    def user_graph(subgraph):
        _, subgraph['user_vec'] = LatentFactor(l2_reg=l2_reg, 
                                   init='normal', 
                                   id_=subgraph['user_id'],
                                   shape=[total_users, dim_user_embed], 
                                    subgraph=subgraph,
                                   scope='user')

    @t.itemgraph(outs=['p_item_vec', 'p_item_bias', 'n_item_vec', 'n_item_bias'],
                ins=['p_item_id', 'n_item_id'])
    def train_item_graph(subgraph):
        _, subgraph['p_item_vec'] = LatentFactor(l2_reg=l2_reg, init='normal', id_=subgraph['p_item_id'],
                    shape=[total_items, dim_item_embed], subgraph=subgraph, scope='item')
        _, subgraph['p_item_bias'] = LatentFactor(l2_reg=l2_reg, init='zero', id_=subgraph['p_item_id'],
                    shape=[total_items, 1], subgraph=subgraph, scope='item_bias')
        _, subgraph['n_item_vec'] = LatentFactor(l2_reg=l2_reg, init='normal', id_=subgraph['n_item_id'],
                    shape=[total_items, dim_item_embed], subgraph=subgraph, scope='item')
        _, subgraph['n_item_bias'] = LatentFactor(l2_reg=l2_reg, init='zero', id_=subgraph['n_item_id'],
                    shape=[total_items, 1], subgraph=subgraph, scope='item_bias')
        
    @s.itemgraph(outs=['item_vec', 'item_bias'], ins=['item_id'])
    def serve_item_graph(subgraph):
        _, subgraph['item_vec'] = LatentFactor(l2_reg=l2_reg, init='normal', id_=subgraph['item_id'],
                    shape=[total_items, dim_item_embed], subgraph=subgraph, scope='item')
        _, subgraph['item_bias'] = LatentFactor(l2_reg=l2_reg, init='zero', id_=subgraph['item_id'],
                    shape=[total_items, 1], subgraph=subgraph, scope='item_bias')

    @t.interactiongraph(ins=['user_vec', 'p_item_vec', 'p_item_bias', 'n_item_vec', 'n_item_bias'])
    def interaction_graph(subgraph):
        PairwiseLog(user_vec=subgraph['user_vec'], 
                    p_item_vec=subgraph['p_item_vec'], 
                    n_item_vec=subgraph['n_item_vec'], 
                    p_item_bias=subgraph['p_item_bias'], 
                    n_item_bias=subgraph['n_item_bias'], 
                    subgraph=subgraph, train=True, scope='PairwiseLog')

    @t.optimizergraph
    def optimizer_graph(subgraph):
        losses = tf.add_n(subgraph.get_global_losses())
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        subgraph.register_global_operation(optimizer.minimize(losses))

    @s.interactiongraph(ins=['user_vec', 'item_vec', 'item_bias'])
    def serve_interaction_graph(subgraph):
        PairwiseLog(user_vec=subgraph['user_vec'], 
                    item_vec=subgraph['item_vec'],
                    item_bias=subgraph['item_bias'], 
                    train=False, subgraph=subgraph, scope='PairwiseLog')
    
    @t.connector
    def train_connect(graph):
        graph.usergraph['user_id'] = graph.inputgraph['user_id']
        graph.itemgraph['p_item_id'] = graph.inputgraph['p_item_id']
        graph.itemgraph['n_item_id'] = graph.inputgraph['n_item_id']
        graph.interactiongraph['user_vec'] = graph.usergraph['user_vec']
        graph.interactiongraph['p_item_vec'] = graph.itemgraph['p_item_vec']
        graph.interactiongraph['n_item_vec'] = graph.itemgraph['n_item_vec']
        graph.interactiongraph['p_item_bias'] = graph.itemgraph['p_item_bias']
        graph.interactiongraph['n_item_bias'] = graph.itemgraph['n_item_bias']
    
    @s.connector
    def serve_connect(graph):
        graph.usergraph['user_id'] = graph.inputgraph['user_id']
        graph.itemgraph['item_id'] = graph.inputgraph['item_id']
        graph.interactiongraph['user_vec'] = graph.usergraph['user_vec']
        graph.interactiongraph['item_vec'] = graph.itemgraph['item_vec']
        graph.interactiongraph['item_bias'] = graph.itemgraph['item_bias']
    
    return rec