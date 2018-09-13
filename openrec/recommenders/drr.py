from openrec.recommenders import Recommender
from openrec.modules.extractions import LatentFactor, MultiLayerFC
from openrec.modules.interactions import PointwiseMLPCE
import tensorflow as tf

def DRR(batch_size, dim_user_embed, dim_item_embed, total_users, total_items,
        l2_reg_embed=None, l2_reg_mlp=None, dropout=None, init_model_dir=None,
        save_model_dir='DRN/', train=True, serve=False):
    rec = Recommender(init_model_dir=init_model_dir, 
                      save_model_dir=save_model_dir, train=train, serve=serve)

    t = rec.traingraph
    s = rec.servegraph

    @t.inputgraph(outs=['user_id', 'item_id', 'label'])
    def train_input_graph(subgraph):
        subgraph['user_id'] = tf.placeholder(tf.int32, shape=[batch_size],
                                             name='user_id')
        subgraph['item_id'] = tf.placeholder(tf.int32, shape=[batch_size],
                                             name='item_id')
        subgraph['label'] = tf.placeholder(tf.int32, shape=[batch_size],
                                           name='label')
        subgraph.register_global_input_mapping({'user_id': subgraph['user_id'],
                                               'item_id': subgraph['item_id'],
                                               'label': subgraph['label']})

    @s.inputgraph(outs=['user_id', 'item_id'])
    def serve_input_graph(subgraph):
        subgraph['user_id'] = tf.placeholder(tf.int32, shape=[None],
                                             name='user_id')
        subgraph['item_id'] = tf.placeholder(tf.int32, shape=[None],
                                             name='item_id')
        subgraph.register_global_input_mapping({'user_id': subgraph['user_id'],
                                                'item_id': subgraph['item_id']})

    @t.usergraph(ins=['user_id'], outs=['user_vec'])
    @s.usergraph(ins=['user_id'], outs=['user_vec'])
    def user_graph(subgraph):
        _, subgraph['user_vec'] = LatentFactor(l2_reg=l2_reg_embed,
                                               init='normal',
                                               id_=subgraph['user_id'],
                                               shape=[total_users,dim_user_embed],
                                               subgraph=subgraph,
                                               scope='user')
    ##TODO: add user features and concat here, use usergraph.extend


    @t.itemgraph(ins=['item_id'], outs=['item_vec', 'item_bias'])
    @s.itemgraph(ins=['item_id'], outs=['item_vec', 'item_bias'])
    def itemgraph(subgraph):
        _, subgraph['item_vec'] = LatentFactor(l2_reg=l2_reg_embed,
                                               init='normal',
                                               id_=subgraph['item_id'],
                                               shape=[total_items,dim_item_embed],
                                               subgraph=subgraph,
                                               scope='item')
        _, subgraph['item_bias'] = LatentFactor(l2_reg=l2_reg_embed, 
                                                init='zero',
                                                id_=subgraph['item_id'],
                                                shape=[total_items,1],
                                                subgraph=subgraph,
                                                scope='item_bias')


    @t.interactiongraph(ins=['user_vec', 'item_vec', 'item_bias', 'label'])
    def train_interaction_graph(subgraph):
        PointwiseMLPCE(user=subgraph['user_vec'], 
                       item=subgraph['item_vec'], 
                       dims=[dim_user_embed+dim_item_embed, 1], 
                       item_bias=subgraph['item_bias'], 
                       l2_reg=l2_reg_mlp, 
                       labels=subgraph['label'],
                       dropout=dropout, 
                       train=True, 
                       subgraph=subgraph,
                       scope='PointwiseMLPCE' 
                       )

    @s.interactiongraph(ins=['user_vec', 'item_vec', 'item_bias'])
    def serve_interaction_graph(subgraph):
        PointwiseMLPCE(user=subgraph['user_vec'], 
                       item=subgraph['item_vec'], 
                       dims=[dim_user_embed+dim_item_embed, 1], 
                       item_bias=subgraph['item_bias'], 
                       l2_reg=l2_reg_mlp, 
                       train=False, 
                       subgraph=subgraph,
                       scope='PointwiseMLPCE' 
                       )

    @t.optimizergraph
    def optimizer_graph(subgraph):
        losses = tf.add_n(subgraph.get_global_losses())
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        subgraph.register_global_operation(optimizer.minimize(losses))
    
    @t.connector
    def train_connect(graph):
        graph.usergraph['user_id'] = graph.inputgraph['user_id']
        graph.itemgraph['item_id'] = graph.inputgraph['item_id']
        graph.interactiongraph['label'] = graph.inputgraph['label']
        graph.interactiongraph['user_vec'] = graph.usergraph['user_vec']
        graph.interactiongraph['item_vec'] = graph.itemgraph['item_vec']
        graph.interactiongraph['item_bias'] = graph.itemgraph['item_bias']
    
    @s.connector
    def serve_connect(graph):
        graph.usergraph['user_id'] = graph.inputgraph['user_id']
        graph.itemgraph['item_id'] = graph.inputgraph['item_id']
        graph.interactiongraph['user_vec'] = graph.usergraph['user_vec']
        graph.interactiongraph['item_vec'] = graph.itemgraph['item_vec']
        graph.interactiongraph['item_bias'] = graph.itemgraph['item_bias']
    
    return rec
