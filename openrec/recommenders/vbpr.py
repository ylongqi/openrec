from openrec.recommenders import BPR
from openrec.modules.extractions import LatentFactor, MultiLayerFC
import tensorflow as tf


def VBPR(batch_size, dim_user_embed, dim_item_embed, dim_v, total_users, total_items, l2_reg_embed=None,
    l2_reg_mlp=None, init_model_dir=None, save_model_dir='Recommender/', train=True, serve=False):
    
    rec = BPR(batch_size=batch_size, 
              dim_user_embed=dim_user_embed, 
              dim_item_embed=dim_item_embed, 
              total_users=total_users, 
              total_items=total_items, 
              l2_reg=l2_reg_embed,
              init_model_dir=init_model_dir, 
              save_model_dir=save_model_dir, 
              train=train, serve=serve)
    
    t = rec.traingraph
    s = rec.servegraph
    
    @t.inputgraph.extend(outs=['p_item_vfeature', 'n_item_vfeature'])
    def train_item_visual_features(subgraph):
        subgraph['p_item_vfeature'] = tf.placeholder(tf.float32, shape=[batch_size, dim_v], name='p_item_vfeature')
        subgraph['n_item_vfeature'] = tf.placeholder(tf.float32, shape=[batch_size, dim_v], name='n_item_vfeature')
        subgraph.update_global_input_mapping({'p_item_vfeature': subgraph['p_item_vfeature'],
                            'n_item_vfeature': subgraph['n_item_vfeature']})
    
    @s.inputgraph.extend(outs=['item_vfeature'])
    def serving_item_visual_features(subgraph):
        subgraph['item_vfeature'] = tf.placeholder(tf.float32, shape=[None, dim_v], name='item_vfeature')
        subgraph.update_global_input_mapping({'item_vfeature': subgraph['item_vfeature']})
    
    @t.itemgraph.extend(ins=['p_item_vfeature', 'n_item_vfeature'])
    def train_add_item_graph(subgraph):
        p_item_vout = MultiLayerFC(in_tensor=subgraph['p_item_vfeature'], l2_reg=l2_reg_mlp, subgraph=subgraph,
                                        dims=[dim_user_embed-dim_item_embed], scope='item_MLP')
        n_item_vout = MultiLayerFC(in_tensor=subgraph['n_item_vfeature'], l2_reg=l2_reg_mlp, subgraph=subgraph,
                                        dims=[dim_user_embed-dim_item_embed], scope='item_MLP')
        subgraph['p_item_vec'] = tf.concat([subgraph['p_item_vec'], p_item_vout], axis=1)
        subgraph['n_item_vec'] = tf.concat([subgraph['n_item_vec'], n_item_vout], axis=1)
        
    @s.itemgraph.extend(ins=['item_vfeature'])
    def serving_add_item_graph(subgraph):
        item_vout = MultiLayerFC(in_tensor=subgraph['item_vfeature'], l2_reg=l2_reg_mlp, subgraph=subgraph,
                                        dims=[dim_user_embed-dim_item_embed], scope='item_MLP')
        subgraph['item_vec'] = tf.concat([subgraph['item_vec'], item_vout], axis=1)
    
    @t.connector.extend
    def train_connect(graph):
        graph.itemgraph['p_item_vfeature'] = graph.inputgraph['p_item_vfeature']
        graph.itemgraph['n_item_vfeature'] = graph.inputgraph['n_item_vfeature']
    
    @s.connector.extend
    def serve_connect(graph):
        graph.itemgraph['item_vfeature'] = graph.inputgraph['item_vfeature']
    
    return rec