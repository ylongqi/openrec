import tensorflow as tf
from openrec.recommenders import BPR
from openrec.modules.interactions import PairwiseEuDist
from openrec.modules.extractions import LatentFactor

def UCML(batch_size, dim_user_embed, dim_item_embed, total_users, total_items, l2_reg=None,
    init_model_dir=None, save_model_dir='Recommender/', train=True, serve=False):
    
    rec = BPR(batch_size=batch_size, 
              dim_user_embed=dim_user_embed,
              dim_item_embed=dim_item_embed,
              total_users=total_users, 
              total_items=total_items, 
              l2_reg=l2_reg,
              init_model_dir=init_model_dir, 
              save_model_dir=save_model_dir, 
              train=train, 
              serve=serve)
    
    t = rec.traingraph
    s = rec.servegraph
    
    def censor_vec(embedding, censor_id):
        unique_censor_id, _ = tf.unique(censor_id)
        embedding_gather = tf.gather(embedding, indices=unique_censor_id)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_gather), axis=1, keepdims=True))
        return tf.scatter_update(embedding, indices=unique_censor_id, updates=embedding_gather / tf.maximum(norm, 1.0))
        
    @t.usergraph.extend
    def censor_user_vec(subgraph):
        user_embedding, _ = LatentFactor(l2_reg=None, 
                                         init='normal', 
                                         id_=None,
                                         shape=[total_users, dim_user_embed], 
                                         scope='user')
        user_censor_ops = censor_vec(user_embedding, subgraph['user_id'])
        subgraph.register_global_operation(user_censor_ops, 'censor_embedding')
    
    @t.itemgraph.extend
    def censor_item_vec(subgraph):
        item_embedding, _ = LatentFactor(l2_reg=None, 
                                         init='normal', 
                                         id_=None,
                                         shape=[total_items, dim_item_embed], 
                                         subgraph=subgraph, 
                                         scope='item')
        item_censor_ops = censor_vec(item_embedding, tf.concat([subgraph['p_item_id'], subgraph['n_item_id']], axis=0))
        subgraph.register_global_operation(item_censor_ops, 'censor_embedding')
    
    @t.interactiongraph(ins=['user_vec', 'p_item_vec', 'n_item_vec', 'p_item_bias', 'n_item_bias'])
    def interaction_graph(subgraph):
        PairwiseEuDist(user_vec=subgraph['user_vec'], 
                       p_item_vec=subgraph['p_item_vec'], 
                       n_item_vec=subgraph['n_item_vec'], 
                       p_item_bias=subgraph['p_item_bias'], 
                       n_item_bias=subgraph['n_item_bias'], 
                       subgraph=subgraph, 
                       train=True, 
                       scope='PairwiseEuDist')
    
    @s.interactiongraph(ins=['user_vec', 'item_vec', 'item_bias'])
    def serving_interaction_graph(subgraph):
        PairwiseEuDist(user_vec=subgraph['user_vec'], 
                       item_vec=subgraph['item_vec'],
                       item_bias=subgraph['item_bias'], 
                       train=False, 
                       subgraph=subgraph, 
                       scope='PairwiseEuDist')
    
    return rec
        