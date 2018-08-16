import tensorflow as tf
from openrec.recommenders import BPR
from openrec.modules.interactions import PairwiseEuDist
from openrec.modules.extractions import LatentFactor

def UCML(batch_size, dim_embed, total_users, total_items, l2_reg=None,
    init_model_dir=None, save_model_dir='Recommender/', training=True, serving=False):
    
    rec = BPR(batch_size=batch_size, 
              dim_embed=dim_embed, 
              total_users=total_users, 
              total_items=total_items, 
              l2_reg=l2_reg,
              init_model_dir=init_model_dir, 
              save_model_dir=save_model_dir, 
              training=training, 
              serving=serving)
    
    def censor_vec(embedding, censor_id):
        unique_censor_id, _ = tf.unique(censor_id)
        embedding_gather = tf.gather(embedding, indices=unique_censor_id)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_gather), axis=1, keepdims=True))
        return tf.scatter_update(embedding, indices=unique_censor_id, updates=embedding_gather / tf.maximum(norm, 1.0))
        
    @rec.TrainingGraph.UserGraph.add([])
    def censor_user_vec(subgraph):
        user_id = subgraph.super.InputGraph.get('user_id')
        user_embedding, _ = LatentFactor(l2_reg=l2_reg, 
                                         init='normal', 
                                         id_=None,
                                         shape=[total_users, dim_embed], 
                                         scope='user')
        user_censor_ops = censor_vec(user_embedding, user_id)
        subgraph.super.register_train_op(user_censor_ops, 'censor_embedding')
    
    @rec.TrainingGraph.ItemGraph.add([])
    def censor_item_vec(subgraph):
        IG = subgraph.super.InputGraph
        item_embedding, _ = LatentFactor(l2_reg=l2_reg, 
                                         init='normal', 
                                         id_=None,
                                         shape=[total_items, dim_embed], 
                                         subgraph=subgraph, 
                                         scope='item')
        item_censor_ops = censor_vec(item_embedding, tf.concat([IG.get('p_item_id'), IG.get('n_item_id')], axis=0))
        subgraph.super.register_train_op(item_censor_ops, 'censor_embedding')
    
    @rec.TrainingGraph.InteractionGraph([])
    def interaction_graph(subgraph):
        UG = subgraph.super.UserGraph
        IG = subgraph.super.ItemGraph
        PairwiseEuDist(user_vec=UG.get('user_vec'), 
                       p_item_vec=IG.get('p_item_vec'), 
                       n_item_vec=IG.get('n_item_vec'), 
                       p_item_bias=IG.get('p_item_bias'), 
                       n_item_bias=IG.get('n_item_bias'), 
                       subgraph=subgraph, 
                       train=True, 
                       scope='PairwiseEuDist')
    
    @rec.ServingGraph.InteractionGraph([])
    def serving_interaction_graph(subgraph):
        UG = subgraph.super.UserGraph
        IG = subgraph.super.ItemGraph
        PairwiseEuDist(user_vec=UG.get('user_vec'), 
                       item_vec=IG.get('item_vec'),
                       item_bias=IG.get('item_bias'), 
                       train=False, 
                       subgraph=subgraph, 
                       scope='PairwiseEuDist')
    
    return rec
        