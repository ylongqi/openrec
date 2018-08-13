from openrec.recommenders import Recommender
from openrec.modules.extractions import LatentFactor
from openrec.modules.interactions import PairwiseLog
import tensorflow as tf

def BPR(batch_size, dim_embed, max_user, max_item, l2_reg=None,
    init_model_dir=None, save_model_dir='Recommender/', training=True, serving=False):
    
    rec = Recommender(init_model_dir=init_model_dir, save_model_dir=save_model_dir, 
                    training=training, serving=serving)
    
    @rec.TrainingGraph.InputGraph(['user_id', 'p_item_id', 'n_item_id'])
    def training_input_graph(subgraph):
        user_id = tf.placeholder(tf.int32, shape=[batch_size], name='user_id')
        p_item_id = tf.placeholder(tf.int32, shape=[batch_size], name='p_item_id')
        n_item_id = tf.placeholder(tf.int32, shape=[batch_size], name='n_item_id')
        
        subgraph.set('user_id', user_id)
        subgraph.set('p_item_id', p_item_id)
        subgraph.set('n_item_id', n_item_id)
        subgraph.super.register_input_mapping({'user_id': user_id,
                            'p_item_id': p_item_id,
                            'n_item_id': n_item_id})

    @rec.ServingGraph.InputGraph(['user_id', 'item_id'])
    def serving_input_graph(subgraph):
        user_id = tf.placeholder(tf.int32, shape=[None], name='user_id')
        item_id = tf.placeholder(tf.int32, shape=[None], name='item_id')
        
        subgraph.set('user_id', user_id)
        subgraph.set('item_id', item_id)
        subgraph.super.register_input_mapping({'user_id': user_id,
                                'item_id': item_id})

    @rec.TrainingGraph.UserGraph(['user_vec'])
    @rec.ServingGraph.UserGraph(['user_vec'])
    def user_graph(subgraph):
        user_id = subgraph.super.InputGraph.get('user_id')
        _, user_vec = LatentFactor(l2_reg=l2_reg, init='normal', ids=user_id,
                    shape=[max_user, dim_embed], scope='user')
        subgraph.set('user_vec', user_vec)

    @rec.TrainingGraph.ItemGraph(['p_item_vec', 'p_item_bias', 'n_item_vec', 'n_item_bias'])
    def training_item_graph(subgraph):
        p_item_id = subgraph.super.InputGraph.get('p_item_id')
        n_item_id = subgraph.super.InputGraph.get('n_item_id')
        _, p_item_vec = LatentFactor(l2_reg=l2_reg, init='normal', ids=p_item_id,
                    shape=[max_item, dim_embed], subgraph=subgraph, scope='item')
        _, p_item_bias = LatentFactor(l2_reg=l2_reg, init='zero', ids=p_item_id,
                    shape=[max_item, 1], subgraph=subgraph, scope='item_bias')
        _, n_item_vec = LatentFactor(l2_reg=l2_reg, init='normal', ids=n_item_id,
                    shape=[max_item, dim_embed], subgraph=subgraph, scope='item')
        _, n_item_bias = LatentFactor(l2_reg=l2_reg, init='zero', ids=n_item_id,
                    shape=[max_item, 1], subgraph=subgraph, scope='item_bias')
        subgraph.set('p_item_vec', p_item_vec)
        subgraph.set('p_item_bias', p_item_bias)
        subgraph.set('n_item_vec', n_item_vec)
        subgraph.set('n_item_bias', n_item_bias)
        
    @rec.ServingGraph.ItemGraph(['item_vec', 'item_bias'])
    def serving_item_graph(subgraph):
        item_id = subgraph.super.InputGraph.get('item_id')
        _, item_vec = LatentFactor(l2_reg=l2_reg, init='normal', ids=item_id,
                    shape=[max_item, dim_embed], subgraph=subgraph, scope='item')
        _, item_bias = LatentFactor(l2_reg=l2_reg, init='zero', ids=item_id,
                    shape=[max_item, 1], subgraph=subgraph, scope='item_bias')
        subgraph.set('item_vec', item_vec)
        subgraph.set('item_bias', item_bias)

    @rec.TrainingGraph.InteractionGraph([])
    def interaction_graph(subgraph):
        user_vec = subgraph.super.UserGraph.get('user_vec')
        p_item_vec = subgraph.super.ItemGraph.get('p_item_vec')
        p_item_bias = subgraph.super.ItemGraph.get('p_item_bias')
        n_item_vec = subgraph.super.ItemGraph.get('n_item_vec')
        n_item_bias = subgraph.super.ItemGraph.get('n_item_bias')
        PairwiseLog(user_vec=user_vec, p_item_vec=p_item_vec, n_item_vec=n_item_vec, 
                    p_item_bias=p_item_bias, n_item_bias=n_item_bias, 
                    subgraph=subgraph, train=True, scope='PairwiseLog')

    @rec.TrainingGraph.OptimizerGraph([])
    def optimizer_graph(subgraph):
        losses = tf.add_n(subgraph.super.get_losses())
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        subgraph.super.register_train_op(optimizer.minimize(losses))

    @rec.ServingGraph.InteractionGraph([])
    def serving_interaction_graph(subgraph):
        user_vec = subgraph.super.UserGraph.get('user_vec')
        item_vec = subgraph.super.ItemGraph.get('item_vec')
        item_bias = subgraph.super.ItemGraph.get('item_bias')
        PairwiseLog(user_vec=user_vec, item_vec=item_vec,
                    item_bias=item_bias, train=False, 
                     subgraph=subgraph, scope='PairwiseLog')
    
    return rec
