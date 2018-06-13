from openrec.recommenders import Recommender
from openrec.modules.extractions import LatentFactor
from openrec.modules.interactions import PointwiseMSE
import tensorflow as tf

def PMF(batch_size, dim_embed, max_user, max_item, l2_reg=None,
    init_model_dir=None, save_model_dir='Recommender/', training=True, serving=False):

    r = Recommender(init_model_dir=init_model_dir, save_model_dir=save_model_dir, 
                    training=training, serving=serving)

    @r.T.InputGraph(['user_id', 'item_id', 'labels'])
    def training_input_graph(sg):
        user_id_input = tf.placeholder(tf.int32, shape=[batch_size], name='user_id')
        item_id_input = tf.placeholder(tf.int32, shape=[batch_size], name='item_id')
        labels = tf.placeholder(tf.float32, shape=[batch_size], name='labels')
        sg.set('user_id', user_id_input)
        sg.set('item_id', item_id_input)
        sg.set('labels', labels)
        sg.super.register_input_mapping({'user_id': user_id_input,
                            'item_id': item_id_input,
                            'labels': labels})

    @r.S.InputGraph(['user_id', 'item_id'])
    def serving_input_graph(sg):
        user_id_input = tf.placeholder(tf.int32, shape=[None], name='user_id')
        item_id_input = tf.placeholder(tf.int32, shape=[None], name='item_id')
        sg.set('user_id', user_id_input)
        sg.set('item_id', item_id_input)
        sg.super.register_input_mapping({'user_id': user_id_input,
                                'item_id': item_id_input})

    @r.T.UserGraph(['user_vec'])
    @r.S.UserGraph(['user_vec'])
    def user_graph(sg):
        user_id_input = sg.super.InputGraph.get('user_id')
        _, user_vec = LatentFactor(l2_reg=l2_reg, init='normal', ids=user_id_input,
                    shape=[max_user, dim_embed], scope='user')
        sg.set('user_vec', user_vec)

    @r.T.ItemGraph(['item_vec', 'item_bias'])
    @r.S.ItemGraph(['item_vec', 'item_bias'])
    def item_graph(sg):
        item_id_input = sg.super.InputGraph.get('item_id')
        _, item_vec = LatentFactor(l2_reg=l2_reg, init='normal', ids=item_id_input,
                    shape=[max_item, dim_embed], subgraph=sg, scope='item')
        _, item_bias = LatentFactor(l2_reg=l2_reg, init='zero', ids=item_id_input,
                    shape=[max_item, 1], subgraph=sg, scope='item_bias')
        sg.set('item_vec', item_vec)
        sg.set('item_bias', item_bias)

    @r.T.InteractionGraph([])
    def interaction_graph(sg):
        user_vec = sg.super.UserGraph.get('user_vec')
        item_vec = sg.super.ItemGraph.get('item_vec')
        item_bias = sg.super.ItemGraph.get('item_bias')
        labels = sg.super.InputGraph.get('labels')
        PointwiseMSE(user_vec=user_vec, item_vec=item_vec,
                    item_bias=item_bias, labels=labels, 
                    a=1.0, b=1.0, sigmoid=False,
                    train=True, subgraph=sg, scope='PointwiseMSE')

    @r.T.OptimizerGraph([])
    def optimizer_graph(sg):
        losses = tf.add_n(sg.super.get_losses())
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        sg.super.register_train_op(optimizer.minimize(losses))

    @r.S.InteractionGraph([])
    def serving_interaction_graph(sg):
        user_vec = sg.super.UserGraph.get('user_vec')
        item_vec = sg.super.ItemGraph.get('item_vec')
        item_bias = sg.super.ItemGraph.get('item_bias')

        PointwiseMSE(user_vec=user_vec, item_vec=item_vec,
                    item_bias=item_bias, a=1.0, b=1.0, sigmoid=False,
                    train=False, subgraph=sg, scope='PointwiseMSE')
    
    return r.build()

def InferPMF(dim_embed, max_item, l2_reg=None,
    init_model_dir=None, save_model_dir=None, training=True, serving=False):
    
    r = Recommender(init_model_dir=init_model_dir, save_model_dir=save_model_dir, 
                          training=training, serving=serving)
    
    @r.T.InputGraph(['user_upvote', 'item_id', 'labels'])
    def input_graph(sg):
        user_upvote = tf.placeholder(tf.int32, shape=[None], name='user_upvote')
        item_id = tf.range(max_item, dtype=tf.int32)
        labels = tf.reduce_sum(tf.one_hot(user_upvote, depth=max_item, dtype=tf.float32), axis=0)
        sg.set('user_upvote', user_upvote)
        sg.set('item_id', item_id)
        sg.set('labels', labels)
        sg.super.register_input_mapping({'user_upvote': user_upvote})
    
    @r.T.ItemGraph(['item_vec', 'item_bias'])
    def item_graph(sg):
        item_id_input = sg.super.InputGraph.get('item_id')
        _, item_vec = LatentFactor(l2_reg=l2_reg, init='normal', ids=item_id_input,
                    shape=[max_item, dim_embed], subgraph=sg, scope='item')
        _, item_bias = LatentFactor(l2_reg=l2_reg, init='zero', ids=item_id_input,
                    shape=[max_item, 1], subgraph=sg, scope='item_bias')
        sg.set('item_vec', item_vec)
        sg.set('item_bias', item_bias)
    
    @r.T.UserGraph(['temp_user_embedding', 'user_vec'])
    def user_graph(sg):
        temp_user_embedding, user_vec = LatentFactor(l2_reg=l2_reg, init='normal', 
                                                ids=tf.zeros([max_item], dtype=tf.int32),
                                                shape=[1, dim_embed], subgraph=sg, scope='temp_user')
        sg.set('user_vec', user_vec)
        sg.set('temp_user_embedding', temp_user_embedding)

    @r.T.OptimizerGraph([])
    def optimizer_graph(sg):
        temp_user_embedding = sg.super.UserGraph.get('temp_user_embedding')
        losses = tf.add_n(sg.super.get_losses())
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        sg.super.register_train_op(tf.variables_initializer([temp_user_embedding] + optimizer.variables()), identifier='init')
        sg.super.register_train_op(optimizer.minimize(losses, var_list=[temp_user_embedding]))
        
    @r.T.InteractionGraph([])
    def interaction_graph(sg):
        user_vec = sg.super.UserGraph.get('user_vec')
        item_vec = sg.super.ItemGraph.get('item_vec')
        item_bias = sg.super.ItemGraph.get('item_bias')
        labels = sg.super.InputGraph.get('labels')
        PointwiseMSE(user_vec=user_vec, item_vec=item_vec,
                    item_bias=item_bias, labels=labels, a=1.0, b=0.25, sigmoid=False,
                    train=True, subgraph=sg, scope='PointwiseMSE')
    
    return r.build()