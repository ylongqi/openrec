from openrec.recommenders import Recommender
from openrec.modules.extractions import LatentFactor
from openrec.modules.interactions import PointwiseMSE
import tensorflow as tf

def PMF(batch_size, dim_embed, max_user, max_item, a=1.0, b=0.01, l2_reg=None,
    init_model_dir=None, save_model_dir='Recommender/', training=True, serving=False):

    rec = Recommender(init_model_dir=init_model_dir, save_model_dir=save_model_dir, 
                    training=training, serving=serving)

    @rec.TrainingGraph.InputGraph(['user_id', 'item_id', 'labels'])
    def training_input_graph(subgraph):
        user_id_input = tf.placeholder(tf.int32, shape=[batch_size], name='user_id')
        item_id_input = tf.placeholder(tf.int32, shape=[batch_size], name='item_id')
        labels = tf.placeholder(tf.float32, shape=[batch_size], name='labels')
        subgraph.set('user_id', user_id_input)
        subgraph.set('item_id', item_id_input)
        subgraph.set('labels', labels)
        subgraph.super.register_input_mapping({'user_id': user_id_input,
                            'item_id': item_id_input,
                            'labels': labels})

    @rec.ServingGraph.InputGraph(['user_id', 'item_id'])
    def serving_input_graph(subgraph):
        user_id_input = tf.placeholder(tf.int32, shape=[None], name='user_id')
        item_id_input = tf.placeholder(tf.int32, shape=[None], name='item_id')
        subgraph.set('user_id', user_id_input)
        subgraph.set('item_id', item_id_input)
        subgraph.super.register_input_mapping({'user_id': user_id_input,
                                'item_id': item_id_input})

    @rec.TrainingGraph.UserGraph(['user_vec'])
    @rec.ServingGraph.UserGraph(['user_vec'])
    def user_graph(subgraph):
        user_id_input = subgraph.super.InputGraph.get('user_id')
        _, user_vec = LatentFactor(l2_reg=l2_reg, init='normal', ids=user_id_input,
                    shape=[max_user, dim_embed], subgraph=subgraph, scope='user')
        subgraph.set('user_vec', user_vec)

    @rec.TrainingGraph.ItemGraph(['item_vec', 'item_bias'])
    @rec.ServingGraph.ItemGraph(['item_vec', 'item_bias'])
    def item_graph(subgraph):
        item_id_input = subgraph.super.InputGraph.get('item_id')
        _, item_vec = LatentFactor(l2_reg=l2_reg, init='normal', ids=item_id_input,
                    shape=[max_item, dim_embed], subgraph=subgraph, scope='item')
        _, item_bias = LatentFactor(l2_reg=l2_reg, init='zero', ids=item_id_input,
                    shape=[max_item, 1], subgraph=subgraph, scope='item_bias')
        subgraph.set('item_vec', item_vec)
        subgraph.set('item_bias', item_bias)

    @rec.TrainingGraph.InteractionGraph([])
    def interaction_graph(subgraph):
        user_vec = subgraph.super.UserGraph.get('user_vec')
        item_vec = subgraph.super.ItemGraph.get('item_vec')
        item_bias = subgraph.super.ItemGraph.get('item_bias')
        labels = subgraph.super.InputGraph.get('labels')
        PointwiseMSE(user_vec=user_vec, item_vec=item_vec,
                    item_bias=item_bias, labels=labels, 
                    a=a, b=b, sigmoid=False,
                    train=True, subgraph=subgraph, scope='PointwiseMSE')

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

        PointwiseMSE(user_vec=user_vec, item_vec=item_vec,
                    item_bias=item_bias, a=a, b=b, sigmoid=False,
                    train=False, subgraph=subgraph, scope='PointwiseMSE')
    
    return rec