from openrec.recommenders import Recommender
from openrec.modules.extractions import LatentFactor, MultiLayerFC
from openrec.modules.interactions import MLPSoftmax
import tensorflow as tf

def YouTubeRec(batch_size, user_dict, item_dict, dim_user_embed, dim_item_embed, max_seq_len, l2_reg_embed=None, l2_reg_mlp=None, dropout=None, init_model_dir=None, save_model_dir='DRN/', train=True, serve=False):

    rec = Recommender(init_model_dir=init_model_dir, 
                      save_model_dir=save_model_dir, train=train, serve=serve)

    @rec.traingraph.inputgraph(outs=['seq_item_id', 'seq_len', 'label'])
    def train_input_graph(subgraph):
        subgraph['seq_item_id'] = tf.placeholder(tf.int32, shape=[batch_size, max_seq_len],
                                                 name='seq_item_id')
        subgraph['seq_len'] = tf.placeholder(tf.int32, shape=[batch_size],
                                             name='seq_len')
        subgraph['label'] = tf.placeholder(tf.int32, shape=[batch_size],
                                           name='label')
        subgraph.register_global_input_mapping({'seq_item_id': subgraph['seq_item_id'],
                                               'seq_len': subgraph['seq_len'],
                                               'label': subgraph['label']})
    


    @rec.servegraph.inputgraph(outs=['seq_item_id', 'seq_len'])
    def serve_input_graph(subgraph):
        subgraph['seq_item_id'] = tf.placeholder(tf.int32, shape=[None, max_seq_len],
                                                 name='seq_item_id')
        subgraph['seq_len'] = tf.placeholder(tf.int32, shape=[None],
                                             name='seq_len')
        subgraph.register_global_input_mapping({'seq_item_id': subgraph['seq_item_id'],
                                               'seq_len': subgraph['seq_len']})


    @rec.traingraph.inputgraph.extend(outs=['user_geo', 'user_gender']) 
    def add_feature(subgraph):
        subgraph['user_gender'] = tf.placeholder(tf.int32, shape=[batch_size], name='user_gender')
        subgraph['user_geo'] = tf.placeholder(tf.int32, shape=[batch_size], name='user_geo')
        
        subgraph.update_global_input_mapping({'user_gender': subgraph['user_gender'],
                                              'user_geo': subgraph['user_geo']
                                             })


    @rec.servegraph.inputgraph.extend(outs=['user_gender', 'user_geo'])
    def add_feature(subgraph):
        subgraph['user_gender'] = tf.placeholder(tf.int32, shape=[None], name='user_gender')
        subgraph['user_geo'] = tf.placeholder(tf.int32, shape=[None], name='user_geo')
        
        subgraph.update_global_input_mapping({'user_gender': subgraph['user_gender'],
                                              'user_geo': subgraph['user_geo']
                                             })

    
    @rec.traingraph.usergraph(ins=['user_geo', 'user_gender'], outs=['user_vec'])
    @rec.servegraph.usergraph(ins=['user_geo', 'user_gender'], outs=['user_vec'])
    def user_graph(subgraph):
        _, user_gender = LatentFactor(shape=[user_dict['gender'], dim_user_embed['gender']],
                                      id_=subgraph['user_gender'],
                                      subgraph=subgraph,
                                      init='normal',
                                      scope='user_gender')

        _, user_geo = LatentFactor(shape=[user_dict['geo'], dim_user_embed['geo']],
                                   id_=subgraph['user_geo'],
                                   subgraph=subgraph,
                                   init='normal',
                                   scope='user_geo')
        subgraph['user_vec'] = tf.concat([user_gender, user_geo], axis=1)


    @rec.traingraph.itemgraph(ins=['seq_item_id', 'seq_len'], outs=['seq_vec'])
    @rec.servegraph.itemgraph(ins=['seq_item_id', 'seq_len'], outs=['seq_vec'])
    def item_graph(subgraph):
        _, subgraph['seq_vec']= LatentFactor(init='normal',
                                       id_=subgraph['seq_item_id'],
                                       shape=[item_dict['id'], dim_item_embed['id']],
                                       subgraph=subgraph,
                                       scope='item')



    @rec.traingraph.interactiongraph(ins=['user_vec', 'seq_vec', 'seq_len', 'label'])
    def train_interaction_graph(subgraph):
        MLPSoftmax(user=subgraph['user_vec'],
                   item=subgraph['seq_vec'], 
                   seq_len=subgraph['seq_len'],
                   max_seq_len=max_seq_len,
                   dims=[dim_user_embed['total'] + dim_item_embed['total'], item_dict['id']], 
                   l2_reg=l2_reg_mlp, 
                   labels=subgraph['label'],
                   dropout=dropout, 
                   train=True, 
                   subgraph=subgraph,
                   scope='MLPSoftmax' 
                  )


    @rec.servegraph.interactiongraph(ins=['user_vec', 'seq_vec', 'seq_len'])
    def serve_interaction_graph(subgraph):
        MLPSoftmax(user=subgraph['user_vec'],
                   item=subgraph['seq_vec'], 
                   seq_len=subgraph['seq_len'],
                   max_seq_len=max_seq_len,
                   dims=[dim_user_embed['total'] + dim_item_embed['total'], item_dict['id']], 
                   l2_reg=l2_reg_mlp, 
                   train=False, 
                   subgraph=subgraph,
                   scope='MLPSoftmax' 
                   )

    @rec.traingraph.optimizergraph
    def optimizer_graph(subgraph):
        losses = tf.add_n(subgraph.get_global_losses())
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        subgraph.register_global_operation(optimizer.minimize(losses))
    

    @rec.traingraph.connector
    @rec.servegraph.connector
    def connect(graph):
        graph.itemgraph['seq_item_id'] = graph.inputgraph['seq_item_id']
        graph.itemgraph['seq_len'] = graph.inputgraph['seq_len']
        graph.usergraph['user_geo'] = graph.inputgraph['user_geo']
        graph.usergraph['user_gender'] = graph.inputgraph['user_gender']
        graph.interactiongraph['seq_len'] = graph.inputgraph['seq_len']
        graph.interactiongraph['seq_vec'] = graph.itemgraph['seq_vec']
        graph.interactiongraph['user_vec'] = graph.usergraph['user_vec']


    @rec.traingraph.connector.extend
    def train_connect(graph):
        graph.interactiongraph['label'] = graph.inputgraph['label']
    

    return rec
