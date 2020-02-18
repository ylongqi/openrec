from openrec.tf1.recommenders import VanillaYouTubeRec
from openrec.tf1.modules.extractions import LatentFactor, MultiLayerFC
from openrec.tf1.modules.interactions import MLPSoftmax
import tensorflow as tf

def YouTubeRec(batch_size, user_dict, item_dict, dim_user_embed, dim_item_embed, max_seq_len, l2_reg_embed=None, l2_reg_mlp=None, dropout=None, init_model_dir=None, save_model_dir='YouTubeRec/', train=True, serve=False):

    rec = VanillaYouTubeRec(batch_size=batch_size,
                            dim_item_embed=dim_item_embed['id'],
                            max_seq_len=max_seq_len,
                            total_items=item_dict['id'],
                            l2_reg_embed=l2_reg_embed,
                            l2_reg_mlp=l2_reg_mlp,
                            dropout=dropout,
                            init_model_dir=init_model_dir,
                            save_model_dir=save_model_dir, 
                            train=train, serve=serve)


    @rec.traingraph.inputgraph.extend(outs=['user_geo', 'user_gender']) 
    def add_feature(subgraph):
        subgraph['user_gender'] = tf.placeholder(tf.int32, shape=[batch_size], name='user_gender')
        subgraph['user_geo'] = tf.placeholder(tf.int32, shape=[batch_size], name='user_geo')
        
        subgraph.update_global_input_mapping({'user_gender': subgraph['user_gender'],
                                              'user_geo': subgraph['user_geo']})


    @rec.servegraph.inputgraph.extend(outs=['user_gender', 'user_geo'])
    def add_feature(subgraph):
        subgraph['user_gender'] = tf.placeholder(tf.int32, shape=[None], name='user_gender')
        subgraph['user_geo'] = tf.placeholder(tf.int32, shape=[None], name='user_geo')
        
        subgraph.update_global_input_mapping({'user_gender': subgraph['user_gender'],
                                              'user_geo': subgraph['user_geo']})

    
    @rec.traingraph.usergraph(ins=['user_geo', 'user_gender'], outs=['user_vec'])
    @rec.servegraph.usergraph(ins=['user_geo', 'user_gender'], outs=['user_vec'])
    def user_graph(subgraph):
        _, user_gender = LatentFactor(l2_reg=l2_reg_embed,
                                      shape=[user_dict['gender'], 
                                      dim_user_embed['gender']],
                                      id_=subgraph['user_gender'],
                                      subgraph=subgraph,
                                      init='normal',
                                      scope='user_gender')

        _, user_geo = LatentFactor(l2_reg=l2_reg_embed,
                                   shape=[user_dict['geo'], dim_user_embed['geo']],
                                   id_=subgraph['user_geo'],
                                   subgraph=subgraph,
                                   init='normal',
                                   scope='user_geo')
        subgraph['user_vec'] = tf.concat([user_gender, user_geo], axis=1)


    @rec.traingraph.interactiongraph(ins=['user_vec', 'seq_item_vec', 'seq_len', 'label'])
    def train_interaction_graph(subgraph):
        MLPSoftmax(user=subgraph['user_vec'],
                   item=subgraph['seq_item_vec'], 
                   seq_len=subgraph['seq_len'],
                   max_seq_len=max_seq_len,
                   dims=[dim_user_embed['total'] + dim_item_embed['total'], item_dict['id']], 
                   l2_reg=l2_reg_mlp, 
                   labels=subgraph['label'],
                   dropout=dropout, 
                   train=True, 
                   subgraph=subgraph,
                   scope='MLPSoftmax')


    @rec.servegraph.interactiongraph(ins=['user_vec', 'seq_item_vec', 'seq_len'])
    def serve_interaction_graph(subgraph):
        MLPSoftmax(user=subgraph['user_vec'],
                   item=subgraph['seq_item_vec'], 
                   seq_len=subgraph['seq_len'],
                   max_seq_len=max_seq_len,
                   dims=[dim_user_embed['total'] + dim_item_embed['total'], item_dict['id']], 
                   l2_reg=l2_reg_mlp, 
                   train=False, 
                   subgraph=subgraph,
                   scope='MLPSoftmax')


    @rec.traingraph.connector.extend
    @rec.servegraph.connector.extend
    def add_user_connect(graph):
        graph.usergraph['user_geo'] = graph.inputgraph['user_geo']
        graph.usergraph['user_gender'] = graph.inputgraph['user_gender']
        graph.interactiongraph['user_vec'] = graph.usergraph['user_vec']


    return rec
