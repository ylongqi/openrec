from openrec.tf1.recommenders import Recommender
from openrec.tf1.modules.extractions import LatentFactor, MultiLayerFC
from openrec.tf1.modules.interactions import MLPSoftmax
import tensorflow as tf

def VanillaYouTubeRec(batch_size, dim_item_embed, max_seq_len, total_items, 
                      l2_reg_embed=None, l2_reg_mlp=None, dropout=None, init_model_dir=None, 
                      save_model_dir='Vanilla_YouTubeRec/', train=True, serve=False):

    rec = Recommender(init_model_dir=init_model_dir, save_model_dir=save_model_dir, 
                      train=train, serve=serve)


    @rec.traingraph.inputgraph(outs=['seq_item_id', 'seq_len', 'label'])
    def train_input_graph(subgraph):
        subgraph['seq_item_id'] = tf.placeholder(tf.int32, shape=[batch_size, max_seq_len], name='seq_item_id')
        subgraph['seq_len'] = tf.placeholder(tf.int32, shape=[batch_size], name='seq_len')
        subgraph['label'] = tf.placeholder(tf.int32, shape=[batch_size], name='label')
        subgraph.register_global_input_mapping({'seq_item_id': subgraph['seq_item_id'],
                                                'seq_len': subgraph['seq_len'],
                                                'label': subgraph['label']})


    @rec.servegraph.inputgraph(outs=['seq_item_id', 'seq_len'])
    def serve_input_graph(subgraph):
        subgraph['seq_item_id'] = tf.placeholder(tf.int32, shape=[None, max_seq_len], name='seq_item_id')
        subgraph['seq_len'] = tf.placeholder(tf.int32, shape=[None], name='seq_len')
        subgraph.register_global_input_mapping({'seq_item_id': subgraph['seq_item_id'],
                                               'seq_len': subgraph['seq_len']})


    @rec.traingraph.itemgraph(ins=['seq_item_id', 'seq_len'], outs=['seq_item_vec'])
    @rec.servegraph.itemgraph(ins=['seq_item_id', 'seq_len'], outs=['seq_item_vec'])
    def item_graph(subgraph):
        _, subgraph['seq_item_vec']= LatentFactor(l2_reg=l2_reg_embed,
                                                  init='normal',
                                                  id_=subgraph['seq_item_id'],
                                                  shape=[total_items,dim_item_embed],
                                                  subgraph=subgraph,
                                                  scope='item')

        
    @rec.traingraph.interactiongraph(ins=['seq_item_vec', 'seq_len', 'label'])
    def train_interaction_graph(subgraph):
        MLPSoftmax(user=None,
                   item=subgraph['seq_item_vec'], 
                   seq_len=subgraph['seq_len'],
                   max_seq_len=max_seq_len,
                   dims=[dim_item_embed, total_items], 
                   l2_reg=l2_reg_mlp, 
                   labels=subgraph['label'],
                   dropout=dropout, 
                   train=True, 
                   subgraph=subgraph,
                   scope='MLPSoftmax')


    @rec.servegraph.interactiongraph(ins=['seq_item_vec', 'seq_len'])
    def serve_interaction_graph(subgraph):
        MLPSoftmax(user=None,
                   item=subgraph['seq_item_vec'], 
                   seq_len=subgraph['seq_len'],
                   max_seq_len=max_seq_len,
                   dims=[dim_item_embed, total_items], 
                   l2_reg=l2_reg_mlp, 
                   train=False, 
                   subgraph=subgraph,
                   scope='MLPSoftmax')

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
        graph.interactiongraph['seq_len'] = graph.inputgraph['seq_len']
        graph.interactiongraph['seq_item_vec'] = graph.itemgraph['seq_item_vec']


    @rec.traingraph.connector.extend
    def train_connect(graph):
        graph.interactiongraph['label'] = graph.inputgraph['label']
    
    
    return rec
