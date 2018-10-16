from openrec.recommenders import Recommender
from openrec.modules.extractions import LatentFactor
from openrec.modules.interactions import RNNSoftmax
import tensorflow as tf

def RNNRec(batch_size, dim_item_embed, max_seq_len, total_items, num_units, l2_reg=None,
    init_model_dir=None, save_model_dir='Recommender/', train=True, serve=False):

    rec = Recommender(init_model_dir=init_model_dir, save_model_dir=save_model_dir, 
                    train=train, serve=serve)
    
    @rec.traingraph.inputgraph(outs=['seq_item_id', 'seq_len', 'label'])
    def f(subgraph):
        subgraph['seq_item_id'] = tf.placeholder(tf.int32, shape=[batch_size, max_seq_len], name='seq_item_id')
        subgraph['seq_len'] = tf.placeholder(tf.int32, shape=[batch_size], name='seq_len')
        subgraph['label'] = tf.placeholder(tf.int32, shape=[batch_size], name='label')
        subgraph.register_global_input_mapping({'seq_item_id': subgraph['seq_item_id'],
                                               'seq_len': subgraph['seq_len'],
                                               'label': subgraph['label']})
    
    @rec.servegraph.inputgraph(outs=['seq_item_id', 'seq_len'])
    def f(subgraph):
        subgraph['seq_item_id'] = tf.placeholder(tf.int32, shape=[None, max_seq_len], name='seq_item_id')
        subgraph['seq_len'] = tf.placeholder(tf.int32, shape=[None], name='seq_len')
        subgraph.register_global_input_mapping({'seq_item_id': subgraph['seq_item_id'],
                                               'seq_len': subgraph['seq_len']})
    
    @rec.traingraph.itemgraph(outs=['seq_item_vec'], ins=['seq_item_id'])
    @rec.servegraph.itemgraph(outs=['seq_item_vec'], ins=['seq_item_id'])
    def f(subgraph):
        _, subgraph['seq_item_vec'] = LatentFactor(l2_reg=l2_reg, 
                                               init='normal', 
                                               id_=subgraph['seq_item_id'],
                                               shape=[total_items, dim_item_embed], 
                                               subgraph=subgraph, 
                                               scope='item')
    
    @rec.traingraph.interactiongraph(ins=['seq_item_vec', 'seq_len', 'label'])
    def f(subgraph):
        RNNSoftmax(seq_item_vec=subgraph['seq_item_vec'], seq_len=subgraph['seq_len'], 
                   num_units=num_units, total_items=total_items, label=subgraph['label'], train=True, 
                   subgraph=subgraph, scope='RNNSoftmax')
    
    @rec.servegraph.interactiongraph(ins=['seq_item_vec', 'seq_len'])
    def f(subgraph):
        RNNSoftmax(seq_item_vec=subgraph['seq_item_vec'], seq_len=subgraph['seq_len'],
                   num_units=num_units, total_items=total_items, train=False, 
                   subgraph=subgraph, scope='RNNSoftmax')
    
    @rec.traingraph.optimizergraph
    def f(subgraph):
        losses = tf.add_n(subgraph.get_global_losses())
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        subgraph.register_global_operation(optimizer.minimize(losses))
    
    @rec.traingraph.connector
    @rec.servegraph.connector
    def f(graph):
        graph.itemgraph['seq_item_id'] = graph.inputgraph['seq_item_id']
        graph.interactiongraph['seq_item_vec'] = graph.itemgraph['seq_item_vec']
        graph.interactiongraph['seq_len'] = graph.inputgraph['seq_len']
        
    @rec.traingraph.connector.extend
    def f(graph):
        graph.interactiongraph['label'] = graph.inputgraph['label']
        
    return rec