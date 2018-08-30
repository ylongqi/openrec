import numpy as np
import tensorflow as tf
from termcolor import colored
from openrec.recommenders import PMF
from openrec.modules.extractions import LatentFactor
import math

def _FastDotProductRecommender(batch_size, dim_embed, total_users, total_items):
    
    rec = PMF(batch_size=batch_size, 
              dim_user_embed=dim_embed, 
              dim_item_embed=dim_embed, 
              total_users=total_users, 
              total_items=total_items,
              init_model_dir=None, 
              save_model_dir=None, 
              train=False, serve=True)
    s = rec.servegraph
    
    @s.inputgraph.extend(outs=['user_lf_cache', 'item_lf_cache', 'item_bias_cache'])
    def _add_input(subgraph):
        subgraph['user_lf_cache'] = tf.placeholder(tf.float32, shape=[total_users, dim_embed], 
                                                   name='user_lf_cache')
        subgraph['item_lf_cache'] = tf.placeholder(tf.float32, shape=[total_items, dim_embed], 
                                                   name='item_lf_cache')
        subgraph['item_bias_cache'] = tf.placeholder(tf.float32, shape=[total_items, 1], 
                                                     name='item_bias_cache')
        
        subgraph.register_global_input_mapping({'user_lf_cache': subgraph['user_lf_cache'],
                                                'item_lf_cache': subgraph['item_lf_cache'],
                                                'item_bias_cache': subgraph['item_bias_cache']}, 'cache')
    
    @s.usergraph.extend(ins=['user_lf_cache'])
    def _add_usergraph(subgraph):
        user_embedding, _ = LatentFactor(id_=None, shape=[total_users, dim_embed], 
                                         subgraph=subgraph, scope='user')
        subgraph.register_global_operation(tf.assign(user_embedding, subgraph['user_lf_cache']), 'cache')
    
    @s.itemgraph.extend(ins=['item_lf_cache', 'item_bias_cache'])
    def _add_itemgraph(subgraph):
        item_embedding, _ = LatentFactor(id_=None, shape=[total_items, dim_embed], 
                                         subgraph=subgraph, scope='item')
        item_bias_embedding, _ = LatentFactor(id_=None, shape=[total_items, 1], 
                                              subgraph=subgraph, scope='item_bias')
        subgraph.register_global_operation(tf.assign(item_embedding, subgraph['item_lf_cache']), 'cache')
        subgraph.register_global_operation(tf.assign(item_bias_embedding, subgraph['item_bias_cache']), 'cache')
    
    @s.connector.extend
    def _add_connect(graph):
        graph.usergraph['user_lf_cache'] = graph.inputgraph['user_lf_cache']
        graph.itemgraph['item_lf_cache'] = graph.inputgraph['item_lf_cache']
        graph.itemgraph['item_bias_cache'] = graph.inputgraph['item_bias_cache']
        
    return rec
    
    
class FastDotProductServer(object):
    
    
    def __init__(self, model, batch_size, total_users, total_items, dim_embed,  
                extract_user_lf_func=None, extract_item_lf_func=None, extract_item_bias_func=None):
        
        assert extract_user_lf_func is not None, "Function for user embedding extraction not specified."
        assert extract_item_lf_func is not None, "Function for item embedding extraction not specified."
        
        self._model = model
        self._fastmodel = _FastDotProductRecommender(batch_size=batch_size,
                                                    dim_embed=dim_embed,
                                                    total_users=total_users,
                                                    total_items=total_items)
        self._extract_user_lf_func = extract_user_lf_func
        self._extract_item_lf_func = extract_item_lf_func
        self._extract_item_bias_func = extract_item_bias_func
        
        self._user_lf_cache = np.zeros((total_users, dim_embed), dtype=np.float32)
        self._item_lf_cache = np.zeros((total_items, dim_embed), dtype=np.float32)
        self._item_bias_cache = np.zeros((total_items, 1), dtype=np.float32)
        
        self._batch_size = batch_size
        self._total_users = total_users
        self._total_items = total_items
        self._model_updated = False
        
    def _cache(self):
        
        # print(colored('..Caching', 'yellow'), end='')
        num_batch = math.ceil(float(self._total_users) / self._batch_size)
        for batch_id in range(num_batch):
            input_batch = np.arange(batch_id*self._batch_size, min((batch_id+1)*self._batch_size, self._total_users))
            self._user_lf_cache[input_batch] = self._extract_user_lf_func(self._model, input_batch)
        # print(colored('..user embedding', 'yellow'), end='')
        
        num_batch = math.ceil(float(self._total_items) / self._batch_size)
        for batch_id in range(num_batch):
            input_batch = np.arange(batch_id*self._batch_size, min((batch_id+1)*self._batch_size, self._total_items))
            self._item_lf_cache[input_batch] = self._extract_item_lf_func(self._model, input_batch)
        
        if self._extract_item_bias_func is not None:
            # print(colored('..item embedding', 'yellow'), end='')
            num_batch = math.ceil(float(self._total_items) / self._batch_size)
            for batch_id in range(num_batch):
                input_batch = np.arange(batch_id*self._batch_size, min((batch_id+1)*self._batch_size, self._total_items))
                self._item_bias_cache[input_batch] = self._extract_item_bias_func(self._model, input_batch).reshape((-1, 1))
            # print(colored('..item bias', 'yellow'), end='\t')
        # else:
            # print(colored('..item embedding', 'yellow'), end='\t')
        
        batch_data = {'user_lf_cache': self._user_lf_cache,
                     'item_lf_cache': self._item_lf_cache,
                     'item_bias_cache': self._item_bias_cache}
        
        self._fastmodel.serve(batch_data=batch_data,
                             input_mapping_id='cache', 
                              operations_id='cache', 
                              losses_id=None, 
                              outputs_id=None)
        
    def build(self):
        
        self._model.build()
        self._fastmodel.build()
            
    def train(self, batch_data, input_mapping_id='default', operations_id='default', losses_id='default', outputs_id='default'):
        
        self._model_updated = True
        return self._model.train(batch_data=batch_data,
                                input_mapping_id=input_mapping_id,
                                operations_id=operations_id,
                                losses_id=losses_id,
                                outputs_id=outputs_id)
        
    def serve(self, batch_data, input_mapping_id='default', operations_id='default', losses_id='default', outputs_id='default'):
        
        if self._model_updated:
            self._cache()
            self._model_updated = False
        return self._fastmodel.serve(batch_data=batch_data,
                                    input_mapping_id=input_mapping_id,
                                    operations_id=operations_id,
                                    losses_id=losses_id,
                                    outputs_id=outputs_id)
    
    def save(self, save_model_dir=None, global_step=None):
        
        self._model.save(save_model_dir=save_model_dir, 
                         global_step=global_step)
        
    def isbuilt(self):
        
        return self._model.isbuilt()