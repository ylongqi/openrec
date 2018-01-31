from __future__ import print_function
from termcolor import colored
from openrec.recommenders import Recommender
from openrec.modules.extractions import LatentFactor
from openrec.modules.interactions import PointwiseMSE
from openrec.modules.extractions import TemporalLatentFactor

class ItrMLP(Recommender):

    def __init__(self, batch_size, dim_embed, max_user, max_item,
                    pretrained_user_embeddings, pretrained_item_embeddings, 
                     user_dims, item_dims, test_batch_size=None, 
                     l2_reg=None, opt='SGD', sess_config=None):

        self._dim_embed = dim_embed
        self._pretrained_user_embeddings = pretrained_user_embeddings
        self._pretrained_item_embeddings = pretrained_item_embeddings
        self._user_dims = user_dims
        self._item_dims = item_dims

        super(ItrMLP, self).__init__(batch_size=batch_size, 
                                  test_batch_size=test_batch_size,
                                  max_user=max_user, 
                                  max_item=max_item, 
                                  l2_reg=l2_reg,
                                  opt=opt, sess_config=sess_config)
    
    def _initialize(self, init_dict):
        
        super(ItrMLP, self)._initialize(init_dict=init_dict)
        print(colored('[Pretrain user MLP into identity]', 'blue'))
        self._get_module('user_vec').pretrain_mlp_as_identity(self._sess)
        print(colored('[Pretrain item MLP into identity]', 'blue'))
        self._get_module('item_vec').pretrain_mlp_as_identity(self._sess)

    def update_embeddings(self):

        self._get_module('user_vec').forward_update_embeddings(self._sess)
        self._get_module('item_vec').forward_update_embeddings(self._sess)

    def _input_mappings(self, batch_data, train):

        if train:
            return {self._get_input('user_id'): batch_data['user_id_input'],
                    self._get_input('item_id'): batch_data['item_id_input'],
                    self._get_input('labels'): batch_data['labels']}
        else:
            return {self._get_input('user_id', train=False): batch_data['user_id_input'],
                   self._get_input('item_id', train=False): batch_data['item_id_input']}

    def _build_user_inputs(self, train=True):
        
        if train:
            self._add_input(name='user_id', dtype='int32', shape=[self._batch_size])
        else:
            self._add_input(name='user_id', dtype='int32', shape=[self._test_batch_size], train=False)
        
    def _build_item_inputs(self, train=True):
        
        if train:
            self._add_input(name='item_id', dtype='int32', shape=[self._batch_size])
        else:
            self._add_input(name='item_id', dtype='int32', shape=[self._test_batch_size], train=False)
    
    def _build_extra_inputs(self, train=True):
        
        if train:
            self._add_input(name='labels', dtype='float32', shape=[self._batch_size])
        else:
            self._add_input(name='labels', dtype='none', train=False)

    def _build_user_extractions(self, train=True):

        self._add_module('user_vec',
                         TemporalLatentFactor(l2_reg=self._l2_reg, init=self._pretrained_user_embeddings, 
                                              ids=self._get_input('user_id', train=train), mlp_dims=self._user_dims,
                                              shape=[self._max_user, self._dim_embed], train=train,
                                              scope='user', reuse=not train),
                         train=train)

    def _build_item_extractions(self, train=True):
        
        self._add_module('item_vec',
                          TemporalLatentFactor(l2_reg=self._l2_reg, init=self._pretrained_item_embeddings, 
                                              ids=self._get_input('item_id', train=train), mlp_dims=self._item_dims,
                                              shape=[self._max_item, self._dim_embed], train=train,
                                              scope='item', reuse=not train),
                          train=train)
        self._add_module('item_bias',
                          LatentFactor(l2_reg=self._l2_reg, init='zero', ids=self._get_input('item_id', train=train),
                                      shape=[self._max_item, 1], scope='item_bias', reuse=not train),
                          train=train)

    def _build_default_interactions(self, train=True):

          self._add_module('interaction',
                          PointwiseMSE(user=self._get_module('user_vec', train=train).get_outputs()[0], 
                                      item=self._get_module('item_vec', train=train).get_outputs()[0],
                                      item_bias=self._get_module('item_bias', train=train).get_outputs()[0], 
                                      labels=self._get_input('labels', train=train), a=1.0, b=1.0, sigmoid=True,
                                      train=train, scope='PointwiseMSE', reuse=not train),
                          train=train)

    def _build_serving_graph(self):

        super(ItrMLP, self)._build_serving_graph()
        self._scores = self._get_module('interaction', train=False).get_outputs()[0]