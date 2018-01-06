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
        self._user_vec.pretrain_mlp_as_identity(self._sess)
        print(colored('[Pretrain item MLP into identity]', 'blue'))
        self._item_vec.pretrain_mlp_as_identity(self._sess)

    def update_embeddings(self):

        self._user_vec.forward_update_embeddings(self._sess)
        self._item_vec.forward_update_embeddings(self._sess)

    def _input_mappings(self, batch_data, train):

        if train:
            return {self._user_id_input: batch_data['user_id_input'],
                    self._item_id_input: batch_data['item_id_input'],
                    self._labels: batch_data['labels']}
        else:
            return {self._user_id_serving: batch_data['user_id_input'],
                   self._item_id_serving: batch_data['item_id_input']}

    def _build_user_inputs(self, train=True):
        
        if train:
            self._user_id_input = self._input(dtype='int32', shape=[self._batch_size], name='user_id_input')
        else:
            self._user_id_serving = self._input(dtype='int32', shape=[self._test_batch_size], name='user_id_serving')
        
    def _build_item_inputs(self, train=True):
        
        if train:
            self._item_id_input = self._input(dtype='int32', shape=[self._batch_size], name='item_id_input')
        else:
            self._item_id_serving = self._input(dtype='int32', shape=[self._test_batch_size], name='item_id_serving')
    
    def _build_extra_inputs(self, train=True):
        
        if train:
            self._labels = self._input(dtype='float32', shape=[self._batch_size], name='labels')

    def _build_user_extractions(self, train=True):

        if train:
            self._user_vec = TemporalLatentFactor(l2_reg=self._l2_reg, init=self._pretrained_user_embeddings, 
                                                  ids=self._user_id_input, mlp_dims=self._user_dims,
                                                  shape=[self._max_user, self._dim_embed], train=True,
                                                  scope='user', reuse=False)
            self._loss_nodes += [self._user_vec]
        else:
            self._user_vec_serving = TemporalLatentFactor(l2_reg=self._l2_reg, init=self._pretrained_user_embeddings, 
                                                          ids=self._user_id_serving, mlp_dims=self._user_dims,
                                                          shape=[self._max_user, self._dim_embed], train=False,
                                                          scope='user', reuse=True)

    def _build_item_extractions(self, train=True):
        
        if train:
            self._item_vec = TemporalLatentFactor(l2_reg=self._l2_reg, init=self._pretrained_item_embeddings, 
                                                  ids=self._item_id_input, mlp_dims=self._item_dims,
                                                  shape=[self._max_item, self._dim_embed], train=True,
                                                  scope='item', reuse=False)
            self._item_bias = LatentFactor(l2_reg=self._l2_reg, init='zero', ids=self._item_id_input,
                                    shape=[self._max_item, 1], scope='item_bias', reuse=False)
            self._loss_nodes += [self._item_vec, self._item_bias]
        else:
            self._item_vec_serving = TemporalLatentFactor(l2_reg=self._l2_reg, init=self._pretrained_item_embeddings, 
                                                          ids=self._item_id_serving, mlp_dims=self._item_dims,
                                                          shape=[self._max_item, self._dim_embed], train=False,
                                                          scope='item', reuse=True)
            self._item_bias_serving = LatentFactor(l2_reg=self._l2_reg, init='zero', ids=self._item_id_serving,
                                    shape=[self._max_item, 1], scope='item_bias', reuse=True)

    def _build_default_interactions(self, train=True):

        if train:
            self._interaction_train = PointwiseMSE(user=self._user_vec.get_outputs()[0], 
                                        item=self._item_vec.get_outputs()[0],
                                        item_bias=self._item_bias.get_outputs()[0], 
                                        labels=self._labels, a=1.0, b=1.0, sigmoid=True,
                                        train=True, scope='PointwiseMSE', reuse=False)
            self._loss_nodes.append(self._interaction_train)
        else:
            self._interaction_serve = PointwiseMSE(user=self._user_vec_serving.get_outputs()[0], 
                                        item=self._item_vec_serving.get_outputs()[0],
                                        item_bias=self._item_bias_serving.get_outputs()[0], sigmoid=True, 
                                        train=False, batch_serving=False, scope='PointwiseMSE', reuse=True)

    def _build_serving_graph(self):

        super(ItrMLP, self)._build_serving_graph()
        self._scores = self._interaction_serve.get_outputs()[0]