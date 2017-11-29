from openrec.recommenders import Recommender
from openrec.modules.extractions import LatentFactor
from openrec.modules.interactions import PointwiseMSE

class PMF(Recommender):

    def __init__(self, batch_size, dim_embed, max_user, max_item,
                    test_batch_size=None, l2_reg=None, opt='SGD', sess_config=None):

        self._dim_embed = dim_embed

        super(PMF, self).__init__(batch_size=batch_size, 
                                  test_batch_size=test_batch_size,
                                  max_user=max_user, 
                                  max_item=max_item, 
                                  l2_reg=l2_reg,
                                  opt=opt, sess_config=sess_config)

    def _input_mappings(self, batch_data, train):

        if train:
            return {self._user_id_input: batch_data['user_id_input'],
                    self._item_id_input: batch_data['item_id_input'],
                    self._labels: batch_data['labels']}
        else:
            return {self._user_id_serving: batch_data['user_id_input']}

    def _build_user_inputs(self, train=True):
        
        if train:
            self._user_id_input = self._input(dtype='int32', shape=[self._batch_size], name='user_id_input')
        else:
            self._user_id_serving = self._input(dtype='int32', shape=[None], name='user_id_serving')
    
    def _build_item_inputs(self, train=True):
        
        if train:
            self._item_id_input = self._input(dtype='int32', shape=[self._batch_size], name='item_id_input')
        else:
            self._item_id_serving = None
    
    def _build_extra_inputs(self, train=True):
        
        if train:
            self._labels = self._input(dtype='float32', shape=[self._batch_size], name='labels')

    def _build_user_extractions(self, train=True):

        if train:
            self._user_vec = LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._user_id_input,
                                    shape=[self._max_user, self._dim_embed], scope='user', reuse=False)
            self._loss_nodes += [self._user_vec]
        else:
            self._user_vec_serving = LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._user_id_serving,
                                    shape=[self._max_user, self._dim_embed], scope='user', reuse=True)
    
    def _build_item_extractions(self, train=True):
        
        if train:
            self._item_vec = LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._item_id_input,
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=False)
            self._item_bias = LatentFactor(l2_reg=self._l2_reg, init='zero', ids=self._item_id_input,
                                    shape=[self._max_item, 1], scope='item_bias', reuse=False)
            self._loss_nodes += [self._item_vec, self._item_bias]
        else:
            self._item_vec_serving = LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._item_id_serving,
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True)
            self._item_bias_serving = LatentFactor(l2_reg=self._l2_reg, init='zero', ids=self._item_id_serving,
                                    shape=[self._max_item, 1], scope='item_bias', reuse=True)

    def _build_default_interactions(self, train=True):

        if train:
            self._interaction_train = PointwiseMSE(user=self._user_vec.get_outputs()[0], 
                                        item=self._item_vec.get_outputs()[0],
                                        item_bias=self._item_bias.get_outputs()[0], 
                                        labels=self._labels, a=1.0, b=1.0, 
                                        train=True, scope='PointwiseMSE', reuse=False)
            self._loss_nodes.append(self._interaction_train)
        else:
            self._interaction_serve = PointwiseMSE(user=self._user_vec_serving.get_outputs()[0], 
                                        item=self._item_vec_serving.get_outputs()[0],
                                        item_bias=self._item_bias_serving.get_outputs()[0], 
                                        train=False, scope='PointwiseMSE', reuse=True)

    def _build_serving_graph(self):

        super(PMF, self)._build_serving_graph()
        self._scores = self._interaction_serve.get_outputs()[0]
