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
            return {self._get_input('user_id'): batch_data['user_id_input'],
                    self._get_input('item_id'): batch_data['item_id_input'],
                    self._get_input('labels'): batch_data['labels']}
        else:
            return {self._get_input('user_id', train=False): batch_data['user_id_input']}

    def _build_user_inputs(self, train=True):
        
        if train:
            self._add_input(name='user_id', dtype='int32', shape=[self._batch_size])
        else:
            self._add_input(name='user_id', dtype='int32', shape=[None], train=False)
    
    def _build_item_inputs(self, train=True):
        
        if train:
            self._add_input(name='item_id', dtype='int32', shape=[self._batch_size])
        else:
            self._add_input(name='item_id', dtype='none', train=False)
    
    def _build_extra_inputs(self, train=True):
        
        if train:
            self._add_input(name='labels', dtype='float32', shape=[self._batch_size])

    def _build_user_extractions(self, train=True):

        self._add_module('user_vec', 
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('user_id', train=train),
                                    shape=[self._max_user, self._dim_embed], scope='user', reuse=not train), 
                         train=train)
    
    def _build_item_extractions(self, train=True):
        
        self._add_module('item_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('item_id', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=not train), 
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
                                        labels=self._get_input('labels'), a=1.0, b=1.0, sigmoid=True, 
                                        train=train, scope='PointwiseMSE', reuse=not train),
                        train=train)

    def _build_serving_graph(self):

        super(PMF, self)._build_serving_graph()
        self._scores = self._get_module('interaction', train=False).get_outputs()[0]
