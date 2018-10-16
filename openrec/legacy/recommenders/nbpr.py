from openrec.legacy.recommenders import BPR
from openrec.legacy.modules.extractions import LatentFactor
from openrec.legacy.modules.interactions import NsLog
import numpy as np

class NBPR(BPR):

    """
    Pure Baysian Personalized Ranking (BPR) [1]_ based Recommender with negative sampling

    """

    def __init__(self, batch_size, max_user, max_item, dim_embed, neg_num=5,
        test_batch_size=None, l2_reg=None, opt='SGD', lr=None, init_dict=None, sess_config=None):

        self._dim_embed = dim_embed
        self._neg_num = neg_num

        super(NBPR, self).__init__(batch_size=batch_size, 
                                  test_batch_size=test_batch_size,
                                  max_user=max_user, 
                                  max_item=max_item, 
                                  dim_embed = dim_embed,
                                  l2_reg=l2_reg,
                                  opt=opt,
                                  lr=lr,
                                  init_dict=init_dict,
                                  sess_config=sess_config)


    def _input_mappings(self, batch_data, train):

        if train:
            return {self._get_input('user_id'): batch_data['user_id_input'],
                    self._get_input('p_item_id'): batch_data['p_item_id_input'],
                    self._get_input('n_item_id'): np.array(batch_data['n_item_id_inputs'].tolist())}
        else:
            return {self._get_input('user_id', train=train): batch_data['user_id_input'],
                   self._get_input('item_id', train=train): batch_data['item_id_input']}

    def _build_item_inputs(self, train=True):

        if train:
            self._add_input(name='p_item_id', dtype='int32', shape=[self._batch_size])
            self._add_input(name='n_item_id', dtype='int32', shape=[self._batch_size, self._neg_num])
        else:
            self._add_input(name='item_id', dtype='int32', shape=[None], train=False)


    def _build_default_interactions(self, train=True):

        if train:
            self._add_module('interaction',
                            NsLog(user=self._get_module('user_vec').get_outputs()[0],
                                  max_item=self._max_item,
                                    p_item=self._get_module('p_item_vec').get_outputs()[0],
                                    n_item=self._get_module('n_item_vec').get_outputs()[0], 
                                    p_item_bias=self._get_module('p_item_bias').get_outputs()[0],
                                    n_item_bias=self._get_module('n_item_bias').get_outputs()[0], 
                                    scope='pairwise_log', reuse=False, train=True),
                            train=True)
        else:
            self._add_module('interaction',
                            NsLog(user=self._get_module('user_vec', train=train).get_outputs()[0],
                                  max_item=self._max_item,
                                        item=self._get_module('item_vec', train=train).get_outputs()[0], 
                                        item_bias=self._get_module('item_bias', train=train).get_outputs()[0],
                                        scope='pairwise_log', reuse=True, train=False),
                            train=False)

  
