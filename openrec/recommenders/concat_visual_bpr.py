from openrec.recommenders import BPR
from openrec.modules.extractions import LatentFactor, MultiLayerFC
from openrec.modules.fusions import Concat

class ConcatVisualBPR(BPR):

    def __init__(self, batch_size, max_user, max_item, dim_embed, dim_ve, item_f_source, 
                item_serving_size=None, l2_reg=None, sess_config=None):
        
        self._dim_ve = dim_ve
        self._item_f_source = item_f_source
        self._item_serving_size = item_serving_size
        
        super(ConcatVisualBPR, self).__init__(batch_size=batch_size, 
                                   max_user=max_user, 
                                   max_item=max_item,
                                   dim_embed=dim_embed, 
                                   l2_reg=l2_reg,
                                  sess_config=sess_config)

    def _build_item_inputs(self, train=True):
        
        super(ConcatVisualBPR, self)._build_item_inputs(train)
        if train:
            self._add_input(name='p_item_vfeature', dtype='float32', shape=[self._batch_size, self._item_f_source.shape[1]])
            self._add_input(name='n_item_vfeature', dtype='float32', shape=[self._batch_size, self._item_f_source.shape[1]])
        else:
            self._add_input(name='item_id', dtype='int32', shape=[None], train=False)
            self._add_input(name='item_vfeature', dtype='float32', shape=[None, self._item_f_source.shape[1]], train=False)

    def _input_mappings(self, batch_data, train):

        default_input_map = super(ConcatVisualBPR, self)._input_mappings(batch_data=batch_data, train=train)
        if train:
            default_input_map[self._get_input('p_item_vfeature')] = self._item_f_source[batch_data['p_item_id_input']]
            default_input_map[self._get_input('n_item_vfeature')] = self._item_f_source[batch_data['n_item_id_input']]
        else:
            default_input_map[self._get_input('item_id', train=False)] = batch_data['item_id_input']
            default_input_map[self._get_input('item_vfeature', train=False)] = self._item_f_source[batch_data['item_id_input']]
        
        return default_input_map
        
    def _build_item_extractions(self, train=True):

        if train:
            self._add_module('p_item_lf',
                            LatentFactor(init='normal', l2_reg=self._l2_reg, ids=self._get_input('p_item_id'), 
                                        shape=[self._max_item, self._dim_embed-self._dim_ve], scope='item', reuse=False))
            self._add_module('p_item_vf',
                            MultiLayerFC(in_tensor=self._get_input('p_item_vfeature'), 
                                        dims=[self._dim_ve], scope='item_MLP', reuse=False))
            self._add_module('p_item_bias',
                            LatentFactor(l2_reg=self._l2_reg, init='zero', ids=self._get_input('p_item_id'),
                                        shape=[self._max_item, 1], scope='item_bias', reuse=False))
            self._add_module('n_item_lf',
                            LatentFactor(init='normal', l2_reg=self._l2_reg, ids=self._get_input('n_item_id'), 
                                        shape=[self._max_item, self._dim_embed-self._dim_ve], scope='item', reuse=True))
            self._add_module('n_item_vf',
                            MultiLayerFC(in_tensor=self._get_input('n_item_vfeature'), 
                                        dims=[self._dim_ve], scope='item_MLP', reuse=True))
            self._add_module('n_item_bias',
                            LatentFactor(l2_reg=self._l2_reg, init='zero', ids=self._get_input('n_item_id'),
                                        shape=[self._max_item, 1], scope='item_bias', reuse=True))
        else:
            self._add_module('item_lf',
                            LatentFactor(init='normal', l2_reg=self._l2_reg, ids=self._get_input('item_id', train=train),
                                        shape=[self._max_item, self._dim_embed-self._dim_ve], scope='item', reuse=True),
                            train=False)
            self._add_module('item_vf',
                            MultiLayerFC(in_tensor=self._get_input('item_vfeature', train=train), 
                                        dims=[self._dim_ve], scope='item_MLP', reuse=True),
                            train=False)
            self._add_module('item_bias',
                            LatentFactor(l2_reg=self._l2_reg, init='zero', ids=self._get_input('item_id', train=train),
                                        shape=[self._max_item, 1], scope='item_bias', reuse=True), 
                             train=False)

    def _build_default_fusions(self, train=True):
        
        if train:
            self._add_module('p_item_vec',
                            Concat(scope='item_concat', reuse=False,
                                    module_list=[self._get_module('p_item_lf'), self._get_module('p_item_vf')]))
            self._add_module('n_item_vec',
                            Concat(scope='item_concat', reuse=True,
                                    module_list=[self._get_module('n_item_lf'), self._get_module('n_item_vf')]))
        else:
            self._add_module('item_vec',
                            Concat(scope='item_concat', reuse=True, 
                                module_list=[self._get_module('item_lf', train=train), self._get_module('item_vf', train=train)]),
                            train=False)

    def _grad_post_processing(self, grad_var_list):
        
        normalized_grad_var_list = []
        for grad, var in grad_var_list:
            if 'item_MLP' in var.name:
                normalized_grad_var_list.append((grad * (1.0 / self._batch_size), var))
            else:
                normalized_grad_var_list.append((grad, var))
        return normalized_grad_var_list
