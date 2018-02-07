from openrec.recommenders import BPR
from openrec.modules.extractions import LatentFactor, MultiLayerFC
from openrec.modules.fusions import Average

class VisualBPR(BPR):

    def __init__(self, batch_size, max_user, max_item, dim_embed,
            dims, item_f_source, test_batch_size=None, item_serving_size=None,
            dropout_rate=None, l2_reg=None, l2_reg_mlp=None, 
            opt='Adam', sess_config=None):

        self._dims = dims
        self._dropout_rate = dropout_rate
        self._item_f_source = item_f_source

        self._l2_reg_mlp = l2_reg_mlp

        super(VisualBPR, self).__init__(batch_size=batch_size, max_user=max_user, max_item=max_item, l2_reg=l2_reg,
                                    test_batch_size=test_batch_size, dim_embed=dim_embed, sess_config=sess_config)

    def _build_item_inputs(self, train=True):
        
        super(VisualBPR, self)._build_item_inputs(train)
        if train:
            self._add_input(name='p_item_vfeature', dtype='float32', shape=[self._batch_size, self._item_f_source.shape[1]])
            self._add_input(name='n_item_vfeature', dtype='float32', shape=[self._batch_size, self._item_f_source.shape[1]])
        else:
            self._add_input(name='item_id', dtype='int32', shape=[None], train=False)
            self._add_input(name='item_vfeature', dtype='float32', shape=[None, self._item_f_source.shape[1]], train=False)

    def _input_mappings(self, batch_data, train):

        default_input_map = super(VisualBPR, self)._input_mappings(batch_data=batch_data, train=train)
        if train:
            default_input_map[self._get_input('p_item_vfeature')] = self._item_f_source[batch_data['p_item_id_input']]
            default_input_map[self._get_input('n_item_vfeature')] = self._item_f_source[batch_data['n_item_id_input']]
        else:
            default_input_map[self._get_input('item_id', train=False)] = batch_data['item_id_input']
            default_input_map[self._get_input('item_vfeature', train=False)] = self._item_f_source[batch_data['item_id_input']]
        
        return default_input_map

    def _build_item_extractions(self, train=True):

        super(VisualBPR, self)._build_item_extractions(train)
        if train:
            self._add_module('p_item_vf',
                            MultiLayerFC(in_tensor=self._get_input('p_item_vfeature'), dropout_mid=self._dropout_rate, train=True,
                                        l2_reg=self._l2_reg_mlp, dims=self._dims, scope='item_visual_embed', reuse=False))
            self._add_module('n_item_vf',
                            MultiLayerFC(in_tensor=self._get_input('n_item_vfeature'), dropout_mid=self._dropout_rate, train=True,
                                        l2_reg=self._l2_reg_mlp, dims=self._dims, scope='item_visual_embed', reuse=True))
        else:
            self._add_module('item_vf',
                            MultiLayerFC(in_tensor=self._get_input('item_vfeature', train=False), train=False,
                                        l2_reg=self._l2_reg_mlp, dims=self._dims, scope='item_visual_embed', reuse=True),
                            train=False)

    def _build_default_fusions(self, train=True):

        if train:
            self._add_module('p_item_vec',
                            Average(scope='item_concat', reuse=False,
                                    module_list=[self._get_module('p_item_vec'), self._get_module('p_item_vf')], weight=2.0))
            self._add_module('n_item_vec',
                            Average(scope='item_concat', reuse=True,
                                    module_list=[self._get_module('n_item_vec'), self._get_module('n_item_vf')], weight=2.0))
        else:
            self._add_module('item_vec',
                            Average(scope='item_concat', reuse=True, 
                                module_list=[self._get_module('item_vec', train=train), self._get_module('item_vf', train=train)], weight=2.0),
                            train=False)

    def _grad_post_processing(self, grad_var_list):
        
        normalized_grad_var_list = []
        for grad, var in grad_var_list:
            if 'item_MLP' in var.name:
                normalized_grad_var_list.append((grad * (1.0 / self._batch_size), var))
            else:
                normalized_grad_var_list.append((grad, var))
        return normalized_grad_var_list
