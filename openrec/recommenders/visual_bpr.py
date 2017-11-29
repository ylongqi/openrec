from openrec.recommenders import BPR
from openrec.modules.extractions import LatentFactor, MultiLayerFC
from openrec.modules.fusions import Average

class VisualBPR(BPR):

    def __init__(self, batch_size, max_user, max_item, dim_embed,
            dims, item_f_source, test_batch_size=None, item_serving_size=None,
            dropout_rate=None, l2_reg_u=None, l2_reg_mlp=None, 
            l2_reg_v=None, opt='Adam', sess_config=None):

        self._dims = dims
        self._dropout_rate = dropout_rate
        self._item_f_source = item_f_source

        self._l2_reg_u = l2_reg_u
        self._l2_reg_mlp = l2_reg_mlp
        self._l2_reg_v = l2_reg_v

        super(VisualBPR, self).__init__(batch_size=batch_size, max_user=max_user, max_item=max_item,
                                    test_batch_size=test_batch_size, dim_embed=dim_embed, sess_config=sess_config)

    def _build_item_inputs(self, train=True):
        
        super(VisualBPR, self)._build_item_inputs(train)
        if train:
            self._p_item_vfeature_input = self._input(dtype='float32', shape=[self._batch_size, self._item_f_source.shape[1]], 
                                                name='p_item_vfeature_input')
            self._n_item_vfeature_input = self._input(dtype='float32', shape=[self._batch_size, self._item_f_source.shape[1]],
                                                name='n_item_vfeature_input')
        else:
            self._item_id_serving = self._input(dtype='int32', shape=[None],
                                                name='item_id_serving')
            self._item_vfeature_serving = self._input(dtype='float32', shape=[None, self._item_f_source.shape[1]], name='item_vfeature_serving')

    def _input_mappings(self, batch_data, train):

        default_input_map = super(VisualBPR, self)._input_mappings(batch_data=batch_data, train=train)
        if train:
            default_input_map[self._p_item_vfeature_input] = self._item_f_source[batch_data['p_item_id_input']]
            default_input_map[self._n_item_vfeature_input] = self._item_f_source[batch_data['n_item_id_input']]
        else:
            default_input_map[self._item_id_serving] = batch_data['item_id_input']
            default_input_map[self._item_vfeature_serving] = self._item_f_source[batch_data['item_id_input']]
        
        return default_input_map

    def _build_item_extractions(self, train=True):

        super(VisualBPR, self)._build_item_extractions(train)
        if train:
            self._loss_nodes.remove(self._p_item_vec)
            self._loss_nodes.remove(self._n_item_vec)
            
            self._p_item_vf = MultiLayerFC(in_tensor=self._p_item_vfeature_input, dropout_mid=self._dropout_rate, 
                                        dims=self._dims, scope='item_visual_embed', reuse=False)
            self._n_item_vf =  MultiLayerFC(in_tensor=self._n_item_vfeature_input, dropout_mid=self._dropout_rate, 
                                        dims=self._dims, scope='item_visual_embed', reuse=True)
        else:
            self._item_vf_serving = MultiLayerFC(in_tensor=self._item_vfeature_serving, 
                                                dims=self._dims, scope='item_visual_embed', reuse=True)

    def _build_default_fusions(self, train=True):

        if train:
            self._p_item_vec = Average(scope='item_concat', reuse=False,
                                    module_list=[self._p_item_vec, self._p_item_vf], weight=2.0)
            self._n_item_vec = Average(scope='item_concat', reuse=True,
                                    module_list=[self._n_item_vec, self._n_item_vf], weight=2.0)
            self._loss_nodes += [self._p_item_vec, self._p_item_bias, self._n_item_vec, self._n_item_bias]
        else:
            self._item_vec_serving = Average(scope='item_concat', reuse=True, 
                                module_list=[self._item_vec_serving, self._item_vf_serving], weight=2.0)

    def _grad_post_processing(self, grad_var_list):
        
        normalized_grad_var_list = []
        for grad, var in grad_var_list:
            if 'item_MLP' in var.name:
                normalized_grad_var_list.append((grad * (1.0 / self._batch_size), var))
            else:
                normalized_grad_var_list.append((grad, var))
        return normalized_grad_var_list