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
            self._p_item_vfeature_input = self._input(dtype='float32', shape=[self._batch_size, self._item_f_source.shape[1]], 
                                                name='p_item_vfeature_input')
            self._n_item_vfeature_input = self._input(dtype='float32', shape=[self._batch_size, self._item_f_source.shape[1]],
                                                name='n_item_vfeature_input')
        else:
            self._item_id_serving = self._input(dtype='int32', shape=[None],
                                                name='item_id_serving')
            self._item_vfeature_serving = self._input(dtype='float32', shape=[None, self._item_f_source.shape[1]], 
                                                name='item_vfeature_serving')

    def _input_mappings(self, batch_data, train):

        default_input_map = super(ConcatVisualBPR, self)._input_mappings(batch_data=batch_data, train=train)
        if train:
            default_input_map[self._p_item_vfeature_input] = self._item_f_source[batch_data['p_item_id_input']]
            default_input_map[self._n_item_vfeature_input] = self._item_f_source[batch_data['n_item_id_input']]
        else:
            default_input_map[self._item_id_serving] = batch_data['item_id_input']
            default_input_map[self._item_vfeature_serving] = self._item_f_source[batch_data['item_id_input']]
        
        return default_input_map
        
    def _build_item_extractions(self, train=True):

        super(ConcatVisualBPR, self)._build_item_extractions(train)

        if train:
            self._loss_nodes.remove(self._p_item_vec)
            self._loss_nodes.remove(self._n_item_vec)

            self._p_item_lf = LatentFactor(init='normal', l2_reg=self._l2_reg, ids=self._p_item_id_input, 
                                        shape=[self._max_item, self._dim_embed-self._dim_ve], scope='item', reuse=False)
            self._p_item_vf = MultiLayerFC(in_tensor=self._p_item_vfeature_input, 
                                        dims=[self._dim_ve], scope='item_MLP', reuse=False)
            self._n_item_lf = LatentFactor(init='normal', l2_reg=self._l2_reg, ids=self._n_item_id_input, 
                                        shape=[self._max_item, self._dim_embed-self._dim_ve], scope='item', reuse=True)
            self._n_item_vf =  MultiLayerFC(in_tensor=self._n_item_vfeature_input, 
                                        dims=[self._dim_ve], scope='item_MLP', reuse=True)
        else:

            self._item_lf_serving = LatentFactor(init='normal', l2_reg=self._l2_reg, ids=self._item_id_serving,
                                                shape=[self._max_item, self._dim_embed-self._dim_ve], scope='item', reuse=True)
            self._item_vf_serving = MultiLayerFC(in_tensor=self._item_vfeature_serving, 
                                                dims=[self._dim_ve], scope='item_MLP', reuse=True)

    def _build_default_fusions(self, train=True):

        if train:
            self._p_item_vec = Concat(scope='item_concat', reuse=False,
                                    module_list=[self._p_item_lf, self._p_item_vf])
            self._n_item_vec = Concat(scope='item_concat', reuse=True,
                                    module_list=[self._n_item_lf, self._n_item_vf])
            self._loss_nodes += [self._p_item_vec, self._n_item_vec]
        else:
            self._item_vec_serving = Concat(scope='item_concat', reuse=True, module_list=[self._item_lf_serving, self._item_vf_serving])

    def _grad_post_processing(self, grad_var_list):
        
        normalized_grad_var_list = []
        for grad, var in grad_var_list:
            if 'item_MLP' in var.name:
                normalized_grad_var_list.append((grad * (1.0 / self._batch_size), var))
            else:
                normalized_grad_var_list.append((grad, var))
        return normalized_grad_var_list
