from openrec.recommenders import BPR
from openrec.modules.extractions import LatentFactor, MultiLayerFC
from openrec.modules.fusions import Average

class FeatureBasedBPR(BPR):

    def __init__(self, batch_size, max_user, max_item, dim_embed, test_batch_size=None, item_serving_size=None,
            opt='Adam', sess_config=None):


        super(FeatureBasedBPR, self).__init__(batch_size=batch_size, max_user=max_user, max_item=max_item,
                                    test_batch_size=test_batch_size, dim_embed=dim_embed, sess_config=sess_config)

    def _build_item_inputs(self, train=True):
        
        super(FeatureBasedBPR, self)._build_item_inputs(train)
        if train:
            self._p_item_genre_input = self._input(dtype='int32', shape=[self._batch_size], name='p_item_genre_input')
            self._n_item_genre_input = self._input(dtype='int32', shape=[self._batch_size], name='n_item_genre_input')
        else:
            self._item_id_serving = None

    def _input_mappings(self, batch_data, train):

        default_input_map = super(FeatureBasedBPR, self)._input_mappings(batch_data=batch_data, train=train)
        if train:
            default_input_map[self._p_item_genre_input] = self._item_f_source[batch_data['p_item_genre_input']]
            default_input_map[self._n_item_genre_input] = self._item_f_source[batch_data['n_item_genre_input']]
        else:
            default_input_map[self._item_id_serving] = batch_data['item_id_input']
            default_input_map[self._item_vfeature_serving] = self._item_f_source[batch_data['item_id_input']]
        
        return default_input_map

    def _build_item_extractions(self, train=True):

        super(FeatureBasedBPR, self)._build_item_extractions(train)
        if train:
            
            self._p_song_genre = LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._p_item_genre_input,
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True)
            self._n_song_genre = LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._n_item_genre_input,
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True)
        else:
            self._item_vec_serving = LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._item_id_serving,
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True)

    def _build_default_fusions(self, train=True):

        if train:
            self._p_item_vec = Average(scope='item_concat', reuse=False,
                                    module_list=[self._p_item_vec, self._p_song_genre], weight=2.0)
            self._n_item_vec = Average(scope='item_concat', reuse=True,
                                    module_list=[self._n_item_vec, self._n_song_genre], weight=2.0)
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