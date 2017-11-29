from openrec.recommenders import VisualPMF
from openrec.modules.extractions import MultiLayerFC
from openrec.modules.fusions import Average

class UserVisualPMF(VisualPMF):

    def __init__(self, batch_size, max_user, max_item, dim_embed, dims_user, dims_item, user_f_source, 
                    item_f_source, test_batch_size=None, item_serving_size=None, dropout_rate=None,
                    l2_reg_u=None, l2_reg_mlp=None, l2_reg_v=None, opt='SGD', sess_config=None):

        self._dims_user = dims_user
        self._user_f_source = user_f_source

        super(UserVisualPMF, self).__init__(batch_size=batch_size, max_user=max_user,
            max_item=max_item, dim_embed=dim_embed, dims=dims_item, item_f_source=item_f_source,
            test_batch_size=test_batch_size, item_serving_size=item_serving_size, dropout_rate=dropout_rate,
            l2_reg_u=l2_reg_u, l2_reg_mlp=l2_reg_mlp, l2_reg_v=l2_reg_v, opt=opt, sess_config=sess_config)

    def _build_user_inputs(self, train=True):
        
        super(UserVisualPMF, self)._build_user_inputs(train)
        if train:
            self._user_feature_input = self._input(dtype='float32', shape=[self._batch_size, self._user_f_source.shape[1]], 
                                                name='user_feature_input')
        else:
            self._user_feature_serving = self._input(dtype='float32', shape=[None, self._user_f_source.shape[1]], 
                                                name='user_feature_serving')

    def _input_mappings(self, batch_data, train):

        default_input_map = super(UserVisualPMF, self)._input_mappings(batch_data=batch_data, train=train)
        if train:
            default_input_map[self._user_feature_input] = self._user_f_source[batch_data['user_id_input']]
        else:
            default_input_map[self._user_feature_serving] = self._user_f_source[batch_data['user_id_input']]
        return default_input_map

    def _build_user_extractions(self, train=True):

        super(UserVisualPMF, self)._build_user_extractions(train)

        if train:
            self._loss_nodes.remove(self._user_vec)
            self._user_f = MultiLayerFC(in_tensor=self._user_feature_input, dims=self._dims_user, l2_reg=self._l2_reg_mlp,
                            dropout_mid=self._dropout_rate, scope='user_MLP', reuse=False)
        else:
            self._user_f_serving = MultiLayerFC(in_tensor=self._user_feature_serving, dims=self._dims_user, l2_reg=self._l2_reg_mlp,
                            dropout_mid=self._dropout_rate, scope='user_MLP', reuse=True)

    def _build_default_fusions(self, train=True):

        super(UserVisualPMF, self)._build_default_fusions(train)
        if train:
            self._user_vec = Average(scope='user_average', reuse=False, module_list=[self._user_vec, self._user_f], weight=2.0)
            self._loss_nodes += [self._user_vec]
        else:
            self._user_vec_serving = Average(scope='user_average', reuse=True, 
                                module_list=[self._user_vec_serving, self._user_f_serving], weight=2.0)