from openrec.recommenders import PMF
from openrec.modules.extractions import SDAE
from openrec.modules.fusions import Average

class CDL(PMF):

    def __init__(self, batch_size, max_user, max_item, dim_embed, item_f, dims, dropout=None, test_batch_size=None,
                    item_serving_size=None, l2_reg_lf=None, l2_reg_mlp=None, l2_reconst=None, opt='SGD',
                    sess_config=None):


        self._item_f = item_f
        self._dims = dims
        self._dropout = dropout

        self._l2_reg_lf = l2_reg_lf
        self._l2_reg_mlp = l2_reg_mlp
        self._l2_reconst = l2_reconst

        super(CDL, self).__init__(batch_size=batch_size, max_user=max_user, max_item=max_item, dim_embed=dim_embed,
                                test_batch_size=test_batch_size, opt=opt, sess_config=sess_config)

    def _build_item_inputs(self, train=True):

        super(CDL, self)._build_item_inputs(train)
        if train:
            self._item_feature_input = self._input(dtype='float32', shape=[self._batch_size, self._item_f.shape[1]], 
                                                name='item_feature_input')
        else:
            self._item_id_serving = self._input(dtype='int32', shape=[None],
                                                name='item_id_serving')
            self._item_feature_serving = self._input(dtype='float32', shape=[None, self._item_f.shape[1]], name='item_feature_serving')

    def _input_mappings(self, batch_data, train):

        default_input_map = super(CDL, self)._input_mappings(batch_data=batch_data, train=train)
        if train:
            default_input_map[self._item_feature_input] = self._item_f[batch_data['item_id_input']]
        else:
            default_input_map[self._item_id_serving] = batch_data['item_id_input']
            default_input_map[self._item_feature_serving] = self._item_f[batch_data['item_id_input']]
        return default_input_map

    def _build_item_extractions(self, train=True):

        super(CDL, self)._build_item_extractions(train)

        if train:
            self._loss_nodes.remove(self._item_lf)
            sdae = SDAE(in_tensor=self._item_feature_input, dims=self._dims, l2_reg=self._l2_reg_mlp,
                        l2_reconst=self._l2_reconst, dropout=self._dropout, scope='AutoEncoder', reuse=False)
            self._item_lf = Average(scope='item_average', reuse=False, module_list=[self._item_lf, sdae], weight=2.0)
            self._loss_nodes += [self._item_lf]
        else:
            sdae = SDAE(in_tensor=self._item_feature_serving, dims=self._dims, l2_reg=self._l2_reg_mlp,
                        l2_reconst=self._l2_reconst, dropout=self._dropout, scope='AutoEncoder', reuse=True)
            self._item_lf_serving = Average(scope='item_average', reuse=True, module_list=[self._item_lf_serving, sdae], weight=2.0)