from openrec.recommenders import PMF
from openrec.modules.extractions import MultiLayerFC
from openrec.modules.fusions import Average

class VisualPMF(PMF):

    def __init__(self, batch_size, max_user, max_item, dim_embed, dims, item_f_source, test_batch_size=None, item_serving_size=None, dropout_rate=None,
                    l2_reg=None, l2_reg_mlp=None, opt='SGD', sess_config=None):

        self._dims = dims
        self._dropout_rate = dropout_rate
        self._item_f_source = item_f_source
        self._item_serving_size = item_serving_size

        self._l2_reg_mlp = l2_reg_mlp

        super(VisualPMF, self).__init__(batch_size=batch_size, max_user=max_user, max_item=max_item, dim_embed=dim_embed,
                                l2_reg=l2_reg, test_batch_size=test_batch_size, opt=opt, sess_config=sess_config)

    def _build_item_inputs(self, train=True):
        
        super(VisualPMF, self)._build_item_inputs(train)
        if train:
            self._add_input(name='item_vfeature', dtype='float32', shape=[self._batch_size, self._item_f_source.shape[1]])
        else:
            self._add_input(name='item_id', dtype='int32', shape=[None], train=False)
            self._add_input(name='item_vfeature', dtype='float32', shape=[None, self._item_f_source.shape[1]], train=False)

    def _input_mappings(self, batch_data, train):

        default_input_map = super(VisualPMF, self)._input_mappings(batch_data=batch_data, train=train)
        if train:
            default_input_map[self._get_input('item_vfeature')] = self._item_f_source[batch_data['item_id_input']]
        else:
            default_input_map[self._get_input('item_id', train=False)] = batch_data['item_id_input']
            default_input_map[self._get_input('item_vfeature', train=False)] = self._item_f_source[batch_data['item_id_input']]
        return default_input_map

    def _build_item_extractions(self, train=True):

        super(VisualPMF, self)._build_item_extractions(train)
        self._add_module('item_vf',
                         MultiLayerFC(in_tensor=self._get_input('item_vfeature', train=train), dims=self._dims, 
                                      l2_reg=self._l2_reg_mlp, dropout_mid=self._dropout_rate, scope='item_MLP', reuse=not train,
                                      train=train),
                         train=train)

    def _build_default_fusions(self, train=True):

        self._add_module('item_vec',
                        Average(scope='item_average', reuse=not train, module_list=[self._get_module('item_vec', train=train), 
                                self._get_module('item_vf', train=train)], weight=2.0),
                        train=train)
