from openrec.recommenders import AddvPMF
from openrec.modules.fusions import Concat
from openrec.modules.extractions import LatentFactor
from openrec.modules.extractions import MultiLayerFC

class ConcatvPMF(AddvPMF):

    def _build_item_extractions(self, train=True):

        if train:
            item_lf_offset = LatentFactor(l2_reg=self._l2_reg_v, init='normal', ids=self._item_id_input,
                                    shape=[self._num_item, self._dim_embed / 2], scope='item_offset', reuse=False)
            mlfc = MultiLayerFC(in_tensor=self._item_vfeatures_input, dims=self._dims, l2_reg=self._l2_reg_mlp,
                            dropout_mid=self._dropout_rate, scope='item_feature_embed', reuse=False)
            self._item_lf = Concat(scope='item_concat', reuse=False, module_list=[item_lf_offset, mlfc])
            self._item_bias = LatentFactor(l2_reg=self._l2_reg, init='zero', ids=self._item_id_input,
                                    shape=[self._num_item, 1], scope='item_bias', reuse=False)
            self._loss_nodes += [self._item_lf, self._item_bias]
        else:
            item_lf_offset_serving = LatentFactor(l2_reg=self._l2_reg_v, init='normal', ids=self._item_id_serving,
                                    shape=[self._num_item, self._dim_embed / 2], scope='item_offset', reuse=True)
            mlfc = MultiLayerFC(in_tensor=self._item_vfeatures_serving, dims=self._dims, l2_reg=self._l2_reg_mlp,
                            dropout_mid=self._dropout_rate, scope='item_feature_embed', reuse=True)
            self._item_lf_serving = Concat(scope='item_concat', reuse=True, module_list=[
                                    item_lf_offset_serving, mlfc])
            self._item_bias_serving = LatentFactor(l2_reg=self._l2_reg, init='zero', ids=self._item_id_serving,
                                    shape=[self._num_item, 1], scope='item_bias', reuse=True)
