from openrec.modules.interactions import PairwiseEuDist
from openrec.recommenders import VisualBPR

class VisualCML(VisualBPR):

    def _build_default_interactions(self, train=True):
        
        if train:
            self._interaction_train = PairwiseEuDist(user=self._user_vec.get_outputs()[0], p_item=self._p_item_vec.get_outputs()[0],
                        n_item=self._n_item_vec.get_outputs()[0], p_item_bias=self._p_item_bias.get_outputs()[0],
                        n_item_bias=self._n_item_bias.get_outputs()[0], train=True, scope='PairwiseEuDist', reuse=False)
            self._loss_nodes.append(self._interaction_train)
        else:
            self._interaction_serve = PairwiseEuDist(user=self._user_vec_serving.get_outputs()[0], item=self._item_vec_serving.get_outputs()[0],
                                    item_bias=self._item_bias_serving.get_outputs()[0], train=False, scope='PairwiseEuDist', reuse=False)