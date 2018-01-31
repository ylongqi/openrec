from openrec.modules.interactions import PairwiseEuDist
from openrec.recommenders import VisualBPR

class VisualCML(VisualBPR):

    def _build_default_interactions(self, train=True):
        
        if train:
            self._add_module('interaction',
                            PairwiseEuDist(user=self._get_module('user_vec').get_outputs()[0], 
                                    p_item=self._get_module('p_item_vec').get_outputs()[0],
                                    n_item=self._get_module('n_item_vec').get_outputs()[0], 
                                    p_item_bias=self._get_module('p_item_bias').get_outputs()[0],
                                    n_item_bias=self._get_module('n_item_bias').get_outputs()[0], 
                                    scope='PairwiseEuDist', reuse=False, train=True),
                            train=True)
        else:
            self._add_module('interaction',
                            PairwiseEuDist(user=self._get_module('user_vec', train=train).get_outputs()[0], 
                                     item=self._get_module('item_vec', train=train).get_outputs()[0],
                                    item_bias=self._get_module('item_bias', train=train).get_outputs()[0],
                                    scope='PairwiseEuDist', reuse=True, train=False),
                            train=False)