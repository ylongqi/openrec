from openrec.recommenders import VisualPMF
from openrec.modules.interactions import PointwiseGeCE

class VisualGMF(VisualPMF):

     def _build_default_interactions(self, train=True):
        
        self._add_module('interaction',
                        PointwiseGeCE(user=self._get_module('user_vec', train=train).get_outputs()[0], 
                                        item=self._get_module('item_vec', train=train).get_outputs()[0],
                                        item_bias=self._get_module('item_bias', train=train).get_outputs()[0], 
                                        labels=self._get_input('labels'), l2_reg=self._l2_reg, 
                                        train=train, scope='PointwiseGeCE', reuse=not train),
                        train=train)
