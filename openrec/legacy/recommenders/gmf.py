from openrec.legacy.recommenders import PMF
from openrec.legacy.modules.interactions import PointwiseGeCE

class GMF(PMF):

    def __init__(self, batch_size, dim_embed, max_user, max_item,
                    test_batch_size=None, l2_reg=None, opt='SGD', sess_config=None):
        
        super(GMF, self).__init__(
            batch_size = batch_size,
            dim_embed = dim_embed,
            max_user = max_user,
            max_item = max_item,
            test_batch_size = test_batch_size,
            l2_reg = l2_reg,
            opt = opt,
            sess_config = sess_config
        )
    
    def _build_default_interactions(self, train=True):
        
        self._add_module(
            'interaction',
            PointwiseGeCE(
                user=self._get_module('user_vec', train=train).get_outputs()[0],
                item=self._get_module('item_vec', train=train).get_outputs()[0],
                item_bias=self._get_module('item_bias', train=train).get_outputs()[0],
                l2_reg=self._l2_reg,
                labels=self._get_input('labels'),
                train=train, scope="PointwiseGeCE", reuse=(not train)
            ),
            train=train
        )

