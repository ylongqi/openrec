from openrec.recommenders import Recommender
from openrec.modules.extractions import LatentFactor
from openrec.modules.interactions import PairwiseLog

class BPR(Recommender):

    """
    Pure Baysian Personalized Ranking (BPR) [1]_ based Recommender

    Parameters
    ----------
    batch_size: int
        Training batch size. Each training instance consists of 
        an user, a positive item, and a negative item.
    max_user: int
        Maximum number of users in the recommendation system.
    max_item: int
        Maximum number of items in the recommendation system.
    dim_embed: int
        Dimensionality of the user/item embedding.
    test_batch_size: int, optional
        Batch size for testing and serving. Each testing/serving bacth consists of
        an user.
    l2_reg: float, optional
        Weight for L2 regularization, i.e., weight decay.
    opt: 'SGD'(default) or 'Adam', optional
        Optimization algorithm, SGD: Stochastic Gradient Descent.
    lr: float, optional
        Initial learning rate.
    init_dict: dict, optional
        Key-value pairs for inital parameter values.
    sess_config: tensorflow.ConfigProto(), optional
        Tensorflow session configuration.

    Notes  
    -----
    BPR recommender is trained on users' implicit feedback signals (e.g., clicks and views). The items 
    clicked or viewed are treated as positive items, and otherwise as negative items. The pure BPR
    recommender does not consider any other auxiliary signals.

    References
    ----------
    .. [1] Rendle, S., Freudenthaler, C., Gantner, Z. and Schmidt-Thieme, L., 2009, June. 
        BPR: Bayesian personalized ranking from implicit feedback. In Proceedings of the 
        twenty-fifth conference on uncertainty in artificial intelligence (pp. 452-461). AUAI Press.
    """

    def __init__(self, batch_size, max_user, max_item, dim_embed, 
        test_batch_size=None, l2_reg=None, opt='SGD', lr=None, init_dict=None, sess_config=None):

        self._dim_embed = dim_embed

        super(BPR, self).__init__(batch_size=batch_size, 
                                  test_batch_size=test_batch_size,
                                  max_user=max_user, 
                                  max_item=max_item, 
                                  l2_reg=l2_reg,
                                  opt=opt,
                                  lr=lr,
                                  init_dict=init_dict,
                                  sess_config=sess_config)

    def _input_mappings(self, batch_data, train):

        if train:
            return {self._get_input('user_id'): batch_data['user_id_input'],
                    self._get_input('p_item_id'): batch_data['p_item_id_input'],
                    self._get_input('n_item_id'): batch_data['n_item_id_input']}
        else:
            return {self._get_input('user_id', train=train): batch_data['user_id_input']}

    def _build_user_inputs(self, train=True):
        
        if train:
            self._add_input(name='user_id', dtype='int32', shape=[self._batch_size])
        else:
            self._add_input(name='user_id', dtype='int32', shape=[self._test_batch_size], train=False)

    def _build_item_inputs(self, train=True):

        if train:
            self._add_input(name='p_item_id', dtype='int32', shape=[self._batch_size])
            self._add_input(name='n_item_id', dtype='int32', shape=[self._batch_size])
        else:
            self._add_input(name='item_id', dtype='none', train=False)

    def _build_user_extractions(self, train=True):
        
        self._add_module('user_vec', 
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('user_id', train=train),
                                    shape=[self._max_user, self._dim_embed], scope='user', reuse=not train), 
                         train=train)

    def _build_item_extractions(self, train=True):

        if train:
            self._add_module('p_item_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('p_item_id', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=False), 
                         train=True)
            self._add_module('p_item_bias',
                         LatentFactor(l2_reg=self._l2_reg, init='zero', ids=self._get_input('p_item_id', train=train),
                                    shape=[self._max_item, 1], scope='item_bias', reuse=False), 
                         train=True)
            self._add_module('n_item_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('n_item_id', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True), 
                         train=True)
            self._add_module('n_item_bias',
                         LatentFactor(l2_reg=self._l2_reg, init='zero', ids=self._get_input('n_item_id', train=train),
                                    shape=[self._max_item, 1], scope='item_bias', reuse=True), 
                         train=True)
        else:
            self._add_module('item_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('item_id', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True), 
                         train=False)
            self._add_module('item_bias',
                         LatentFactor(l2_reg=self._l2_reg, init='zero', ids=self._get_input('item_id', train=train),
                                    shape=[self._max_item, 1], scope='item_bias', reuse=True), 
                         train=False)

    def _build_default_interactions(self, train=True):

        if train:
            self._add_module('interaction',
                            PairwiseLog(user=self._get_module('user_vec').get_outputs()[0], 
                                    p_item=self._get_module('p_item_vec').get_outputs()[0],
                                    n_item=self._get_module('n_item_vec').get_outputs()[0], 
                                    p_item_bias=self._get_module('p_item_bias').get_outputs()[0],
                                    n_item_bias=self._get_module('n_item_bias').get_outputs()[0], 
                                    scope='pairwise_log', reuse=False, train=True),
                            train=True)
        else:
            self._add_module('interaction',
                            PairwiseLog(user=self._get_module('user_vec', train=train).get_outputs()[0],
                                        item=self._get_module('item_vec', train=train).get_outputs()[0], 
                                        item_bias=self._get_module('item_bias', train=train).get_outputs()[0],
                                        scope='pairwise_log', reuse=True, train=False),
                            train=False)

    def _build_serving_graph(self):
        
        super(BPR, self)._build_serving_graph()
        self._scores = self._get_module('interaction', train=False).get_outputs()[0]
