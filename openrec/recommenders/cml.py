import tensorflow as tf
from openrec.recommenders import BPR
from openrec.modules.interactions import PairwiseEuDist

class CML(BPR):

    def _build_post_training_ops(self):
        unique_user_id, _ = tf.unique(self._user_id_input)
        unique_item_id, _ = tf.unique(tf.concat([self._p_item_id_input, self._n_item_id_input], axis=0))
        return [self._user_vec.censor_l2_norm_op(censor_id_list=unique_user_id),
                self._p_item_vec.censor_l2_norm_op(censor_id_list=unique_item_id)]

    def _build_interactions(self, train=True):

        if train:
            self._interaction_train = PairwiseEuDist(user=self._user_vec.get_outputs()[0], 
                                                p_item=self._p_item_vec.get_outputs()[0],
                                                n_item=self._n_item_vec.get_outputs()[0], 
                                                p_item_bias=self._p_item_bias.get_outputs()[0],
                                                n_item_bias=self._n_item_bias.get_outputs()[0], 
                                                train=True, scope='pairwise_eu_dist', reuse=False)
            self._loss_nodes.append(self._interaction_train)
        else:
            self._interaction_serve = PairwiseEuDist(user=self._user_vec_serving.get_outputs()[0], 
                                                    item=self._item_vec_serving.get_outputs()[0],
                                                    item_bias=self._item_bias_serving.get_outputs()[0], 
                                                    train=False, scope='CML', reuse=False)
