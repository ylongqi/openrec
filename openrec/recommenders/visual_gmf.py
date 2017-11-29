from openrec.recommenders import VisualPMF
from openrec.modules.interactions import PointwiseGeCE

class VisualGMF(VisualPMF):

     def _build_default_interactions(self, train=True):

        if train:
            self._interaction_train = PointwiseGeCE(user=self._user_vec.get_outputs()[0], 
                                                    item=self._item_vec.get_outputs()[0],
                                                    item_bias=self._item_bias.get_outputs()[0], 
                                                    labels=self._labels, l2_reg=self._l2_reg,
                                                    train=True, scope='PointwiseGeCE', reuse=False)
            self._loss_nodes.append(self._interaction_train)
        else:
            self._interaction_serve = PointwiseGeCE(user=self._user_vec_serving.get_outputs()[0], 
                                                    item=self._item_vec_serving.get_outputs()[0],
                                                    item_bias=self._item_bias_serving.get_outputs()[0],
                                                    train=False, scope='PointwiseGeCE', reuse=True)

    