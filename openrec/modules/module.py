import tensorflow as tf


class Module(object):

    def __init__(self, train=True, l2_reg=None, scope=None, reuse=False):

        self._scope = self.__class__.__name__ if scope is None else scope
        self._reuse = reuse
        self._l2_reg = l2_reg

        self._loss = 0.0
        self._outputs = []

        if train:
            self._build_shared_graph()
            self._build_training_graph()
        else:
            self._build_shared_graph()
            self._build_serving_graph()

    def _build_shared_graph(self):
        return None

    def _build_training_graph(self):
        return None

    def _build_serving_graph(self):
        return None

    def get_outputs(self):
        return self._outputs

    def get_loss(self):
        return self._loss

