from __future__ import print_function
import tensorflow as tf
from termcolor import colored
from openrec.modules.extractions import Extraction

class LookUp(Extraction):

    def __init__(self, embed, ids=None, scope=None, reuse=False):

        assert embed is not None, 'embed cannot be None'
        self._embed = embed
        self._ids = ids
        super(LookUp, self).__init__(scope=scope, reuse=reuse)

    def _build_shared_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):

            self.embedding = tf.Variable(self._embed, trainable=False, name='embedding',dtype=tf.float32)
            if self._ids is not None:
                self._outputs.append(tf.nn.embedding_lookup(self.embedding, self._ids))
            else:
                self._outputs.append(self.embedding)
