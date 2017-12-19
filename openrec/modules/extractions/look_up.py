from __future__ import print_function
import tensorflow as tf
from termcolor import colored
from openrec.modules.extractions import Extraction

class LookUp(Extraction):

    """
    The LookUp module maps (embeds) input ids into *fixed* representations. The representations are \
    not be updated during training. The module outputs a tensor with shape \
    **shape(ids) + [embedding dimensionality]**.

    Parameters
    ----------
    embed: numpy array
        Fixed embedding matrix.
    ids: Tensorflow tensor, optional
        List of ids to retrieve embeddings. If *None*, the whole embedding matrix is returned.
    scope: str, optional
        Scope for module variables.
    reuse: bool, optional
        Whether or not to reuse module variables.
    """

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
