import tensorflow as tf
from openrec.modules.extractions import Extraction

class LatentFactor(Extraction):

    """
    The LatentFactor module maps (embeds) input ids into latent representations. The module \
    outputs a tensor with shape **shape(ids) + [embedding dimensionality]**.

    Parameters
    ----------
    shape: list
        Shape of the embedding matrix, i.e. [number of unique ids, embedding dimensionality].
    init: str, optional
        Embedding initialization. *'zero'* or *'normal'* (default).
    ids: Tensorflow tensor, optionl
        List of ids to retrieve embeddings. If *None*, the whole embedding matrix is returned.
    l2_reg: float, optional
        Weight for L2 regularization, i.e., weight decay.
    scope: str, optional
        Scope for module variables.
    reuse: bool, optional
        Whether or not to reuse module variables.
    """

    def __init__(self, shape, init='normal', ids=None, l2_reg=None, scope=None, reuse=False):

        assert shape is not None, 'shape cannot be None'

        if init == 'normal':
            self._initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)
        elif init == 'zero':
            self._initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
        self._shape = shape
        self._ids = ids

        super(LatentFactor, self).__init__(l2_reg=l2_reg, scope=scope, reuse=reuse)

    def _build_shared_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):
            
            self._embedding = tf.get_variable('embedding', shape=self._shape, trainable=True,
                                      initializer=self._initializer)

            if self._ids is not None:
                self._outputs.append(tf.nn.embedding_lookup(self._embedding, self._ids))

                if self._l2_reg is not None:
                    self._loss = self._l2_reg * tf.nn.l2_loss(self._outputs[0])
                    
            else:
                self._outputs.append(self._embedding)

    def censor_l2_norm_op(self, censor_id_list=None, max_norm=1):

        """Limit the norm of embeddings.
        
        Parameters
        ----------
        censor_id_list: list or Tensorflow tensor
            list of embeddings to censor (indexed by ids).
        max_norm: float, optional
            Maximum norm.

        Returns
        -------
        Tensorflow operator
            An operator for post-training execution.
        """

        
        embedding_gather = tf.gather(self._embedding, indices=censor_id_list)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_gather), axis=1, keep_dims=True))
        return tf.scatter_update(self._embedding, indices=censor_id_list, updates=embedding_gather / tf.maximum(norm, max_norm))

