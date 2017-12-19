from __future__ import print_function
import tensorflow as tf
from termcolor import colored
from openrec.modules.extractions import Extraction
from openrec.modules.extractions import MultiLayerFC

class SDAE(Extraction):

    """
    The SDAE module implements Stacked Denoising Autoencoders [bn]_. It outputs SDAE's bottleneck representations \
    (i.e., the encoder outputs). 

    Parameters
    ----------
    in_tensor: Tensorflow tensor
        An input tensor with shape **[*, feature dimensionality]**
    dims: list
        Specify the *feature size* of each **encoding layer**'s outputs. For example, setting **dims=[512, 258, 128]** to create \
        an three-layer encoder with output shape **[*, 512]**, **[*, 256]**, and **[*, 128]**, and a two-layer decoder with \
        output shape **[*, 256]** and **[*, 512]**.
    dropout: float, optional
        Dropout rate for the input tensor. If *None*, no dropout is used for the input tensor.
    l2_reconst: float, optional
        Weight for reconstruction loss.
    train: bool, optionl
        An indicator for training or servining phase.
    l2_reg: float, optional
        Weight for L2 regularization, i.e., weight decay.
    scope: str, optional
        Scope for module variables.
    reuse: bool, optional
        Whether or not to reuse module variables.
    
    References
    ----------
    .. [bn] Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y. and Manzagol, P.A., 2010. Stacked denoising autoencoders: \
        Learning useful representations in a deep network with a local denoising criterion. Journal of Machine Learning \
        Research, 11(Dec), pp.3371-3408.
    """

    def __init__(self, in_tensor, dims, dropout=None, 
        l2_reconst=1.0, train=True, l2_reg=None, scope=None, reuse=False):

        assert dims is not None, 'dims cannot be None'
        assert in_tensor is not None, 'in_tensor cannot be None'

        self._in_tensor = in_tensor
        self._dims = dims
        self._dropout = dropout
        self._l2_reconst = l2_reconst

        super(SDAE, self).__init__(train=train, l2_reg=l2_reg, scope=scope, reuse=reuse)

    def _build_shared_graph(self):

        with tf.variable_scope(self._scope, reuse=self._reuse):

            _encoder = MultiLayerFC(l2_reg=self._l2_reg, in_tensor=self._in_tensor, dims=self._dims[1:], scope='encoder',
                            dropout_in=self._dropout, dropout_mid=self._dropout, reuse=self._reuse)
            _decoder = MultiLayerFC(l2_reg=self._l2_reg, in_tensor=_encoder.get_outputs()[0], dims=self._dims[::-1][1:],
                            scope='decoder', relu_in=True, dropout_in=self._dropout, relu_mid=True,
                            dropout_mid=self._dropout, relu_out=True, dropout_out=self._dropout, reuse=self._reuse)

            self._outputs += _encoder.get_outputs()
            self._loss = _encoder.get_loss() + _decoder.get_loss()
            self._loss += self._l2_reconst * tf.nn.l2_loss(_decoder.get_outputs()[0] - self._in_tensor)
