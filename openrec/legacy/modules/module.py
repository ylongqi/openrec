import tensorflow as tf


class Module(object):

    """
    The module is the OpenRec abstraction for modules. A module may belong to one of the three categories, \
    **extractions**, **fusions**, and **interactions**, depending on its functionality (Read [1]_ for details).

    Parameters
    ----------
    train: bool, optional
        An indicator for training or servining phase.
    l2_reg: float, optional
        Weight for L2 regularization, i.e., weight decay.
    scope: str, optional
        Scope for module variables.
    reuse: bool, optional
        Whether or not to reuse module variables.

    Notes
    -----
    The module abstraction is used to construct recommenders. It should be extended by all module implementations. \
    During initialization, functions :code:`self._build_shared_graph`, :code:`self._build_training_graph`, and \
    :code:`self._build_serving_graph` are called as follows.
    
    .. image:: module.png
        :scale: 50 %
        :alt: The structure of the module abstraction
        :align: center

    A module implementation should follow two steps below:
    
    * **Build computational graphs.** Override :code:`self._build_shared_graph()`, :code:`self._build_training_graph()`,\
     and/or :code:`self._build_serving_graph()` functions to build training/serving computational graphs.

    * **Define a loss and an output list.** Define a loss (:code:`self._loss`) to be included in training and an output \
    list of Tensorflow tensors (:code:`self._outputs`).

    References
    ----------
    .. [1] Yang, L., Bagdasaryan, E., Gruenstein, J., Hsieh, C., and Estrin, D., 2018, June. 
        OpenRec: A Modular Framework for Extensible and Adaptable Recommendation Algorithms.
        In Proceedings of WSDM'18, February 5-9, 2018, Marina Del Rey, CA, USA.
    """

    def __init__(self, train=True, l2_reg=None, scope=None, reuse=False):

        self._scope = self.__class__.__name__ if scope is None else scope
        self._reuse = reuse
        self._l2_reg = l2_reg

        self._loss = 0.0
        self._outputs = []
        self._train = train
        
        if train:
            self._build_shared_graph()
            self._build_training_graph()
        else:
            self._build_shared_graph()
            self._build_serving_graph()

    def _build_shared_graph(self):

        """Build shared computational graphs across training and serving (may be overridden).
        """
        pass

    def _build_training_graph(self):

        """Build training-specific computational graphs (may be overridden).
        """
        pass

    def _build_serving_graph(self):

        """Build serving-specific computational graphs (may be overridden).
        """
        pass

    def get_outputs(self):

        """Retrieve the output list of Tensorflow tensors.

        Returns
        -------
        list
            An output list of Tensorflow tensors
        """

        return self._outputs

    def get_loss(self):

        """Retrieve the training loss.

        Returns
        -------
        float or Tensor
            Training loss
        """

        return self._loss

