import tensorflow as tf
import numpy as np

class Recommender(object):
    
    """
    The Recommender is the OpenRec abstraction [1]_ for recommendation algorithms.

    Parameters
    ----------
    batch_size: int
        Training batch size. The structure of a training instance varies across recommenders.
    max_user: int
        Maximum number of users in the recommendation system.
    max_item: int
        Maximum number of items in the recommendation system.
    extra_interactions_funcs: list, optional
        List of functions to build extra interaction modules.
    extra_fusions_funcs: list, optional
        List of functions to build extra fusion modules.
    test_batch_size: int, optional
        Batch size for testing and serving. The structure of a testing/serving instance varies across recommenders.
    l2_reg: float, optional
        Weight for L2 regularization, i.e., weight decay.
    opt: 'SGD'(default) or 'Adam', optional
        Optimization algorithm, SGD: Stochastic Gradient Descent.
    init_dict: dict, optional
        Key-value pairs for initial parameter values.
    sess_config: tensorflow.ConfigProto(), optional
        Tensorflow session configuration.

    Notes  
    -----
    .. highlight:: python

    The recommender abstraction defines the procedures to build a recommendation computational graph and exposes 
    interfaces for training and evaluation. During training, for each batch, the :code:`self.train` function should be 
    called with a :code:`batch_data` input,
    
    .. code:: python

        recommender_instance.train(batch_data)

    and during testing/serving, the *serve* function should be called with a *batch_data* input:

    .. code:: python

        recommender_instance.serve(batch_data)

    A recommender contains four major components: **inputs**, **extractions**, **fusions**, and 
    **interactions**. The figure below shows the order of which each related function is called. The 
    :code:`train` parameter in each function is used to build different computational graphs for training 
    and serving.

    .. image:: recommender.png
        :scale: 50 %
        :alt: The structure of the recommender abstraction
        :align: center

    A new recommender class should be inherent from the *Recommender* class. Follow the steps below to override 
    corresponding functions. To make a recommender easily extensible, it is **NOT** recommended to override functions 
    :code:`self._build_inputs`, :code:`self._build_fusions`, and :code:`self._build_interactions`. 
    
        * **Define inputs.** Override functions :code:`self._build_user_inputs`, :code:`self._build_item_inputs`, \
        and :code:`self._build_extra_inputs` to define inputs for users', items', and contextual data sources \
        respectively. An input should be defined using the *input* function as follows.
    
        .. code:: python

            self._add_input(name='input_name', dtype='float32', shape=data_shape, train=True)

        * **Define input mappings.** Override the function :code:`self._input_mappings` to feed a *batch_data* into \
        the defined inputs. The mapping should be specified using a python dict where a *key* corresponds to an input \
        object retrieved by :code:`self._get_input(input_name, train=train)`, and a *value* corresponds to a \
        :code:`batch_data` value.

        * **Define extraction modules.** Override functions :code:`self._build_user_extractions`, \
        :code:`self._build_item_extractions`, and :code:`self._build_extra_extractions` to define extraction \
        modules for users, items, and extra contexts respectively. Use :code:`self._add_module` to construct a \
        module, and :code:`self._get_input`/:code:`self._get_module` to retrieve an existing input/module.

        * **Define fusion modules.** Override the function :code:`self._build_default_fusions` to build fusion modules.\
        Custom functions can also be used as long as they are included in the input :code:`extra_fusions_funcs` list. \
        Use :code:`self._add_module` to construct a module, and :code:`self._get_input`/:code:`self._get_module` to \
        retrieve an existing input/module.

        * **Define interaction modules.** Override the fuction :code:`build_default_interactions` to build interaction \
        modules. Custom functions can also be used as long as they are included in the input \
        :code:`extra_interactions_funcs` list. Use :code:`self._add_module` to construct a module, and \
        :code:`self._get_input`/:code:`self._get_module` to retrieve an existing input/module.

    When (:code:`train==False`), a variable named :code:`self._scores` should be defined for *user-item scores*. \
    Such a score is higher if an item should be ranked higher in the recommendation list.

    References
    ----------
    .. [1] Yang, L., Bagdasaryan, E., Gruenstein, J., Hsieh, C., and Estrin, D., 2018, June. 
        OpenRec: A Modular Framework for Extensible and Adaptable Recommendation Algorithms.
        In Proceedings of WSDM'18, February 5-9, 2018, Marina Del Rey, CA, USA.
    """

    def __init__(self, batch_size, max_user, max_item, extra_interactions_funcs=[],
                    extra_fusions_funcs=[], test_batch_size=None, l2_reg=None, opt='SGD', lr=None, 
                    init_dict=None, sess_config=None):
        
        self._str_to_dtype = {
            'float16': tf.float16,
            'float32': tf.float32,
            'float64': tf.float64,
            'int8': tf.int8,
            'int16': tf.int16,
            'int32': tf.int32,
            'int64': tf.int64,
            'bool': tf.bool,
            'string': tf.string
        }
        
        self._batch_size = batch_size
        self._test_batch_size = test_batch_size
        self._max_user = max_user
        self._max_item = max_item
        self._l2_reg = l2_reg
        self._opt = opt

        if lr is None:
            if self._opt == 'Adam':
                self._lr = 0.001
            elif self._opt == 'SGD':
                self._lr = 0.005
        else:
            self._lr = lr

        self._loss_nodes = []
        self._inputs_store = {'train':{}, 'serving': {}}
        self._modules_store = {'train':{}, 'serving': {}}
        
        self._interactions_funcs = [self._build_default_interactions] + extra_interactions_funcs
        self._fusions_funcs = [self._build_default_fusions] + extra_fusions_funcs

        self._build_training_graph()
        self._build_post_training_graph()
        self._build_serving_graph()
        if sess_config is None:
            self._sess = tf.Session()
        else:
            self._sess = tf.Session(config=sess_config)
        self._initialize(init_dict)
        self._saver = tf.train.Saver(max_to_keep=None)

    
    def train(self, batch_data):

        """Train the model with an input batch_data.

        Parameters
        ----------
        batch_data: dict
            A batch of training data.
        """

        _, loss = self._sess.run([self._train_op, self._loss],
                                 feed_dict=self._input_mappings(batch_data, train=True))
        return loss

    def serve(self, batch_data):

        """Evaluate the model with an input batch_data.

        Parameters
        ----------
        batch_data: dict
            A batch of testing or serving data.
        """

        scores = self._sess.run(self._scores, 
                            feed_dict=self._input_mappings(batch_data, train=False))

        return scores

    def compute_module_outputs(self, name, batch_data, train=True):

        """Compute the outputs of a module, specified by the name and the train flag.

        Parameters
        ----------
        name: str
            The module name.
        batch_data: dict
            A batch of training or serving data.
        train: bool
            Specify the computational graph (train/serving) to compute outputs.
        
        Returns
        -------
        A list of Numpy arrays
            The outputs of the specified module.
        """
        
        module = self._get_module(name=name, train=train)
        
        if len(module.get_outputs()) == 0:
            return []
        
        return self._sess.run(module.get_outputs(), feed_dict=self._input_mappings(batch_data, train=train))

    def compute_module_loss(self, name, batch_data, train=True):

        """Compute the loss of a module, specified by the name and the train flag.

        Parameters
        ----------
        name: str
            The module name.
        batch_data: dict
            A batch of training or serving data.
        train: bool
            Specify the computational graph (train/serving) to compute loss.
        
        Returns
        -------
        Numpy array
            The loss of the specified module.
        """

        module = self._get_module(name=name, train=train)
        
        if type(module.get_loss()) == float:
            return module.get_loss()

        return self._sess.run(module.get_loss(), feed_dict=self._input_mappings(batch_data, train=train))

    def save(self, save_dir, step):

        """Save a trained model to disk.

        Parameters
        ----------
        save_str: str
            Path to save the model.
        step: int
            training step.
        """

        self._saver.save(self._sess, save_dir, global_step=step)

    def load(self, load_dir):

        """Load a saved model from disk.

        Parameters
        ----------
        load_str: str
            Path to the saved model.
        """

        self._saver.restore(self._sess, load_dir)
    
    def _add_module(self, name, module, train_loss=None, train=True):
        
        """Add a module - overwrite if :code:`name` exists.

        Parameters
        ----------
        name: str
            Module name.
        module: Module
            Module instance.
        train_loss: bool, optional
            Whether or not to include the output loss in the training loss (Default: include losses from all modules).
        train: bool, optional
            Specify the computational graph (train/serving) to add the module.
        """
        if train_loss is None:
            train_loss = train
            
        if train:
            self._modules_store['train'][name] = module
        else:
            self._modules_store['serving'][name] = module
        
        if train_loss:
            self._loss_nodes.append(module)
    
    def _get_module(self, name, train=True):
        
        """Retrieve a module.

        Parameters
        ----------
        name: str
            The module name.
        train: bool
            Specify training or serving graph.

        Returns
        -------
        Module
            The module specified by the name and the :code:`train` flag.
        """
        if train:
            return self._modules_store['train'][name]
        else:
            return self._modules_store['serving'][name]
    
    def _add_input(self, name, dtype='float32', shape=None, train=True):
        
        """Add an input - overwrite if :code:`name` exists.

        Parameters
        ----------
        name: str
            The input name.
        dtype: str
            Data type: "float16", "float32" (default), "float64", "int8", "int16", "int32", "int64", "bool", "string" \
            or "none".
        shape: list or tuple
            Input shape.
        train: bool
            Specify training or serving graph.
        """
        
        if train:
            if dtype=='none':
                self._inputs_store['train'][name] = None
            else:
                self._inputs_store['train'][name] = self._input(dtype=dtype, shape=shape, name=name+'_train')
        else:
            if dtype=='none':
                self._inputs_store['serving'][name] = None
            else:
                self._inputs_store['serving'][name] = self._input(dtype=dtype, shape=shape, name=name+'_serving')
    
    def _get_input(self, name, train=True):
        
        """Retrieve an input.

        Parameters
        ----------
        name: str
            Input name.
        train: bool
            Specify training or serving graph.

        Returns
        -------
        Tensorflow placeholder
            The input specified by the name and the :code:`train` flag.
        """
        if train:
            return self._inputs_store['train'][name]
        else:
            return self._inputs_store['serving'][name]
        
    def _initialize(self, init_dict):

        """Initialize model parameters (do NOT override).

        Parameters
        ----------
        init_dict: dict
            Key-value pairs for initial parameter values.

        """
        if init_dict is None:
            self._sess.run(tf.global_variables_initializer())
        else:
            self._sess.run(tf.global_variables_initializer(), feed_dict=init_dict)

    
    def _input(self, dtype='float32', shape=None, name=None):
        
        """Define an input for the recommender.

        Parameters
        ----------
        dtype: str
            Data type: "float16", "float32", "float64", "int8", "int16", "int32", "int64", "bool", or "string".
        shape: list or tuple
            Input shape.
        name: str
            Name of the input.

        Returns
        -------
        Tensorflow placeholder
            Defined tensorflow placeholder.
        """
        if dtype not in self._str_to_dtype:
            raise ValueError
        else:
            return tf.placeholder(self._str_to_dtype[dtype], shape=shape, name=name)

    def _input_mappings(self, batch_data, train):
        
        """Define mappings from input training batch to defined inputs.

        Parameters
        ----------
        batch_data: dict
            A training batch.
        train: bool
            An indicator for training or servining phase.

        Returns
        -------
        dict
            The mapping where a *key* corresponds to an input object, and a *value* corresponds to a \
            :code:`batch_data` value.
        """

        return {}

    def _build_inputs(self, train=True):

        """Call sub-functions to build inputs (do NOT override).

        Parameters
        ----------
        train: bool
            An indicator for training or servining phase.

        """
        
        self._build_user_inputs(train=train)
        self._build_item_inputs(train=train)
        self._build_extra_inputs(train=train)

    def _build_user_inputs(self, train=True):

        """Build inputs for users' data sources (should be overriden)

        Parameters
        ----------
        train: bool
            An indicator for training or servining phase.

        """

        pass

    def _build_item_inputs(self, train=True):
        
        """Build inputs for items' data sources (should be overriden)

        Parameters
        ----------
        train: bool
            An indicator for training or servining phase.

        """

        pass

    def _build_extra_inputs(self, train=True):

        """Build inputs for contextual data sources (should be overriden)

        Parameters
        ----------
        train: bool
            An indicator for training or servining phase.

        """

        pass

    def _build_extractions(self, train=True):

        """Call sub-functions to build extractions (do NOT override).

        Parameters
        ----------
        train: bool
            An indicator for training or servining phase.

        """
        
        self._build_user_extractions(train=train)
        self._build_item_extractions(train=train)
        self._build_extra_extractions(train=train)
        
    def _build_user_extractions(self, train=True):

        """Build extraction modules for users' data sources (should be overriden)

        Parameters
        ----------
        train: bool
            An indicator for training or servining phase.

        """

        pass

    def _build_item_extractions(self, train=True):

        """Build extraction modules for items' data sources (should be overriden)

        Parameters
        ----------
        train: bool
            An indicator for training or servining phase.

        """

        pass

    def _build_extra_extractions(self, train=True):

        """Build extraction modules for contextual data sources (may be overriden)

        Parameters
        ----------
        train: bool
            An indicator for training or servining phase.

        """

        pass

    def _build_fusions(self, train=True):

        """Call sub-functions to build fusions (do NOT override).

        Parameters
        ----------
        train: bool
            An indicator for training or servining phase.

        """

        for func in self._fusions_funcs:
            func(train)
    
    def _build_default_fusions(self, train=True):

        """Build default fusion modules (may be overriden).

        Parameters
        ----------
        train: bool
            An indicator for training or servining phase.

        """

        pass

    def _build_interactions(self, train=True):

        """Call sub-functions to build interactions (do NOT override).

        Parameters
        ----------
        train: bool
            An indicator for training or servining phase.

        """
        
        for func in self._interactions_funcs:
            func(train)

    def _build_default_interactions(self, train=True):

        """Build default interaction modules (may be overriden).

        Parameters
        ----------
        train: bool
            An indicator for training or servining phase.

        """

        pass

    def _build_post_training_ops(self):

        """Build post-training operators (may be overriden).

        Returns
        -------
        list
            A list of Tensorflow operators.
        """
        return []

    def _build_optimizer(self):

        """Build an optimizer for model training.
        """
        
        self._loss = tf.add_n([node.get_loss() for node in self._loss_nodes])

        if self._opt == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)

        grad_var_list = optimizer.compute_gradients(self._loss)
        self._train_op = optimizer.apply_gradients(self._grad_post_processing(grad_var_list))

    def _grad_post_processing(self, grad_var_list):

        """Post-process gradients before updating variables.
        
        Parameters
        ----------
        grad_var_list: list
            A list of tuples (gradients, variable).

        Returns
        -------
        list
            A list of updated tuples (updated gradients, variables).
        """

        return grad_var_list

    def _build_training_graph(self):

        """Call sub-functions to build training graph (do NOT override).
        """

        self._loss_nodes = []
        self._build_inputs(train=True)
        self._build_extractions(train=True)
        self._build_fusions(train=True)
        self._build_interactions(train=True)
        self._build_optimizer()

    def _build_post_training_graph(self):

        """Build post-training graph (do NOT override).
        """

        if hasattr(self, '_train_op'):
            with tf.control_dependencies([self._train_op]):
                self._post_training_op = self._build_post_training_ops()

    def _build_serving_graph(self):

        """Call sub-functions to build serving graph (do NOT override).
        """

        self._build_inputs(train=False)
        self._build_extractions(train=False)
        self._build_fusions(train=False)
        self._build_interactions(train=False)
