import tensorflow as tf
import numpy as np

class Recommender(object):

    def __init__(self, batch_size, test_batch_size, max_user, max_item, l2_reg, opt='SGD', extra_interactions_funcs=[],
                    extra_fusions_funcs=[], lr=None, init_dict=None, sess_config=None):

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
                self._lr = 0.05
        else:
            self._lr = lr

        self._loss_nodes = []
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

    def _initialize(self, init_dict):
        if init_dict is None:
            self._sess.run(tf.global_variables_initializer())
        else:
            self._sess.run(tf.global_variables_initializer(), feed_dict=init_dict)

    def train(self, batch_data):
        _, loss = self._sess.run([self._train_op, self._loss],
                                 feed_dict=self._input_mappings(batch_data, train=True))
        return loss

    def serve(self, batch_data):
        scores = self._sess.run(self._scores, 
                            feed_dict=self._input_mappings(batch_data, train=False))

        return scores
    
    def save(self, save_dir, step):
        self._saver.save(self._sess, save_dir, global_step=step)

    def load(self, load_dir):
        self._saver.restore(self._sess, load_dir)
    
    def _input(self, dtype='float32', shape=None, name=None):
        
        exec("tf_dtype = tf.%s" % dtype)
        return tf.placeholder(tf_dtype, shape=shape, name=name)

    def _input_mappings(self, batch_data, train):
        return {}

    def _build_inputs(self, train=True):
        
        self._build_user_inputs(train=train)
        self._build_item_inputs(train=train)
        self._build_extra_inputs(train=train)

    def _build_user_inputs(self, train=True):
        return None

    def _build_item_inputs(self, train=True):
        return None

    def _build_extra_inputs(self, train=True):
        return None

    def _build_extractions(self, train=True):
        
        self._build_user_extractions(train=train)
        self._build_item_extractions(train=train)
        self._build_extra_extractions(train=train)
        
    def _build_user_extractions(self, train=True):
        return None

    def _build_item_extractions(self, train=True):
        return None

    def _build_extra_extractions(self, train=True):
        return None

    def _build_fusions(self, train=True):
        
        for func in self._fusions_funcs:
            func(train)
    
    def _build_default_fusions(self, train=True):
        return None

    def _build_interactions(self, train=True):
        
        for func in self._interactions_funcs:
            func(train)

    def _build_default_interactions(self, train=True):
        return None

    def _build_post_training_ops(self):
        return []

    def _build_optimizer(self):
        
        self._loss = tf.add_n([node.get_loss() for node in self._loss_nodes])

        if self._opt == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)

        grad_var_list = optimizer.compute_gradients(self._loss)
        self._train_op = optimizer.apply_gradients(self._grad_post_processing(grad_var_list))

    def _grad_post_processing(self, grad_var_list):
        return grad_var_list

    def _build_training_graph(self):
        self._loss_nodes = []
        self._build_inputs(train=True)
        self._build_extractions(train=True)
        self._build_fusions(train=True)
        self._build_interactions(train=True)
        self._build_optimizer()

    def _build_post_training_graph(self):

        if hasattr(self, '_train_op'):
            with tf.control_dependencies([self._train_op]):
                self._post_training_op = self._build_post_training_ops()

    def _build_serving_graph(self):

        self._build_inputs(train=False)
        self._build_extractions(train=False)
        self._build_fusions(train=False)
        self._build_interactions(train=False)
