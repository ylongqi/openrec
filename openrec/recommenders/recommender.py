import tensorflow as tf
import numpy as np
import os

class _RecommenderGraph(object):

    class _SubGraph(object):

        def __init__(self, rec_graph):

            self.super = rec_graph
            self._tensor_store = dict()
            self._build_funcs = []
            self._is_built = False

        def __call__(self, keys):
            for key in keys:
                self._tensor_store[key] = None
            def assign_build_func(build_func):
                self._build_funcs = [build_func]
                return build_func
            return assign_build_func

        def add(self, keys):
            for key in keys:
                self._tensor_store[key] = None
            def add_build_func(build_func):
                self._build_funcs.append(build_func)
                return build_func
            return add_build_func

        def build(self):
            if not self._is_built:
                for build_func in self._build_funcs:
                    build_func(self)
                self._is_built = True

        def copy(self, sub_graph):
            self._build_funcs = sub_graph.build_funcs
            self._tensor_store = sub_graph.tensor_store.copy()

        @property
        def tensor_store(self):
            return self._tensor_store

        @property
        def build_funcs(self):
            return self._build_funcs

        def set(self, key, value):
            assert key in self._tensor_store, "\"%s\" Tensor is not registered" % key
            self._tensor_store[key] = value

        def get(self, key):
            assert key in self._tensor_store, "\"%s\" Tensor is not registered" % key
            if not self._is_built:
                self._build_func(self)
                self._is_built = True
            assert self._tensor_store[key] is not None, "Registered \"%s\" Tensor is not defined" % key
            return self._tensor_store[key]

    def __init__(self):
        
        self._tf_graph = tf.Graph()
        self.InputGraph = self._SubGraph(self)
        self.UserGraph = self._SubGraph(self)
        self.ItemGraph = self._SubGraph(self)
        self.ContextGraph = self._SubGraph(self)
        self.FusionGraph = self._SubGraph(self)
        self.InteractionGraph = self._SubGraph(self)
        self.OptimizerGraph = self._SubGraph(self)

        self._train_op_identifier_set = set()
        self._loss_identifier_set = set()
        self._output_identifier_set = set()
        self._input_mapping_dict = dict()

    def register_input_mapping(self, input_mapping, identifier='default'):

        self._input_mapping_dict[identifier] = input_mapping

    def register_train_op(self, train_op, identifier='default'):

        self._train_op_identifier_set.add(identifier)
        tf.add_to_collection('openrec.recommender.train_ops.'+identifier, train_op)

    def register_loss(self, loss, identifier='default'):

        self._loss_identifier_set.add(identifier)
        tf.add_to_collection('openrec.recommender.losses.'+identifier, loss)

    def register_output(self, output, identifier='default'):
        
        self._output_identifier_set.add(identifier)
        tf.add_to_collection('openrec.recommender.outputs.'+identifier, output)
    
    @property
    def tf_graph(self):
        return self._tf_graph

    def build(self):

         with self._tf_graph.as_default():
            self.InputGraph.build()
            self.UserGraph.build()
            self.ItemGraph.build()
            self.FusionGraph.build()
            self.InteractionGraph.build()
            self.OptimizerGraph.build()

    def get_input_mapping(self, identifier='default'):

        return self._input_mapping_dict[identifier]

    def get_train_ops(self, identifier='default'):
        
        with self._tf_graph.as_default():
            return tf.get_collection('openrec.recommender.train_ops.'+identifier)

    def get_losses(self, identifier='default'):
        
        with self._tf_graph.as_default():
            return tf.get_collection('openrec.recommender.losses.'+identifier)

    def get_outputs(self, identifier='default'):

        with self._tf_graph.as_default():
            return tf.get_collection('openrec.recommender.outputs.'+identifier)

class Recommender(object):

    def __init__(self, _sentinel=None, init_model_dir=None, save_model_dir=None, training=True, serving=False):

        self._training = training
        self._serving = serving
        self._init_model_dir = init_model_dir
        self._save_model_dir = save_model_dir
        self._flag_updated = False
        
        self.TrainingGraph = _RecommenderGraph()
        self.ServingGraph = _RecommenderGraph()

        self.T = self.TrainingGraph
        self.S = self.ServingGraph
    
    def _generate_feed_dict(self, batch_data, input_map):

        feed_dict = dict()
        for key in input_map:
            feed_dict[input_map[key]] = batch_data[key]
        return feed_dict

    def train(self, batch_data, input_mapping_id='default', train_ops_id='default', losses_id='default', outputs_id='default'):

        assert self._training, "Training is disabled"
        feed_dict = self._generate_feed_dict(batch_data, 
                                            self.T.get_input_mapping(input_mapping_id))
        train_ops = self.T.get_train_ops(train_ops_id)
        losses_nodes = self.T.get_losses(losses_id)
        if len(losses_nodes) > 0:
            losses = [tf.add_n(losses_nodes)]
        else:
            losses = []
        outputs = self.T.get_outputs(outputs_id)
        results = self._tf_training_sess.run(train_ops+losses+outputs,
                                 feed_dict=feed_dict)
        
        return_dict = {'losses': results[len(train_ops):len(train_ops)+len(losses)],
                      'outputs': results[-len(outputs):]}
        
        self._flag_updated = True
        return return_dict

    def serve(self, batch_data, input_mapping_id='default', losses_id='default', outputs_id='default'):

        assert self._serving, "Serving is disabled"
        
        if self._flag_updated:
            self._save_and_load_for_serving()
            self._flag_updated = False
            
        feed_dict = self._generate_feed_dict(batch_data, self.S.get_input_mapping(input_mapping_id))
        losses_nodes = self.S.get_losses(losses_id)
        if len(losses_nodes) > 0:
            losses = [tf.add_n(losses_nodes)]
        else:
            losses = []
        outputs = self.S.get_outputs(outputs_id)
        results = self._tf_serving_sess.run(losses+outputs, 
                            feed_dict=feed_dict)

        return {'losses': results[:len(losses)], 'outputs': results[len(losses):]}

    def _save_and_load_for_serving(self):
        
        assert self._save_model_dir is not None, 'save_model_dir is not specified'
        if self._training:
            self._tf_training_saver.save(self._tf_training_sess, os.path.join(self._save_model_dir, 'model.ckpt'))
        if self._serving:
            self._tf_serving_saver.restore(self._tf_serving_sess, os.path.join(self._save_model_dir, 'model.ckpt'))
    
    def _optimistic_restore(self, session, save_file):
        
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                if var.name.split(':')[0] in saved_shapes and len(var.shape) > 0])
        restore_vars = []
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = tf.get_variable(saved_var_name)
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_file)
            
    def build(self):

        if self._training:
            self.TrainingGraph.build()
            with self.TrainingGraph.tf_graph.as_default():
                self._tf_training_sess = tf.Session()
                self._tf_training_sess.run(tf.global_variables_initializer())
                self._tf_training_saver = tf.train.Saver(tf.global_variables())
                if self._init_model_dir is not None:
                    self._optimistic_restore(self._tf_training_sess, os.path.join(self._init_model_dir, 'model.ckpt'))

        if self._serving:
            self.ServingGraph.build()
            with self.ServingGraph.tf_graph.as_default():
                self._tf_serving_sess = tf.Session()
                self._tf_serving_sess.run(tf.global_variables_initializer())
                self._tf_serving_saver = tf.train.Saver(tf.global_variables())
                if self._init_model_dir is not None:
                    self._optimistic_restore(self._tf_serving_sess, os.path.join(self._init_model_dir, 'model.ckpt'))

        return self