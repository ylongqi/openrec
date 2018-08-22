import tensorflow as tf
import numpy as np
import os    
    
class _RecommenderGraph(object):

    class _SubGraph(object):
        
        class _Port(object):
            
            def __init__(self):
                self.s = None
        
        class _InPort(_Port):
            
            def assign(self, subgraph, key):
                self.s = {'subgraph':subgraph, 'key':key}
            
            def retrieve(self):
                if self.s is None:
                    return None
                else:
                    return self.s['subgraph'][self.s['key']]
        
        class _OutPort(_Port):
            
            def assign(self, tensor):
                self.s = tensor
            
            def retrieve(self):
                return self.s

        def __init__(self, rec_graph):

            self.super = rec_graph
            self._port_store = dict()
            self._build_funcs = []
            self._build_mode = False
            self._is_built = False
            
        def __getitem__(self, key):
            
            assert key in self._port_store, "%s port is not found." % key
            if self._build_mode:
                if not self._is_built:
                    self.build()
                    self._is_built = True
                return self._port_store[key].retrieve()
            else:
                assert isinstance(self._port_store[key], self._OutPort), "[Connect Error] Getting a value from the %s in-port" % key
                return self, key
        
        def __setitem__(self, key, value):
            
            assert key in self._port_store, "%s port is not found." % key
            if self._build_mode:
                assert isinstance(self._port_store[key], self._OutPort), "[Build Error] Assigning a value to the %s in-port" % key
                self._port_store[key].assign(value)
            else:
                assert isinstance(self._port_store[key], self._InPort), "[Connect Error] Assigning a value to the %s out-port" % key
                self._port_store[key].assign(value[0], value[1])
                
        def __call__(self, ins=[], outs=[]):
            
            assert isinstance(ins, list), "ins should be a list of strings."
            assert isinstance(outs, list), "outs should be a list of strings"
            for in_ in ins:
                self._port_store[in_] = self._InPort()
            for out_ in outs:
                self._port_store[out_] = self._OutPort()
            
            def add_build_func(build_func):
                self._build_funcs = [build_func]
                return build_func
            return add_build_func

#         def ins(self, *keys):
            
#             for key in keys:
#                 self._port_store[key] = self._InPort()
#             return self
            
#         def outs(self, *keys):
            
#             for key in keys:
#                 self._port_store[key] = self._OutPort()
#             return self
            
        def add(self, ins=[], outs=[]):
            for in_ in ins:
                self._port_store[in_] = self._InPort()
            for out_ in outs:
                self._port_store[out_] = self._OutPort()
            
            def add_build_func(build_func):
                self._build_funcs.append(build_func)
                return build_func
            return build_func
        
        def ready(self):
            
            self._build_mode = True
            
        def build(self):
            if not self._is_built:
                self._is_built = True
                for build_func in self._build_funcs:
                    build_func(self)
        
        def register_global_input_mapping(self, input_mapping, identifier='default'):

            self.super.register_input_mapping(input_mapping, identifier)

        def register_global_train_op(self, train_op, identifier='default'):
            
            self.super.register_train_op(train_op, identifier)

        def register_global_loss(self, loss, identifier='default'):

            self.super.register_loss(loss, identifier)

        def register_global_output(self, output, identifier='default'):
            
            self.super.register_output(output, identifier)
        
        def get_global_input_mapping(self, identifier='default'):

            self.super.get_input_mapping(identifier)

        def get_global_train_ops(self, identifier='default'):

            return self.super.get_train_ops(identifier)

        def get_global_losses(self, identifier='default'):

            return self.super.get_losses(identifier)
            
        def get_global_outputs(self, identifier='default'):

            return self.super.get_outputs(identifier)
            
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
            
            self.InputGraph.ready()
            self.UserGraph.ready()
            self.ItemGraph.ready()
            self.FusionGraph.ready()
            self.InteractionGraph.ready()
            self.OptimizerGraph.ready()
            
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
        self._flag_isbuilt = False
        
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
        assert self._flag_isbuilt, "Training graph is not built"
        
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
    
    def get_parameter(self, name):
        
        tensor = self.TrainingGraph.tf_graph.get_tensor_by_name(name)
        return self._tf_training_sess.run(tensor)
        
    def serve(self, batch_data, input_mapping_id='default', losses_id='default', outputs_id='default'):

        assert self._serving, "Serving is disabled"
        assert self._flag_isbuilt, "Serving graph is not built"
        
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
    
    def save(self, save_model_dir=None, global_step=None):
        
        if save_model_dir is None:
            save_model_dir = self._save_model_dir
        with self.TrainingGraph.tf_graph.as_default():
            self._tf_training_saver.save(self._tf_training_sess, 
                                         os.path.join(save_model_dir, 'model.ckpt'),
                                        global_step=global_step)
    
    def restore(self, save_model_dir=None, restore_training=False, restore_serving=False):
        
        if save_model_dir is None:
            save_model_dir = self._save_model_dir
        if restore_training:
            assert self._training is not None, 'Training is not enabled.'
            with self.TrainingGraph.tf_graph.as_default():
                self._optimistic_restore(self._tf_training_sess, os.path.join(save_model_dir, 'model.ckpt'))
        if restore_serving:
            assert self._serving is not None, 'Serving is not enabled.'
            with self.ServingGraph.tf_graph.as_default():
                self._optimistic_restore(self._tf_serving_sess, os.path.join(save_model_dir, 'model.ckpt'))
            
    def _save_and_load_for_serving(self):
        
        assert self._save_model_dir is not None, 'save_model_dir is not specified'
        if self._training:
            self.save()
        if self._serving:
            self.restore(restore_serving=True)
    
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
        print('... restored variables:', ','.join([var.name for var in restore_vars]))
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_file)
            
    def build(self):

        if self._training:
            self.TrainingGraph.build()
            with self.TrainingGraph.tf_graph.as_default():
                self._tf_training_sess = tf.Session()
                self._tf_training_sess.run(tf.global_variables_initializer())
                self._tf_training_saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        if self._serving:
            self.ServingGraph.build()
            with self.ServingGraph.tf_graph.as_default():
                self._tf_serving_sess = tf.Session()
                self._tf_serving_sess.run(tf.global_variables_initializer())
                self._tf_serving_saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        
        if self._init_model_dir is not None:
            self.restore(save_model_dir=self._init_model_dir,
                        restore_training=self._training,
                        restore_serving=self._serving)
        
        self._flag_isbuilt = True
        
        return self
    
    def isbuilt(self):
        
        return self._flag_isbuilt