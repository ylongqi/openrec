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

            self._super = rec_graph
            self._port_store = dict()
            self._build_funcs = []
            self._build_mode = False
            self._is_built = False
            
        def __getitem__(self, key):
            
            assert key in self._port_store, "%s port is not found." % key
            if self._build_mode:
                assert self._is_built, "[Build Error] Getting a value from an unconstructed graph."
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
                
        def __call__(self, build_func=None, ins=[], outs=[], overwrite=True):
            
            assert isinstance(ins, list), "ins should be a list of strings."
            assert isinstance(outs, list), "outs should be a list of strings"
            
            if overwrite:
                self._port_store = {}
                self._build_funcs = []
            
            for in_ in ins:
                self._port_store[in_] = self._InPort()
            for out_ in outs:
                self._port_store[out_] = self._OutPort()
            
            if build_func is None:
                def add_build_func(build_func):
                    self._build_funcs.append(build_func)
                    return build_func
                return add_build_func
            else:
                self._build_funcs.append(build_func)
                return build_func
        
        def get_intrinsics(self):
            
            return self._port_store, self._build_funcs
            
        def copy(self, subgraph):
            
            self._port_store, self._build_funcs = subgraph.get_intrinsics()
        
        def ready(self):
            
            self._build_mode = True
            
        def build(self):
            if not self._is_built:
                self._is_built = True
                for build_func in self._build_funcs:
                    build_func(self)
        
        def register_global_input_mapping(self, input_mapping, identifier='default'):

            self._super.register_input_mapping(input_mapping, identifier)
        
        def update_global_input_mapping(self, update_input_mapping, identifier='default'):
            
            self._super.update_input_mapping(update_input_mapping, identifier)

        def register_global_train_op(self, train_op, identifier='default'):
            
            self._super.register_train_op(train_op, identifier)

        def register_global_loss(self, loss, identifier='default'):

            self._super.register_loss(loss, identifier)

        def register_global_output(self, output, identifier='default'):
            
            self._super.register_output(output, identifier)
        
        def get_global_input_mapping(self, identifier='default'):

            self._super.get_input_mapping(identifier)

        def get_global_train_ops(self, identifier='default'):

            return self._super.get_train_ops(identifier)

        def get_global_losses(self, identifier='default'):

            return self._super.get_losses(identifier)
            
        def get_global_outputs(self, identifier='default'):

            return self._super.get_outputs(identifier)
            
    def __init__(self):
        
        self._tf_graph = tf.Graph()
        self.inputgraph = self._SubGraph(self)
        self.usergraph = self._SubGraph(self)
        self.itemgraph = self._SubGraph(self)
        self.contextgraph = self._SubGraph(self)
        self.fusiongraph = self._SubGraph(self)
        self.interactiongraph = self._SubGraph(self)
        self.optimizergraph = self._SubGraph(self)

        self._train_op_identifier_set = set()
        self._loss_identifier_set = set()
        self._output_identifier_set = set()
        self._input_mapping_dict = dict()
        
        self._connect_funcs = []
    
    def __setattr__(self, name, value):
        
        if name in set(['inputgraph', 'usergraph', 'itemgraph', 'contextgraph',
                      'fusiongraph', 'interactiongraph', 'optimizergraph']):
            if name in self.__dict__:
                self.__dict__[name].copy(value)
            else:
                self.__dict__[name] = value
        else:
            self.__dict__[name] = value

    def register_input_mapping(self, input_mapping, identifier='default'):

        self._input_mapping_dict[identifier] = input_mapping
    
    def update_input_mapping(self, update_input_mapping, identifier='default'):
        
        self._input_mapping_dict[identifier].update(update_input_mapping)

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
            
            assert len(self._connect_funcs)>0, "Graph connection is not specified"
            for connect_func in self._connect_funcs:
                connect_func(self)
                
            self.inputgraph.ready()
            self.usergraph.ready()
            self.itemgraph.ready()
            self.contextgraph.ready()
            self.fusiongraph.ready()
            self.interactiongraph.ready()
            self.optimizergraph.ready()
            
            with tf.variable_scope('inputgraph', reuse=tf.AUTO_REUSE):
                self.inputgraph.build()
            with tf.variable_scope('usergraph', reuse=tf.AUTO_REUSE):
                self.usergraph.build()
            with tf.variable_scope('itemgraph', reuse=tf.AUTO_REUSE):
                self.itemgraph.build()
            with tf.variable_scope('contextgraph', reuse=tf.AUTO_REUSE):
                self.contextgraph.build()
            with tf.variable_scope('fusiongraph', reuse=tf.AUTO_REUSE):
                self.fusiongraph.build()
            with tf.variable_scope('interactiongraph', reuse=tf.AUTO_REUSE):
                self.interactiongraph.build()
            with tf.variable_scope('optimizergraph', reuse=tf.AUTO_REUSE):  
                self.optimizergraph.build()
    
    def connect(self, connect_func=None, overwrite=True):

        if overwrite:
            self._connect_funcs = []

        if connect_func is None:
            def add_connect_func(connect_func):
                self._connect_funcs.append(connect_func)
                return connect_func
            return add_connect_func
        else:
            self._connect_funcs.append(connect_func)
            return connect_func
        
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

    def __init__(self, _sentinel=None, init_model_dir=None, save_model_dir=None, train=True, serve=False):

        self._train = train
        self._serve = serve
        self._init_model_dir = init_model_dir
        self._save_model_dir = save_model_dir
        
        self._flag_updated = False
        self._flag_isbuilt = False
        
        self.traingraph = _RecommenderGraph()
        self.servegraph = _RecommenderGraph()

        self.T = self.traingraph
        self.S = self.servegraph
    
    def _generate_feed_dict(self, batch_data, input_map):

        feed_dict = dict()
        for key in input_map:
            feed_dict[input_map[key]] = batch_data[key]
        return feed_dict

    def train(self, batch_data, input_mapping_id='default', train_ops_id='default', losses_id='default', outputs_id='default'):

        assert self._train, "Train is disabled"
        assert self._flag_isbuilt, "Train graph is not built"
        
        feed_dict = self._generate_feed_dict(batch_data, 
                                            self.T.get_input_mapping(input_mapping_id))
        train_ops = self.T.get_train_ops(train_ops_id)
        losses_nodes = self.T.get_losses(losses_id)
        if len(losses_nodes) > 0:
            losses = [tf.add_n(losses_nodes)]
        else:
            losses = []
        outputs = self.T.get_outputs(outputs_id)
        results = self._tf_train_sess.run(train_ops+losses+outputs,
                                 feed_dict=feed_dict)
        
        return_dict = {'losses': results[len(train_ops):len(train_ops)+len(losses)],
                      'outputs': results[-len(outputs):]}
        
        self._flag_updated = True
        return return_dict
    
    def train_inspect_ports(self, batch_data, ports=[], input_mapping_id='default'):
        
        assert self._train, "Train is disabled"
        assert self._flag_isbuilt, "Train graph is not built"
        
        feed_dict = self._generate_feed_dict(batch_data, 
                                            self.T.get_input_mapping(input_mapping_id))
        
        results = self._tf_train_sess.run(ports,
                                 feed_dict=feed_dict)
        return results
        
    def serve(self, batch_data, input_mapping_id='default', losses_id='default', outputs_id='default'):

        assert self._serve, "serve is disabled"
        assert self._flag_isbuilt, "serve graph is not built"
        
        if self._flag_updated:
            self._save_and_load_for_serve()
            self._flag_updated = False
            
        feed_dict = self._generate_feed_dict(batch_data, self.S.get_input_mapping(input_mapping_id))
        losses_nodes = self.S.get_losses(losses_id)
        if len(losses_nodes) > 0:
            losses = [tf.add_n(losses_nodes)]
        else:
            losses = []
        outputs = self.S.get_outputs(outputs_id)
        results = self._tf_serve_sess.run(losses+outputs, 
                            feed_dict=feed_dict)

        return {'losses': results[:len(losses)], 'outputs': results[len(losses):]}
    
    def serve_inspect_ports(self, batch_data, ports=[], input_mapping_id='default'):
        
        assert self._serve, "serve graph is disabled"
        assert self._flag_isbuilt, "serve graph is not built"
        
        feed_dict = self._generate_feed_dict(batch_data, 
                                            self.T.get_input_mapping(input_mapping_id))
        
        results = self._tf_serve_sess.run(ports,
                                 feed_dict=feed_dict)
        return results
    
    def save(self, save_model_dir=None, global_step=None):
        
        if save_model_dir is None:
            save_model_dir = self._save_model_dir
        with self.traingraph.tf_graph.as_default():
            self._tf_train_saver.save(self._tf_train_sess, 
                                         os.path.join(save_model_dir, 'model.ckpt'),
                                        global_step=global_step)
    
    def restore(self, save_model_dir=None, restore_train=False, restore_serve=False):
        
        if save_model_dir is None:
            save_model_dir = self._save_model_dir
        if restore_train:
            assert self._train is not None, 'train is not enabled.'
            with self.traingraph.tf_graph.as_default():
                self._optimistic_restore(self._tf_train_sess, os.path.join(save_model_dir, 'model.ckpt'))
        if restore_serve:
            assert self._serve is not None, 'serve is not enabled.'
            with self.servegraph.tf_graph.as_default():
                self._optimistic_restore(self._tf_serve_sess, os.path.join(save_model_dir, 'model.ckpt'))
            
    def _save_and_load_for_serve(self):
        
        assert self._save_model_dir is not None, 'save_model_dir is not specified'
        if self._train:
            self.save()
        if self._serve:
            self.restore(restore_serve=True)
    
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

        if self._train:
            self.traingraph.build()
            with self.traingraph.tf_graph.as_default():
                config = tf.ConfigProto()
                config.gpu_options.allow_growth=True
                self._tf_train_sess = tf.Session(config=config)
                self._tf_train_sess.run(tf.global_variables_initializer())
                self._tf_train_saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        if self._serve:
            self.servegraph.build()
            with self.servegraph.tf_graph.as_default():
                config = tf.ConfigProto()
                config.gpu_options.allow_growth=True
                self._tf_serve_sess = tf.Session(config=config)
                self._tf_serve_sess.run(tf.global_variables_initializer())
                self._tf_serve_saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        
        if self._init_model_dir is not None:
            self.restore(save_model_dir=self._init_model_dir,
                        restore_train=self._train,
                        restore_serve=self._serve)
        
        self._flag_isbuilt = True
        
        return self
    
    def isbuilt(self):
        
        return self._flag_isbuilt