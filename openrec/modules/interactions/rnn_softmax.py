import tensorflow as tf

def RNNSoftmax(seq_item_vec, total_items, seq_len, num_units, cell_type='gru', softmax_samples=None,
                   label=None, train=True, subgraph=None, scope=None):
    
    with tf.variable_scope(scope, default_name='RNNSoftmax', reuse=tf.AUTO_REUSE):
        if cell_type == 'gru':
            rnn_cell = tf.nn.rnn_cell.GRUCell(num_units)
        elif cell_type == 'lstm':
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units)
        else:
            assert False, "Invalid RNN cell type."

        _, rnn_state = tf.nn.dynamic_rnn(cell=rnn_cell, 
                                              inputs=seq_item_vec, 
                                              sequence_length=seq_len,
                                              dtype=tf.float32)
        weight = tf.get_variable('weights', shape=[total_items, num_units], trainable=True,
                                      initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('biases', shape=[total_items], trainable=True,
                                    initializer=tf.zeros_initializer())
        if train:
            if softmax_samples is not None:
                loss = tf.nn.sampled_sparse_softmax_loss(weight=weight, bias=bias, num_sampled=softmax_samples, 
                                                         num_classes=total_items, labels=label, inputs=rnn_state)
            else:
                logits = tf.matmul(rnn_state, tf.transpose(weight)) + bias
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
            subgraph.register_global_loss(tf.reduce_mean(loss))
        else:
            logits = tf.matmul(rnn_state, tf.transpose(weight)) + bias
            subgraph.register_global_output(tf.squeeze(logits))