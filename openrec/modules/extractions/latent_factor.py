import tensorflow as tf

def LatentFactor(shape, id_=None, l2_reg=None, init='normal', 
                subgraph=None, scope=None):

    if init == 'normal':
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)
    elif init == 'zero':
        initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    else:
        initializer = tf.constant_initializer(value=init, dtype=tf.float32)

    with tf.variable_scope(scope, default_name='latentfactor', reuse=tf.AUTO_REUSE):
        embedding = tf.get_variable('embedding', shape=shape, trainable=True,
                                      initializer=initializer)
        if id_ is None:
            output = None
        else:
            output = tf.nn.embedding_lookup(embedding, id_)
        
        if l2_reg is not None:
            subgraph.register_global_loss(l2_reg * tf.nn.l2_loss(output))

    return embedding, output