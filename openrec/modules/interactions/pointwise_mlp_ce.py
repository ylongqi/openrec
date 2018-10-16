import tensorflow as tf
from openrec.modules.extractions import MultiLayerFC


def PointwiseMLPCE(user, item, dims, subgraph, item_bias=None, extra=None,
                   l2_reg=None, labels=None, dropout=None, train=None, scope=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        
        if extra is not None:
            in_tensor = tf.concat([user, item, extra], axis=1)
        else:
            in_tensor = tf.concat([user, item], axis=1)
        if train: 
            reg = MultiLayerFC(
                in_tensor=in_tensor,
                dims=dims,
                subgraph=subgraph,
                bias_in=True,
                bias_mid=True,
                bias_out=False,
                dropout_mid=dropout,
                l2_reg=l2_reg,
                scope='mlp_reg')
        else:
            reg = MultiLayerFC(in_tensor=in_tensor,
                               dims=dims,
                               subgraph=subgraph,
                               bias_in=True,
                               bias_mid=True,
                               bias_out=False,
                               l2_reg=l2_reg,
                               scope='mlp_reg')

        logits = reg#.get_outputs()[0]
        if item_bias is not None:
            logits += item_bias
        
        if train:
            labels_float = tf.reshape(tf.to_float(labels), (-1, 1))
            subgraph.register_global_loss(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_float, logits=logits)))
            subgraph.register_global_output(logits)
        else:
            subgraph.register_global_output(tf.sigmoid(logits))
