import tensorflow as tf
from tensorflow.keras.layers import Layer

class PointwiseMSELoss(Layer):
    
    def __init__(self, a=1.0, b=1.0, sigmoid=False):

        super(PointwiseMSELoss, self).__init__()
        self._a = a
        self._b = b
        self._sigmoid = sigmoid
        
    def __call__(self, user_vec, item_vec, item_bias, label):
        
        outputs = super(PointwiseMSELoss, self).__call__((user_vec, item_vec, item_bias, label))
        return outputs
    
    def call(self, inputs):
        
        user_vec, item_vec, item_bias, label = inputs

        dot_user_item = tf.math.reduce_sum(tf.math.multiply(user_vec, item_vec),
                                  axis=1, keepdims=False, name="dot_user_item")

        if self._sigmoid:
            prediction = tf.math.sigmoid(dot_user_item + tf.reshape(item_bias, [-1]))
        else:
            prediction = dot_user_item + tf.reshape(item_bias, [-1])

        label_weight = (self._a - self._b) * label + self._b
        return tf.math.reduce_sum(label_weight * tf.square(label - prediction))
        