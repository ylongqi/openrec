import tensorflow as tf
import numpy as np

class DictMean:
    
    def __init__(self, state_shape):
        
        self._states = {}
        for key in state_shape:
            shape = state_shape[key]
            self._states[key] = {'sum': tf.Variable(tf.zeros(shape, dtype=tf.float32)),
                                'count': tf.Variable(tf.zeros([], dtype=tf.float32))}
    
    def reset_states(self):
        
        for key in self._states:
            self._states[key]['sum'].assign(tf.zeros(tf.shape(self._states[key]['sum']), 
                                                     dtype=tf.float32))
            self._states[key]['count'].assign(0.)
    
    def update_state(self, state):
        
        for key in state:
            self._states[key]['sum'].assign_add(tf.math.reduce_sum(state[key], axis=0))
            self._states[key]['count'].assign_add(tf.cast(tf.shape(state[key])[0], tf.float32))
        
    def result(self):
        
        result = {}
        for key in self._states:
            result[key] = self._states[key]['sum'] / self._states[key]['count']
        return result
            