import tensorflow as tf
import numpy as np

def _log2(value):
    
    return tf.math.log(value) / tf.math.log(2.0)

def AUC(pos_mask, pred, excl_mask):
    
    def _map_fn(tups):
        
        user_pos_mask, user_pred, user_excl_mask = tups
        
        eval_mask = tf.math.logical_not(tf.math.logical_or(user_pos_mask, user_excl_mask))
        eval_pred = user_pred[eval_mask]
        pos_pred = user_pred[user_pos_mask]
        eval_num = tf.math.count_nonzero(eval_mask, dtype=tf.int32)
        user_auc = tf.math.count_nonzero(eval_pred <= tf.reshape(pos_pred, (-1, 1)), dtype=tf.float32) \
                    / tf.cast(tf.size(pos_pred) * eval_num, dtype=tf.float32)
        
        return user_auc
        
    auc = tf.map_fn(_map_fn, (pos_mask, pred, excl_mask), parallel_iterations=10, dtype=tf.float32)
            
    return auc


def NDCG(pos_mask, pred, excl_mask, at=[100]):
    
    def _map_fn(tups):
        
        user_pos_mask, user_pred, user_excl_mask = tups
        user_pred = tf.math.exp(user_pred) * tf.cast(tf.math.logical_not(user_excl_mask), tf.float32)
        pos_pred = user_pred[user_pos_mask]
        rank_above = tf.math.count_nonzero(user_pred > tf.reshape(pos_pred, (-1, 1)), axis=1, dtype=tf.float32)
        rank_above = tf.tile(tf.expand_dims(rank_above, 0), [len(at), 1])
        tf_at = tf.reshape(tf.constant(at, dtype=tf.float32), [-1, 1])
        log_recipr = tf.math.reciprocal(_log2(rank_above+2)) 
        
        user_ndcg = tf.reduce_sum(log_recipr * tf.cast(rank_above < tf_at, tf.float32),
                                  axis=1)
        
        return user_ndcg
        
    ndcg = tf.map_fn(_map_fn, (pos_mask, pred, excl_mask), parallel_iterations=10, dtype=tf.float32)
            
    return ndcg


def Recall(pos_mask, pred, excl_mask, at=[100]):
    
    
    def _map_fn(tups):
        
        user_pos_mask, user_pred, user_excl_mask = tups
        user_pred = tf.math.exp(user_pred) * tf.cast(tf.math.logical_not(user_excl_mask), tf.float32)
        pos_pred = user_pred[user_pos_mask]
        rank_above = tf.math.count_nonzero(user_pred > tf.reshape(pos_pred, (-1, 1)), axis=1, dtype=tf.float32)
        rank_above = tf.tile(tf.expand_dims(rank_above, 0), [len(at), 1])
        tf_at = tf.reshape(tf.constant(at, dtype=tf.float32), [-1, 1]) 
        
        user_recall = tf.math.count_nonzero(rank_above < tf_at, axis=1, dtype=tf.float32) / \
                        tf.cast(tf.size(pos_pred), tf.float32)
        
        return user_recall
        
    recall = tf.map_fn(_map_fn, (pos_mask, pred, excl_mask), parallel_iterations=10, dtype=tf.float32)
            
    return recall