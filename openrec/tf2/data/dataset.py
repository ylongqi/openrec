from openrec.tf2.data import _ParallelDataset
from openrec.tf2.data import _DataStore
from tensorflow.data import Dataset as tf_Dataset
import tensorflow as tf
import numpy as np
import random

class Dataset:
    
    def __init__(self, raw_data, total_users, total_items, implicit_negative=True, 
                 num_negatives=None, seed=None, sortby=None, asc=True, name=None):
        
        self.datastore = _DataStore(raw_data=raw_data, 
                                total_users=total_users, 
                                total_items=total_items, 
                                implicit_negative=implicit_negative, 
                                num_negatives=num_negatives, 
                                seed=seed, sortby=sortby, name=name, asc=asc)
    
    def _build_dataset(self, generator, output_types, output_shapes, 
                       batch_size, buffer_size, num_parallel_calls, take=None):
        
        if num_parallel_calls > 1:
            return _ParallelDataset(generator=generator,
                                    output_types=output_types, 
                                    output_shapes=output_shapes,
                                    batch_size=batch_size, 
                                    num_parallel_calls=num_parallel_calls, 
                                    take=take)
        else:
            dataset = tf_Dataset.from_generator(generator=generator,
                                 output_types=output_types,
                                 output_shapes=output_shapes).batch(batch_size).prefetch(buffer_size)
            
            if take is not None:
                dataset = dataset.take(take)
            
            return dataset
        
    def pairwise(self, batch_size, num_parallel_calls=1, buffer_size=1, take=None):
        
            
        def _generator(datastore=self.datastore):
                
            while True:
                entry = datastore.next_random_record()
                user_id = entry['user_id']
                p_item_id = entry['item_id']
                n_item_id = datastore.sample_negative_items(user_id)[0]
                yield {'user_id': user_id,
                       'p_item_id': p_item_id, 
                       'n_item_id': n_item_id}
        
        output_types = {'user_id': tf.int32, 
                        'p_item_id': tf.int32, 
                        'n_item_id': tf.int32}
        output_shapes = {'user_id':[], 
                        'p_item_id':[], 
                        'n_item_id':[]}
        
        return self._build_dataset(generator=_generator,
                                   output_types=output_types,
                                   output_shapes=output_shapes,
                                   batch_size=batch_size,
                                   buffer_size=buffer_size,
                                   num_parallel_calls=num_parallel_calls, 
                                   take=take)
        
        
    def evaluation(self, batch_size, excl_datasets=[], buffer_size=1):

        def _generator(datastore=self.datastore, excl_datasets=excl_datasets):

            eval_users = datastore.warm_users()

            for user_id in eval_users:

                pos_mask_npy = np.zeros(datastore.total_items(), dtype=np.bool)  # Reset pos_mask
                positive_items = datastore.get_positive_items(user_id)
                pos_mask_npy[positive_items] = True

                if datastore.contain_negatives():
                    excl_mask_npy = np.ones(datastore.total_items(), dtype=np.bool)  # Reset excl_mask
                    excl_mask_npy[positive_items] = False
                    negative_items = datastore.get_negative_items(user_id)
                    excl_mask_npy[negative_items] = False
                else:
                    excl_mask_npy = np.zeros(datastore.total_items(), dtype=np.bool)  # Reset excl_mask

                excl_positive_items = []
                for excl_d in excl_datasets:
                    excl_positive_items += excl_d.datastore.get_positive_items(user_id)
                excl_mask_npy[excl_positive_items] = True

                yield {'user_id': user_id,
                      'pos_mask': pos_mask_npy,
                      'excl_mask': excl_mask_npy}
            
        output_types = {'user_id': tf.int32,
                       'pos_mask': tf.bool,
                       'excl_mask': tf.bool}
        output_shapes = {'user_id': [], 
                         'pos_mask': [self.datastore.total_items()], 
                         'excl_mask': [self.datastore.total_items()]}
        
        return self._build_dataset(generator=_generator,
                                  output_types=output_types,
                                  output_shapes=output_shapes,
                                  batch_size=batch_size,
                                  buffer_size=buffer_size,
                                  num_parallel_calls=1)
        