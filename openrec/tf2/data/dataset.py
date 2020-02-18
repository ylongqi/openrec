from openrec.tf2.data import _ParallelDataset
from openrec.tf2.data import _DataStore
import tensorflow as tf
import numpy as np
import random

def _pairwise_generator(datastore):
                
    while True:
        entry = datastore.next_random_record()
        user_id = entry['user_id']
        p_item_id = entry['item_id']
        n_item_id = datastore.sample_negative_items(user_id)[0]
        yield {'user_id': user_id,
                'p_item_id': p_item_id, 
                'n_item_id': n_item_id}

def _stratified_pointwise_generator(datastore, pos_ratio):

    while True:
        if random.random() <= pos_ratio:
            entry = datastore.next_random_record()
            yield {'user_id': entry['user_id'],
                    'item_id': entry['item_id'], 
                    'label': 1.0}
        else:
            user_id = random.randint(0, datastore.total_users()-1)
            item_id = random.randint(0, datastore.total_items()-1)
            while datastore.is_positive(user_id, item_id):
                user_id = random.randint(0, datastore.total_users()-1)
                item_id = random.randint(0, datastore.total_items()-1)
            yield {'user_id': user_id,
                    'item_id': item_id, 
                    'label': 0.0}

def _per_pos_stratified_pointwise_generator(datastore, pos_ratio):

    num_negative_per_positive = int((1 - pos_ratio) / pos_ratio)

    while True:

        entry = datastore.next_random_record()
        user_id = entry['user_id']
        p_item_id = entry['item_id']
        yield {'user_id': user_id,
                'item_id': p_item_id, 
                'label': 1.0}
        
        count = 0
        for n_item_id in random.sample(range(datastore.total_items()), k=num_negative_per_positive + 1):
            if n_item_id == p_item_id:
                continue
            yield {'user_id': user_id,
                    'item_id': n_item_id, 
                    'label': 0.0}
            count += 1
            if count >= num_negative_per_positive:
                break

def _evaluation_generator(datastore, excl_datasets):

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

class Dataset:
    
    def __init__(self, raw_data, total_users, total_items, implicit_negative=True, 
                 num_negatives=None, seed=None, sortby=None, asc=True, name=None):
        
        self.datastore = _DataStore(raw_data=raw_data, 
                                total_users=total_users, 
                                total_items=total_items, 
                                implicit_negative=implicit_negative, 
                                num_negatives=num_negatives, 
                                seed=seed, sortby=sortby, name=name, asc=asc)
    
    def _build_dataset(self, generator, generator_params, output_types, output_shapes, 
                       batch_size, num_parallel_calls, take=None):
        
        
        return _ParallelDataset(generator=generator,
                                generator_params=generator_params,
                                output_types=output_types, 
                                output_shapes=output_shapes,
                                batch_size=batch_size, 
                                num_parallel_calls=num_parallel_calls, 
                                take=take)
        
    def pairwise(self, batch_size, num_parallel_calls=1, take=None):
        
        output_types = {'user_id': tf.int32, 
                        'p_item_id': tf.int32, 
                        'n_item_id': tf.int32}
        output_shapes = {'user_id':[], 
                        'p_item_id':[], 
                        'n_item_id':[]}
        
        return self._build_dataset(generator=_pairwise_generator,
                                   generator_params=(self.datastore, ),
                                   output_types=output_types,
                                   output_shapes=output_shapes,
                                   batch_size=batch_size,
                                   num_parallel_calls=num_parallel_calls, 
                                   take=take)
    
    def stratified_pointwise(self, batch_size, pos_ratio=0.5, num_parallel_calls=1, take=None):

        output_types = {'user_id': tf.int32, 
                        'item_id': tf.int32, 
                        'label': tf.float32}
        output_shapes = {'user_id':[], 
                        'item_id':[], 
                        'label':[]}
        
        return self._build_dataset(generator=_stratified_pointwise_generator,
                                   generator_params=(self.datastore, pos_ratio),
                                   output_types=output_types,
                                   output_shapes=output_shapes,
                                   batch_size=batch_size,
                                   num_parallel_calls=num_parallel_calls, 
                                   take=take)
    
    def per_pos_stratified_pointwise(self, batch_size, pos_ratio=0.5, num_parallel_calls=1, take=None):

        output_types = {'user_id': tf.int32, 
                        'item_id': tf.int32, 
                        'label': tf.float32}
        output_shapes = {'user_id':[], 
                        'item_id':[], 
                        'label':[]}
        
        return self._build_dataset(generator=_per_pos_stratified_pointwise_generator,
                                   generator_params=(self.datastore, pos_ratio),
                                   output_types=output_types,
                                   output_shapes=output_shapes,
                                   batch_size=batch_size,
                                   num_parallel_calls=num_parallel_calls, 
                                   take=take)

    def evaluation(self, batch_size, excl_datasets=[]):
            
        output_types = {'user_id': tf.int32,
                       'pos_mask': tf.bool,
                       'excl_mask': tf.bool}
        output_shapes = {'user_id': [], 
                         'pos_mask': [self.datastore.total_items()], 
                         'excl_mask': [self.datastore.total_items()]}
        
        return self._build_dataset(generator=_evaluation_generator,
                                  generator_params=(self.datastore, excl_datasets),
                                  output_types=output_types,
                                  output_shapes=output_shapes,
                                  batch_size=batch_size,
                                  num_parallel_calls=1)
        