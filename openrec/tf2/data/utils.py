import multiprocessing as mp
import tensorflow as tf
import numpy as np
import random

class _DataStore(object):
    
    def __init__(self, raw_data, total_users, total_items, implicit_negative=True, 
                 num_negatives=None, seed=None, sortby=None, asc=True, name=None):
        
        self.name = name
        random.seed(seed)
        if type(raw_data) == np.ndarray:
            self._raw_data = raw_data
        else:
            raise TypeError("Unsupported data input schema. Please use structured numpy array.")
        self._rand_ids = []
        
        self._total_users = total_users
        self._total_items = total_items
        
        self._sortby = sortby
        
        self._index_store = dict()
        self._implicit_negative = implicit_negative
        self._num_negatives = num_negatives
        if self._implicit_negative:
            self._index_store['positive'] = dict()
            for ind, entry in enumerate(self._raw_data):
                if entry['user_id'] not in self._index_store['positive']:
                    self._index_store['positive'][entry['user_id']] = dict()
                self._index_store['positive'][entry['user_id']][entry['item_id']] = ind
            self._index_store['positive_sets'] = dict()
            for user_id in self._index_store['positive']:
                self._index_store['positive_sets'][user_id] = set(self._index_store['positive'][user_id])
            if num_negatives is not None:
                self._index_store['negative'] = dict()
                for user_id in self._index_store['positive']:
                    self._index_store['negative'][user_id] = dict()
                    shuffled_items = np.random.permutation(self._total_items)
                    for item in shuffled_items:
                        if item not in self._index_store['positive'][user_id]:
                            self._index_store['negative'][user_id][item] = None
                        if len(self._index_store['negative'][user_id]) == num_negatives:
                            break
                self._index_store['negative_sets'] = dict()
                for user_id in self._index_store['negative']:
                    self._index_store['negative_sets'][user_id] = set(self._index_store['negative'][user_id])
        else:
            self._index_store['positive'] = dict()
            self._index_store['negative'] = dict()
            for ind, entry in enumerate(self._raw_data):
                if entry['label'] > 0:
                    if entry['user_id'] not in self._index_store['positive']:
                        self._index_store['positive'][entry['user_id']] = dict()
                    self._index_store['positive'][entry['user_id']][entry['item_id']] = ind
                else:
                    if entry['user_id'] not in self._index_store['negative']:
                        self._index_store['negative'][entry['user_id']] = dict()
                    self._index_store['negative'][entry['user_id']][entry['item_id']] = ind
            self._index_store['positive_sets'] = dict()
            for user_id in self._index_store['positive']:
                self._index_store['positive_sets'][user_id] = set(self._index_store['positive'][user_id])
            self._index_store['negative_sets'] = dict()
            for user_id in self._index_store['negative']:
                self._index_store['negative_sets'][user_id] = set(self._index_store['negative'][user_id])
        
        if self._sortby is not None:
            self._index_store['positive_sorts'] = dict()
            for user_id in self._index_store['positive_sets']:
                self._index_store['positive_sorts'][user_id] = sorted(list(self._index_store['positive_sets'][user_id]),
                                                                    key=lambda item:\
                                             self._raw_data[self._index_store['positive'][user_id][item]][self._sortby],
                                                                    reverse=not asc)
    def contain_negatives(self):
        
        if self._implicit_negative and self._num_negatives is None:
            return False
        else:
            return True
    
    def next_random_record(self):
        
        if len(self._rand_ids) == 0:
            self._rand_ids = list(range(len(self._raw_data)))
            random.shuffle(self._rand_ids)
        return self._raw_data[self._rand_ids.pop()]
        
    def is_positive(self, user_id, item_id):
        
        if user_id in self._index_store['positive'] and item_id in self._index_store['positive'][user_id]:
            return True
        return False
    
    def sample_positive_items(self, user_id, num_samples=1):
        
        if user_id in self._index_store['positive_sets']:
            return random.sample(self._index_store['positive_sets'][user_id], num_samples)
        else:
            return []
        
    def sample_negative_items(self, user_id, num_samples=1):
        
        if 'negative_sets' in self._index_store:
            if user_id in self._index_store['negative_sets']:
                return random.sample(self._index_store['negative_sets'][user_id], num_samples)
            else:
                return []
        else:
            sample_id = random.randint(0, self._total_items-1)
            sample_set = set()
            while len(sample_set) < num_samples:
                if user_id not in self._index_store['positive_sets'] or sample_id not in self._index_store['positive_sets'][user_id]:
                    sample_set.add(sample_id)
                sample_id = random.randint(0, self._total_items-1)
            return list(sample_set)
        
    def get_positive_items(self, user_id, sort=False):

        if user_id in self._index_store['positive_sets']:
            if sort:
                assert self._sortby is not None, "sortby key is not specified."
                return self._index_store['positive_sorts'][user_id]
            else:
                return list(self._index_store['positive_sets'][user_id])
        else:
            return []
    
    def get_negative_items(self, user_id):
        
        if 'negative_sets' in self._index_store:
            if user_id in self._index_store['negative_sets']:
                return list(self._index_store['negative_sets'][user_id])
            else:
                return []
        else:
            negative_items = []
            for item_id in range(self._total_items):
                if item_id not in self._index_store['positive_sets'][user_id]:
                    negative_items.append(item_id)
            return negative_items
    
    def warm_users(self, threshold=1):
        
        users_list = []
        for user_id in self._index_store['positive']:
            if len(self._index_store['positive'][user_id]) >= threshold:
                users_list.append(user_id)
        return users_list
    
    def total_users(self):
        
        return self._total_users

    def total_items(self):
        
        return self._total_items
    
    def total_records(self):

        return len(self._raw_data)


def _process(q, generator, generator_params, output_shapes, batch_size):
            
    batch_data = {key:[] for key in output_shapes}
    num_data_points = 0

    for single_data in generator(*generator_params):
        for key in single_data:
            batch_data[key].append(single_data[key])
        num_data_points += 1
        if num_data_points == batch_size:
            q.put(batch_data)
            batch_data = {key:[] for key in output_shapes}
            num_data_points = 0
    
    if num_data_points > 0:
        q.put(batch_data)
    q.put(None)

class _ParallelDataset:
    
    def __init__(self, generator, generator_params, output_types, output_shapes, 
                 num_parallel_calls, batch_size, take):
        
        ctx = mp.get_context('spawn')
        self._q = ctx.Queue(maxsize=num_parallel_calls)
        self._output_types = output_types
        self._take = take
        self._count = 0
        
        self._p_list = []
        
        for i in range(num_parallel_calls):
            self._p_list.append(ctx.Process(target=_process, args=(self._q, generator, generator_params, output_shapes, batch_size)))
            self._p_list[i].daemon = True
            self._p_list[i].start()
    
    def __iter__(self):
        
        return self
    
    def __next__(self):
        
        if self._take is None or self._count < self._take:
            batch_data = self._q.get()
            if batch_data is None:
                raise StopIteration()
            else:
                self._count += 1
                return {key:tf.constant(batch_data[key], dtype=self._output_types[key]) for key in batch_data}
        else:
            raise StopIteration()
        
    