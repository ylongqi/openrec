import numpy as np
import random

class Dataset(object):
    
    def __init__(self, raw_data, total_users, total_items, implicit_negative=True, 
                 num_negatives=None, name='dataset', seed=100, sortby=None, asc=True):
        
        random.seed(seed)
        self.name = name
        if type(raw_data) == np.ndarray:
            self._raw_data = raw_data
        else:
            raise TypeError("Unsupported data input schema. Please use structured numpy array.")
        self._rand_ids = []
        
        self._total_users = total_users
        self._total_items = total_items
        
        self._sortby = sortby
        
        self._index_store = dict()
        if implicit_negative:
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