import numpy as np
from scipy.sparse import csr_matrix
import collections

from tqdm import tqdm

class Dataset(object):

    def __init__(self, raw_data, max_user, max_item, name='dataset'):

        self.name = name
        if type(raw_data) == csr_matrix:
            self.data = self._csr_to_structured_array(raw_data)
        elif type(raw_data) == collections.OrderedDict:
            self.data = self._tuples_to_structured_array(raw_data)
        elif type(raw_data) == np.ndarray:
            self.data = raw_data
        else:
            raise TypeError("Unsupported data input schema. Please use csr, tuples, or structured numpy array.")

        self.gy_user_item = dict()
        for ind, entry in enumerate(self.data):
            if entry['user_id'] not in self.gy_user_item:
                self.gy_user_item[entry['user_id']] = dict()
            if entry['item_id'] not in self.gy_user_item[entry['user_id']]:
                self.gy_user_item[entry['user_id']][entry['item_id']] = []
            self.gy_user_item[entry['user_id']][entry['item_id']].append(entry)

        self.gy_item_user = dict()
        for ind, entry in enumerate(self.data):
            if entry['item_id'] not in self.gy_item_user:
                self.gy_item_user[entry['item_id']] = dict()
            if entry['user_id'] not in self.gy_item_user[entry['item_id']]:
                self.gy_item_user[entry['item_id']][entry['user_id']] = []
            self.gy_item_user[entry['item_id']][entry['user_id']].append(entry)

        self.users = np.array(self.gy_user_item.keys())
        self.items = np.array(self.gy_item_user.keys())
        self.num_user = len(self.users)
        self.num_item = len(self.items)

        self.max_user = max_user
        self.max_item = max_item

    def shuffle(self):
        np.random.shuffle(self.data)

    def sample_nagatives(self, num=1000):

        self.sampled_user_negatives = {}
        for user in self.users:
            shuffled_items = np.random.permutation(self.max_item)
            subsamples = []
            for item in shuffled_items:
                if item not in self.gy_user_item[user]:
                    subsamples.append(item)
                if len(subsamples) == num:
                    break
            self.sampled_user_negatives[user] = subsamples

    def _csr_to_structured_array(self, csr_matrix):

        user_inds, item_inds = csr_matrix.nonzero()
        structured_arr = np.zeros(len(user_inds), dtype=[('user_id', np.int32), 
                                                         ('item_id', np.int32), 
                                                         ('interaction',np.float32)])
        for i in len(user_inds):
            structured_arr[i] = (user_inds[i], item_inds[i], csr_matrix[user_inds[i], item_inds[i]])
        return structured_arr

    def _tuples_to_structured_array(self,user_item_dict):
        
        num_interactions = 0
        for user in user_item_dict:
            num_interactions += len(user_item_dict[user])

        structured_arr = np.zeros(num_interactions, dtype=[('user_id', np.int32), 
                                                ('item_id', np.int32), 
                                                ('interaction', np.float32)])

        index = 0
        for user in user_item_dict:
            for item in user_item_dict[user]:
                structured_arr[index] = (user, item, 1.0)
                index += 1
        return structured_arr
