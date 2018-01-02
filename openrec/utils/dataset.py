import numpy as np
from scipy.sparse import csr_matrix
import collections

from tqdm import tqdm

class Dataset(object):

    """
    The Dataset class stores and parses a sequence of data points for training or evaluation. 

    Parameters
    ----------
    raw_data: numpy structured array
        Input raw data. Other legacy formats (e.g., sparse matrix) are supported but not recommended.
    max_user: int
        Maximum number of users in the recommendation system.
    max_item: int
        Maximum number of items in the recommendation system.
    name: str
        Name of the dataset. 

    Notes  
    -----
    The Dataset class parses the input :code:`raw_data` into structured dictionaries (consumed by samplers or model trainer). 
    This class expects :code:`raw_data` as a numpy structured array, where each row represents a data 
    point and contains *at least* two keys:
    
    * :code:`user_id`: the user involved in the interaction.
    * :code:`item_id`: the item involved in the interaction.

    :code:`raw_data` might contain other keys, such as :code:`timestamp`, and :code:`location`, etc.
    based on the use cases of different recommendation systems. An user should be uniquely and numerically indexed 
    from 0 to :code:`total_number_of_users - 1`. The items should be indexed likewise.
    """

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

        self._gb_user_item = dict()
        for ind, entry in enumerate(self.data):
            if entry['user_id'] not in self._gb_user_item:
                self._gb_user_item[entry['user_id']] = dict()
            if entry['item_id'] not in self._gb_user_item[entry['user_id']]:
                self._gb_user_item[entry['user_id']][entry['item_id']] = []
            self._gb_user_item[entry['user_id']][entry['item_id']].append(entry)

        self._gb_item_user = dict()
        for ind, entry in enumerate(self.data):
            if entry['item_id'] not in self._gb_item_user:
                self._gb_item_user[entry['item_id']] = dict()
            if entry['user_id'] not in self._gb_item_user[entry['item_id']]:
                self._gb_item_user[entry['item_id']][entry['user_id']] = []
            self._gb_item_user[entry['item_id']][entry['user_id']].append(entry)

        self._users = np.array(self._gb_user_item.keys())
        self._items = np.array(self._gb_item_user.keys())
        self._num_user = len(self._users)
        self._num_item = len(self._items)

        self._max_user = max_user
        self._max_item = max_item

    def contain_user(self, user_id):
        """Check whether or not an user is involved in any interaction.

        Parameters
        ----------
        user_id: int
            target user id.

        Returns
        -------
        bool
            A boolean indicator
        """
        return user_id in self._gb_user_item

    def contain_item(self, item_id):
        """Check whether or not an item is involved in any interaction.

        Parameters
        ----------
        item_id: int
            target item id.

        Returns
        -------
        bool
            A boolean indicator
        """
        return item_id in self._gb_item_user

    def get_interactions_by_user_gb_item(self, user_id):
        """Retrieve the interactions (grouped by item ids) involve a specific user.

        Parameters
        ----------
        user_id: int
            target user id.

        Returns
        -------
        dict
            Interactions grouped by item ids.
        """
        return self._gb_user_item[user_id]

    def get_interactions_by_item_gb_user(self, item_id):
        """Retrieve the interactions (grouped by user ids) involve a specific item.

        Parameters
        ----------
        item_id: int
            target item id.

        Returns
        -------
        dict
            Interactions grouped by user ids.
        """
        return self._gb_item_user[item_id]

    def get_unique_user_list(self):
        """Retrieve a list of unique user ids.

        Returns
        -------
        numpy array
            A list of unique user ids.
        """
        return self._users

    def get_unique_item_list(self):
        """Retrieve a list of unique item ids.

        Returns
        -------
        numpy array
            A list of unique item ids.
        """
        return self._items

    def unique_user_count(self):
        """Number of unique users.

        Returns
        -------
        int
            Number of unique users.
        """
        return self._num_user

    def unique_item_count(self):
        """Number of unique items.

        Returns
        -------
        int
            Number of unique items.
        """
        return self._num_item

    def max_user(self):
        """Maximum number of users.

        Returns
        -------
        int
            Maximum number of users.
        """
        return self._max_user

    def max_item(self):
        """Maximum number of items.

        Returns
        -------
        int
            Maximum number of items.
        """
        return self._max_item

    def shuffle(self):
        """Shuffle the dataset entries.
        """
        np.random.shuffle(self.data)

    def get_sampled_items_by_user(self, user_id):
        """Retrieve a list of items NOT interacted with an user. The function :code:`sample_negatives` needs to be called first to sample \
        negatives.
        
        Parameters
        ----------
        user_id: int
            target user id.

        Returns
        -------
        numpy array
            A list of item ids.
        """
        if self._sampled_user_negatives is None:
            print 'Retrieval failed ... negatives are not sampled.'
        else:
            return self._sampled_user_negatives[user_id]

    def sample_nagatives(self, num=1000):
        """Sample a given number of items NOT interacted with each user.
        
        Parameters
        ----------
        num: int
            Number of negative items sampled for each user.
        """
        self._sampled_user_negatives = {}
        for user in self._users:
            shuffled_items = np.random.permutation(self._max_item)
            subsamples = []
            for item in shuffled_items:
                if item not in self._gb_user_item[user]:
                    subsamples.append(item)
                if len(subsamples) == num:
                    break
            self._sampled_user_negatives[user] = subsamples

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
