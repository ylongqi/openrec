import collections
import numpy as np
from scipy.sparse import csr_matrix

class Dataset(object):

    
    """
    The Dataset class stores a sequence of data points for training or evaluation. 

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
    The Dataset class expects :code:`raw_data` as a numpy structured array, where each row represents a data 
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
        
        self._max_user = max_user
        self._max_item = max_item
    
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
    
    def _csr_to_structured_array(self, csr_matrix):

        user_inds, item_inds = csr_matrix.nonzero()
        structured_arr = np.zeros(len(user_inds), dtype=[('user_id', np.int32), 
                                                         ('item_id', np.int32)])
        for i in len(user_inds):
            structured_arr[i] = (user_inds[i], item_inds[i])
        return structured_arr

    def _tuples_to_structured_array(self,user_item_dict):
        
        num_interactions = 0
        for user in user_item_dict:
            num_interactions += len(user_item_dict[user])

        structured_arr = np.zeros(num_interactions, dtype=[('user_id', np.int32), ('item_id', np.int32)])

        index = 0
        for user in user_item_dict:
            for item in user_item_dict[user]:
                structured_arr[index] = (user, item)
                index += 1
        return structured_arr