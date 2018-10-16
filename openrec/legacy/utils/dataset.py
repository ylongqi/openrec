import numpy as np

class Dataset(object):

    
    """
    The Dataset class stores a sequence of data points for training or evaluation. 

    Parameters
    ----------
    raw_data: numpy structured array
        Input raw data.
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
        if type(raw_data) == np.ndarray:
            self.raw_data = raw_data
        else:
            raise TypeError("Unsupported data input schema. Please use structured numpy array.")
        
        self.data = None
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
        if self.data is None:
            self.data = self.raw_data.copy()
            
        np.random.shuffle(self.data)