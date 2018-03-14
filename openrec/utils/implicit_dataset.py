import numpy as np
from openrec.utils import Dataset

class ImplicitDataset(Dataset):

    """
    The ImplicitDataset class stores and parses a sequence of user implicit feedback for training or evaluation. It extends the 
    functionality of the Dataset class.

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
    The ImplicitDataset class parses the input :code:`raw_data` into structured dictionaries (consumed by samplers or model 
    trainer). This class expects :code:`raw_data` as a numpy structured array, where each row represents a data 
    point and contains *at least* two keys:
    
    * :code:`user_id`: the user involved in the interaction.
    * :code:`item_id`: the item involved in the interaction.

    :code:`raw_data` might contain other keys, such as :code:`timestamp`, and :code:`location`, etc.
    based on the use cases of different recommendation systems. An user should be uniquely and numerically indexed 
    from 0 to :code:`total_number_of_users - 1`. The items should be indexed likewise.
    """

    def __init__(self, raw_data, max_user, max_item, name='dataset'):

        super(ImplicitDataset, self).__init__(raw_data=raw_data, max_user=max_user, 
                                              max_item=max_item, name=name)

        self._gb_user_item = dict()
        for entry in self.raw_data:
            if entry['user_id'] not in self._gb_user_item:
                self._gb_user_item[entry['user_id']] = list()
            self._gb_user_item[entry['user_id']].append(entry['item_id'])
        
        self._gb_item_user = dict()
        for entry in self.raw_data:
            if entry['item_id'] not in self._gb_item_user:
                self._gb_item_user[entry['item_id']] = list()
            self._gb_item_user[entry['item_id']].append(entry['user_id'])
        
        self._num_user = len(self._gb_user_item)
        self._num_item = len(self._gb_item_user)

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
        list
            Items that have interacted with given user.
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
        list
            Users that have interacted with given item.
        """
        return self._gb_item_user[item_id]

    def get_unique_user_list(self):
        """Retrieve a list of unique user ids.

        Returns
        -------
        numpy array
            A list of unique user ids.
        """
        return np.array(list(self._gb_user_item.keys()))

    def get_unique_item_list(self):
        """Retrieve a list of unique item ids.

        Returns
        -------
        numpy array
            A list of unique item ids.
        """
        return np.array(list(self._gb_item_user.keys()))

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
