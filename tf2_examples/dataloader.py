import tensorflow as tf
import numpy as np

def load_amazon_book(dataset_folder='dataset/'):

    raw_data = dict()
    raw_data['total_users'] = 99473
    raw_data['total_items'] = 450166

    raw_data['train_data'] = np.load(dataset_folder + 'amazon/user_data_train.npy')
    raw_data['val_data'] = np.load(dataset_folder + 'amazon/user_data_val.npy')
    raw_data['test_data'] = np.load(dataset_folder + 'amazon/user_data_test.npy')

    raw_data['item_features'] = np.array(np.memmap(dataset_folder + 'amazon/book_features_update.mem', 
                                dtype=np.float32, mode='r', shape=(raw_data['max_item'], 4096)))
    raw_data['user_features'] = np.load(dataset_folder + 'amazon/user_features_categories.npy')
    return raw_data

def load_citeulike(dataset_folder='dataset/'):

    raw_data = dict()
    raw_data['total_users'] = 5551
    raw_data['total_items'] = 16980
    
    raw_data['train_data'] = np.load(dataset_folder + 'citeulike/user_data_train.npy')
    raw_data['val_data'] = np.load(dataset_folder + 'citeulike/user_data_val.npy')
    raw_data['test_data'] = np.load(dataset_folder + 'citeulike/user_data_test.npy')
    
    return raw_data

def load_tradesy(dataset_folder='dataset/'):

    raw_data = dict()
    raw_data['total_users'] = 19243
    raw_data['total_items'] = 165906
    
    raw_data['train_data'] = np.load(dataset_folder + 'tradesy/user_data_train.npy')
    raw_data['val_data'] = np.load(dataset_folder + 'tradesy/user_data_val.npy')
    raw_data['test_data'] = np.load(dataset_folder + 'tradesy/user_data_test.npy')

    raw_data['item_features'] = np.load(dataset_folder + 'tradesy/item_features.npy') / 32.671101
    return raw_data

def load_criteo(dataset_folder='dataset/'):
    
    # Data processing code adapted from https://github.com/facebookresearch/dlrm
    # Follow steps in https://github.com/ylongqi/dlrm/blob/master/data_utils.py to generate kaggle_processed.npz
    # Or using `./download_dataset.sh criteo` command to download the processed data.
    
    with np.load(dataset_folder + 'criteo/kaggle_processed.npz') as data:

        X_int = data["X_int"]
        X_cat = data["X_cat"]
        y = data["y"]
        counts = data["counts"]
    
    indices = np.arange(len(y))
    indices = np.array_split(indices, 7)
    for i in range(len(indices)):
        indices[i] = np.random.permutation(indices[i])
    
    train_indices = np.concatenate(indices[:-1])
    test_indices = indices[-1]
    val_indices, test_indices = np.array_split(test_indices, 2)
    train_indices = np.random.permutation(train_indices)
    
    raw_data = dict()
    
    raw_data['counts'] = counts
    
    raw_data['X_cat_train'] = X_cat[train_indices].astype(np.int32)
    raw_data['X_int_train'] = np.log(X_int[train_indices]+1).astype(np.float32)
    raw_data['y_train'] = y[train_indices].astype(np.float32)
    
    raw_data['X_cat_val'] = X_cat[val_indices]
    raw_data['X_int_val'] = np.log(X_int[val_indices]+1).astype(np.float32)
    raw_data['y_val'] = y[val_indices]
    
    raw_data['X_cat_test'] = X_cat[test_indices]
    raw_data['X_int_test'] = np.log(X_int[test_indices]+1).astype(np.float32)
    raw_data['y_test'] = y[test_indices]
    
    return raw_data
    