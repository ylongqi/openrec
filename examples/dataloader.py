from six.moves import cPickle as pickle
import numpy as np
from scipy.sparse import csr_matrix

def load_amazon_book():

    raw_data = dict()
    raw_data['max_user'] = 99473
    raw_data['max_item'] = 450166

    with open('dataset/amazon/user_data_train_dict.p', 'rb') as fin:
        raw_data['train_data'] = pickle.load(fin)
    with open('dataset/amazon/user_data_vali_dict.p', 'rb') as fin:
        raw_data['val_data'] = pickle.load(fin)
    with open('dataset/amazon/user_data_test_dict.p', 'rb') as fin:
        raw_data['test_data'] = pickle.load(fin)

    raw_data['item_features'] = np.array(np.memmap('dataset/amazon/book_features_update.mem', 
                                dtype=np.float32, mode='r', shape=(raw_data['max_item'], 4096)))
    raw_data['user_features'] = np.load('dataset/amazon/user_features_categories.npy')
    return raw_data

def load_citeulike():

    raw_data = dict()
    raw_data['max_user'] = 5551
    raw_data['max_item'] = 16980

    with open('dataset/citeulike/user_data_train_dict.p', 'rb') as fin:
        raw_data['train_data'] = pickle.load(fin)
    with open('dataset/citeulike/user_data_vali_dict.p', 'rb') as fin:
        raw_data['val_data'] = pickle.load(fin)
    with open('dataset/citeulike/user_data_test_dict.p', 'rb') as fin:
        raw_data['test_data'] = pickle.load(fin)

    return raw_data

def load_tradesy():

    raw_data = dict()
    raw_data['max_user'] = 19243
    raw_data['max_item'] = 165906

    with open('dataset/tradesy/user_data_train_dict.p', 'rb') as fin:
        raw_data['train_data'] = pickle.load(fin)
    with open('dataset/tradesy/user_data_vali_dict.p', 'rb') as fin:
        raw_data['val_data'] = pickle.load(fin)
    with open('dataset/tradesy/user_data_test_dict.p', 'rb') as fin:
        raw_data['test_data'] = pickle.load(fin)

    raw_data['item_features'] = np.load('dataset/tradesy/item_features.npy') / 32.671101
    return raw_data


