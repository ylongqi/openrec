import numpy as np
from scipy.sparse import csr_matrix

def load_amazon_book():

    raw_data = dict()
    raw_data['max_user'] = 99473
    raw_data['max_item'] = 450166

    raw_data['train_data'] = np.load('dataset/amazon/user_data_train.npy')
    raw_data['val_data'] = np.load('dataset/amazon/user_data_val.npy')
    raw_data['test_data'] = np.load('dataset/amazon/user_data_test.npy')

    raw_data['item_features'] = np.array(np.memmap('dataset/amazon/book_features_update.mem', 
                                dtype=np.float32, mode='r', shape=(raw_data['max_item'], 4096)))
    raw_data['user_features'] = np.load('dataset/amazon/user_features_categories.npy')
    return raw_data

def load_citeulike():

    raw_data = dict()
    raw_data['max_user'] = 5551
    raw_data['max_item'] = 16980
    
    raw_data['train_data'] = np.load('dataset/citeulike/user_data_train.npy')
    raw_data['val_data'] = np.load('dataset/citeulike/user_data_val.npy')
    raw_data['test_data'] = np.load('dataset/citeulike/user_data_test.npy')
    
    return raw_data

def load_tradesy():

    raw_data = dict()
    raw_data['max_user'] = 19243
    raw_data['max_item'] = 165906
    
    raw_data['train_data'] = np.load('dataset/tradesy/user_data_train.npy')
    raw_data['val_data'] = np.load('dataset/tradesy/user_data_val.npy')
    raw_data['test_data'] = np.load('dataset/tradesy/user_data_test.npy')

    raw_data['item_features'] = np.load('dataset/tradesy/item_features.npy') / 32.671101
    return raw_data


def load_music():
    raw_data = dict()
    raw_data['max_user'] = 29163
    raw_data['max_item'] = 2296834

    print 'oh shit 1'
    raw_data['train_data'] = np.load('dataset/music/train_structured_arr.p')
    print 'oh shit 2'
    raw_data['val_data'] = np.load('dataset/music/val_structured_arr.p')
    print 'oh shit 3'
    raw_data['test_data'] = np.load('dataset/music/test_structured_arr.p')
    print 'oh shit 4'
    raw_data['song_to_genre'] = np.load('dataset/music/song_to_genre.p')
    print 'oh shit 5' 
    return raw_data


