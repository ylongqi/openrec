from __future__ import print_function
from openrec.algorithms.basic_cml import BasicCML
from openrec.utils.samplers import PairwiseSampler
from openrec.utils.evaluators import AUC
from ..utils.dataset import Dataset
import numpy as np
import random
from tqdm import tqdm
import math

print('==> Load dataset')

'''
from .dataloader.tradesy import *

BATCH_SIZE = 10000
TEST_BATCH_SIZE = 100
DISPLAY_ITR = 10000
'''

from .dataloader.citeulike import *

BATCH_SIZE = 1000
TEST_BATCH_SIZE = 100
DISPLAY_ITR = 10000

dataset = Dataset(user_data_train_dict,user_data_vali_dict,user_data_test_dict)

cml_model = BasicCML(batch_size=BATCH_SIZE, num_user=NUM_USER, num_item=NUM_ITEM, dim_embed=20)
sampler = PairwiseSampler(batch_size=BATCH_SIZE, train_data=dataset.train, num_process=1)
evaluator = AUC()

acc_loss = 0

for itr in range(int(5e8)):
    sampler.next_batch(cml_model)
    loss = cml_model.train()
    acc_loss += loss

    if itr % DISPLAY_ITR == 0:
        print ('==> iteration %d, loss %f' % (itr, acc_loss / DISPLAY_ITR))
        acc_loss = 0

        print('==> Validation %d itr' % itr)
        val_auc_list = []
        for val_itr in tqdm(range(int(math.ceil(NUM_USER / TEST_BATCH_SIZE)))):
            rankings = cml_model.serve(np.arange(val_itr*TEST_BATCH_SIZE, min(NUM_USER, (val_itr + 1) * TEST_BATCH_SIZE)))
            for val_ind, ranking in enumerate(rankings):
                user_id = val_itr * TEST_BATCH_SIZE + val_ind
                val_auc_list.append(evaluator.calculate(pos_samples=dataset.vali.get_reduced_user_interactions(user_id),predictions=ranking,neg_samples=dataset.all.get_reduced_user_interactions(user_id,pos=False)))

        print ('==> Validation AUC: %f' % np.mean(val_auc_list))

        print ('==> Test %d itr' % itr)
        test_auc_list = []
        for test_itr in tqdm(range(int(math.ceil(NUM_USER / TEST_BATCH_SIZE)))):
            rankings = cml_model.serve(np.arange(val_itr*TEST_BATCH_SIZE, min(NUM_USER, (val_itr + 1) * TEST_BATCH_SIZE)))

            for test_ind, ranking in enumerate(rankings):
                user_id = test_itr * TEST_BATCH_SIZE + test_ind
                test_auc_list.append(evaluator.calculate(pos_samples=dataset.test.get_reduced_user_interactions(user_id),predictions=ranking,neg_samples=dataset.all.get_reduced_user_interactions(user_id,pos=False)))

        print ('==> Test AUC: %f' % np.mean(test_auc_list))
