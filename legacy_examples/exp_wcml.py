import os
import sys
sys.path.append(os.getcwd())

from openrec.legacy import ImplicitModelTrainer
from openrec.legacy.utils import ImplicitDataset
from openrec.legacy.recommenders import WCML
from openrec.legacy.utils.evaluators import AUC
from openrec.legacy.utils.samplers import NPairwiseSampler
import dataloader

raw_data = dataloader.load_citeulike()
batch_size = 2000
test_batch_size = 100
display_itr = 500

train_dataset = ImplicitDataset(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
val_dataset = ImplicitDataset(raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')
test_dataset = ImplicitDataset(raw_data['test_data'], raw_data['max_user'], raw_data['max_item'], name='Test')

model = WCML(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(), 
    dim_embed=20, neg_num=5, l2_reg=None, opt='Adam', sess_config=None)
sampler = NPairwiseSampler(batch_size=batch_size, dataset=train_dataset, negativenum=5, num_process=5)
model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size,
                                     train_dataset=train_dataset, model=model, sampler=sampler)
auc_evaluator = AUC()

model_trainer.train(num_itr=int(1e5), display_itr=display_itr, eval_datasets=[val_dataset],
                    evaluators=[auc_evaluator], num_negatives=200)
