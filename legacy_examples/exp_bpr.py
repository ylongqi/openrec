import os
import sys
sys.path.append(os.getcwd())

from openrec.legacy import ImplicitModelTrainer
from openrec.legacy.utils import ImplicitDataset
from openrec.legacy.recommenders import BPR
from openrec.legacy.utils.evaluators import AUC
from openrec.legacy.utils.samplers import PairwiseSampler
from config import sess_config
import dataloader

raw_data = dataloader.load_citeulike()
batch_size = 1000
test_batch_size = 100
display_itr = 10000

train_dataset = ImplicitDataset(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
val_dataset = ImplicitDataset(raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')
test_dataset = ImplicitDataset(raw_data['test_data'], raw_data['max_user'], raw_data['max_item'], name='Test')

bpr_model = BPR(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(), 
                dim_embed=20, opt='Adam', sess_config=sess_config)
sampler = PairwiseSampler(batch_size=batch_size, dataset=train_dataset, num_process=5)
model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size, 
    train_dataset=train_dataset, model=bpr_model, sampler=sampler)
auc_evaluator = AUC()

model_trainer.train(num_itr=int(1e5), display_itr=display_itr, eval_datasets=[val_dataset, test_dataset],
                    evaluators=[auc_evaluator])
