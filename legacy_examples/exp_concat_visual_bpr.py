import os
import sys
sys.path.append(os.getcwd())

from openrec.legacy import ImplicitModelTrainer
from openrec.legacy.utils import ImplicitDataset
from openrec.legacy.recommenders import ConcatVisualBPR
from openrec.legacy.utils.evaluators import AUC
from openrec.legacy.utils.samplers import PairwiseSampler
from config import sess_config
import dataloader

raw_data = dataloader.load_tradesy()
batch_size = 1000
test_batch_size = 100
item_serving_size = 1000
display_itr = 10000

train_dataset = ImplicitDataset(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
val_dataset = ImplicitDataset(raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')
test_dataset = ImplicitDataset(raw_data['test_data'], raw_data['max_user'], raw_data['max_item'], name='Test')

model = ConcatVisualBPR(batch_size=batch_size, max_user=raw_data['max_user'], max_item=raw_data['max_item'], item_serving_size=item_serving_size,
                dim_embed=20, dim_ve=10, item_f_source=raw_data['item_features'], l2_reg=None, sess_config=sess_config)
sampler = PairwiseSampler(batch_size=batch_size, dataset=train_dataset, num_process=5)
model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size, item_serving_size=item_serving_size,
    train_dataset=train_dataset, model=model, sampler=sampler)

auc_evaluator = AUC()

model_trainer.train(num_itr=int(1e5), display_itr=display_itr, eval_datasets=[val_dataset, test_dataset],
                    evaluators=[auc_evaluator], num_negatives=1000)
