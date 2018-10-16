import os
import sys
sys.path.append(os.getcwd())

from openrec.legacy import ImplicitModelTrainer
from openrec.legacy.utils import ImplicitDataset
from openrec.legacy.recommenders import PMF
from openrec.legacy.utils.evaluators import AUC, Recall
from openrec.legacy.utils.samplers import PointwiseSampler
from config import sess_config
import dataloader

raw_data = dataloader.load_citeulike()
batch_size = 1000
test_batch_size = 100
display_itr = 10000

train_dataset = ImplicitDataset(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
val_dataset = ImplicitDataset(raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')
test_dataset = ImplicitDataset(raw_data['test_data'], raw_data['max_user'], raw_data['max_item'], name='Test')

model = PMF(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(), 
                dim_embed=50, opt='Adam', sess_config=sess_config)
sampler = PointwiseSampler(batch_size=batch_size, dataset=train_dataset, pos_ratio=0.2, num_process=5)
model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size, 
    train_dataset=train_dataset, model=model, sampler=sampler)

auc_evaluator = AUC()
recall_evaluator = Recall(recall_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

model_trainer.train(num_itr=int(1e5), display_itr=display_itr, eval_datasets=[val_dataset, test_dataset],
                    evaluators=[auc_evaluator, recall_evaluator])
