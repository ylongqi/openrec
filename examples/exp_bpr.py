import os
import sys
sys.path.append(os.getcwd())

from openrec import ImplicitModelTrainer
from openrec.utils import Dataset
from openrec.recommenders import BPR
from openrec.utils.evaluators import AUC
from openrec.utils.samplers import RandomPairwiseSampler
from openrec.utils.samplers import EvaluationSampler
from config import sess_config
import dataloader

raw_data = dataloader.load_citeulike()
batch_size = 1000
eval_it = 10000
save_it = eval_it

train_dataset = Dataset(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
val_dataset = Dataset(raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val', num_negatives=500)
test_dataset = Dataset(raw_data['test_data'], raw_data['max_user'], raw_data['max_item'], name='Test', num_negatives=500)

train_sampler = RandomPairwiseSampler(batch_size=batch_size, dataset=train_dataset, num_process=5)
val_sampler = EvaluationSampler(val_dataset)
test_sampler = EvaluationSampler(test_dataset)

bpr_model = BPR(batch_size=batch_size, total_users=train_dataset.total_users(), total_items=train_dataset.total_items(), 
                dim_embed=50, save_model_dir='bpr_recommender/', training=True, serving=True)

model_trainer = ImplicitModelTrainer(model=bpr_model, serving_batch_size=batch_size)

auc_evaluator = AUC()
model_trainer.train(total_it=int(1e5), eval_it=eval_it, save_it=save_it, train_sampler=train_sampler, 
                    eval_samplers=[val_sampler, test_sampler], evaluators=[auc_evaluator])