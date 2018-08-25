import os
import sys
sys.path.append(os.getcwd())

from openrec import ImplicitModelTrainer
from openrec.utils import Dataset
from openrec.recommenders import UCML
from openrec.utils.evaluators import AUC, Recall
from openrec.utils.samplers import RandomPairwiseSampler
from openrec.utils.samplers import EvaluationSampler
from config import sess_config
import dataloader

raw_data = dataloader.load_citeulike()
dim_embed = 50
total_it = int(1e5)
batch_size = 1000
eval_it = 10000
save_it = eval_it

train_dataset = Dataset(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
val_dataset = Dataset(raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val', num_negatives=500)
test_dataset = Dataset(raw_data['test_data'], raw_data['max_user'], raw_data['max_item'], name='Test', num_negatives=500)

train_sampler = RandomPairwiseSampler(batch_size=batch_size, dataset=train_dataset, num_process=5)
val_sampler = EvaluationSampler(batch_size=batch_size, dataset=val_dataset)
test_sampler = EvaluationSampler(batch_size=batch_size, dataset=test_dataset)

model = UCML(batch_size=batch_size, total_users=train_dataset.total_users(), total_items=train_dataset.total_items(), 
                dim_user_embed=dim_embed, dim_item_embed=dim_embed, save_model_dir='ucml_recommender/', train=True, serve=True)

def train_it_func(model, batch_data):
    loss = model.train(batch_data)['losses'][0]
    model.train(batch_data, train_ops_id='censor_embedding')
    return loss

model_trainer = ImplicitModelTrainer(model=model,
                                     train_it_func=train_it_func)

auc_evaluator = AUC()
recall_evaluator = Recall(recall_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  
model_trainer.train(total_it=int(1e5), eval_it=eval_it, save_it=save_it, train_sampler=train_sampler, 
                    eval_samplers=[val_sampler, test_sampler], evaluators=[auc_evaluator, recall_evaluator])
