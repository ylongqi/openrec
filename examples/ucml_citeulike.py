from openrec import ModelTrainer
from openrec.utils import Dataset
from openrec.recommenders import UCML
from openrec.utils.evaluators import AUC, Recall
from openrec.utils.samplers import RandomPairwiseSampler
from openrec.utils.samplers import EvaluationSampler
import dataloader

raw_data = dataloader.load_citeulike()
dim_embed = 50
total_iter = int(1e5)
batch_size = 1000
eval_iter = 10000
save_iter = eval_iter

train_dataset = Dataset(raw_data['train_data'], raw_data['total_users'], raw_data['total_items'], name='Train')
val_dataset = Dataset(raw_data['val_data'], raw_data['total_users'], raw_data['total_items'], name='Val', num_negatives=500)
test_dataset = Dataset(raw_data['test_data'], raw_data['total_users'], raw_data['total_items'], name='Test', num_negatives=500)

train_sampler = RandomPairwiseSampler(batch_size=batch_size, dataset=train_dataset, num_process=5)
val_sampler = EvaluationSampler(batch_size=batch_size, dataset=val_dataset)
test_sampler = EvaluationSampler(batch_size=batch_size, dataset=test_dataset)

model = UCML(batch_size=batch_size, total_users=train_dataset.total_users(), total_items=train_dataset.total_items(), 
                dim_user_embed=dim_embed, dim_item_embed=dim_embed, save_model_dir='ucml_recommender/', train=True, serve=True)

def train_iter_func(model, batch_data):
    loss = model.train(batch_data)['losses'][0]
    model.train(batch_data, operations_id='censor_embedding')
    return loss

model_trainer = ModelTrainer(model=model,
                            train_iter_func=train_iter_func)

auc_evaluator = AUC()
recall_evaluator = Recall(recall_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  
model_trainer.train(total_iter=int(1e5), eval_iter=eval_iter, save_iter=save_iter, train_sampler=train_sampler, 
                    eval_samplers=[val_sampler, test_sampler], evaluators=[auc_evaluator, recall_evaluator])
