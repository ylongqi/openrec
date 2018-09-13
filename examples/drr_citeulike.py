from openrec import ModelTrainer
from openrec.utils import Dataset
from openrec.recommenders import DRR
from openrec.utils.evaluators import AUC, Recall
from openrec.utils.samplers import EvaluationSampler,StratifiedPointwiseSampler
import dataloader

raw_data = dataloader.load_citeulike()
dim_user_embed = 10
dim_item_embed = 10
total_it = int(1e5)
batch_size = 1000
eval_it = 100
save_it = eval_it


train_dataset = Dataset(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
val_dataset = Dataset(raw_data['val_data'], raw_data['max_user'],
        raw_data['max_item'], name='Val', num_negatives=100)
test_dataset = Dataset(raw_data['test_data'], raw_data['max_user'],
                       raw_data['max_item'], name='Test', num_negatives=100)    


train_sampler = StratifiedPointwiseSampler(pos_ratio=0.5, batch_size=batch_size, dataset=train_dataset, num_process=5)
val_sampler = EvaluationSampler(batch_size=batch_size, dataset=val_dataset)
test_sampler = EvaluationSampler(batch_size=batch_size, dataset=test_dataset)

model = DRR(batch_size=batch_size, 
             total_users=train_dataset.total_users(), 
             total_items=train_dataset.total_items(), 
             dim_user_embed=dim_user_embed, 
             dim_item_embed=dim_item_embed, 
             save_model_dir='drr_recommender/', 
             l2_reg_embed=0.001,
             l2_reg_mlp=0.001,
             train=True, serve=True)


model_trainer = ModelTrainer(model=model)

auc_evaluator = AUC()
recall_evaluator = Recall(recall_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  

model_trainer.train(total_it=total_it, eval_it=eval_it, save_it=save_it,train_sampler=train_sampler, eval_samplers=[val_sampler,test_sampler], evaluators=[auc_evaluator, recall_evaluator])
