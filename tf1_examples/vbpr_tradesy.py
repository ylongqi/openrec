from openrec import ModelTrainer
from openrec import FastDotProductServer
from openrec.tf1.utils import Dataset
from openrec.tf1.recommenders import VBPR
from openrec.tf1.utils.evaluators import AUC, Recall
from openrec.tf1.utils.samplers import VBPRPairwiseSampler
from openrec.tf1.utils.samplers import EvaluationSampler
import dataloader

raw_data = dataloader.load_tradesy()
dim_user_embed = 50
dim_item_embed = 25
total_iter = int(1e5)
batch_size = 10000
eval_iter = 100
save_iter = eval_iter

train_dataset = Dataset(raw_data['train_data'], raw_data['total_users'], raw_data['total_items'], name='Train')
val_dataset = Dataset(raw_data['val_data'], raw_data['total_users'], raw_data['total_items'], name='Val', num_negatives=1000)
test_dataset = Dataset(raw_data['test_data'], raw_data['total_users'], raw_data['total_items'], name='Test', num_negatives=1000)

train_sampler = VBPRPairwiseSampler(batch_size=batch_size, dataset=train_dataset, 
                                      item_vfeature=raw_data['item_features'], num_process=5)
val_sampler = EvaluationSampler(batch_size=batch_size, dataset=val_dataset)
test_sampler = EvaluationSampler(batch_size=batch_size, dataset=test_dataset)

_, dim_v = raw_data['item_features'].shape

model = VBPR(batch_size=batch_size, 
             dim_v=dim_v, 
             total_users=train_dataset.total_users(), 
             total_items=train_dataset.total_items(), 
             dim_user_embed=dim_user_embed, 
             dim_item_embed=dim_item_embed, 
             save_model_dir='vbpr_recommender/', 
             l2_reg_embed=0.001,
             l2_reg_mlp=0.001,
             train=True, serve=True)


def extract_user_lf_func(model, user_id):
    return model.serve_inspect_ports({'user_id':user_id},
                             ports=[model.servegraph.usergraph['user_vec']])[0]
    
def extract_item_lf_func(model, item_id, item_vfeature=raw_data['item_features']):
    return model.serve_inspect_ports({'item_id':item_id,
                                     'item_vfeature': item_vfeature[item_id]},
                                    ports=[model.servegraph.interactiongraph['item_vec']])[0]
    
def extract_item_bias_func(model, item_id):
    return model.serve_inspect_ports({'item_id':item_id},
                                    ports=[model.servegraph.interactiongraph['item_bias']])[0]
    

fastmodel = FastDotProductServer(model=model, batch_size=batch_size,
                     dim_embed=dim_user_embed, 
                     total_users=train_dataset.total_users(), 
                     total_items=train_dataset.total_items(), 
                     extract_user_lf_func=extract_user_lf_func, 
                     extract_item_lf_func=extract_item_lf_func, 
                     extract_item_bias_func=extract_item_bias_func)

model_trainer = ModelTrainer(model=fastmodel)

auc_evaluator = AUC()
recall_evaluator = Recall(recall_at=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  
model_trainer.train(total_iter=total_iter, eval_iter=eval_iter, save_iter=save_iter, train_sampler=train_sampler,
                    eval_samplers=[val_sampler, test_sampler], evaluators=[auc_evaluator, recall_evaluator])