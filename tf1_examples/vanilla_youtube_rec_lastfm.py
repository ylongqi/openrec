from openrec import ModelTrainer
from openrec.tf1.utils import Dataset
from openrec.tf1.recommenders import VanillaYouTubeRec
from openrec.tf1.utils.evaluators import AUC, Recall
from openrec.tf1.utils.samplers import TemporalSampler, TemporalEvaluationSampler
import numpy as np

total_users = 992
total_items = 14598 
train_data = np.load('dataset/lastfm/lastfm_train.npy')
test_data = np.load('dataset/lastfm/lastfm_test.npy')

dim_item_embed = 50
max_seq_len = 20
total_iter = int(1e5)
batch_size = 100
eval_iter = 100
save_iter = eval_iter

train_dataset = Dataset(train_data, total_users, total_items, sortby='ts',
                        name='Train')
test_dataset = Dataset(test_data, total_users, total_items, sortby='ts',
                       name='Test')    


train_sampler = TemporalSampler(batch_size=batch_size, max_seq_len=max_seq_len,
                               dataset=train_dataset, num_process=1)
test_sampler = TemporalEvaluationSampler(dataset=test_dataset,
                                         max_seq_len=max_seq_len)


model = VanillaYouTubeRec(batch_size=batch_size, 
                          total_items=train_dataset.total_items(), 
                          max_seq_len=max_seq_len,
                          dim_item_embed=dim_item_embed, 
                          save_model_dir='vanilla_youtube_recommender/', 
                          train=True, serve=True)


model_trainer = ModelTrainer(model=model)

auc_evaluator = AUC()
recall_evaluator = Recall(recall_at=[100, 200, 300, 400, 500])  

model_trainer.train(total_iter=total_iter, eval_iter=eval_iter,
                    save_iter=save_iter,train_sampler=train_sampler,
                    eval_samplers=[test_sampler], evaluators=[auc_evaluator, recall_evaluator])
