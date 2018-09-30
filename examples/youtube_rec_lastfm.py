from openrec import ModelTrainer
from openrec.utils import Dataset
from openrec.recommenders import YouTubeRec
from openrec.utils.evaluators import AUC, Recall
from openrec.utils.samplers import YouTubeSampler, YouTubeEvaluationSampler
import numpy as np

train_data = np.load('dataset/lastfm/lastfm_train.npy')
test_data = np.load('dataset/lastfm/lastfm_test.npy')
user_feature = np.load('dataset/lastfm/user_feature.npy')

total_users = 992
total_items = 14598 

user_dict = {'gender': 3, 'geo': 67}
item_dict = {'id': total_items}

dim_item_embed = {'total': 50, 'id': 50}
dim_user_embed = {'total': 30, 'geo': 20, 'gender': 10}
max_seq_len = 20

total_iter = int(1e5)
batch_size = 100
eval_iter = 1000
save_iter = eval_iter

train_dataset = Dataset(train_data, total_users, total_items, sortby='ts',
                        name='Train')
test_dataset = Dataset(test_data, total_users, total_items, sortby='ts',
                       name='Test')    


train_sampler = YouTubeSampler(user_feature=user_feature, batch_size=batch_size, max_seq_len=max_seq_len, dataset=train_dataset, num_process=1)
test_sampler = YouTubeEvaluationSampler(user_feature=user_feature, dataset=test_dataset, max_seq_len=max_seq_len)


model = YouTubeRec(batch_size=batch_size, 
                   user_dict=user_dict,
                   item_dict=item_dict,
                   max_seq_len=max_seq_len,
                   dim_item_embed=dim_item_embed, 
                   dim_user_embed=dim_user_embed,
                   save_model_dir='youtube_recommender/', 
                   train=True, serve=True)


model_trainer = ModelTrainer(model=model)

auc_evaluator = AUC()
recall_evaluator = Recall(recall_at=[100, 200, 300, 400, 500])  

model_trainer.train(total_iter=total_iter, eval_iter=eval_iter,
                    save_iter=save_iter,train_sampler=train_sampler,
                    eval_samplers=[test_sampler], evaluators=[auc_evaluator, recall_evaluator])
