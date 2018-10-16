from openrec import ModelTrainer
from openrec.utils import Dataset
from openrec.recommenders import RNNRec
from openrec.utils.evaluators import AUC, Recall
from openrec.utils.samplers import TemporalSampler
from openrec.utils.samplers import TemporalEvaluationSampler
import numpy as np

lastfm_train = np.load('dataset/lastfm/lastfm_train.npy')
lastfm_test = np.load('dataset/lastfm/lastfm_test.npy')
total_users = 992
total_items = 14598

dim_item_embed = 50
max_seq_len = 100
num_units = 32
batch_size = 256
total_iter = int(1e5)
eval_iter = 100
save_iter = eval_iter

train_dataset = Dataset(raw_data=lastfm_train, total_users=total_users,
                        total_items=total_items, sortby='ts', name='Train')
test_dataset = Dataset(raw_data=lastfm_test, total_users=total_users,
                        total_items=total_items, sortby='ts', name='Test')

train_sampler = TemporalSampler(batch_size=batch_size, max_seq_len=max_seq_len, dataset=train_dataset, num_process=1)
test_sampler = TemporalEvaluationSampler(dataset=test_dataset, max_seq_len=max_seq_len)

rnn_model = RNNRec(batch_size=batch_size, dim_item_embed=dim_item_embed, max_seq_len=max_seq_len, total_items=train_dataset.total_items(), 
                   num_units=num_units, save_model_dir='rnn_recommender/', train=True, serve=True)

model_trainer = ModelTrainer(model=rnn_model)

auc_evaluator = AUC()
recall_evaluator = Recall(recall_at=[100, 500])   
model_trainer.train(total_iter=total_iter, eval_iter=eval_iter, save_iter=save_iter, train_sampler=train_sampler, 
                    eval_samplers=[test_sampler], evaluators=[auc_evaluator, recall_evaluator])