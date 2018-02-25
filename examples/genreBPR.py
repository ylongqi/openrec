import os
import sys
sys.path.append(os.getcwd())

from openrec import ImplicitModelTrainer
from openrec.utils import ImplicitDataset
from openrec.recommenders import FeatureBasedBPR
from openrec.utils.evaluators import AUC
from openrec.utils.samplers import PairwiseSampler
from config import sess_config
import dataloader


print 'nah'
raw_data = dataloader.load_music()
print 'yah'
batch_size = 1000
test_batch_size = 100
display_itr = 10000

print 'yuck'
train_dataset = ImplicitDataset(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
print 'h'
val_dataset = ImplicitDataset(raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')
print 'e'
test_dataset = ImplicitDataset(raw_data['test_data'], raw_data['max_user'], raw_data['max_item'], name='Test')
print 'here 1'

bpr_model = FeatureBasedBPR(batch_size=batch_size, max_user=raw_data['max_user'], max_item=raw_data['max_item'], 
                dim_embed=20, sess_config=sess_config, opt='Adam')

# bpr_model = FeatureBasedBPR(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(), 
#                 dim_embed=20, opt='Adam', sess_config=sess_config)

print 'here 2'
sampler = GeneralSampler(batch_size=batch_size, dataset=train_dataset, num_process=1, genre_f = raw_data['song_to_genre'])
print 'here 3'
model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size, 
    train_dataset=train_dataset, model=bpr_model, sampler=sampler)
print 'here 4'
auc_evaluator = AUC()
print 'here 5'
model_trainer.train(num_itr=int(1e6), display_itr=display_itr, eval_datasets=[val_dataset, test_dataset],
                    evaluators=[auc_evaluator])
print 'here 6'