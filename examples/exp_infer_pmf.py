from openrec.utils import ImplicitDataset
from openrec.recommenders import InferPMF
import numpy as np
import dataloader

raw_data = dataloader.load_citeulike()

finetune_iterations = 500
user_id = 100

train_dataset = ImplicitDataset(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
val_dataset = ImplicitDataset(raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')
test_dataset = ImplicitDataset(raw_data['test_data'], raw_data['max_user'], raw_data['max_item'], name='Test')

model = InferPMF(max_item=train_dataset.max_item(), dim_embed=50, 
                 init_model_dir='pmf_recommender/', training=True, serving=False)

user_upvote = np.array(list(train_dataset.get_interactions_by_user_gb_item(user_id)), np.int32)
val_item = list(val_dataset.get_interactions_by_user_gb_item(user_id))[0]
test_item = list(test_dataset.get_interactions_by_user_gb_item(user_id))[0]
model.train({'user_upvote':user_upvote}, train_ops_id='init')

for i in range(finetune_iterations):
    results = model.train({'user_upvote':user_upvote})

#Output scores for all items: results['outputs'][0]
loss = results['losses'][0]
val_rank = np.sum((results['outputs'][0] > results['outputs'][0][val_item]).astype(np.int32))
test_rank = np.sum((results['outputs'][0] > results['outputs'][0][test_item]).astype(np.int32))

print('Fine-tuning loss:', loss)
print('Rank of the validation item:', val_rank)
print('Rank of the testing item:', test_rank)