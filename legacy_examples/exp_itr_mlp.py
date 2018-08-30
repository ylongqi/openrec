import numpy as np
from openrec.legacy import ItrMLPModelTrainer
from openrec.legacy.utils import Dataset
from openrec.legacy.recommenders import ItrMLP
from openrec.legacy.utils.evaluators import MSE
from openrec.legacy.utils.samplers import ExplicitSampler

batch_size = 32
test_batch_size = 32
display_itr = 4096
update_itr = 4096

max_user = 480189
max_item = 17770

pretrained_user_embeddings = np.load('dataset/netflix/pretrained_user_embeddings.npy')
pretrained_item_embeddings = np.load('dataset/netflix/pretrained_item_embeddings.npy')
netflix_ratings = np.load('dataset/netflix/netflix_ratings_formatted.npy')

train_dataset = Dataset(netflix_ratings[:-int(1e7)], max_user=max_user, max_item=max_item, name='Train')
val_dataset = Dataset(netflix_ratings[-int(1e7):-int(5e6)], max_user=max_user, max_item=max_item, name='Val')
test_dataset = Dataset(netflix_ratings[-int(5e6):], max_user=max_user, max_item=max_item, name='Test')

model = ItrMLP(batch_size=batch_size, max_user=max_user, max_item=max_item, dim_embed=20, opt='SGD',
              pretrained_user_embeddings=pretrained_user_embeddings, pretrained_item_embeddings=pretrained_item_embeddings,
              user_dims=[30, 30, 20], item_dims=[30, 30, 20], test_batch_size=test_batch_size)

sampler = ExplicitSampler(batch_size=batch_size, dataset=train_dataset, chronological=True)
model_trainer = ItrMLPModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size, 
    train_dataset=train_dataset, model=model, sampler=sampler)

mse_evaluator = MSE()

model_trainer.train(num_itr=int(1e5), display_itr=display_itr, update_itr=update_itr,
                    eval_datasets=[val_dataset, test_dataset],
                    evaluators=[mse_evaluator])
