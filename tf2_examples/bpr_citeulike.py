from openrec.tf2.data import Dataset
from openrec.tf2.recommenders import BPR
from openrec.tf2.metrics import AUC, NDCG, Recall, DictMean
from tqdm.auto import tqdm
import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers

import dataloader

raw_data = dataloader.load_citeulike('../dataset/')
dim_embed = 50
total_iter = int(1e5)
batch_size = 1000
eval_interval = 1000
save_interval = eval_interval

train_dataset = Dataset(raw_data=raw_data['train_data'], 
                        total_users=raw_data['total_users'], 
                        total_items=raw_data['total_items'])

val_dataset = Dataset(raw_data=raw_data['val_data'], 
                      total_users=raw_data['total_users'], 
                      total_items=raw_data['total_items'])

bpr_model = BPR(total_users=raw_data['total_users'], 
                total_items=raw_data['total_items'], 
                dim_user_embed=dim_embed, 
                dim_item_embed=dim_embed)

optimizer = optimizers.Adam()

@tf.function
def train_step(user_id, p_item_id, n_item_id):
    with tf.GradientTape() as tape:
        loss_value = bpr_model(user_id, p_item_id, n_item_id)
    gradients = tape.gradient(loss_value, bpr_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, bpr_model.trainable_variables))
    return loss_value

@tf.function
def eval_step(user_id, pos_mask, excl_mask):
    pred = bpr_model.inference(user_id)
    auc = AUC(pos_mask=pos_mask, pred=pred, excl_mask=excl_mask)
    recall = Recall(pos_mask=pos_mask, pred=pred, excl_mask=excl_mask, at=[50, 100])
    return {'AUC': auc, 'Recall':recall}

average_loss = tf.keras.metrics.Mean()
average_metrics = DictMean({'AUC': [], 'Recall': [2]})

for train_iter, batch_data in enumerate(train_dataset.pairwise(batch_size=batch_size, 
                                                               num_parallel_calls=5)):
    loss = train_step(**batch_data)
    average_loss.update_state(loss)
    print('%d iter training.' % train_iter, end='\r')
    
    if train_iter % eval_interval == 0:
        for eval_batch_data in tqdm(val_dataset.evaluation(batch_size=batch_size,
                                                          excl_datasets=[train_dataset]),
                                    leave=False, desc='%d iter evaluation' % train_iter):
            eval_results = eval_step(**eval_batch_data)
            average_metrics.update_state(eval_results)
        result = average_metrics.result()
        print("Iter: %d, Loss: %.2f, AUC: %.4f, Recall(50, 100): %s" % (train_iter, average_loss.result().numpy(), 
                                                    result['AUC'].numpy(), result['Recall'].numpy()))
        average_loss.reset_states()
        average_metrics.reset_states()