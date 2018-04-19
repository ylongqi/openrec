from __future__ import division
import pickle
import numpy as np
from openrec.utils.evaluators import AUC, Recall, Precision, NDCG
#import matplotlib.pyplot as plt

###
infilename = "./others-gmf-citeulike-test_evaluate_partial.pickle"
trainset_path = "/Users/xuan/Documents/Specialization Project/openrec/dataset/citeulike/user_data_train.npy"
###

#
trainset = np.load(trainset_path)
trainset = trainset['item_id']
frequency = dict()
for i in trainset:
    if i in frequency:
        frequency[i] += 1
    else:
        frequency[i] = 1
#

auc_evaluator = AUC()
recall_evaluator = Recall(recall_at=[10])
precision_evaluator = Precision(precision_at=[10])
ndcg_evaluator = NDCG(ndcg_at=[10])

f = open(infilename, 'rb')
p = pickle.load(f)
f.close()

score_per_item = dict()
count_per_item = dict()

for user in p['users']:
    neg_scores = p['results'][user][:p['num_negatives']]
    for i in range(len(p['user_items'][user][p['num_negatives'] : ])):
        pos_score = p['results'][user][p['num_negatives'] + i]
        rank_above = np.array([ float(np.sum(neg_scores > pos_score)) ])
        #print(rank_above)
        #print(p['num_negatives'])
        negative_num = float(p['num_negatives'])
        #print("#####")
        curr_score_auc = auc_evaluator.compute(rank_above, negative_num)
        curr_score_recall = recall_evaluator.compute(rank_above, negative_num)[0]
        curr_score_precision = precision_evaluator.compute(rank_above, negative_num)[0]
        curr_score_ndcg = ndcg_evaluator.compute(rank_above, negative_num)[0]
        if p['user_items'][user][p['num_negatives'] + i] not in score_per_item:
            score_per_item[ p['user_items'][user][p['num_negatives'] + i] ] = list()
        if p['user_items'][user][p['num_negatives'] + i] not in count_per_item:
            count_per_item[ p['user_items'][user][p['num_negatives'] + i] ] = 0.0
        score_per_item[ p['user_items'][user][p['num_negatives'] + i] ].append( (curr_score_auc, curr_score_recall, curr_score_ndcg, curr_score_precision) )
        count_per_item[ p['user_items'][user][p['num_negatives'] + i] ] += 1

# calculate per-item scores
per_item_auc = dict()
per_item_recall = dict()
per_item_ndcg = dict()
per_item_precision = dict()

for key in score_per_item.keys():
    curr_auc = 0.0
    curr_recall = 0.0
    curr_ndcg = 0.0
    curr_precision = 0.0
    for tup in score_per_item[key]:
        curr_auc += tup[0]
        curr_recall += tup[1]
        curr_ndcg += tup[2]
        curr_precision += tup[3]
    per_item_auc[ key ] = curr_auc / count_per_item[key]
    per_item_recall[ key ] = curr_recall / count_per_item[key]
    per_item_ndcg[ key ] = curr_ndcg / count_per_item[key]
    per_item_precision[ key ] = curr_precision / count_per_item[key]

# result - uniform
print("Uniform: ")
numerator_auc = 0.0
denominator = len(per_item_auc.keys())
numerator_recall = 0.0
numerator_ndcg = 0.0
numerator_precision = 0.0
for key in per_item_auc:
    numerator_auc += per_item_auc[key]
    numerator_recall += per_item_recall[key]
    numerator_ndcg += per_item_ndcg[key]
    numerator_precision += per_item_precision[key]
print("auc", numerator_auc / denominator)
print("recall", numerator_recall / denominator)
print("ndcg", numerator_ndcg / denominator)
print("precision", numerator_precision / denominator)

# result - natural
print("Natural: ")
numerator_auc = 0.0
denominator = 0.0
numerator_recall = 0.0
numerator_ndcg = 0.0
numerator_precision = 0.0
for key in per_item_auc:
    numerator_auc += per_item_auc[key] * count_per_item[key]
    numerator_recall += per_item_recall[key] * count_per_item[key]
    numerator_ndcg += per_item_ndcg[key] * count_per_item[key]
    numerator_precision += per_item_precision[key] * count_per_item[key]
    denominator += count_per_item[key]
print("auc", numerator_auc / denominator)
print("recall", numerator_recall / denominator)
print("ndcg", numerator_ndcg / denominator)
print("precision", numerator_precision / denominator)

# result - unbiased
print("Unbiased: ")
score_per_activelevel = dict()
count_per_activelevel = dict()
for key in per_item_auc:
    curr_frequency = frequency.get(key, 0)
    if curr_frequency not in score_per_activelevel:
        score_per_activelevel[ curr_frequency ] = 0.0
    if curr_frequency not in count_per_activelevel:
        count_per_activelevel[ curr_frequency ] = 0.0
    score_per_activelevel[ curr_frequency ] += per_item_auc[key]
    count_per_activelevel[ curr_frequency ] += 1
XandY = list()
maxX = np.amax(count_per_activelevel.keys())
for key in count_per_activelevel:
    XandY.append((key / maxX, score_per_activelevel[key] / count_per_activelevel[key]))
XandY.sort(key = lambda x : x[0])
X = [aa[0] for aa in XandY]
Y = [aa[1] for aa in XandY]
area = 0.0
for i in range(1, len(X)):
    area += (Y[i-1] + Y[i]) * (X[i] - X[i-1])
area /= 2
print(area)

#plt.plot(X, Y)
#plt.show()
