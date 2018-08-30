from __future__ import print_function
from tqdm import tqdm
import math
from termcolor import colored
import numpy as np
import sys

class ItrMLPModelTrainer(object):
    
    def __init__(self, batch_size, test_batch_size, train_dataset, model, sampler):

        self._batch_size = batch_size
        self._test_batch_size = test_batch_size

        self._train_dataset = train_dataset
        self._max_item = self._train_dataset.max_item()

        self._model = model
        self._sampler = sampler
    
    def train(self, num_itr, display_itr, update_itr, eval_datasets=[], evaluators=[]):
        
        acc_loss = 0
        
        for itr in range(num_itr):
            batch_data = self._sampler.next_batch()
            loss = self._model.train(batch_data)
            acc_loss += loss
            if itr % display_itr == 0 and itr > 0:
                print(colored('[Itr %d]' % itr, 'red'), 'loss: %f' % (acc_loss/display_itr),
                        'mse: %f' % (acc_loss * 2 / (display_itr * self._batch_size)))
                for dataset in eval_datasets:
                    print(colored('..(dataset: %s) evaluation' % dataset.name, 'green'))
                    sys.stdout.flush()
                    eval_results = self._evaluate(eval_dataset=dataset, evaluators=evaluators)
                    for key, result in eval_results.items():
                        average_result = np.mean(result, axis=0)
                        if type(average_result) is np.ndarray:
                            print(colored('..(dataset: %s)' % dataset.name, 'green'),
                                key, ' '.join([str(s) for s in average_result]))
                        else:
                            print(colored('..(dataset: %s)' % dataset.name, 'green'), \
                                key, average_result)
                acc_loss = 0
            
            if itr % update_itr == 0 and itr > 0:
                self._model.update_embeddings()
            
    
    def _evaluate(self, eval_dataset, evaluators):
        
        metric_results = {}
        for evaluator in evaluators:
            metric_results[evaluator.name] = []
            
        num_entries = len(eval_dataset.data)
        batch_data = {'user_id_input': np.zeros(self._test_batch_size, np.int32),
                     'item_id_input': np.zeros(self._test_batch_size, np.int32)}
        
        for ind in tqdm(range(int(num_entries / self._test_batch_size))):
            entries = eval_dataset.data[ind*self._test_batch_size:(ind+1)*self._test_batch_size]
            user = entries['user_id']
            item = entries['item_id']
            labels = entries['label']
            
            batch_data['user_id_input'][:len(user)] = user
            batch_data['item_id_input'][:len(item)] = item
            
            for evaluator in evaluators:
                results = evaluator.compute(predictions=self._model.serve(batch_data)[:len(user)],
                                 labels=labels)
                metric_results[evaluator.name].append(results[:len(user)])
        
        for evaluator in evaluators:
            metric_results[evaluator.name] = np.concatenate(metric_results[evaluator.name])
        
        return metric_results