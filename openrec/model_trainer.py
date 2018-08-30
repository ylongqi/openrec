from __future__ import print_function
from openrec.utils.evaluators import EvalManager
from termcolor import colored
import sys
import numpy as np

class ModelTrainer(object):

    def __init__(self, model, train_it_func=None, eval_it_func=None):

        self._model = model
        # self._serve_batch_size = serve_batch_size
        if not self._model.isbuilt():
            self._model.build()
        
        if train_it_func is None:
            self._train_it_func = self._default_train_it_func
        else:
            self._train_it_func = train_it_func
        
        if eval_it_func is None:
            self._eval_it_func = self._default_eval_it_func
        else:
            self._eval_it_func = eval_it_func
        
        self._trained_it = 0
        
    def _default_train_it_func(self, model, batch_data):
        return model.train(batch_data)['losses'][0]
    
    def _default_eval_it_func(self, model, batch_data):
        return model.serve(batch_data)['outputs'][0]
    
    def _evaluate(self, eval_sampler):
        
        metric_results = {}
        for evaluator in self._eval_manager.evaluators:
            metric_results[evaluator.name] = []
        
        completed_user_count = 0
        pos_items, batch_data = eval_sampler.next_batch()
        while batch_data is not None:
            all_scores = []
            all_pos_items = []
            while len(batch_data) > 0:
                all_scores.append(self._eval_it_func(self._model, batch_data))
                all_pos_items += pos_items
                pos_items, batch_data = eval_sampler.next_batch()
            result = self._eval_manager.full_eval(pos_samples=all_pos_items,
                                                  excl_pos_samples=[],
                                                predictions=np.concatenate(all_scores, axis=0))
            completed_user_count += 1
            print('...Evaluated %d users' % completed_user_count, end='\r')
            for key in result:
                metric_results[key].append(result[key])
            pos_items, batch_data = eval_sampler.next_batch()
            
        return metric_results

    def train(self, total_it, eval_it, save_it, train_sampler, start_it=0, eval_samplers=[], evaluators=[]):
        
        acc_loss = 0
        self._eval_manager = EvalManager(evaluators=evaluators)
        
        print(colored('[Training starts, total_it: %d, eval_it: %d, save_it: %d]' \
                          % (total_it, eval_it, save_it), 'blue'))
        
        for it in range(total_it):
            batch_data = train_sampler.next_batch()
            loss = self._train_it_func(self._model, batch_data)
            acc_loss += loss
            self._trained_it += 1
            print('..Trained for %d iterations.' % it, end='\r')
            if (it + 1) % save_it == 0:
                self._model.save(global_step=self._trained_it)
                print(' '*len('..Trained for %d iterations.' % it), end='\r')
                print(colored('[it %d]' % self._trained_it, 'red'), 'Model saved.')
            if (it + 1) % eval_it == 0:
                print(' '*len('..Trained for %d iterations.' % it), end='\r')
                print(colored('[it %d]' % self._trained_it, 'red'), 'loss: %f' % (acc_loss/eval_it))
                for sampler in eval_samplers:
                    print(colored('..(dataset: %s) evaluation' % sampler.name, 'green'))
                    sys.stdout.flush()
                    eval_results = self._evaluate(sampler)
                    for key, result in eval_results.items():
                        average_result = np.mean(result, axis=0)
                        if type(average_result) is np.ndarray:
                            print(colored('..(dataset: %s)' % sampler.name, 'green'), \
                                key, ' '.join([str(s) for s in average_result]))
                        else:
                            print(colored('..(dataset: %s)' % sampler.name, 'green'), \
                                key, average_result)
                acc_loss = 0