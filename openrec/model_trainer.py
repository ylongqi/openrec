from __future__ import print_function
from openrec.utils.evaluators import EvalManager
from termcolor import colored
import sys
import numpy as np

class ModelTrainer(object):

    def __init__(self, model):

        self._model = model
        
        if not self._model.isbuilt():
            self._model.build()
        
        self._trained_it = 0
        
    def _default_train_iter_func(self, model, _input, step, write_summary):
        
        return np.sum(model.train(_input, step=step, 
                                  write_summary=write_summary)['losses'])
    
    def _default_eval_iter_func(self, model, _input):
        
        return np.squeeze(model.serve(_input)['outputs'])
    
    def _evaluate(self, eval_sampler):
        
        metric_results = {}
        for evaluator in self._eval_manager.evaluators:
            metric_results[evaluator.name] = []
        
        batch_data = eval_sampler.next_batch()
        while batch_data is not None:
            scores = self._eval_iter_func(model=self._model, 
                                          _input=batch_data['input'])
            pos_mask = batch_data['pos_mask']
            excl_mask = batch_data['excl_mask'] if 'excl_mask' in batch_data else None
            
            result = {}
            for evaluator in self._evaluators:
                result[evaluator.name] = evaluator.compute(pos_mask=pos_mask, 
                                                           pred=scores, 
                                                           excl_mask=excl_mask)
            
            '''
            result = self._eval_manager.full_eval(pos_samples=all_pos_items,
                                                  excl_pos_samples=[],
                                                predictions=np.concatenate(all_scores, axis=0))
            '''
            completed_user_count = batch_data['progress']
            print('...Evaluated %d users' % completed_user_count, end='\r')
            for key in result:
                metric_results[key].append(result[key])
            batch_data = eval_sampler.next_batch()
            
        return metric_results

    def train(self, total_iter, eval_iter, save_iter, train_sampler, start_iter=0, 
              eval_samplers=[], evaluators=[], train_iter_func=None, eval_iter_func=None):
        
        if train_iter_func is None:
            self._train_iter_func = self._default_train_iter_func
        else:
            self._train_iter_func = train_iter_func
        
        if eval_iter_func is None:
            self._eval_iter_func = self._default_eval_iter_func
        else:
            self._eval_iter_func = eval_iter_func
            
        acc_loss = 0
        self._eval_manager = EvalManager(evaluators=evaluators)
        self._evaluators = evaluators
        
        train_sampler.reset()
        for sampler in eval_samplers:
            sampler.reset()
        
        print(colored('[Training starts, total_iter: %d, eval_iter: %d, save_iter: %d]' \
                          % (total_iter, eval_iter, save_iter), 'blue'))
        
        epoch = 0
        for _iter in range(total_iter):
            batch_data = train_sampler.next_batch()
            if batch_data is None:
                epoch += 1
                print('##### %d epoch completed #####' % epoch)
                continue
                
            loss = self._train_iter_func(model=self._model, 
                                         _input=batch_data['input'], 
                                         step=self._trained_it,
                                         write_summary=(self._trained_it % eval_iter == 0))
            acc_loss += loss
            
            if self._model.is_summary():
                self._model.add_scalar_summary(tag='train_loss',
                                              value=loss,
                                              step=self._trained_it)
            
            print('..Trained for %d iterations.' % _iter, end='\r')
            if _iter % save_iter == 0 and _iter != 0:
                self._model.save(global_step=self._trained_it)
                print(' '*len('..Trained for %d iterations.' % _iter), end='\r')
                print(colored('[iter %d]' % self._trained_it, 'red'), 'Model saved.')
            if _iter % eval_iter == 0:
                print(' '*len('..Trained for %d iterations.' % _iter), end='\r')
                if _iter == 0:
                    average_loss = acc_loss
                else:
                    average_loss = acc_loss / eval_iter
                print(colored('[iter %d]' % self._trained_it, 'red'), 'loss: %f' % average_loss)
                for sampler in eval_samplers:
                    print(colored('..(dataset: %s) evaluation' % sampler.name, 'green'))
                    sys.stdout.flush()
                    eval_results = self._evaluate(sampler)
                    for key, result in eval_results.items():
                        average_result = np.mean(np.vstack(result), axis=0)
                        if type(average_result) is np.ndarray:
                            print(colored('..(dataset: %s)' % sampler.name, 'green'), \
                                key, ' '.join([str(s) for s in average_result]))
                            if self._model.is_summary():
                                for _i in range(len(average_result)):
                                    self._model.add_scalar_summary(tag='%s:%s[%d]' % (sampler.name, key, _i),
                                                              value=average_result[_i],
                                                              step=self._trained_it)
                        else:
                            print(colored('..(dataset: %s)' % sampler.name, 'green'), \
                                key, average_result)
                            if self._model.is_summary():
                                self._model.add_scalar_summary(tag='%s:%s' % (sampler.name, key),
                                                              value=average_result,
                                                              step=self._trained_it)
                acc_loss = 0
            
            self._trained_it += 1
