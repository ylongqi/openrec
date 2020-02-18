import numpy as np
from tqdm import tqdm

class ImplicitEvalManager(object):

    def __init__(self, evaluators=[]):

        self.evaluators = evaluators

    def _full_rank(self, pos_samples, excl_pos_samples, predictions):

        
        pos_samples = np.array(pos_samples, dtype=np.int32)
        pos_predictions = predictions[pos_samples]

        excl_pos_samples_set = set(excl_pos_samples)
        rank_above = np.zeros(len(pos_samples))

        pos_samples_len = len(pos_samples)
        for ind in range(len(predictions)):
            if ind not in excl_pos_samples_set:
                for pos_ind in range(pos_samples_len):
                    if pos_predictions[pos_ind] < predictions[ind]:
                        rank_above[pos_ind] += 1

        return rank_above, len(predictions) - len(excl_pos_samples)

    def _partial_rank(self, pos_scores, neg_scores):

        pos_scores = np.array(pos_scores)
        rank_above = np.zeros(len(pos_scores))
        pos_scores_len = len(pos_scores)

        for score in neg_scores:
            for pos_ind in range(pos_scores_len):
                if pos_scores[pos_ind] < score:
                    rank_above[pos_ind] += 1

        return rank_above, len(neg_scores)

    def full_eval(self, pos_samples, excl_pos_samples, predictions):

        results = {}
        rank_above, negative_num = self._full_rank(pos_samples, excl_pos_samples, predictions)
        for evaluator in self.evaluators:
            if evaluator.etype == 'rank':
                results[evaluator.name] = evaluator.compute(rank_above=rank_above, negative_num=negative_num)

        return results

    def partial_eval(self, pos_scores, neg_scores):

        results = {}
        rank_above, negative_num = self._partial_rank(pos_scores, neg_scores)
        for evaluator in self.evaluators:
            if evaluator.etype == 'rank':
                results[evaluator.name] = evaluator.compute(rank_above=rank_above, negative_num=negative_num)

        return results