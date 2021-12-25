"""

Metrics on AttackQueries
---------------------------------------------------------------------

"""

import numpy as np

from codeattack.attack_results import SkippedAttackResult
from codeattack.metrics import Metric


class MRR(Metric):
    def __init__(self):
        self.all_metrics = {}

    def calculate(self, results):
        """Calculates all metrics related to number of queries in an attack.

        Args:
            results (``AttackResult`` objects):
                Attack results for each instance in dataset
        """

        self.results = results
        ori_results = [res.original_result for res in results]
        per_results = [res.perturbed_result for res in results]

        self.all_metrics["original_mrr"] = self.cal_mrr(ori_results)
        self.all_metrics["perturbed_mrr"] = self.cal_mrr(per_results)

        return self.all_metrics

    def cal_mrr(self, results):
        nl_vecs = []
        code_vecs = []
        for result in results:
            code_vecs.append(result.raw_output[0:1])
            nl_vecs.append(result.raw_output[1:2])
        code_vecs=np.concatenate(code_vecs,0)
        nl_vecs=np.concatenate(nl_vecs,0)
        scores=np.matmul(nl_vecs,code_vecs.T)
        ranks=[]
        for i in range(len(scores)):
            score=scores[i,i]
            rank=1
            for j in range(len(scores)):
                if i!=j and scores[i,j]>=score:
                    rank+=1
            ranks.append(1/rank)
        return round(float(np.mean(ranks)), 2)
