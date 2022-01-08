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
        return round(float(np.mean(ranks)), 4)

class MAP(Metric):
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

        self.all_metrics["original_map"] = self.cal_map(ori_results)
        self.all_metrics["perturbed_map"] = self.cal_map(per_results)

        return self.all_metrics

    def cal_map(self, results):
        code_vecs = []
        labels = []
        for result in results:
            code_vecs.append(result.raw_output[0:1])
            labels.append(result.ground_truth_output)
        code_vecs=np.concatenate(code_vecs,0)

        scores=np.matmul(code_vecs,code_vecs.T)
        dic={}
        for i in range(scores.shape[0]):
            scores[i,i]=-1000000
            if int(labels[i]) not in dic:
                dic[int(labels[i])]=-1
            dic[int(labels[i])]+=1
        sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
        MAP=[]
        for i in range(scores.shape[0]):
            label=int(labels[i])
            Avep = []
            for j in range(dic[label]):
                index=sort_ids[i,j]
                if int(labels[index])==label:
                    Avep.append((len(Avep)+1)/(j+1))
            MAP.append(sum(Avep)/dic[label])
        return round(float(np.mean(MAP)), 4)