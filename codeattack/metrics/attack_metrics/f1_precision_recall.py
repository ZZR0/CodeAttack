"""

Metrics on AttackSuccessRate
---------------------------------------------------------------------

"""

from codeattack.metrics import Metric
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


class F1(Metric):
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

        self.all_metrics["original_f1"] = self.cal(ori_results)
        self.all_metrics["perturbed_f1"] = self.cal(per_results)

        return self.all_metrics

    def cal(self, results):
        y_trues = []
        y_preds = []
        for result in results:
            y_trues.append(result.ground_truth_output)
            y_preds.append(result.output)

        f1=f1_score(y_trues, y_preds, average='macro')
        return round(f1, 4)


class Precision(Metric):
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

        self.all_metrics["original_precision"] = self.cal(ori_results)
        self.all_metrics["perturbed_precision"] = self.cal(per_results)

        return self.all_metrics

    def cal(self, results):
        y_trues = []
        y_preds = []
        for result in results:
            y_trues.append(result.ground_truth_output)
            y_preds.append(result.output)

        precision=precision_score(y_trues, y_preds, average='macro')   
        return round(precision, 4)


class Recall(Metric):
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

        self.all_metrics["original_recall"] = self.cal(ori_results)
        self.all_metrics["perturbed_recall"] = self.cal(per_results)

        return self.all_metrics

    def cal(self, results):
        y_trues = []
        y_preds = []
        for result in results:
            y_trues.append(result.ground_truth_output)
            y_preds.append(result.output)

        recall=recall_score(y_trues, y_preds, average='macro')
        return round(recall, 4)