from collections import defaultdict
import numpy as np
from allennlp.predictors.predictor import Predictor

class NLIAttentionPredictions:

    def __init__(self):
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz", "textual_entailment")

    def get_predictions(self, premise, hypothesis):

        result = self.predictor.predict(hypothesis=hypothesis, premise=premise)
        preds = []
        ans = []
        for i in range(len(result['premise_tokens'])):
            attn = result['p2h_attention'][i]

            for j in range(len(result['hypothesis_tokens'])):
                index =  j

                if result['hypothesis_tokens'][index] == '@@NULL@@':
                    continue

                preds.append((j,attn[j]))

        ans = defaultdict(list)

        for i in range(len(preds)):
            ans[preds[i][0]].append(preds[i][1])

        for k,v in ans.items():
            ans[k] = [np.mean(np.array(v))]

        return ans