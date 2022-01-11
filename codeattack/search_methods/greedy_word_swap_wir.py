"""
Greedy Word Swap with Word Importance Ranking
===================================================


When WIR method is set to ``unk``, this is a reimplementation of the search
method from the paper: Is BERT Really Robust?

A Strong Baseline for Natural Language Attack on Text Classification and
Entailment by Jin et. al, 2019. See https://arxiv.org/abs/1907.11932 and
https://github.com/jind11/TextFooler.
"""

import random
import numpy as np
import torch
from torch.nn.functional import softmax

from codeattack.goal_function_results import GoalFunctionResultStatus
from codeattack.search_methods import SearchMethod
from codeattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)
from codeattack.attention_models.han import HAN
from codeattack.attention_models.utils import *
from codeattack.attention_models.dan_snli import NLIAttentionPredictions
from codeattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder

use = UniversalSentenceEncoder()
class GreedyWordSwapWIR(SearchMethod):
    """An attack that greedily chooses from a list of possible perturbations in
    order of index, after ranking indices by importance.

    Args:
        wir_method: method for ranking most important words
        model_wrapper: model wrapper used for gradient-based ranking
    """

    def __init__(self, wir_method="unk", attention_model_path=None):
        self.wir_method = wir_method
        self.attention_model_path = attention_model_path
        if self.attention_model_path == "mnli":
            self.nli_preds = NLIAttentionPredictions()

    def _get_index_order(self, initial_text):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""
        len_text = len(initial_text.words)

        if self.wir_method == "unk":
            leave_one_texts = [
                initial_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])
        elif self.wir_method == "lsh_with_attention":

          # The implementation of Attention + LSH Ranking step  (https://arxiv.org/abs/2109.04775)
            if self.attention_model_path != "mnli":

                #Load Hierarchical Attention Network (HAN) for classification task
                han = HAN(path=self.attention_model_path)
                doc, score, word_alpha, sentence_alphas = han.classify(" ".join(initial_text.words[:1000]))
                scrs = []
                stop_words = stop_word_set()
                word_alpha = word_alpha.detach().cpu().numpy()
                for i in range(len(word_alpha)):
                    for j in range(len(word_alpha[i])):
                        if doc[i][j] not in stop_words and len(doc[i][j]) > 2:
                            scrs.append(word_alpha[i][j])
                        else:
                            scrs.append(-101.0)
                for i in range(len(scrs), len(initial_text.words)):
                    scrs.append(-101.0)

            elif self.attention_model_path == "mnli":

               # Load Decompose Attention Model (DA) for entailment task

                len_premise = len(initial_text.words_per_input[0])

                premise = " ".join(initial_text.words_per_input[0])
                hypothesis = " ".join(initial_text.words_per_input[1])
                scrs = []

                hypothesis_scores = self.nli_preds.get_predictions(premise, hypothesis)

                for i in range(len_text):
                    if i < len_premise:
                        scrs.append(-101.0)
                    else:
                        scrs.append(hypothesis_scores[i - len_premise][0])

            scrs = np.asarray(scrs)
            index_scores = scrs
            search_over = False
            saliency_scores = np.array([result for result in scrs])
            softmax_saliency_scores = softmax(
                torch.Tensor(saliency_scores), dim=0
            ).numpy()

            #Scores due to attention model
            index_scores = softmax_saliency_scores
            delta_ps = []

            # LSH step
            # Substitute each word with all candidates from the search space
            for idx in range(len_text):
                transformed_text_candidates = self.get_transformations(
                    initial_text,
                    original_text=initial_text,
                    indices_to_modify=[idx],
                )
                if not transformed_text_candidates:
                    delta_ps.append(0.0)
                    continue
                text_to_encode = []
                idx_to_txt = {}
                k=0

                for txt in transformed_text_candidates:
                    idx_to_txt[str(k)] = txt
                    res = txt.text_window_around_index(idx,10)
                    text_to_encode.append(res)
                    k+=1

                # Encode all the generates input texts using sentence encoder
                embeddings = use.encode(text_to_encode)

                lsh = LSH(512) #dimension size of USE embeddings is 512

                for t in range(len(embeddings)):
                    lsh.add(embeddings[t],str(t))
                table = lsh.get_result()
                transformed_text_candidates = []

                #Get a random candidate from each bucket
                for key,value in table.table.items():
                    val = random.choice(value)
                    transformed_text_candidates.append(idx_to_txt[val])

                #The final candidate is the one which causes the maximum change in
                #target model's confidence score
                swap_results, _ = self.get_goal_results(transformed_text_candidates)
                score_change = [result.score for result in swap_results]
                max_score_change = np.max(score_change)
                delta_ps.append(max_score_change)
            index_scores = (softmax_saliency_scores) * (delta_ps)

        elif self.wir_method == "weighted-saliency":
            # first, compute word saliency
            leave_one_texts = [
                initial_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            saliency_scores = np.array([result.score for result in leave_one_results])

            softmax_saliency_scores = softmax(
                torch.Tensor(saliency_scores), dim=0
            ).numpy()

            # compute the largest change in score we can find by swapping each word
            delta_ps = []
            for idx in range(len_text):
                transformed_text_candidates = self.get_transformations(
                    initial_text,
                    original_text=initial_text,
                    indices_to_modify=[idx],
                )
                if not transformed_text_candidates:
                    # no valid synonym substitutions for this word
                    delta_ps.append(0.0)
                    continue
                swap_results, _ = self.get_goal_results(transformed_text_candidates)
                score_change = [result.score for result in swap_results]
                if not score_change:
                    delta_ps.append(0.0)
                    continue
                max_score_change = np.max(score_change)
                delta_ps.append(max_score_change)

            index_scores = softmax_saliency_scores * np.array(delta_ps)

        elif self.wir_method == "delete":
            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "gradient":
            victim_model = self.get_victim_model()
            index_scores = np.zeros(initial_text.num_words)
            grad_output = victim_model.get_grad(initial_text.tokenizer_input)
            gradient = grad_output["gradient"]
            word2token_mapping = initial_text.align_with_model_tokens(victim_model)
            for i, word in enumerate(initial_text.words):
                matched_tokens = word2token_mapping[i]
                if not matched_tokens:
                    index_scores[i] = 0.0
                else:
                    agg_grad = np.mean(gradient[matched_tokens], axis=0)
                    index_scores[i] = np.linalg.norm(agg_grad, ord=1)

            search_over = False

        elif self.wir_method == "random":
            index_order = np.arange(len_text)
            np.random.shuffle(index_order)
            search_over = False
        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")

        if self.wir_method != "random":
            index_order = (-index_scores).argsort()

        return index_order, search_over

    def perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text

        # Sort words by order of importance
        index_order, search_over = self._get_index_order(attacked_text)

        i = 0
        cur_result = initial_result
        results = None
        while i < len(index_order) and not search_over:
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            i += 1
            if len(transformed_text_candidates) == 0:
                continue
            results, search_over = self.get_goal_results(transformed_text_candidates)
            results = sorted(results, key=lambda x: -x.score)
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score:
                cur_result = results[0]
            else:
                continue
            # If we succeeded, return the index with best similarity.
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                best_result = cur_result
                # @TODO: Use vectorwise operations
                max_similarity = -float("inf")
                for result in results:
                    if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        break
                    candidate = result.attacked_text
                    try:
                        similarity_score = candidate.attack_attrs["similarity_score"]
                    except KeyError:
                        # If the attack was run without any similarity metrics,
                        # candidates won't have a similarity score. In this
                        # case, break and return the candidate that changed
                        # the original score the most.
                        break
                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                        best_result = result
                return best_result

        return cur_result

    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word swap and deletion transformations."""
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    @property
    def is_black_box(self):
        if self.wir_method == "gradient":
            return False
        else:
            return True

    def extra_repr_keys(self):
        return ["wir_method"]
