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

class RandomSearch(SearchMethod):
    """An attack that greedily chooses from a list of possible perturbations in
    order of index, after ranking indices by importance.

    Args:
        wir_method: method for ranking most important words
        model_wrapper: model wrapper used for gradient-based ranking
    """

    def __init__(self, trials):
        self.trials = trials

    def _get_index_order(self, initial_text):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""
        len_text = len(initial_text.words)

        index_order = np.arange(len_text)
        np.random.shuffle(index_order)
        search_over = False

        return index_order, search_over

    def perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text
        # Sort words by order of importance
        index_order, search_over = self._get_index_order(attacked_text)
        cur_result = initial_result

        i = 0
        results = None
        while i < len(index_order) and not search_over:

            for _ in range(self.trials):
                transformed_text_candidates = self.get_transformations(
                    cur_result.attacked_text,
                    original_text=initial_result.attacked_text,
                    indices_to_modify=[index_order[i]],
                )
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
                
            i += 1

        return cur_result

    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word swap and deletion transformations."""
        return True

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["trials"]
