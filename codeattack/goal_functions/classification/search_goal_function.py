"""
Determine for if an attack has been successful in Classification
---------------------------------------------------------------------
"""


import numpy as np
import torch

from codeattack.goal_function_results import SearchGoalFunctionResult
from codeattack.goal_function_results.search_goal_function_result import CloneGoalFunctionResult
from codeattack.goal_functions import GoalFunction


class SearchGoalFunction(GoalFunction):
    """A goal function defined on a model that outputs a probability for some
    number of classes."""

    def _process_model_outputs(self, inputs, scores):
        """Processes and validates a list of model outputs.

        This is a task-dependent operation. For example, classification
        outputs need to have a softmax applied.
        """
        # Automatically cast a list or ndarray of predictions to a tensor.
        if isinstance(scores, list) or isinstance(scores, np.ndarray):
            scores = torch.tensor(scores)

        # Ensure the returned value is now a tensor.
        if not isinstance(scores, torch.Tensor):
            raise TypeError(
                "Must have list, np.ndarray, or torch.Tensor of "
                f"scores. Got type {type(scores)}"
            )

        # Validation check on model score dimensions
        if scores.ndim != 3:
            # If model somehow returns too may dimensions, throw an error.
            raise ValueError(
                f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
            )
        elif scores.shape[0] != len(inputs):
            # If model returns an incorrect number of scores, throw an error.
            raise ValueError(
                f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
            )
        return scores.cpu()

    def _is_goal_complete(self, model_output, _):
        return False

    def _get_score(self, model_output, _):
        # If the model outputs a single number and the ground truth output is
        # a float, we assume that this is a regression task.
        loss=-(model_output[0]*model_output[1]).sum(-1)
        return loss

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return SearchGoalFunctionResult

    def extra_repr_keys(self):
        return []

    def _get_displayed_output(self, raw_output):
        return 0

class CloneGoalFunction(SearchGoalFunction):

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return CloneGoalFunctionResult
