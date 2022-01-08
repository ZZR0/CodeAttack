"""

Determine successful in untargeted Classification
----------------------------------------------------
"""

from typing import List
import numpy as np
import torch

from .classification_goal_function import ClassificationGoalFunction


class DefectClassification(ClassificationGoalFunction):
    """An untargeted attack on classification models which attempts to minimize
    the score of the correct label until it is no longer the predicted label.

    Args:
        target_max_score (float): If set, goal is to reduce model output to
            below this score. Otherwise, goal is to change the overall predicted
            class.
    """

    def __init__(self, *args, target_max_score=None, **kwargs):
        self.target_max_score = target_max_score
        super().__init__(*args, **kwargs)

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
        if scores.ndim == 1:
            # Unsqueeze prediction, if it's been squeezed by the model.
            if len(inputs) == 1:
                scores = scores.unsqueeze(dim=0)
            else:
                raise ValueError(
                    f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
                )
        elif scores.ndim != 2:
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
        if self.target_max_score:
            return model_output[self.ground_truth_output] < self.target_max_score
        elif (model_output.numel() == 1):
            return abs(self.ground_truth_output - model_output.item()) >= 0.5
        else:
            return model_output.argmax() != self.ground_truth_output

    def _get_score(self, model_output, _):
        # If the model outputs a single number and the ground truth output is
        # a float, we assume that this is a regression task.
        if (model_output.numel() == 1):
            return abs(model_output.item() - self.ground_truth_output)
        else:
            return 1 - model_output[self.ground_truth_output]
    
    def _get_displayed_output(self, raw_output, thresh=0.5):
        if len(raw_output) == 2:
            return 1 if raw_output[1] > thresh else 0
        return 1 if raw_output > thresh else 0
