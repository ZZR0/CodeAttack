"""
MORPHEUS2020
===============
(It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations)


"""
from codeattack import Attack
from codeattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from codeattack.goal_functions import MinimizeBleu
from codeattack.search_methods import GreedySearch
from codeattack.transformations import WordSwapInflections

from .attack_recipe import AttackRecipe


class MorpheusTan2020(AttackRecipe):
    """Samson Tan, Shafiq Joty, Min-Yen Kan, Richard Socher.

    It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations

    https://www.aclweb.org/anthology/2020.acl-main.263/
    """

    @staticmethod
    def build(model_wrapper):

        #
        # Goal is to minimize BLEU score between the model output given for the
        # perturbed input sequence and the reference translation
        #
        goal_function = MinimizeBleu(model_wrapper)

        # Swap words with their inflections
        transformation = WordSwapInflections()

        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]

        #
        # Greedily swap words (see pseudocode, Algorithm 1 of the paper).
        #
        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)
