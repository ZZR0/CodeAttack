"""

TextFooler (Is BERT Really Robust?)
===================================================
A Strong Baseline for Natural Language Attack on Text Classification and Entailment)

"""

from codeattack import Attack
from codeattack.constraints.grammaticality import PartOfSpeech
from codeattack.constraints.overlap.max_words_perturbed import MaxWordsPerturbed
from codeattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from codeattack.constraints.semantics import KeyWord
from codeattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from codeattack.goal_functions import UntargetedClassification, SearchGoalFunction
from codeattack.search_methods import GreedyWordSwapWIR
from codeattack.transformations import WordSwapEmbedding

from .attack_recipe import AttackRecipe


class CodeSearchAttack(AttackRecipe):

    @staticmethod
    def build(model_wrapper):
        #
        # Swap words with their 50 closest embedding nearest-neighbors.
        # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
        #
        transformation = WordSwapEmbedding(max_candidates=50)
        #
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        constraints = [RepeatModification()]
        constraints.append(MaxWordsPerturbed(max_num_words=5))
        constraints.append(KeyWord())
        #
        # Goal is untargeted classification
        #
        # goal_function = UntargetedClassification(model_wrapper)
        goal_function = SearchGoalFunction(model_wrapper)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)
