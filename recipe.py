from codeattack import Attack
from codeattack.models.wrappers import ModelWrapper
from codeattack.search_methods import GreedyWordSwapWIR
from codeattack.transformations import WordSwapEmbedding, WordSwapGradientBased, WordSwapRandom
from codeattack.constraints.overlap.max_words_perturbed import MaxWordsPerturbed
from codeattack.constraints.pre_transformation import (
    RepeatModification,
)
from codeattack.constraints.semantics import KeyWord
from codeattack.attack_recipes import AttackRecipe

class RandomAttack(AttackRecipe):

    @staticmethod
    def build(model_wrapper, goal_function):
        transformation = WordSwapRandom(max_candidates=50)

        constraints = [RepeatModification()]
        constraints.append(MaxWordsPerturbed(max_num_words=5))
        constraints.append(KeyWord())


        search_method = GreedyWordSwapWIR(wir_method="random")

        return Attack(goal_function, constraints, transformation, search_method)

class TextFoolerAttack(AttackRecipe):

    @staticmethod
    def build(model_wrapper, goal_function):
        #
        # Swap words with their 50 closest embedding nearest-neighbors.
        # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
        #
        transformation = WordSwapEmbedding(max_candidates=50)
        # transformation = WordSwapGradientBased(model_wrapper)
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
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)