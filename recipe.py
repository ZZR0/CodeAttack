from codeattack import Attack
from codeattack.search_methods import (
    GreedyWordSwapWIR,
    ParticleSwarmOptimization,
    HardLabelSearch,
    RandomSearch
)
from codeattack.transformations import (
    WordSwapEmbedding, 
    WordSwapGradientBased, 
    WordSwapRandom,
    WordSwapMaskedLM,
    WordSwapHowNet,
    WordSwapWordNet,
)
from codeattack.constraints.overlap.max_words_perturbed import MaxWordsPerturbed
from codeattack.constraints.pre_transformation import (
    RepeatModification,
)
from codeattack.constraints.semantics import KeyWord
from codeattack.attack_recipes import AttackRecipe

def replace_tokens(code):
    code = code.replace("\n", "NEW_LINE")
    code = code.replace("\t", "INDENT")
    code = code.replace("{", "OPEN_CURLY_TOKEN")
    code = code.replace("}", "CLOSE_CURLY_TOKEN")
    code = code.replace("<", "SMALLER_TOKEN")
    code = code.replace(">", "GREATER_TOKEN")
    code = code.replace("[", "OPEN_SQUARE_TOKEN")
    code = code.replace("]", "CLOSE_SQUARE_TOKEN")
    code = code.replace("$", "DOLLAR_TOKEN")
    return code

class RandomAttack(AttackRecipe):

    @staticmethod
    def build(model_wrapper, goal_function):
        transformation = WordSwapRandom(max_candidates=50)

        constraints = [RepeatModification()]
        constraints.append(MaxWordsPerturbed(max_num_words=5))
        constraints.append(KeyWord())


        search_method = GreedyWordSwapWIR(wir_method="random")

        return Attack(goal_function, constraints, transformation, search_method)

class RandomPlusAttack(AttackRecipe):

    @staticmethod
    def build(model_wrapper, goal_function):
        transformation = WordSwapRandom(max_candidates=50)

        # constraints = [RepeatModification()]
        constraints = []
        constraints.append(MaxWordsPerturbed(max_num_words=5))
        constraints.append(KeyWord())


        search_method = RandomSearch(trials=10)

        return Attack(goal_function, constraints, transformation, search_method)

class PSOAttack(AttackRecipe):

    @staticmethod
    def build(model_wrapper, goal_function):
        transformation = WordSwapHowNet()

        constraints = [RepeatModification()]
        constraints.append(MaxWordsPerturbed(max_num_words=5))
        constraints.append(KeyWord())


        search_method = ParticleSwarmOptimization(pop_size=60, max_iters=20)

        return Attack(goal_function, constraints, transformation, search_method)

class BERTAttack(AttackRecipe):

    @staticmethod
    def build(model_wrapper, goal_function):
        transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=48)

        constraints = [RepeatModification()]
        constraints.append(MaxWordsPerturbed(max_num_words=5))
        constraints.append(KeyWord())


        search_method = GreedyWordSwapWIR(wir_method="unk")

        return Attack(goal_function, constraints, transformation, search_method)


class BAEAttack(AttackRecipe):

    @staticmethod
    def build(model_wrapper, goal_function):
        transformation = WordSwapMaskedLM(
            method="bae", max_candidates=50, min_confidence=0.0
        )

        constraints = [RepeatModification()]
        constraints.append(MaxWordsPerturbed(max_num_words=5))
        constraints.append(KeyWord())


        search_method = GreedyWordSwapWIR(wir_method="delete")

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


class LSHAttentionAttack(AttackRecipe):
    """An implementation of the paper "A Strong Baseline for
    Query Efficient Attacks in a Black Box Setting", Maheshwary et al., 2021.
    The attack jointly leverages attention mechanism and locality sensitive hashing
    (LSH) to rank input words and reduce the number of queries required to attack
    target models. The attack iscevaluated on four different search spaces.
    https://arxiv.org/abs/2109.04775
    """

    @staticmethod
    def build(model, goal_function, attention_model="attention_models/yelp/han_model_yelp.pt"):
        transformation = WordSwapWordNet()

        constraints = [RepeatModification()]
        constraints.append(MaxWordsPerturbed(max_num_words=5))
        constraints.append(KeyWord())

        search_method = GreedyWordSwapWIR("lsh_with_attention", attention_model_path=attention_model)

        return Attack(goal_function, constraints, transformation, search_method)


class HardLabelAttack(AttackRecipe):
    @staticmethod
    def build(model_wrapper, goal_function):
        transformation = WordSwapEmbedding(max_candidates=50)

        constraints = [RepeatModification()]
        constraints.append(MaxWordsPerturbed(max_num_words=5))
        constraints.append(KeyWord())

        search_method = HardLabelSearch(pop_size=30, max_iters=100)

        return Attack(goal_function, constraints, transformation, search_method)


class GreedyRandomAttack(AttackRecipe):

    @staticmethod
    def build(model_wrapper, goal_function):
        transformation = WordSwapRandom(max_candidates=50)

        constraints = [RepeatModification()]
        constraints.append(MaxWordsPerturbed(max_num_words=5))
        constraints.append(KeyWord())


        search_method = GreedyWordSwapWIR(wir_method="unk")

        return Attack(goal_function, constraints, transformation, search_method)