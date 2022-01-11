
"""
Attention + LSH
=======
(A Strong Baseline for Query Efficient Attacks in a Black Box Setting)
"""
from codeattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    InputColumnModification
)
from codeattack.constraints.grammaticality.language_models import (
    LearningToWriteLanguageModel,
)
from codeattack.constraints.grammaticality import PartOfSpeech
from codeattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from codeattack.transformations import WordSwapHowNet
from codeattack.goal_functions import UntargetedClassification
from codeattack.search_methods import GreedyWordSwapWIR
from codeattack import Attack
from codeattack.transformations import WordSwapWordNet
from codeattack.transformations import WordSwapEmbedding
from codeattack.constraints.semantics import WordEmbeddingDistance
from codeattack.constraints.overlap import MaxWordsPerturbed

from .attack_recipe import AttackRecipe


class LSHWithAttentionWordNet(AttackRecipe):
    """An implementation of the paper "A Strong Baseline for
    Query Efficient Attacks in a Black Box Setting", Maheshwary et al., 2021.
    The attack jointly leverages attention mechanism and locality sensitive hashing
    (LSH) to rank input words and reduce the number of queries required to attack
    target models. The attack iscevaluated on four different search spaces.
    https://arxiv.org/abs/2109.04775
    """

    @staticmethod
    def build(model, attention_model=None):
        transformation = WordSwapWordNet()
        constraints = [RepeatModification(), StopwordModification()]
        goal_function = UntargetedClassification(model)

        search_method = GreedyWordSwapWIR("lsh_with_attention", attention_model_path=attention_model)
        return Attack(goal_function, constraints, transformation, search_method)


class LSHWithAttentionHowNet(AttackRecipe):

    @staticmethod
    def build(model, attention_model=None):
        #
        # Swap words with their synonyms extracted based on the HowNet.
        #
        transformation = WordSwapHowNet()
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"hypothesis"}
        )
        constraints.append(input_column_modification)
        #
        # Use untargeted classification for demo, can be switched to targeted one
        #
        goal_function = UntargetedClassification(model)
        search_method = GreedyWordSwapWIR(wir_method="lsh_with_attention", attention_model_path=attention_model)

        return Attack(goal_function, constraints, transformation, search_method)

class LSHWithAttentionEmbedding(AttackRecipe):

    @staticmethod
    def build(model, attention_model=None):
        #
        # Swap words with their 50 closest embedding nearest-neighbors.
        # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
        #
        transformation = WordSwapEmbedding(max_candidates=50)
        #
        # Don't modify the same word twice or the stopwords defined
        # in the TextFooler public implementation.
        #
        # fmt: off
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        )
        # fmt: on
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        # Minimum word embedding cosine similarity of 0.5.
        # (The paper claims 0.7, but analysis of the released code and some empirical
        # results show that it's 0.5.)
        #
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))
        #
        # Only replace words with the same part of speech (or nouns with verbs)
        #
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        #
        # Universal Sentence Encoder with a minimum angular similarity of ε = 0.5.
        #
        # In the TextFooler code, they forget to divide the angle between the two
        # embeddings by pi. So if the original threshold was that 1 - sim >= 0.5, the
        # new threshold is 1 - (0.5) / pi = 0.840845057
        #
        use_constraint = UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR(wir_method="lsh_with_attention", attention_model_path=attention_model)

        return Attack(goal_function, constraints, transformation, search_method)


class LSHWithAttentionEmbeddingGen(AttackRecipe):

    def build(model, attention_model=None):
        transformation = WordSwapEmbedding(max_candidates=8)
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        # Maximum words perturbed percentage of 20%
        #
        constraints.append(MaxWordsPerturbed(max_percent=0.2))
        #
        # Maximum word embedding euclidean distance of 0.5.
        #
        constraints.append(WordEmbeddingDistance(max_mse_dist=0.5))
        #
        # Language Model
        #
        #
        #
        # constraints.append(LearningToWriteLanguageModel(window_size=5))
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model)
        #
        # Perform word substitution with a genetic algorithm.
        #
        search_method = GreedyWordSwapWIR(wir_method="weighted-saliency", attention_model_path=attention_model)

        return Attack(goal_function, constraints, transformation, search_method)