# Quiet TensorFlow.
import os
import json
import torch
import argparse
import numpy as np
import torch.nn as nn

import codeattack
from codeattack import Attacker
from codeattack import Attack
from codeattack.models.wrappers import ModelWrapper, model_wrapper
from codeattack.search_methods import GreedyWordSwapWIR
from codeattack.transformations import WordSwapEmbedding, WordSwapGradientBased
from codeattack.constraints.overlap.max_words_perturbed import MaxWordsPerturbed
from codeattack.constraints.pre_transformation import (
    RepeatModification,
)
from codeattack.constraints.semantics import KeyWord
from codeattack.goal_functions import MinimizeBleu
from codeattack.attack_recipes import AttackRecipe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', action='store', dest='save_dir', required=True,
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--n_alt_iters', type=int)
    parser.add_argument('--z_epsilon', type=int)
    parser.add_argument('--u_pgd_epochs', type=int)
    
    parser.add_argument('--task', type=str, default="summarization")
    parser.add_argument('--model', type=str, default="codebert")
    parser.add_argument('--lang', type=str, default="java")
    parser.add_argument('--num_examples', type=int, default=100)
    parser.add_argument('--max_source_length', type=int, default=256)
    parser.add_argument('--max_target_length', type=int, default=128)
    parser.add_argument('--beam_size', type=int, default=10)

    parser.add_argument("--model_name_or_path", default=None, type=str, 
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=-1, type=int)
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
 
    args = parser.parse_args()

    return args



def build_dataset(args):
    dataset = []
    with open(args.test_data_file) as f:
        for line in f:
            js=json.loads(line.strip())
            code=' '.join(js['code'].split())
            adv_code=' '.join(js['adv'].split())
            nl=' '.join(js['nl'].split())
            site_map = js["site_map"]
            dataset += [((code, adv_code, nl), nl, site_map)]

    dataset = codeattack.datasets.Dataset(dataset, input_columns=["code", "adv", "nl"])
    return dataset

class SummarizationAttack(AttackRecipe):

    @staticmethod
    def build(model_wrapper):
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
        goal_function = MinimizeBleu(model_wrapper, model_batch_size=16)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)

def get_wrapper(args):
    if args.model == "codebert":
        from models.codebert_models import build_wrapper
    elif args.model == "codegpt":
        from models.codegpt_models import build_wrapper
    elif args.model == "codet5":
        from models.codet5_models import build_wrapper
    elif args.model == "codetrans":
        from models.codetrans_models import build_wrapper
    elif args.model == "contracode":
        from models.contracode_models import build_wrapper
    elif args.model == "cotext":
        from models.cotext_models import build_wrapper
    elif args.model == "graphcodebert":
        from models.graphcodebert_models import build_wrapper
    elif args.model == "plbart":
        from models.plbart_models import build_wrapper
    
    return build_wrapper(args)

if __name__ == "__main__":
    args = parse_args()

    model_wrapper = get_wrapper(args)

    recipe = SummarizationAttack.build(model_wrapper)

    dataset = build_dataset(args)
    attack_args = codeattack.AttackArgs(num_examples=args.num_examples)
    attacker = Attacker(recipe, dataset, attack_args=attack_args)
    results = attacker.attack_dataset()
