# Quiet TensorFlow.
import json
import argparse

import codeattack
from codeattack import Attacker
from codeattack.models.wrappers import ModelWrapper, model_wrapper
from codeattack.goal_functions import UntargetedClassification, SearchGoalFunction
from recipe import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', action='store', dest='save_dir', required=True,
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--n_alt_iters', type=int)
    parser.add_argument('--z_epsilon', type=int)
    parser.add_argument('--u_pgd_epochs', type=int)
    
    parser.add_argument('--task', type=str, default="search")
    parser.add_argument('--model', type=str, default="codebert")
    parser.add_argument('--lang', type=str, default="java")
    parser.add_argument('--recipe', type=str, default="textfooler")

    parser.add_argument('--num_examples', type=int, default=-1)
    parser.add_argument('--max_source_length', type=int, default=256)
    parser.add_argument('--max_target_length', type=int, default=256)
    parser.add_argument("--model_name_or_path", default=None, type=str, 
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=-1, type=int)
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--parallel", action='store_true')
 
    args = parser.parse_args()

    return args


def build_dataset(args):
    dataset = []
    with open(args.test_data_file) as f:
        for line in f:
            js=json.loads(line.strip())
            code=' '.join(js['function'].split())
            adv_code=' '.join(js['adv_func'].split())
            if args.model == "cotext":
                code = replace_tokens(code)
                adv_code = replace_tokens(adv_code)
            nl=' '.join(js['docstring_summary'].split())
            site_map = js["site_map"]
            label = 1
            dataset += [((adv_code, nl), label, site_map)]

    dataset = codeattack.datasets.Dataset(dataset, input_columns=["adv", "nl"])
    return dataset


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

def get_recipe(args, model_wrapper, goal_function):
    if args.recipe == "textfooler":
        recipe = TextFoolerAttack.build(model_wrapper, goal_function)
    elif args.recipe == "pso":
        recipe = PSOAttack.build(model_wrapper, goal_function)
    elif args.recipe == "bertattack":
        recipe = BERTAttack.build(model_wrapper, goal_function)
    elif args.recipe == "bae":
        recipe = BAEAttack.build(model_wrapper, goal_function)
    elif args.recipe == "lsh":
        recipe = LSHAttentionAttack.build(model_wrapper, goal_function)
    elif args.recipe == "hard":
        recipe = HardLabelAttack.build(model_wrapper, goal_function)
    elif args.recipe == "random":
        recipe = RandomAttack.build(model_wrapper, goal_function)
    else:
        print("Wrong Recipe.")
    return recipe


if __name__ == "__main__":
    args = parse_args()

    model_wrapper = get_wrapper(args)
    goal_function = SearchGoalFunction(model_wrapper, model_batch_size=16, query_budget=200)
    recipe = get_recipe(args, model_wrapper, goal_function)

    dataset = build_dataset(args)
    attack_args = codeattack.AttackArgs(num_examples=args.num_examples, parallel=args.parallel)
    attacker = Attacker(recipe, dataset, attack_args=attack_args)
    results = attacker.attack_dataset()
