# Quiet TensorFlow.
import json
import copy
import argparse

import codeattack
from codeattack import Attacker
from codeattack.goal_functions import DefectClassification, UntargetedClassification
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
    
    parser.add_argument('--task', type=str, default="clone_bcb")
    parser.add_argument('--model', type=str, default="codebert")
    parser.add_argument('--lang', type=str, default="java")
    parser.add_argument('--recipe', type=str, default="textfooler")
    parser.add_argument('--num_examples', type=int, default=-1)
    parser.add_argument('--max_source_length', type=int, default=400)
    parser.add_argument('--max_target_length', type=int, default=400)
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

def update_code(adv_code, site_map, start=0):
    new_site_map = {}
    for idx, key in enumerate(site_map.keys()):
        new_key = key.replace("@R_{}@".format(idx), "@R_{}@".format(idx+start))
        adv_code = adv_code.replace(key, new_key)
        new_site_map[new_key] = site_map[key]
    return adv_code, new_site_map

def build_dataset(args):
    test_file = "./dataset/clone_detection_bigclonebench/test.txt"
    url_to_code={}
    with open(args.test_data_file) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            url_to_code[js['idx']]={"code":js["code"], "adv":js['adv'], "site_map":js["site_map"]}

    dataset = []
    with open(test_file) as f:
        for line in f.readlines()[:10000]:
            line=line.strip()
            url1,url2,label=line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label=='0':
                label=0
            else:
                label=1
            
            code1 = url_to_code[url1]["code"]
            adv_code1 = url_to_code[url1]["adv"]
            site_map1 = copy.deepcopy(url_to_code[url1]["site_map"])
            code2 = url_to_code[url2]["code"]
            adv_code2 = url_to_code[url2]["adv"]
            site_map2 = copy.deepcopy(url_to_code[url2]["site_map"])

            adv_code2, site_map2 = update_code(adv_code2, site_map2, start=len(site_map1))
            site_map1.update(site_map2)
            dataset += [((adv_code1, adv_code2), label, site_map1)]

    dataset = codeattack.datasets.Dataset(dataset, input_columns=["adv1", "adv2"])
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
    elif args.recipe == "random":
        recipe = RandomAttack.build(model_wrapper, goal_function)
    else:
        print("Wrong Recipe.")
    return recipe


if __name__ == "__main__":
    args = parse_args()

    model_wrapper = get_wrapper(args)
    goal_function = UntargetedClassification(model_wrapper, model_batch_size=16)
    recipe = get_recipe(args, model_wrapper, goal_function)

    dataset = build_dataset(args)
    attack_args = codeattack.AttackArgs(num_examples=args.num_examples, parallel=args.parallel)
    attacker = Attacker(recipe, dataset, attack_args=attack_args)
    results = attacker.attack_dataset()
