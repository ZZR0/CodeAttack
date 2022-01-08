# Quiet TensorFlow.
import os
import json
import torch
import argparse
import numpy as np
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

import codeattack
from codeattack import Attacker
from codesearch import CodeSearchAttack
from codeattack.models.wrappers import ModelWrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', action='store', dest='save_dir', required=True,
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--n_alt_iters', type=int)
    parser.add_argument('--z_epsilon', type=int)
    parser.add_argument('--u_pgd_epochs', type=int)
    
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

class SearchModelWrapper(ModelWrapper):
    """Loads a PyTorch model (`nn.Module`) and tokenizer.

    Args:
        model (torch.nn.Module): PyTorch model
        tokenizer: tokenizer whose output can be packed as a tensor and passed to the model.
            No type requirement, but most have `tokenizer` method that accepts list of strings.
    """

    def __init__(self, model, tokenizer, args):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"PyTorch model must be torch.nn.Module, got type {type(model)}"
            )

        self.model = model
        self.tokenizer = tokenizer
        self.max_source_length = args.max_source_length
        self.args = args

    def to(self, device):
        self.model.to(device)
    
    def get_ids(self, source):
        source_tokens=self.tokenizer.tokenize(source)[:self.max_source_length-2]
        source_tokens =[self.tokenizer.cls_token]+source_tokens+[self.tokenizer.sep_token]
        source_ids =  self.tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = self.max_source_length - len(source_ids)
        source_ids+=[self.tokenizer.pad_token_id]*padding_length
        return source_ids

    def __call__(self, text_input_list, batch_size=32):

        model_device = next(self.model.parameters()).device
        code_ids = [self.get_ids(text[1]) for text in text_input_list]
        nl_ids = [self.get_ids(text[2]) for text in text_input_list]

        code_ids = torch.tensor(code_ids).to(model_device)
        nl_ids = torch.tensor(nl_ids).to(model_device)

        with torch.no_grad():
            outputs = self.model(code_inputs=code_ids, nl_inputs=nl_ids)

        return outputs

def build_dataset(args):
    dataset = []
    with open(args.test_data_file) as f:
        for line in f:
            js=json.loads(line.strip())
            code=' '.join(js['function'].split())
            adv_code=' '.join(js['adv_func'].split())
            nl=' '.join(js['docstring_summary'].split())
            site_map = js["site_map"]
            label = 1
            dataset += [((code, adv_code, nl), label, site_map)]

    dataset = codeattack.datasets.Dataset(dataset, input_columns=["code", "adv", "nl"])
    return dataset

if __name__ == "__main__":
    args = parse_args()
    config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, use_fast=True)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model = codeattack.models.helpers.search_model.build_and_load_model(model, config, tokenizer, args)
    model = codeattack.models.wrappers.PyTorchModelWrapper(model, tokenizer)

    model_wrapper = SearchModelWrapper(model, tokenizer, args)

    # Create the recipe: PWWS uses a WordNet transformation.
    recipe = CodeSearchAttack.build(model_wrapper)

    dataset = build_dataset(args)

    attacker = Attacker(recipe, dataset)
    results = attacker.attack_dataset()
