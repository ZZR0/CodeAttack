"""
ModelArgs Class
===============
"""


from dataclasses import dataclass
import json
import os

import transformers

import codeattack
from codeattack.shared.utils import ARGS_SPLIT_TOKEN, load_module_from_file
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

MODELS = {"codebert": {"config":(RobertaConfig, "microsoft/codebert-base"), 
                       "model": (RobertaModel, "microsoft/codebert-base"), 
                       "tokenizer": (RobertaTokenizer, "microsoft/codebert-base")}}

@dataclass
class ModelArgs:
    """Arguments for loading base/pretrained or trained models."""

    model: str = None
    model_from_file: str = None

    @classmethod
    def _add_parser_args(cls, parser):
        """Adds model-related arguments to an argparser."""

        parser.add_argument(
            "--model",
            type=str,
            required=False,
            default="codebert",
            help="Name of or path to a pre-trained TextAttack model to load. Choices: "
        )

        parser.add_argument(
            "--model_from_file",
            type=str,
            required=False,
            default="",
            help="Name of or path to a pre-trained TextAttack model to load. Choices: "
        )

        parser.add_argument(
            "--task",
            type=str,
            required=True,
            default="search",
        )

        parser.add_argument("--max_source_length",type=int,default=400,)
        parser.add_argument("--max_target_length",type=int,default=400,)

        return parser

    @classmethod
    def _create_model_from_args(cls, args):
        """Given ``ModelArgs``, return specified
        ``textattack.models.wrappers.ModelWrapper`` object."""

        assert isinstance(
            args, cls
        ), f"Expect args to be of type `{type(cls)}`, but got type `{type(args)}`."

        if (args.model in MODELS):
            # Support loading models automatically from the HuggingFace model hub.
            colored_model_name = codeattack.shared.utils.color_text(
                args.model, color="blue", method="ansi"
            )
            codeattack.shared.logger.info(
                f"Loading pre-trained model from : {colored_model_name}"
            )
            config = MODELS[args.model]["config"][0].from_pretrained(
                MODELS[args.model]["config"][1]
            )
            tokenizer = MODELS[args.model]["tokenizer"][0].from_pretrained(
                MODELS[args.model]["tokenizer"][1], use_fast=True
            )
            model = MODELS[args.model]["model"][0].from_pretrained(
                MODELS[args.model]["model"][1], config=config
            )
            model = codeattack.models.helpers.search_model.build_and_load_model(model, config, tokenizer, args)
            
            model = codeattack.models.wrappers.PyTorchModelWrapper(model, tokenizer)

        else:
            raise ValueError(f"Error: unsupported TextAttack model {args.model}")

        assert isinstance(
            model, codeattack.models.wrappers.ModelWrapper
        ), "`model` must be of type `textattack.models.wrappers.ModelWrapper`."
        return model
