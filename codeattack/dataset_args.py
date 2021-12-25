"""
DatasetArgs Class
=================
"""

from dataclasses import dataclass

import codeattack
import json

@dataclass
class DatasetArgs:
    """Arguments for loading dataset from command line input."""

    dataset_from_file: str = None
    filter_by_labels: list = None

    @classmethod
    def _add_parser_args(cls, parser):
        """Adds dataset-related arguments to an argparser."""

        parser.add_argument(
            "--dataset_from_file",
            type=str,
            required=True,
            default=None,
            help="Dataset to load depending on the name of the model",
        )
        return parser

    @classmethod
    def _create_dataset_from_args(cls, args):
        """Given ``DatasetArgs``, return specified
        ``textattack.dataset.Dataset`` object."""

        assert isinstance(
            args, cls
        ), f"Expect args to be of type `{type(cls)}`, but got type `{type(args)}`."

        if args.dataset_from_file and args.task:
            codeattack.shared.logger.info(
                f"Loading dataset from file: {args.dataset_from_file}"
            )

            try:
                dataset = []
                with open(args.dataset_from_file) as f:
                    for line in f:
                        js=json.loads(line.strip())
                        code=' '.join(js['function'].split())
                        adv_code=' '.join(js['adv_func'].split())
                        nl=' '.join(js['docstring_summary'].split())
                        site_map = js["site_map"]
                        label = 1
                        dataset += [((code, adv_code, nl), label, site_map)]
                
                dataset = codeattack.datasets.Dataset(dataset, input_columns=["code", "adv", "nl"])
            except Exception:
                raise ValueError(f"Failed to import file {args.dataset_from_file}")
        else:
            raise ValueError("Must supply pretrained model or dataset")

        assert isinstance(
            dataset, codeattack.datasets.Dataset
        ), "Loaded `dataset` must be of type `textattack.datasets.Dataset`."

        if args.filter_by_labels:
            dataset.filter_by_labels_(args.filter_by_labels)

        return dataset
